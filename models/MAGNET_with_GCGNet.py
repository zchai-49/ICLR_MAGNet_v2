# MAGNET_with_GCGNet.py
# MAGNet as a plug-and-play module on GCGNet (mode 2); mode 3 is vanilla GCGNet.
# Injection (hybrid mode 2): MAGNet's per-patch graph bias [B,P,d] -> MLP to 1 channel -> upsample to the input length ->
# sigmoid gate (initialised at -2, i.e. mostly closed) -> added to the ENDOGENOUS channel of GCGNet's input after
# GCGNet's own per-window normalisation and before its patch embedding, so it reaches both the VAE forecast path
# and the graph/GCN path. ngc_lag=1, history-only input.
#
# GCGNet's training objective uses the true future (endogenous + exogenous) only as a graph-teacher signal
# (`if self.training`); at evaluation both futures are replaced with zeros, and passing exog_future in eval mode raises.
import os
import sys
import types

import torch
import torch.nn as nn


def _forward_with_endo_bias(self, history, exog_future, endo_future=None, endo_bias=None):
    """GCGNetModel.forward, unchanged except that `endo_bias` [B, input_len, 1] is added to the normalised
    endogenous history right after GCGNet's own per-window normalisation."""
    import torch
    endo_history = history[:, -self.input_len:, :self.endo_num]
    exog_history = history[:, -self.input_len:, self.endo_num:]
    exog_future = exog_future[:, :self.pred_len, :]
    endo_future = endo_future[:, :self.pred_len, :]

    if self.use_norm:
        endo_means = endo_history.mean(1, keepdim=True).detach()
        endo_stdev = torch.sqrt(torch.var(endo_history, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        exog = exog_history
        exog_means = exog.mean(1, keepdim=True).detach()
        exog_stdev = torch.sqrt(torch.var(exog, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()

        endo_history = self.sample_norm(endo_history, endo_means, endo_stdev)
        if self.training:
            endo_future = self.sample_norm(endo_future, endo_means, endo_stdev)
        exog_history = self.sample_norm(exog_history, exog_means, exog_stdev)
        exog_future = self.sample_norm(exog_future, exog_means, exog_stdev)

    if endo_bias is not None:
        endo_history = endo_history + endo_bias

    history = torch.cat([endo_history, exog_history], dim=-1)[:, -self.input_len:, :]
    history = history.permute(0, 2, 1)

    patch_history, _ = self.patch_embedding(history)
    patch_history = patch_history.reshape(-1, self.var_num, self.input_patch_num * self.d_model)
    predict_future, mu, logvar = self.vae(patch_history)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    predict_full_sequence = torch.cat([history, predict_future], dim=-1)
    patch_predict_full_sequence, _ = self.patch_embedding(predict_full_sequence)
    patch_predict_full_sequence = patch_predict_full_sequence.reshape(
        -1, self.var_num * (self.input_patch_num + self.pred_patch_num), self.d_model)
    predict_adj, mu, logvar = self.graph_discriminator(patch_predict_full_sequence)
    graph_kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    graph_loss = 0
    if self.training:
        target_future = torch.cat([endo_future, exog_future], dim=-1).permute(0, 2, 1)
        target_full_sequence = torch.cat([history, target_future], dim=-1)
        patch_target_full_sequence, _ = self.patch_embedding(target_full_sequence)
        patch_target_full_sequence = patch_target_full_sequence.reshape(
            -1, self.var_num * (self.input_patch_num + self.pred_patch_num), self.d_model)
        target_adj, _, _ = self.graph_discriminator(patch_target_full_sequence)
        graph_loss = self.graph_criterion(predict_adj, target_adj.detach())

    predict_adj, moe_loss = self.sparsifier(predict_adj, is_training=self.training)
    patch_full_enc = self.gcn(predict_adj, patch_predict_full_sequence)
    patch_full_enc = patch_full_enc.reshape(-1, self.var_num, self.input_patch_num + self.pred_patch_num, self.d_model)
    endo_target_dec = self.head(patch_full_enc[:, :, :, :]).permute(0, 2, 1)
    if self.use_norm:
        endo_target_dec = self.sample_denorm(endo_target_dec, endo_means, endo_stdev)
    return endo_target_dec, graph_loss + moe_loss + kl_loss + graph_kl_loss


class MAGNETWithGCGNet(nn.Module):
    def __init__(self, magnet_config: dict, gcg_config, mode: int = 3,
                 use_text: bool = True):
        super().__init__()
        assert mode in (2, 3), "mode 2 = GCGNet + MAGNet, mode 3 = vanilla GCGNet"
        assert not gcg_config.use_future_exog, "honest setting only"
        self.mode = mode
        self.pred_len = gcg_config.pred_len
        bb_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backbones")
        if bb_root not in sys.path:
            sys.path.insert(0, bb_root)

        from ts_benchmark.baselines.GCGNet.models.gcgnet_model import GCGNetModel
        self.gcg = GCGNetModel(gcg_config)
        self.input_len = self.gcg.input_len

        if mode == 2:
            from MAGNET_graph_only import MAGNETGraphOnly
            d_graph = magnet_config.get('d_model', 64)
            self.magnet_graph = MAGNETGraphOnly(
                n_exo          = magnet_config['n_exo'],
                patch_len      = magnet_config['patch_len'],
                d_model        = d_graph,
                n_heads        = magnet_config.get('n_heads', 4),
                ngc_hidden_dim = magnet_config.get('ngc_hidden_dim', 32),
                dropout        = magnet_config.get('dropout', 0.1),
                dataset        = magnet_config['dataset'],
                term           = magnet_config['term'],
                use_text       = use_text,
                ngc_lag        = magnet_config.get('ngc_lag', 1),
            )
            self.graph_proj = nn.Sequential(nn.Linear(d_graph, 16), nn.GELU(), nn.Linear(16, 1))
            self.bias_scale = nn.Parameter(torch.tensor(-2.0))   # sigmoid(-2) ~ 0.12: starts mostly closed
            # GCGNetModel.forward is patched to accept the bias (added in normalised space)
            self.gcg.forward = types.MethodType(_forward_with_endo_bias, self.gcg)
        else:
            self.magnet_graph = None

    def forward(
        self,
        exo_patches   : torch.Tensor,         # [B, P, patch_len, n_exo]
        endo_patches  : torch.Tensor,         # [B, P, patch_len, 1]
        exog_future   : torch.Tensor = None,  # [B, pred_len, n_exo] (training-time teacher only)
        time_features : torch.Tensor = None,  # unused
        endo_future   : torch.Tensor = None,  # [B, pred_len] (training-time teacher only)
        dataset       : int = None,
    ):
        if exog_future is not None and not self.training:
            raise ValueError("future exogenous values are only accepted in training mode (GCGNet's teacher signal); "
                             "evaluation uses the observed history only")
        B, P, pl, n_exo = exo_patches.shape
        endo_seq = endo_patches.reshape(B, P * pl, 1)
        exo_seq  = exo_patches.reshape(B, P * pl, n_exo)
        history  = torch.cat([endo_seq, exo_seq], dim=-1)  # endo first, as series_dim=1

        if self.training:
            assert exog_future is not None and endo_future is not None
            if exog_future.dim() == 4:
                exog_future = exog_future.reshape(B, -1, n_exo)
            endo_future = endo_future.reshape(B, -1, 1)
        else:
            # never let eval see real future values
            exog_future = history.new_zeros(B, self.pred_len, n_exo)
            endo_future = history.new_zeros(B, self.pred_len, 1)

        if self.mode == 2:
            bias = self.graph_proj(self.magnet_graph(exo_patches, endo_patches))       # [B, P, 1]
            bias = torch.nn.functional.interpolate(bias.permute(0, 2, 1), size=self.input_len,
                                                   mode='linear', align_corners=False).permute(0, 2, 1)
            endo_bias = torch.sigmoid(self.bias_scale) * bias                            # [B, input_len, 1]
            output, aux_loss = self.gcg(history, exog_future, endo_future, endo_bias=endo_bias)
        else:
            output, aux_loss = self.gcg(history, exog_future, endo_future)
        pred = output[:, :, 0]  # [B, pred_len]
        if self.training:
            return pred, aux_loss
        return pred
