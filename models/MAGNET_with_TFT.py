# MAGNET_with_TFT.py
# Mode 2: MAGNET causal graph bias injected into TFT's history/future representations
# Mode 3: Vanilla TFT only (no MAGNET)
#
# This is a single-"observed"-variable (endogenous-only) Temporal Fusion Transformer built from the
# Time-Series-Library TFT blocks (VariableSelectionNetwork, TemporalFusionDecoder; vendored in
# backbones/time_series_library, MIT license). No static covariates are used. The exogenous series never
# reach the backbone directly, only through MAGNET's causal graph, and no future values of any variable
# are used: the future branch is fed a learned placeholder query, so the LSTM's carried state (and, in
# mode 2, MAGNET's graph bias computed from the history) is the only information about what is ahead.

import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from MAGNET_graph_only import MAGNETGraphOnly
from time_series_library.layers.Embed import DataEmbedding
from time_series_library.models.TemporalFusionTransformer import (
    VariableSelectionNetwork, TemporalFusionDecoder,
)


# ── Backbone (reuses vendored TFT building blocks) ─────────────────────────────

class TFTBackbone(nn.Module):
    """
    Self-contained TFT backbone, endo-only (1 "observed" variable, no static
    covariates). Accepts an optional graph_bias [B, seq_len+pred_len, d_model]
    added onto the history/future representations right before they enter
    TFT's LSTM encoder-decoder.
    """
    def __init__(
        self,
        seq_len : int,
        pred_len: int,
        d_model : int = 64,
        n_heads : int = 4,
        dropout : float = 0.1,
    ):
        super().__init__()
        self.seq_len  = seq_len
        self.pred_len = pred_len

        self.enc_embedding = DataEmbedding(1, d_model, dropout=dropout)
        self.history_vsn   = VariableSelectionNetwork(d_model, 1, dropout=dropout)
        # No real future-known covariates (see module docstring) — a single
        # learned query stands in for them; the LSTM's carried hidden state
        # and (in mode 2) the graph bias are what actually inform the future.
        self.future_query = nn.Parameter(torch.zeros(1, pred_len, d_model))

        decoder_configs = types.SimpleNamespace(
            seq_len=seq_len, pred_len=pred_len, d_model=d_model,
            n_heads=n_heads, dropout=dropout, c_out=1,
        )
        self.temporal_fusion_decoder = TemporalFusionDecoder(decoder_configs)

    def forward(self, x, graph_bias=None):
        """
        x          : [B, seq_len, 1]  endo series, already normalized
        graph_bias : [B, seq_len+pred_len, d_model] or None
        returns    : [B, pred_len]
        """
        emb = self.enc_embedding(x, None)                      # [B, seq_len, d_model]
        history_input = self.history_vsn(emb.unsqueeze(-2))    # [B, seq_len, d_model]
        future_input  = self.future_query.expand(x.size(0), -1, -1)  # [B, pred_len, d_model]

        if graph_bias is not None:
            history_input = history_input + graph_bias[:, :self.seq_len]
            future_input  = future_input  + graph_bias[:, self.seq_len:]

        # No static context (c_c/c_h/c_e=None) — TFT's own GRN/LSTM code
        # already degrades gracefully to zero-initialized state / no context
        # bias when these are None, so nothing extra is needed here.
        dec_out = self.temporal_fusion_decoder(
            history_input, future_input, None, None, None
        )                                                       # [B, pred_len, 1]
        return dec_out.squeeze(-1)                               # [B, pred_len]


# ── Main wrapper ──────────────────────────────────────────────────────────────

class MAGNETWithTFT(nn.Module):
    """
    Hybrid model: TFT backbone + optional MAGNET causal graph bias.

    Mode 2 — MAGNET graph bias injected into TFT's history/future inputs:
        1. MAGNETGraphOnly produces patch-level bias [B, P, d_graph]
        2. Projected to TFT's d_model -> upsampled (linear interp) from P
           patches to the full seq_len+pred_len length
        3. Gated (sigmoid, init -2.0 short-term / -3.5 long-term — same
           term-aware convention as MAGNET_with_iTrans.py, since long-horizon
           graph signal is noisier) and added onto the history/future inputs
           right before TFT's LSTM encoder-decoder runs

    Mode 3 — Vanilla TFT:
        Endo-only, no MAGNET, no real future covariates (future branch is a
        learned placeholder — see TFTBackbone docstring).
    """

    def __init__(self, magnet_config: dict, tft_config: dict, mode: int = 2,
                 use_text: bool = True):
        super().__init__()
        self.mode = mode
        term = magnet_config.get('term', 'short')

        seq_len  = tft_config['seq_len']
        pred_len = tft_config['pred_len']
        d_model  = tft_config.get('d_model', 64)
        n_heads  = tft_config.get('n_heads', 4)
        dropout  = tft_config.get('dropout', 0.1)

        # ── MAGNET graph module (mode 2 only) ─────────────────────────────
        if self.mode == 2:
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
            self.graph_proj = nn.Sequential(
                nn.Linear(d_graph, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(0.1 if term == 'short' else 0.3),
            )
            gate_init = -2.0 if term == 'short' else -3.5
            self.bias_gate = nn.Parameter(torch.tensor(gate_init))
        else:
            self.magnet_graph = None
            self.graph_proj   = None

        # ── TFT backbone ─────────────────────────────────────────────────
        self.backbone = TFTBackbone(
            seq_len  = seq_len,
            pred_len = pred_len,
            d_model  = d_model,
            n_heads  = n_heads,
            dropout  = dropout,
        )

    def forward(
        self,
        exo_patches   : torch.Tensor,         # [B, P, patch_len, n_exo]
        endo_patches  : torch.Tensor,         # [B, P, patch_len, 1]
        exog_future   : torch.Tensor = None,  # unused, kept for API compat
        time_features : torch.Tensor = None,  # unused
        dataset       : int = None,
    ) -> torch.Tensor:
        if exog_future is not None:
            raise ValueError("future exogenous values are not accepted: forecasts use the observed history only")

        B, P, pl, _ = endo_patches.shape

        endo_history = endo_patches.reshape(B, P * pl, 1)  # [B, seq_len, 1]

        means = endo_history.mean(dim=1, keepdim=True).detach()
        stdev = torch.sqrt(
            torch.var(endo_history, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        endo_norm = (endo_history - means) / stdev

        m = means[:, 0, 0].unsqueeze(1)   # [B, 1]
        s = stdev[:, 0, 0].unsqueeze(1)   # [B, 1]

        if self.mode == 3:
            out = self.backbone(endo_norm)
            return out * s + m

        graph_bias = self.magnet_graph(exo_patches, endo_patches)  # [B, P, d_graph]
        graph_proj = self.graph_proj(graph_bias)                    # [B, P, d_model]

        total_len = self.backbone.seq_len + self.backbone.pred_len
        graph_bias_full = F.interpolate(
            graph_proj.permute(0, 2, 1),   # [B, d_model, P]
            size=total_len,
            mode='linear',
            align_corners=False,
        ).permute(0, 2, 1)                  # [B, total_len, d_model]
        graph_bias_full = torch.sigmoid(self.bias_gate) * graph_bias_full

        out = self.backbone(endo_norm, graph_bias=graph_bias_full)
        return out * s + m
