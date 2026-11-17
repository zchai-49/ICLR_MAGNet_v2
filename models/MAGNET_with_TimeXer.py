# MAGNET_with_TimeXer.py
# Mode 2: MAGNET graph bias injected into TimeXer's patch embeddings
# Mode 3: Vanilla TimeXer only

import torch
import torch.nn as nn

from MAGNET_graph_only import MAGNETGraphOnly
from timexer.model.timexer_model import timexer_model


class TimeXerWithMAGNET(nn.Module):
    """
    Hybrid model: TimeXer forecaster + optional MAGNET causal graph bias.

    Mode 2 — MAGNET graph bias injected into TimeXer:
        1. TimeXer does its own patch embedding on raw endo/exo sequences
        2. MAGNETGraphOnly produces a causal bias [B, P, d_graph]
        3. Bias is projected to TimeXer's d_model and added to endo patch embeddings
        4. TimeXer encoder + head runs as normal
        5. Output is denormed by TimeXer's own normalization

    Mode 3 — Vanilla TimeXer only:
        TimeXer runs forward_only with raw sequences, no MAGNET involvement.
    """

    def __init__(self, magnet_config: dict, timexer_config, mode: int = 2,
                 graph_proj_arch: str = 'bottleneck', use_text: bool = True):
        super().__init__()
        self.mode = mode

        # ── MAGNET graph module (mode 2 only) ──────────────────────────────
        if self.mode == 2:
            self.magnet_graph = MAGNETGraphOnly(
                n_exo         = magnet_config['n_exo'],
                patch_len     = magnet_config['patch_len'],
                d_model       = magnet_config.get('d_model', 64),
                n_heads       = magnet_config.get('n_heads', 4),
                ngc_hidden_dim= magnet_config.get('ngc_hidden_dim', 32),
                dropout       = magnet_config.get('dropout', 0.1),
                dataset       = magnet_config['dataset'],
                term          = magnet_config['term'],
                use_text      = use_text,
                ngc_lag       = magnet_config.get('ngc_lag', 1),
            )
            # Project graph d_model → TimeXer d_model
            #
            # NOTE: two architectures exist in the wild. Checkpoints saved
            # before the bottleneck refactor use 'legacy' (single Linear
            # straight to timexer d_model); newer runs use 'bottleneck'.
            # Callers loading an existing checkpoint should inspect its
            # state_dict and pass the matching graph_proj_arch — see
            # detect_graph_proj_arch() in test_best_TimeXer.py.
            d_model_graph = magnet_config.get('d_model', 64)
            if graph_proj_arch == 'legacy':
                self.graph_proj = nn.Sequential(
                    nn.Linear(d_model_graph, timexer_config.d_model),
                    nn.LayerNorm(timexer_config.d_model),
                    nn.Dropout(0.1),
                )
            else:
                d_mid = max(d_model_graph * 2, 128)
                self.graph_proj = nn.Sequential(
                    nn.Linear(d_model_graph, d_mid),
                    nn.GELU(),
                    nn.LayerNorm(d_mid),
                    nn.Linear(d_mid, timexer_config.d_model),
                    nn.LayerNorm(timexer_config.d_model),
                    nn.Dropout(0.1),
                )
        else:
            self.magnet_graph = None
            self.graph_proj   = None

        # ── TimeXer (always present) ────────────────────────────────────────
        self.timexer = timexer_model(timexer_config)
        self.patch_importance = nn.Linear(timexer_config.d_model, 1)

        # Global injection strength — sigmoid-bounded, initialized near zero
        # so training starts close to the vanilla TimeXer backbone and the
        # causal-graph channel only opens up where gradients say it helps.
        # (See forward(): replaces the old softmax(...)*P re-weighting,
        # whose max per-patch amplification grows with patch count P and was
        # destabilizing long-term runs, where P is ~4x larger than short-term.)
        if self.mode == 2:
            self.injection_scale = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        exo_patches   : torch.Tensor,         # [B, P, patch_len, n_exo]
        endo_patches  : torch.Tensor,         # [B, P, patch_len, 1]
        exog_future   : torch.Tensor = None,  # [B, pred_len, n_exo]
        time_features : torch.Tensor = None,
        dataset       : int = None,
    ) -> torch.Tensor:
        if exog_future is not None:
            raise ValueError("future exogenous values are not accepted: forecasts use the observed history only")

        B, P, pl, n_exo = exo_patches.shape

        # Reconstruct raw sequences for TimeXer
        exo_history  = exo_patches.reshape(B, P * pl, n_exo)   # [B, seq_len, n_exo]
        endo_history = endo_patches.reshape(B, P * pl, 1)       # [B, seq_len, 1]

        # TimeXer's exogenous cross-attention sees the observed history only.
        full_exog = exo_history

        # ── MODE 3: Vanilla TimeXer ─────────────────────────────────────────
        if self.mode == 3:
            dec_out = self.timexer.forward_only(endo_history, full_exog)
            return dec_out[:, :, 0]  # [B, pred_len]

        # ── MODE 2: TimeXer + MAGNET graph bias ─────────────────────────────

        # 1. Compute MAGNET causal graph bias
        graph_bias = self.magnet_graph(exo_patches, endo_patches)  # [B, P, d_graph]
        graph_bias = self.graph_proj(graph_bias)                    # [B, P, d_timexer]

        # Bounded per-patch gate (0,1) x bounded global scale (0,1), both
        # learned. Unlike softmax(...)*P — mean-preserving but unbounded in
        # the max (grows with P) — this can never amplify a patch beyond 1x,
        # and starts near 0 so early training matches the vanilla backbone.
        patch_gate = torch.sigmoid(self.patch_importance(graph_bias))  # [B, P, 1]
        injection_strength = torch.sigmoid(self.injection_scale)        # scalar
        graph_bias_weighted = injection_strength * patch_gate * graph_bias

        # 2. Run TimeXer with graph bias injection
        dec_out = self.timexer.forward_with_magnet(
            endo_history = endo_history,   # [B, seq_len, 1]
            full_exog    = full_exog,      # [B, seq+pred, n_exo]
            graph_bias   = graph_bias_weighted,     # [B, P, d_timexer]
        )  # [B, pred_len]

        return dec_out