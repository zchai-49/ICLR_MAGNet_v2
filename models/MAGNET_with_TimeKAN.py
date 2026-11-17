# MAGNET_with_TimeKAN.py
# Mode 2: MAGNET causal graph bias injected into TimeKAN's patch embeddings
# Mode 3: Vanilla TimeKAN only (no MAGNET)
#
# TimeKAN (ICLR 2025) forecasts by building a multi-resolution pyramid of the
# series, decomposing each resolution into a low-frequency (trend) and
# high-frequency (seasonal) part via FFT, mixing information across
# resolutions (season fine→coarse, trend coarse→fine, mirroring TimeMixer's
# PDM block), and replacing the mixing/projection Linears with KAN layers
# (learnable per-edge spline activations) whose polynomial order is varied by
# frequency content — sharper/higher-order splines for the seasonal path,
# smoother/lower-order for the trend path ("multi-order KAN").
#
# This is a from-scratch, self-contained reimplementation of that idea (no
# official TimeKAN code is vendored anywhere in this repo) adapted to fit the
# patch-based interface the rest of this codebase already uses:
#   - The pyramid is built by patch-embedding the endo series first (raw
#     seq_len -> P = seq_len/patch_len, same patch_len=12 convention as every
#     other MAGNET_with_*.py backbone), then average-pooling the EMBEDDED
#     sequence to build coarser scales — cheaper than the paper's raw-series
#     pyramid + independent per-scale embedding, and keeps every KAN layer's
#     matrices small (tens of positions, not hundreds), which matters since a
#     KANLinear costs ~(grid_size+spline_order)x more than a plain nn.Linear.
#   - Frequency decomposition per scale is a real FFT low-pass/residual split
#     (rfft, zero out all but the lowest `low_freq_ratio` bins, irfft), not a
#     moving-average approximation like DLinear/TimeMixer typically use.
#   - Graph bias is injected the same way as MAGNET_with_PatchTST.py: added
#     directly to the finest-resolution patch embeddings before the pyramid
#     and mixing run, gated by a learned sigmoid (init -2.0, "start mostly
#     closed", same convention used across every other backbone here).

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from MAGNET_graph_only import MAGNETGraphOnly


# ── KAN building block (efficient-kan style B-spline linear layer) ───────────

class KANLinear(nn.Module):
    """
    Learnable per-edge activation function = base term (SiLU(x) @ W_base) +
    a B-spline term (grid_size + spline_order learnable coefficients per
    input-output edge). `spline_order` controls how sharp/expressive the
    spline can be — this is the knob TimeKAN's "multi-order" mixing varies
    between its seasonal (higher order) and trend (lower order) paths.

    Operates on the last dimension of any-shaped input, like nn.Linear.
    """

    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 spline_scale=1.0, grid_range=(-1.0, 1.0)):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.grid_size    = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * h
            + grid_range[0]
        ).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)  # [in_features, grid_size + 2*spline_order + 1]

        self.base_weight   = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        self.base_activation = nn.SiLU()

        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.1 * spline_scale)

    def b_splines(self, x):
        # x: [N, in_features] -> [N, in_features, grid_size + spline_order]
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left  = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)])
            right = (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-k])
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        return bases

    def forward(self, x):
        orig_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_out   = F.linear(self.base_activation(x), self.base_weight)
        spline_in  = self.b_splines(x).reshape(x.size(0), -1)
        spline_out = F.linear(spline_in, self.spline_weight.reshape(self.out_features, -1))

        out = base_out + spline_out
        return out.reshape(*orig_shape[:-1], self.out_features)


# ── Frequency decomposition (FFT low-pass / high-pass residual split) ────────

class FrequencyDecomp(nn.Module):
    """
    Splits x into a trend (low-frequency, via FFT low-pass) and seasonal
    (high-frequency residual) component. `low_ratio` is the fraction of the
    lowest rfft bins (including DC) kept for the trend reconstruction.
    """

    def __init__(self, low_ratio=0.2):
        super().__init__()
        self.low_ratio = low_ratio

    def forward(self, x):  # x: [B, T, D]
        T = x.shape[1]
        Xf = torch.fft.rfft(x, dim=1)
        n_freq = Xf.shape[1]
        cutoff = max(1, int(round(n_freq * self.low_ratio)))
        trend_f = torch.zeros_like(Xf)
        trend_f[:, :cutoff, :] = Xf[:, :cutoff, :]
        trend = torch.fft.irfft(trend_f, n=T, dim=1)
        seasonal = x - trend
        return seasonal, trend


# ── Backbone ──────────────────────────────────────────────────────────────────

class TimeKANBackbone(nn.Module):
    """
    Self-contained TimeKAN-style backbone, endo-only (enc_in=1).
    Accepts an optional patch_bias [B, P, d_model] added to the finest-scale
    patch embeddings before the multi-resolution pyramid is built.
    """

    def __init__(
        self,
        seq_len             : int,
        pred_len            : int,
        patch_len           : int   = 12,
        d_model              : int  = 64,
        e_layers             : int  = 2,
        down_sampling_window : int  = 2,
        down_sampling_layers : int  = 2,
        low_freq_ratio        : float = 0.2,
        grid_size              : int = 5,
        season_spline_order    : int = 3,
        trend_spline_order     : int = 2,
        dropout                : float = 0.1,
    ):
        super().__init__()
        assert seq_len % patch_len == 0, "seq_len must be divisible by patch_len"
        self.seq_len    = seq_len
        self.pred_len   = pred_len
        self.patch_len  = patch_len
        self.P          = seq_len // patch_len
        self.d_model    = d_model
        self.e_layers   = e_layers
        self.down_sampling_window = down_sampling_window

        # Cap the number of pyramid levels so the coarsest scale never
        # collapses below 2 patches (matters for small-P datasets, e.g. the
        # hydro short-term setting where P can be as low as 5).
        L, length = 0, self.P
        scale_lengths = [self.P]
        for _ in range(down_sampling_layers):
            nxt = length // down_sampling_window
            if nxt < 2:
                break
            length = nxt
            scale_lengths.append(length)
            L += 1
        self.L = L                          # number of extra (coarser) scales
        self.scale_lengths = scale_lengths  # length L+1, finest -> coarsest

        # ── Patch embedding (finest scale) ─────────────────────────────────
        self.patch_embed = nn.Sequential(
            nn.Linear(patch_len, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # ── Frequency decomposition (shared module, stateless) ─────────────
        self.freq_decomp = FrequencyDecomp(low_ratio=low_freq_ratio)

        # ── Per-layer mixing KAN modules ────────────────────────────────────
        self.season_down = nn.ModuleList([
            nn.ModuleList([
                KANLinear(scale_lengths[i - 1], scale_lengths[i],
                          grid_size=grid_size, spline_order=season_spline_order)
                for i in range(1, L + 1)
            ])
            for _ in range(e_layers)
        ])
        self.trend_up = nn.ModuleList([
            nn.ModuleList([
                KANLinear(scale_lengths[i + 1], scale_lengths[i],
                          grid_size=grid_size, spline_order=trend_spline_order)
                for i in range(L - 1, -1, -1)   # built finest-first: index i -> i-th entry
            ])
            for _ in range(e_layers)
        ])
        self.norms = nn.ModuleList([
            nn.ModuleList([nn.LayerNorm(d_model) for _ in range(L + 1)])
            for _ in range(e_layers)
        ])
        self.dropout = nn.Dropout(dropout)

        # ── Future projection (one KAN head per scale, averaged) ───────────
        self.future_proj = nn.ModuleList([
            KANLinear(scale_lengths[i], pred_len,
                      grid_size=grid_size, spline_order=season_spline_order)
            for i in range(L + 1)
        ])
        self.out_proj = nn.Linear(d_model, 1)

    def _pool(self, x):  # x: [B, T, D] -> [B, T//window, D]
        return F.avg_pool1d(
            x.permute(0, 2, 1), kernel_size=self.down_sampling_window
        ).permute(0, 2, 1)

    def forward(self, x, patch_bias=None):
        """
        x          : [B, seq_len, 1]  endo series, already normalized
        patch_bias : [B, P, d_model] or None
        returns    : [B, pred_len]
        """
        B = x.shape[0]

        patches = x.reshape(B, self.P, self.patch_len)   # [B, P, patch_len]
        emb = self.patch_embed(patches)                  # [B, P, d_model]

        if patch_bias is not None:
            emb = emb + patch_bias

        # Build the multi-resolution pyramid (finest -> coarsest).
        scales = [emb]
        for _ in range(self.L):
            scales.append(self._pool(scales[-1]))

        # ── e_layers of KAN-based season/trend cross-scale mixing ─────────
        for l in range(self.e_layers):
            seasons, trends = [], []
            for s in scales:
                se, tr = self.freq_decomp(s)
                seasons.append(se)
                trends.append(tr)

            # season mixing: fine -> coarse
            season_out = [seasons[0]]
            for i in range(1, self.L + 1):
                proj = self.season_down[l][i - 1](
                    season_out[i - 1].permute(0, 2, 1)
                ).permute(0, 2, 1)
                season_out.append(seasons[i] + proj)

            # trend mixing: coarse -> fine
            trend_out = [None] * (self.L + 1)
            trend_out[self.L] = trends[self.L]
            for j, i in enumerate(range(self.L - 1, -1, -1)):
                proj = self.trend_up[l][j](
                    trend_out[i + 1].permute(0, 2, 1)
                ).permute(0, 2, 1)
                trend_out[i] = trends[i] + proj

            scales = [
                self.norms[l][i](self.dropout(scales[i] + season_out[i] + trend_out[i]))
                for i in range(self.L + 1)
            ]

        # ── Future multi-resolution prediction (averaged across scales) ───
        preds = []
        for i in range(self.L + 1):
            p = self.future_proj[i](scales[i].permute(0, 2, 1)).permute(0, 2, 1)
            preds.append(p)  # [B, pred_len, d_model]

        out = torch.stack(preds, dim=0).mean(dim=0)   # [B, pred_len, d_model]
        out = self.out_proj(out)                       # [B, pred_len, 1]
        return out[:, :, 0]


# ── Main wrapper ──────────────────────────────────────────────────────────────

class MAGNETWithTimeKAN(nn.Module):
    """
    Hybrid model: TimeKAN backbone + optional MAGNET causal graph bias.

    Mode 2 — MAGNET graph bias injected into TimeKAN's patch embeddings:
        1. MAGNETGraphOnly produces patch-level bias [B, P, d_graph]
        2. Bias projected to TimeKAN's d_model, interpolated to match the
           backbone's own patch count (in case patch_len configs differ)
        3. Gated (sigmoid, init -2.0) and added to the finest-scale patch
           embeddings before the pyramid/mixing stack runs

    Mode 3 — Vanilla TimeKAN:
        Pure TimeKAN on the normalized endo sequence, no MAGNET involvement.
    """

    def __init__(self, magnet_config: dict, timekan_config: dict, mode: int = 2,
                 use_text: bool = True):
        super().__init__()
        self.mode = mode

        seq_len   = timekan_config['seq_len']
        pred_len  = timekan_config['pred_len']
        patch_len = timekan_config.get('patch_len', 12)
        d_model   = timekan_config.get('d_model', 64)
        e_layers  = timekan_config.get('e_layers', 2)
        down_sampling_window = timekan_config.get('down_sampling_window', 2)
        down_sampling_layers = timekan_config.get('down_sampling_layers', 2)
        low_freq_ratio        = timekan_config.get('low_freq_ratio', 0.2)
        grid_size             = timekan_config.get('grid_size', 5)
        season_spline_order   = timekan_config.get('season_spline_order', 3)
        trend_spline_order    = timekan_config.get('trend_spline_order', 2)
        dropout               = timekan_config.get('dropout', 0.1)

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
                nn.Dropout(0.1),
            )
            self.bias_gate = nn.Parameter(torch.tensor(-2.0))
        else:
            self.magnet_graph = None
            self.graph_proj   = None

        # ── TimeKAN backbone ────────────────────────────────────────────────
        self.backbone = TimeKANBackbone(
            seq_len              = seq_len,
            pred_len             = pred_len,
            patch_len            = patch_len,
            d_model              = d_model,
            e_layers             = e_layers,
            down_sampling_window = down_sampling_window,
            down_sampling_layers = down_sampling_layers,
            low_freq_ratio       = low_freq_ratio,
            grid_size            = grid_size,
            season_spline_order  = season_spline_order,
            trend_spline_order   = trend_spline_order,
            dropout              = dropout,
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

        graph_bias = self.magnet_graph(exo_patches, endo_patches)  # [B, P_g, d_graph]
        graph_proj = self.graph_proj(graph_bias)                   # [B, P_g, d_model]

        target_P = self.backbone.P
        if graph_proj.shape[1] != target_P:
            graph_proj = F.interpolate(
                graph_proj.permute(0, 2, 1), size=target_P,
                mode='linear', align_corners=False,
            ).permute(0, 2, 1)
        patch_bias = torch.sigmoid(self.bias_gate) * graph_proj

        out = self.backbone(endo_norm, patch_bias=patch_bias)
        return out * s + m
