# MAGNET_with_DLinear_fusion.py
# Copy of MAGNET_with_DLinear.py — do NOT edit that file, it backs existing
# DLinear-hybrid results. Adds THREE new, optional injection mechanisms on
# top of the original pre-decomposition additive bias:
#
#   use_scalar_injection        — the ORIGINAL mechanism (graph_bias -> 1D ->
#     upsampled to seq_len -> added to raw endo BEFORE trend/seasonal
#     decomposition, gated by bias_scale). Kept as its own flag.
#
#   use_dual_injection           — inject SEPARATELY into the trend and
#     seasonal components AFTER decomposition (not combined before),
#     independently gated (trend_gate, seasonal_gate). MAGNet's causal
#     signal may matter differently for slow trend vs. fast seasonal
#     structure — DLinear is the one backbone here where that split is
#     architecturally free to test, since decomposition is already explicit.
#
#   use_postlinear_film_injection — DLinear's core computation IS just two
#     Linear layers (linear_trend, linear_seasonal); this modulates their
#     OUTPUTS (not the raw input) via graph_bias-derived (gamma, beta) at
#     pred_len resolution — the natural DLinear-shaped analogue of FiLM,
#     since there's no embedding stage to hook into pre-linear.
#
#   use_gated_mixing_injection   — DLinear normally sums
#     seasonal_out + trend_out unconditionally (implicitly a fixed 1:1
#     mix). This lets graph_bias derive a per-sample mixing weight instead:
#     out = (1+w)*seasonal_out + (1-w)*trend_out, where
#     w = tanh(mix_logit) * sigmoid(mix_gate) — bounded in (-1, 1), and w=0
#     (cold start, mix_gate≈-2.0) reproduces the original unweighted sum
#     exactly. Specific to DLinear's explicit decomposition structure.
#
# 2026-07-24: three MORE injection mechanisms, added to give the model more ways to
# use the graph bias. All 4 mechanisms above
# only ever inject graph_bias as a smooth interpolated bias or a global
# mean-pooled modulation — none let the model dynamically attend to WHICH
# patches matter, and none touch the linear weights themselves. These three
# target that gap:
#
#   use_crossattn_injection      — Cross-attention injection. A learned query attends over
#     MAGNet's full per-patch graph_bias sequence (real QKV attention, not
#     mean-pooling or interpolation) producing a context vector fed into the
#     SAME post-linear FiLM hook as use_postlinear_film_injection, but via
#     its own gate/projections (crossattn_film_gate) — isolates whether
#     attention-based pooling beats naive mean-pooling for identical FiLM
#     injection.
#
#   use_hyper_weight_injection   — a hypernetwork predicts a rank-4 additive
#     delta to linear_trend/linear_seasonal's weight matrices PER SAMPLE,
#     conditioned on mean-pooled graph_bias: deltaW = U @ V^T (U:[pred_len,
#     r], V:[seq_len,r]). Every mechanism above only adds a bias before or
#     after the linear map; this is the first that changes the seq_len->
#     pred_len mapping itself. LoRA-style cold start — V is left at default
#     random init, U is zero-init, so deltaW=0 exactly at step 0 but
#     gradients flow into U immediately (dL/dU = dL/d(deltaW) @ V, nonzero
#     since V!=0), avoiding the symmetric zero-init dead zone of
#     zero-initializing both factors.
#
#   use_gated_scale_injection    — the ORIGINAL scalar_injection is purely
#     additive (x = x + causal_bias, pre-decomposition). This instead
#     multiplicatively gates the raw input before decomposition:
#     x = x * (1 + sigmoid(scale_gate) * tanh(scale_bias)), letting MAGNet's
#     signal suppress or amplify existing structure instead of only
#     shifting it. Cheapest of the three to test — near-identical code path
#     to scalar_injection, good sanity baseline before the pricier two above.
#
# All new gates follow the established cold-start (-2.0 sigmoid init) +
# high-LR-param-group convention (see run_fusion_retrain.py's docstring for
# why — gate-starvation finding, confirmed across 24+ runs).
#
# Verified bit-identical to MAGNET_with_DLinear.py's mode=2 output when only
# use_scalar_injection=True and the other three flags are False.

import torch
import torch.nn as nn
import torch.nn.functional as F

from MAGNET_graph_only import MAGNETGraphOnly


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size=25):
        super().__init__()
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):  # x: [B, seq_len, 1]
        x_t = x.permute(0, 2, 1)
        trend = self.avg(x_t)
        if trend.shape[-1] != x_t.shape[-1]:
            trend = trend[:, :, :x_t.shape[-1]]
        trend = trend.permute(0, 2, 1)
        seasonal = x - trend
        return seasonal, trend


class DLinearBackbone(nn.Module):
    # 2026-07-26 additions — backbone-CAPACITY experiments, distinct from the
    # injection mechanisms above (those all still funnel through a plain
    # linear seq_len->pred_len map; these change that map itself). Added
    # because the remaining bottleneck looks like DLinear's bare-linear
    # capacity ceiling rather than the injection. All three
    # default OFF and are bit-identical to the original when unused.
    #
    #   use_mlp_backbone       — replace linear_trend/linear_seasonal (each a
    #     single Linear(seq_len,pred_len), no nonlinearity at all) with a
    #     2-layer MLP (Linear->GELU->Linear). Gives the seq_len->pred_len map
    #     actual nonlinear capacity for the first time. Loses the original's
    #     "1/seq_len init ≈ moving-average" cold start since an MLP can't
    #     replicate that trick — standard init instead.
    #
    #   use_residual_head      — keeps linear_trend/linear_seasonal and their
    #     cold-start init EXACTLY as before; adds one small nonlinear residual
    #     block AFTER seasonal_out+trend_out are combined. Zero-initialized
    #     final layer so residual contributes exactly 0 at step 0 (same
    #     cold-start philosophy as the gates above) — safest of the three
    #     since it can only ever add capacity on top of the proven default.
    #
    #   use_multiscale_decomp  — normal DLinear decomposes with ONE fixed
    #     moving_avg kernel. This runs several kernel sizes in parallel and
    #     learns a softmax-weighted combination, targeting price series'
    #     multiple periodicities (daily + weekly) that a single trend
    #     definition can't capture. Still feeds into linear_trend/seasonal
    #     unchanged — only the decomposition step is extended.
    def __init__(self, seq_len, pred_len, moving_avg=25,
                 use_mlp_backbone=False, mlp_hidden_dim=64,
                 use_residual_head=False, residual_hidden_dim=64,
                 use_multiscale_decomp=False, multiscale_kernels=(13, 25, 51)):
        super().__init__()
        self.seq_len  = seq_len
        self.pred_len = pred_len

        self.use_mlp_backbone      = use_mlp_backbone
        self.use_residual_head     = use_residual_head
        self.use_multiscale_decomp = use_multiscale_decomp

        if use_multiscale_decomp:
            self.decomps = nn.ModuleList([SeriesDecomposition(kernel_size=k) for k in multiscale_kernels])
            self.multiscale_weights = nn.Parameter(torch.zeros(len(multiscale_kernels)))
            # kept for shape/back-compat introspection elsewhere; unused when multiscale is on
            self.decomp = self.decomps[0]
        else:
            self.decomp = SeriesDecomposition(kernel_size=moving_avg)

        if use_mlp_backbone:
            self.linear_trend = nn.Sequential(
                nn.Linear(seq_len, mlp_hidden_dim), nn.GELU(), nn.Linear(mlp_hidden_dim, pred_len))
            self.linear_seasonal = nn.Sequential(
                nn.Linear(seq_len, mlp_hidden_dim), nn.GELU(), nn.Linear(mlp_hidden_dim, pred_len))
        else:
            self.linear_trend    = nn.Linear(seq_len, pred_len)
            self.linear_seasonal = nn.Linear(seq_len, pred_len)
            nn.init.constant_(self.linear_trend.weight,    1.0 / seq_len)
            nn.init.constant_(self.linear_seasonal.weight, 1.0 / seq_len)

        if use_residual_head:
            self.residual_head = nn.Sequential(
                nn.Linear(pred_len, residual_hidden_dim), nn.GELU(), nn.Linear(residual_hidden_dim, pred_len))
            # zero-init final layer -> residual contributes exactly 0 at step 0
            nn.init.zeros_(self.residual_head[-1].weight)
            nn.init.zeros_(self.residual_head[-1].bias)

        # New: independent gates for trend/seasonal post-decomposition
        # injection.
        self.trend_gate    = nn.Parameter(torch.tensor(-2.0))
        self.seasonal_gate = nn.Parameter(torch.tensor(-2.0))
        # New: shared gate for post-linear FiLM (trend and seasonal branches
        # share one gate — they're modulated by independently-projected
        # gamma/beta, so a shared gate just controls overall FiLM strength).
        self.film_gate = nn.Parameter(torch.tensor(-2.0))
        # New: gate bounding how far the causal-derived mixing weight can
        # deviate from the original unweighted 1:1 sum.
        self.mix_gate = nn.Parameter(torch.tensor(-2.0))
        # 2026-07-24 additions — see module docstring.
        self.crossattn_film_gate = nn.Parameter(torch.tensor(-2.0))
        self.hyper_gate          = nn.Parameter(torch.tensor(-2.0))
        self.scale_gate          = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x, causal_bias=None, trend_bias=None, seasonal_bias=None,
                scale_bias=None,
                film_gamma_trend=None, film_beta_trend=None,
                film_gamma_seasonal=None, film_beta_seasonal=None,
                crossattn_gamma_trend=None, crossattn_beta_trend=None,
                crossattn_gamma_seasonal=None, crossattn_beta_seasonal=None,
                mix_logit=None,
                hyper_u_trend=None, hyper_v_trend=None,
                hyper_u_seasonal=None, hyper_v_seasonal=None):
        """
        x                : [B, seq_len, 1]  (already normalized)
        causal_bias      : [B, seq_len, 1]  optional — scalar injection (pre-decomp)
        trend_bias       : [B, seq_len, 1]  optional — dual injection (trend)
        seasonal_bias    : [B, seq_len, 1]  optional — dual injection (seasonal)
        scale_bias       : [B, seq_len, 1]  optional — gated_scale injection (pre-decomp, multiplicative)
        film_gamma/beta_*: [B, 1, pred_len] optional — post-linear FiLM (mean-pooled)
        crossattn_gamma/beta_*: [B, 1, pred_len] optional — post-linear FiLM (attention-pooled)
        mix_logit        : [B, 1, 1]        optional — causal-gated mixing
        hyper_u/v_*      : [B, pred_len, r] / [B, seq_len, r] optional — low-rank weight delta
        returns          : [B, pred_len]
        """
        if causal_bias is not None:
            x = x + causal_bias
        if scale_bias is not None:
            x = x * (1 + torch.sigmoid(self.scale_gate) * torch.tanh(scale_bias))

        if self.use_multiscale_decomp:
            weights = F.softmax(self.multiscale_weights, dim=0)
            seasonal, trend = None, None
            for w, d in zip(weights, self.decomps):
                s_i, t_i = d(x)
                seasonal = w * s_i if seasonal is None else seasonal + w * s_i
                trend    = w * t_i if trend    is None else trend    + w * t_i
        else:
            seasonal, trend = self.decomp(x)  # each [B, seq_len, 1]

        if trend_bias is not None:
            trend = trend + torch.sigmoid(self.trend_gate) * trend_bias
        if seasonal_bias is not None:
            seasonal = seasonal + torch.sigmoid(self.seasonal_gate) * seasonal_bias

        seasonal = seasonal.permute(0, 2, 1)  # [B, 1, seq_len]
        trend    = trend.permute(0, 2, 1)

        seasonal_out = self.linear_seasonal(seasonal)  # [B, 1, pred_len]
        trend_out    = self.linear_trend(trend)

        if hyper_u_trend is not None and hyper_v_trend is not None:
            # deltaW: [B, pred_len, seq_len], matching linear_trend.weight's shape
            deltaW = torch.bmm(hyper_u_trend, hyper_v_trend.transpose(1, 2))
            delta_out = torch.bmm(trend, deltaW.transpose(1, 2))  # [B,1,pred_len]
            trend_out = trend_out + torch.sigmoid(self.hyper_gate) * delta_out
        if hyper_u_seasonal is not None and hyper_v_seasonal is not None:
            deltaW = torch.bmm(hyper_u_seasonal, hyper_v_seasonal.transpose(1, 2))
            delta_out = torch.bmm(seasonal, deltaW.transpose(1, 2))
            seasonal_out = seasonal_out + torch.sigmoid(self.hyper_gate) * delta_out

        if film_gamma_seasonal is not None and film_beta_seasonal is not None:
            film_g = torch.sigmoid(self.film_gate)
            seasonal_out = seasonal_out * (1 + film_g * film_gamma_seasonal) + film_g * film_beta_seasonal
        if film_gamma_trend is not None and film_beta_trend is not None:
            film_g = torch.sigmoid(self.film_gate)
            trend_out = trend_out * (1 + film_g * film_gamma_trend) + film_g * film_beta_trend

        if crossattn_gamma_seasonal is not None and crossattn_beta_seasonal is not None:
            ca_g = torch.sigmoid(self.crossattn_film_gate)
            seasonal_out = seasonal_out * (1 + ca_g * crossattn_gamma_seasonal) + ca_g * crossattn_beta_seasonal
        if crossattn_gamma_trend is not None and crossattn_beta_trend is not None:
            ca_g = torch.sigmoid(self.crossattn_film_gate)
            trend_out = trend_out * (1 + ca_g * crossattn_gamma_trend) + ca_g * crossattn_beta_trend

        if mix_logit is not None:
            mix_w = torch.tanh(mix_logit) * torch.sigmoid(self.mix_gate)  # [B,1,1], in (-1,1)
            out = (1 + mix_w) * seasonal_out + (1 - mix_w) * trend_out
        else:
            out = seasonal_out + trend_out

        if self.use_residual_head:
            out = out + self.residual_head(out)

        return out.squeeze(1)  # [B, pred_len]


# MAGNet's NGC module always feeds each exogenous variable from the patch ONE step
# earlier than the endogenous patch it explains (temporal precedence). Not configurable.
NGC_LAG = 1


class MAGNETWithDLinear(nn.Module):
    def __init__(self, magnet_config: dict, dlinear_config: dict, mode: int = 2,
                 use_text: bool = True,
                 use_scalar_injection: bool = True,
                 use_dual_injection: bool = False,
                 use_postlinear_film_injection: bool = False,
                 use_gated_mixing_injection: bool = False,
                 use_crossattn_injection: bool = False,
                 use_hyper_weight_injection: bool = False,
                 use_gated_scale_injection: bool = False,
                 crossattn_n_queries: int = 1,
                 use_instance_norm: bool = True,
                 use_mlp_backbone: bool = False, mlp_hidden_dim: int = 64,
                 use_residual_head: bool = False, residual_hidden_dim: int = 64,
                 use_multiscale_decomp: bool = False, multiscale_kernels: tuple = (13, 25, 51),
                 use_crossattn_deep_film: bool = False, crossattn_film_hidden_dim: int = 64):
        super().__init__()
        if magnet_config.get('ngc_lag', NGC_LAG) != NGC_LAG:
            raise ValueError(f"MAGNet uses ngc_lag={NGC_LAG} in every configuration; "
                             f"got ngc_lag={magnet_config['ngc_lag']}")
        self.mode = mode
        self.use_instance_norm = use_instance_norm
        self.use_scalar_injection          = use_scalar_injection
        self.use_dual_injection            = use_dual_injection
        self.use_postlinear_film_injection = use_postlinear_film_injection
        self.use_gated_mixing_injection    = use_gated_mixing_injection
        self.use_crossattn_injection       = use_crossattn_injection
        self.use_hyper_weight_injection    = use_hyper_weight_injection
        self.use_gated_scale_injection     = use_gated_scale_injection

        seq_len    = dlinear_config['seq_len']
        pred_len   = dlinear_config['pred_len']
        moving_avg = dlinear_config.get('moving_avg', 25)
        self._seq_len  = seq_len
        self._pred_len = pred_len

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
                ngc_lag        = NGC_LAG,
                use_gvar_causal           = magnet_config.get('use_gvar_causal', False),
                use_sparse_granger_causal = magnet_config.get('use_sparse_granger_causal', False),
                sparse_granger_proximal   = magnet_config.get('sparse_granger_proximal', False),
                sparse_granger_contemporaneous = magnet_config.get('sparse_granger_contemporaneous', False),
            )

            if self.use_scalar_injection:
                self.graph_proj = nn.Sequential(nn.Linear(d_graph, 16), nn.GELU(), nn.Linear(16, 1))
                self.bias_scale = nn.Parameter(torch.tensor(0.1))
            else:
                self.graph_proj = None
                self.bias_scale = None

            if self.use_dual_injection:
                self.trend_proj    = nn.Sequential(nn.Linear(d_graph, 16), nn.GELU(), nn.Linear(16, 1))
                self.seasonal_proj = nn.Sequential(nn.Linear(d_graph, 16), nn.GELU(), nn.Linear(16, 1))
            else:
                self.trend_proj = self.seasonal_proj = None

            if self.use_postlinear_film_injection:
                # Pool over patches -> [B, d_graph] -> per-branch (gamma, beta) at pred_len resolution.
                self.film_gamma_trend_proj    = nn.Linear(d_graph, pred_len)
                self.film_beta_trend_proj     = nn.Linear(d_graph, pred_len)
                self.film_gamma_seasonal_proj = nn.Linear(d_graph, pred_len)
                self.film_beta_seasonal_proj  = nn.Linear(d_graph, pred_len)
                for proj in (self.film_gamma_trend_proj, self.film_beta_trend_proj,
                             self.film_gamma_seasonal_proj, self.film_beta_seasonal_proj):
                    nn.init.zeros_(proj.weight); nn.init.zeros_(proj.bias)
            else:
                self.film_gamma_trend_proj = self.film_beta_trend_proj = None
                self.film_gamma_seasonal_proj = self.film_beta_seasonal_proj = None

            if self.use_gated_mixing_injection:
                self.mix_proj = nn.Linear(d_graph, 1)
            else:
                self.mix_proj = None

            if self.use_crossattn_injection:
                n_heads_ca = magnet_config.get('n_heads', 4)
                self.crossattn = nn.MultiheadAttention(
                    embed_dim=d_graph, num_heads=n_heads_ca, batch_first=True,
                )
                self.crossattn_n_queries = crossattn_n_queries
                self.crossattn_query = nn.Parameter(torch.randn(1, crossattn_n_queries, d_graph) * 0.02)
                # use_crossattn_deep_film: at long horizons, a bare Linear(d_graph,
                # pred_len) is a huge, abrupt expansion (e.g. 32->360 for Sdwpfh2/
                # long) -- underdetermined and prone to noisy modulation. This adds
                # a hidden layer to soften that jump. Only the FINAL layer is
                # zero-init (same cold-start contract: FiLM contributes exactly 0
                # at step 0) -- the hidden layer needs its normal random init to be
                # a useful nonlinear basis at all.
                if use_crossattn_deep_film:
                    def _make_film_proj():
                        proj = nn.Sequential(nn.Linear(d_graph, crossattn_film_hidden_dim), nn.GELU(),
                                              nn.Linear(crossattn_film_hidden_dim, pred_len))
                        nn.init.zeros_(proj[-1].weight); nn.init.zeros_(proj[-1].bias)
                        return proj
                    self.ca_gamma_trend_proj    = _make_film_proj()
                    self.ca_beta_trend_proj     = _make_film_proj()
                    self.ca_gamma_seasonal_proj = _make_film_proj()
                    self.ca_beta_seasonal_proj  = _make_film_proj()
                else:
                    self.ca_gamma_trend_proj    = nn.Linear(d_graph, pred_len)
                    self.ca_beta_trend_proj     = nn.Linear(d_graph, pred_len)
                    self.ca_gamma_seasonal_proj = nn.Linear(d_graph, pred_len)
                    self.ca_beta_seasonal_proj  = nn.Linear(d_graph, pred_len)
                    for proj in (self.ca_gamma_trend_proj, self.ca_beta_trend_proj,
                                 self.ca_gamma_seasonal_proj, self.ca_beta_seasonal_proj):
                        nn.init.zeros_(proj.weight); nn.init.zeros_(proj.bias)
            else:
                self.crossattn = None
                self.crossattn_query = None
                self.ca_gamma_trend_proj = self.ca_beta_trend_proj = None
                self.ca_gamma_seasonal_proj = self.ca_beta_seasonal_proj = None

            if self.use_hyper_weight_injection:
                # LoRA-style cold start: V (input-side) left at default random
                # init, U (output-side) zero-init, so deltaW=U@V^T=0 exactly
                # at step 0 but dL/dU = dL/d(deltaW) @ V is nonzero from the
                # first step (V!=0) — avoids the dead zone of zero-initing
                # both factors (see module docstring).
                self.hyper_rank = 4
                r = self.hyper_rank
                self.hyper_u_trend_proj    = nn.Linear(d_graph, pred_len * r)
                self.hyper_v_trend_proj    = nn.Linear(d_graph, seq_len * r)
                self.hyper_u_seasonal_proj = nn.Linear(d_graph, pred_len * r)
                self.hyper_v_seasonal_proj = nn.Linear(d_graph, seq_len * r)
                nn.init.zeros_(self.hyper_u_trend_proj.weight);    nn.init.zeros_(self.hyper_u_trend_proj.bias)
                nn.init.zeros_(self.hyper_u_seasonal_proj.weight); nn.init.zeros_(self.hyper_u_seasonal_proj.bias)
            else:
                self.hyper_rank = None
                self.hyper_u_trend_proj = self.hyper_v_trend_proj = None
                self.hyper_u_seasonal_proj = self.hyper_v_seasonal_proj = None

            if self.use_gated_scale_injection:
                self.scale_proj = nn.Sequential(nn.Linear(d_graph, 16), nn.GELU(), nn.Linear(16, 1))
            else:
                self.scale_proj = None

            self._seq_len_for_interp = seq_len
        else:
            self.magnet_graph = None
            self.graph_proj = self.bias_scale = None
            self.trend_proj = self.seasonal_proj = None
            self.film_gamma_trend_proj = self.film_beta_trend_proj = None
            self.film_gamma_seasonal_proj = self.film_beta_seasonal_proj = None
            self.mix_proj = None
            self.crossattn = None
            self.crossattn_query = None
            self.ca_gamma_trend_proj = self.ca_beta_trend_proj = None
            self.ca_gamma_seasonal_proj = self.ca_beta_seasonal_proj = None
            self.hyper_rank = None
            self.hyper_u_trend_proj = self.hyper_v_trend_proj = None
            self.hyper_u_seasonal_proj = self.hyper_v_seasonal_proj = None
            self.scale_proj = None

        self.backbone = DLinearBackbone(
            seq_len=seq_len, pred_len=pred_len, moving_avg=moving_avg,
            use_mlp_backbone=use_mlp_backbone, mlp_hidden_dim=mlp_hidden_dim,
            use_residual_head=use_residual_head, residual_hidden_dim=residual_hidden_dim,
            use_multiscale_decomp=use_multiscale_decomp, multiscale_kernels=multiscale_kernels)

    def _project_and_upsample(self, graph_bias_raw, proj_module):
        proj = proj_module(graph_bias_raw)  # [B, P, 1]
        full = F.interpolate(
            proj.permute(0, 2, 1), size=self._seq_len, mode='linear', align_corners=False,
        ).permute(0, 2, 1)  # [B, seq_len, 1]
        return full

    def forward(
        self,
        exo_patches   : torch.Tensor,
        endo_patches  : torch.Tensor,
        exog_future   : torch.Tensor = None,  # must stay None: no future exogenous values, ever
        time_features : torch.Tensor = None,
        dataset       : int = None,
    ) -> torch.Tensor:
        if exog_future is not None:
            raise ValueError("MAGNet+DLinear does not accept future exogenous values; "
                             "forecasts are conditioned on the observed history only.")

        B, P, pl, _ = endo_patches.shape

        endo_history = endo_patches.reshape(B, P * pl, 1)

        if self.use_instance_norm:
            means = endo_history.mean(dim=1, keepdim=True).detach()
            stdev = torch.sqrt(
                torch.var(endo_history, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            endo_norm = (endo_history - means) / stdev
            m = means[:, 0, 0].unsqueeze(1)
            s = stdev[:, 0, 0].unsqueeze(1)
        else:
            # Data is already globally StandardScaler-normalized at load time
            # (MAGNET_data.py fits on train, applies to train/val/test) -- skip
            # the extra per-window instance-norm layer, matching common
            # benchmark convention (single normalization, not stacked). No-op denorm (m=0, s=1).
            endo_norm = endo_history
            m = torch.zeros(B, 1, device=endo_history.device, dtype=endo_history.dtype)
            s = torch.ones(B, 1, device=endo_history.device, dtype=endo_history.dtype)

        if self.mode == 3:
            out = self.backbone(endo_norm)
            return out * s + m

        graph_bias_raw = self.magnet_graph(exo_patches, endo_patches)  # [B, P, d_graph]

        if self.use_scalar_injection:
            causal_bias = torch.sigmoid(self.bias_scale) * self._project_and_upsample(graph_bias_raw, self.graph_proj)
        else:
            causal_bias = None

        if self.use_dual_injection:
            trend_bias    = self._project_and_upsample(graph_bias_raw, self.trend_proj)
            seasonal_bias = self._project_and_upsample(graph_bias_raw, self.seasonal_proj)
        else:
            trend_bias = seasonal_bias = None

        if self.use_postlinear_film_injection:
            graph_pooled = graph_bias_raw.mean(dim=1)  # [B, d_graph]
            film_gamma_trend    = self.film_gamma_trend_proj(graph_pooled).unsqueeze(1)    # [B,1,pred_len]
            film_beta_trend     = self.film_beta_trend_proj(graph_pooled).unsqueeze(1)
            film_gamma_seasonal = self.film_gamma_seasonal_proj(graph_pooled).unsqueeze(1)
            film_beta_seasonal  = self.film_beta_seasonal_proj(graph_pooled).unsqueeze(1)
        else:
            film_gamma_trend = film_beta_trend = film_gamma_seasonal = film_beta_seasonal = None

        if self.use_gated_mixing_injection:
            graph_pooled_mix = graph_bias_raw.mean(dim=1)  # [B, d_graph]
            mix_logit = self.mix_proj(graph_pooled_mix).unsqueeze(1)  # [B,1,1]
        else:
            mix_logit = None

        if self.use_crossattn_injection:
            B_ = graph_bias_raw.shape[0]
            query = self.crossattn_query.expand(B_, -1, -1)  # [B, n_queries, d_graph]
            context, _ = self.crossattn(query, graph_bias_raw, graph_bias_raw)
            # n_queries>1: distinct learned queries can specialize on different
            # aspects of the patch sequence during attention (nonlinear softmax),
            # then get pooled -- richer than a single query even after averaging.
            context = context.mean(dim=1)  # [B, d_graph]
            crossattn_gamma_trend    = self.ca_gamma_trend_proj(context).unsqueeze(1)
            crossattn_beta_trend     = self.ca_beta_trend_proj(context).unsqueeze(1)
            crossattn_gamma_seasonal = self.ca_gamma_seasonal_proj(context).unsqueeze(1)
            crossattn_beta_seasonal  = self.ca_beta_seasonal_proj(context).unsqueeze(1)
        else:
            crossattn_gamma_trend = crossattn_beta_trend = None
            crossattn_gamma_seasonal = crossattn_beta_seasonal = None

        if self.use_hyper_weight_injection:
            graph_pooled_hyper = graph_bias_raw.mean(dim=1)  # [B, d_graph]
            B_ = graph_pooled_hyper.shape[0]
            r = self.hyper_rank
            hyper_u_trend    = self.hyper_u_trend_proj(graph_pooled_hyper).view(B_, self._pred_len, r)
            hyper_v_trend    = self.hyper_v_trend_proj(graph_pooled_hyper).view(B_, self._seq_len, r)
            hyper_u_seasonal = self.hyper_u_seasonal_proj(graph_pooled_hyper).view(B_, self._pred_len, r)
            hyper_v_seasonal = self.hyper_v_seasonal_proj(graph_pooled_hyper).view(B_, self._seq_len, r)
        else:
            hyper_u_trend = hyper_v_trend = hyper_u_seasonal = hyper_v_seasonal = None

        if self.use_gated_scale_injection:
            scale_bias = self._project_and_upsample(graph_bias_raw, self.scale_proj)
        else:
            scale_bias = None

        out = self.backbone(
            endo_norm, causal_bias=causal_bias, trend_bias=trend_bias, seasonal_bias=seasonal_bias,
            scale_bias=scale_bias,
            film_gamma_trend=film_gamma_trend, film_beta_trend=film_beta_trend,
            film_gamma_seasonal=film_gamma_seasonal, film_beta_seasonal=film_beta_seasonal,
            crossattn_gamma_trend=crossattn_gamma_trend, crossattn_beta_trend=crossattn_beta_trend,
            crossattn_gamma_seasonal=crossattn_gamma_seasonal, crossattn_beta_seasonal=crossattn_beta_seasonal,
            mix_logit=mix_logit,
            hyper_u_trend=hyper_u_trend, hyper_v_trend=hyper_v_trend,
            hyper_u_seasonal=hyper_u_seasonal, hyper_v_seasonal=hyper_v_seasonal,
        )
        return out * s + m
