# inserting endo node
# causality metric (exo->endo) and one-way endo node learning

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

_CAUSAL_TEMP = float(os.environ.get("MAGNET_CAUSAL_TEMP", "1"))

class CausalWeightFused(nn.Module):
    """
    Exo-endo causal weights fused with Rq text.
 
    Replaces the NGC softmax output in endo_insertion.py
    by additionally fusing with relationship-level text.
 
    For each exo variable i:
        numerical = ngc_score_i                          scalar
        text      = rq_emb[exo_i_global, endo_global]   [d_model]
        fused     = MLP(concat(numerical, text))         scalar
 
    Then softmax over exo dimension.

    Args:
        n_exo      : number of exo variables
        d_model    : embedding dimension
        hidden_dim : MLP hidden size
        learn_temperature : EXPERIMENTAL (2026-09-14, see
                    diagnose_ngc_causal_weights_collapse.py / conversation):
                    the collapse diagnostic found causal_weights near-uniform
                    (norm_entropy ~0.95-0.98) for several configs even after
                    the ngc_window structural fix -- consistent with
                    causal_scores being a small, close-together raw MSE-gap
                    that softmax maps to ~uniform regardless of whether the
                    underlying ranking is informative, combined with no
                    training pressure on the scores' SCALE (only an indirect
                    gradient through the downstream forecast loss). When
                    True, adds a single learnable scalar that multiplies the
                    raw scores before softmax -- init 1.0 so behavior is
                    identical to before at step 0, letting gradient descent
                    find whatever scale best serves the downstream loss
                    instead of guessing a fixed temperature. Default False =
                    exact previous behavior (no new parameter at all).

    Input:
        ngc_scores : [B, n_exo]  raw NGC scores (before softmax)
        rq_emb     : [n_vars, n_vars, d_model]

    Output:
        causal_weights : [B, n_exo, 1]
    """

    def __init__(self, n_exo: int, d_model: int, hidden_dim: int = 32,
                 learn_temperature: bool = False):
        super().__init__()
        self.n_exo = n_exo
        self.learn_temperature = learn_temperature
        if learn_temperature:
            self.temperature_scale = nn.Parameter(torch.tensor(1.0))

        # shared RqEdgeFusion for all exo-endo pairs
        self.mlp = nn.Sequential(
            nn.Linear(1 + d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        ngc_scores : torch.Tensor,   # [B, n_exo]  raw scores before softmax
        rq_emb     : torch.Tensor,   # [n_vars, n_vars, d_model]
    ) -> torch.Tensor:

        B, n_exo = ngc_scores.shape
        if rq_emb is not None:
            rq_causal = rq_emb[1:, 0]                    # [n_exo, d_model]
            num = ngc_scores.unsqueeze(-1)               # [B, n_exo, 1]
            txt = rq_causal.unsqueeze(0).expand(B, -1, -1)  # [B, n_exo, d_model]
            combined = torch.cat([num, txt], dim=-1)     # [B, n_exo, 1+d_model]
            fused_scores = self.mlp(combined).squeeze(-1)  # [B, n_exo]
            if self.learn_temperature:
                fused_scores = fused_scores * self.temperature_scale
            causal_weights = F.softmax(fused_scores, dim=1).unsqueeze(-1)
        else:
            scores = ngc_scores * self.temperature_scale if self.learn_temperature else ngc_scores
            causal_weights = F.softmax(scores, dim=1).unsqueeze(-1)

        return causal_weights
    
class _DLinear(nn.Module):
    """
    DLinear-inspired backbone adapted for patch embeddings.
 
    Original DLinear (Zeng et al., 2023) decomposes a time series into
    trend + residual and applies a separate linear layer to each.
 
    Adaptation for NGC (operating on d_model features, not raw time steps):
      - trend    branch : linear(d_in → d_model) on smoothed input
      - seasonal branch : linear(d_in → d_model) on residual
      - output          : trend + seasonal
 
    This is lightweight (2 linear layers, no nonlinearity) and directly
    comparable to the original DLinear design philosophy.
 
    Input  : [B, d_in]
    Output : [B, d_model]
    """
    def __init__(self, d_in: int, d_model: int):
        super().__init__()
        self.trend_linear    = nn.Linear(d_in, d_model)
        self.seasonal_linear = nn.Linear(d_in, d_model)
 
        # learned smooth kernel to extract trend component
        # kernel_size=3 gives a minimal moving-average-like decomposition
        self.kernel_size = min(3, d_in)
        padding = self.kernel_size // 2
        self.avg_pool = nn.AvgPool1d(
            kernel_size=self.kernel_size,
            stride=1,
            padding=padding,
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, d_in]
        # decompose along feature dimension
        # trend: smoothed version via average pooling over features
        trend    = self.avg_pool(x.unsqueeze(1)).squeeze(1)  # [B, d_in]
        # trim if avg_pool adds extra dim due to padding
        trend    = trend[:, :x.shape[1]]
        seasonal = x - trend                                  # [B, d_in]
 
        out = self.trend_linear(trend) + self.seasonal_linear(seasonal)
        return out   # [B, d_model]
    
class _Mixer(nn.Module):
    """
    MLP-Mixer-inspired backbone adapted for NGC patch embeddings.
 
    Original MLP-Mixer (Tolstikhin et al., 2021) alternates between:
      (1) token-mixing  MLP : mixes across sequence positions
      (2) channel-mixing MLP: mixes across feature dimensions
 
    Adaptation for NGC (single vector input [B, d_in], no sequence dim):
      - We treat d_in as the 'token' dimension
      - channel-mixing: MLP over d_in features → projects to d_model
      - token-mixing:   MLP over d_model → same d_model (self-mixing)
      - both with LayerNorm and residual connections
 
    Input  : [B, d_in]
    Output : [B, d_model]
    """
    def __init__(self, d_in: int, hidden_dim: int, d_model: int):
        super().__init__()
 
        # channel-mixing: projects from d_in to d_model
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in,    hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
 
        # token-mixing: self-mixing within d_model space
        self.token_mix = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
 
        self.norm_out = nn.LayerNorm(d_model)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # channel mix: [B, d_in] → [B, d_model]
        out = self.channel_mix(x)          # [B, d_model]
 
        # token mix with residual: [B, d_model] → [B, d_model]
        out = out + self.token_mix(out)    # [B, d_model]
 
        return self.norm_out(out)          # [B, d_model]
 
 
class _TransformerBackbone(nn.Module):
    """
    Lightweight single-layer Transformer encoder adapted for NGC.
 
    Since NGC operates on single patch embeddings [B, d_model] rather
    than sequences, we use a minimal Transformer with:
      - positional encoding via learned linear projection
      - 1 self-attention layer (single encoder block)
      - feedforward sublayer
      - LayerNorm + residual (standard Transformer block)
 
    The input [B, d_in] is treated as a single-token sequence.
    Self-attention over a single token degenerates to a gated projection,
    which is equivalent to a learned weighted transformation of the input.
    This keeps it lightweight while matching the Transformer architecture.
 
    For the unrestricted path, d_in = d_model*2 (endo + exo concatenated),
    so the Transformer sees a richer input and can learn to gate which
    part of the combined representation is more predictive of endo.
 
    Input  : [B, d_in]
    Output : [B, d_model]
    """
    def __init__(
        self,
        d_in      : int,
        d_model   : int,
        n_heads   : int = 4,
        ffn_dim   : int = 128,
        dropout   : float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
 
        # project input to d_model (handles d_in ≠ d_model)
        self.input_proj = nn.Linear(d_in, d_model)
 
        # single Transformer encoder layer
        self.attn_layer = nn.TransformerEncoderLayer(
            d_model   = d_model,
            nhead     = n_heads,
            dim_feedforward = ffn_dim,
            dropout   = dropout,
            batch_first = True,     # [B, seq, d] convention
            activation  = 'gelu',
        )
 
        self.norm = nn.LayerNorm(d_model)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, d_in]
        # project to d_model and add sequence dimension
        out = self.input_proj(x).unsqueeze(1)   # [B, 1, d_model]
 
        # Transformer encoder (single token — attention is a gated projection)
        out = self.attn_layer(out)               # [B, 1, d_model]
 
        # remove sequence dimension
        out = out.squeeze(1)                     # [B, d_model]
        return self.norm(out)

# causality measure using Neural Granger Causality
class NeuralGrangerCausality(nn.Module):
    """
    Computes directed causal weights from each exo variable to endo
    using Neural Granger Causality on ENCODED patch embeddings from TSPreprocessor.
 
    Each patch embedding [d_model] is a rich learned summary of patch_len
    raw timesteps - richer than raw values for causality estimation.
 
      restricted  : LSTM(endo_emb)              -> pred_endo_emb
      unrestricted: LSTM(endo_emb + exo_i_emb)  -> pred_endo_emb
 
      causal_score_i = MSE(restricted) - MSE(unrestricted)
      -> positive = exo_i helps predict endo = causal
 
    Runs per patch -> each graph gets its OWN causal weights.
 
    Optimization: instead of n_exo separate LSTM calls (sequential),
    we merge the exo dimension into the batch dimension and run ONE
    shared LSTM call — GPU processes all exo variables in parallel.
 
      Before: n_exo LSTM calls × kernel overhead = slow
      After:  1 LSTM call on [B*n_exo, 1, d_model*2] = fast
 
    The shared weights are a reasonable inductive bias — the mechanism
    for detecting causality is the same regardless of which exo variable
    we're testing. Inputs still differ per exo var so outputs differ too.
 
    Args:
        n_exo      : number of exogenous variables
        d_model    : patch embedding dimension
        hidden_dim : LSTM hidden size
 
    Input:
        exo_emb_ts  : [B, n_exo, d_model]
        endo_emb_ts : [B, 1,     d_model]
 
    Output:
        causal_scores : [B, n_exo]  raw scores (before softmax)
    """
 
    def __init__(self, n_exo: int, d_model: int, hidden_dim: int = 32,
                 use_current_and_lag: bool = False,
                 classical_prior: torch.Tensor = None,  # EXPERIMENTAL (2026-09-15): see
                             # NeuralGrangerCausalityWindowed's classical_prior docstring --
                             # same mechanism, ported to this (non-windowed, ngc_lag-based)
                             # class so it can be tested directly on the full_lag1 base
                             # instead of only on top of the w5c window+contemporaneous
                             # base. When set, forward(return_aux=True) computes a
                             # KL-distillation aux loss against this prior INSTEAD of the
                             # older reconstruction-quality aux loss below (that one is
                             # kept as the fallback when classical_prior is None, for any
                             # existing caller -- exact previous behavior, unaffected).
                 ):
        super().__init__()
        self.n_exo     = n_exo
        self.hidden_dim = hidden_dim
        if classical_prior is not None:
            self.register_buffer('classical_prior', torch.as_tensor(classical_prior, dtype=torch.float32))
        else:
            self.classical_prior = None
        # EXPERIMENTAL (2026-09-13): when True, the unrestricted branch gets
        # BOTH exo(p) and exo(p-lag) concatenated (not just one), so it can
        # detect trend/change in exo, not only a fixed-lag level. Widens
        # lstm_unrestricted's input accordingly. See forward()'s
        # exo_emb_ts_extra param and MAGNETGraphOnly's ngc_lag docstring.
        # NOTE: this makes unrestricted's input strictly bigger than
        # restricted's by more than "has exo" -- part of any measured causal
        # gap could be a capacity artifact (more numbers to fit), not only
        # genuine causal signal. Known limitation, not yet controlled for.
        self.use_current_and_lag = use_current_and_lag

        # restricted LSTM: endo only
        self.lstm_restricted = nn.LSTM(
            input_size  = d_model,
            hidden_size = hidden_dim,
            batch_first = True,
        )

        # ONE shared unrestricted LSTM — replaces n_exo separate LSTMs
        # all exo variables processed in parallel via batch dimension expansion
        self.lstm_unrestricted = nn.LSTM(
            input_size  = d_model * (3 if use_current_and_lag else 2),
            hidden_size = hidden_dim,
            batch_first = True,
        )

        # Ablation A: using GRU
        #self.gru_restricted = nn.GRU(
        #    input_size = d_model, 
        #    hidden_size = hidden_dim, 
        #    batch_first = True)
        #self.gru_unrestricted = nn.GRU(
        #    input_size = d_model*2, 
        #    hidden_size = hidden_dim, 
        #    batch_first = True)

        # Ablation B: using MLP
        #self.mlp_restricted = nn.Sequential(
        #    nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, d_model)
        #)
        #self.mlp_unrestricted = nn.Sequential(
        #    nn.Linear(d_model*2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, d_model)
        #)

        # Ablation C: using LinearNL
        #self.linear_restricted = nn.Sequential(
        #    nn.Linear(d_model, hidden_dim),
        #    nn.ReLU(),
        #    nn.Linear(hidden_dim, d_model)
        #)

        #self.linear_unrestricted = nn.Sequential(
        #    nn.Linear(d_model*2, hidden_dim),
        #    nn.ReLU(),
        #    nn.Linear(hidden_dim, d_model)
        #)

        #self.dlinear_restricted   = _DLinear(d_model, d_model)
        #self.dlinear_unrestricted = _DLinear(d_model*2, d_model)

        #self.mixer_restricted   = _Mixer(d_model, hidden_dim, d_model)
        #self.mixer_unrestricted = _Mixer(d_model*2, hidden_dim, d_model)

        #self.trf_restricted   = _TransformerBackbone(d_model, d_model)
        #self.trf_unrestricted = _TransformerBackbone(d_model*2, d_model)
 
        # shared prediction head
        self.pred_head = nn.Linear(hidden_dim, d_model) # *2 the hidden_dim if Ablation D
 
    def forward(
        self,
        exo_emb_ts  : torch.Tensor,   # [B, n_exo, d_model]
        endo_emb_ts : torch.Tensor,   # [B, 1,     d_model]
        exo_emb_ts_extra : torch.Tensor = None,  # [B, n_exo, d_model], optional --
                          # a SECOND exo embedding (e.g. the contemporaneous
                          # patch when exo_emb_ts carries the lagged one)
                          # concatenated alongside exo_emb_ts when
                          # use_current_and_lag=True, so the unrestricted
                          # branch sees exo's level AND its lag together
                          # (trend-sensitive), not just one or the other.
        return_aux  : bool = False,   # EXPERIMENTAL: also return an auxiliary
                          # loss (see below) that directly supervises the
                          # restricted/unrestricted predictors. Default False
                          # keeps the original single-tensor return.
        aux_detach_inputs : bool = True,  # EXPERIMENTAL (2026-09-13): when
                          # return_aux=True, the FIRST attempt (aux_weight=0.1,
                          # no detach) made things worse in every one of 3
                          # test cases -- the aux loss also backprops into the
                          # SHARED upstream embedding layers (endo_proj,
                          # exo_proj, exo_node, endo_node) that the main
                          # forecast task depends on, pulling them toward
                          # what's good for NGC's reconstruction subtask
                          # instead. aux_detach_inputs=True reruns
                          # lstm_restricted/lstm_unrestricted/pred_head a
                          # SECOND time on .detach()'d copies of their inputs
                          # purely to compute aux_loss, so its gradient only
                          # reaches those three submodules' own weights, never
                          # anything upstream -- the main causal_scores path
                          # below is completely unaffected either way.
    ):

        B, n_exo, d_model = exo_emb_ts.shape

        endo_input  = endo_emb_ts # unsqueeze for 3 dimensional only              # [B, 1, d_model]
        endo_target = endo_emb_ts          # [B, 1, d_model]

        # ── restricted: endo alone ────────────────────────────────────────────
        lstm_out, _     = self.lstm_restricted(endo_input)   # [B, 1, hidden]
        pred_restricted = self.pred_head(lstm_out)            # [B, 1, d_model]
        #pred_restricted = lstm_out
        loss_restricted = F.mse_loss(
            pred_restricted, endo_target, reduction='none'
        ).mean(dim=-1).mean(dim=-1)                     # [B]

        # ── unrestricted: endo + all exo vars in ONE parallel call ────────────
        # expand endo to match all exo vars: [B, n_exo, d_model]
        endo_expanded = endo_input.expand(B, n_exo, d_model)  # [B, n_exo, d_model]

        # concat endo + exo (+ optional second exo view) along feature dim
        if self.use_current_and_lag and exo_emb_ts_extra is not None:
            combined = torch.cat([endo_expanded, exo_emb_ts, exo_emb_ts_extra], dim=-1)
        else:
            combined = torch.cat([endo_expanded, exo_emb_ts], dim=-1)
        d_in = combined.shape[-1]

        # merge batch and exo dims: [B*n_exo, 1, d_in]
        # each exo variable becomes a separate "batch item"
        combined = combined.reshape(B * n_exo, 1, d_in)

        # ONE LSTM call — GPU processes all exo vars simultaneously
        lstm_out_all, _ = self.lstm_unrestricted(combined)    # [B*n_exo, 1, hidden]
        pred_all        = self.pred_head(lstm_out_all)        # [B*n_exo, 1, d_model]
        #pred_all = lstm_out_all

        # reshape back: [B, n_exo, d_model]
        pred_all = pred_all.reshape(B, n_exo, d_model)

        # compute loss per exo variable
        # endo_target: [B, 1, d_model] -> expand to [B, n_exo, d_model]
        endo_target_expanded = endo_target.expand(B, n_exo, d_model)
        loss_unrestricted = F.mse_loss(
            pred_all, endo_target_expanded, reduction='none'
        ).mean(dim=-1)                                        # [B, n_exo]

        # causal score per exo: how much does including exo_i help?
        # loss_restricted: [B] -> [B, 1] -> broadcast to [B, n_exo]
        causal_scores = loss_restricted.unsqueeze(1) - loss_unrestricted
        #               [B, n_exo]
        # (unaffected by aux_detach_inputs -- always computed from the live,
        # non-detached tensors above)

        if return_aux:
            # EXPERIMENTAL (2026-09-13) auxiliary loss: causal_scores' only
            # gradient, previously, came indirectly through softmax(causal_scores)
            # -> aggregator attention -> backbone -> final forecast MSE, several
            # layers removed from what these two LSTMs are actually predicting.
            # That's a long, weak, noisy path -- directly supervising both
            # branches' own reconstruction-of-endo objective gives them a clean
            # signal independent of that chain, so the restricted/unrestricted
            # GAP (what causal_scores actually is) reflects real fit quality
            # rather than whatever the downstream task's indirect gradient
            # happened to nudge them toward.
            if aux_detach_inputs:
                # Rerun the SAME lstm_restricted/lstm_unrestricted/pred_head
                # (shared weights, intentional) on detached copies of their
                # inputs, purely for the aux loss -- .detach() stops gradient
                # from reaching endo_input/exo_emb_ts's own graph, so nothing
                # upstream of this module (exo_proj, endo_proj, exo_node,
                # endo_node) sees this loss at all. The live causal_scores
                # computation above is untouched by this branch.
                endo_input_d = endo_input.detach()
                exo_emb_ts_d = exo_emb_ts.detach()

                lstm_out_d, _ = self.lstm_restricted(endo_input_d)
                pred_restricted_d = self.pred_head(lstm_out_d)
                loss_restricted_aux = F.mse_loss(
                    pred_restricted_d, endo_input_d, reduction='none'
                ).mean(dim=-1).mean(dim=-1)

                endo_expanded_d = endo_input_d.expand(B, n_exo, d_model)
                if self.use_current_and_lag and exo_emb_ts_extra is not None:
                    combined_d = torch.cat(
                        [endo_expanded_d, exo_emb_ts_d, exo_emb_ts_extra.detach()], dim=-1)
                else:
                    combined_d = torch.cat([endo_expanded_d, exo_emb_ts_d], dim=-1)
                combined_d = combined_d.reshape(B * n_exo, 1, combined_d.shape[-1])

                lstm_out_all_d, _ = self.lstm_unrestricted(combined_d)
                pred_all_d = self.pred_head(lstm_out_all_d).reshape(B, n_exo, d_model)
                endo_target_d_expanded = endo_input_d.expand(B, n_exo, d_model)
                loss_unrestricted_aux = F.mse_loss(
                    pred_all_d, endo_target_d_expanded, reduction='none'
                ).mean(dim=-1)

                if self.classical_prior is not None:
                    causal_scores_aux = loss_restricted_aux.unsqueeze(1) - loss_unrestricted_aux
                else:
                    aux_loss = loss_restricted_aux.mean() + loss_unrestricted_aux.mean()
            else:
                if self.classical_prior is not None:
                    causal_scores_aux = causal_scores
                else:
                    aux_loss = loss_restricted.mean() + loss_unrestricted.mean()

            if self.classical_prior is not None:
                mean_scores = causal_scores_aux.mean(dim=0)                     # [n_exo]
                log_model_dist = F.log_softmax(mean_scores, dim=0)
                # log1p before softmax (EXPERIMENTAL, 2026-09-15): raw F-statistics have
                # no natural scale -- softmax of raw values makes the target distribution's
                # sharpness an artifact of the F-stats' absolute magnitude (DE's modest
                # 116 vs 98 gap and Energy's stark 1556 vs 475 gap both collapsed to a
                # near-one-hot target), not the RELATIVE strength of evidence. log1p
                # compresses that so the KL target reflects "which is more likely causal"
                # calibrated to how much more, rather than forcing near-total certainty
                # regardless of the actual gap -- see conversation: first classical-prior
                # runs got the ranking right (favorite matched classical GC every time) but
                # still cost MAE vs plain full_lag1, suspected to be this over-sharpening
                # discarding genuinely useful (if not strictly causal) blended signal.
                prior_vals = torch.log1p(self.classical_prior.to(mean_scores.device, mean_scores.dtype))
                prior_dist = F.softmax(prior_vals, dim=0)
                aux_loss = F.kl_div(log_model_dist, prior_dist, reduction='sum')
            return causal_scores, aux_loss
        return causal_scores                                  # [B, n_exo]

class NeuralGrangerCausalityWindowed(nn.Module):
    """
    Windowed variant of NeuralGrangerCausality -- fixes two structural
    issues found while investigating the causal_weights collapse diagnostic
    (see diagnose_ngc_causal_weights_collapse.py / conversation):

    1. The original restricted branch reconstructs endo(t) FROM endo(t)
       itself (endo_input == endo_target) -- a near-trivial identity task
       an LSTM+Linear solves almost perfectly regardless of exo, so
       loss_restricted collapses toward ~0 and causal_scores becomes a tiny,
       noisy residual dominated by whatever small edge the unrestricted
       branch's extra exo channel can squeeze out. This is NOT classic
       Granger causality (predict FUTURE from PAST); it's an autoencoding
       redundancy test.
    2. Both LSTMs are called on a sequence of length 1 (one patch embedding
       per call) -- with seq_len=1 an LSTM has no timesteps to recur over,
       so it's functionally just an expensive gated affine transform, never
       exploiting temporal structure at all.

    This variant restores both: restricted/unrestricted now consume a REAL
    K-patch PAST window (endo(t-K..t-1), optionally + exo(t-K..t-1)) and
    must predict the CURRENT endo(t) embedding as an actual held-out target
    -- a genuine forecasting task the LSTM can only solve by using
    information in the window, with K>1 real timesteps for the recurrence
    to matter.

    Input:
        endo_window : [B, K, d_model]         endo embeddings, patches t-K..t-1
        exo_window  : [B, K, n_exo, d_model]   exo embeddings,  patches t-K..t-1
        endo_target : [B, 1, d_model]          endo embedding at patch t (held out)
        exo_current : [B, n_exo, d_model], optional -- patch t's exo embedding
                      (CONTEMPORANEOUS with the target, unlike exo_window which
                      is strictly past). See forward() docstring for why this
                      exists -- ground-truth Granger causality checks on raw
                      data (Sdwpfm2 wind speed -> power, see conversation)
                      found the TRUE causal relationship for some drivers is
                      near-instantaneous (correlation peaks at the SAME patch,
                      decaying sharply one patch back), which a strictly-past
                      window structurally cannot see. None (default) = exact
                      previous behavior.

    Output:
        causal_scores : [B, n_exo]  raw scores (before softmax)
    """

    def __init__(self, n_exo: int, d_model: int, hidden_dim: int = 32,
                 score_type: str = 'diff',  # EXPERIMENTAL (2026-09-15): 'diff' is the
                             # exact previous behavior (raw MSE gap). 'log_ratio' instead
                             # computes log(loss_restricted+eps) - log(loss_unrestricted+eps)
                             # -- i.e. log(loss_restricted / loss_unrestricted), a RELATIVE
                             # improvement (ratio) rather than an ABSOLUTE one (difference),
                             # closer to how classical Granger causality's F-test compares
                             # residual variances via a ratio (with dof correction) rather
                             # than a raw subtraction. Motivation: the diagnostic found NGC's
                             # collapse cases have causal_scores staying small/close together
                             # across exo regardless of correctness -- a raw MSE difference
                             # is sensitive to the absolute loss SCALE (which varies a lot
                             # across datasets/windows), whereas the ratio is scale-invariant,
                             # so two exo with a genuinely different relative fit should stay
                             # separated in log-space even when absolute losses are both small.
                 classical_prior: torch.Tensor = None,  # EXPERIMENTAL (2026-09-15): optional
                             # [n_exo] fixed prior (e.g. a classical Granger-causality
                             # F-statistic per exo, computed offline on the raw series --
                             # see compute_classical_ngc_priors.py) used ONLY when
                             # forward(return_aux=True) -- see that docstring. None
                             # (default) = no aux loss available, exact previous behavior.
                 ):
        super().__init__()
        self.n_exo = n_exo
        assert score_type in ('diff', 'log_ratio')
        self.score_type = score_type
        if classical_prior is not None:
            self.register_buffer('classical_prior', torch.as_tensor(classical_prior, dtype=torch.float32))
        else:
            self.classical_prior = None

        self.lstm_restricted = nn.LSTM(
            input_size=d_model, hidden_size=hidden_dim, batch_first=True)
        self.lstm_unrestricted = nn.LSTM(
            input_size=d_model * 2, hidden_size=hidden_dim, batch_first=True)
        self.pred_head = nn.Linear(hidden_dim, d_model)

    def _compute_scores(self, endo_window: torch.Tensor, exo_window: torch.Tensor,
                         endo_target: torch.Tensor, exo_current: torch.Tensor = None) -> torch.Tensor:
        B, K, n_exo, d_model = exo_window.shape

        if exo_current is not None:
            zero_endo_step = torch.zeros(B, 1, d_model, device=endo_window.device, dtype=endo_window.dtype)
            endo_window = torch.cat([endo_window, zero_endo_step], dim=1)              # [B, K+1, d_model]
            exo_window = torch.cat([exo_window, exo_current.unsqueeze(1)], dim=1)      # [B, K+1, n_exo, d_model]
            K = K + 1

        # ── restricted: endo's own K-patch PAST predicts endo's CURRENT ──────
        lstm_out, _ = self.lstm_restricted(endo_window)          # [B, K, hidden]
        pred_restricted = self.pred_head(lstm_out[:, -1, :])      # [B, d_model] (last step only)
        loss_restricted = F.mse_loss(
            pred_restricted, endo_target.squeeze(1), reduction='none'
        ).mean(dim=-1)                                            # [B]

        # ── unrestricted: endo's + exo_i's K-patch PAST predict endo's CURRENT
        endo_expanded = endo_window.unsqueeze(1).expand(B, n_exo, K, d_model)   # [B, n_exo, K, d_model]
        exo_perm = exo_window.permute(0, 2, 1, 3)                               # [B, n_exo, K, d_model]
        combined = torch.cat([endo_expanded, exo_perm], dim=-1)                 # [B, n_exo, K, 2*d_model]
        combined = combined.reshape(B * n_exo, K, 2 * d_model)

        lstm_out_all, _ = self.lstm_unrestricted(combined)        # [B*n_exo, K, hidden]
        pred_all = self.pred_head(lstm_out_all[:, -1, :])          # [B*n_exo, d_model]
        pred_all = pred_all.reshape(B, n_exo, d_model)

        endo_target_expanded = endo_target.squeeze(1).unsqueeze(1).expand(B, n_exo, d_model)
        loss_unrestricted = F.mse_loss(
            pred_all, endo_target_expanded, reduction='none'
        ).mean(dim=-1)                                              # [B, n_exo]

        if self.score_type == 'log_ratio':
            eps = 1e-6
            causal_scores = torch.log(loss_restricted.unsqueeze(1) + eps) - torch.log(loss_unrestricted + eps)
        else:
            causal_scores = loss_restricted.unsqueeze(1) - loss_unrestricted   # [B, n_exo]
        return causal_scores

    def forward(self, endo_window: torch.Tensor, exo_window: torch.Tensor,
                endo_target: torch.Tensor, exo_current: torch.Tensor = None,
                return_aux: bool = False,  # EXPERIMENTAL (2026-09-15): when True AND
                          # self.classical_prior is set, also returns an aux_loss that
                          # directly distills the classical Granger-causality ranking into
                          # causal_scores -- a KL divergence between softmax(classical_prior)
                          # and softmax(mean causal_scores over the batch). Unlike the base
                          # NeuralGrangerCausality class's return_aux (which supervises the
                          # restricted/unrestricted branches' own reconstruction quality,
                          # independent of any external signal), this gives NGC's RANKING an
                          # actual ground-truth target instead of relying on the weak,
                          # indirect gradient through softmax->attention->forecast loss.
                aux_detach_inputs: bool = True,  # see NeuralGrangerCausality.forward's
                          # aux_detach_inputs docstring -- same gotcha applies here: without
                          # detaching, this aux loss's gradient also reaches the SHARED
                          # upstream embeddings (exo_proj/endo_proj/exo_node/endo_node),
                          # pulling them toward whatever's good for matching the prior
                          # instead of the main forecast task. True (default) reruns the
                          # score computation on detached copies purely for the aux loss, so
                          # its gradient only reaches lstm_restricted/lstm_unrestricted/
                          # pred_head's own weights.
                ) -> torch.Tensor:
        """
        exo_current (EXPERIMENTAL, 2026-09-15): when given, both branches get
        one extra timestep appended -- endo's extra step is a ZERO placeholder
        (never reveals endo(t) itself, which would reintroduce the original
        trivial-reconstruction problem this class exists to avoid), exo's
        extra step is its TRUE current-patch value. This tests "does exo's
        CONTEMPORANEOUS value help predict endo(t), beyond endo's own past" --
        instantaneous causality, alongside the existing lagged/precedence
        test -- without letting endo cheat. Both LSTMs get the same +1 extra
        timestep (restricted too, via the same zero-padded endo), so the
        comparison stays symmetric: the only difference between the two
        branches remains "has exo at all", not sequence length. None
        (default) = exact previous behavior, no extra timestep.
        """
        causal_scores = self._compute_scores(endo_window, exo_window, endo_target, exo_current)

        if not return_aux:
            return causal_scores

        if self.classical_prior is None:
            aux_loss = causal_scores.new_zeros(())
            return causal_scores, aux_loss

        if aux_detach_inputs:
            scores_for_aux = self._compute_scores(
                endo_window.detach(), exo_window.detach(), endo_target.detach(),
                exo_current.detach() if exo_current is not None else None,
            )
        else:
            scores_for_aux = causal_scores

        mean_scores = scores_for_aux.mean(dim=0)                            # [n_exo]
        log_model_dist = F.log_softmax(mean_scores, dim=0)
        # log1p before softmax -- see NeuralGrangerCausality.forward's identical fix
        # docstring (endo_insertion.py) for why: raw F-stat softmax over-sharpens the
        # KL target regardless of the actual relative strength of evidence.
        prior_vals = torch.log1p(self.classical_prior.to(mean_scores.device, mean_scores.dtype))
        prior_dist = F.softmax(prior_vals, dim=0)
        aux_loss = F.kl_div(log_model_dist, prior_dist, reduction='sum')
        return causal_scores, aux_loss


# NGC ablations
class AttentionAggregator(nn.Module):
    """
    Computes endo node embedding for ONE patch by attending over exo nodes.
 
    Asymmetric attention:
      - Endo = Query   (what exo info is relevant to me?)
      - Exo  = Key/Value (here is what we offer)
 
    Attention scores biased by NGC causal weights from Step 1.
 
    Args:
        d_model : embedding dimension
        n_heads : attention heads
        dropout : dropout rate
 
    Input:
        exo_emb        : [B, n_exo, d_model]  encoded, after ExoGAT
        endo_emb       : [B, 1,     d_model]  encoded, from TSPreprocessor
        causal_weights : [B, n_exo, 1]        from NGC for THIS patch
 
    Output:
        endo_node_emb  : [B, 1, d_model]
    """
 
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
 
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
 
        self.W_q     = nn.Linear(d_model, d_model)
        self.W_k     = nn.Linear(d_model, d_model)
        self.W_v     = nn.Linear(d_model, d_model)
        self.W_out   = nn.Linear(d_model, d_model)
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
 
    def forward(
        self,
        exo_emb        : torch.Tensor,   # [B, n_exo, d_model]
        endo_emb       : torch.Tensor,   # [B, 1,     d_model]
        causal_weights : torch.Tensor,   # [B, n_exo, 1]
        endo_text_bias : torch.Tensor = None,  # [d_model]      — added to Q only
        exo_text_bias  : torch.Tensor = None,  # [n_exo, d_model] — added to K only
        exo_gate       : torch.Tensor = None,  # scalar in (0, 1), optional -- scales the
                          # message gathered from exo before the residual (causal gate,
                          # see MAGNETGraphOnly.enable_causal_options). None = exact
                          # previous behavior.
    ) -> torch.Tensor:
        """
        endo_text_bias / exo_text_bias: optional text-derived vectors added to
        the query/key BEFORE the dot product — NOT to V. This makes text a
        routing signal (which exo variable is relevant) rather than content
        injected into what gets transmitted. Unlike adding text as a constant
        node feature (which is identical for every sample and so cannot carry
        per-sample information), biasing Q/K means text interacts with the
        actual sample content through the dot product (endo·exo_text_bias,
        exo·endo_text_bias) — a genuinely sample-dependent contribution.
        Both default to None (added to Q/K unchanged), so existing callers
        (model.py's standalone MAGNET) are completely unaffected.
        """

        B, n_exo, d_model = exo_emb.shape
        residual = endo_emb

        endo_q_in = endo_emb + endo_text_bias if endo_text_bias is not None else endo_emb
        exo_k_in  = exo_emb  + exo_text_bias  if exo_text_bias  is not None else exo_emb

        # endo is Q, exo are K and V — V stays pure numeric content
        Q = self.W_q(endo_q_in)                       # [B, 1,     d_model]
        K = self.W_k(exo_k_in)                        # [B, n_exo, d_model]
        V = self.W_v(exo_emb)                         # [B, n_exo, d_model]
 
        # split into heads
        def split_heads(x, seq_len):
            return x.reshape(B, seq_len, self.n_heads, self.d_head) \
                    .permute(0, 2, 1, 3)
 
        Q = split_heads(Q, 1)                         # [B, n_heads, 1,     d_head]
        K = split_heads(K, n_exo)                     # [B, n_heads, n_exo, d_head]
        V = split_heads(V, n_exo)                     # [B, n_heads, n_exo, d_head]
 
        # scaled dot-product attention
        scale = self.d_head ** 0.5
        attn  = torch.matmul(Q, K.transpose(-1, -2)) / scale
        #       [B, n_heads, 1, n_exo]
 
        # bias with per-patch causal weights
        # [B, n_exo, 1] -> [B, 1, 1, n_exo]
        causal_bias = causal_weights.squeeze(-1).unsqueeze(1).unsqueeze(2)
        # option 2 (2026-11-14): softer Granger steering, log-weight bias divided by MAGNET_CAUSAL_TEMP (default 1)
        attn = F.softmax(attn + causal_bias.log().clamp(min=-1e9) / _CAUSAL_TEMP, dim=-1)
 
        # endo gathers from exo
        out = torch.matmul(attn, V)                   # [B, n_heads, 1, d_head]
 
        # merge heads
        out = out.permute(0, 2, 1, 3).reshape(B, 1, d_model)
 
        # output projection + residual + norm
        out = self.W_out(out)
        if exo_gate is not None:
            out = exo_gate * out
        out = self.norm(residual + self.dropout(out))
 
        return out
    
class VqCrossAttention(nn.Module):
    """
    Cross-attention over a REAL multi-token text sequence.

    Previously this received one pooled [CLS] vector as its only key, which
    made softmax over a single key always evaluate to 1.0 — i.e. attention
    degenerated into a fixed scalar gate on one vector, with nothing to
    select between. Now the key/value sequence is the variable's full
    per-token BERT embedding (optionally padded, masked out via vq_mask),
    plus an optional extra "stats" token carrying the dynamic per-patch
    statistics — so the query can genuinely choose which sub-fact of the
    text (or the current stats) is relevant, instead of always drinking
    from a single fixed cup.
    """
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.d_model = d_model

        self.W_q    = nn.Linear(d_model, d_model)
        self.W_k    = nn.Linear(d_model, d_model)
        self.W_v    = nn.Linear(d_model, d_model)
        self.W_out  = nn.Linear(d_model, d_model)
        self.norm   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        patch_emb : torch.Tensor,             # [N, d_model]  N = B*P (flattened)
        vq_tokens : torch.Tensor,              # [T, d_model]  static per-variable text tokens (already projected), possibly padded
        vq_mask   : torch.Tensor = None,       # [T]           True = real token, False = pad
        stats_emb : torch.Tensor = None,       # [N, d_model]  optional dynamic per-patch stats, appended as one extra token
    ) -> torch.Tensor:

        N        = patch_emb.shape[0]
        T        = vq_tokens.shape[0]
        residual = patch_emb                           # save for residual

        # patch_emb as Query: [N, 1, d_model]
        Q = self.W_q(patch_emb).unsqueeze(1)

        # static text tokens, broadcast over N: [N, T, d_model]
        vq = vq_tokens.unsqueeze(0).expand(N, T, self.d_model)
        mask = vq_mask.unsqueeze(0).expand(N, T) if vq_mask is not None else None

        # dynamic per-patch stats as one extra key/value token
        if stats_emb is not None:
            stats_tok = stats_emb.unsqueeze(1)                       # [N, 1, d_model]
            kv_in     = torch.cat([vq, stats_tok], dim=1)            # [N, T+1, d_model]
            if mask is not None:
                extra = torch.ones(N, 1, dtype=torch.bool, device=mask.device)
                mask  = torch.cat([mask, extra], dim=1)              # [N, T+1]
        else:
            kv_in = vq

        K  = self.W_k(kv_in)                          # [N, T(+1), d_model]
        V  = self.W_v(kv_in)
        T_kv = K.shape[1]

        # split into heads
        def split_heads(x, seq_len):
            return x.reshape(N, seq_len, self.n_heads, self.d_head) \
                    .permute(0, 2, 1, 3)              # [N, n_heads, seq_len, d_head]

        Q = split_heads(Q, 1)
        K = split_heads(K, T_kv)
        V = split_heads(V, T_kv)

        # scaled dot-product attention over the real multi-token sequence
        scale = self.d_head ** 0.5
        attn  = torch.matmul(Q, K.transpose(-1, -2)) / scale
        #       [N, n_heads, 1, T_kv]

        if mask is not None:
            # mask out padded key positions before softmax
            mask_bias = mask.unsqueeze(1).unsqueeze(1)     # [N, 1, 1, T_kv]
            attn = attn.masked_fill(~mask_bias, float('-inf'))

        attn  = F.softmax(attn, dim=-1)               # [N, n_heads, 1, T_kv]

        # weighted value
        out   = torch.matmul(attn, V)                 # [N, n_heads, 1, d_head]

        # merge heads
        out   = out.permute(0, 2, 1, 3)               # [N, 1, n_heads, d_head]
        out   = out.reshape(N, self.d_model)          # [N, d_model]

        # output projection + residual + norm
        out   = self.W_out(out)                       # [N, d_model]
        out   = self.norm(residual + self.dropout(out))

        return out                                    # [N, d_model]
    
class GatedCausalAttention(nn.Module):
    """
    Replaces NGC with learnable cross-attention between endo and exo.
    
    Instead of LSTM-based Granger causality scores, directly computes
    how much each exo variable's embedding relates to endo via attention,
    then gates with a learned per-variable scalar.
    
    Input:
        exo_emb_ts  : [B, n_exo, d_model]
        endo_emb_ts : [B, 1,     d_model]
    Output:
        causal_weights : [B, n_exo, 1]
    """
    def __init__(self, n_exo: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.n_exo = n_exo

        # endo attends to each exo to compute relevance scores
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)

        # per-variable learned gate — how much does exo_i causally matter?
        self.gate = nn.Linear(d_model, 1)

        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale   = d_model ** 0.5

    def forward(
        self,
        exo_emb_ts  : torch.Tensor,  # [B, n_exo, d_model]
        endo_emb_ts : torch.Tensor,  # [B, 1,     d_model]
    ) -> torch.Tensor:

        B, n_exo, d_model = exo_emb_ts.shape

        # endo as query, exo as keys
        Q = self.W_q(endo_emb_ts)              # [B, 1, d_model]
        K = self.W_k(exo_emb_ts)              # [B, n_exo, d_model]

        # scaled dot-product: how much does endo attend to each exo?
        attn = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        # [B, 1, n_exo] → [B, n_exo]
        attn_scores = attn.squeeze(1)

        # learned gate per exo variable — captures directionality
        # exo_emb carries content info, gate learns which content causes endo
        gate_scores = self.gate(exo_emb_ts).squeeze(-1)  # [B, n_exo]

        # combine attention + gate → final causal scores
        causal_scores = attn_scores + gate_scores         # [B, n_exo]

        # softmax to get normalized weights
        causal_weights = F.softmax(causal_scores, dim=-1).unsqueeze(-1)
        # [B, n_exo, 1]

        return causal_weights
    
class EndoNodeConstruction(nn.Module):
    """
    Applies Vq cross-attention to all exo and endo patch embeddings.
 
    Uses SEPARATE cross-attention modules for exo and endo nodes
    since they have different roles in the graph.
 
    Processes all variables and all patches efficiently by looping
    over the patch and variable dimensions.
 
    Args:
        n_exo   : number of exogenous variables
        d_model : embedding dimension
        n_heads : attention heads
        dropout : dropout rate
 
    Input:
        exo_emb_ts  : [B, num_patches, n_exo, d_model]  from TSPreprocessor
        endo_emb_ts : [B, num_patches, 1,     d_model]  from TSPreprocessor
        vq_emb      : [n_vars, d_model]                 precomputed text embeddings
                      ordered as [endo_0, exo_0, exo_1, ...]
                      matching DATASET_INFO['all_vars'] order
 
    Output:
        exo_fused  : [B, num_patches, n_exo, d_model]
        endo_fused : [B, num_patches, 1,     d_model]
    """
 
    def __init__(
        self,
        n_exo   : int,
        d_model : int,
        n_heads : int   = 4,
        dropout : float = 0.1,
    ):
        super().__init__()
 
        # separate cross-attention for exo and endo
        self.endo_cross_attn  = VqCrossAttention(d_model, n_heads, dropout)
 
    def forward(self, endo_emb_ts, vq_tokens, vq_mask=None, stats_emb=None):
        # vq_tokens: [1, T, d_model]  static per-token text (already projected), endo has exactly one variable
        # vq_mask  : [1, T]           True = real token, False = pad (optional)
        # stats_emb: [B, P, d_model]
        B, P, _, d_model = endo_emb_ts.shape
        endo_flat = endo_emb_ts.reshape(B * P, d_model)

        if vq_tokens is None:
            # no-text ablation: skip cross-attention entirely if there's
            # also no stats to inject, otherwise fall back to stats-only
            if stats_emb is None:
                return endo_flat.unsqueeze(2).reshape(B, P, 1, d_model)
            stats_flat = stats_emb.reshape(B * P, d_model)
            fused = self.endo_cross_attn(
                endo_flat,
                torch.zeros(1, d_model, device=endo_flat.device, dtype=endo_flat.dtype),
                stats_emb=stats_flat,
            )
            return fused.unsqueeze(2).reshape(B, P, 1, d_model)

        vq_v   = vq_tokens[0]                     # [T, d_model]
        mask_v = vq_mask[0] if vq_mask is not None else None
        stats_flat = stats_emb.reshape(B * P, d_model) if stats_emb is not None else None

        fused = self.endo_cross_attn(endo_flat, vq_v, vq_mask=mask_v, stats_emb=stats_flat)
        return fused.unsqueeze(2).reshape(B, P, 1, d_model)

# summary function  
class EndoNodeInsertion(nn.Module):
    """
    Full endo node insertion - runs per patch.
 
    Both NGC and AttentionAggregator are called inside the per-patch loop.
    NGC uses TSPreprocessor embeddings, AttentionAggregator uses GAT-updated exo.
 
    Usage inside per-patch loop:
        for p in range(num_patches):
            endo_node_emb, causal_w = endo_insertion(
                exo_emb_ts  = exo_emb_ts[:, p, :, :],   # [B, n_exo, d_model]
                endo_emb_ts = endo_emb_ts[:, p, :, :],  # [B, 1,     d_model]
                exo_emb_gat = exo_emb_gat[:, p, :, :],  # [B, n_exo, d_model]
            )
 
    Args:
        n_exo      : number of exogenous variables
        d_model    : embedding dimension
        n_heads    : attention heads
        hidden_dim : LSTM hidden size for NGC
        dropout    : dropout rate
    """
 
    def __init__(
        self,
        n_exo      : int,
        d_model    : int,
        n_heads    : int   = 4,
        hidden_dim : int   = 32,
        dropout    : float = 0.1,
        ngc_use_current_and_lag : bool = False,
        ngc_window : int = 0,  # see NeuralGrangerCausalityWindowed docstring --
                    # 0 (default) keeps the original single-patch NGC exactly
                    # as before; >0 swaps in the K-patch-window NGC instead.
        ngc_window_contemporaneous : bool = False,  # EXPERIMENTAL (2026-09-15):
                    # see NeuralGrangerCausalityWindowed.forward's exo_current
                    # docstring -- lets the windowed unrestricted branch also
                    # see exo's CURRENT patch (not just its strictly-past
                    # window), while endo's own current patch stays hidden
                    # (zero-padded) to avoid the trivial-reconstruction issue
                    # the window fix was built to solve. Only meaningful when
                    # ngc_window > 0. False (default) = exact previous behavior.
        ngc_learn_temperature : bool = False,  # see CausalWeightFused docstring
                    # -- orthogonal to ngc_window/ngc_lag (scales whatever
                    # raw scores they produce, doesn't change how they're
                    # computed). False (default) = exact previous behavior.
        ngc_score_type : str = 'diff',  # EXPERIMENTAL (2026-09-15): see
                    # NeuralGrangerCausalityWindowed.__init__'s score_type docstring.
                    # Only meaningful when ngc_window > 0. 'diff' (default) = exact
                    # previous behavior.
        ngc_classical_prior : torch.Tensor = None,  # EXPERIMENTAL (2026-09-15): see
                    # NeuralGrangerCausalityWindowed.forward's return_aux docstring.
                    # Only meaningful when ngc_window > 0. None (default) = no aux
                    # loss available, exact previous behavior.
    ):
        super().__init__()
        self.ngc_window = ngc_window
        self.ngc_window_contemporaneous = ngc_window_contemporaneous
        self.ngc        = NeuralGrangerCausality(n_exo, d_model, hidden_dim,
                                                  use_current_and_lag=ngc_use_current_and_lag,
                                                  classical_prior=ngc_classical_prior)
        if ngc_window > 0:
            self.ngc_windowed = NeuralGrangerCausalityWindowed(
                n_exo, d_model, hidden_dim,
                score_type=ngc_score_type, classical_prior=ngc_classical_prior)
        self.causal = CausalWeightFused(n_exo, d_model, hidden_dim,
                                         learn_temperature=ngc_learn_temperature)
        #self.gated = GatedCausalAttention(n_exo, d_model, dropout)
        self.aggregator = AttentionAggregator(d_model, n_heads, dropout)
 
    def forward(
        self,
        exo_emb_ts  : torch.Tensor,   # [B, n_exo, d_model]  from Fusion
        endo_emb_ts : torch.Tensor,   # [B, 1,     d_model]  from Fusion in exo-graph
        exo_emb_gat : torch.Tensor,   # [B, n_exo, d_model]  from ExoGAT
        rq_emb : torch.Tensor,
        endo_text_bias : torch.Tensor = None,  # [d_model]        — passed to aggregator's Q
        exo_text_bias  : torch.Tensor = None,  # [n_exo, d_model] — passed to aggregator's K
        exo_emb_ts_lagged : torch.Tensor = None,  # [B, n_exo, d_model], same shape as
                          # exo_emb_ts but from an EARLIER patch — see ngc_lag docstring
                          # in MAGNETGraphOnly. None (default) = exact previous behavior
                          # (NGC compares endo vs. endo+exo at the SAME patch).
        exo_emb_ts_extra  : torch.Tensor = None,  # [B, n_exo, d_model], optional --
                          # passed straight through to NGC's exo_emb_ts_extra (see
                          # NeuralGrangerCausality.forward docstring). Only used when
                          # self.ngc.use_current_and_lag is True.
        return_aux_loss   : bool = False,  # EXPERIMENTAL: when True, returns a
                          # 3-tuple (endo_node_emb, causal_weights, aux_loss)
                          # instead of the usual 2-tuple. See
                          # NeuralGrangerCausality.forward's return_aux docstring.
        endo_emb_window : torch.Tensor = None,  # [B, K, d_model], optional -- K-patch
                          # PAST endo window (patches t-K..t-1, current EXCLUDED). When
                          # given together with exo_emb_window and self.ngc_window > 0,
                          # NGC uses NeuralGrangerCausalityWindowed instead of the
                          # single-patch NeuralGrangerCausality -- see that class's
                          # docstring for why (fixes the "restricted reconstructs endo
                          # from itself" + "LSTM sees only 1 timestep" issues).
        exo_emb_window  : torch.Tensor = None,  # [B, K, n_exo, d_model], optional --
                          # K-patch PAST exo window, paired with endo_emb_window.
        exo_gate : torch.Tensor = None,  # scalar, optional -- passed to the aggregator
                          # (scales the exo message). None = exact previous behavior.
        causal_weights_override : torch.Tensor = None,  # [B, n_exo, 1], optional --
                          # causal weights from an alternative estimator (GVAR / sparse-input
                          # Granger, see causal_alternatives.py). When given, NGC is skipped
                          # and these weights bias the aggregator instead. None (default) =
                          # exact previous behavior.
    ):
        """
        Args:
            exo_emb_ts  : exo embeddings straight from Fusion
                          -> used for NGC (captures raw temporal pattern)
            endo_emb_ts : endo embedding from TSPreprocessor
                          -> used for NGC and as endo Query in aggregator
            exo_emb_gat : exo embeddings AFTER ExoGAT update
                          -> used as Key/Value in aggregator (inter-exo aware)
            endo_text_bias / exo_text_bias : optional, see AttentionAggregator.
                          Default None — existing callers unaffected.
            exo_emb_ts_lagged : optional lagged exo embedding fed to NGC INSTEAD of
                          exo_emb_ts, so NGC tests "does exo's PAST embedding help
                          predict endo's CURRENT embedding" (real Granger precedence)
                          rather than "does exo's embedding at the SAME instant help
                          reconstruct endo's embedding at that instant" (contemporaneous
                          redundancy — the previous, default behavior). Only NGC's input
                          changes; ExoGraph and the aggregator's K/V still use the
                          contemporaneous exo_emb_gat, unaffected.
        """

        # Step 1: NGC on TSPreprocessor embeddings -> per-patch causal weights
        if causal_weights_override is not None:
            causal_weights = causal_weights_override
            aux_loss = causal_weights.new_zeros(())
        elif self.ngc_window > 0 and endo_emb_window is not None and exo_emb_window is not None:
            if return_aux_loss:
                ngc_scores, aux_loss = self.ngc_windowed(
                    endo_emb_window, exo_emb_window, endo_emb_ts,
                    exo_current=exo_emb_ts if self.ngc_window_contemporaneous else None,
                    return_aux=True)
            else:
                ngc_scores = self.ngc_windowed(
                    endo_emb_window, exo_emb_window, endo_emb_ts,
                    exo_current=exo_emb_ts if self.ngc_window_contemporaneous else None)
        else:
            ngc_exo_input = exo_emb_ts_lagged if exo_emb_ts_lagged is not None else exo_emb_ts
            if return_aux_loss:
                ngc_scores, aux_loss = self.ngc(
                    ngc_exo_input, endo_emb_ts, exo_emb_ts_extra=exo_emb_ts_extra, return_aux=True)
            else:
                ngc_scores = self.ngc(ngc_exo_input, endo_emb_ts, exo_emb_ts_extra=exo_emb_ts_extra)
        if causal_weights_override is None:
            causal_weights = self.causal(ngc_scores, rq_emb)

        # Ablation: random weights
        #B, exo_n, _ = exo_emb_ts.shape
        #causal_weights = torch.ones(B, exo_n, 1, device=exo_emb_ts.device) / exo_n

        # Ablation: gated causal attention
        #causal_weights = self.gated(exo_emb_ts, endo_emb_ts)

        #                [B, n_exo, 1]

        # Step 2: aggregator uses GAT-updated exo as Key/Value
        #         and TSPreprocessor endo as Query
        endo_node_emb = self.aggregator(
            exo_emb_gat, endo_emb_ts, causal_weights,
            endo_text_bias=endo_text_bias, exo_text_bias=exo_text_bias,
            exo_gate=exo_gate,
        )
        #               [B, 1, d_model]

        if return_aux_loss:
            return endo_node_emb, causal_weights, aux_loss
        return endo_node_emb, causal_weights

if __name__ == '__main__':
    B           = 16
    num_patches = 8
    n_exo       = 4
    n_vars      = 5
    d_model     = 64
    n_heads     = 4
 
    endo_insertion = EndoNodeInsertion(
        n_exo=n_exo, d_model=d_model, n_heads=n_heads, hidden_dim=32
    )
    node_construction = EndoNodeConstruction(n_exo, d_model, n_heads)
 
    # from TSPreprocessor [B, num_patches, n_vars, d_model]
    exo_fused  = torch.randn(B, num_patches, n_exo, d_model)
    endo_emb_ts_all = torch.randn(B, num_patches, 1,     d_model)
 
    # from ExoGAT [B, num_patches, n_exo, d_model]
    exo_emb_gat_all = torch.randn(B, num_patches, n_exo, d_model)
    vq_emb = torch.randn(n_vars, d_model)
    rq_emb = torch.randn(n_vars, n_vars, d_model)
 
    # ── per-patch loop ────────────────────────────────────────────────────────
    endo_node_patches     = []
    causal_weight_patches = []
 
    endo_fused = node_construction(endo_emb_ts_all, vq_emb)
    for p in range(num_patches):
        endo_node_emb, causal_w = endo_insertion(
            exo_emb_ts  = exo_fused[:, p, :, :],   # [B, n_exo, d_model]
            endo_emb_ts = endo_fused[:, p, :, :],  # [B, 1,     d_model]
            exo_emb_gat = exo_emb_gat_all[:, p, :, :],  # [B, n_exo, d_model]
            rq_emb = rq_emb 
        )
        endo_node_patches.append(endo_node_emb)
        causal_weight_patches.append(causal_w)
 
    # stack -> endo time series for Transformer
    endo_series   = torch.stack(endo_node_patches,     dim=1).squeeze(2)
    causal_series = torch.stack(causal_weight_patches, dim=1)

    print("\n── CausalWeightsWithRq ──────────────────────────────")
    causal = CausalWeightFused(n_exo=n_exo, d_model=d_model)
    ngc = NeuralGrangerCausality(n_exo, d_model)
    ngc_scores = ngc(exo_fused[:,0,:,:], endo_emb_ts_all[:,0,:,:])
    cw     = causal(ngc_scores, rq_emb)
    print(f"  input  ngc_scores : {ngc_scores.shape}")  # [16, 2]
    print(f"  output causal_w   : {cw.shape}")           # [16, 2, 1]
    print(f"  sample 0:")
    print(f"    occ -> flow : {cw[0,0].item():.4f}")
    print(f"    spd -> flow : {cw[0,1].item():.4f}")
    print(f"    sum (=1.0)  : {cw[0].sum().item():.4f}")
 
    print("── Per-patch NGC causal weights ──────────────────────")
    print(f"  causal_series : {causal_series.shape}")   # [16, 8, 2, 1]
    print(f"  (each patch has its own causal weights)")
    print()
    print("  sample 0, all patches:")
    print(f"  {'patch':<8} {'occupancy->flow':<20} {'speed->flow':<20}")
    for p in range(num_patches):
        cw = causal_series[0, p]
        print(f"  {p:<8} {cw[0].item():<20.4f} {cw[1].item():<20.4f}")
 
    print(f"\n── Output shapes ─────────────────────────────────────")
    print(f"  exo_emb_ts_all  : {exo_fused.shape}")   # [16, 8, 2, 64]
    print(f"  endo_emb_ts_all : {endo_fused.shape}")  # [16, 8, 1, 64]
    print(f"  exo_emb_gat_all : {exo_emb_gat_all.shape}")  # [16, 8, 2, 64]
    print(f"  endo_series     : {endo_series.shape}")       # [16, 8, 64]
 
    print(f"\n── Parameter count ───────────────────────────────────")
    total = sum(p.numel() for p in endo_insertion.parameters())
    print(f"  EndoNodeInsertion: {total:,} parameters")
    for name, module in endo_insertion.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {params:,} parameters")