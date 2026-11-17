import final_defaults  # noqa: F401  official configuration (SIG v2), see final_defaults.py
import os
# causal_alternatives.py
# Two drop-in alternatives to NeuralGrangerCausality (endo_insertion.py) for MAGNet's
# exo->endo causal weights. Selected with MAGNETGraphOnly(use_gvar_causal=True) or
# MAGNETGraphOnly(use_sparse_granger_causal=True); NGC stays the default and is untouched.
#
# Unlike NGC, both modules
#   * work on the RAW (standardized) patch values, not on learned embeddings,
#   * predict endo patch p from the PAST patches p-1..p-K of every variable (endo + all exo
#     jointly, i.e. conditional rather than pairwise Granger causality),
#   * are trained on their own forecasting loss (returned as an auxiliary loss), and
#   * give per-exo weights in [0, 1) that are NOT normalized across exo:
#         w_j = s_j / (s_j + s_endo)
#     where s_j is exo j's Granger strength and s_endo the endo variable's own
#     autoregressive strength. All weights go to 0 when no exo variable helps.
#
# A: GVARCausality        (Marcinkevicz & Vogt, "Interpretable models for Granger causality
#                          using self-explaining neural networks", ICLR 2021)
#    Per-window, per-lag coefficients; s_j(p) = sum_k |psi_{k,j}(p)|.
# B: SparseInputGranger   (Tank et al., "Neural Granger causality", TPAMI 2021, cMLP)
#    Group lasso on the first-layer weights; s_j = sum_k ||W1[:, j, k, :]||_F (static per dataset).

import torch
import torch.nn as nn
import torch.nn.functional as F


def lagged_patches(x, n_lags):
    """x: [B, P, pl, V] -> [B, P, K, pl, V], entry k-1 holds patch p-k.
    Patches before the window start reuse patch 0 (same boundary rule as ngc_lag)."""
    B, P, pl, V = x.shape
    pad = x[:, :1].expand(B, n_lags, pl, V)
    xp = torch.cat([pad, x], dim=1)                                  # [B, P+K, pl, V]
    return torch.stack([xp[:, n_lags - k: n_lags - k + P] for k in range(1, n_lags + 1)], dim=2)


def strength_to_weights(s_exo, s_endo, eps=1e-6):
    """Exo strength relative to endo's own autoregressive strength, in [0, 1)."""
    return s_exo / (s_exo + s_endo.unsqueeze(-1) + eps)


def target_mask(P, n_lags, device):
    """Patches with a full real history; falls back to p >= 1 when the window is short."""
    first = n_lags if P > n_lags else 1
    m = torch.zeros(P, dtype=torch.bool, device=device)
    m[first:] = True
    if P == 1:                        # single patch (no_patching ablation): only the lag-0 term can be fitted
        m[0] = True
    return m


class GVARCausality(nn.Module):
    """
    Self-explaining Granger model (GVAR):
        endo(p) = sum_k sum_j psi_{k,j}(x(p-k)) * x_j(p-k) + b
    psi_k = g_k(x(p-k)) is produced per window by a small MLP per lag, so the causal
    coefficients vary over time (per patch), like NGC's per-patch weights.

    Input : x [B, P, pl, V] raw patches, variable 0 = endo, 1.. = exo
    Output: weights [B, P, n_exo], aux_loss (scalar), coeffs [B, P, K, V]
    """

    def __init__(self, n_vars, patch_len, n_lags=3, hidden_dim=32,
                 sparsity=0.01, smoothness=0.01):
        super().__init__()
        self.n_vars, self.n_lags = n_vars, n_lags
        self.sparsity, self.smoothness = sparsity, smoothness
        self.generators = nn.ModuleList([
            nn.Sequential(nn.Linear(n_vars * patch_len, hidden_dim), nn.ReLU(),
                          nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                          nn.Linear(hidden_dim, n_vars))
            for _ in range(n_lags)])
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, P, pl, V = x.shape
        K = min(self.n_lags, max(P - 1, 1))
        lags = lagged_patches(x, K)                                   # [B, P, K, pl, V]
        coeffs = torch.stack([self.generators[k](lags[:, :, k].reshape(B, P, pl * V))
                              for k in range(K)], dim=2)               # [B, P, K, V]
        pred = (coeffs.unsqueeze(3) * lags).sum(dim=(2, 4)) + self.bias  # [B, P, pl]

        mask = target_mask(P, K, x.device)
        mse = F.mse_loss(pred[:, mask], x[:, mask, :, 0])
        aux = mse + self.sparsity * coeffs.abs().mean()
        if P > 1:
            aux = aux + self.smoothness * (coeffs[:, 1:] - coeffs[:, :-1]).abs().mean()

        strength = coeffs.abs().sum(dim=2)                            # [B, P, V]
        weights = strength_to_weights(strength[..., 1:], strength[..., 0])
        return weights, aux, coeffs


class SparseInputGranger(nn.Module):
    """
    Component-wise MLP (cMLP) for the endo target with a group-lasso penalty on each
    (variable, lag) block of the first layer. A block driven to ~0 means that variable at
    that lag does not Granger-cause endo. The strength is a static, per-dataset quantity.

    proximal=True replaces the penalty gradient by a proximal (group soft-thresholding) step
    after every optimizer step, as in Tank et al.'s GISTA training: call prox_step(lr) after
    optimizer.step(). Unlike the penalty, this drives non-causal groups to exactly zero.

    contemporaneous=True adds a lag-0 group per EXO variable: endo patch p is also predicted from
    exo patch p (same patch, still inside the observed history window -- no future values). PCMCI+
    on the training data finds most exo->endo dependence at lag 0 (e.g. wind speed -> power), which a
    lags>=1-only model cannot see. Endo's own patch p is never an input (it is the target). The lag-0
    weights live in a separate parameter W0 (absent when False, so older checkpoints load unchanged)
    and are penalized / proximally thresholded exactly like the lagged groups.

    Input : x [B, P, pl, V] raw patches, variable 0 = endo, 1.. = exo
    Output: weights [B, P, n_exo], aux_loss (scalar), group_norms [K, V]
    """

    def __init__(self, n_vars, patch_len, n_lags=3, hidden_dim=32, group_lasso=0.01, proximal=False,
                 contemporaneous=False):
        super().__init__()
        self.n_vars, self.n_lags, self.group_lasso = n_vars, n_lags, group_lasso
        self.proximal, self.contemporaneous = proximal, contemporaneous
        # lag-0 as a NUISANCE control (2026-11-13, MAGNET_LAG0_NUISANCE=1): the lag-0 groups are still fitted (and
        # penalised) so they absorb same-patch co-movement, but only the LAGGED groups set the attention weights.
        self.lag0_in_weights = os.environ.get("MAGNET_LAG0_NUISANCE") != "1"
        # idea A (2026-11-14): heavier group-lasso penalty on the lag-0 groups (MAGNET_LAG0_PENALTY = multiplier c)
        self.lag0_penalty = float(os.environ.get("MAGNET_LAG0_PENALTY", "1"))
        # idea B (2026-11-14): two-stage / residual fitting (MAGNET_LAG0_RESIDUAL=1): the lagged model predicts the
        # target patch; the lag-0 groups (own small head) may only explain what the lagged prediction leaves over
        self.lag0_residual = os.environ.get("MAGNET_LAG0_RESIDUAL") == "1" and contemporaneous
        if self.lag0_residual:
            self.b0 = nn.Parameter(torch.zeros(hidden_dim))
            self.out0 = nn.Linear(hidden_dim, patch_len)
        fan_in = n_vars * n_lags + (n_vars - 1 if contemporaneous else 0)
        self.W1 = nn.Parameter(torch.empty(hidden_dim, n_vars, n_lags, patch_len))
        nn.init.uniform_(self.W1, -1 / (fan_in * patch_len) ** 0.5, 1 / (fan_in * patch_len) ** 0.5)
        if contemporaneous:
            self.W0 = nn.Parameter(torch.empty(hidden_dim, n_vars - 1, patch_len))   # exo only
            nn.init.uniform_(self.W0, -1 / (fan_in * patch_len) ** 0.5, 1 / (fan_in * patch_len) ** 0.5)
        else:
            self.W0 = None
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.out = nn.Linear(hidden_dim, patch_len)

    def group_norms(self, n_lags=None):
        W = self.W1[:, :, :self.n_lags if n_lags is None else n_lags]
        return W.pow(2).sum(dim=(0, 3)).add(1e-12).sqrt().t()       # [K, V]

    def contemporaneous_norms(self):
        """[n_exo] norms of the lag-0 exo groups (None without contemporaneous inputs)."""
        return None if self.W0 is None else self.W0.pow(2).sum(dim=(0, 2)).add(1e-12).sqrt()

    def forward(self, x):
        B, P, pl, V = x.shape
        # K <= P - 1: with a single patch (the no_patching ablation) there is no earlier patch, so only the lag-0 exo
        # groups remain (padding a lag from patch 0 would feed the target patch to its own predictor). Identical to
        # min(n_lags, max(P - 1, 1)) for every P >= 2.
        K = min(self.n_lags, P - 1)
        if K > 0:
            lags = lagged_patches(x, K)                               # [B, P, K, pl, V]
            h = torch.einsum("bpkiv,hvki->bph", lags, self.W1[:, :, :K]) + self.b1
        else:
            h = self.b1.expand(B, P, -1)
        mask = target_mask(P, K, x.device)
        if self.W0 is not None and getattr(self, "lag0_residual", False):
            pred_lag = self.out(F.relu(h))                                            # lagged-only prediction
            h0 = torch.einsum("bpiv,hvi->bph", x[..., 1:], self.W0) + self.b0
            pred0 = self.out0(F.relu(h0))                                             # explains the residual only
            aux = F.mse_loss(pred_lag[:, mask], x[:, mask, :, 0]) + \
                F.mse_loss(pred0[:, mask], (x[..., 0] - pred_lag).detach()[:, mask])
        else:
            if self.W0 is not None:
                h = h + torch.einsum("bpiv,hvi->bph", x[..., 1:], self.W0)
            pred = self.out(F.relu(h))                                # [B, P, pl]
            aux = F.mse_loss(pred[:, mask], x[:, mask, :, 0])
        norms = self.group_norms(K)                                   # [K, V]
        c_norms = self.contemporaneous_norms()                        # [n_exo] or None
        if not self.proximal:
            aux = aux + self.group_lasso * (norms.sum() + (getattr(self, "lag0_penalty", 1.0) * c_norms.sum()
                                                           if c_norms is not None else 0))

        strength = norms.sum(dim=0)                                   # [V]
        if c_norms is not None and getattr(self, "lag0_in_weights", True):
            strength = torch.cat([strength[:1], strength[1:] + c_norms])
        weights = strength_to_weights(strength[1:], strength[0]).expand(B, P, V - 1)
        return weights, aux, norms

    @torch.no_grad()
    def prox_step(self, lr):
        """Group soft-thresholding of every (variable, lag) block of W1 with threshold lr * group_lasso."""
        norms = self.W1.pow(2).sum(dim=(0, 3), keepdim=True).sqrt()  # [1, V, K, 1]
        self.W1.mul_(torch.clamp(1 - lr * self.group_lasso / norms.clamp_min(1e-12), min=0.0))
        if self.W0 is not None:
            n0 = self.W0.pow(2).sum(dim=(0, 2), keepdim=True).sqrt()  # [1, n_exo, 1]
            self.W0.mul_(torch.clamp(1 - lr * self.group_lasso / n0.clamp_min(1e-12), min=0.0))


def exo_strength_ratio(estimator):
    """sum_j s_j / (sum_j s_j + s_endo) of a SparseInputGranger: how much of the target's Granger-predictive
    strength comes from exo rather than its own past. Used as the causal gate (rebuttal option 1)."""
    s = estimator.group_norms().sum(dim=0)                         # [V], lags 1..K
    c = estimator.contemporaneous_norms()
    s_exo = s[1:].sum() + (c.sum() if c is not None else 0)
    return s_exo / (s_exo + s[0] + 1e-6)


class GainHead(nn.Module):
    """
    Amortised occlusion gain (rebuttal option 2): predicts, from the raw window, how much the forecast loss rises when
    each exo variable is shuffled across the batch, c_j(window) = loss(shuffle j) - loss(full), in units of a running
    std of the observed gains. Its output modulates the static sparse-Granger weights per window:
        w_j(window) = w_j^static * 2 * sigmoid(c_hat_j)        (factor 1 when the predicted gain is 0)

    Input : x [B, P, pl, V] raw patches (variable 0 = endo)
    Output: c_hat [B, n_exo]
    """

    def __init__(self, n_vars, seq_len, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_vars * seq_len, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_vars - 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)                            # starts at factor 1 = plain static weights
        self.register_buffer("scale", torch.tensor(0.0))             # running std of observed gains (0 = unset)

    def forward(self, x):
        return self.net(x.flatten(1))

    @torch.no_grad()
    def update_scale(self, gains, momentum=0.99):
        if gains.numel() < 2:                                        # a 1-window last batch has no std (NaN)
            return
        std = gains.std().clamp_min(1e-6)
        self.scale.copy_(std if self.scale == 0 else momentum * self.scale + (1 - momentum) * std)
