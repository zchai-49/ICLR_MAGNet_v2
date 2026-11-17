import final_defaults  # noqa: F401  official configuration (SIG v2), see final_defaults.py
import os
import json
# MAGNET_graph_only.py
# Lightweight MAGNET — strips everything down to just the causal graph + text.
# No TSPreprocessor, no MultiLevelPatch, no normalization, no stats conditioning.
# Output: a causal graph bias [B, P, d_model] to inject into the downstream backbone's representation.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from exo_graph import ExoNodeConstruction, ExoGraph
from endo_insertion import EndoNodeConstruction, EndoNodeInsertion
from text_projection import TextProjection
from causal_alternatives import GVARCausality, SparseInputGranger, GainHead, exo_strength_ratio

import os as _os
_TEXT_EMB_ROOT = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data", "text_emb")


# TextTokenPooler (learned attention-pooling over full per-token text,
# replacing plain [CLS]-pooling) — built 2026-07-08, currently unused. Not
# considering the text-ablation upgrade path for now, so MAGNETGraphOnly
# below is reverted to the pre-2026-07-08 (1 July) [CLS]-pooled text_bias,
# which is what the reported hyperparameter searches were run
# against. Re-enable by restoring endo_text_pool/exo_text_pool below and
# swapping _load_text_embeddings back to the *_vq_tokens/*_vq_mask files.
class TextTokenPooler(nn.Module):
    """
    Learned attention-pooling over a variable's FULL per-token text
    embedding, replacing naive [CLS]-token pooling.

    Previously each variable's text collapsed to BERT's [CLS] vector before
    ever reaching this module — [CLS] from a non-fine-tuned bert-base-uncased
    is a known-weak sentence representative (it wasn't trained to summarize
    meaning via that token). Here a small learnable query attends over every
    real (non-padded) token and produces one summary vector per variable, so
    gradient gets to decide which sub-facts of the text matter, instead of
    always taking whatever bert-base-uncased happened to put in position 0.

    Output is still exactly one vector per variable (this module feeds
    endo_text_bias / exo_text_bias, which are added directly into Q/K — a
    single bias vector per position, not a key/value sequence — so unlike
    VqCrossAttention's fix, the "multi-token" gain here is entirely in HOW
    the summary vector is computed, not in what shape it ends up as).
    """
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.d_model = d_model

        self.query  = nn.Parameter(torch.randn(d_model) * 0.02)
        self.W_k    = nn.Linear(d_model, d_model)
        self.W_v    = nn.Linear(d_model, d_model)
        self.W_out  = nn.Linear(d_model, d_model)
        self.norm   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # tokens : [n_vars, T, d_model]   already projected to d_model
        # mask   : [n_vars, T]            True = real token, False = pad
        n_vars, T, d_model = tokens.shape

        Q = self.query.unsqueeze(0).expand(n_vars, -1).unsqueeze(1)  # [n_vars, 1, d_model]
        K = self.W_k(tokens)                                         # [n_vars, T, d_model]
        V = self.W_v(tokens)

        def split_heads(x, seq_len):
            return x.reshape(n_vars, seq_len, self.n_heads, self.d_head) \
                    .permute(0, 2, 1, 3)                              # [n_vars, n_heads, seq_len, d_head]

        Qh = split_heads(Q, 1)
        Kh = split_heads(K, T)
        Vh = split_heads(V, T)

        scale = self.d_head ** 0.5
        attn  = torch.matmul(Qh, Kh.transpose(-1, -2)) / scale        # [n_vars, n_heads, 1, T]

        if mask is not None:
            mask_bias = mask.unsqueeze(1).unsqueeze(1)                # [n_vars, 1, 1, T]
            attn = attn.masked_fill(~mask_bias, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        out  = torch.matmul(attn, Vh)                                 # [n_vars, n_heads, 1, d_head]
        out  = out.permute(0, 2, 1, 3).reshape(n_vars, self.d_model)

        out = self.W_out(self.dropout(out))
        out = self.norm(out)
        return out                                                    # [n_vars, d_model]


class MAGNETGraphOnly(nn.Module):
    """
    Minimal MAGNET encoder — causal graph + text only.

    Input:
        exo_patches  : [B, P, patch_len, n_exo]
        endo_patches : [B, P, patch_len, 1]

    Output:
        graph_bias   : [B, P, d_model]  — causal context to add to the backbone's representation
    """

    def __init__(
        self,
        n_exo        : int,
        patch_len    : int,
        d_model      : int = 64,
        n_heads      : int = 4,
        ngc_hidden_dim: int = 32,
        dropout      : float = 0.1,
        dataset      : int = None,
        term         : str = "short",
        bert_dim     : int = 768,
        use_text     : bool = True,   # ablation switch: text (multi-modal) conditioning
        text_variant : str = "new",   # which text_emb/*_<variant>_*.pt files to load —
                                       # "new" = current default prompt (variable+relation,
                                       # with numbers); see text_generate_ablation.py for
                                       # the other content/numbers ablation variants.
        node_text_fusion: str = "qk_bias",  # HOW text enters, independent of whether it
                                       # enters at all (use_text). Three variants (see
                                       # run_full_text_variant_ablation.py):
                                       #   "qk_bias" (default, = every existing PnP result
                                       #     for every backbone) — node construction gets a
                                       #     ZERO text token (node_emb ~= patch_emb through a
                                       #     residual VqCrossAttention no-op); text instead
                                       #     enters ONLY as a learned Q/K bias inside the
                                       #     exo->endo NGC causal attention.
                                       #   "add" — node_emb = patch_emb + text_bias (same
                                       #     pooled/gated/normed text_bias qk_bias uses),
                                       #     simple elementwise addition, no attention. NGC
                                       #     Q/K bias is disabled (clean swap, not stacked).
                                       #   "attn" — node_emb = CrossAttention(patch_emb as Q,
                                       #     real per-token text as K/V), via the existing
                                       #     ExoNodeConstruction/EndoNodeConstruction +
                                       #     VqCrossAttention machinery (previously always
                                       #     fed a zero dummy token). NGC Q/K bias disabled.
        ngc_lag      : int = 1,       # EXPERIMENTAL (2026-09-13): patches are otherwise
                                       # fully independent here (B*P flatten below) -- NGC's
                                       # restricted/unrestricted LSTM comparison is therefore
                                       # contemporaneous (same patch index) by default, which
                                       # is a same-instant redundancy test, not real Granger
                                       # causality (no temporal precedence). ngc_lag>0 feeds
                                       # NGC exo_i's embedding from `ngc_lag` patches EARLIER
                                       # instead, while endo's target patch and everything
                                       # else (ExoGraph GAT, aggregator K/V) stay contemporaneous
                                       # and unchanged. 0 (default) = exact previous behavior.
        ngc_use_current_and_lag : bool = False,  # EXPERIMENTAL: requires ngc_lag>0.
                                       # NGC's unrestricted branch sees BOTH exo(p) and
                                       # exo(p-ngc_lag) concatenated, instead of only the
                                       # lagged one -- lets it detect trend/change in exo,
                                       # not just a fixed-lag level. See NeuralGrangerCausality
                                       # docstring for the capacity-matching caveat this
                                       # introduces (unrestricted's input is now bigger than
                                       # restricted's by more than "has exo").
        ngc_window   : int = 0,       # EXPERIMENTAL (2026-09-14): replaces ngc_lag's
                                       # single-patch shift with a REAL K-patch past window
                                       # fed to NeuralGrangerCausalityWindowed (see its
                                       # docstring) -- both the "restricted reconstructs
                                       # endo from itself" triviality AND the "LSTM sees
                                       # only 1 timestep" issue that ngc_lag alone doesn't
                                       # fix. Mutually exclusive with ngc_lag (windowing
                                       # subsumes a fixed single-step lag). 0 (default) =
                                       # exact previous behavior.
        ngc_learn_temperature : bool = False,  # EXPERIMENTAL (2026-09-14): see
                                       # CausalWeightFused docstring in endo_insertion.py.
                                       # Orthogonal to ngc_lag/ngc_window -- adds one
                                       # learnable scalar that multiplies whichever raw
                                       # NGC scores those produce, before the final
                                       # softmax, to address entropy staying near-uniform
                                       # even after the window fix (a scale problem, not
                                       # necessarily an information problem). Init 1.0 =
                                       # exact previous behavior at step 0. False (default)
                                       # = no new parameter at all.
        ngc_window_contemporaneous : bool = False,  # EXPERIMENTAL (2026-09-15): see
                                       # NeuralGrangerCausalityWindowed.forward's
                                       # exo_current docstring in endo_insertion.py.
                                       # Ground-truth Granger causality checks on raw
                                       # data found some drivers' true causal effect is
                                       # near-instantaneous (same patch), which the
                                       # strictly-past ngc_window cannot see by
                                       # construction. When True (and ngc_window > 0),
                                       # the windowed unrestricted branch also gets
                                       # exo's CURRENT patch. Only meaningful with
                                       # ngc_window > 0. False (default) = exact
                                       # previous behavior.
        ngc_score_type : str = 'diff',  # EXPERIMENTAL (2026-09-15): see
                                       # NeuralGrangerCausalityWindowed.__init__'s
                                       # score_type docstring in endo_insertion.py --
                                       # 'diff' (raw MSE gap, default) or 'log_ratio'
                                       # (scale-invariant log-ratio, closer to a
                                       # classical Granger F-test). Only meaningful
                                       # when ngc_window > 0.
        ngc_classical_prior : torch.Tensor = None,  # EXPERIMENTAL (2026-09-15): see
                                       # NeuralGrangerCausalityWindowed.forward's
                                       # return_aux docstring in endo_insertion.py --
                                       # a fixed [n_exo] classical Granger-causality
                                       # prior used to distill a ranking target into
                                       # causal_scores. Only meaningful when
                                       # ngc_window > 0 and forward(return_aux_loss=True).
        use_gvar_causal : bool = False,  # EXPERIMENTAL (2026-10-26): replace NGC's exo->endo
                                       # weights with GVAR (option A, causal_alternatives.py).
        use_sparse_granger_causal : bool = False,  # EXPERIMENTAL (2026-10-26): replace NGC's
                                       # weights with the sparse-input cMLP (option B).
                                       # Both default False = NGC, exact previous behavior.
                                       # Either one adds an auxiliary loss, exposed as
                                       # self.causal_aux_loss after each forward().
        causal_n_lags : int = 3,       # lags (in patches) seen by options A/B
        sparse_granger_proximal : bool = False,  # option B only: proximal group-lasso step
                                       # (exact zeros) instead of the penalty gradient; the
                                       # training loop must call causal_estimator.prox_step(lr)
        sparse_granger_contemporaneous : bool = False,  # option B only: also feed each exo's
                                       # SAME patch (lag 0, still observed history) -- see
                                       # SparseInputGranger docstring
    ):
        super().__init__()
        self.n_exo    = n_exo
        self.d_model  = d_model
        self.term     = term
        self.dataset  = dataset
        self.use_text = use_text
        self.text_variant = text_variant
        assert node_text_fusion in ("qk_bias", "add", "attn")
        self.node_text_fusion = node_text_fusion
        if ngc_lag != 1 or ngc_window != 0:
            raise ValueError("Official MAGNet requires ngc_lag=1 (and no NGC window); "
                             f"got ngc_lag={ngc_lag}, ngc_window={ngc_window}")
        assert not (ngc_use_current_and_lag and ngc_lag == 0), \
            "ngc_use_current_and_lag requires ngc_lag > 0 (nothing to combine with lag=0)"
        assert ngc_window >= 0
        assert not (ngc_window > 0 and ngc_lag > 0), \
            "ngc_window and ngc_lag are mutually exclusive (windowing subsumes a fixed lag)"
        self.ngc_lag = ngc_lag
        self.ngc_use_current_and_lag = ngc_use_current_and_lag
        self.ngc_window = ngc_window
        self.ngc_window_contemporaneous = ngc_window_contemporaneous
        self.ngc_learn_temperature = ngc_learn_temperature

        # Simple linear patch projections — no complex TSPreprocessor
        self.exo_proj  = nn.Linear(patch_len, d_model)
        self.endo_proj = nn.Linear(patch_len, d_model)

        if self.use_text:
            # Text embeddings — core MAGNET contribution
            self.endo_vq_projector = TextProjection(bert_dim, d_model)
            self.exo_vq_projector  = TextProjection(bert_dim, d_model)

            # Text is a single STATIC vector per dataset (no batch/patch
            # dependence at all — this module has no stats conditioning to
            # give it per-sample variation, unlike the full MAGNET encoder).
            # TextProjection's raw output has whatever scale a freshly
            # initialized 768->d_model MLP happens to produce, which has no
            # reason to match the patch embeddings' scale. Cross-attention
            # here has a key/value sequence length of 1, so softmax collapses
            # to a no-op and the "attention" output is just W_v(text) added
            # to the residual — a scale mismatch there isn't something later
            # attention layers can average away, it directly perturbs the
            # residual stream. LayerNorm fixes the scale regardless of the
            # projector's init, independently for endo vs. exo text.
            self.endo_text_norm = nn.LayerNorm(d_model)
            self.exo_text_norm  = nn.LayerNorm(d_model)

            # Learnable gate on the (now well-scaled) text contribution.
            # Initialized at -2.0 -> sigmoid ~= 0.12, i.e. mostly closed, so
            # training starts close to the text-free graph and only opens
            # the channel where gradients show it helps — same "start near
            # zero" pattern used for the other injection scales, avoiding a
            # 0.5-open default silently perturbing every run from step 1. Adds a
            # single scalar parameter (plus two parameter-free-hyperparam
            # LayerNorms), so it needs no hparam re-search: d_model,
            # e_layers, lr decisions made by the existing hparam search are
            # unaffected — this only needs retraining with those same values.
            self.text_gate = nn.Parameter(torch.tensor(-2.0))
        else:
            self.endo_vq_projector = None
            self.exo_vq_projector  = None
            self.endo_text_norm    = None
            self.exo_text_norm     = None
            self.text_gate         = None

        # Causal graph modules — the unique MAGNET contribution
        self.exo_node = ExoNodeConstruction(
            n_exo=n_exo, d_model=d_model,
            n_heads=n_heads, dropout=dropout,
        )
        self.exo_graph = ExoGraph(
            n_exo=n_exo, d_model=d_model,
            n_heads=n_heads, dropout=dropout,
        )
        self.endo_node = EndoNodeConstruction(
            n_exo=n_exo, d_model=d_model,
            n_heads=n_heads, dropout=dropout,
        )
        self.endo_insertion = EndoNodeInsertion(
            n_exo=n_exo, d_model=d_model,
            n_heads=n_heads, hidden_dim=ngc_hidden_dim,
            dropout=dropout, ngc_use_current_and_lag=ngc_use_current_and_lag,
            ngc_window=ngc_window, ngc_window_contemporaneous=ngc_window_contemporaneous,
            ngc_learn_temperature=ngc_learn_temperature,
            ngc_score_type=ngc_score_type, ngc_classical_prior=ngc_classical_prior,
        )

        # Plug-and-play detach_gg (2026-11-10): MAGNET_PNP_DETACH_GG=1 gives every backbone wrapper the sparse-input
        # Granger graph with detached weights and the fixed cross-fit gate, without touching the backbone trainers.
        _pnp_dg = os.environ.get("MAGNET_PNP_DETACH_GG") == "1"
        if _pnp_dg:
            use_sparse_granger_causal, sparse_granger_contemporaneous = True, True
        assert not (use_gvar_causal and use_sparse_granger_causal), \
            "use_gvar_causal and use_sparse_granger_causal are mutually exclusive"
        if use_gvar_causal:
            self.causal_estimator = GVARCausality(1 + n_exo, patch_len, n_lags=causal_n_lags,
                                                  hidden_dim=ngc_hidden_dim)
        elif use_sparse_granger_causal:
            self.causal_estimator = SparseInputGranger(1 + n_exo, patch_len, n_lags=causal_n_lags, group_lasso=0.01,
                                                       hidden_dim=ngc_hidden_dim,
                                                       proximal=sparse_granger_proximal,
                                                       contemporaneous=sparse_granger_contemporaneous)
        else:
            self.causal_estimator = None
        self.causal_aux_loss = None
        if _pnp_dg:
            ids = {5: "BE", 6: "DE", 7: "FR", 8: "NP", 9: "PJM", 10: "Energy", 11: "Colbun", 12: "Rapel",
                   13: "Sdwpfh1", 14: "Sdwpfh2", 15: "Sdwpfm1", 16: "Sdwpfm2",
                   18: "Syn5", 19: "Syn20", 20: "Syn50", 21: "Syn20weak", 22: "Syn20null"}
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            gg = json.load(open(os.path.join(root, "configs",
                                             os.environ.get("MAGNET_GG_FILE", "granger_gain_crossfit.json"))))
            self.detach_causal = True
            self.enable_causal_options(gate="fixed",
                                       gate_init=min(max(gg[f"{ids[dataset]}/{term}"]["ridge"]["gain"], 0.0), 1.0))
            self._pnp_self_train, self._est_opt = True, None

        # Load text embeddings — pooled [CLS] vectors for "qk_bias"/"add" (the
        # gate/norm machinery below operates on one vector per variable), or
        # real per-token sequences for "attn" (needs a genuine K/V sequence
        # for cross-attention, a single pooled vector would make softmax
        # degenerate to 1.0 over one key).
        if self.use_text and dataset is not None:
            if self.node_text_fusion == "attn":
                self._load_text_tokens(dataset, term)
            else:
                self._load_text_embeddings(dataset, term)

    def _load_text_tokens(self, dataset, term):
        # Same dataset->file-prefix mapping as _load_text_embeddings, just
        # pointed at the *_vq_tokens.pt/*_vq_mask.pt files (full per-token
        # BERT sequence, pre-dating TextTokenPooler) instead of the
        # *_vq_emb.pt pooled [CLS] vector.
        paths = {
            5:  ("BE",     "be"),
            6:  ("DE",     "de"),
            7:  ("FR",     "fr"),
            8:  ("NP",     "np"),
            9:  ("PJM",    "pjm"),
            10: ("energy", "Energy"),
            11: ("Colbun", "Colbun"),
            12: ("Rapel",  "Rapel"),
            13: ("wind/Sdwpfh1", "Sdwpfh1"),
            14: ("wind/Sdwpfh2", "Sdwpfh2"),
            15: ("wind/Sdwpfm1", "Sdwpfm1"),
            16: ("wind/Sdwpfm2", "Sdwpfm2"),
        }
        if dataset not in paths:
            raise ValueError(f"Unknown dataset id: {dataset}")

        folder, base_name = paths[dataset]
        prefix = f"{base_name}_{self.text_variant}"
        base = f"{_TEXT_EMB_ROOT}/{folder}"

        endo_tokens = torch.load(f"{base}/{prefix}_{term}_endo_vq_tokens.pt")  # [1,     T, bert_dim]
        endo_mask   = torch.load(f"{base}/{prefix}_{term}_endo_vq_mask.pt")   # [1,     T]
        exo_tokens  = torch.load(f"{base}/{prefix}_{term}_exo_vq_tokens.pt")  # [n_exo, T, bert_dim]
        exo_mask    = torch.load(f"{base}/{prefix}_{term}_exo_vq_mask.pt")    # [n_exo, T]

        self.register_buffer('bert_endo_vq_tokens', endo_tokens)
        self.register_buffer('bert_endo_vq_mask',   endo_mask)
        self.register_buffer('bert_exo_vq_tokens',  exo_tokens)
        self.register_buffer('bert_exo_vq_mask',    exo_mask)

    def _load_text_embeddings(self, dataset, term):
        paths = {
            5:  ("BE",     "be"),
            6:  ("DE",     "de"),
            7:  ("FR",     "fr"),
            8:  ("NP",     "np"),
            9:  ("PJM",    "pjm"),
            10: ("energy", "Energy"),
            11: ("Colbun", "Colbun"),
            12: ("Rapel",  "Rapel"),
            13: ("wind/Sdwpfh1", "Sdwpfh1"),
            14: ("wind/Sdwpfh2", "Sdwpfh2"),
            15: ("wind/Sdwpfm1", "Sdwpfm1"),
            16: ("wind/Sdwpfm2", "Sdwpfm2"),
            17: ("weather", "Weather"),          # rebuttal (models/text_generate_generic.py)
            18: ("synthetic/Syn5", "Syn5"),
            19: ("synthetic/Syn20", "Syn20"),
            20: ("synthetic/Syn50", "Syn50"),
            21: ("synthetic/Syn20weak", "Syn20weak"),
            22: ("synthetic/Syn20null", "Syn20null"),
        }
        if dataset not in paths:
            raise ValueError(f"Unknown dataset id: {dataset}")

        folder, base_name = paths[dataset]
        prefix = f"{base_name}_{self.text_variant}"
        base = f"{_TEXT_EMB_ROOT}/{folder}"

        # Pooled [CLS]-only embeddings (pre-2026-07-08 format) — matches
        # what the reported hyperparameter searches were run
        # against. Per-token *_vq_tokens.pt/*_vq_mask.pt files (for
        # TextTokenPooler) still exist on disk if this gets re-enabled later.
        endo_vq = torch.load(f"{base}/{prefix}_{term}_endo_vq_emb.pt")  # [1,     bert_dim]
        exo_vq  = torch.load(f"{base}/{prefix}_{term}_exo_vq_emb.pt")   # [n_exo, bert_dim]

        self.register_buffer('bert_endo_vq', endo_vq)
        self.register_buffer('bert_exo_vq',  exo_vq)

    def _alt_causal_weights(self, exo_patches, endo_patches):
        """Options A/B: causal weights [B*P, n_exo, 1] from the raw patches. Stores the
        estimator's own loss in self.causal_aux_loss and its coefficients / group norms
        in self.causal_coeffs (for visualization)."""
        B, P = endo_patches.shape[:2]
        if getattr(self, "uniform_causal", False):   # rebuttal control: no causal bias (log 1 = 0 in the attention)
            self.causal_aux_loss = None
            return endo_patches.new_ones(B * P, self.n_exo, 1)
        if self.causal_estimator is None:
            self.causal_aux_loss = None
            return None
        x = torch.cat([endo_patches, exo_patches], dim=-1)               # [B, P, pl, 1+n_exo]
        if os.environ.get("MAGNET_SIG_PREWHITEN") == "1":                # idea C (2026-11-14): seasonal pre-whitening
            x = _seasonal_difference(x, {11: 7, 12: 7, 15: 48, 16: 48}.get(self.dataset, 24))   # daily period in steps (Colbun/Rapel daily data: weekly; Sdwpfm 30-min)
        weights, self.causal_aux_loss, self.causal_coeffs = self.causal_estimator(x)
        if getattr(self, "_pnp_self_train", False) and self.training and torch.is_grad_enabled():
            # the backbone trainers do not know the Granger loss; weights are detached, so the estimator can train
            # itself here on L_G alone (equivalent to adding L_G to the total loss)
            if self._est_opt is None:
                self._est_opt = torch.optim.Adam(self.causal_estimator.parameters(), lr=5e-4)
            self._est_opt.zero_grad()
            self.causal_aux_loss.backward()
            self._est_opt.step()
            self.causal_aux_loss = None
        if getattr(self, "detach_causal", False):    # pure Granger graph: only the estimator's own loss trains it
            weights = weights.detach()
        if getattr(self, "gain_head", None) is not None:              # option 2: per-window occlusion gain
            self.gain_pred = self.gain_head(x)                       # [B, n_exo]
            weights = weights * (2 * torch.sigmoid(self.gain_pred)).unsqueeze(1)
        return weights.clamp(min=1e-6).reshape(B * P, self.n_exo, 1)

    def enable_causal_options(self, gate="none", gate_init=0.0, gain_head=False, seq_len=None):
        """Rebuttal options on top of a sparse-Granger estimator (all off by default, so existing checkpoints load
        unchanged). Call before .to(device) and before building the optimizer.
          gate="estimator": exo message scaled by g = sigmoid(logit(r) + b), r = exo_strength_ratio (detached),
                            b learnable, init 0 -> g starts at r            (option 1)
          gate="free"     : g = sigmoid(b), b learnable, init gate_init    (option 1 control: same place, same init,
                            not driven by the estimator)
          gate="fixed"    : g = gate_init, a constant (option 1 redesign: the cell's held-out Granger gain)
          gain_head=True  : GainHead modulates the static weights per window (option 2); needs seq_len"""
        assert gate in ("none", "estimator", "free", "fixed"), gate
        self.causal_gate_mode = gate
        if gate == "fixed":                      # constant gate value (e.g. the held-out Granger gain), no parameter
            self.register_buffer("causal_gate_value", torch.tensor(float(gate_init)))
        elif gate != "none":
            assert self.causal_estimator is not None, "the causal gate needs a causal estimator"
            self.causal_gate_bias = nn.Parameter(torch.tensor(0.0 if gate == "estimator" else float(gate_init)))
        if gain_head:
            assert self.causal_estimator is not None and seq_len is not None
            self.gain_head = GainHead(1 + self.n_exo, seq_len)
            self.gain_pred = None

    def _causal_gate(self):
        mode = getattr(self, "causal_gate_mode", "none")
        if mode == "none":
            return None
        if mode == "fixed":
            return self.causal_gate_value
        if mode == "free":
            return torch.sigmoid(self.causal_gate_bias)
        r = exo_strength_ratio(self.causal_estimator).detach().clamp(1e-4, 1 - 1e-4)
        return torch.sigmoid(torch.log(r / (1 - r)) + self.causal_gate_bias)

    def forward(self, exo_patches, endo_patches, return_aux_loss: bool = False):
        """
        exo_patches  : [B, P, patch_len, n_exo]
        endo_patches : [B, P, patch_len, 1]
        returns      : [B, P, d_model]
        """
        B, P, pl, _ = endo_patches.shape

        # Simple linear projection — [B, P, n_exo, patch_len] → [B, P, n_exo, d_model]
        exo_emb_ts  = self.exo_proj(exo_patches.permute(0, 1, 3, 2))   # [B, P, n_exo, d_model]
        endo_emb_ts = self.endo_proj(endo_patches.permute(0, 1, 3, 2))  # [B, P, 1, d_model]

        # Text fusion — see node_text_fusion docstring in __init__ for the
        # three variants. "qk_bias" (default) matches every existing PnP
        # result across all backbones: node construction gets a zero token,
        # text only biases NGC's Q/K below. "add"/"attn" fuse text directly
        # into the node embedding instead, and disable the NGC Q/K bias
        # (endo_text_bias/exo_text_bias = None) so text enters exactly once.
        if self.use_text and self.node_text_fusion in ("qk_bias", "add"):
            gate = torch.sigmoid(self.text_gate)

            endo_pooled = self.endo_vq_projector(self.bert_endo_vq).squeeze(0)  # [d_model]
            exo_pooled  = self.exo_vq_projector(self.bert_exo_vq)               # [n_exo, d_model]

            endo_text_bias = gate * self.endo_text_norm(endo_pooled)  # [d_model]
            exo_text_bias  = gate * self.exo_text_norm(exo_pooled)    # [n_exo, d_model]
        else:
            endo_text_bias = None
            exo_text_bias  = None

        if self.use_text and self.node_text_fusion == "add":
            # Simple additive fusion — no attention, no ExoNodeConstruction/
            # EndoNodeConstruction pass at all (that module is a no-op residual
            # transform under a zero token anyway). Result still flows into
            # exo_graph/endo_insertion below, same as every other variant —
            # this only changes how text enters, not the causal graph itself.
            exo_fused  = exo_emb_ts + exo_text_bias.unsqueeze(0).unsqueeze(0)      # [B, P, n_exo, d_model]
            endo_fused = (endo_emb_ts.squeeze(2) + endo_text_bias).unsqueeze(2)    # [B, P, 1, d_model]

        elif self.use_text and self.node_text_fusion == "attn":
            # Real cross-attention fusion — patch_emb as Query, the variable's
            # full per-token text sequence as Key/Value (masked to real
            # tokens), via the SAME ExoNodeConstruction/EndoNodeConstruction +
            # VqCrossAttention machinery "qk_bias" mode starves with a zero
            # token. Residual + LayerNorm are internal to VqCrossAttention.
            endo_vq = self.endo_vq_projector(self.bert_endo_vq_tokens)  # [1,     T, d_model]
            exo_vq  = self.exo_vq_projector(self.bert_exo_vq_tokens)    # [n_exo, T, d_model]
            exo_fused  = self.exo_node(exo_emb_ts, exo_vq, vq_mask=self.bert_exo_vq_mask, stats_emb=None)    # [B, P, n_exo, d_model]
            endo_fused = self.endo_node(endo_emb_ts, endo_vq, vq_mask=self.bert_endo_vq_mask, stats_emb=None)  # [B, P, 1, d_model]

        else:
            # "qk_bias" (default, matches all existing PnP results) — node
            # construction always gets a zero vq_emb (text moved out of here
            # entirely), so this is identical whether use_text is True or
            # False: just the learned residual projection + norm of the raw
            # patch embedding via VqCrossAttention's W_v bias term.
            # A single dummy (unpadded) token — VqCrossAttention expects a
            # real token sequence [n_vars, T, d_model]; T=1 zero content
            # reproduces the previous single-vector zero-key behavior exactly.
            zero_endo_vq = torch.zeros(1, 1, self.d_model, device=endo_patches.device, dtype=endo_patches.dtype)
            zero_exo_vq  = torch.zeros(self.n_exo, 1, self.d_model, device=endo_patches.device, dtype=endo_patches.dtype)
            exo_fused  = self.exo_node(exo_emb_ts, zero_exo_vq, vq_mask=None, stats_emb=None)   # [B, P, n_exo, d_model]
            endo_fused = self.endo_node(endo_emb_ts, zero_endo_vq, vq_mask=None, stats_emb=None) # [B, P, 1, d_model]

        # EXPERIMENTAL ngc_lag (see __init__ docstring): build a lagged copy of
        # exo_fused BEFORE the B*P flatten below destroys the patch axis --
        # torch.roll wraps around (patch 0 would otherwise silently borrow
        # patch P-1's, i.e. the FUTURE's, embedding), so boundary patches
        # [0:ngc_lag] are overwritten back to their own contemporaneous value
        # (the only well-defined fallback when there's no earlier patch yet).
        if self.ngc_lag > 0:
            exo_fused_lagged = torch.roll(exo_fused, shifts=self.ngc_lag, dims=1)
            exo_fused_lagged[:, :self.ngc_lag] = exo_fused[:, :self.ngc_lag]
            exo_all_lagged = exo_fused_lagged.reshape(B * P, self.n_exo, -1)
        else:
            exo_all_lagged = None

        # ngc_window (see NeuralGrangerCausalityWindowed / __init__ docstring):
        # build, for every patch p, a K-patch PAST window (p-K..p-1) BEFORE
        # the B*P flatten destroys the patch axis. Replicate-pad the first K
        # patches with patch 0 (same boundary convention as ngc_lag above --
        # no earlier patch exists yet, so the earliest available one stands
        # in), then a sliding unfold gives every window in one shot.
        if self.ngc_window > 0:
            K = self.ngc_window
            exo_pad  = exo_fused[:, :1].expand(-1, K, -1, -1)          # [B, K, n_exo, d_model]
            endo_pad = endo_fused[:, :1].expand(-1, K, -1, -1)         # [B, K, 1, d_model]
            exo_padded  = torch.cat([exo_pad, exo_fused], dim=1)       # [B, P+K, n_exo, d_model]
            endo_padded = torch.cat([endo_pad, endo_fused], dim=1)     # [B, P+K, 1, d_model]

            # unfold(dim=1, size=K, step=1) appends a new last dim of size K;
            # window i (0-indexed) covers padded positions [i, i+K-1] = original
            # patches [i-K, i-1] -- exactly the K patches strictly BEFORE patch i.
            exo_windows = exo_padded.unfold(1, K, 1)[:, :P]             # [B, P, n_exo, d_model, K]
            exo_windows = exo_windows.permute(0, 1, 4, 2, 3)            # [B, P, K, n_exo, d_model]
            endo_windows = endo_padded.unfold(1, K, 1)[:, :P]           # [B, P, 1, d_model, K]
            endo_windows = endo_windows.permute(0, 1, 4, 2, 3)          # [B, P, K, 1, d_model]

            exo_window_all  = exo_windows.reshape(B * P, K, self.n_exo, -1)   # [B*P, K, n_exo, d_model]
            endo_window_all = endo_windows.reshape(B * P, K, -1)              # [B*P, K, d_model]
        else:
            exo_window_all  = None
            endo_window_all = None

        # Flatten B*P for graph
        exo_all  = exo_fused.reshape(B * P, self.n_exo, -1)  # [B*P, n_exo, d_model]
        endo_all = endo_fused.reshape(B * P, 1, -1)           # [B*P, 1, d_model]

        # Exo-exo GAT
        # skip_exo_gat (module ablation on detach_gg, 2026-11-11): endo attends to the pre-GAT exo embeddings
        exo_gat = exo_all if getattr(self, "skip_exo_gat", False) else self.exo_graph(exo_all, None)[0]

        # Exo-endo NGC insertion — text enters here as a Q/K bias ONLY in
        # "qk_bias" mode. "add"/"attn" already fused text into the node
        # embedding above, so the bias is forced to None here to avoid
        # injecting text twice through two different mechanisms.
        ngc_endo_text_bias = endo_text_bias if self.node_text_fusion == "qk_bias" else None
        ngc_exo_text_bias  = exo_text_bias  if self.node_text_fusion == "qk_bias" else None
        endo_insertion_out = self.endo_insertion(
            exo_all, endo_all, exo_gat, None,
            endo_text_bias=ngc_endo_text_bias, exo_text_bias=ngc_exo_text_bias,
            exo_emb_ts_lagged=exo_all_lagged,
            exo_emb_ts_extra=exo_all if self.ngc_use_current_and_lag else None,
            endo_emb_window=endo_window_all, exo_emb_window=exo_window_all,
            return_aux_loss=return_aux_loss,
            causal_weights_override=self._alt_causal_weights(exo_patches, endo_patches),
            exo_gate=self._causal_gate(),
        )  # [B*P, 1, d_model]

        # Return causal bias [B, P, d_model]
        if return_aux_loss:
            endo_out, _, aux_loss = endo_insertion_out
            return endo_out.squeeze(1).reshape(B, P, -1), aux_loss
        endo_out, _ = endo_insertion_out
        return endo_out.squeeze(1).reshape(B, P, -1)

    def forward_with_causal_weights(self, exo_patches, endo_patches):
        """
        Identical to forward(), but also returns the per-exo-variable NGC
        causal weights that self.endo_insertion computes internally and
        forward() normally discards (via `endo_out, _ = self.endo_insertion(...)`).
        Added for interpretability/visualization (e.g. plotting which exo
        variable the graph leans on per patch) — forward() itself is left
        untouched so every existing caller (MAGNET_with_*.py wrappers) is
        unaffected.

        exo_patches  : [B, P, patch_len, n_exo]
        endo_patches : [B, P, patch_len, 1]
        returns      : (graph_bias [B, P, d_model], causal_weights [B, P, n_exo])
                        causal_weights are softmax-normalized over the exo
                        dimension (sum to 1 per patch per sample) — see
                        CausalWeightFused in endo_insertion.py.
        """
        B, P, pl, _ = endo_patches.shape

        exo_emb_ts  = self.exo_proj(exo_patches.permute(0, 1, 3, 2))
        endo_emb_ts = self.endo_proj(endo_patches.permute(0, 1, 3, 2))

        if self.use_text:
            gate = torch.sigmoid(self.text_gate)
            endo_pooled = self.endo_vq_projector(self.bert_endo_vq).squeeze(0)
            exo_pooled  = self.exo_vq_projector(self.bert_exo_vq)
            endo_text_bias = gate * self.endo_text_norm(endo_pooled)
            exo_text_bias  = gate * self.exo_text_norm(exo_pooled)
        else:
            endo_text_bias = None
            exo_text_bias  = None

        zero_endo_vq = torch.zeros(1, 1, self.d_model, device=endo_patches.device, dtype=endo_patches.dtype)
        zero_exo_vq  = torch.zeros(self.n_exo, 1, self.d_model, device=endo_patches.device, dtype=endo_patches.dtype)
        exo_fused  = self.exo_node(exo_emb_ts, zero_exo_vq, vq_mask=None, stats_emb=None)
        endo_fused = self.endo_node(endo_emb_ts, zero_endo_vq, vq_mask=None, stats_emb=None)

        exo_all  = exo_fused.reshape(B * P, self.n_exo, -1)
        endo_all = endo_fused.reshape(B * P, 1, -1)

        exo_gat = exo_all if getattr(self, "skip_exo_gat", False) else self.exo_graph(exo_all, None)[0]

        # same lagged NGC input as forward(): patch p is explained by exo patch p-1
        exo_all_lagged = None
        if self.ngc_lag > 0:
            exo_fused_lagged = torch.roll(exo_fused, shifts=self.ngc_lag, dims=1)
            exo_fused_lagged[:, :self.ngc_lag] = exo_fused[:, :self.ngc_lag]
            exo_all_lagged = exo_fused_lagged.reshape(B * P, self.n_exo, -1)

        # causal_weights: [B*P, n_exo, 1] — this is the tensor forward() discards
        endo_out, causal_weights = self.endo_insertion(
            exo_all, endo_all, exo_gat, None,
            endo_text_bias=endo_text_bias, exo_text_bias=exo_text_bias,
            exo_emb_ts_lagged=exo_all_lagged,
            causal_weights_override=self._alt_causal_weights(exo_patches, endo_patches),
            exo_gate=self._causal_gate(),
        )

        graph_bias     = endo_out.squeeze(1).reshape(B, P, -1)
        causal_weights = causal_weights.squeeze(-1).reshape(B, P, self.n_exo)
        return graph_bias, causal_weights

    def forward_with_intermediate_embeddings(self, exo_patches, endo_patches):
        """
        Identical to forward(), but also returns the pre-graph endo patch
        embedding (raw_emb) alongside the usual post-graph causal bias
        (graph_bias) — forward() itself is left untouched, same "pure
        addition" pattern as forward_with_causal_weights(). Added for the
        t-SNE representation visualization (option 4 in
        visualize_ablations.py): raw_emb is just endo_proj(endo_patches),
        the representation BEFORE any exo-exo GAT or exo-endo NGC causal
        attention touches it; graph_bias is the SAME final output forward()
        returns, i.e. AFTER both. Both live in the same [B, P, d_model]
        space (this dataset/backbone's own d_model), which is what makes
        them a fair apples-to-apples before/after comparison.

        exo_patches  : [B, P, patch_len, n_exo]
        endo_patches : [B, P, patch_len, 1]
        returns      : (raw_emb [B, P, d_model], graph_bias [B, P, d_model])
        """
        B, P, pl, _ = endo_patches.shape

        exo_emb_ts  = self.exo_proj(exo_patches.permute(0, 1, 3, 2))
        endo_emb_ts = self.endo_proj(endo_patches.permute(0, 1, 3, 2))
        raw_emb = endo_emb_ts.squeeze(2)  # [B, P, d_model], pre-graph

        if self.use_text:
            gate = torch.sigmoid(self.text_gate)
            endo_pooled = self.endo_vq_projector(self.bert_endo_vq).squeeze(0)
            exo_pooled  = self.exo_vq_projector(self.bert_exo_vq)
            endo_text_bias = gate * self.endo_text_norm(endo_pooled)
            exo_text_bias  = gate * self.exo_text_norm(exo_pooled)
        else:
            endo_text_bias = None
            exo_text_bias  = None

        zero_endo_vq = torch.zeros(1, 1, self.d_model, device=endo_patches.device, dtype=endo_patches.dtype)
        zero_exo_vq  = torch.zeros(self.n_exo, 1, self.d_model, device=endo_patches.device, dtype=endo_patches.dtype)
        exo_fused  = self.exo_node(exo_emb_ts, zero_exo_vq, vq_mask=None, stats_emb=None)
        endo_fused = self.endo_node(endo_emb_ts, zero_endo_vq, vq_mask=None, stats_emb=None)

        exo_all  = exo_fused.reshape(B * P, self.n_exo, -1)
        endo_all = endo_fused.reshape(B * P, 1, -1)

        exo_gat = exo_all if getattr(self, "skip_exo_gat", False) else self.exo_graph(exo_all, None)[0]

        endo_out, _ = self.endo_insertion(
            exo_all, endo_all, exo_gat, None,
            endo_text_bias=endo_text_bias, exo_text_bias=exo_text_bias,
        )

        graph_bias = endo_out.squeeze(1).reshape(B, P, -1)
        return raw_emb, graph_bias


def _seasonal_difference(x, s):
    """Pre-whitening for the Granger estimator only (idea C): every series of the window (flattened over patches) is
    replaced by its seasonal difference x_t - x_{t-s} (s = 24 hourly, 7 daily), removing the level and the shared
    daily / weekly cycle, i.e. the same-time co-movement that is not a genuine instantaneous dependence. The first s
    steps (no t-s inside the window) are set to 0. x [B, P, pl, V] -> same shape."""
    B, P, pl, V = x.shape
    f = x.reshape(B, P * pl, V)
    d = torch.zeros_like(f)
    if P * pl > s:
        d[:, s:] = f[:, s:] - f[:, :-s]
    return d.reshape(B, P, pl, V)
