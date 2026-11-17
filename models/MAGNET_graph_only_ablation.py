# MAGNET_graph_only_ablation.py
# Module-ablation variants of MAGNETGraphOnly used by the five-way MAGNet+DLinear ablation
# (full / no_text / no_exo_gat / no_ngc / no_patching); see train/run_dlinear_ablation.py.
#   no_exo_gat -- endo attends directly to the exogenous node embeddings, skipping the exo-exo GAT
#                 (the NGC weighting is unchanged and uses the same lagged input as the full module)
#   no_ngc     -- the exo->endo NGC causal weight is replaced by a random trainable parameter
#                 (the exo-exo graph stays real)
# no_text is MAGNETGraphOnly(use_text=False); no_patching is the full module with a single patch (patch_len = seq_len).

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from exo_graph import ExoNodeConstruction, ExoGAT, ExoGraph
from endo_insertion import EndoNodeConstruction, AttentionAggregator, EndoNodeInsertion
from causal_alternatives import SparseInputGranger
from text_projection import TextProjection

_TEXT_EMB_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "text_emb")


_TEXT_EMB_PATHS = {
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


class MAGNETGraphOnlyNoExoGAT(nn.Module):
    """
    Main-module ablation variant "no_exo_gat": endo attends directly to
    exo's post-node-construction patch embeddings (exo_fused), skipping
    ExoGraph's cosine-similarity edge weights + GAT update entirely.

    NGC is UNCHANGED by this ablation — EndoNodeInsertion's NGC input was
    already the pre-GAT "straight from Fusion" exo embedding (see its own
    docstring: exo_emb_ts "captures raw temporal pattern"), so removing
    ExoGraph only changes what AttentionAggregator uses as Key/Value, not
    what NGC scores. Concretely, both the 1st and 3rd positional args to
    EndoNodeInsertion.forward (normally exo_all, exo_gat) are exo_all here.
    Isolates "does exo-exo correlation reasoning matter" independently from
    "does the exo->endo causal weighting matter" (that's no_ngc, below) —
    the two are orthogonal axes.

    Text fusion identical to MAGNETGraphOnly's default "qk_bias" mode.

    Input:
        exo_patches  : [B, P, patch_len, n_exo]
        endo_patches : [B, P, patch_len, 1]

    Output:
        bias         : [B, P, d_model]
    """

    def __init__(
        self,
        n_exo         : int,
        patch_len     : int,
        d_model       : int = 64,
        n_heads       : int = 4,
        ngc_hidden_dim: int = 32,
        dropout       : float = 0.1,
        dataset       : int = None,
        term          : str = "short",
        bert_dim      : int = 768,
        use_text      : bool = True,
        text_variant  : str = "new",
        ngc_lag       : int = 1,
        use_sparse_granger_causal      : bool = False,  # 2026-10-27: same re-route as MAGNETGraphOnly -- the
        sparse_granger_contemporaneous : bool = False,  # exo->endo weights come from SparseInputGranger instead
        sparse_granger_proximal        : bool = False,  # of NGC (False = exact previous behavior)
    ):
        super().__init__()
        self.ngc_lag  = ngc_lag
        self.n_exo    = n_exo
        self.d_model  = d_model
        self.term     = term
        self.dataset  = dataset
        self.use_text = use_text
        self.text_variant = text_variant

        self.exo_proj  = nn.Linear(patch_len, d_model)
        self.endo_proj = nn.Linear(patch_len, d_model)

        if self.use_text:
            self.endo_vq_projector = TextProjection(bert_dim, d_model)
            self.exo_vq_projector  = TextProjection(bert_dim, d_model)
            self.endo_text_norm    = nn.LayerNorm(d_model)
            self.exo_text_norm     = nn.LayerNorm(d_model)
            self.text_gate = nn.Parameter(torch.tensor(-2.0))
        else:
            self.endo_vq_projector = None
            self.exo_vq_projector  = None
            self.endo_text_norm    = None
            self.exo_text_norm     = None
            self.text_gate         = None

        # Node construction — identical to MAGNETGraphOnly's "qk_bias" path.
        self.exo_node  = ExoNodeConstruction(n_exo=n_exo, d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.endo_node = EndoNodeConstruction(n_exo=n_exo, d_model=d_model, n_heads=n_heads, dropout=dropout)

        # NOTE: deliberately no self.exo_graph here — that IS the ablation.
        # EndoNodeInsertion reused UNMODIFIED (real NGC internally).
        self.endo_insertion = EndoNodeInsertion(
            n_exo=n_exo, d_model=d_model, n_heads=n_heads,
            hidden_dim=ngc_hidden_dim, dropout=dropout,
        )
        self.causal_estimator = (SparseInputGranger(1 + n_exo, patch_len, hidden_dim=ngc_hidden_dim,
                                                    proximal=sparse_granger_proximal,
                                                    contemporaneous=sparse_granger_contemporaneous)
                                 if use_sparse_granger_causal else None)
        self.causal_aux_loss = None

        if self.use_text and dataset is not None:
            self._load_text_embeddings(dataset, term)

    def _load_text_embeddings(self, dataset, term):
        if dataset not in _TEXT_EMB_PATHS:
            raise ValueError(f"Unknown dataset id: {dataset}")
        folder, base_name = _TEXT_EMB_PATHS[dataset]
        prefix = f"{base_name}_{self.text_variant}"
        base   = f"{_TEXT_EMB_ROOT}/{folder}"

        endo_vq = torch.load(f"{base}/{prefix}_{term}_endo_vq_emb.pt")  # [1,     bert_dim]
        exo_vq  = torch.load(f"{base}/{prefix}_{term}_exo_vq_emb.pt")   # [n_exo, bert_dim]

        self.register_buffer('bert_endo_vq', endo_vq)
        self.register_buffer('bert_exo_vq',  exo_vq)

    def forward(self, exo_patches, endo_patches):
        B, P, pl, _ = endo_patches.shape

        exo_emb_ts  = self.exo_proj(exo_patches.permute(0, 1, 3, 2))    # [B, P, n_exo, d_model]
        endo_emb_ts = self.endo_proj(endo_patches.permute(0, 1, 3, 2))  # [B, P, 1, d_model]

        if self.use_text:
            gate = torch.sigmoid(self.text_gate)
            endo_pooled = self.endo_vq_projector(self.bert_endo_vq).squeeze(0)  # [d_model]
            exo_pooled  = self.exo_vq_projector(self.bert_exo_vq)               # [n_exo, d_model]
            endo_text_bias = gate * self.endo_text_norm(endo_pooled)  # [d_model]
            exo_text_bias  = gate * self.exo_text_norm(exo_pooled)    # [n_exo, d_model]
        else:
            endo_text_bias = None
            exo_text_bias  = None

        zero_endo_vq = torch.zeros(1, 1, self.d_model, device=endo_patches.device, dtype=endo_patches.dtype)
        zero_exo_vq  = torch.zeros(self.n_exo, 1, self.d_model, device=endo_patches.device, dtype=endo_patches.dtype)
        exo_fused  = self.exo_node(exo_emb_ts, zero_exo_vq, vq_mask=None, stats_emb=None)   # [B, P, n_exo, d_model]
        endo_fused = self.endo_node(endo_emb_ts, zero_endo_vq, vq_mask=None, stats_emb=None) # [B, P, 1, d_model]

        exo_all  = exo_fused.reshape(B * P, self.n_exo, -1)  # [B*P, n_exo, d_model]
        endo_all = endo_fused.reshape(B * P, 1, -1)           # [B*P, 1, d_model]

        # No ExoGraph step — endo attends directly to exo_all (post node
        # construction, pre-GAT) as BOTH the NGC input and the aggregator's
        # Key/Value, instead of the GAT-refined exo_gat.
        # Same lagged NGC input as the full module (patch p is explained by exo patch p-1; boundary patches
        # keep their own contemporaneous embedding), so this ablation removes only the exo-exo GAT.
        exo_fused_lagged = torch.roll(exo_fused, shifts=self.ngc_lag, dims=1)
        exo_fused_lagged[:, :self.ngc_lag] = exo_fused[:, :self.ngc_lag]
        exo_all_lagged = exo_fused_lagged.reshape(B * P, self.n_exo, -1)

        override = None
        if self.causal_estimator is not None:
            w, self.causal_aux_loss, self.causal_coeffs = self.causal_estimator(torch.cat([endo_patches, exo_patches], -1))
            override = w.clamp(min=1e-6).reshape(B * P, self.n_exo, 1)

        endo_out, _ = self.endo_insertion(
            exo_all, endo_all, exo_all, None,
            endo_text_bias=endo_text_bias, exo_text_bias=exo_text_bias,
            exo_emb_ts_lagged=exo_all_lagged, causal_weights_override=override,
        )  # [B*P, 1, d_model]

        return endo_out.squeeze(1).reshape(B, P, -1)


class MAGNETGraphOnlyRandomNGC(nn.Module):
    """
    Main-module ablation variant "no_ngc": mirror image of
    MAGNETGraphOnlyNoExoGAT above. Keeps ExoGraph's REAL cosine-similarity
    exo-exo edge weights and GAT update, but replaces the exo->endo causal
    weight — normally NeuralGrangerCausality's restricted/unrestricted LSTM
    comparison — with a randomly-initialized, trainable [n_exo] parameter
    that never looks at the data.

    Distinct from MAGNETGraphOnlyRandomCausal ("no_causal_graph", above),
    which randomizes BOTH the exo-exo edge weights AND the exo->endo causal
    weight together — using that for "does learned causality matter" would
    double-count with no_exo_gat. This variant isolates NGC alone, so
    no_exo_gat and no_ngc are orthogonal, non-overlapping axes.

    Text fusion identical to MAGNETGraphOnly's default "qk_bias" mode.

    Input:
        exo_patches  : [B, P, patch_len, n_exo]
        endo_patches : [B, P, patch_len, 1]

    Output:
        bias         : [B, P, d_model]
    """

    def __init__(
        self,
        n_exo        : int,
        patch_len    : int,
        d_model      : int = 64,
        n_heads      : int = 4,
        dropout      : float = 0.1,
        dataset      : int = None,
        term         : str = "short",
        bert_dim     : int = 768,
        use_text     : bool = True,
        text_variant : str = "new",
    ):
        super().__init__()
        self.n_exo    = n_exo
        self.d_model  = d_model
        self.term     = term
        self.dataset  = dataset
        self.use_text = use_text
        self.text_variant = text_variant

        self.exo_proj  = nn.Linear(patch_len, d_model)
        self.endo_proj = nn.Linear(patch_len, d_model)

        if self.use_text:
            self.endo_vq_projector = TextProjection(bert_dim, d_model)
            self.exo_vq_projector  = TextProjection(bert_dim, d_model)
            self.endo_text_norm    = nn.LayerNorm(d_model)
            self.exo_text_norm     = nn.LayerNorm(d_model)
            self.text_gate = nn.Parameter(torch.tensor(-2.0))
        else:
            self.endo_vq_projector = None
            self.exo_vq_projector  = None
            self.endo_text_norm    = None
            self.exo_text_norm     = None
            self.text_gate         = None

        self.exo_node  = ExoNodeConstruction(n_exo=n_exo, d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.endo_node = EndoNodeConstruction(n_exo=n_exo, d_model=d_model, n_heads=n_heads, dropout=dropout)

        # Exo-exo aggregation — REAL ExoGraph (cosine-similarity edge
        # weights + GAT), completely unmodified.
        self.exo_graph = ExoGraph(n_exo=n_exo, d_model=d_model, n_heads=n_heads, dropout=dropout)

        # Exo->endo aggregation — AttentionAggregator reused unmodified;
        # causal weights are a random trainable [n_exo] parameter instead of
        # NeuralGrangerCausality's restricted/unrestricted LSTM comparison.
        self.aggregator = AttentionAggregator(d_model, n_heads, dropout)
        self.random_causal_logits = nn.Parameter(torch.randn(n_exo) * 0.02)

        if self.use_text and dataset is not None:
            self._load_text_embeddings(dataset, term)

    def _load_text_embeddings(self, dataset, term):
        if dataset not in _TEXT_EMB_PATHS:
            raise ValueError(f"Unknown dataset id: {dataset}")
        folder, base_name = _TEXT_EMB_PATHS[dataset]
        prefix = f"{base_name}_{self.text_variant}"
        base   = f"{_TEXT_EMB_ROOT}/{folder}"

        endo_vq = torch.load(f"{base}/{prefix}_{term}_endo_vq_emb.pt")  # [1,     bert_dim]
        exo_vq  = torch.load(f"{base}/{prefix}_{term}_exo_vq_emb.pt")   # [n_exo, bert_dim]

        self.register_buffer('bert_endo_vq', endo_vq)
        self.register_buffer('bert_exo_vq',  exo_vq)

    def forward(self, exo_patches, endo_patches):
        B, P, pl, _ = endo_patches.shape

        exo_emb_ts  = self.exo_proj(exo_patches.permute(0, 1, 3, 2))    # [B, P, n_exo, d_model]
        endo_emb_ts = self.endo_proj(endo_patches.permute(0, 1, 3, 2))  # [B, P, 1, d_model]

        if self.use_text:
            gate = torch.sigmoid(self.text_gate)
            endo_pooled = self.endo_vq_projector(self.bert_endo_vq).squeeze(0)  # [d_model]
            exo_pooled  = self.exo_vq_projector(self.bert_exo_vq)               # [n_exo, d_model]
            endo_text_bias = gate * self.endo_text_norm(endo_pooled)  # [d_model]
            exo_text_bias  = gate * self.exo_text_norm(exo_pooled)    # [n_exo, d_model]
        else:
            endo_text_bias = None
            exo_text_bias  = None

        zero_endo_vq = torch.zeros(1, 1, self.d_model, device=endo_patches.device, dtype=endo_patches.dtype)
        zero_exo_vq  = torch.zeros(self.n_exo, 1, self.d_model, device=endo_patches.device, dtype=endo_patches.dtype)
        exo_fused  = self.exo_node(exo_emb_ts, zero_exo_vq, vq_mask=None, stats_emb=None)   # [B, P, n_exo, d_model]
        endo_fused = self.endo_node(endo_emb_ts, zero_endo_vq, vq_mask=None, stats_emb=None) # [B, P, 1, d_model]

        exo_all  = exo_fused.reshape(B * P, self.n_exo, -1)  # [B*P, n_exo, d_model]
        endo_all = endo_fused.reshape(B * P, 1, -1)           # [B*P, 1, d_model]

        exo_gat, _ = self.exo_graph(exo_all, None)  # real cosine-similarity edge weights + GAT

        # Random trainable exo->endo causal weights (replaces NGC), softmaxed
        # and broadcast over the flattened batch — same contract
        # NeuralGrangerCausality+CausalWeightFused's output has.
        causal_weights = F.softmax(self.random_causal_logits, dim=-1) \
                           .unsqueeze(0).unsqueeze(-1).expand(B * P, -1, -1)  # [B*P, n_exo, 1]

        endo_out = self.aggregator(
            exo_gat, endo_all, causal_weights,
            endo_text_bias=endo_text_bias, exo_text_bias=exo_text_bias,
        )  # [B*P, 1, d_model]

        return endo_out.squeeze(1).reshape(B, P, -1)
