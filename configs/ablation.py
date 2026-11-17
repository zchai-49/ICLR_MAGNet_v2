"""Five-way module ablation of MAGNet+DLinear: full / no_text / no_exo_gat / no_ngc / no_patching.

Every ablated model reuses the deployed side's hyperparameters, injection recipe and training regime; only the ablated
component changes.
    no_text     text conditioning removed (MAGNETGraphOnly(use_text=False))
    no_exo_gat  exo-exo GAT removed (endo attends directly to the exogenous node embeddings; NGC unchanged)
    no_ngc      exo->endo NGC causal weight replaced by a random trainable parameter (exo-exo graph unchanged)
    no_patching one patch covering the whole window (patch_len = seq_len)
"""
import inspect
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "configs"))
import lag1_config as C  # noqa: E402
from MAGNET_graph_only_ablation import MAGNETGraphOnlyNoExoGAT, MAGNETGraphOnlyRandomNGC  # noqa: E402

VARIANTS = ["full", "no_text", "no_exo_gat", "no_ngc", "no_patching"]


def build_ablation_model(dataset_name, term, side, variant="full", causal="ngc"):
    """Returns (model, patch_len) for one ablation variant of one deployed side.
    `causal` is the exo->endo estimator of the full model (default ngc). no_ngc removes the estimator, so it is the same
    model whatever `causal` is; for a sparse-Granger full model the other variants keep that estimator."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    cell = C.cell_config(dataset_name, term)
    dataset_id = cell["dataset_id"]
    seq_len, _ = C.get_seq_horizon(dataset_id, term)
    if variant == "no_ngc":
        causal = "ngc"                                     # the random-weight module replaces any estimator
    if variant == "no_patching":
        return C.build_model(dataset_name, term, side, patch_len=seq_len, causal=causal)
    if variant == "no_text":
        return C.build_model(dataset_name, term, side, use_text=False, causal=causal)
    model, patch_len = C.build_model(dataset_name, term, side, causal=causal)
    if variant in ("no_exo_gat", "no_ngc"):
        s = cell["sides"][side]
        d_graph = s["hparams"]["magnet_d_model"]
        cls = MAGNETGraphOnlyNoExoGAT if variant == "no_exo_gat" else MAGNETGraphOnlyRandomNGC
        kw = dict(n_exo=C.DATASETS[dataset_id][1], patch_len=patch_len, d_model=d_graph,
                  n_heads=C.MAGNET_N_HEADS.get(d_graph, 4), dropout=0.2 if term == "long" else 0.1,
                  dataset=dataset_id, term=term)
        if "ngc_hidden_dim" in inspect.signature(cls.__init__).parameters:
            kw["ngc_hidden_dim"] = s.get("ngc_hidden_dim") or 32
        if variant == "no_exo_gat" and causal != "ngc":
            assert causal.startswith("sparse_granger"), f"no_exo_gat is wired for sparse-Granger estimators only: {causal}"
            kw.update(use_sparse_granger_causal=True,
                      sparse_granger_contemporaneous=causal.endswith("_lag0"),
                      sparse_granger_proximal=causal.startswith("sparse_granger_prox"))
        model.magnet_graph = cls(**kw)
    return model, patch_len
