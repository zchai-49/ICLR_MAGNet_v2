"""Deployment configuration for the official MAGNet+DLinear models (no future exogenous input).

deployment_final.json lists, for every (dataset, term) cell of the paper, which ensemble
side(s) are deployed and each side's hyperparameters / capacity recipe.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("models", "data", "configs"):
    _p = os.path.join(ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import final_defaults  # noqa: E402,F401  official configuration (SIG v2), see models/final_defaults.py

from MAGNET_with_DLinear_fusion import MAGNETWithDLinear  # noqa: E402

SEEDS = [0, 42, 123, 2024, 3407]
MAGNET_N_HEADS = {32: 2, 64: 4, 128: 8}

# id: (csv path relative to ROOT/data/dataset, number of exogenous variables, name)
DATASETS = {
    5:  ("EPF/BE.csv",         2, "BE"),
    6:  ("EPF/DE.csv",         2, "DE"),
    7:  ("EPF/FR.csv",         2, "FR"),
    8:  ("EPF/NP.csv",         2, "NP"),
    9:  ("EPF/PJM.csv",        2, "PJM"),
    10: ("energy/Energy.csv",  5, "Energy"),
    11: ("hydro/Colbun.csv",   2, "Colbun"),
    12: ("hydro/Rapel.csv",    2, "Rapel"),
    13: ("wind/Sdwpfh1.csv",   6, "Sdwpfh1"),
    14: ("wind/Sdwpfh2.csv",   6, "Sdwpfh2"),
    15: ("wind/Sdwpfm1.csv",   6, "Sdwpfm1"),
    16: ("wind/Sdwpfm2.csv",   6, "Sdwpfm2"),
    # synthetic datasets with a known dependency structure (scripts/make_synthetic.py); generic wide-CSV loader
    18: ("generic/Syn5.csv",       5, "Syn5"),
    19: ("generic/Syn20.csv",     20, "Syn20"),
    20: ("generic/Syn50.csv",     50, "Syn50"),
    21: ("generic/Syn20weak.csv", 20, "Syn20weak"),
    22: ("generic/Syn20null.csv", 20, "Syn20null"),
}
SYNTHETIC = {18, 19, 20, 21, 22}   # synthetic: text from statistics-only prompts (models/text_generate_generic.py), on by MAGNET_SYN_TEXT=1 (default)
NAME_TO_ID = {v[2]: k for k, v in DATASETS.items()}
DIFF_DATASETS = {11}            # long-term Colbun is modeled on first differences
TTA_DATASETS = {8, 13, 14, 15, 16}  # test-time noise averaging on the noisy datasets

# MAGNET_DEPLOYMENT (default configs/deployment_final.json, models/final_defaults.py) selects the deployment file
with open(os.environ.get("MAGNET_DEPLOYMENT", os.path.join(ROOT, "configs", "deployment_final.json"))) as _f:
    DEPLOYMENT = json.load(_f)


def csv_path(dataset_id):
    return os.path.join(ROOT, "data", "dataset", DATASETS[dataset_id][0])


def get_seq_horizon(dataset_id, term):
    if dataset_id in (11, 12):
        return (60, 10) if term == "short" else (180, 30)
    return (168, 24) if term == "short" else (720, 360)


# Every injection mechanism of the original DLinear-fusion screen (MAGNET/run_dlinear_fusion_screen.py VARIANTS +
# the original scalar injection); 2026-11-10 fully validation-based re-selection screens all of them.
MECHANISMS = {
    "scalar_only": dict(use_scalar_injection=True),
    "dual_only": dict(use_scalar_injection=False, use_dual_injection=True),
    "postlinear_film_only": dict(use_scalar_injection=False, use_postlinear_film_injection=True),
    "gated_mixing_only": dict(use_scalar_injection=False, use_gated_mixing_injection=True),
    "crossattn_only": dict(use_scalar_injection=False, use_crossattn_injection=True),
    "hyper_weight_only": dict(use_scalar_injection=False, use_hyper_weight_injection=True),
    "gated_scale_only": dict(use_scalar_injection=False, use_gated_scale_injection=True),
    "dual_gated_scale": dict(use_scalar_injection=False, use_dual_injection=True, use_gated_scale_injection=True),
    "dual_crossattn": dict(use_scalar_injection=False, use_dual_injection=True, use_crossattn_injection=True),
    "dual_hyper_weight": dict(use_scalar_injection=False, use_dual_injection=True, use_hyper_weight_injection=True),
    "dual_gated_scale_crossattn": dict(use_scalar_injection=False, use_dual_injection=True,
                                       use_gated_scale_injection=True, use_crossattn_injection=True),
    "crossattn_mq": dict(use_scalar_injection=False, use_dual_injection=True,          # = dual_crossattn_mq
                         use_crossattn_injection=True, crossattn_n_queries=4),
}


def side_injection_kwargs(side):
    if side not in MECHANISMS:
        raise ValueError(side)
    return dict(MECHANISMS[side])


def cell_config(dataset_name, term):
    return DEPLOYMENT[f"{dataset_name}/{term}"]


CAUSAL_METHODS = ("ngc", "gvar", "sparse_granger", "sparse_granger_prox",
                  "sparse_granger_lag0", "sparse_granger_prox_lag0")


def build_model(dataset_name, term, side, patch_len=None, use_text=True, causal="ngc"):
    """MAGNet+DLinear for one (dataset, term, ensemble side). Returns (model, patch_len).
    `patch_len` overrides the deployed patch length (used by the no_patching ablation); `use_text=False` drops the text.
    `causal` picks the exo->endo causal-weight estimator: "ngc" (default), "gvar", "sparse_granger" (group-lasso
    penalty) or "sparse_granger_prox" (proximal group lasso, exact zeros); a "_lag0" suffix adds same-patch exo inputs."""
    assert causal in CAUSAL_METHODS, causal
    cell = cell_config(dataset_name, term)
    dataset_id = cell["dataset_id"]
    # 2026-11-14: synthetic text (models/text_generate_generic.py, statistics-only prompts) enabled by MAGNET_SYN_TEXT=1
    use_text = use_text and (dataset_id not in SYNTHETIC or os.environ.get("MAGNET_SYN_TEXT") == "1")
    s = cell["sides"][side]
    seq_len, horizon = get_seq_horizon(dataset_id, term)
    n_exo = DATASETS[dataset_id][1]
    patch_len = patch_len or s["patch_len"]
    d_graph = s["hparams"]["magnet_d_model"]
    magnet_config = {
        "n_exo": n_exo, "patch_len": patch_len, "num_patches": seq_len // patch_len,
        "d_model": d_graph, "n_heads": MAGNET_N_HEADS.get(d_graph, 4),
        "ngc_hidden_dim": s.get("ngc_hidden_dim") or 32,
        "dropout": 0.2 if term == "long" else 0.1,
        "dataset": dataset_id, "term": term,
        "use_gvar_causal": causal == "gvar", "use_sparse_granger_causal": causal.startswith("sparse_granger"),
        "sparse_granger_proximal": causal.startswith("sparse_granger_prox"),
        "sparse_granger_contemporaneous": causal.endswith("_lag0"),
    }
    dlinear_config = {"seq_len": seq_len, "pred_len": horizon,
                      "moving_avg": s["hparams"]["moving_avg"]}
    kwargs = dict(side_injection_kwargs(side), **s["kwargs"])
    model = MAGNETWithDLinear(magnet_config, dlinear_config, mode=2, use_text=use_text,
                              use_instance_norm=False, **kwargs)
    return model, patch_len


def checkpoint_path(dataset_name, term, side, seed):
    return os.path.join(ROOT, "checkpoints", f"{term}_{dataset_name}_{side}", f"seed_{seed}", "ckpt.pt")
