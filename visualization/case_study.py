#!/usr/bin/env python3
"""Case study figure (paper: FR): ground truth vs MAGNet+DLinear (5-seed ensemble over the deployed sides, shipped checkpoints)
vs DLinear alone (5-seed ensemble), on the test windows where MAGNet improves most, with the exogenous variable of largest
Granger weight plotted over the horizon (display only; models never see future exogenous values).
DLinear-alone checkpoints are not shipped: missing ones are trained first (train/train_dlinear_alone.py, ~minutes per seed).
    python3 visualization/case_study.py                                   # FR, short and long
    python3 visualization/case_study.py --datasets FR Syn20 --terms short --gpu 0
-> figures/case_study_<dataset>_<term>.{png,pdf}"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("configs", "scripts", "train", "visualization", "models", "data"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import torch  # noqa: E402
import make_figures as MF  # noqa: E402
import train_options as O  # noqa: E402

C = MF.C
for _i, _n in ((18, "Syn5"), (19, "Syn20"), (20, "Syn50"), (21, "Syn20weak"), (22, "Syn20null")):
    MF.PU._EXO_JSON[_i] = f"generic/{_n}.json"


def magnet_models(ds, term, device, ckpt_root):
    cell = f"{ds}/{term}"
    for side in C.DEPLOYMENT[cell]["deployed_sides"]:
        sel = C.DEPLOYMENT[cell]["sides"][side]["selected"]; saved = json.loads(json.dumps(C.DEPLOYMENT[cell]["sides"][side]))
        O.apply_candidate(ds, term, side, sel["hp_rank"], sel["fix"], sel["regime"])
        for seed in C.SEEDS:
            p = os.path.join(ckpt_root, f"{term}_{ds}_{side}", f"seed_{seed}", "ckpt.pt")
            if os.path.exists(p):
                model, patch_len = O.build_option_model(ds, term, side, "detach_gg")
                model.load_state_dict(torch.load(p, map_location=device), strict=True)
                yield side, seed, model.to(device).eval(), patch_len
        C.DEPLOYMENT[cell]["sides"][side] = saved


MF.magnet_models = magnet_models

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", nargs="+", default=["FR"])
    ap.add_argument("--terms", nargs="+", default=["short", "long"]); ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    device = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    for term in a.terms:
        for ds in a.datasets:
            MF.figure_case_study(ds, term, device, os.path.join(ROOT, "checkpoints", "magnet_dlinear"),
                                 os.path.join(ROOT, "runs", "dlinear_alone"), True, os.path.join(ROOT, "figures"), None)
