#!/usr/bin/env python3
"""Evaluate the official MAGNet+DLinear checkpoints (final configuration, models/final_defaults.py) on the test split.

For every requested (dataset, term) cell: predictions of every available seed of each deployed side are averaged per side,
then over the deployed sides (the reported number is this ensemble over seeds 0, 42, 123, 2024, 3407). Cells with only
some seeds present are evaluated on those seeds and marked "partial" next to the reported 5-seed number
(results/table1_main.json).

    python3 scripts/evaluate_magnet.py                                  # every cell that has checkpoints
    python3 scripts/evaluate_magnet.py --datasets FR Syn20 --terms short --gpu 0
-> results/evaluation.json"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("configs", "train", "scripts", "data", "models"):
    sys.path.insert(0, os.path.join(ROOT, d))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import lag1_config as C  # noqa: E402
import train_options as O  # noqa: E402
from MAGNET_data import get_dataloaders  # noqa: E402
from evaluate import predict  # noqa: E402  (history-only prediction + test-time averaging, shared with training)

REAL = ["BE", "DE", "FR", "NP", "PJM", "Energy", "Colbun", "Rapel", "Sdwpfh1", "Sdwpfh2", "Sdwpfm1", "Sdwpfm2"]
SYN = ["Syn5", "Syn20", "Syn50", "Syn20weak", "Syn20null"]




def evaluate_cell(ds, term, device, ckpt_root, seeds):
    c = f"{ds}/{term}"; e = C.cell_config(ds, term); i = C.NAME_TO_ID[ds]; L, H = C.get_seq_horizon(i, term)
    side_preds, t, used = [], None, {}
    for side in e["deployed_sides"]:
        sel = e["sides"][side]["selected"]; saved = json.loads(json.dumps(C.DEPLOYMENT[c]["sides"][side]))
        O.apply_candidate(ds, term, side, sel["hp_rank"], sel["fix"], sel["regime"])
        P = []
        for sd in seeds:
            ck = os.path.join(ckpt_root, f"{term}_{ds}_{side}", f"seed_{sd}", "ckpt.pt")
            if not os.path.exists(ck):
                continue
            model, pl = O.build_option_model(ds, term, side, "detach_gg")
            model.load_state_dict(torch.load(ck, map_location=device), strict=True)
            te = get_dataloaders(npz_path=C.csv_path(i), sensor_id=0, seq_len=L, forecast_horizon=H, patch_len=pl,
                                 batch_size=64, dataset_id=i)[2]
            with torch.no_grad():
                p, t = predict(model.to(device).eval(), te, i, term, device)
            P.append(np.asarray(p)); used.setdefault(side, []).append(sd)
        C.DEPLOYMENT[c]["sides"][side] = saved
        if not P:
            return None
        side_preds.append(np.mean(P, 0))
    err = np.mean(side_preds, 0) - np.asarray(t)
    return {"mae": float(np.abs(err).mean()), "mse": float((err ** 2).mean()), "seeds": used}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=REAL + SYN)
    ap.add_argument("--terms", nargs="+", default=["short", "long"])
    ap.add_argument("--seeds", type=int, nargs="+", default=C.SEEDS)
    ap.add_argument("--ckpt-root", default=os.path.join(ROOT, "checkpoints", "magnet_dlinear"))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "evaluation.json"))
    a = ap.parse_args()
    device = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    ref = json.load(open(os.path.join(ROOT, "results", "table1_main.json")))
    out = {}
    for term in a.terms:
        for ds in a.datasets:
            r = evaluate_cell(ds, term, device, a.ckpt_root, a.seeds)
            c = f"{ds}/{term}"
            if r is None:
                print(f"{c}: no checkpoints in {a.ckpt_root} - skipped"); continue
            n = min(len(v) for v in r["seeds"].values()); r["status"] = "full" if n == 5 else f"partial ({n}/5 seeds)"
            rep = ref.get(c, {}).get("MAGNet"); r["reported_5seed"] = rep
            out[c] = r
            print(f"{c}: MAE {r['mae']:.4f} MSE {r['mse']:.4f} [{r['status']}]"
                  + (f"   reported (5-seed ensemble) MAE {rep[0]:.4f} MSE {rep[1]:.4f}" if rep else ""), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
