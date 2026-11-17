#!/usr/bin/env python3
"""Test 1 summary (is the MAGNet signal used by each backbone?): per backbone, mean test MAE/MSE intact vs with the MAGNet
branch's exogenous input shuffled, over 12 real + 5 synthetic datasets (short, long, overall), and per data group.
Sources: runs/pnp/<bb>/<ds>/<term>/seed_<s>/result.json["shuffle"] (experiments/table2_pnp/run.sh, MAGNET_PNP_SHUFFLE=1) and
runs/test1/DLinear.json (dlinear_shuffle.py). Cells whose cross-fit gate is <= 0 have the branch switched off (dMAE = 0
exactly); diverged seeds are dropped.     python3 experiments/test1_branch_shuffle/summarize.py   -> results/test1_reproduced.md"""
import json, os, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REAL = ["BE", "DE", "FR", "NP", "PJM", "Energy", "Colbun", "Rapel", "Sdwpfh1", "Sdwpfh2", "Sdwpfm1", "Sdwpfm2"]
SYN = ["Syn5", "Syn20", "Syn50", "Syn20weak", "Syn20null"]; BB = ["DAG", "GCGNet", "TimeXer", "TimeKAN", "PatchTST", "TFT"]
G = json.load(open(os.path.join(ROOT, "configs", "granger_gain_crossfit.json")))
def cell(b, d, t):
    xs = []
    if b == "DLinear":
        R = json.load(open(os.path.join(ROOT, "runs", "test1", "DLinear.json")))
        xs = [(v["intact_mae"], v["shuffled_mae"], v["intact_mse"], v["shuffled_mse"]) for k, v in R.items() if k.startswith(f"{d}/{t}|")]
    else:
        for s in (0, 42, 123, 2024, 3407):
            f = os.path.join(ROOT, "runs", "pnp", b, d, t, f"seed_{s}", "result.json")
            if not os.path.exists(f): continue
            r = json.load(open(f)); sh = r.get("shuffle")
            if sh and np.isfinite(sh["intact_mae"]): xs.append((sh["intact_mae"], sh["shuffled_mae"], sh["intact_mse"], sh["shuffled_mse"]))
            elif G[f"{d}/{t}"]["ridge"]["gain"] <= 0:
                mae, mse = (r["test_metrics"]["mae"], r["test_metrics"]["mse"]) if "test_metrics" in r else (r["test_mae"], r["test_mse"])
                if np.isfinite(mae): xs.append((mae, mae, mse, mse))
    return np.mean(xs, 0) if xs else None
rows = []
for b in BB + ["DLinear"]:
    V = {t: [cell(b, d, t) for d in REAL + SYN] for t in ("short", "long")}
    for lab, ts in (("short", ("short",)), ("long", ("long",)), ("overall", ("short", "long"))):
        A = [x for t in ts for x in V[t]]
        if any(x is None for x in A): rows.append(f"| {b} | {lab} | incomplete |||||"); continue
        m = np.mean(A, 0)
        rows.append(f"| {b} | {lab} | {m[0]:.4f} | {m[1]:.4f} | {100 * (m[1] / m[0] - 1):+.2f}% | {m[2]:.4f} | {m[3]:.4f} | {100 * (m[3] / m[2] - 1):+.2f}% |")
md = "\n".join(["# Test 1 (MAGNet-branch exogenous shuffle), real + synthetic (17 datasets per term)", "",
                "| Backbone | Term | MAE intact | MAE shuffled | ΔMAE | MSE intact | MSE shuffled | ΔMSE |", "|---|---|---|---|---|---|---|---|"] + rows)
open(os.path.join(ROOT, "results", "test1_reproduced.md"), "w").write(md + "\n"); print(md)
