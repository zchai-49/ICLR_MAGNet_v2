#!/usr/bin/env python3
"""Table 2: vanilla backbone vs MAGNet+backbone, test MAE/MSE per dataset x term and averages (real, synthetic, all 17).
+MAGNet = 5-seed mean of runs/pnp (seeds that diverged to NaN are dropped and counted); vanilla backbones and DLinear(+MAGNet)
are the Table 1 numbers (results/table1_main.json; vanilla baselines are trained with their official code, not shipped).
    python3 experiments/table2_pnp/build_table.py [--runs runs/pnp]     -> results/table2_reproduced.md"""
import argparse, json, os, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REAL = ["BE", "DE", "FR", "NP", "PJM", "Energy", "Colbun", "Rapel", "Sdwpfh1", "Sdwpfh2", "Sdwpfm1", "Sdwpfm2"]
SYN = ["Syn5", "Syn20", "Syn50", "Syn20weak", "Syn20null"]; BB = ["DAG", "GCGNet", "TimeXer", "TimeKAN", "DLinear", "PatchTST", "TFT"]
ap = argparse.ArgumentParser(); ap.add_argument("--runs", default=os.path.join(ROOT, "runs", "pnp")); a = ap.parse_args()
T1 = json.load(open(os.path.join(ROOT, "results", "table1_main.json")))
def mag(b, d, t):
    if b == "DLinear": return T1[f"{d}/{t}"]["MAGNet"], 5
    m = []
    for s in (0, 42, 123, 2024, 3407):
        f = os.path.join(a.runs, b, d, t, f"seed_{s}", "result.json")
        if os.path.exists(f):
            r = json.load(open(f)); x = (r["test_metrics"]["mae"], r["test_metrics"]["mse"]) if "test_metrics" in r else (r["test_mae"], r["test_mse"])
            if np.isfinite(x[0]): m.append(x)
    return (list(np.mean(m, 0)), len(m)) if m else (None, 0)
V = {(b, d, t): (T1[f"{d}/{t}"][b], *mag(b, d, t)) for b in BB for d in REAL + SYN for t in ("short", "long")}
md = []
for k, name in ((0, "MAE"), (1, "MSE")):
    md += [f"## Test {name}", "", "| Dataset | Term | " + " | ".join(f"{b} | {b}+MAGNet" for b in BB) + " |", "|---|---" + "|---" * (2 * len(BB)) + "|"]
    for d in REAL + SYN:
        for t in ("short", "long"):
            cells = []
            for b in BB:
                v, m, n = V[(b, d, t)]
                cells += [f"{v[k]:.4f}", "—" if m is None else f"{m[k]:.4f}" + (f" ({n}/5)" if n < 5 else "")]
            md.append(f"| {d} | {t} | " + " | ".join(cells) + " |")
    for grp, DS in (("real", REAL), ("synthetic", SYN), ("all", REAL + SYN)):
        for lab, ts in (("short", ("short",)), ("long", ("long",)), ("short+long", ("short", "long"))):
            cells = []
            for b in BB:
                vs = [V[(b, d, t)] for d in DS for t in ts]
                cells += [f"{np.mean([x[0][k] for x in vs]):.4f}",
                          "—" if any(x[1] is None for x in vs) else f"{np.mean([x[1][k] for x in vs]):.4f}"]
            md.append(f"| **{grp} avg** | {lab} | " + " | ".join(cells) + " |")
    md.append("")
open(os.path.join(ROOT, "results", "table2_reproduced.md"), "w").write("\n".join(md)); print("\n".join(md))
