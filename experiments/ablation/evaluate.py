#!/usr/bin/env python3
"""Ablation table: test MAE/MSE of each column (5-seed ensemble over the deployed sides) per dataset x term, plus averages
over real (12), synthetic (5) and all 17 datasets (short, long, short+long). Columns without checkpoints are skipped.
    python3 experiments/ablation/evaluate.py [--gpu 0]      -> results/ablation_reproduced.{json,md}"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from common import ROOT, REAL, SYN, RUNS, FINAL_CKPTS, ensemble, md_table  # noqa: E402

COLS = [("Full", "detach_gg"), ("w/o Text", "no_text"), ("w/o Exo", "no_exo"), ("w/o Graph", "dlinear_only"),
        ("w/ Exo MLP", "exo_mlp"), ("w/ Cross-attn", "xattn_dlinear")]
ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, default=0); a = ap.parse_args()
dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
R = {}
for term in ("short", "long"):
    for ds in REAL + SYN:
        for name, opt in COLS:
            root = FINAL_CKPTS if opt == "detach_gg" else os.path.join(RUNS, "ablation", opt)
            env = {"MAGNET_SYN_TEXT": "0"} if (ds in SYN and opt != "detach_gg") else None
            r = ensemble(ds, term, opt, root, device=dev, env=env)
            if r:
                R.setdefault(f"{ds}/{term}", {})[name] = r[:2]; print(ds, term, name, [round(x, 4) for x in r[:2]], flush=True)
json.dump(R, open(os.path.join(ROOT, "results", "ablation_reproduced.json"), "w"), indent=1)
names = [n for n, _ in COLS]; md = []
for k, metric in ((0, "MAE"), (1, "MSE")):
    rows = [[c] + [f"{R[c][n][k]:.4f}" if n in R[c] else "—" for n in names] for c in R]
    for grp, DS in (("real", REAL), ("synthetic", SYN), ("all", REAL + SYN)):
        for lab, ts in (("short", ("short",)), ("long", ("long",)), ("short+long", ("short", "long"))):
            cells = [f"{d}/{t}" for d in DS for t in ts]
            rows.append([f"**{grp} avg {lab}**"] + [f"{np.mean([R[c][n][k] for c in cells]):.4f}" if all(n in R.get(c, {}) for c in cells) else "—" for n in names])
    md += [f"## Test {metric}", "", md_table(["Cell"] + names, rows), ""]
open(os.path.join(ROOT, "results", "ablation_reproduced.md"), "w").write("\n".join(md)); print("\n".join(md))
