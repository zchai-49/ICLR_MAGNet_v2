#!/usr/bin/env python3
"""Text ablation tables (encoder / generator / prompt), test MAE and MSE of the ensemble over the deployed sides x seeds {0, 42}
for NP, Energy, Rapel, Sdwpfm1, Syn20 (short, long), with an average row. Bold = best text variant (w/o Text excluded).
    python3 experiments/text_ablation/evaluate.py [--gpu 0]    -> results/text_ablation_reproduced.md"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from common import ROOT, RUNS, FINAL_CKPTS, ensemble, md_table  # noqa: E402

CELLS = [(d, t) for d in ("NP", "Energy", "Rapel", "Sdwpfm1", "Syn20") for t in ("short", "long")]
TABLES = {"Text encoder (same Llama-3.2-3B text)": [("Deployed (BERT)", "detach_gg"), ("BERT re-enc.", "text_encbert"), ("MiniLM", "text_encminilm"),
                                                      ("RoBERTa", "text_encroberta"), ("MPNet", "text_encmpnet"), ("BGE", "text_encbge")],
          "Text generator (BERT fixed)": [("Llama-3.2-1B", "text_llama1b"), ("Llama-3.2-3B (deployed)", "detach_gg"), ("Llama-3.1-8B", "text_llama8b"),
                                          ("Qwen2.5-3B", "text_qwen"), ("Qwen2.5-7B", "text_qwen7b"), ("DeepSeek-7B", "text_deepseek")],
          "Text prompt (Llama-3.2-3B + BERT fixed)": [("Ours", "detach_gg"), ("P1 simple", "text_psimple"), ("P2 no causal Qs", "text_pnocausal"),
                                                      ("P3 no stats", "text_pnostats"), ("P4 stats only", "text_pstats"), ("P5 name+desc", "text_pnamedesc"),
                                                      ("w/o Text", "no_text")]}
ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, default=0); a = ap.parse_args()
dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
cache = {}
def get(opt, d, t):
    if (opt, d, t) not in cache:
        if opt == "detach_gg" or (opt == "text_encbert" and d == "Syn20"):
            root, o = FINAL_CKPTS, "detach_gg"
        else:
            root, o = os.path.join(RUNS, "ablation" if opt == "no_text" else "text", opt), opt
        r = ensemble(d, t, o, root, seeds=(0, 42), device=dev)
        cache[(opt, d, t)] = r[:2] if r else None
    return cache[(opt, d, t)]
md = []
for title, cols in TABLES.items():
    for k, metric in ((0, "MAE"), (1, "MSE")):
        rows, allv = [], {n: [] for n, _ in cols}
        for d, t in CELLS:
            v = {n: get(o, d, t) for n, o in cols}
            txt = [n for n in v if v[n] and n != "w/o Text"]; best = min(txt, key=lambda n: v[n][k]) if txt else None
            rows.append([f"{d}/{t}"] + ["—" if v[n] is None else (f"**{v[n][k]:.4f}**" if n == best else f"{v[n][k]:.4f}") for n, _ in cols])
            for n in v: allv[n].append(np.nan if v[n] is None else v[n][k])
        rows.append(["**Average**"] + [f"{np.mean(allv[n]):.4f}" for n, _ in cols])
        md += [f"### {title} — test {metric}", "", md_table(["Cell"] + [n for n, _ in cols], rows), ""]
open(os.path.join(ROOT, "results", "text_ablation_reproduced.md"), "w").write("\n".join(md)); print("\n".join(md))
