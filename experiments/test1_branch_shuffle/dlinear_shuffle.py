#!/usr/bin/env python3
"""Test 1 for MAGNet+DLinear on the shipped checkpoints: per deployed side x seed, test MAE/MSE intact vs with the MAGNet
graph branch's exogenous input replaced by the exogenous input of a random OTHER test window (forward pre-hook on
model.magnet_graph, rng seeded by the training seed); the DLinear backbone keeps the true exogenous series.
    python3 experiments/test1_branch_shuffle/dlinear_shuffle.py [--gpu 0]   -> runs/test1/DLinear.json"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from common import REAL, SYN, SEEDS, RUNS, FINAL_CKPTS, C, O, get_dataloaders, predict  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, default=0); a = ap.parse_args()
dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
os.makedirs(os.path.join(RUNS, "test1"), exist_ok=True); OUT = os.path.join(RUNS, "test1", "DLinear.json")
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
for term in ("short", "long"):
    for ds in REAL + SYN:
        c = f"{ds}/{term}"; e = C.cell_config(ds, term); i = C.NAME_TO_ID[ds]; L, H = C.get_seq_horizon(i, term)
        for side in e["deployed_sides"]:
            sel = e["sides"][side]["selected"]; saved = json.loads(json.dumps(C.DEPLOYMENT[c]["sides"][side]))
            O.apply_candidate(ds, term, side, sel["hp_rank"], sel["fix"], sel["regime"])
            model, pl = O.build_option_model(ds, term, side, "detach_gg")
            te = get_dataloaders(npz_path=C.csv_path(i), sensor_id=0, seq_len=L, forecast_horizon=H, patch_len=pl, batch_size=64, dataset_id=i)[2]
            pool = torch.cat([b[0] for b in te])
            for sd in SEEDS:
                k = f"{c}|{side}|{sd}"; ck = os.path.join(FINAL_CKPTS, f"{term}_{ds}_{side}", f"seed_{sd}", "ckpt.pt")
                if k in R or not os.path.exists(ck):
                    continue
                model.load_state_dict(torch.load(ck, map_location=dev)); model.to(dev).eval()
                p0, t = predict(model, te, i, term, dev)
                rng = np.random.default_rng(sd)
                h = model.magnet_graph.register_forward_pre_hook(
                    lambda _m, x: (pool[torch.from_numpy(rng.integers(0, len(pool), size=x[0].shape[0]))].to(x[0].device),) + tuple(x[1:]))
                p1, _ = predict(model, te, i, term, dev); h.remove()
                p0, p1, t = map(np.asarray, (p0, p1, t))
                R[k] = {"intact_mae": float(np.abs(p0 - t).mean()), "shuffled_mae": float(np.abs(p1 - t).mean()),
                        "intact_mse": float(((p0 - t) ** 2).mean()), "shuffled_mse": float(((p1 - t) ** 2).mean())}
                json.dump(R, open(OUT, "w"), indent=1); print(k, round(R[k]["intact_mae"], 4), round(R[k]["shuffled_mae"], 4), flush=True)
            C.DEPLOYMENT[c]["sides"][side] = saved
