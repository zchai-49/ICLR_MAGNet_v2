#!/usr/bin/env python3
"""Vanilla DLinear for the rebuttal datasets, chosen on validation only (2026-11-12): grid moving_avg {13, 25, 51} x
lr {1e-3, 1e-4} with seed 0 (train/train_dlinear_alone.py training loop), best validation MAE wins, then 5 seeds.
Writes runs/dlinear_alone/<term>_<ds>/seed_<s>/ckpt.pt (+ grid/ and choice.json).
    MAGNET_VAL_LOOKBACK=1 python3 train/dlinear_alone_valsel.py --cells Syn20/short Weather/long --gpu 0"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import train_dlinear_alone as D  # noqa: E402
import torch  # noqa: E402
from MAGNET_data import get_dataloaders  # noqa: E402
from trainer import run_epoch  # noqa: E402

OUT = os.path.join(ROOT, "runs", "dlinear_alone")


def fit(ds, term, seed, ma, lr, gpu, path):
    if os.path.exists(path + ".json"):
        return json.load(open(path + ".json"))["val"]
    D.HPARAMS[f"{ds}/{term}"] = {"moving_avg": ma, "lr": lr}
    dev = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    i = D.C.NAME_TO_ID[ds]; torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    L, H = D.C.get_seq_horizon(i, term)
    m = D.build_dlinear_alone(ds, term).to(dev)
    tr, va, _ = get_dataloaders(npz_path=D.C.csv_path(i), sensor_id=0, seq_len=L, forecast_horizon=H, patch_len=12,
                                batch_size=32 if term == "long" else 64, dataset_id=i)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4 if term == "short" else 1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200, eta_min=1e-6)
    best, bad, state = float("inf"), 0, None
    for ep in range(200):
        run_epoch(m, tr, opt, dev, i, term, train=True); v = run_epoch(m, va, opt, dev, i, term, train=False); sch.step()
        if v < best:
            best, bad, state = v, 0, {k: x.detach().cpu().clone() for k, x in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 15:
                break
    os.makedirs(os.path.dirname(path), exist_ok=True); torch.save(state, path)
    json.dump({"val": best, "moving_avg": ma, "lr": lr, "seed": seed, "epochs": ep + 1}, open(path + ".json", "w"))
    return best


ap = argparse.ArgumentParser(); ap.add_argument("--cells", nargs="+", required=True); ap.add_argument("--gpu", type=int, default=0)
a = ap.parse_args()
for c in a.cells:
    ds, term = c.split("/")
    grid = {(ma, lr): fit(ds, term, 0, ma, lr, a.gpu, os.path.join(OUT, "grid", f"{term}_{ds}", f"ma{ma}_lr{lr}", "ckpt.pt"))
            for ma in (13, 25, 51) for lr in (1e-3, 1e-4)}
    ma, lr = min(grid, key=grid.get)
    json.dump({"moving_avg": ma, "lr": lr, "grid_val": {f"{k[0]}/{k[1]}": v for k, v in grid.items()}},
              open(os.path.join(OUT, "grid", f"{term}_{ds}", "choice.json"), "w"), indent=1)
    for sd in (0, 42, 123, 2024, 3407):
        fit(ds, term, sd, ma, lr, a.gpu, D.ckpt_path(OUT, ds, term, sd))
    print(f"{c}: chose moving_avg={ma} lr={lr} (val {grid[(ma, lr)]:.4f}); 5 seeds done", flush=True)
