#!/usr/bin/env python3
"""Train DLinear alone (no MAGNet; mode 3) for the case-study comparison in visualization/make_figures.py.

    python3 train/train_dlinear_alone.py --dataset Energy --term short --seeds 2024 0 --gpu 0

Hyperparameters (moving average, learning rate) are in configs/dlinear_alone_hparams.json. AdamW, cosine schedule, up to
200 epochs with patience 15, batch size 64 (short) / 32 (long). Checkpoints go to results/dlinear_alone/<term>_<dataset>/seed_<seed>/.
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "configs"))
sys.path.insert(0, os.path.join(ROOT, "train"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import lag1_config as C  # noqa: E402
from MAGNET_data import get_dataloaders  # noqa: E402
from MAGNET_with_DLinear_fusion import MAGNETWithDLinear  # noqa: E402
from trainer import run_epoch  # noqa: E402

HPARAMS = json.load(open(os.path.join(ROOT, "configs", "dlinear_alone_hparams.json")))


def build_dlinear_alone(dataset, term):
    dataset_id = C.NAME_TO_ID[dataset]
    seq_len, horizon = C.get_seq_horizon(dataset_id, term)
    cfg = {"seq_len": seq_len, "pred_len": horizon, "moving_avg": HPARAMS[f"{dataset}/{term}"]["moving_avg"]}
    return MAGNETWithDLinear({}, cfg, mode=3, use_instance_norm=False)


def ckpt_path(root, dataset, term, seed):
    return os.path.join(root, f"{term}_{dataset}", f"seed_{seed}", "ckpt.pt")


def train_alone(dataset, term, seed, gpu=0, epochs=200, out_root=None):
    out_root = out_root or os.path.join(ROOT, "runs", "dlinear_alone")
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    dataset_id = C.NAME_TO_ID[dataset]
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
    seq_len, horizon = C.get_seq_horizon(dataset_id, term)
    model = build_dlinear_alone(dataset, term).to(device)
    train_l, val_l, _ = get_dataloaders(npz_path=C.csv_path(dataset_id), sensor_id=0, seq_len=seq_len,
                                        forecast_horizon=horizon, patch_len=12,
                                        batch_size=32 if term == "long" else 64, dataset_id=dataset_id)
    lr = HPARAMS[f"{dataset}/{term}"]["lr"]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4 if term == "short" else 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    best, bad, state, t0 = float("inf"), 0, None, time.time()
    for epoch in range(1, epochs + 1):
        run_epoch(model, train_l, opt, device, dataset_id, term, train=True)
        val = run_epoch(model, val_l, opt, device, dataset_id, term, train=False)
        sched.step()
        if val < best:
            best, bad, state = val, 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 15:
                break
    path = ckpt_path(out_root, dataset, term, seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"DLinear-alone {dataset}/{term}/seed_{seed}: val MAE {best:.4f} ({epoch} epochs, {time.time()-t0:.0f}s) -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(C.NAME_TO_ID))
    ap.add_argument("--term", required=True, choices=["short", "long"])
    ap.add_argument("--seeds", type=int, nargs="+", default=C.SEEDS)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--out-root", default=None)
    a = ap.parse_args()
    root = a.out_root or os.path.join(ROOT, "results", "dlinear_alone")
    for seed in a.seeds:
        if os.path.exists(ckpt_path(root, a.dataset, a.term, seed)):
            print(f"[skip] seed {seed} exists")
            continue
        train_alone(a.dataset, a.term, seed, a.gpu, a.epochs, root)


if __name__ == "__main__":
    main()
