# hparam_MAGNET_GCGNet.py
# Training / evaluation code for MAGNet+GCGNet (hybrid mode 2, ngc_lag=1, history-only input).
# GCGNet's own hyperparameters are fixed per dataset (GCGNET_HPARAMS, from the GCGNet reference implementation);
# the only searched hyperparameter is MAGNet's graph width magnet_d_model in {32, 64, 128}. Each value is trained with
# 5 seeds and the winner is the value with the lowest MEAN validation MAE (`python3 pnp/hparam_MAGNET_GCGNet.py`).
# GCGNet's training objective uses the true future as a graph-teacher signal (training loader only,
# `return_future_exog=True`); validation and test use the observed history only.
import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)
for _d in ("models", "data", "backbones"):
    sys.path.insert(0, os.path.join(PKG_ROOT, _d))

from MAGNET_data import get_dataloaders
from MAGNET_with_GCGNet import MAGNETWithGCGNet

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

SEEDS = [0, 42, 123, 2024, 3407]
DMS = [32, 64, 128]                       # searched: MAGNet graph width
MAGNET_N_HEADS = {32: 2, 64: 4, 128: 8}
PATCH_LEN = 12
NGC_LAG = 1
NUM_EPOCHS, PATIENCE = 50, 5

DATASET_PATHS = {
    5:  (f"{PKG_ROOT}/data/dataset/EPF/BE.csv",        2, "BE"),
    6:  (f"{PKG_ROOT}/data/dataset/EPF/DE.csv",        2, "DE"),
    7:  (f"{PKG_ROOT}/data/dataset/EPF/FR.csv",        2, "FR"),
    8:  (f"{PKG_ROOT}/data/dataset/EPF/NP.csv",        2, "NP"),
    9:  (f"{PKG_ROOT}/data/dataset/EPF/PJM.csv",       2, "PJM"),
    10: (f"{PKG_ROOT}/data/dataset/energy/Energy.csv", 5, "Energy"),
    11: (f"{PKG_ROOT}/data/dataset/hydro/Colbun.csv",  2, "Colbun"),
    12: (f"{PKG_ROOT}/data/dataset/hydro/Rapel.csv",   2, "Rapel"),
    13: (f"{PKG_ROOT}/data/dataset/wind/Sdwpfh1.csv",  6, "Sdwpfh1"),
    14: (f"{PKG_ROOT}/data/dataset/wind/Sdwpfh2.csv",  6, "Sdwpfh2"),
    15: (f"{PKG_ROOT}/data/dataset/wind/Sdwpfm1.csv",  6, "Sdwpfm1"),
    16: (f"{PKG_ROOT}/data/dataset/wind/Sdwpfm2.csv",  6, "Sdwpfm2"),
    18: (f"{PKG_ROOT}/data/dataset/generic/Syn5.csv", 5, "Syn5"),
    19: (f"{PKG_ROOT}/data/dataset/generic/Syn20.csv", 20, "Syn20"),
    20: (f"{PKG_ROOT}/data/dataset/generic/Syn50.csv", 50, "Syn50"),
    21: (f"{PKG_ROOT}/data/dataset/generic/Syn20weak.csv", 20, "Syn20weak"),
    22: (f"{PKG_ROOT}/data/dataset/generic/Syn20null.csv", 20, "Syn20null"),   # synthetic (rebuttal 2026-11-15)
}
WIND_DATASETS = {13, 14, 15, 16}
DIFF_DATASETS = {11}
TTA_DATASETS = {8, 13, 14, 15, 16}        # test-time averaging over 5 passes (GCGNet's VAE is stochastic)
RESULTS_ROOT = Path(PKG_ROOT) / "runs" / "pnp_search" / "gcgnet"

GCGNET_HPARAMS = {
    "BE":      dict(batch_size=32, d_ff=512, d_model=128, dropout=0,   lr=1e-4, lradj="type3", n_heads=2, rank=4),
    "Colbun":  dict(batch_size=64, d_ff=256, d_model=512, dropout=0.1, lr=1e-3, lradj="type3", n_heads=2, rank=4),
    "DE":      dict(batch_size=32, d_ff=512, d_model=64,  dropout=0,   lr=1e-3, lradj="type3", n_heads=2, rank=4),
    "Energy":  dict(batch_size=64, d_ff=128, d_model=64,  dropout=0.1, lr=1e-3, lradj="type3", n_heads=8, rank=4),
    "FR":      dict(batch_size=32, d_ff=256, d_model=64,  dropout=0,   lr=1e-3, lradj="type3", n_heads=2, rank=4),
    "NP":      dict(batch_size=64, d_ff=256, d_model=128, dropout=0.1, lr=1e-3, lradj="type3", n_heads=2, rank=4),
    "PJM":     dict(batch_size=64, d_ff=512, d_model=256, dropout=0,   lr=1e-3, lradj="type3", n_heads=2, rank=4),
    "Rapel":   dict(batch_size=32, d_ff=256, d_model=64,  dropout=0.1, lr=1e-3, lradj="type3", n_heads=2, rank=4),
    "Sdwpfh1": dict(batch_size=32, d_ff=64,  d_model=64,  dropout=0.2, lr=1e-3, lradj="type3", n_heads=2, rank=4),
    "Sdwpfh2": dict(batch_size=64, d_ff=512, d_model=64,  dropout=0.2, lr=1e-3, lradj="type3", n_heads=2, rank=4),
    "Sdwpfm1": dict(batch_size=64, d_ff=64,  d_model=64,  dropout=0.2, lr=1e-4, lradj="type3", n_heads=4, rank=4),
    "Sdwpfm2": dict(batch_size=32, d_ff=256, d_model=512, dropout=0.2, lr=1e-4, lradj="type3", n_heads=2, rank=4),
}


def get_seq_horizon(dataset_id, term):
    if dataset_id in (11, 12):
        return (60, 10) if term == "short" else (180, 30)
    return (168, 24) if term == "short" else (720, 360)


def lr_lambda_factory(lradj):
    """GCGNet/DAG's adjust_learning_rate as a multiplicative LambdaLR factor (their epochs are 1-indexed)."""
    def fn(epoch):
        e = epoch + 1
        if lradj == "type1":
            return 0.5 ** ((e - 1) // 1)
        if lradj == "type3":
            return 1.0 if e < 3 else 0.9 ** ((e - 3) // 1)
        return 1.0
    return fn


def loss_for(dataset_id, term, pred, target, last_known):
    if dataset_id in DIFF_DATASETS and term == "long":
        pred = last_known.unsqueeze(1) + pred.cumsum(dim=-1)
        target = last_known.unsqueeze(1) + target.cumsum(dim=-1)
        return nn.functional.l1_loss(pred, target)
    if dataset_id in WIND_DATASETS:
        return ((1 + torch.abs(target)) * torch.abs(pred - target)).mean()
    if term == "long":
        return nn.functional.smooth_l1_loss(pred, target, beta=0.5)
    return nn.functional.l1_loss(pred, target)


def run_epoch(model, loader, optimizer, device, dataset_id, term, train):
    model.train() if train else model.eval()
    preds, targets = [], []
    with (torch.enable_grad() if train else torch.no_grad()):
        for batch in loader:
            if train:   # training loader carries the true future exogenous series (GCGNet's teacher signal)
                exo_p, endo_p, target, _m, _s, exo_future, time_f, last_known = batch
                exo_future = exo_future.to(device)
            else:
                exo_p, endo_p, target, _m, _s, time_f, last_known = batch
                exo_future = None
            exo_p, endo_p, target = exo_p.to(device), endo_p.to(device), target.to(device)
            last_known = last_known.to(device)
            out = model(exo_p, endo_p, exog_future=exo_future, endo_future=target if train else None)
            pred, aux = (out[0], out[1]) if isinstance(out, (tuple, list)) else (out, 0.0)
            loss = loss_for(dataset_id, term, pred, target, last_known) + aux
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5 if dataset_id == 13 else 1.0)
                optimizer.step()
            preds.append(pred.detach().cpu())
            targets.append(target.detach().cpu())
    return torch.mean(torch.abs(torch.cat(preds) - torch.cat(targets))).item()


def evaluate(model, test_loader, dataset_id, term, device):
    model.eval()
    runs, all_targets = [], None
    with torch.no_grad():
        for _ in range(5 if dataset_id in TTA_DATASETS else 1):
            rp, tg = [], []
            for exo_p, endo_p, target, _m, _s, _tf, last_known in test_loader:
                pred = model(exo_p.to(device), endo_p.to(device))
                if dataset_id in DIFF_DATASETS and term == "long":
                    pred = last_known.to(device).unsqueeze(1) + pred.cumsum(dim=-1)
                    target = last_known.unsqueeze(1) + target.cumsum(dim=-1)
                rp.append(pred.cpu()); tg.append(target.cpu())
            runs.append(torch.cat(rp))
            if all_targets is None:
                all_targets = torch.cat(tg)
    p = torch.stack(runs).mean(0)
    return {"mae": torch.mean(torch.abs(p - all_targets)).item(), "mse": torch.mean((p - all_targets) ** 2).item()}


def train_one_config(dataset_id, term, magnet_d_model, device, seed, out_dir, ngc_lag=NGC_LAG):
    """Train MAGNet+GCGNet for one (dataset, term, magnet_d_model, seed); returns (best_val_mae, result dict)."""
    csv_path, n_exo, ds_name = DATASET_PATHS[dataset_id]
    hp = GCGNET_HPARAMS[ds_name]
    seq_len, horizon = get_seq_horizon(dataset_id, term)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for attempt, divisor in enumerate([1, 2, 4, 8]):          # halve the batch size on out-of-memory
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
        bs = max(1, hp["batch_size"] // divisor)
        kw = dict(npz_path=csv_path, sensor_id=0, seq_len=seq_len, forecast_horizon=horizon, patch_len=PATCH_LEN,
                  batch_size=bs, dataset_id=dataset_id)
        train_loader = get_dataloaders(**kw, return_future_exog=True)[0]
        _, val_loader, test_loader = get_dataloaders(**kw)

        class Cfg: pass
        cfg = Cfg()
        cfg.seq_len, cfg.pred_len, cfg.patch_len = seq_len, horizon, horizon    # GCGNet: patch_len == horizon
        cfg.enc_in, cfg.series_dim = n_exo + 1, 1
        cfg.d_model, cfg.d_ff, cfg.rank, cfg.e_layers = hp["d_model"], hp["d_ff"], hp["rank"], 1
        cfg.n_heads, cfg.dropout = hp["n_heads"], hp["dropout"]
        cfg.use_norm, cfg.use_future_exog = True, False
        magnet_config = {"n_exo": n_exo, "patch_len": PATCH_LEN, "num_patches": seq_len // PATCH_LEN,
                         "d_model": magnet_d_model, "n_heads": MAGNET_N_HEADS[magnet_d_model],
                         "ngc_hidden_dim": 16 if (dataset_id == 11 and term == "long") else 32,
                         "dropout": 0.1 if term == "short" else 0.2, "dataset": dataset_id, "term": term,
                         "ngc_lag": ngc_lag}
        model = MAGNETWithGCGNet(magnet_config, cfg, mode=2).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(hp["lradj"]))
        best_val, bad, best_state, epoch, t0 = float("inf"), 0, None, 0, time.time()
        try:
            for epoch in range(1, NUM_EPOCHS + 1):
                run_epoch(model, train_loader, opt, device, dataset_id, term, train=True)
                val = run_epoch(model, val_loader, opt, device, dataset_id, term, train=False)
                sched.step()
                if val < best_val:
                    best_val, bad = val, 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= PATIENCE:
                        break
        except torch.cuda.OutOfMemoryError:
            del model; torch.cuda.empty_cache(); gc.collect()
            if attempt == 3:
                return best_val, None
            continue
        break

    if best_state is None:          # 2026-11-15: validation never finite (diverged, e.g. NaN loss) -> record as failed
        (out_dir / "result.json").write_text(json.dumps({"dataset": ds_name, "term": term, "magnet_d_model": magnet_d_model,
            "seed": seed, "gcgnet_hparams": hp, "diverged": True, "best_val_mae": float("nan"),
            "test_metrics": {"mae": float("nan"), "mse": float("nan")}}, indent=2))
        del model; torch.cuda.empty_cache(); gc.collect()
        return best_val, None
    model.load_state_dict(best_state)
    m = evaluate(model, test_loader, dataset_id, term, device)
    shuf = None
    if os.environ.get("MAGNET_PNP_SHUFFLE") == "1":   # Test 1 (branch shuffle), experiments/test1_branch_shuffle, same as scripts/branch_shuffle_gcgnet.py
        pool = torch.cat([b[0] for b in test_loader]); model.eval()
        torch.manual_seed(0); np.random.seed(0); m0 = evaluate(model, test_loader, dataset_id, term, device)
        rng = np.random.default_rng(seed)
        h = model.magnet_graph.register_forward_pre_hook(
            lambda _m, a: (pool[torch.from_numpy(rng.integers(0, len(pool), size=a[0].shape[0]))].to(a[0].device),) + tuple(a[1:]))
        torch.manual_seed(0); np.random.seed(0); m1 = evaluate(model, test_loader, dataset_id, term, device); h.remove()
        shuf = {"intact_mae": m0["mae"], "shuffled_mae": m1["mae"], "intact_mse": m0["mse"], "shuffled_mse": m1["mse"]}
    res = {"dataset": ds_name, "term": term, "magnet_d_model": magnet_d_model, "seed": seed, "gcgnet_hparams": hp,
           "batch_size_used": bs, "best_val_mae": best_val, "stopped_at_epoch": epoch, "test_metrics": m,
           "elapsed_s": round(time.time() - t0), "ngc_lag": ngc_lag}
    if shuf is not None:
        res["shuffle"] = shuf
    if os.environ.get("MAGNET_SAVE_CKPT") == "1":         # rebuttal 2026-11-12: keep weights for the branch-shuffle test
        torch.save(best_state, out_dir / "ckpt.pt")
    (out_dir / "result.json").write_text(json.dumps(res, indent=2))
    del model; torch.cuda.empty_cache(); gc.collect()
    return best_val, res


def main():
    ap = argparse.ArgumentParser(description="5-seed search of magnet_d_model for MAGNet+GCGNet on one dataset/term")
    ap.add_argument("--dataset", required=True, choices=[v[2] for v in DATASET_PATHS.values()])
    ap.add_argument("--term", required=True, choices=["short", "long"])
    ap.add_argument("--dms", type=int, nargs="+", default=DMS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_id = {v[2]: k for k, v in DATASET_PATHS.items()}[a.dataset]
    means = {}
    for dm in a.dms:
        vals = []
        for s in a.seeds:
            d = RESULTS_ROOT / a.term / a.dataset / f"mdm{dm}" / f"seed_{s}"
            if (d / "result.json").exists():
                r = json.loads((d / "result.json").read_text())
                v = r["best_val_mae"]
            else:
                v, r = train_one_config(dataset_id, a.term, dm, device, s, d)
                if r is None:
                    continue
            vals.append(v)
            print(f"{a.dataset}/{a.term} magnet_d_model={dm} seed={s}: val MAE {v:.4f}", flush=True)
        if len(vals) == len(a.seeds):
            means[dm] = statistics.mean(vals)
    if means:
        win = min(means, key=means.get)
        print(f"winner magnet_d_model={win} (mean val MAE {means[win]:.4f}); all: {means}")


if __name__ == "__main__":
    main()
