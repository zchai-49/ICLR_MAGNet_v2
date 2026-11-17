# hparam_MAGNET_TimeKAN.py
# Search space and training code (train_one_config) for MAGNet+TimeKAN as a plug-and-play module
# (hybrid mode 2, ngc_lag=1, history-only input).
import torch
import torch.nn as nn
import gc
import numpy as np
import json
import time
import math
import argparse
import itertools
from pathlib import Path
from datetime import datetime

import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)
for _d in ("models", "data", "backbones"):
    sys.path.insert(0, os.path.join(PKG_ROOT, _d))

from MAGNET_data import get_dataloaders, REVIN_DATASETS
from MAGNET_with_TimeKAN import MAGNETWithTimeKAN

# ── Search space ──────────────────────────────────────────────────────────────

SEARCH_SPACE = {
    'tk_d_model' : [32, 64, 128],
    'tk_e_layers': [1, 2],
    'lr'         : [1e-4, 5e-4],
}

# MAGNET graph d_model per dataset — same as standalone MAGNET configs
MAGNET_HPARAMS = {
    (5,  'short'): (256, 4), (5,  'long'): (64,  4),
    (6,  'short'): (64,  4), (6,  'long'): (64,  8),
    (7,  'short'): (64,  4), (7,  'long'): (256, 4),
    (8,  'short'): (64,  8), (8,  'long'): (64,  4),
    (9,  'short'): (64,  4), (9,  'long'): (128, 8),
    (10, 'short'): (64,  4), (10, 'long'): (64,  4),
    (11, 'short'): (256, 4), (11, 'long'): (32,  2),
    (12, 'short'): (64,  4), (12, 'long'): (64,  4),
    (13, 'short'): (128, 8), (13, 'long'): (64,  4),
    (14, 'short'): (64,  4), (14, 'long'): (256, 8),
    (15, 'short'): (64,  4), (15, 'long'): (64,  4),
    (16, 'short'): (128, 4), (16, 'long'): (256, 4),
}

# ── Dataset registry ──────────────────────────────────────────────────────────

DATASET_PATHS = {
    5:  (f"{PKG_ROOT}/data/dataset/EPF/BE.csv",          2, "BE",      "/BE"),
    6:  (f"{PKG_ROOT}/data/dataset/EPF/DE.csv",          2, "DE",      "/DE"),
    7:  (f"{PKG_ROOT}/data/dataset/EPF/FR.csv",          2, "FR",      "/FR"),
    8:  (f"{PKG_ROOT}/data/dataset/EPF/NP.csv",          2, "NP",      "/NP"),
    9:  (f"{PKG_ROOT}/data/dataset/EPF/PJM.csv",         2, "PJM",     "/PJM"),
    10: (f"{PKG_ROOT}/data/dataset/energy/Energy.csv",   5, "Energy",  "/energy"),
    11: (f"{PKG_ROOT}/data/dataset/hydro/Colbun.csv",    2, "Colbun",  "/Colbun"),
    12: (f"{PKG_ROOT}/data/dataset/hydro/Rapel.csv",     2, "Rapel",   "/Rapel"),
    13: (f"{PKG_ROOT}/data/dataset/wind/Sdwpfh1.csv",    6, "Sdwpfh1", "/Sdwpfh1"),
    14: (f"{PKG_ROOT}/data/dataset/wind/Sdwpfh2.csv",    6, "Sdwpfh2", "/Sdwpfh2"),
    15: (f"{PKG_ROOT}/data/dataset/wind/Sdwpfm1.csv",    6, "Sdwpfm1", "/Sdwpfm1"),
    16: (f"{PKG_ROOT}/data/dataset/wind/Sdwpfm2.csv",    6, "Sdwpfm2", "/Sdwpfm2"),
    18: (f"{PKG_ROOT}/data/dataset/generic/Syn5.csv", 5, "Syn5", "/Syn5"),
    19: (f"{PKG_ROOT}/data/dataset/generic/Syn20.csv", 20, "Syn20", "/Syn20"),
    20: (f"{PKG_ROOT}/data/dataset/generic/Syn50.csv", 50, "Syn50", "/Syn50"),
    21: (f"{PKG_ROOT}/data/dataset/generic/Syn20weak.csv", 20, "Syn20weak", "/Syn20weak"),
    22: (f"{PKG_ROOT}/data/dataset/generic/Syn20null.csv", 20, "Syn20null", "/Syn20null"),   # synthetic (rebuttal 2026-11-15)
}

WIND_DATASETS = {13, 14, 15, 16}
DIFF_DATASETS = {11}

RESULTS_ROOT = Path(PKG_ROOT) / "runs" / "pnp_search" / "timekan"

# ── Metrics ───────────────────────────────────────────────────────────────────

def mae(pred, target):
    return torch.mean(torch.abs(pred - target)).item()

def rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()

# ── Single training run ───────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device, dataset, train=True, term="short"):
    model.train() if train else model.eval()
    all_preds, all_targets = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for exo_patches, endo_patches, target, mean, std, time_features, last_known in loader:
            exo_patches   = exo_patches.to(device)
            endo_patches  = endo_patches.to(device)
            target        = target.to(device)
            time_features = time_features.to(device)
            last_known    = last_known.to(device)

            pred = model(exo_patches, endo_patches,
                         time_features=time_features)
            if isinstance(pred, (tuple, list)):
                pred = pred[0]
            if len(pred.shape) == 3:
                pred = pred.squeeze(-1)

            if dataset in DIFF_DATASETS and term == 'long':
                pred   = last_known.unsqueeze(1) + pred.cumsum(dim=-1)
                target = last_known.unsqueeze(1) + target.cumsum(dim=-1)
                loss   = nn.functional.l1_loss(pred, target)
            else:
                if term == "long":
                    if dataset in WIND_DATASETS:
                        weights = 1 + torch.abs(target)
                        loss = (weights * torch.abs(pred - target)).mean()
                    else:
                        loss = nn.functional.smooth_l1_loss(pred, target, beta=0.5)
                else:
                    if dataset in WIND_DATASETS:
                        weights = 1 + torch.abs(target)
                        loss = (weights * torch.abs(pred - target)).mean()
                    else:
                        loss = nn.functional.l1_loss(pred, target)

            if train:
                optimizer.zero_grad()
                loss.backward()
                clip = 0.5 if dataset == 13 else 1.0
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
                optimizer.step()

            all_preds.append(pred.detach().cpu())
            all_targets.append(target.detach().cpu())

    all_preds   = torch.cat(all_preds,   dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    return {
        'mae': mae(all_preds, all_targets),
        'mse': nn.functional.mse_loss(all_preds, all_targets).item(),
    }


def train_one_config(
    dataset_id, term, hparams, device,
    train_loader, val_loader,
    n_exo, seq_len, horizon, patch_len,
    save_dir,
    seed=2024, ngc_lag=1,
):
    """Train a single hparam config. Returns best val MAE and config dict."""

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    d_model_magnet, n_heads_magnet = MAGNET_HPARAMS.get((dataset_id, term), (64, 4))

    magnet_config = {
        'n_exo'         : n_exo,
        'patch_len'     : patch_len,
        'd_model'       : d_model_magnet,
        'n_heads'       : n_heads_magnet,
        'ngc_hidden_dim': 16 if (dataset_id == 11 and term == 'long') else 32,
        'dropout'       : 0.1 if term == 'short' else 0.3,
        'dataset'       : dataset_id,
        'term'          : term,
        'ngc_lag'       : ngc_lag,
    }

    timekan_config = {
        'seq_len'  : seq_len,
        'pred_len' : horizon,
        'patch_len': patch_len,
        'd_model'  : hparams['tk_d_model'],
        'e_layers' : hparams['tk_e_layers'],
        'down_sampling_window': 2,
        'down_sampling_layers': 2,
        'dropout'  : 0.1,
    }

    try:
        model = MAGNETWithTimeKAN(magnet_config, timekan_config, mode=2).to(device)
    except Exception as e:
        return float('inf'), None

    total_params = sum(p.numel() for p in model.parameters())

    # LR warmup datasets (same as standalone MAGNET)
    warmup_short = {7, 8, 10, 12, 13}
    warmup_long  = {5, 8, 10, 12, 13}
    use_warmup   = (
        (term == "short" and dataset_id in warmup_short) or
        (term == "long"  and dataset_id in warmup_long)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hparams['lr'],
        weight_decay=1e-4 if term == 'short' else 1e-3,
    )

    epochs   = 200
    patience = 15

    if use_warmup:
        warmup_epochs = 5
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress)) * \
                   (1 - 1e-6 / hparams['lr']) + 1e-6 / hparams['lr']
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )

    config_str = (f"dm{hparams['tk_d_model']}_"
                  f"el{hparams['tk_e_layers']}_"
                  f"lr{hparams['lr']}")
    resume_path = save_dir / f"resume_{config_str}.pt"

    best_val_mae   = float('inf')
    patience_count = 0
    best_state     = None
    start_epoch    = 1

    if resume_path.exists():
        rck = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(rck['model_state'])
        optimizer.load_state_dict(rck['optimizer_state'])
        scheduler.load_state_dict(rck['scheduler_state'])
        best_val_mae   = rck['best_val_mae']
        patience_count = rck['patience_count']
        best_state     = rck['best_state']
        start_epoch    = rck['epoch'] + 1
        torch.set_rng_state(rck['rng_state'])
        torch.cuda.set_rng_state_all(rck['cuda_rng_state'])
        np.random.set_state(rck['np_rng_state'])
        print(f"      resuming {config_str} from epoch {start_epoch}", flush=True)

    try:
        for epoch in range(start_epoch, epochs + 1):
            run_epoch(model, train_loader, optimizer, device,
                      dataset_id, train=True, term=term)
            val_m = run_epoch(model, val_loader, optimizer, device,
                              dataset_id, train=False, term=term)
            scheduler.step()

            if val_m['mae'] < best_val_mae:
                best_val_mae   = val_m['mae']
                patience_count = 0
                best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_count += 1

            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_val_mae': best_val_mae,
                'patience_count': patience_count,
                'best_state': best_state,
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'np_rng_state': np.random.get_state(),
            }, resume_path)

            if patience_count >= patience:
                break

    except torch.cuda.OutOfMemoryError:
        print(f"      OOM — skipping config")
        del model
        torch.cuda.empty_cache()
        gc.collect()
        resume_path.unlink(missing_ok=True)
        return float('inf'), None
    except Exception as e:
        print(f"      Error: {e}")
        del model
        torch.cuda.empty_cache()
        gc.collect()
        resume_path.unlink(missing_ok=True)
        return float('inf'), None

    resume_path.unlink(missing_ok=True)

    # Save best checkpoint for this config
    ckpt_path = save_dir / f"ckpt_{config_str}.pt"
    torch.save(best_state, ckpt_path)

    result = {
        'hparams'      : hparams,
        'magnet_config': magnet_config,
        'tk_config'    : timekan_config,
        'total_params' : total_params,
        'best_val_mae' : best_val_mae,
        'ckpt_path'    : str(ckpt_path),
    }

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return best_val_mae, result


# ── Main search loop ──────────────────────────────────────────────────────────

def search(dataset_id, term, device):
    if dataset_id not in DATASET_PATHS:
        print(f"Unknown dataset: {dataset_id}")
        return

    npz_path, n_exo, ds_name, _ = DATASET_PATHS[dataset_id]

    # Sequence and horizon
    if dataset_id in {11, 12}:
        seq_len  = 60  if term == 'short' else 180
        horizon  = 10  if term == 'short' else 30
    else:
        seq_len  = 168 if term == 'short' else 720
        horizon  = 24  if term == 'short' else 360

    patch_len  = 12
    batch_size = 64  # fixed for search speed

    # Save dir
    save_dir = RESULTS_ROOT / ds_name / term
    save_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already done
    best_cfg_path = save_dir / "best_config.json"
    if best_cfg_path.exists():
        print(f"[{ds_name} {term}] Already searched — skipping.")
        return

    print(f"\n{'='*60}")
    print(f"[{ds_name}] Term={term} | seq={seq_len} | horizon={horizon}")
    print(f"{'='*60}")

    # Dataloaders
    try:
        train_loader, val_loader, _ = get_dataloaders(
            npz_path         = npz_path,
            sensor_id        = 0,
            seq_len          = seq_len,
            forecast_horizon = horizon,
            patch_len        = patch_len,
            batch_size       = batch_size,
            dataset_id       = dataset_id,
        )
    except Exception as e:
        print(f"  Data error: {e}")
        return

    # Build all configs
    keys   = list(SEARCH_SPACE.keys())
    values = list(SEARCH_SPACE.values())
    all_configs = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    total = len(all_configs)

    print(f"  Searching {total} configs ({total} = "
          f"{len(SEARCH_SPACE['tk_d_model'])} d_model × "
          f"{len(SEARCH_SPACE['tk_e_layers'])} e_layers × "
          f"{len(SEARCH_SPACE['lr'])} lr)")

    # ── Resume from previous crash ────────────────────────────────────────
    progress_path = save_dir / "search_progress.json"
    completed_keys = set()
    all_results  = []
    best_val_mae = float('inf')
    best_result  = None

    if progress_path.exists():
        try:
            with open(progress_path) as f:
                prev = json.load(f)
            all_results  = prev.get('all_results', [])
            best_val_mae = prev.get('best_val_mae', float('inf'))
            best_result  = prev.get('best_config', None)
            completed_keys = {str(sorted(r['hparams'].items()))
                              for r in all_results}
            print(f"  Resuming — {len(completed_keys)} configs done, "
                  f"best val MAE so far: {best_val_mae:.4f}")
        except Exception as e:
            print(f"  Warning: could not read search_progress.json: {e}")

    for i, hparams in enumerate(all_configs, 1):
        # Skip already-completed configs
        if str(sorted(hparams.items())) in completed_keys:
            print(f"  [{i}/{total}] SKIP (already done)")
            continue
        config_str = (f"d_model={hparams['tk_d_model']} "
                      f"e_layers={hparams['tk_e_layers']} "
                      f"lr={hparams['lr']:.0e}")
        print(f"\n  [{i}/{total}] {config_str}")

        t0 = time.time()
        val_mae, result = train_one_config(
            dataset_id, term, hparams, device,
            train_loader, val_loader,
            n_exo, seq_len, horizon, patch_len,
            save_dir,
        )
        elapsed = time.time() - t0

        if result is not None:
            result['elapsed_s'] = round(elapsed)
            all_results.append(result)
            print(f"  → val MAE: {val_mae:.4f} ({elapsed:.0f}s)")

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_result  = result
                print(f"  ★ New best!")
        else:
            print(f"  → FAILED")

        # Save progress after each config
        progress = {
            'dataset'      : ds_name,
            'term'         : term,
            'timestamp'    : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'completed'    : i,
            'total'        : total,
            'best_val_mae' : best_val_mae,
            'best_config'  : best_result,
            'all_results'  : sorted(all_results, key=lambda x: x['best_val_mae']),
        }
        with open(save_dir / "search_progress.json", 'w') as f:
            json.dump(progress, f, indent=2, default=str)

    # Save final best config
    if best_result:
        with open(best_cfg_path, 'w') as f:
            json.dump(best_result, f, indent=2, default=str)

        print(f"\n  ✓ Best config for [{ds_name} {term}]:")
        print(f"    d_model={best_result['hparams']['tk_d_model']} "
              f"e_layers={best_result['hparams']['tk_e_layers']} "
              f"lr={best_result['hparams']['lr']:.0e}")
        print(f"    val MAE: {best_val_mae:.4f}")

    return best_result


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index to use (e.g. 0, 1, 2, 3)')
    parser.add_argument('--datasets', type=int, nargs='+',
                        default=list(range(5, 17)),
                        help='Dataset IDs to search (e.g. 5 6 7 8 9)')
    parser.add_argument('--terms', type=str, nargs='+',
                        default=['short', 'long'],
                        help='Terms to search: short long')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing GPU {args.gpu} → device: {device}")
    print(f"Datasets: {args.datasets}")
    print(f"Terms:    {args.terms}")
    print(f"Search space: {SEARCH_SPACE}")
    print(f"Configs per dataset-term: "
          f"{len(SEARCH_SPACE['tk_d_model']) * len(SEARCH_SPACE['tk_e_layers']) * len(SEARCH_SPACE['lr'])}\n")

    for dataset_id in args.datasets:
        for term in args.terms:
            search(dataset_id, term, device)
            cleanup()

    print("\n=== HPARAM SEARCH COMPLETE ===")

    # Print final summary across all searched datasets
    print("\n=== BEST CONFIGS SUMMARY ===")
    print(f"{'Dataset':<12} {'Term':<7} {'d_model':<10} {'e_layers':<10} {'lr':<10} {'Val MAE'}")
    print("─" * 60)

    for dataset_id in args.datasets:
        if dataset_id not in DATASET_PATHS:
            continue
        _, _, ds_name, _ = DATASET_PATHS[dataset_id]
        for term in args.terms:
            cfg_path = RESULTS_ROOT / ds_name / term / "best_config.json"
            if cfg_path.exists():
                with open(cfg_path) as f:
                    cfg = json.load(f)
                h = cfg['hparams']
                print(f"{ds_name:<12} {term:<7} "
                      f"{h['tk_d_model']:<10} "
                      f"{h['tk_e_layers']:<10} "
                      f"{h['lr']:<10} "
                      f"{cfg['best_val_mae']:.4f}")
            else:
                print(f"{ds_name:<12} {term:<7} {'NOT RUN'}")
