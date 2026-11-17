# hparam_MAGNET_DAG.py
# Search space and training code (train_one_config) for MAGNet+DAG (hybrid mode 2, ngc_lag=1).
# DAG's training objective uses the true future exogenous series as a teacher signal (auxiliary causality loss);
# validation and test use the observed history only (infer_use_future=False).
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
from MAGNET_with_DAG import MAGNETWithDAG

# Without this, cudnn picks non-deterministic conv/attention kernels regardless
# of the manual_seed() calls in train_one_config, so re-running the SAME hparams
# through a 200-epoch/patience-15 training loop produces a genuinely different
# model each time (confirmed on NP long: vanilla test MAE swung 0.5 -> 1.7 -> 2.6
# across three identical-hparams retrains). See same fix in test_best_iTrans.py
# and test_best_DAG.py.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ── Search space ──────────────────────────────────────────────────────────────
# Joint search: DAG backbone + MAGNet graph d_model + lr

SEARCH_SPACE = {
    'dag_d_model'   : [128, 256],         # DAG temporal/cov encoder width
    'dag_e_layers'  : [1, 2],             # DAG encoder depth
    'magnet_d_model': [32, 64, 128],      # MAGNet graph size
    'lr'            : [1e-4, 5e-4],
}
# 2 × 2 × 3 × 2 = 24 configs per dataset-term

MAGNET_N_HEADS = {32: 2, 64: 4, 128: 8}

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
RESULTS_ROOT  = Path(PKG_ROOT) / "runs" / "pnp_search" / "dag"

# ── Metrics ───────────────────────────────────────────────────────────────────

def mae(pred, target):
    return torch.mean(torch.abs(pred - target)).item()

# ── Training epoch ────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device, dataset, train=True, term="short"):
    model.train() if train else model.eval()
    all_preds, all_targets = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for batch in loader:
            if len(batch) == 8:   # training loader also carries the true future exogenous series (teacher)
                exo_patches, endo_patches, target, mean, std, exo_future, time_features, last_known = batch
            else:
                exo_patches, endo_patches, target, mean, std, time_features, last_known = batch
                exo_future = None
            exo_patches   = exo_patches.to(device)
            endo_patches  = endo_patches.to(device)
            target        = target.to(device)
            time_features = time_features.to(device)
            last_known    = last_known.to(device)
            # future exogenous values are used only as DAG's training-time teacher, never at evaluation
            exo_future = exo_future.to(device) if (train and exo_future is not None) else None

            raw = model(exo_patches, endo_patches,
                        exog_future=exo_future, time_features=time_features)
            # DAG returns (pred, causality_loss) while training, pred only at eval
            if isinstance(raw, (tuple, list)):
                pred, causality_loss = raw
            else:
                pred, causality_loss = raw, None
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
                # DAG's own auxiliary causality loss (exog-forecast + covariance
                # consistency, already beta-scaled inside DAGModel) — fold it
                # into the task loss so its causality heads get supervised too.
                if causality_loss is not None and torch.is_tensor(causality_loss):
                    total_loss = loss + causality_loss
                else:
                    total_loss = loss
                optimizer.zero_grad()
                total_loss.backward()
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

# ── Single config training ────────────────────────────────────────────────────

def train_one_config(
    dataset_id, term, hparams, device, npz_path,
    n_exo, seq_len, horizon, patch_len,
    save_dir,
    node_text_fusion="qk_bias",  # ablation switch for run_full_text_variant_ablation.py —
                                 # default "qk_bias" reproduces this function's original
                                 # behavior exactly, so search()/retrain_dag.py callers
                                 # (which never pass this) are unaffected.
    seed: int = 2024,            # default 2024 reproduces the original single-seed
                                 # behavior exactly; hparam_MAGNET_DAG_multiseed.py
                                 # passes 0/42/123/2024/3407 for the 5-seed search.
    num_workers: int = 0,        # default 0 = original behavior. >0 only parallelises
                                 # the DataLoader's CPU work (patchify per window); the
                                 # RandomSampler permutation is still generated in the
                                 # main process, so batch composition/order — and hence
                                 # the trained model — is bit-identical to num_workers=0.
    ngc_lag: int = 1,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Adaptive batch size — DAG runs two encoders (temporal + covariance),
    # heavier than TimeXer at the same d_model
    if hparams['dag_d_model'] == 256 and hparams['dag_e_layers'] >= 2 and term == 'long':
        batch_size = 16
    elif hparams['dag_d_model'] == 256 or term == 'long':
        batch_size = 32
    else:
        batch_size = 64

    train_loader, val_loader, _ = get_dataloaders(
        npz_path           = npz_path,
        sensor_id          = 0,
        seq_len            = seq_len,
        forecast_horizon   = horizon,
        patch_len          = patch_len,
        batch_size         = batch_size,
        dataset_id         = dataset_id,
        num_workers        = num_workers,
        persistent_workers = num_workers > 0,
        return_future_exog = True,   # training loop only: DAG's teacher signal
    )

    d_graph = hparams['magnet_d_model']
    n_heads = MAGNET_N_HEADS.get(d_graph, 4)

    magnet_config = {
        'n_exo'         : n_exo,
        'patch_len'     : patch_len,
        'num_patches'   : seq_len // patch_len,
        'd_model'       : d_graph,
        'n_heads'       : n_heads,
        'ngc_hidden_dim': 16 if (dataset_id == 11 and term == 'long') else 32,
        'dropout'       : 0.1 if term == 'short' else 0.2,
        'dataset'       : dataset_id,
        'term'          : term,
        'ngc_lag'       : ngc_lag,
    }

    # DAG config
    class DagConfig:
        pass

    dag_cfg = DagConfig()
    dag_cfg.seq_len          = seq_len
    dag_cfg.pred_len         = horizon
    dag_cfg.patch_len        = 16
    dag_cfg.stride           = 8
    dag_cfg.use_c            = True
    dag_cfg.use_t            = True
    dag_cfg.use_c_exog       = True
    dag_cfg.use_t_exog       = True
    dag_cfg.alpha            = 0.2
    dag_cfg.beta             = 0.1
    dag_cfg.series_dim       = 1
    dag_cfg.infer_use_future = False   # no future exogenous values at inference
    dag_cfg.enc_in           = n_exo + 1
    dag_cfg.d_model          = hparams['dag_d_model']
    dag_cfg.d_ff             = hparams['dag_d_model'] * 4
    dag_cfg.n_heads          = 8 if hparams['dag_d_model'] >= 256 else 4
    dag_cfg.e_layers         = hparams['dag_e_layers']
    dag_cfg.dropout          = 0.2
    dag_cfg.factor           = 1
    dag_cfg.activation       = "gelu"
    dag_cfg.criterion        = nn.L1Loss()

    try:
        model = MAGNETWithDAG(magnet_config, dag_cfg, mode=2,
                               node_text_fusion=node_text_fusion).to(device)
    except Exception as e:
        print(f"      Model init error: {e}")
        return float('inf'), None

    total_params = sum(p.numel() for p in model.parameters())

    # LR warmup (same datasets as standalone MAGNet)
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

    best_val_mae   = float('inf')
    patience_count = 0
    best_state     = None

    try:
        for epoch in range(1, epochs + 1):
            run_epoch(model, train_loader, optimizer, device,
                      dataset_id, train=True, term=term)
            val_m = run_epoch(model, val_loader, optimizer, device,
                              dataset_id, train=False, term=term)
            scheduler.step()

            if val_m['mae'] < best_val_mae:
                best_val_mae   = val_m['mae']
                patience_count = 0
                best_state     = {k: v.cpu().clone()
                                  for k, v in model.state_dict().items()}
            else:
                patience_count += 1
                if patience_count >= patience:
                    break

    except torch.cuda.OutOfMemoryError:
        print(f"      OOM — skipping config")
        del model
        torch.cuda.empty_cache()
        gc.collect()
        return float('inf'), None
    except Exception as e:
        print(f"      Error: {e}")
        del model
        torch.cuda.empty_cache()
        gc.collect()
        return float('inf'), None

    # Save checkpoint
    config_str = (f"dagdm{hparams['dag_d_model']}_"
                  f"el{hparams['dag_e_layers']}_"
                  f"dm{hparams['magnet_d_model']}_"
                  f"lr{hparams['lr']}")
    ckpt_path = save_dir / f"ckpt_{config_str}.pt"
    if best_state is not None:
        torch.save(best_state, ckpt_path)

    result = {
        'hparams'      : hparams,
        'magnet_config': magnet_config,
        'dag_config'   : {
            'd_model'   : dag_cfg.d_model,
            'd_ff'      : dag_cfg.d_ff,
            'n_heads'   : dag_cfg.n_heads,
            'e_layers'  : dag_cfg.e_layers,
            'seq_len'   : dag_cfg.seq_len,
            'pred_len'  : dag_cfg.pred_len,
            'patch_len' : dag_cfg.patch_len,
            'stride'    : dag_cfg.stride,
            'enc_in'    : dag_cfg.enc_in,
            'alpha'     : dag_cfg.alpha,
            'beta'      : dag_cfg.beta,
        },
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

    if dataset_id in {11, 12}:
        seq_len = 60  if term == 'short' else 180
        horizon = 10  if term == 'short' else 30
    else:
        seq_len = 168 if term == 'short' else 720
        horizon = 24  if term == 'short' else 360

    patch_len = 12

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

    keys        = list(SEARCH_SPACE.keys())
    values      = list(SEARCH_SPACE.values())
    all_configs = [dict(zip(keys, c)) for c in itertools.product(*values)]
    total       = len(all_configs)

    print(f"  Searching {total} configs "
          f"({len(SEARCH_SPACE['dag_d_model'])} dag_d_model × "
          f"{len(SEARCH_SPACE['dag_e_layers'])} e_layers × "
          f"{len(SEARCH_SPACE['magnet_d_model'])} magnet_dm × "
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
        config_str = (f"dag_dm={hparams['dag_d_model']} "
                      f"el={hparams['dag_e_layers']} "
                      f"mag_dm={hparams['magnet_d_model']} "
                      f"lr={hparams['lr']:.0e}")
        print(f"\n  [{i}/{total}] {config_str}")

        t0 = time.time()
        val_mae, result = train_one_config(
            dataset_id, term, hparams, device, npz_path,
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

        # Save progress
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

    if best_result:
        with open(best_cfg_path, 'w') as f:
            json.dump(best_result, f, indent=2, default=str)
        print(f"\n  ✓ Best config for [{ds_name} {term}]:")
        h = best_result['hparams']
        print(f"    dag_d_model={h['dag_d_model']} e_layers={h['dag_e_layers']} "
              f"magnet_dm={h['magnet_d_model']} lr={h['lr']:.0e}")
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
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--datasets', type=int, nargs='+',
                        default=list(range(5, 17)))
    parser.add_argument('--terms', type=str, nargs='+',
                        default=['short', 'long'])
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing GPU {args.gpu} → device: {device}")
    print(f"Datasets: {args.datasets}")
    print(f"Terms:    {args.terms}")
    print(f"Search space: {SEARCH_SPACE}")
    n_configs = (len(SEARCH_SPACE['dag_d_model']) *
                 len(SEARCH_SPACE['dag_e_layers']) *
                 len(SEARCH_SPACE['magnet_d_model']) *
                 len(SEARCH_SPACE['lr']))
    print(f"Configs per dataset-term: {n_configs}\n")

    for dataset_id in args.datasets:
        for term in args.terms:
            search(dataset_id, term, device)
            cleanup()

    print("\n=== HPARAM SEARCH COMPLETE ===")
    print("\n=== BEST CONFIGS SUMMARY ===")
    print(f"{'Dataset':<12} {'Term':<7} {'dag_dm':<8} {'el':<5} "
          f"{'mag_dm':<9} {'lr':<10} {'Val MAE'}")
    print("─" * 58)

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
                      f"{h['dag_d_model']:<8} "
                      f"{h['dag_e_layers']:<5} "
                      f"{h['magnet_d_model']:<9} "
                      f"{h['lr']:<10} "
                      f"{cfg['best_val_mae']:.4f}")
            else:
                print(f"{ds_name:<12} {term:<7} NOT RUN")
