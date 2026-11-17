# hparam_MAGNET_TFT_multiseed.py
# 5-seed hyperparameter search for MAGNet+TFT (plug-and-play, hybrid mode 2, ngc_lag=1, history-only
# input). Every config of the search space is trained with seeds {0,42,123,2024,3407} and the
# winner is the config with the lowest MEAN validation MAE; its per-seed test MAE/MSE are then
# evaluated (mean over seeds is what we report). Fully resumable.
#
# Usage (one dataset/term per process):
#   python3 pnp/hparam_MAGNET_TFT_multiseed.py --gpu 0 --dataset 10 --term short
#   python3 pnp/hparam_MAGNET_TFT_multiseed.py --gpu 0 --dataset 10 --term short --eval-only
import argparse
import os

_p = argparse.ArgumentParser(add_help=False)
_p.add_argument('--gpu', type=int, default=0)
_a, _ = _p.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(_a.gpu)

import gc
import json
import time
import math
import statistics
import itertools
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

from hparam_MAGNET_TFT import SEARCH_SPACE, DATASET_PATHS, train_one_config, PKG_ROOT
from test_best_TFT import evaluate, get_seq_horizon, TTA_DATASETS
from MAGNET_with_TFT import MAGNETWithTFT
from MAGNET_data import get_dataloaders

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

SEEDS_DEFAULT = [0, 42, 123, 2024, 3407]
RESULTS_ROOT = Path(PKG_ROOT) / "runs" / "pnp_search" / "tft_lag1_multiseed"
PATCH_LEN = 12
BATCH_SIZE = 64
NGC_LAG = 1


def batch_size_for(term):
    # long-term (seq_len=720/180) OOM'd at batch=64 under current heavy
    # external GPU contention (<2GB free) — drop to 16 for long, keep the
    # original 64 for short (seq_len=168/60, much lighter).
    return 16 if term == 'long' else BATCH_SIZE


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.floating,)):
            return round(float(o), 6)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def cfg_key(h):
    return f"dm{h['tft_d_model']}_lr{h['lr']}"


def all_configs():
    keys = list(SEARCH_SPACE.keys())
    return [dict(zip(keys, c)) for c in itertools.product(*SEARCH_SPACE.values())]


def _prune_ckpts(base_dir, keep_key):
    for d in base_dir.glob("dm*"):
        if d.name == keep_key:
            continue
        for ck in d.glob("seed_*/ckpt_*.pt"):
            try:
                ck.unlink()
            except OSError:
                pass


# ── Search ────────────────────────────────────────────────────────────────────

def run_search(dataset_id, term, seeds, device):
    npz_path, n_exo, ds_name, _ = DATASET_PATHS[dataset_id]
    seq_len, horizon = get_seq_horizon(dataset_id, term)
    base_dir = RESULTS_ROOT / ds_name / term
    base_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = get_dataloaders(
        npz_path=npz_path, sensor_id=0, seq_len=seq_len, forecast_horizon=horizon,
        patch_len=PATCH_LEN, batch_size=batch_size_for(term), dataset_id=dataset_id,
    )

    configs = all_configs()
    print(f"\n{'='*70}\n[{ds_name} {term}] MAGNet+TFT multiseed search: "
          f"{len(configs)} configs x {len(seeds)} seeds\n{'='*70}", flush=True)

    best_mean = float('inf')
    best_key = None

    for ci, h in enumerate(configs, 1):
        k = cfg_key(h)
        cdir = base_dir / k
        seed_vals = {}
        for seed in seeds:
            sdir = cdir / f"seed_{seed}"
            rp = sdir / "result.json"
            if rp.exists():
                seed_vals[seed] = json.loads(rp.read_text()).get('val_mae', float('inf'))
                print(f"  [{ci}/{len(configs)}] {k} seed={seed}  SKIP "
                      f"(val {seed_vals[seed]:.4f})", flush=True)
                continue
            sdir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            val_mae, result = train_one_config(
                dataset_id, term, h, device,
                train_loader, val_loader,
                n_exo, seq_len, horizon, PATCH_LEN, sdir,
                seed=seed, ngc_lag=NGC_LAG,
            )
            dt = time.time() - t0
            if result is None:
                seed_vals[seed] = float('inf')
                print(f"  [{ci}/{len(configs)}] {k} seed={seed}  FAILED "
                      f"(no result, {dt:.0f}s) — will retry on re-run", flush=True)
                cleanup()
                continue
            rec = {
                'dataset': ds_name, 'term': term, 'hparams': h, 'seed': seed,
                'val_mae': val_mae, 'elapsed_s': round(dt),
                'ckpt_path': result.get('ckpt_path'),
                'magnet_config': result.get('magnet_config'),
                'tft_config': result.get('tft_config'),
                'ok': True,
            }
            rp.write_text(json.dumps(rec, indent=2, cls=_Enc))
            seed_vals[seed] = val_mae
            print(f"  [{ci}/{len(configs)}] {k} seed={seed}  val {val_mae:.4f}  ({dt:.0f}s)",
                  flush=True)
            cleanup()

        finite = [v for v in seed_vals.values() if math.isfinite(v)]
        mean_val = statistics.mean(finite) if len(finite) == len(seeds) else float('inf')
        _write_progress(base_dir, ds_name, term, configs, seeds)

        if mean_val < best_mean:
            best_mean, best_key = mean_val, k
            _prune_ckpts(base_dir, best_key)
            print(f"  -> new best: {k}  mean val MAE {mean_val:.4f}", flush=True)
        elif math.isfinite(mean_val):
            for ck in cdir.glob("seed_*/ckpt_*.pt"):
                try:
                    ck.unlink()
                except OSError:
                    pass

    print(f"\n[{ds_name} {term}] search done. best config: {best_key} "
          f"(mean val MAE {best_mean:.4f})", flush=True)
    return best_key


def _write_progress(base_dir, ds_name, term, configs, seeds):
    rows = []
    for h in configs:
        k = cfg_key(h)
        done = sum(1 for s in seeds if (base_dir / k / f"seed_{s}" / "result.json").exists())
        vals = []
        for s in seeds:
            rp = base_dir / k / f"seed_{s}" / "result.json"
            if rp.exists():
                vals.append(json.loads(rp.read_text()).get('val_mae', float('inf')))
        mean_val = (statistics.mean(vals) if len(vals) == len(seeds)
                    and all(math.isfinite(v) for v in vals) else None)
        rows.append({'config': k, 'hparams': h, 'seeds_done': done,
                     'mean_val_mae': mean_val})
    rows.sort(key=lambda r: (r['mean_val_mae'] is None, r['mean_val_mae'] or 0))
    (base_dir / "search_progress.json").write_text(json.dumps({
        'dataset': ds_name, 'term': term,
        'updated': datetime.now().isoformat(timespec='seconds'),
        'configs_complete': sum(1 for r in rows if r['seeds_done'] == len(seeds)),
        'configs_total': len(configs),
        'rows': rows,
    }, indent=2, cls=_Enc))


# ── Winner selection + test eval ──────────────────────────────────────────────

def _select_winner(base_dir, seeds):
    best = (float('inf'), None)
    for cdir in sorted(base_dir.glob("dm*")):
        vals = []
        for s in seeds:
            rp = cdir / f"seed_{s}" / "result.json"
            if not rp.exists():
                break
            r = json.loads(rp.read_text())
            vals.append(r.get('val_mae', float('inf')))
        if len(vals) != len(seeds) or not all(math.isfinite(v) for v in vals):
            continue
        m = statistics.mean(vals)
        if m < best[0]:
            best = (m, cdir.name)
    return best[1]


def run_eval(dataset_id, term, seeds, device):
    npz_path, n_exo, ds_name, _ = DATASET_PATHS[dataset_id]
    seq_len, horizon = get_seq_horizon(dataset_id, term)
    base_dir = RESULTS_ROOT / ds_name / term

    n_configs = len(all_configs())
    complete = sum(
        1 for cdir in base_dir.glob("dm*")
        if all((cdir / f"seed_{s}" / "result.json").exists() for s in seeds)
    )
    MIN_COMPLETE = n_configs - 2 if n_configs > 2 else n_configs
    if complete < MIN_COMPLETE:
        print(f"[{ds_name} {term}] only {complete}/{n_configs} configs have all "
              f"{len(seeds)} seeds (need >= {MIN_COMPLETE}) — not finalizing yet.",
              flush=True)
        raise SystemExit(2)

    winner = _select_winner(base_dir, seeds)
    if winner is None:
        print(f"[{ds_name} {term}] no config has all {len(seeds)} seeds yet — "
              f"cannot pick a winner.", flush=True)
        raise SystemExit(2)
    wdir = base_dir / winner
    print(f"\n[{ds_name} {term}] winner = {winner}", flush=True)

    train_loader, val_loader, test_loader = get_dataloaders(
        npz_path=npz_path, sensor_id=0, seq_len=seq_len,
        forecast_horizon=horizon, patch_len=PATCH_LEN,
        batch_size=16, dataset_id=dataset_id,
    )

    per_seed = {}
    hparams = None
    for s in seeds:
        r = json.loads((wdir / f"seed_{s}" / "result.json").read_text())
        hparams = r['hparams']
        mcfg, tcfg = r['magnet_config'], r['tft_config']
        ckpt = r.get('ckpt_path')

        if not ckpt or not Path(ckpt).exists():
            print(f"  seed={s}: checkpoint missing, retraining …", flush=True)
            tl, vl, _ = get_dataloaders(
                npz_path=npz_path, sensor_id=0, seq_len=seq_len, forecast_horizon=horizon,
                patch_len=PATCH_LEN, batch_size=batch_size_for(term), dataset_id=dataset_id,
            )
            _, res = train_one_config(dataset_id, term, hparams, device, tl, vl,
                                      n_exo, seq_len, horizon, PATCH_LEN,
                                      wdir / f"seed_{s}", seed=s, ngc_lag=NGC_LAG)
            if res is None:
                print(f"  seed={s}: retrain FAILED, skipping seed", flush=True)
                continue
            mcfg, tcfg, ckpt = res['magnet_config'], res['tft_config'], res['ckpt_path']
            r.update(ckpt_path=ckpt, magnet_config=mcfg, tft_config=tcfg)
            (wdir / f"seed_{s}" / "result.json").write_text(json.dumps(r, indent=2, cls=_Enc))

        model = MAGNETWithTFT(mcfg, tcfg, mode=2).to(device)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state, strict=False)
        m = evaluate(model, test_loader, dataset_id, term,
                     use_tta=(dataset_id in TTA_DATASETS))
        del model
        cleanup()
        per_seed[s] = {'val_mae': r['val_mae'], 'test_mae': m['mae'], 'test_mse': m['mse']}
        print(f"  seed={s}: val {r['val_mae']:.4f}  test MAE {m['mae']:.4f}  "
              f"test MSE {m['mse']:.4f}", flush=True)

    if not per_seed:
        print(f"[{ds_name} {term}] no seeds evaluated.", flush=True)
        return

    maes = [v['test_mae'] for v in per_seed.values()]
    mses = [v['test_mse'] for v in per_seed.values()]
    agg = {
        'mae_mean': statistics.mean(maes),
        'mae_std': statistics.pstdev(maes) if len(maes) > 1 else 0.0,
        'mse_mean': statistics.mean(mses),
        'mse_std': statistics.pstdev(mses) if len(mses) > 1 else 0.0,
    }

    out = {
        'dataset': ds_name, 'term': term,
        'method': 'MAGNet+TFT (hybrid, mode=2), multiseed hparam search',
        'selection': 'min mean val MAE across seeds',
        'seeds': seeds,
        'winner_config': winner,
        'winner_hparams': hparams,
        'per_seed': per_seed,
        'magnet_tft_multiseed': agg,
    }

    (base_dir / "best_config_multiseed.json").write_text(json.dumps(out, indent=2, cls=_Enc))
    print(f"\n[{ds_name} {term}] MAGNet+TFT  MAE {agg['mae_mean']:.4f}±{agg['mae_std']:.4f}"
          f"\n  wrote {base_dir/'best_config_multiseed.json'}", flush=True)


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--dataset', type=int, required=True)
    ap.add_argument('--term', choices=['short', 'long'], required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS_DEFAULT)
    ap.add_argument('--eval-only', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"GPU {args.gpu} -> {device} | dataset={args.dataset} term={args.term} "
          f"seeds={args.seeds} eval_only={args.eval_only}", flush=True)

    if not args.eval_only:
        run_search(args.dataset, args.term, args.seeds, device)
    run_eval(args.dataset, args.term, args.seeds, device)
