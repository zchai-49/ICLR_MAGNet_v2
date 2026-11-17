#!/usr/bin/env python3
"""MAGNet as a plug-and-play (PnP) module on other backbones (PatchTST, TimeKAN, TimeXer, TFT, DAG, GCGNet).

    python3 train/run_pnp.py --backbone PatchTST --dataset Energy --term short --gpu 0
    python3 train/run_pnp.py --backbone TFT --dataset Syn20 --term long --seeds 0 42 --gpu 0

For every seed this trains MAGNet+<backbone> (hybrid mode 2, history-only input; the MAGNet branch is the final SIG v2 graph
with detached weights and the fixed cross-fit gate, models/final_defaults.py) with the selected hyperparameters, evaluates the
best-validation checkpoint on the test set and writes runs/pnp/<backbone>/<dataset>/<term>/seed_<seed>/result.json.
  real datasets      configs/pnp_selected_hparams.json  (per backbone x dataset x term)
  synthetic datasets the PnP hparams of the template dataset chosen on VALIDATION (seeds 0/42) among Energy / NP /
                     Sdwpfh1 (+ Sdwpfm1 for GCGNet): configs/pnp_syn_selected.json; --hp-from <template> forces one
Table 2 reports the 5-seed mean (seeds 0, 42, 123, 2024, 3407), see experiments/table2_pnp.
MAGNET_PNP_SHUFFLE=1 additionally records the branch-shuffle Test 1 in result.json["shuffle"] (experiments/test1_branch_shuffle);
MAGNET_PNP_DROP_CKPT=1 deletes the checkpoint after evaluation (no PnP checkpoints are shipped).
The original hyperparameter searches are pnp/hparam_MAGNET_<backbone>[_multiseed].py.
"""
import argparse
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--gpu", type=int, default=0)
os.environ["CUDA_VISIBLE_DEVICES"] = str(_p.parse_known_args()[0].gpu)

for _d in ("pnp", "models", "data", "backbones"):
    sys.path.insert(0, os.path.join(ROOT, _d))

import torch  # noqa: E402
from MAGNET_data import get_dataloaders  # noqa: E402

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

SEEDS = [0, 42, 123, 2024, 3407]
PATCH_LEN = 12
NGC_LAG = 1


def load_backbone(name):
    if name == "PatchTST":
        import hparam_MAGNET_PatchTST as H, test_best_PatchTST as T
        from MAGNET_with_PatchTST import MAGNETWithPatchTST as M
        return H, T, M, "ptst_config"
    if name == "TimeKAN":
        import hparam_MAGNET_TimeKAN as H, test_best_TimeKAN as T
        from MAGNET_with_TimeKAN import MAGNETWithTimeKAN as M
        return H, T, M, "tk_config"
    if name == "TFT":
        import hparam_MAGNET_TFT as H, test_best_TFT as T
        from MAGNET_with_TFT import MAGNETWithTFT as M
        return H, T, M, "tft_config"
    if name == "TimeXer":
        import hparam_MAGNET_TimeXer as H, test_best_TimeXer as T
        from MAGNET_with_TimeXer import TimeXerWithMAGNET as M
        return H, T, M, "tx_config"
    if name == "DAG":
        import hparam_MAGNET_DAG as H, test_best_DAG as T
        from MAGNET_with_DAG import MAGNETWithDAG as M
        return H, T, M, "dag_config"
    raise ValueError(name)


def branch_shuffle(model, test_loader, ev, seed):
    """Test 1: test MAE intact vs with the MAGNet graph branch's exogenous input replaced by a random OTHER test
    window's (forward pre-hook on model.magnet_graph; backbone keeps the true exo). Seeding as branch_shuffle_test.py."""
    import numpy as np
    pool = torch.cat([b[0] for b in test_loader]); model.eval()
    torch.manual_seed(0); np.random.seed(0); m0 = ev()
    rng = np.random.default_rng(seed)
    h = model.magnet_graph.register_forward_pre_hook(
        lambda _m, a: (pool[torch.from_numpy(rng.integers(0, len(pool), size=a[0].shape[0]))].to(a[0].device, a[0].dtype),) + tuple(a[1:]))
    try:
        torch.manual_seed(0); np.random.seed(0); m1 = ev()
    finally:
        h.remove()
    return {"intact_mae": m0["mae"], "shuffled_mae": m1["mae"], "intact_mse": m0["mse"], "shuffled_mse": m1["mse"]}


def run_gcgnet(args, hparams, reported):
    """MAGNet+GCGNet: GCGNet's own hyperparameters are fixed per dataset; only magnet_d_model comes from the search."""
    import hparam_MAGNET_GCGNet as G
    dataset_id = {v[2]: k for k, v in G.DATASET_PATHS.items()}[args.dataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"MAGNet+GCGNet {args.dataset}/{args.term}  magnet_d_model={hparams['magnet_d_model']}  seeds={args.seeds}", flush=True)
    per_seed = {}
    for seed in args.seeds:
        sdir = os.path.join(args.out_root, "GCGNet", args.dataset, args.term, f"seed_{seed}")
        val_mae, res = G.train_one_config(dataset_id, args.term, hparams["magnet_d_model"], device, seed, sdir)
        if res is None:
            print(f"seed {seed}: training failed", flush=True)
            continue
        m = res["test_metrics"]
        per_seed[seed] = m
        rep = reported["per_seed"].get(str(seed))
        print(f"seed {seed}: val {val_mae:.4f}  test MAE {m['mae']:.4f} MSE {m['mse']:.4f}"
              + (f"   (reported MAE {rep['mae']:.4f} MSE {rep['mse']:.4f})" if rep else ""), flush=True)
    if per_seed:
        print(f"\nMEAN over {len(per_seed)} seed(s): MAE {statistics.mean(v['mae'] for v in per_seed.values()):.4f}  "
              f"MSE {statistics.mean(v['mse'] for v in per_seed.values()):.4f}   "
              f"(reported, 5-seed mean: MAE {reported['mae_mean']:.4f}  MSE {reported['mse_mean']:.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["PatchTST", "TimeKAN", "TimeXer", "TFT", "DAG", "GCGNet"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--term", required=True, choices=["short", "long"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out-root", default=os.path.join(ROOT, "runs", "pnp"))
    ap.add_argument("--hp-from", default=None, help="borrow the selected hparams of this real dataset")
    args = ap.parse_args()

    # --hp-from <real dataset> (rebuttal 2026-11-15): a dataset without its own selection (synthetic) borrows that
    # template's selected PnP hparams, its MAGNet graph size and (GCGNet) its GCGNet hparams; chosen on validation.
    if args.hp_from is None and args.dataset.startswith("Syn"):   # synthetic: validation-selected template (configs/pnp_syn_selected.json)
        args.hp_from = json.load(open(os.path.join(ROOT, "configs", "pnp_syn_selected.json")))[f"{args.backbone}/{args.dataset}/{args.term}"]["winner"]
    src = args.hp_from or args.dataset
    sel = json.load(open(os.path.join(ROOT, "configs", "pnp_selected_hparams.json")))[args.backbone][f"{src}/{args.term}"]
    hparams = sel["hparams"]
    # reported = our Table 2 run of this cell (SIG v2), results/table2_pnp_per_seed.json
    reported = json.load(open(os.path.join(ROOT, "results", "table2_pnp_per_seed.json")))[f"{args.backbone}/{args.dataset}/{args.term}"]
    if args.backbone == "GCGNet":
        if args.hp_from:
            import hparam_MAGNET_GCGNet as G
            G.GCGNET_HPARAMS[args.dataset] = dict(G.GCGNET_HPARAMS[args.hp_from])
        return run_gcgnet(args, hparams, reported)
    H, T, Model, cfg_key = load_backbone(args.backbone)
    if args.hp_from and hasattr(H, "MAGNET_HPARAMS"):
        ids = {v[2]: k for k, v in H.DATASET_PATHS.items()}
        if (ids[args.hp_from], args.term) in H.MAGNET_HPARAMS:
            H.MAGNET_HPARAMS[(ids[args.dataset], args.term)] = H.MAGNET_HPARAMS[(ids[args.hp_from], args.term)]
    dataset_id = {v[2]: k for k, v in H.DATASET_PATHS.items()}[args.dataset]
    csv_path, n_exo, ds_name, _ = H.DATASET_PATHS[dataset_id]
    seq_len, horizon = T.get_seq_horizon(dataset_id, args.term)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"MAGNet+{args.backbone} {args.dataset}/{args.term}  hparams={hparams}  seeds={args.seeds}", flush=True)

    test_loader = get_dataloaders(npz_path=csv_path, sensor_id=0, seq_len=seq_len, forecast_horizon=horizon,
                                  patch_len=PATCH_LEN, batch_size=16, dataset_id=dataset_id)[2]
    train_loader = val_loader = None
    if args.backbone not in ("TimeXer", "DAG"):  # these build their own loaders (batch size depends on the dataset)
        bs = 16 if args.term == "long" else 64
        train_loader, val_loader, _ = get_dataloaders(npz_path=csv_path, sensor_id=0, seq_len=seq_len,
                                                      forecast_horizon=horizon, patch_len=PATCH_LEN,
                                                      batch_size=bs, dataset_id=dataset_id)
    per_seed = {}
    for seed in args.seeds:
        sdir = os.path.join(args.out_root, args.backbone, args.dataset, args.term, f"seed_{seed}")
        os.makedirs(sdir, exist_ok=True)
        from pathlib import Path
        if args.backbone in ("TimeXer", "DAG"):
            val_mae, res = H.train_one_config(dataset_id, args.term, hparams, device, csv_path, n_exo,
                                              seq_len, horizon, PATCH_LEN, Path(sdir), seed=seed, ngc_lag=NGC_LAG)
        else:
            val_mae, res = H.train_one_config(dataset_id, args.term, hparams, device, train_loader, val_loader,
                                              n_exo, seq_len, horizon, PATCH_LEN, Path(sdir), seed=seed,
                                              ngc_lag=NGC_LAG)
        if res is None:
            print(f"seed {seed}: training failed", flush=True)
            continue
        state = torch.load(res["ckpt_path"], map_location=device)
        if args.backbone == "TimeXer":
            tx_cfg = T.build_timexer_config(res[cfg_key], n_exo, seq_len, horizon, PATCH_LEN, 16)
            model = Model(res["magnet_config"], tx_cfg, mode=2, graph_proj_arch=T.detect_graph_proj_arch(state))
            use_tta = True
        elif args.backbone == "DAG":
            model = Model(res["magnet_config"], T.build_dag_config(res[cfg_key], n_exo, seq_len, horizon), mode=2)
            use_tta = dataset_id in T.TTA_DATASETS
        else:
            model = Model(res["magnet_config"], res[cfg_key], mode=2)
            use_tta = dataset_id in T.TTA_DATASETS
        model.to(device).load_state_dict(state, strict=False)
        m = T.evaluate(model, test_loader, dataset_id, args.term, use_tta=use_tta)
        per_seed[seed] = {"val_mae": val_mae, "test_mae": m["mae"], "test_mse": m["mse"]}
        if os.environ.get("MAGNET_PNP_SHUFFLE") == "1":   # Test 1 (branch shuffle), experiments/test1_branch_shuffle, same as scripts/branch_shuffle_test.py
            per_seed[seed]["shuffle"] = branch_shuffle(model, test_loader, lambda: T.evaluate(model, test_loader, dataset_id, args.term, use_tta=use_tta), seed)
        rep = reported["per_seed"].get(str(seed))
        with open(os.path.join(sdir, "result.json"), "w") as f:
            json.dump({"backbone": args.backbone, "dataset": args.dataset, "term": args.term, "seed": seed,
                       "hparams": hparams, "ngc_lag": NGC_LAG, **per_seed[seed], "reported": rep}, f, indent=2)
        print(f"seed {seed}: val {val_mae:.4f}  test MAE {m['mae']:.4f} MSE {m['mse']:.4f}"
              + (f"   (reported MAE {rep['mae']:.4f} MSE {rep['mse']:.4f})" if rep else ""), flush=True)
        if os.environ.get("MAGNET_PNP_DROP_CKPT") == "1":   # disk (2026-11-15): best + resume checkpoints
            import glob as _g
            for _f in [res["ckpt_path"]] + _g.glob(os.path.join(sdir, "resume_*.pt")):
                if os.path.exists(_f):
                    os.remove(_f)
        del model
        torch.cuda.empty_cache()

    if per_seed:
        mae = statistics.mean(v["test_mae"] for v in per_seed.values())
        mse = statistics.mean(v["test_mse"] for v in per_seed.values())
        print(f"\nMEAN over {len(per_seed)} seed(s): MAE {mae:.4f}  MSE {mse:.4f}   "
              f"(reported, 5-seed mean: MAE {reported['mae_mean']:.4f}  MSE {reported['mse_mean']:.4f})")


if __name__ == "__main__":
    main()
