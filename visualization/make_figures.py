#!/usr/bin/env python3
"""Figures for MAGNet+DLinear: NGC causal-weight heatmap and case study (code only; nothing is precomputed).

    python3 visualization/make_figures.py --figure heatmap    --datasets Energy Colbun DE Sdwpfh1 --terms short long
    python3 visualization/make_figures.py --figure case_study --datasets Energy --terms short --train-missing
    python3 visualization/make_figures.py --causal sparse_granger_lag0 --figure lag_heatmap --datasets Energy DE Sdwpfm2 ...

heatmap     One panel per dataset: the exo->endo NGC causal weights of the trained MAGNet+DLinear models (mean over the test set,
            over the checkpoints present in checkpoints/ for every deployed side, resampled to a common number of patches).
case_study  Test windows where MAGNet+DLinear improves most over DLinear alone: ground truth and both forecasts, plus the exogenous
            variable with the highest mean causal weight. DLinear-alone checkpoints come from results/dlinear_alone/ (train them with
            train/train_dlinear_alone.py, or pass --train-missing). The exogenous series in the lower panel is the TRUE series over the
            forecast horizon, shown for context only; it is read by a separate loader and never passed to any model.
lag_heatmap (sparse-Granger estimators only, --causal sparse_granger*) One pair of panels per dataset: left, the learned Granger
            strength of every (exogenous variable, lag) group -- lag 0 = same patch, lags 1-3 = earlier patches -- with the target's
            own past as a reference row (mean over the checkpoints present); right, the PCMCI+ reference graph on the training data
            (|MCI| partial correlation, * = p < 0.05; results/causal_alt/pcmci_reference.json).
--causal picks the exo->endo estimator the checkpoints were trained with (default ngc). For a non-NGC estimator the default
checkpoint root is results/causal_alt/<causal>/ and figures go to figures/<causal>/ (NGC figures are never overwritten). The
patch-position heatmap is skipped for the sparse-Granger estimators, whose weights do not vary over patches.
Figures go to figures/{heatmap,case_study,lag_heatmap}/ as PNG and PDF.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("configs", "scripts", "train", "visualization"):
    sys.path.insert(0, os.path.join(ROOT, _d))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import lag1_config as C  # noqa: E402
import plot_utils as PU  # noqa: E402
from MAGNET_data import get_dataloaders  # noqa: E402
from evaluate import predict  # noqa: E402
from train_dlinear_alone import build_dlinear_alone, ckpt_path as alone_ckpt_path, train_alone  # noqa: E402


CAUSAL = "ngc"   # set from --causal in main()


def magnet_models(ds, term, device, ckpt_root):
    """Yield (side, seed, model, patch_len) for every checkpoint present."""
    for side in C.cell_config(ds, term)["deployed_sides"]:
        for seed in C.SEEDS:
            p = os.path.join(ckpt_root, f"{term}_{ds}_{side}", f"seed_{seed}", "ckpt.pt")
            if os.path.exists(p):
                model, patch_len = C.build_model(ds, term, side, causal=CAUSAL)
                model.load_state_dict(torch.load(p, map_location=device), strict=True)
                yield side, seed, model.to(device).eval(), patch_len


def test_loader(ds, term, patch_len, future_exog=False):
    dataset_id = C.NAME_TO_ID[ds]
    seq_len, horizon = C.get_seq_horizon(dataset_id, term)
    return get_dataloaders(npz_path=C.csv_path(dataset_id), sensor_id=0, seq_len=seq_len, forecast_horizon=horizon,
                           patch_len=patch_len, batch_size=32, dataset_id=dataset_id,
                           return_future_exog=future_exog)[2]


@torch.no_grad()
def causal_weights(model, loader, device, max_batches=None):
    out = []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        _, cw = model.magnet_graph.forward_with_causal_weights(batch[0].to(device), batch[1].to(device))  # [B, P, n_exo]
        out.append(cw.cpu())
    return torch.cat(out)


def figure_heatmap(datasets, term, device, ckpt_root, out_dir, max_batches):
    results = {}
    for ds in datasets:
        dataset_id = C.NAME_TO_ID[ds]
        per_model = []
        for side, seed, model, patch_len in magnet_models(ds, term, device, ckpt_root):
            cw = causal_weights(model, test_loader(ds, term, patch_len), device, max_batches)
            per_model.append(cw.mean(0).numpy())                      # [P, n_exo]
        if not per_model:
            print(f"  [skip] {ds}/{term}: no checkpoints in {ckpt_root}")
            continue
        p = max(h.shape[0] for h in per_model)
        hm = np.stack([PU.resample_patches(h, p) for h in per_model]).mean(0)
        names = PU.get_exo_names(dataset_id)
        results[f"{PU.DATASET_DOMAIN[dataset_id]} ({ds})"] = (hm.T, names)
        print(f"  {ds}/{term}: {len(per_model)} models, dominant variable {names[int(hm.mean(0).argmax())]}")
    if results:
        PU.plot_ngc_heatmap_grid(results, os.path.join(out_dir, "heatmap", f"ngc_heatmap_{term}.png"))


def figure_lag_heatmap(datasets, term, device, ckpt_root, out_dir):
    pc_path = os.path.join(ROOT, "results", "causal_alt", "pcmci_reference.json")
    pc = json.load(open(pc_path)) if os.path.exists(pc_path) else {}
    panels = {}
    for ds in datasets:
        mats = []
        for side, seed, model, patch_len in magnet_models(ds, term, device, ckpt_root):
            est = model.magnet_graph.causal_estimator
            g = est.group_norms().detach().cpu().numpy()                     # [K, V]: lags 1..K, var 0 = target
            c = est.contemporaneous_norms()
            n_exo = g.shape[1] - 1
            m = np.full((n_exo + 1, g.shape[0] + 1), np.nan)                 # rows: exo..., target; cols: lag 0..K
            m[:n_exo, 1:] = g[:, 1:].T
            m[n_exo, 1:] = g[:, 0]
            if c is not None:
                m[:n_exo, 0] = c.detach().cpu().numpy()
            mats.append(m)
        if not mats:
            print(f"  [skip] {ds}/{term}: no checkpoints in {ckpt_root}")
            continue
        names = PU.get_exo_names(C.NAME_TO_ID[ds]) + ["own past (target)"]
        ref = pc.get(f"{ds}/{term}", {}).get("parcorr")
        ref_m = ref_sig = None
        if ref:
            ref_m = np.abs(np.array(ref["mci_exo_to_endo"]))                  # [n_exo, tau 0..3]
            ref_m = np.vstack([ref_m, np.r_[np.nan, np.abs(ref["mci_endo_autoreg"])]])
            ref_sig = np.vstack([np.array(ref["p_exo_to_endo"]) < 0.05, np.zeros((1, ref_m.shape[1]), bool)])
        panels[ds] = (np.nanmean(np.stack(mats), 0), names, ref_m, ref_sig)
        top = names[int(np.nanargmax(np.nansum(panels[ds][0][:-1], 1)))]
        print(f"  {ds}/{term}: {len(mats)} models, strongest exogenous driver {top}")
    if panels:
        PU.plot_lag_heatmap_pairs(panels, term, os.path.join(out_dir, "lag_heatmap", f"lag_heatmap_{term}.png"))


def figure_case_study(ds, term, device, ckpt_root, alone_root, train_missing, out_dir, max_batches, n_windows=2):
    dataset_id = C.NAME_TO_ID[ds]
    side_preds, gt, first, seeds_used = {}, None, None, set()
    for side, seed, model, patch_len in magnet_models(ds, term, device, ckpt_root):
        p, t = predict(model, test_loader(ds, term, patch_len), dataset_id, term, device)
        side_preds.setdefault(side, []).append(p)
        seeds_used.add(seed)
        gt = t
        first = first or (model, patch_len)
    if not side_preds:
        print(f"  [skip] {ds}/{term}: no MAGNet checkpoints in {ckpt_root}")
        return
    pred_h = torch.stack([torch.stack(v).mean(0) for v in side_preds.values()]).mean(0)

    alone_preds = []
    for seed in sorted(seeds_used):
        path = alone_ckpt_path(alone_root, ds, term, seed)
        if not os.path.exists(path):
            if not train_missing:
                print(f"  [skip] no DLinear-alone checkpoint for seed {seed}: run train/train_dlinear_alone.py or pass --train-missing")
                continue
            train_alone(ds, term, seed, device.index or 0, 200, alone_root)
        m = build_dlinear_alone(ds, term).to(device)
        m.load_state_dict(torch.load(path, map_location=device), strict=True)
        alone_preds.append(predict(m.eval(), test_loader(ds, term, 12), dataset_id, term, device)[0])
    if not alone_preds:
        return
    pred_v = torch.stack(alone_preds).mean(0)

    model, patch_len = first
    cw = causal_weights(model, test_loader(ds, term, patch_len), device, max_batches).mean(1)      # [N, n_exo]
    # display only: true exogenous series over the forecast horizon, from a separate loader; never passed to a model
    exo_future = torch.cat([b[5] for b in test_loader(ds, term, patch_len, future_exog=True)])
    n = min(len(pred_h), len(pred_v), len(gt), len(cw), len(exo_future))
    pred_h, pred_v, gt, cw, exo_future = pred_h[:n], pred_v[:n], gt[:n], cw[:n], exo_future[:n]
    err_h, err_v = (pred_h - gt).abs().mean(-1), (pred_v - gt).abs().mean(-1)
    improvement = err_v - err_h
    print(f"  {ds}/{term}: MAGNet+DLinear MAE {err_h.mean():.4f} | DLinear alone {err_v.mean():.4f} | N={n}")
    idx_sel, gap = [], max(1, n // (4 * n_windows))
    for idx in torch.argsort(improvement, descending=True).tolist():
        if all(abs(idx - j) >= gap for j in idx_sel):
            idx_sel.append(idx)
        if len(idx_sel) == n_windows:
            break
    names = PU.get_exo_names(dataset_id)
    dom = int(cw.mean(0).argmax())
    windows = [{"window_idx": i, "gt": gt[i].numpy(), "pred_hybrid": pred_h[i].numpy(), "pred_vanilla": pred_v[i].numpy(),
                "improvement": improvement[i].item(), "exo_name": names[dom], "exo_series": exo_future[i, :, dom].numpy(),
                "exo_weight": cw[i, dom].item()} for i in idx_sel]
    PU.plot_case_studies(windows, ds, term, os.path.join(out_dir, "case_study", f"case_study_{ds}_{term}.png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure", choices=["heatmap", "case_study", "lag_heatmap", "all"], default="all")
    ap.add_argument("--causal", default="ngc", choices=C.CAUSAL_METHODS,
                    help="exo->endo estimator of the checkpoints (default ngc = the shipped MAGNet+DLinear models)")
    ap.add_argument("--datasets", nargs="+", default=["Energy"])
    ap.add_argument("--terms", nargs="+", default=["short", "long"])
    ap.add_argument("--ckpt-root", default=None, help="default: checkpoints/ (ngc) or results/causal_alt/<causal>/")
    ap.add_argument("--dlinear-alone-root", default=os.path.join(ROOT, "results", "dlinear_alone"))
    ap.add_argument("--train-missing", action="store_true", help="train missing DLinear-alone checkpoints for the case study")
    ap.add_argument("--out-dir", default=None, help="default: figures/ (ngc) or figures/<causal>/")
    ap.add_argument("--max-batches", type=int, default=None, help="cap test batches per model (default: full test set)")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    global CAUSAL
    CAUSAL = a.causal
    a.ckpt_root = a.ckpt_root or (os.path.join(ROOT, "checkpoints") if a.causal == "ngc"
                                  else os.path.join(ROOT, "results", "causal_alt", a.causal))
    a.out_dir = a.out_dir or (os.path.join(ROOT, "figures") if a.causal == "ngc"
                              else os.path.join(ROOT, "figures", a.causal))
    sparse = a.causal.startswith("sparse_granger")
    device = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    for term in a.terms:
        if a.figure in ("lag_heatmap", "all") and sparse:
            print(f"\n== lag heatmap ({a.causal}) vs PCMCI+, {term} ==")
            figure_lag_heatmap(a.datasets, term, device, a.ckpt_root, a.out_dir)
        if a.figure == "heatmap" and sparse:
            print("  [skip] patch-position heatmap: sparse-Granger weights are constant over patches; use --figure lag_heatmap")
        if a.figure in ("heatmap", "all") and not sparse:
            print(f"\n== NGC heatmap, {term} ==")
            figure_heatmap(a.datasets, term, device, a.ckpt_root, a.out_dir, a.max_batches)
        if a.figure in ("case_study", "all"):
            for ds in a.datasets:
                print(f"\n== case study {ds}/{term} ==")
                figure_case_study(ds, term, device, a.ckpt_root, a.dlinear_alone_root, a.train_missing, a.out_dir, a.max_batches)


if __name__ == "__main__":
    main()
