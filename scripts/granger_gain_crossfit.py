#!/usr/bin/env python3
"""Final Granger-gain protocol (fixed 2026-11-09, BEFORE any test result of the detach_gg deployment):
cross-fitted, blocked, linear + nonlinear, held-out horizon-matched gain.

Per cell (train split fits every predictor; validation is only used to choose and to measure):
  candidates   restricted = target's own lookback, full = restricted + exo features, for
                 ridge  x {all exo lags, last 24 steps, 12-step patch means, each single exo variable's patch means}
                 GBM    x {last 24 steps, patch means}  (HistGradientBoosting, fixed settings, predicting the first
                        8 principal components of the forecast window; inputs = target patch means + last 24 steps)
  cross-fit    validation split into K = 4 contiguous blocks. For block k: on the OTHER blocks (minus a gap of
               horizon windows next to block k, so no forecast target overlaps; no gap if < 50 windows would remain) choose each ridge alpha and then the candidate with the best
               gain; measure that candidate's restricted and full MSE on block k
  gain         g = 1 - sum_k MSE_full,k / sum_k MSE_restricted,k   (pooled over the 4 held-out blocks)
  uncertainty  se = std of the 4 per-block gains / sqrt(4); also reported: per-block gains, the chosen candidates
  gate         clip(g, 0, 1)                     (primary, point estimate)
               clip(g - se, 0, 1)                (reported as the shrunk alternative; c = 1 fixed in advance)

    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" python3 scripts/granger_gain_crossfit.py
        -> configs/granger_gain_crossfit.json  ([cell]["ridge"]["gain"] = g, so train_options can read it)
"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "configs")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
import lag1_config as C
from MAGNET_data import get_dataloaders
from granger_gain_check import arrays, MAX_TRAIN

OUT = os.path.join(ROOT, "configs", os.environ.get("MAGNET_GG_OUT", "granger_gain_crossfit.json"))   # fold B: MAGNET_FOLD=B MAGNET_GG_OUT=granger_gain_crossfit_foldB.json
ALPHAS = (1e-1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6)
K = 4


def ridge_path(Ztr, Ytr, Zva):
    """Validation predictions for every alpha, from one eigendecomposition. -> [n_alpha, n_val, H]"""
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6
    A = (Ztr - mu) / sd; B = (Zva - mu) / sd
    ym = Ytr.mean(0)
    s, V = np.linalg.eigh(A.T @ A)
    R = V.T @ (A.T @ (Ytr - ym))
    BV = B @ V
    return np.stack([BV @ (R / (s + a)[:, None]) + ym for a in ALPHAS])


def gbm_pred(Ztr, Ytr, Zva, seed=0):
    k = min(8, Ytr.shape[1])
    pca = PCA(k, random_state=seed).fit(Ytr)
    C_tr = pca.transform(Ytr)
    out = np.zeros((len(Zva), k))
    for j in range(k):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                          early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
                                          random_state=seed)
        out[:, j] = m.fit(Ztr, C_tr[:, j]).predict(Zva)
    return pca.inverse_transform(out)[None]                     # [1, n_val, H]


def main():
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for term in ("short", "long"):
        for ds in C.NAME_TO_ID:
            key = f"{ds}/{term}"
            if key in out:
                continue
            i = C.NAME_TO_ID[ds]
            L, H = C.get_seq_horizon(i, term)
            tr, va, _ = get_dataloaders(npz_path=C.csv_path(i), sensor_id=0, seq_len=L, forecast_horizon=H,
                                        patch_len=L, batch_size=256, dataset_id=i)
            Etr, Xtr, Ytr = arrays(tr, MAX_TRAIN); Eva, Xva, Yva = arrays(va)        # validation stays in time order
            n = Xtr.shape[1] // L; P = 12 if L % 12 == 0 else 6; Kl = min(24, L)
            X3tr, X3va = Xtr.reshape(len(Xtr), n, L), Xva.reshape(len(Xva), n, L)
            pm = lambda X3: X3.reshape(len(X3), X3.shape[1], -1, P).mean(-1)         # noqa: E731
            exo = {"full": (Xtr, Xva),
                   "lastK": (X3tr[:, :, -Kl:].reshape(len(Xtr), -1), X3va[:, :, -Kl:].reshape(len(Xva), -1)),
                   "patchmean": (pm(X3tr).reshape(len(Xtr), -1), pm(X3va).reshape(len(Xva), -1))}
            for j in range(n):
                exo[f"var{j}"] = (pm(X3tr)[:, j], pm(X3va)[:, j])
            # validation predictions of every candidate: {name: (restricted [A, n, H], full [A, n, H])}
            preds = {}
            r_lin = ridge_path(Etr, Ytr, Eva)
            for k, (a, b) in exo.items():
                preds[f"ridge/{k}"] = (r_lin, ridge_path(np.hstack([Etr, a]), Ytr, np.hstack([Eva, b])))
            Etr_g = np.hstack([pm(Etr[:, None])[:, 0], Etr[:, -Kl:]]); Eva_g = np.hstack([pm(Eva[:, None])[:, 0], Eva[:, -Kl:]])
            r_gbm = gbm_pred(Etr_g, Ytr, Eva_g)
            for k in ("lastK", "patchmean"):
                a, b = exo[k]
                preds[f"gbm/{k}"] = (r_gbm, gbm_pred(np.hstack([Etr_g, a]), Ytr, np.hstack([Eva_g, b])))
            # squared errors per window, averaged over the horizon: {name: (restricted [A, n], full [A, n])}
            se = {c: (((r - Yva) ** 2).mean(-1), ((f - Yva) ** 2).mean(-1)) for c, (r, f) in preds.items()}

            N = len(Yva); blocks = np.array_split(np.arange(N), K); gap = H   # forecast targets of windows < H apart overlap
            fold, num, den, picks = [], 0.0, 0.0, []
            for b in blocks:
                sel = np.ones(N, bool); sel[max(b[0] - gap, 0): b[-1] + gap + 1] = False
                if sel.sum() < 50:                                   # tiny validation split: drop the gap
                    sel = np.ones(N, bool); sel[b] = False
                best = None
                for c, (er, ef) in se.items():
                    ar, af = er[:, sel].mean(1).argmin(), ef[:, sel].mean(1).argmin()     # alpha per model
                    g_sel = 1 - ef[af, sel].mean() / er[ar, sel].mean()
                    if best is None or g_sel > best[0]:
                        best = (g_sel, c, ar, af)
                _, c, ar, af = best
                er, ef = se[c]
                mr, mf = er[ar, b].mean(), ef[af, b].mean()
                fold.append(float(1 - mf / mr)); num += mf * len(b); den += mr * len(b); picks.append(c)
            g = float(1 - num / den); s = float(np.std(fold, ddof=1) / np.sqrt(K))
            out[key] = {"ridge": {"gain": g}, "gain": g, "se": s, "fold_gains": fold, "picks": picks,
                        "gate": min(max(g, 0.0), 1.0), "gate_shrunk": min(max(g - s, 0.0), 1.0), "n_val": N, "gap": gap}
            print(f"{key:14s} gain {g:+.3f} ± {s:.3f}  gate {out[key]['gate']:.3f}  shrunk {out[key]['gate_shrunk']:.3f}"
                  f"  folds {' '.join(f'{x:+.3f}' for x in fold)}  picks {','.join(picks)}", flush=True)
            json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
