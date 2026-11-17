#!/usr/bin/env python3
"""Option-1 redesign check: does a horizon-matched, held-out Granger gain predict how much MAGNet relies on exo?

Per cell (training split only for fitting):
  restricted  predicts the forecast window (next H steps of the target) from the target's own lookback
  full        predicts it from the target's lookback plus every exo variable's lookback (lags 0.. within the window)
  gain        g = 1 - MSE_full / MSE_restricted on the second half of the validation split
              (the first half early-stops the MLPs), i.e. the share of the remaining error that exo explains beyond
              the target's own past, at the actual horizon
Two predictor families: closed-form ridge (alpha by first-half validation) and a 2-layer MLP.
Compared (Spearman, across the 24 cells and within each horizon) with the measured reliance of the trained
sparse_granger_lag0 MAGNet on exo: test-MAE increase when exo is shuffled (graph_usage_summary.json).

    CUDA_VISIBLE_DEVICES="" python3 scripts/granger_gain_check.py      # -> configs/granger_gain_check.json (diagnostic only)
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "configs"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from scipy import stats  # noqa: E402
import lag1_config as C  # noqa: E402
from MAGNET_data import get_dataloaders  # noqa: E402

MAX_TRAIN = 8000
OUT = os.path.join(ROOT, "results", "causal_alt", "granger_gain_check.json")


def arrays(loader, limit=None, seed=0):
    X, E, Y = [], [], []
    for exo_p, endo_p, target, *_ in loader:
        E.append(endo_p.flatten(1))                                  # [B, seq_len]
        X.append(exo_p.permute(0, 3, 1, 2).flatten(1))               # [B, n_exo * seq_len]
        Y.append(target)
    E, X, Y = torch.cat(E).numpy(), torch.cat(X).numpy(), torch.cat(Y).numpy()
    if limit and len(Y) > limit:
        idx = np.random.default_rng(seed).choice(len(Y), limit, replace=False)
        E, X, Y = E[idx], X[idx], Y[idx]
    return E, X, Y


def ridge(Ztr, Ytr, Zva, Yva, Zte):
    best = None
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6
    A, B, T = (Ztr - mu) / sd, (Zva - mu) / sd, (Zte - mu) / sd
    A1 = np.hstack([A, np.ones((len(A), 1))])
    for alpha in (1e-1, 1, 10, 100, 1000):
        W = np.linalg.solve(A1.T @ A1 + alpha * np.eye(A1.shape[1]), A1.T @ Ytr)
        mse = ((np.hstack([B, np.ones((len(B), 1))]) @ W - Yva) ** 2).mean()
        if best is None or mse < best[0]:
            best = (mse, W)
    return np.hstack([T, np.ones((len(T), 1))]) @ best[1]


def mlp(Ztr, Ytr, Zva, Yva, Zte, seed=0):
    torch.manual_seed(seed)
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6
    f = lambda Z: torch.tensor((Z - mu) / sd, dtype=torch.float32)  # noqa: E731
    A, B, T, ya, yb = f(Ztr), f(Zva), f(Zte), torch.tensor(Ytr), torch.tensor(Yva)
    net = nn.Sequential(nn.Linear(A.shape[1], 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, Ytr.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    best, bad, state = float("inf"), 0, None
    for epoch in range(200):
        net.train()
        for i in torch.randperm(len(A)).split(256):
            loss = ((net(A[i]) - ya[i]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = ((net(B) - yb) ** 2).mean().item()
        if v < best - 1e-6:
            best, bad, state = v, 0, {k: x.clone() for k, x in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 10:
                break
    net.load_state_dict(state)
    with torch.no_grad():
        return net(T).numpy()


def main():
    torch.set_num_threads(4)
    usage = {(r["dataset"], r["term"]): r for r in
             json.load(open(os.path.join(ROOT, "results", "causal_alt", "graph_usage_summary.json")))
             if r["method"] == "sparse_granger_lag0"}
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for term in ("short", "long"):
        for ds in C.NAME_TO_ID:
            key = f"{ds}/{term}"
            if key in out:
                continue
            i = C.NAME_TO_ID[ds]
            seq_len, horizon = C.get_seq_horizon(i, term)
            tr, va, _ = get_dataloaders(npz_path=C.csv_path(i), sensor_id=0, seq_len=seq_len,
                                        forecast_horizon=horizon, patch_len=seq_len, batch_size=256, dataset_id=i)
            Etr, Xtr, Ytr = arrays(tr, MAX_TRAIN)
            Eva, Xva, Yva = arrays(va)
            h = len(Yva) // 2                                            # first half: early stop / alpha
            res = {}
            for name, fit in (("ridge", ridge), ("mlp", mlp)):
                pr = fit(Etr, Ytr, Eva[:h], Yva[:h], Eva[h:])
                pf = fit(np.hstack([Etr, Xtr]), Ytr, np.hstack([Eva[:h], Xva[:h]]), Yva[:h],
                         np.hstack([Eva[h:], Xva[h:]]))
                mr, mf = ((pr - Yva[h:]) ** 2).mean(), ((pf - Yva[h:]) ** 2).mean()
                res[name] = {"mse_restricted": float(mr), "mse_full": float(mf), "gain": float(1 - mf / mr)}
            u = usage.get((ds, term), {})
            out[key] = dict(res, exo_shuffle_pct=u.get("exo_shuffle_pct"), graph_off_pct=u.get("graph_off_pct"))
            print(f"{key:14s} ridge gain {res['ridge']['gain']:+.3f}  mlp gain {res['mlp']['gain']:+.3f}  "
                  f"| MAGNet exo-shuffle +{u.get('exo_shuffle_pct', float('nan')):.1f}%", flush=True)
            json.dump(out, open(OUT, "w"), indent=1)

    summary = {}
    for scope in ("all", "short", "long"):
        keys = [k for k in out if k != "summary" and (scope == "all" or k.endswith(scope))]
        rel = [out[k]["exo_shuffle_pct"] for k in keys]
        for name in ("ridge", "mlp"):
            g = [out[k][name]["gain"] for k in keys]
            rho, p = stats.spearmanr(g, rel)
            summary[f"{scope}/{name}"] = {"spearman": float(rho), "p": float(p), "n": len(keys),
                                          "mean_gain": float(np.mean(g))}
            print(f"{scope:5s} {name:5s}: Spearman(gain, exo reliance) = {rho:+.2f} (p={p:.3f}, n={len(keys)}), "
                  f"mean gain {np.mean(g):+.3f}", flush=True)
    out["summary"] = summary
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
