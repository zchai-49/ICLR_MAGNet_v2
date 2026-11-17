#!/usr/bin/env python3
"""Synthetic exogenous-forecasting benchmark with a KNOWN dependency structure (rebuttal, 2026-11-12).

Hourly series, T = 20000 steps. N exogenous variables X1..XN (AR(1) + daily seasonality + noise), one target OT:
    OT_t = 0.6 OT_{t-1} + 0.3 sin(2 pi t / 24) + beta * [ tanh(X1_{t-24}) + 0.5 (X2_{t-30}^2 - 1) + X3_{t-36} ] + sigma e_t
  direct parents      X1, X2, X3 at lags 24 / 30 / 36 (>= the 24-step horizon, so the exo signal is usable)
  indirect (chain)    X4 -> X1 (X1_t += 0.7 X4_{t-12}): X4 Granger-causes OT only THROUGH X1 (exo-exo structure)
  proxies             max(1, N // 5) variables = a direct parent + noise (corr ~0.95 with it, no own effect on OT):
                      marginally predictive, not conditionally (Granger) once the parent is known
  irrelevant          the rest: independent AR(1) + seasonality
beta is set so the exogenous term explains a fraction `share` of OT's variance (after the AR filter); share = 0 gives
the null dataset (no exogenous variable matters; the cross-fit gate should be 0).

Writes data/dataset/generic/<name>.csv (wide: date, OT, X1..XN), <name>.json (endo/exo metadata for the loader) and
<name>_truth.json (roles of every variable).
    python3 scripts/make_synthetic.py
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "dataset", "generic")
T = 20000
SETS = {"Syn5": (5, 0.5), "Syn20": (20, 0.5), "Syn50": (50, 0.5), "Syn20weak": (20, 0.2), "Syn20null": (20, 0.0)}


def ar_filter(u, phi=0.6):
    y = np.zeros_like(u)
    for t in range(1, len(u)):
        y[t] = phi * y[t - 1] + u[t]
    return y


def make(name, n, share, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(T + 200)
    X = np.zeros((len(t), n))
    for j in range(n):
        phi, amp, ph = rng.uniform(0.8, 0.95), rng.uniform(0.3, 1.0), rng.uniform(0, 2 * np.pi)
        e = rng.normal(size=len(t))
        for k in range(1, len(t)):
            X[k, j] = phi * X[k - 1, j] + 0.4 * e[k]
        X[:, j] += amp * np.sin(2 * np.pi * t / 24 + ph)
    X[12:, 0] += 0.7 * X[:-12, 3]                                  # chain X4 -> X1
    X = (X - X.mean(0)) / X.std(0)
    n_proxy = max(1, n // 5)
    proxies = list(range(4, 4 + n_proxy))
    for i, j in enumerate(proxies):                                # proxies of the direct parents X1, X2, X3
        X[:, j] = X[:, i % 3] + 0.3 * rng.normal(size=len(t))
    lag = {0: 24, 1: 30, 2: 36}
    ex = np.zeros(len(t))
    ex[24:] += np.tanh(X[:-24, 0]); ex[30:] += 0.5 * (X[:-30, 1] ** 2 - 1); ex[36:] += X[:-36, 2]
    base = ar_filter(0.3 * np.sin(2 * np.pi * t / 24) + 0.5 * rng.normal(size=len(t)))
    exf = ar_filter(ex)
    beta = 0.0 if share == 0 else np.sqrt(share / (1 - share) * base[200:].var() / exf[200:].var())
    y = base + beta * exf
    X, y = X[200:], y[200:]
    os.makedirs(OUT, exist_ok=True)
    cols = ["OT"] + [f"X{j + 1}" for j in range(n)]
    df = pd.DataFrame(np.column_stack([y, X]), columns=cols)
    df.insert(0, "date", pd.date_range("2020-01-01", periods=T, freq="h").astype(str))
    df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
    role = {j: "irrelevant" for j in range(n)}
    role.update({0: "parent", 1: "parent", 2: "parent", 3: "chain (X4 -> X1)"})
    role.update({j: f"proxy of X{i % 3 + 1}" for i, j in enumerate(proxies)})
    info = {"context": "synthetic", "endo": [{"name": "OT", "index": 0, "type": "endo", "description": "target"}],
            "exo": [{"name": f"X{j + 1}", "index": j + 1, "type": "exo", "description": "exogenous"} for j in range(n)]}
    json.dump(info, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1)
    truth = {"n_exo": n, "share": share, "beta": float(beta), "seed": seed, "lags": {f"X{j + 1}": l for j, l in lag.items()},
             "roles": {f"X{j + 1}": r for j, r in role.items()}, "direct_parents": ["X1", "X2", "X3"]}
    json.dump(truth, open(os.path.join(OUT, f"{name}_truth.json"), "w"), indent=1)
    c = [abs(np.corrcoef(y[36:], X[:-36, j])[0, 1]) for j in range(n)]
    print(f"{name}: n_exo {n}, share {share}, beta {beta:.3f}, |corr(OT_t, X_t-36)| parents "
          f"{np.round(c[:3], 2)}, chain {c[3]:.2f}, proxies {np.round([c[j] for j in proxies[:3]], 2)}")


if __name__ == "__main__":
    for i, (name, (n, share)) in enumerate(SETS.items()):
        make(name, n, share, seed=100 + i)
