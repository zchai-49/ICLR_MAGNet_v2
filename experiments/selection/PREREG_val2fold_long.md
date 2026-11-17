# Pre-registered long-horizon selection rule (fixed 2026-11-10, before any fold-B result exists)

Applies to the 12 long-horizon cells of the detach_gg deployment. Short horizons keep the existing val selection.

Data
- fold A: existing checkpoints (train [0,70%)), validation [70%,80%) with the lookback border
  (validation inputs may start seq_len steps before 70%; targets stay inside the fold).
- fold B: new checkpoints (train [0,60%)), validation [60%,70%), lookback border.
- Test data are never read during selection.

Per side s and candidate config c (hp_rank, fix, regime):
- A(c) = mean fold-A val MAE over seeds {0, 42} available for c; B(c) = fold-B val MAE (seed 0).
- nA(c) = A(c) / median_c' A(c'), nB(c) = B(c) / median_c' B(c')     (medians over candidates of that cell-side)
- sd(c) = std (ddof=1) of the fold-A normalized val MAE over seeds {0, 42} if c has both; otherwise the median sd over
  the candidates of that cell-side that have both (0 if none).
- score(c) = (nA(c) + nB(c)) / 2 + 1.0 * sd(c);  choose argmin. Candidates need fold A seed 0 and fold B seed 0.

Deployed sides: among {side1, side2, side1+side2}, the one with the smallest mean over folds of
(ensemble val MAE / that fold's mean over the three options), using seed-mean predictions of the chosen configs.

Then: 5 seeds (0, 42, 123, 2024, 3407) of the chosen configs on fold A; test = P1 ensemble and P2 seed mean vs the
10 baselines, reported whatever the outcome. Known caveat: the detach_gg gate value per cell was computed from fold-A
validation, which includes fold B's validation period.
