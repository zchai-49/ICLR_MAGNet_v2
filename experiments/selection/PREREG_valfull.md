# Fully validation-based re-selection (pre-registered 2026-11-10, before any run of this protocol)

Model: MAGNet+DLinear, sparse-input Granger graph (lag 0), detach_gg (detached weights, fixed cross-fit Granger-gain
gate; fold B uses its own gate computed on fold-B data, granger_gain_crossfit_foldB.json). Test data are never read
until the final deployment is fixed; the result is reported whatever it is.

Validation (both horizons, same protocol): lookback border (validation inputs may reach seq_len back into train,
targets stay in the fold).
  fold A: train [0,70%), validate [70%,80%)      fold B: train [0,60%), validate [60%,70%)
Normalized score of a candidate within a comparison set = its val MAE / the median val MAE of the set on that fold.

Mechanism shortlist (union of the old 6-dataset screen's top mechanisms by test and by validation):
  short: crossattn_mq, dual_crossattn, dual_gated_scale_crossattn, dual_gated_scale, dual_only
  long : dual_only, gated_mixing_only, postlinear_film_only, crossattn_mq
Hparam grid (18): moving_avg {13,25,51} x magnet_d_model {32,64,128} x lr {1e-4,5e-4}; default = index 8 (25, 64, 1e-4).

Steps (seed 0 unless stated):
 1 screen   every shortlisted mechanism, 12 datasets, default hparams, folds A and B
 2 mech     per horizon: mean over datasets and folds of the normalized score; the best TWO mechanisms are that
            horizon's two sides for every dataset
 3 hp       per (cell, side): all 18 hparams on fold A; top 3 by fold-A val -> fold B
 4 hp pick  among the top 3: mean over folds of the normalized score (set = the 3)
 5 fix      at the chosen hparams: fix {none, patch8, mlp_backbone, residual_head, deep_film if the mechanism has
            cross-attention} x regime {default, warmup}, fold A; top 3 -> fold B; rank by the 2-fold score
 6 seed     the top 3 (fix, regime) get seed 42 on both folds;
            score = mean over folds of the normalized seed-mean val MAE + 1.0 x sd over seeds {0,42} of the
            normalized fold-A score; lowest wins
 7 sides    deployed = the one of {side1, side2, side1+side2} with the lowest mean over folds of
            (ensemble val MAE of the seed-mean predictions / mean over the three options)
 8 final    5 seeds (0, 42, 123, 2024, 3407) on fold A of the chosen config of each deployed side; test = P1
            ensemble and P2 seed mean, MAE and MSE, vs the 10 baselines.

Amendment (2026-11-10 17:30, before any screen score was examined; 77 fold-A screen runs existed, none inspected):
step 2 uses the 6 representative datasets of the original screen (NP, PJM, Energy, Colbun, Sdwpfh1, Sdwpfm2; all four
domains) instead of all 12, to save compute. Steps 3+ still cover all 12 datasets.

Amendment (2026-11-10 21:30, before any hparam result was ranked; ~30 grid runs existed, none inspected): steps 3-4 use a
9-config grid = the default (index 8) + the 8 configs most often in the validation top-3 of the earlier hparam search
(configs/valsel_hparam_provenance.json): indices 0, 1, 2, 4, 6, 7, 8, 12, 13 of the 18-config grid.
