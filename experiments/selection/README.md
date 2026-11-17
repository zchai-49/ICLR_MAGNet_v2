# Validation-only selection of the MAGNet+DLinear deployment

`configs/deployment_final.json` (which injection mechanisms = ensemble "sides", hyperparameters, enhancer ("fix") and
training regime per dataset/horizon) was chosen by the pre-registered protocol in `PREREG_valfull.md`, using validation
data only (two rolling folds: A = train [0,70%) / val [70,80%); B = train [0,60%) / val [60,70%)). Test data were not read
until the deployment was fixed.

    screen -> mech -> hp (fold A) -> hpB -> hppick -> fix (fold A) -> fixB -> fixpick -> seed -> final

`select_valfull.py <step>` implements each step: it reads the validation results of the previous step and writes the
next job list `valfull_jobs_<step>.json`. A job `[cell, side, hp_rank, fix, regime, seed, fold]` is one training run:

```bash
# fold A                                                     # fold B: prefix MAGNET_FOLD=B MAGNET_GG_FILE=granger_gain_crossfit_foldB.json
MAGNET_DEPLOYMENT=configs/deployment_valfull.json python3 train/train_options.py --option detach_gg \
    --dataset NP --term short --side dual_only --seeds 0 --hp-rank 4 --fix residual_head --regime warmup \
    --out-root experiments/selection/valfull/hp4_residual_head_warmup          # fold B: experiments/selection/valfull_foldB/...
```

(`hparam_candidates` of `configs/deployment_valfull.json` is the 18-config grid; the rank indexes it.) The final step
writes the deployment with the chosen `selected` entry per side.

* `valfull/state.json`: the decisions for the 12 real datasets (step by step, with the validation scores).
* `valfull/state_rebuttal.json`, `valfull/state_synlong.json`: the same protocol for the synthetic datasets (short, long),
  run with `MAGNET_VF_TAG=rebuttal|synlong MAGNET_VF_CELLS=Syn5/short,...`; they start at step `mech` with the mechanisms of
  the real-data screen.
* `PREREG_val2fold_long.md`: the earlier two-fold rule for long horizons that this protocol extends.

Note: the real-data selection ran before lag 0 became a nuisance and pre-whitening was added (both validated to be neutral
on real data, so the selected configurations were kept and only retrained); the synthetic long-horizon selection ran with
the final configuration. The gate files are computed by `scripts/granger_gain_crossfit.py` (fold B: `MAGNET_FOLD=B
MAGNET_GG_OUT=granger_gain_crossfit_foldB.json`).
