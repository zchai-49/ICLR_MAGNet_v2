# MAGNet: official code (final version)

**MAGNet** is a causal-graph module for forecasting a target series from exogenous variables. It builds patch-level
nodes for the target and every exogenous variable, propagates information between exogenous variables with a graph
attention network, and lets the target attend to them. The attention is biased by a learned Granger-causal graph and by
LLM-generated text about each variable. MAGNet is used:

* with a **DLinear** backbone (the default model, "MAGNet" in the tables), shipped with **all final checkpoints**;
* as a **plug-and-play (PnP)** module on **DAG, GCGNet, TimeXer, TimeKAN, PatchTST and TFT** (code + selected
  hyperparameters, no checkpoints).

Final configuration (what this package runs by default, see `models/final_defaults.py`):

| Component | Final choice |
|---|---|
| Causal graph | **Sparse-input Granger (SIG)** estimator: component-wise MLP on the raw patches with a group-lasso penalty per (variable, lag); K = 3 lags (patches), λ = 0.01 |
| Lag 0 | fitted and penalised as a **nuisance** control (absorbs same-patch co-movement) but **not** used for the attention weights; only the lagged groups set them |
| Pre-whitening | SIG inputs are **seasonally differenced** (s = 24 hourly, 48 for the 30-min Sdwpfm data, 7 for daily Colbun/Rapel) |
| Graph training | weights **detached** (the estimator learns only from its own Granger loss) |
| Gate | the exogenous message is scaled by a **fixed cross-fit Granger gain** per dataset/horizon (`configs/granger_gain_crossfit.json`, computed on train/validation only) |
| Text | Llama-3.2-3B-Instruct answers fixed questions per variable, conditioned on its description and training-split statistics; BERT [CLS] embeddings |
| MAGNet+DLinear | validation-selected ensemble of 1–2 injection mechanisms ("sides") x 5 seeds per dataset/horizon (`configs/deployment_final.json`) |

Datasets: **12 real** — EPF electricity prices (BE, DE, FR, NP, PJM), Energy, Hydro (Colbun, Rapel), Wind (Sdwpfh1,
Sdwpfh2, Sdwpfm1, Sdwpfm2) — and **5 synthetic** datasets with a known dependency structure (Syn5, Syn20, Syn50,
Syn20weak, Syn20null), each with a short (168 → 24; hydro 60 → 10) and a long (720 → 360; hydro 180 → 30) horizon.
**No future exogenous values** are ever given to a model (history-only forecasting).

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python3 tests/test_invariants.py        # final configuration active, all 285 checkpoints load, history-only input
```

Tested with Python 3.10, torch 2.11 (CUDA 13), numpy 2.2, pandas 2.3, scikit-learn 1.7. Every script runs on CPU or GPU
(`--gpu N`). No environment variable is needed: `models/final_defaults.py` sets the final configuration as defaults, and
exporting a `MAGNET_*` variable overrides it (this is how the ablations / controls run).

## Layout

```
models/        MAGNET_graph_only.py      MAGNet module (exo/endo patch nodes, exo-exo GAT, causal attention, text bias)
               causal_alternatives.py    SparseInputGranger (SIG, final), GVAR (alternative), lag-0 nuisance
               exo_graph.py endo_insertion.py text_projection.py controls.py (Exo-MLP / cross-attention controls)
               MAGNET_with_DLinear_fusion.py   MAGNet+DLinear (all injection mechanisms / enhancers)
               MAGNET_with_{DAG,GCGNet,TimeXer,TimeKAN,PatchTST,TFT}.py   MAGNet as a PnP module (mode 2) / vanilla (mode 3)
               MAGNET_graph_only_ablation.py   module ablations
               text_generate.py          text generation + BERT embeddings (real datasets)
               text_generate_generic.py  same for the synthetic datasets (statistics-only prompts)
               text_generate_variants.py text_encoder_variants.py text_prompt_variants.py   text ablations
               final_defaults.py         the official configuration
backbones/     vendored TimeXer, Time-Series-Library (TFT), DAG / GCGNet code used by the wrappers
pnp/           per-backbone PnP trainers (train_one_config) and the original hyperparameter searches
data/          MAGNET_data.py (loading, scaling, patching; history-only batches)
               dataset/   12 real CSVs + variable descriptors; generic/Syn*.csv + Syn*_truth.json (ground-truth roles)
               text_emb/  generated text (*_new_text_log.json) and BERT embeddings (*_vq_emb.pt) for all 17 datasets
configs/       lag1_config.py            datasets, horizons, model builder, deployment loader
               deployment_final.json     final MAGNet+DLinear deployment (34 cells: sides, hparams, enhancer, regime)
               granger_gain_crossfit.json (+ _foldB)   cross-fit Granger gain per cell (gate)
               pnp_selected_hparams.json selected PnP hparams per backbone x real dataset x term
               pnp_syn_selected.json     synthetic PnP: template chosen on validation per backbone x dataset x term
               dlinear_alone_hparams.json   DLinear without MAGNet (validation-selected)
               deployment_valfull.json valsel_hparam_provenance.json   candidate grid of the selection protocol
               ablation.py               ablation model builders
train/         train_magnet.py           train the final MAGNet+DLinear (all deployed sides x seeds of a cell)
               train_options.py          same trainer with ablation / control / text options (train_magnet.py wraps it)
               trainer.py                shared training loop, losses, optimiser and schedules
               run_pnp.py                train + evaluate MAGNet+<backbone> with the selected hyperparameters
               train_dlinear_alone.py dlinear_alone_valsel.py   DLinear without MAGNet
scripts/       evaluate_magnet.py        evaluate MAGNet+DLinear checkpoints (5-seed ensemble) vs the reported numbers
               evaluate.py               shared prediction / metric helpers
               make_synthetic.py         generates the 5 synthetic datasets
               granger_gain_crossfit.py  computes the gate file (+ granger_gain_check.py helpers)
experiments/   one folder per table: run.sh (training commands) + an evaluation / table script (see below)
visualization/ case_study.py (FR case-study figure), make_figures.py, plot_utils.py
checkpoints/   magnet_dlinear/<term>_<dataset>_<side>/seed_<seed>/ckpt.pt (+ result.json): ALL final checkpoints
results/       the final tables we report (markdown + json), see below
tests/         test_invariants.py
```

## 1. MAGNet+DLinear (shipped checkpoints)

All 285 final checkpoints ship (17 datasets x 2 horizons, every deployed side x seeds 0, 42, 123, 2024, 3407). A reported
number is the prediction ensemble: average over the 5 seeds of each side, then over the deployed sides.

```bash
python3 scripts/evaluate_magnet.py                              # all 34 cells -> results/evaluation.json
python3 scripts/evaluate_magnet.py --datasets FR Syn20 --terms short --gpu 0
```

Each line prints the test MAE/MSE next to the reported number (`results/table1_main.json`); e.g. FR short 0.2069 / 0.3647,
FR long 0.3186 / 0.5922 (identical to the paper). To retrain a cell from scratch (existing checkpoints are skipped, so
point `--out-root` elsewhere to retrain):

```bash
python3 train/train_magnet.py --dataset FR --term short --gpu 0 --out-root runs/retrain     # deployed sides x 5 seeds
python3 scripts/evaluate_magnet.py --datasets FR --terms short --ckpt-root runs/retrain
```

Training: AdamW (weight decay 1e-3), cosine schedule, up to 200 epochs, early stopping on validation MAE (patience 15),
batch size 32; L1 loss (short) / Smooth-L1 β = 0.5 (long); magnitude-weighted L1 on the wind datasets; Colbun long on first
differences; regime `warmup` = 30-epoch attention warm-up with gate learning rate x50. Per-seed results vary slightly
across GPUs / library versions; compare 5-seed ensembles.

## 2. Synthetic datasets

`python3 scripts/make_synthetic.py` regenerates `data/dataset/generic/Syn{5,20,50,20weak,20null}.{csv,json}` and
`*_truth.json` (hourly, 20 000 steps, split 70/10/20). The target is
OT_t = 0.6·OT_{t-1} + 0.3·sin(2πt/24) + β·[tanh(X1_{t-24}) + 0.5(X2²_{t-30} − 1) + X3_{t-36}] + noise, with three direct
parents X1–X3, a chain variable X4 → X1, proxies of the parents (noisy copies) and irrelevant AR(1)+seasonal variables;
β sets the exogenous share of OT's variance (50 %; 20 % for Syn20weak; 0 for Syn20null). Their text is generated from
statistics-only prompts: `python3 models/text_generate_generic.py --name Syn20 --folder synthetic/Syn20`.

## 3. Reproducing the experiments (code only, no checkpoints)

Training outputs go to `runs/`. Every `run.sh` is a plain list of independent training commands (`GPU=<id>`); run them in
parallel as you like, then the evaluation script builds the table into `results/*_reproduced.md`.

| Experiment | Train | Table |
|---|---|---|
| **Table 1** main results (MAGNet column) | shipped checkpoints / `train/train_magnet.py` | `scripts/evaluate_magnet.py` |
| **Ablation** (Full, w/o Text, w/o Exo, w/o Graph, w/ Exo MLP, w/ Cross-attn; 17 datasets) | `experiments/ablation/run.sh` | `experiments/ablation/evaluate.py` |
| **Table 2** MAGNet as PnP on 6 backbones (17 datasets x 5 seeds) | `experiments/table2_pnp/run.sh` | `experiments/table2_pnp/build_table.py` |
| **Test 1** branch shuffle (is the MAGNet signal used?) | recorded by the Table 2 runs (`MAGNET_PNP_SHUFFLE=1`); DLinear: `experiments/test1_branch_shuffle/dlinear_shuffle.py` | `experiments/test1_branch_shuffle/summarize.py` |
| **Text ablation** (encoder, generator, prompt) | `experiments/text_ablation/run.sh` | `experiments/text_ablation/evaluate.py` |
| **Selection protocol** that produced `deployment_final.json` | `experiments/selection/` (see its README) | — |
| **Case-study figure** (FR) | — | `python3 visualization/case_study.py` → `figures/case_study/` |

**PnP (Table 2).** `python3 train/run_pnp.py --backbone TFT --dataset Syn20 --term long --seeds 0 42` trains and
evaluates MAGNet+TFT with the selected hyperparameters: real datasets from `configs/pnp_selected_hparams.json`; synthetic
datasets borrow the selected PnP hyperparameters of the template dataset (Energy / NP / Sdwpfh1, + Sdwpfm1 for GCGNet)
chosen on validation (`configs/pnp_syn_selected.json`, full list in `results/hparams_synthetic.md`). Each seed's result is
printed next to our run of the same cell (`results/table2_pnp_per_seed.json`). Some MAGNet+GCGNet synthetic seeds diverge
at learning rate 1e-3; Table 2 averages the converged seeds and marks the count.

**Test 1.** Test MAE/MSE with the MAGNet branch's exogenous input replaced by that of a random other test window (the
backbone keeps the true exogenous series). A rise means the backbone uses the MAGNet signal. Cells whose cross-fit gate is
≤ 0 have the branch switched off (change exactly 0).

**Ablation details.** Each column keeps the cell's deployed sides, hyperparameters, enhancers and regime and changes one
module (`train/train_options.py` lists every option). The synthetic ablation columns are trained without text
(`MAGNET_SYN_TEXT=0`), as reported.

**Baselines.** The vanilla baselines are not part of this package: DAG, GCGNet, PatchTST, TimeXer, TimeKAN, TFT, T3Time,
TimeCMA and Time-LLM are available in their own official repositories, which we used to train them. Their test numbers are
in `results/table1_main.*` (real datasets: 5-seed means; synthetic non-LLM baselines: the validation-chosen candidate among
the Energy / NP / Sdwpfh1 hyperparameters, 5-seed mean; the LLM baselines on synthetic data: seed 0). DLinear without MAGNet
is included (`train/dlinear_alone_valsel.py`, `train/train_dlinear_alone.py`), since DLinear is our default backbone.

## 4. Final results (in `results/`)

| File | Content |
|---|---|
| `table1_main.md` / `.json` | Table 1: MAGNet vs 10 baselines, 12 real + 5 synthetic datasets, MAE and MSE, averages |
| `table2_pnp.md` / `.json`, `table2_pnp_per_seed.json` | Table 2: vanilla vs +MAGNet for 7 backbones, per dataset and averages |
| `ablation_combined.md`, `ablation_real_cells.json`, `synthetic_{short,long}.json` | ablation columns (real + synthetic) |
| `test1_branch_shuffle_combined.md`, `test1_branch_shuffle.md` | Test 1 |
| `text_ablation.md` / `.json` | text encoder / generator / prompt ablations |
| `hparams_synthetic.md` | selected hyperparameters of every MAGNet+backbone on the synthetic datasets |
| `real_main_cells.json` | per-cell MAGNet ensemble numbers on the real datasets |

Overall test averages over all 17 datasets (Table 1): MAGNet MAE 0.3983 (short) / 0.5468 (long) / 0.4725 (both),
MSE 0.4090 / 0.6481 / 0.5286 — the lowest of all 11 models.

## 5. Text

`data/text_emb/<dataset>/<dataset>_new_text_log.json` holds the generated text and `*_new_{short,long}_{endo,exo}_vq_emb.pt`
its BERT embeddings (shipped for all 17 datasets). `python3 models/text_generate.py` (real datasets) and
`models/text_generate_generic.py` (synthetic) regenerate them; they need `transformers`, `bitsandbytes`, `accelerate` and
access to `meta-llama/Llama-3.2-3B-Instruct` (gated: accept the license, `huggingface-cli login`). Generation is greedy, but
4-bit kernels are not bit-identical across GPUs.

## Protocol notes

* Splits are chronological 70 / 10 / 20 (train / validation / test). Data are standardised with training-split statistics;
  metrics are on the standardised scale, target channel only.
* All selection (deployment sides, hyperparameters, enhancers, synthetic PnP templates, gates) used validation data only;
  test numbers were never used to choose anything.
* Test-time averaging over two extra noise-perturbed passes on NP and the four wind datasets.
* Third-party code in `backbones/`: TimeXer, Time-Series-Library (TFT blocks; MIT) and the DAG / GCGNet reference
  implementations (MIT, `backbones/ts_benchmark/LICENSE_DAG_GCGNet`). PatchTST and TimeKAN are compact re-implementations
  inside their wrappers.
