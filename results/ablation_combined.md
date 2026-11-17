# Ablation, real + synthetic combined (17 datasets per term), SIG v2, test, 5-seed ensembles

| Metric | Full | w/o Text | w/o Exo | w/o Graph | w/ Exo MLP | w/ Cross-attn |
|---|---|---|---|---|---|---|
| MAE overall avg | **0.4725** | 0.4769 | 0.4928 | 0.4943 | 0.5268 | 0.4858 |
| MAE short avg | **0.3983** | 0.4071 | 0.4269 | 0.4297 | 0.4398 | 0.4014 |
| MAE long avg | **0.5468** | 0.5468 | 0.5586 | 0.5590 | 0.6138 | 0.5702 |
| MSE overall avg | **0.5286** | 0.5367 | 0.5684 | 0.5624 | 0.5986 | 0.5461 |
| MSE short avg | 0.4090 | 0.4249 | 0.4631 | 0.4611 | 0.4660 | **0.4075** |
| MSE long avg | **0.6481** | 0.6484 | 0.6738 | 0.6637 | 0.7312 | 0.6848 |

Real: ptest_ablation_v2.json; synthetic: reb_results_v3.json (short), reb_results_long.json (long); synthetic Full uses text, synthetic ablation columns are trained without text.
