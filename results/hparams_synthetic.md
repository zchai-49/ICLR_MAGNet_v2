# Synthetic MAGNet+backbone (PnP) hyperparameters, SIG v2 — template chosen on validation (seeds 0/42) among the selected PnP hparams of Energy / NP / Sdwpfh1 (+ Sdwpfm1 for GCGNet)

## MAGNet+DAG

| Dataset | Term | Template | dag_d_model | dag_e_layers | magnet_d_model | lr |
|---|---|---|---|---|---|---|
| Syn5 | short | NP | 128 | 1 | 64 | 0.0005 |
| Syn20 | short | NP | 128 | 1 | 64 | 0.0005 |
| Syn50 | short | NP | 128 | 1 | 64 | 0.0005 |
| Syn20weak | short | Sdwpfh1 | 128 | 2 | 32 | 0.0005 |
| Syn20null | short | NP | 128 | 1 | 64 | 0.0005 |
| Syn5 | long | NP | 128 | 1 | 32 | 0.0001 |
| Syn20 | long | NP | 128 | 1 | 32 | 0.0001 |
| Syn50 | long | NP | 128 | 1 | 32 | 0.0001 |
| Syn20weak | long | NP | 128 | 1 | 32 | 0.0001 |
| Syn20null | long | Energy | 128 | 1 | 64 | 0.0005 |

## MAGNet+GCGNet

| Dataset | Term | Template | magnet_d_model | gcg_batch_size | gcg_d_ff | gcg_d_model | gcg_dropout | gcg_lr | gcg_lradj | gcg_n_heads | gcg_rank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Syn5 | short | NP | 32 | 64 | 256 | 128 | 0.1 | 0.001 | type3 | 2 | 4 |
| Syn20 | short | Energy | 64 | 64 | 128 | 64 | 0.1 | 0.001 | type3 | 8 | 4 |
| Syn50 | short | Sdwpfm1 | 64 | 64 | 64 | 64 | 0.2 | 0.0001 | type3 | 4 | 4 |
| Syn20weak | short | NP | 32 | 64 | 256 | 128 | 0.1 | 0.001 | type3 | 2 | 4 |
| Syn20null | short | Energy | 64 | 64 | 128 | 64 | 0.1 | 0.001 | type3 | 8 | 4 |
| Syn5 | long | Energy | 32 | 64 | 128 | 64 | 0.1 | 0.001 | type3 | 8 | 4 |
| Syn20 | long | NP | 32 | 64 | 256 | 128 | 0.1 | 0.001 | type3 | 2 | 4 |
| Syn50 | long | Sdwpfh1 | 32 | 32 | 64 | 64 | 0.2 | 0.001 | type3 | 2 | 4 |
| Syn20weak | long | Sdwpfm1 | 128 | 64 | 64 | 64 | 0.2 | 0.0001 | type3 | 4 | 4 |
| Syn20null | long | Sdwpfh1 | 32 | 32 | 64 | 64 | 0.2 | 0.001 | type3 | 2 | 4 |

## MAGNet+TimeXer

| Dataset | Term | Template | tx_d_model | tx_e_layers | magnet_d_model | lr |
|---|---|---|---|---|---|---|
| Syn5 | short | Sdwpfh1 | 256 | 2 | 128 | 0.0001 |
| Syn20 | short | Sdwpfh1 | 256 | 2 | 128 | 0.0001 |
| Syn50 | short | Sdwpfh1 | 256 | 2 | 128 | 0.0001 |
| Syn20weak | short | Sdwpfh1 | 256 | 2 | 128 | 0.0001 |
| Syn20null | short | NP | 512 | 2 | 128 | 0.0005 |
| Syn5 | long | Sdwpfh1 | 512 | 3 | 128 | 0.0005 |
| Syn20 | long | Sdwpfh1 | 512 | 3 | 128 | 0.0005 |
| Syn50 | long | Energy | 256 | 2 | 32 | 0.0001 |
| Syn20weak | long | Sdwpfh1 | 512 | 3 | 128 | 0.0005 |
| Syn20null | long | Sdwpfh1 | 512 | 3 | 128 | 0.0005 |

## MAGNet+TimeKAN

| Dataset | Term | Template | tk_d_model | tk_e_layers | lr | magnet_d_model | magnet_heads |
|---|---|---|---|---|---|---|---|
| Syn5 | short | Energy | 64 | 1 | 0.0005 | 64 | 4 |
| Syn20 | short | Energy | 64 | 1 | 0.0005 | 64 | 4 |
| Syn50 | short | Energy | 64 | 1 | 0.0005 | 64 | 4 |
| Syn20weak | short | NP | 128 | 1 | 0.0005 | 64 | 8 |
| Syn20null | short | Sdwpfh1 | 32 | 1 | 0.0005 | 128 | 8 |
| Syn5 | long | Sdwpfh1 | 64 | 2 | 0.0005 | 64 | 4 |
| Syn20 | long | NP | 128 | 1 | 0.0001 | 64 | 4 |
| Syn50 | long | NP | 128 | 1 | 0.0001 | 64 | 4 |
| Syn20weak | long | NP | 128 | 1 | 0.0001 | 64 | 4 |
| Syn20null | long | NP | 128 | 1 | 0.0001 | 64 | 4 |

## MAGNet+PatchTST

| Dataset | Term | Template | ptst_d_model | ptst_e_layers | lr | magnet_d_model | magnet_heads |
|---|---|---|---|---|---|---|---|
| Syn5 | short | NP | 64 | 2 | 0.0005 | 64 | 8 |
| Syn20 | short | NP | 64 | 2 | 0.0005 | 64 | 8 |
| Syn50 | short | NP | 64 | 2 | 0.0005 | 64 | 8 |
| Syn20weak | short | NP | 64 | 2 | 0.0005 | 64 | 8 |
| Syn20null | short | Energy | 256 | 3 | 0.0001 | 64 | 4 |
| Syn5 | long | Sdwpfh1 | 256 | 2 | 0.0005 | 64 | 4 |
| Syn20 | long | Energy | 64 | 2 | 0.0001 | 64 | 4 |
| Syn50 | long | Sdwpfh1 | 256 | 2 | 0.0005 | 64 | 4 |
| Syn20weak | long | Energy | 64 | 2 | 0.0001 | 64 | 4 |
| Syn20null | long | Energy | 64 | 2 | 0.0001 | 64 | 4 |

## MAGNet+TFT

| Dataset | Term | Template | tft_d_model | lr | magnet_d_model | magnet_heads |
|---|---|---|---|---|---|---|
| Syn5 | short | NP | 32 | 0.0005 | 64 | 8 |
| Syn20 | short | NP | 32 | 0.0005 | 64 | 8 |
| Syn50 | short | NP | 32 | 0.0005 | 64 | 8 |
| Syn20weak | short | Energy | 64 | 0.0005 | 64 | 4 |
| Syn20null | short | NP | 32 | 0.0005 | 64 | 8 |
| Syn5 | long | NP | 64 | 0.0005 | 64 | 4 |
| Syn20 | long | Sdwpfh1 | 32 | 0.0005 | 64 | 4 |
| Syn50 | long | NP | 64 | 0.0005 | 64 | 4 |
| Syn20weak | long | Sdwpfh1 | 32 | 0.0005 | 64 | 4 |
| Syn20null | long | NP | 64 | 0.0005 | 64 | 4 |

## MAGNet+DLinear (main model; validation-selected per side, 5-seed ensemble over the deployed sides)

| Dataset | Term | Side | Enhancer | Regime | hp rank | moving_avg | MAGNet d_model | lr | patch_len |
|---|---|---|---|---|---|---|---|---|---|
| Syn5 | short | dual_gated_scale | residual_head | warmup | 1 | 13 | 32 | 0.0005 | 12 |
| Syn5 | short | dual_only | none | warmup | 0 | 13 | 32 | 0.0001 | 12 |
| Syn20 | short | dual_gated_scale | residual_head | default | 4 | 13 | 128 | 0.0001 | 12 |
| Syn20 | short | dual_only | residual_head | default | 4 | 13 | 128 | 0.0001 | 12 |
| Syn50 | short | dual_gated_scale | residual_head | warmup | 4 | 13 | 128 | 0.0001 | 12 |
| Syn50 | short | dual_only | patch8 | warmup | 4 | 13 | 128 | 0.0001 | 8 |
| Syn20weak | short | dual_gated_scale | patch8 | default | 1 | 13 | 32 | 0.0005 | 8 |
| Syn20weak | short | dual_only | patch8 | warmup | 0 | 13 | 32 | 0.0001 | 8 |
| Syn20null | short | dual_only | patch8 | default | 0 | 13 | 32 | 0.0001 | 8 |
| Syn5 | long | dual_only | mlp_backbone | warmup | 4 | 13 | 128 | 0.0001 | 12 |
| Syn5 | long | crossattn_mq | mlp_backbone | default | 4 | 13 | 128 | 0.0001 | 12 |
| Syn20 | long | dual_only | mlp_backbone | default | 2 | 13 | 64 | 0.0001 | 12 |
| Syn20 | long | crossattn_mq | mlp_backbone | default | 4 | 13 | 128 | 0.0001 | 12 |
| Syn50 | long | dual_only | mlp_backbone | warmup | 4 | 13 | 128 | 0.0001 | 12 |
| Syn50 | long | crossattn_mq | deep_film | default | 4 | 13 | 128 | 0.0001 | 12 |
| Syn20weak | long | dual_only | none | default | 4 | 13 | 128 | 0.0001 | 12 |
| Syn20weak | long | crossattn_mq | deep_film | default | 4 | 13 | 128 | 0.0001 | 12 |
| Syn20null | long | crossattn_mq | mlp_backbone | warmup | 0 | 13 | 32 | 0.0001 | 12 |
