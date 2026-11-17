#!/bin/bash
# Ablation columns (Table: Full | w/o Text | w/o Exo | w/o Graph | w/ Exo MLP | w/ Cross-attn), 12 real + 5 synthetic datasets
# x short/long x 5 seeds, each with the cell's deployed sides / hyperparameters / enhancers (only the module changes):
#   no_text        MAGNet without text                     dlinear_only   the DLinear backbone + enhancers alone (no graph)
#   no_exo         no exogenous input                       exo_mlp        dlinear_only + a 2-layer exo MLP (matched size)
#   xattn_dlinear  dlinear_only + one plain cross-attention block (no graph, no Granger weights, no gate, no text)
# Full = the shipped checkpoints (checkpoints/magnet_dlinear). The synthetic ablation columns are trained WITHOUT text
# (MAGNET_SYN_TEXT=0), as in the paper (synthetic Full uses text). Usage: GPU=0 bash experiments/ablation/run.sh
cd "$(dirname "$0")/../.."
GPU=${GPU:-0}
for opt in no_text no_exo dlinear_only exo_mlp xattn_dlinear; do
  for term in short long; do
    for ds in BE DE FR NP PJM Energy Colbun Rapel Sdwpfh1 Sdwpfh2 Sdwpfm1 Sdwpfm2; do
      python3 train/train_magnet.py --option $opt --dataset $ds --term $term --gpu $GPU --out-root runs/ablation/$opt
    done
    for ds in Syn5 Syn20 Syn50 Syn20weak Syn20null; do
      MAGNET_SYN_TEXT=0 python3 train/train_magnet.py --option $opt --dataset $ds --term $term --gpu $GPU --out-root runs/ablation/$opt
    done
  done
done
