#!/bin/bash
# Table 2: MAGNet as a plug-and-play module on DAG, GCGNet, TimeXer, TimeKAN, PatchTST, TFT; 12 real + 5 synthetic datasets
# x short/long x seeds 0 42 123 2024 3407, with the selected hyperparameters (configs/pnp_selected_hparams.json, synthetic:
# configs/pnp_syn_selected.json). MAGNET_PNP_SHUFFLE=1 also records Test 1 (branch shuffle) of every run in its
# result.json; MAGNET_PNP_DROP_CKPT=1 deletes the PnP checkpoints after evaluation. Results: runs/pnp/<bb>/<ds>/<term>/seed_<s>/.
# Usage: GPU=0 BACKBONES="PatchTST TFT" bash experiments/table2_pnp/run.sh      (each run is independent -> parallelise freely)
cd "$(dirname "$0")/../.."
GPU=${GPU:-0}; BACKBONES=${BACKBONES:-"DAG GCGNet TimeXer TimeKAN PatchTST TFT"}
export MAGNET_PNP_SHUFFLE=1 MAGNET_PNP_DROP_CKPT=1
for bb in $BACKBONES; do for term in short long; do
  for ds in BE DE FR NP PJM Energy Colbun Rapel Sdwpfh1 Sdwpfh2 Sdwpfm1 Sdwpfm2 Syn5 Syn20 Syn50 Syn20weak Syn20null; do
    for seed in 0 42 123 2024 3407; do
      [ -f runs/pnp/$bb/$ds/$term/seed_$seed/result.json ] || python3 train/run_pnp.py --backbone $bb --dataset $ds --term $term --seeds $seed --gpu $GPU
    done
  done
done; done
