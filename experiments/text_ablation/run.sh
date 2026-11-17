#!/bin/bash
# Text ablations on NP, Energy, Rapel, Sdwpfm1 (ids 8 10 12 15) + Syn20, short/long, seeds 0 and 42 (one change at a time):
#   encoder    same Llama-3.2-3B text re-encoded: BERT (re-encoded), MiniLM, RoBERTa, MPNet, BGE   (models/text_encoder_variants.py)
#   generator  Llama-3.2-1B, Llama-3.1-8B, Qwen2.5-3B, Qwen2.5-7B, DeepSeek-7B (same prompts, BERT)   (models/text_generate_variants.py)
#   prompt     P1 simple, P2 no causal questions, P3 no statistics, P4 statistics only (no LLM), P5 name+description (no LLM)
#              (models/text_prompt_variants.py)
# Deployed column = the shipped checkpoints (Llama-3.2-3B + BERT + our prompt); "w/o Text" = experiments/ablation no_text.
# Needs the HF models (meta-llama/* are gated: huggingface-cli login). Usage: GPU=0 bash experiments/text_ablation/run.sh
cd "$(dirname "$0")/../.."
GPU=${GPU:-0}; export CUDA_VISIBLE_DEVICES=$GPU
# 1. text + embeddings
python3 models/text_encoder_variants.py --datasets 8 10 12 15 --generic Syn20
for v in llama1b llama8b qwen qwen7b deepseek psimple pnocausal pnostats pstats pnamedesc; do
  python3 models/text_generate_variants.py --variant $v --datasets 8 10 12 15
  python3 models/text_generate_generic.py --name Syn20 --folder synthetic/Syn20 --variant $v
done
# 2. MAGNet+DLinear with each text variant (deployed sides, seeds 0 and 42)
for opt in text_encbert text_encminilm text_encroberta text_encmpnet text_encbge text_llama1b text_llama8b text_qwen text_qwen7b \
           text_deepseek text_psimple text_pnocausal text_pnostats text_pstats text_pnamedesc; do
  for ds in NP Energy Rapel Sdwpfm1 Syn20; do for term in short long; do
    [ $opt = text_encbert ] && [ $ds = Syn20 ] && continue      # Syn20 logs the full text: re-encoded BERT = deployed
    python3 train/train_magnet.py --option $opt --dataset $ds --term $term --seeds 0 42 --gpu 0 --out-root runs/text/$opt
  done; done
done
