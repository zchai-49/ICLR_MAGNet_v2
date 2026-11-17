#!/usr/bin/env python3
"""Train the official MAGNet+DLinear model (final configuration, models/final_defaults.py) for one (dataset, term) cell:
every deployed ensemble side (configs/deployment_final.json) x the requested seeds, with each side's validation-selected
hyperparameters, enhancer ("fix") and training regime. Existing checkpoints are skipped.

    python3 train/train_magnet.py --dataset FR --term short                    # all deployed sides x seeds 0,42,123,2024,3407
    python3 train/train_magnet.py --dataset Syn20 --term long --seeds 0 --gpu 1
    python3 train/train_magnet.py --dataset NP --term short --option no_exo   # an ablation / control (see train/train_options.py)

Checkpoints: checkpoints/magnet_dlinear/<term>_<dataset>_<side>/seed_<seed>/ckpt.pt (+ result.json); ablation options go to
runs/opt_<option>/ unless --out-root is given. This is train/train_options.py with --option detach_gg (= the final model) and
the shipped checkpoint folder as defaults."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "train"))
argv = sys.argv[1:]
if "--option" not in argv:
    argv = ["--option", "detach_gg"] + argv
if "--out-root" not in argv and argv[argv.index("--option") + 1] == "detach_gg":
    argv += ["--out-root", os.path.join(ROOT, "checkpoints", "magnet_dlinear")]
sys.argv = [os.path.join(ROOT, "train", "train_options.py")] + argv
import train_options  # noqa: E402

train_options.main()
