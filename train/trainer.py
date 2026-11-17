#!/usr/bin/env python3
"""Shared MAGNet+DLinear training utilities (used by train/train_options.py, train/train_magnet.py and the DLinear-alone
trainers): training constants, the loss per dataset/horizon, one epoch of training/evaluation, and the optimiser /
learning-rate schedule of each training regime.

  loss       L1 (short) / Smooth-L1 beta=0.5 (long); magnitude-weighted L1 on the wind datasets; Colbun long on first
             differences (L1 on the reconstructed level)
  optimiser  AdamW, weight decay 1e-3, cosine schedule over EPOCHS; regime attn/gate_warmup_30ep_GATE_LR_MULT50: 30-epoch
             warm-up of the attention / gate parameters with gate learning rate x50 and a 0.05 floor; 5-epoch linear warm-up
             on the WARMUP_LONG long-horizon datasets
"""
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "configs"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import lag1_config as C  # noqa: E402
from MAGNET_data import get_dataloaders  # noqa: E402
from evaluate import predict, metrics  # noqa: E402

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

EPOCHS, PATIENCE, BATCH_SIZE, WEIGHT_DECAY = 200, 15, 32, 1e-3
WIND = {13, 14, 15, 16}
GATE_LR_MULT = 50.0
GATE_NAMES = {"backbone.trend_gate", "backbone.seasonal_gate", "backbone.film_gate", "backbone.mix_gate",
              "backbone.crossattn_film_gate", "backbone.hyper_gate", "backbone.scale_gate"}
ATTN_PREFIXES = ("crossattn.", "crossattn_query")
STAB_GATE_LR_MULT, WARMUP_EPOCHS, FLOOR_MULT = 50.0, 30, 0.05
WARMUP_LONG = {5, 8, 10, 12, 13}  # long-term datasets that use a 5-epoch linear LR warmup
CAUSAL_AUX_WEIGHT = 1.0  # weight of the GVAR / sparse-Granger estimator's own loss (unused for NGC)


def loss_fn(pred, target, dataset_id, term, last_known):
    if dataset_id in C.DIFF_DATASETS and term == "long":
        pred = last_known.unsqueeze(1) + pred.cumsum(-1)
        target = last_known.unsqueeze(1) + target.cumsum(-1)
        return nn.functional.l1_loss(pred, target), pred, target
    if dataset_id in WIND:
        return ((1 + target.abs()) * (pred - target).abs()).mean(), pred, target
    if term == "long":
        return nn.functional.smooth_l1_loss(pred, target, beta=0.5), pred, target
    return nn.functional.l1_loss(pred, target), pred, target


def run_epoch(model, loader, optimizer, device, dataset_id, term, train, weighted=False):
    model.train() if train else model.eval()
    preds, targs = [], []
    with (torch.enable_grad() if train else torch.no_grad()):
        for exo_p, endo_p, target, _m, _s, _tf, last_known in loader:
            exo_p, endo_p, target, last_known = (exo_p.to(device), endo_p.to(device),
                                                 target.to(device), last_known.to(device))
            pred = model(exo_p, endo_p)  # history only: no future exogenous values
            if pred.dim() == 3:
                pred = pred.squeeze(-1)
            if weighted:  # magnitude-weighted L1 (NP/long)
                loss = ((1 + target.abs()) * (pred - target).abs()).mean()
            else:
                loss, pred, target = loss_fn(pred, target, dataset_id, term, last_known)
            # options A/B (causal_alternatives.py): the causal estimator's own Granger loss
            aux = getattr(getattr(model, "magnet_graph", None), "causal_aux_loss", None)
            if train and aux is not None:
                loss = loss + CAUSAL_AUX_WEIGHT * aux
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5 if dataset_id == 13 else 1.0)
                optimizer.step()
                est = getattr(getattr(model, "magnet_graph", None), "causal_estimator", None)
                if getattr(est, "proximal", False):   # option B, proximal group lasso
                    est.prox_step(optimizer.param_groups[0]["lr"])
            preds.append(pred.detach().cpu())
            targs.append(target.detach().cpu())
    return torch.mean(torch.abs(torch.cat(preds) - torch.cat(targs))).item()


def build_optimizer(model, term, dataset_id, regime, lr):
    def kinds(split_attn):
        gate, attn, other = [], [], []
        for n, p in model.named_parameters():
            if n in GATE_NAMES:
                gate.append(p)
            elif split_attn and n.startswith(ATTN_PREFIXES):
                attn.append(p)
            else:
                other.append(p)
        groups = [("other", other, lr, WEIGHT_DECAY)]
        if attn:
            groups.append(("attn", attn, lr, WEIGHT_DECAY))
        if gate:
            groups.append(("gate", gate, lr * STAB_GATE_LR_MULT, 0.0))
        return groups

    if regime in ("gate_warmup_30ep_GATE_LR_MULT50", "attn_warmup_30ep_GATE_LR_MULT50"):
        special = "gate" if regime.startswith("gate") else "attn"
        groups = kinds(split_attn=(special == "attn"))
        opt = torch.optim.AdamW([dict(params=p, lr=l, weight_decay=w) for _, p, l, w in groups])

        def factory(kind):
            def f(epoch):
                cosine = 0.5 * (1 + math.cos(math.pi * epoch / EPOCHS))
                if kind == special:
                    warm = min(1.0, (epoch + 1) / WARMUP_EPOCHS)
                    return (FLOOR_MULT + (1 - FLOOR_MULT) * warm) * cosine
                return cosine
            return f
        return opt, torch.optim.lr_scheduler.LambdaLR(opt, [factory(k) for k, *_ in groups])

    gate = [p for n, p in model.named_parameters() if n in GATE_NAMES]
    other = [p for n, p in model.named_parameters() if n not in GATE_NAMES]
    groups = [{"params": other, "lr": lr, "weight_decay": WEIGHT_DECAY}]
    if gate:
        groups.append({"params": gate, "lr": lr * GATE_LR_MULT, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)
    if regime == "weighted_loss_WARMUP_LONG_5ep" or (term == "long" and dataset_id in WARMUP_LONG):
        def lr_lambda(epoch):
            if epoch < 5:
                return (epoch + 1) / 5
            progress = (epoch - 5) / (EPOCHS - 5)
            return 0.5 * (1 + math.cos(math.pi * progress)) * (1 - 1e-6 / lr) + 1e-6 / lr
        return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return opt, torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
