# Evaluation helpers for MAGNet+TimeXer (plug-and-play): test-set evaluation with optional
# test-time noise averaging (TTA) and first-difference handling for Colbun/long.
# Models are conditioned on the observed history only (no future exogenous values).

import torch

import torch.nn as nn

import json

import numpy as np

import argparse

from pathlib import Path

import sys

import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

PKG_ROOT = os.path.dirname(ROOT_DIR)

for _d in ("models", "data", "backbones"):
    sys.path.insert(0, os.path.join(PKG_ROOT, _d))

from MAGNET_data import get_dataloaders

from MAGNET_with_TimeXer import TimeXerWithMAGNET

DEVICE           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DIFF_DATASETS = {11}

TTA_DATASETS  = {8, 13, 14, 15, 16}

def mae(pred, target):
    return torch.mean(torch.abs(pred - target)).item()

def rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()

def mape(pred, target, eps=1.0):
    return torch.mean(torch.abs((pred - target) / (target.abs() + eps))).item() * 100

def get_seq_horizon(dataset_id, term):
    if dataset_id in (11, 12):
        return (60, 10) if term == 'short' else (180, 30)
    return (168, 24) if term == 'short' else (720, 360)

def evaluate(model, test_loader, dataset_id, term, use_tta=True):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for exo_patches, endo_patches, target, mean, std, time_features, last_known in test_loader:
            exo_patches   = exo_patches.to(DEVICE)
            endo_patches  = endo_patches.to(DEVICE)
            target        = target.to(DEVICE)
            time_features = time_features.to(DEVICE)
            last_known    = last_known.to(DEVICE)

            # never pass the true future exog into the model at eval time.
            raw  = model(exo_patches, endo_patches,
                         time_features=time_features)
            pred = raw[0] if isinstance(raw, (tuple, list)) else raw
            if len(pred.shape) == 3:
                pred = pred.squeeze(-1)

            if use_tta and dataset_id in TTA_DATASETS:
                tta_preds = [pred]
                for _ in range(2):
                    exo_n  = exo_patches  + 0.01 * torch.randn_like(exo_patches)
                    endo_n = endo_patches + 0.01 * torch.randn_like(endo_patches)
                    r = model(exo_n, endo_n, time_features=time_features)
                    p = r[0] if isinstance(r, (tuple, list)) else r
                    if len(p.shape) == 3:
                        p = p.squeeze(-1)
                    tta_preds.append(p)
                pred = torch.stack(tta_preds).mean(0)

            if dataset_id in DIFF_DATASETS and term == 'long':
                pred_d = (last_known.unsqueeze(1) + pred.cumsum(dim=-1)).cpu()
                targ_d = (last_known.unsqueeze(1) + target.cumsum(dim=-1)).cpu()
            else:
                pred_d = pred.cpu()
                targ_d = target.cpu()

            all_preds.append(pred_d)
            all_targets.append(targ_d)

    all_preds   = torch.cat(all_preds,   dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    return {
        'mae' : mae(all_preds, all_targets),
        'mse' : nn.functional.mse_loss(all_preds, all_targets).item(),
        'rmse': rmse(all_preds, all_targets),
        'mape': mape(all_preds, all_targets),
    }

def build_timexer_config(tx_cfg_dict, n_exo, seq_len, horizon, patch_len, batch_size):
    class TimeXerConfig:
        pass
    cfg = TimeXerConfig()
    cfg.enc_in        = n_exo + 1
    cfg.series_dim    = 1
    cfg.seq_len       = seq_len
    cfg.pred_len      = horizon
    cfg.patch_len     = patch_len
    cfg.d_model       = tx_cfg_dict['d_model']
    cfg.d_ff          = tx_cfg_dict['d_ff']
    cfg.n_heads       = tx_cfg_dict['n_heads']
    cfg.e_layers      = tx_cfg_dict['e_layers']
    cfg.dropout       = 0.2
    cfg.factor        = 1
    cfg.activation    = "gelu"
    cfg.use_future    = 0   # history-only exo embedding width
    cfg.batch_size    = batch_size
    cfg.freq          = "h"
    cfg.lradj         = "type3"
    cfg.output_attention = 0
    cfg.seg_len       = 6
    cfg.win_size      = 2
    cfg.period_len    = 4
    cfg.fc_dropout    = 0.2
    cfg.moving_avg    = 25
    return cfg

def detect_graph_proj_arch(state_dict):
    """graph_proj was refactored mid hparam-search: older checkpoints hold a
    single Linear->LayerNorm->Dropout ('legacy'); newer ones hold a 2-layer
    bottleneck Linear->GELU->LayerNorm->Linear->LayerNorm->Dropout
    ('bottleneck', index 3 is the second Linear). Inspect the saved keys
    instead of assuming, so either vintage of checkpoint loads correctly."""
    return 'bottleneck' if 'graph_proj.3.weight' in state_dict else 'legacy'
