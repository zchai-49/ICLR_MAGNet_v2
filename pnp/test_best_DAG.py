# Evaluation helpers for MAGNet+DAG: test-set evaluation on the observed history only (infer_use_future=False),
# optional test-time noise averaging (TTA), first-difference handling for Colbun/long.

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

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def build_dag_config(dag_cfg_dict, n_exo, seq_len, horizon):
    class DagConfig:
        pass
    cfg = DagConfig()
    cfg.seq_len          = seq_len
    cfg.pred_len         = horizon
    cfg.patch_len        = dag_cfg_dict['patch_len']
    cfg.stride           = dag_cfg_dict['stride']
    cfg.use_c            = True
    cfg.use_t            = True
    cfg.use_c_exog       = True
    cfg.use_t_exog       = True
    cfg.alpha            = dag_cfg_dict['alpha']
    cfg.beta             = dag_cfg_dict['beta']
    cfg.series_dim       = 1
    cfg.infer_use_future = False   # no future exogenous values at inference
    cfg.enc_in           = n_exo + 1
    cfg.d_model          = dag_cfg_dict['d_model']
    cfg.d_ff             = dag_cfg_dict['d_ff']
    cfg.n_heads          = dag_cfg_dict['n_heads']
    cfg.e_layers         = dag_cfg_dict['e_layers']
    cfg.dropout          = 0.2
    cfg.factor           = 1
    cfg.activation       = "gelu"
    cfg.criterion        = nn.L1Loss()
    return cfg

def evaluate(model, test_loader, dataset_id, term, use_tta=True, device=None):
    if device is None:
        device = DEVICE
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for exo_patches, endo_patches, target, mean, std, time_features, last_known in test_loader:
            exo_patches   = exo_patches.to(device)
            endo_patches  = endo_patches.to(device)
            target        = target.to(device)
            time_features = time_features.to(device)
            last_known    = last_known.to(device)

            raw  = model(exo_patches, endo_patches,
                         time_features=time_features)
            pred = raw[0] if isinstance(raw, (tuple, list)) else raw
            if len(pred.shape) == 3:
                pred = pred.squeeze(-1)

            # TTA on noisy datasets (mode 2 only)
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
