#!/usr/bin/env python3
"""MAGNet+DLinear trainer with options: detach_gg = the FINAL model (SIG graph, detached weights, fixed cross-fit gate;
lag-0 nuisance + pre-whitening via models/final_defaults.py), plus every ablation / control / text option used in the paper.

    python3 train/train_options.py --option gate_est --dataset NP --term short --seeds 0 42 --gpu 0

Options (results go to runs/opt_<option>/ by default; train/train_magnet.py runs --option detach_gg = the final model<term>_<dataset>_<side>/seed_<seed>/):
  base       plain sparse_granger_lag0 (no detach, no gate)
  pre        estimator pre-trained alone on the training split, then joint training as usual (control for the gates)
  gate_est   pre + causal gate g = sigmoid(logit(r) + b) on the exo message, r = exo/endo strength ratio  (option 1)
  gate_free  pre + free gate g = sigmoid(b), b initialised at logit(r) from the pre-trained estimator    (control)
  exo_mask   each step averages the loss on the clean batch and on a batch with one exo shuffled           (control)
  gain       exo_mask + GainHead: per-window weights from the predicted occlusion gain                     (option 2)
  gate_gg    fixed gate g = clip(held-out Granger gain, 0, 1) on the exo message (option 1 redesign; gain from
             scripts/granger_gain_check.py, ridge, horizon-matched, train/validation only)
  gate_gg_free  free learnable gate initialised at the same value                                         (control)
  detach     graph weights detached: the estimator learns only from its own Granger loss (pure Granger graph)
  detach_gg  detach + gate_gg (the candidate "sound" combination)
Matched controls and text variants (same deployment and training; no causal-module change):
  dlinear_only     the cell's DLinear backbone + enhancers alone (graph removed), same two-side ensemble
  exo_mlp          dlinear_only + a 2-layer exo MLP with about the graph branch's parameter count
  xattn_dlinear    dlinear_only + one plain cross-attention block (target patches -> exo patch tokens), no graph,
                   no Granger weights, no gate, no text (models/controls.py XAttnDLinear)
  plain_xattn      graph kept, but no causal bias, no text bias and no cosine edge bias (plain cross-attention)
  no_cosine        base with uniform exo-exo edge weights (the one-at-a-time ablation we did not have)
  no_text          base without text (same-host copy of the module ablation's no_text)
  no_exo_gat       module ablation: endo attends to the pre-GAT exo embeddings (exo-exo GAT skipped)
  no_causal        module ablation: uniform exo weights (no Granger log-w bias); the fixed gate is kept
  no_patching      module ablation: a single patch (patch_len = seq_len)
  text_<v>         base with the text embeddings of models/text_generate_variants.py variant <v>
                   (text_enc*: models/text_encoder_variants.py, same text re-encoded by 5 encoders)
Everything else (deployment, hparams, regime, loss, early stopping on validation MAE) is train/trainer.py's.
Model selection between options must use best_val_mae only; test metrics are recorded but not for choosing.
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import trainer as T  # noqa: E402  (shared training utilities; sets up sys.path for configs/ and scripts/)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import lag1_config as C  # noqa: E402
import ablation as AB  # noqa: E402
from MAGNET_data import get_dataloaders  # noqa: E402
from evaluate import predict, metrics  # noqa: E402
from causal_alternatives import exo_strength_ratio  # noqa: E402
from controls import ExoMLPDLinear, XAttnDLinear  # noqa: E402

OPTIONS = ("base", "pre", "gate_est", "gate_free", "exo_mask", "gain", "gate_gg", "gate_gg_free", "detach", "detach_gg",
           "dlinear_only", "exo_mlp", "plain_xattn", "no_cosine", "no_text",
           "text_regen", "text_anon", "text_names_only", "text_qwen", "text_minilm", "no_exo",
           "no_exo_gat", "no_causal", "no_patching",
           "text_encbert", "text_encminilm", "text_encroberta", "text_encmpnet", "text_encbge", "xattn_dlinear",
           "text_llama8b", "text_qwen7b", "text_deepseek", "text_gpt35", "text_llama1b",
           "text_psimple", "text_pnocausal", "text_pnostats", "text_pstats", "text_pnamedesc")   # prompt ablation 2026-11-16
CAUSAL = os.environ.get("MAGNET_CAUSAL", "sparse_granger_lag0")   # 2026-11-13: MAGNET_CAUSAL=sparse_granger drops the lag-0 term (test)
# Granger-gain gate file. FINAL (2026-11-09, chosen before any test result): cross-fitted gain (4 blocked validation
# folds, ridge + GBM candidates, scripts/granger_gain_crossfit.py). Earlier: v2 granger_gain_nested.json,
# v1 granger_gain_check.json.
GG_FILE = os.environ.get("MAGNET_GG_FILE", "granger_gain_crossfit.json")
# Compose a control / text option with a causal option (2026-11-09): MAGNET_ON_TOP=detach_gg runs e.g. no_text as
# "detach_gg minus text" (the control's model change + detach_gg's detached graph and fixed Granger-gain gate).
CAUSAL_OPTS = ("base", "pre", "gate_est", "gate_free", "exo_mask", "gain", "gate_gg", "gate_gg_free", "detach",
               "detach_gg", "no_exo")
ON_TOP = os.environ.get("MAGNET_ON_TOP") or None
# option 1 (2026-11-14): MAGNET_GATE_LEARN=1 makes detach_gg's fixed cross-fit gate a learnable gate g = sigmoid(b),
# b initialised at logit(cross-fit gain) (clipped to [0.02, 0.98]) -- the gate becomes a prior, not a cap
GATE_LEARN = os.environ.get("MAGNET_GATE_LEARN") == "1"
_GATE_MODE = {"gate_est": "estimator", "gate_free": "free", "gate_gg": "fixed", "gate_gg_free": "free",
              "detach_gg": "free" if GATE_LEARN else "fixed", "no_exo": "fixed"}
GAIN_WEIGHT = 1.0
PRETRAIN_EPOCHS, PRETRAIN_PATIENCE, PRETRAIN_LR = 100, 5, 1e-3
# The causal gate's bias deliberately stays at the base LR (not GATE_LR_MULT): with the 50x multiplier both gates
# moved ~1.3 logits in one epoch in the smoke test, washing out the estimator-driven vs free-gate comparison.


def pretrain_estimator(est, train_l, val_l, device):
    """Fit the sparse-Granger estimator alone (its own Granger loss + group lasso) on the training split, early-stopped
    on the validation split. Returns the number of epochs run."""
    opt = torch.optim.Adam(est.parameters(), lr=PRETRAIN_LR)
    best, bad, best_state = float("inf"), 0, None
    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        est.train()
        for exo_p, endo_p, *_ in train_l:
            _, aux, _ = est(torch.cat([endo_p, exo_p], dim=-1).to(device))
            opt.zero_grad()
            aux.backward()
            opt.step()
        est.eval()
        with torch.no_grad():
            val = np.mean([est(torch.cat([e, x], dim=-1).to(device))[1].item() for x, e, *_ in val_l])
        if val < best - 1e-6:
            best, bad, best_state = val, 0, {k: v.detach().clone() for k, v in est.state_dict().items()}
        else:
            bad += 1
            if bad >= PRETRAIN_PATIENCE:
                break
    est.load_state_dict(best_state)
    return epoch


def shuffle_one_exo(exo_p):
    """Copy of the batch with one random exo variable permuted across windows (same counterfactual as the
    permutation-importance metric). Returns (shuffled batch, index of the exo)."""
    j = int(torch.randint(exo_p.shape[-1], (1,)))
    out = exo_p.clone()
    out[..., j] = exo_p[torch.randperm(exo_p.shape[0], device=exo_p.device)][..., j]
    return out, j


def run_epoch(model, loader, optimizer, device, dataset_id, term, train, option, weighted=False):
    model.train() if train else model.eval()
    mg = model.magnet_graph
    preds, targs = [], []

    def forward_loss(exo, endo, target, last_known):
        pred = model(exo, endo)
        if pred.dim() == 3:
            pred = pred.squeeze(-1)
        if weighted:
            return ((1 + target.abs()) * (pred - target).abs()).mean(), pred, target
        return T.loss_fn(pred, target, dataset_id, term, last_known)

    with (torch.enable_grad() if train else torch.no_grad()):
        for exo_p, endo_p, target, _m, _s, _tf, last_known in loader:
            exo_p, endo_p, target, last_known = (exo_p.to(device), endo_p.to(device),
                                                 target.to(device), last_known.to(device))
            loss, pred, targ = forward_loss(exo_p, endo_p, target, last_known)
            aux = getattr(mg, "causal_aux_loss", None)                 # estimator loss on the clean batch only
            if train and option in ("exo_mask", "gain"):
                gain_pred = mg.gain_pred if option == "gain" else None
                exo_s, j = shuffle_one_exo(exo_p)
                loss_s, pred_s, targ_s = forward_loss(exo_s, endo_p, target, last_known)
                if option == "gain":
                    gains = ((pred_s - targ_s).abs().mean(-1) - (pred - targ).abs().mean(-1)).detach()   # [B]
                    mg.gain_head.update_scale(gains)
                    gain_loss = nn.functional.mse_loss(gain_pred[:, j], gains / mg.gain_head.scale)
                loss = 0.5 * (loss + loss_s)
                if option == "gain":
                    loss = loss + GAIN_WEIGHT * gain_loss
            if train and aux is not None:
                loss = loss + T.CAUSAL_AUX_WEIGHT * aux
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5 if dataset_id == 13 else 1.0)
                optimizer.step()
            preds.append(pred.detach().cpu())
            targs.append(targ.detach().cpu())
    return torch.mean(torch.abs(torch.cat(preds) - torch.cat(targs))).item()


def build_option_model(dataset, term, side, option, gate_init=0.0, on_top=None):
    """Model for one option (no pre-training); evaluation code rebuilds checkpoints with this."""
    model, patch_len = build_model(dataset, term, side, option)
    seq_len, _ = C.get_seq_horizon(C.cell_config(dataset, term)["dataset_id"], term)
    if model.magnet_graph is not None:
        model.magnet_graph.enable_causal_options(
            gate=_GATE_MODE.get(on_top or option, "none"), gate_init=gate_init,
            gain_head=option == "gain", seq_len=seq_len)        # gate values come from the checkpoint
    return model, patch_len


FIXES = {"none": ({}, None), "mlp_backbone": ({"use_mlp_backbone": True}, None),
         "residual_head": ({"use_residual_head": True}, None), "patch8": ({}, "patch8"),
         "deep_film": ({"use_crossattn_deep_film": True}, None)}


def apply_candidate(dataset, term, side, hp_rank, fix, regime):
    """Validation re-selection: set one side of the loaded deployment to a candidate (in memory only)."""
    s = C.cell_config(dataset, term)["sides"][side]
    s["hparams"] = s["hparam_candidates"][hp_rank]
    kwargs, patch = FIXES[fix]
    s["kwargs"] = dict(kwargs)
    if patch:
        seq_len, _ = C.get_seq_horizon(C.NAME_TO_ID[dataset], term)
        s["patch_len"] = 8 if seq_len % 8 == 0 else 6               # hydro windows (60/180) take 6, as before
    if regime == "warmup":
        s["training_regime"] = ("gate_warmup_30ep_GATE_LR_MULT50" if side == "gated_mixing_only"
                                else "attn_warmup_30ep_GATE_LR_MULT50")


def build_model(dataset, term, side, option):
    """The deployed sparse_granger_lag0 model, modified for a control / text option (identical for the others)."""
    variant = ("no_text" if option in ("no_text", "plain_xattn") else
               "no_patching" if option == "no_patching" else "full")
    model, patch_len = AB.build_ablation_model(dataset, term, side, variant, causal=CAUSAL)
    mg = model.magnet_graph
    if option == "plain_xattn":
        mg.uniform_causal = True
        mg.exo_graph.edge_weights.uniform = True
    elif option == "no_exo_gat":
        mg.skip_exo_gat = True
    elif option == "no_causal":
        mg.uniform_causal = True
    elif option == "no_cosine":
        mg.exo_graph.edge_weights.uniform = True
    elif option.startswith("text_"):
        mg.text_variant = option[len("text_"):]
        mg._load_text_embeddings(mg.dataset, mg.term)
    elif option in ("dlinear_only", "exo_mlp", "xattn_dlinear"):
        # graph parameters the deployed model actually uses (NGC and its fusion MLP are bypassed by the override)
        unused = {id(p) for name in ("ngc", "ngc_windowed", "causal", "gated")
                  for p in (getattr(mg.endo_insertion, name).parameters() if hasattr(mg.endo_insertion, name) else [])}
        graph_params = sum(p.numel() for p in mg.parameters() if id(p) not in unused)
        model.mode, model.magnet_graph = 3, None
        if option == "exo_mlp":
            cell = C.cell_config(dataset, term)
            seq_len, horizon = C.get_seq_horizon(cell["dataset_id"], term)
            model = ExoMLPDLinear(model, C.DATASETS[cell["dataset_id"]][1], seq_len, horizon, graph_params)
        if option == "xattn_dlinear":
            cell = C.cell_config(dataset, term)
            seq_len, horizon = C.get_seq_horizon(cell["dataset_id"], term)
            d = cell["sides"][side]["hparams"]["magnet_d_model"]
            model = XAttnDLinear(model, C.DATASETS[cell["dataset_id"]][1], patch_len, seq_len // patch_len, horizon, d,
                                 n_heads=C.MAGNET_N_HEADS.get(d, 4), dropout=0.2 if term == "long" else 0.1)
    return model, patch_len


def train_one(dataset, term, side, seed, gpu, out_root, option):
    cell = C.cell_config(dataset, term)
    s = cell["sides"][side]
    dataset_id = cell["dataset_id"]
    regime, lr = s["training_regime"], s["hparams"]["lr"]
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    seq_len, horizon = C.get_seq_horizon(dataset_id, term)
    model, patch_len = build_model(dataset, term, side, option)
    model.to(device)
    train_l, val_l, test_l = get_dataloaders(npz_path=C.csv_path(dataset_id), sensor_id=0, seq_len=seq_len,
                                             forecast_horizon=horizon, patch_len=patch_len,
                                             batch_size=T.BATCH_SIZE, dataset_id=dataset_id)
    t0 = time.time()
    go = option if (option in CAUSAL_OPTS or not ON_TOP) else ON_TOP     # the causal option in effect
    pre_epochs, ratio = None, None
    pre_s = None
    if go in ("pre", "gate_est", "gate_free"):
        tp = time.time()
        pre_epochs = pretrain_estimator(model.magnet_graph.causal_estimator, train_l, val_l, device)
        ratio = float(exo_strength_ratio(model.magnet_graph.causal_estimator))
        pre_s = time.time() - tp
    r = min(max(ratio if ratio is not None else 0.5, 1e-4), 1 - 1e-4)
    gate_init = math.log(r / (1 - r))
    if go in ("gate_gg", "gate_gg_free", "detach_gg"):
        gg = json.load(open(os.path.join(T.ROOT, "configs", GG_FILE)))
        ratio = min(max(gg[f"{dataset}/{term}"]["ridge"]["gain"], 0.0), 1.0)
        if os.environ.get("MAGNET_GATE_FLOOR"):      # Wind idea 2 (2026-11-14): gate floor g = max(gain, g_min); g_min=1 -> exo fully on
            ratio = max(ratio, float(os.environ["MAGNET_GATE_FLOOR"]))
        gate_init = ratio if go in ("gate_gg", "detach_gg") else math.log(max(ratio, 0.02) / (1 - min(ratio, 0.98)))
        if go == "detach_gg" and GATE_LEARN:         # option 1: the cross-fit gate as a learnable prior
            gate_init = math.log(max(ratio, 0.02) / (1 - min(ratio, 0.98)))
    if go == "no_exo":                  # restricted model of the model-based Granger test: exo message x 0
        gate_init = 0.0
    if go in ("detach", "detach_gg") and model.magnet_graph is not None:
        model.magnet_graph.detach_causal = True
    if model.magnet_graph is not None:
        model.magnet_graph.enable_causal_options(
            gate=_GATE_MODE.get(
                go, "none"), gate_init=gate_init, gain_head=go == "gain", seq_len=seq_len)
    model.to(device)
    opt, sched = T.build_optimizer(model, term, dataset_id, regime, lr)
    weighted = regime == "weighted_loss_WARMUP_LONG_5ep"

    best_val, bad, best_state = float("inf"), 0, None
    cuda = device.type == "cuda"
    sync = (lambda: torch.cuda.synchronize(device)) if cuda else (lambda: None)
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    train_s, val_s, best_epoch = [], [], 0              # timing (2026-11-09): per-epoch train / validation seconds
    for epoch in range(1, T.EPOCHS + 1):
        sync(); te = time.time()
        run_epoch(model, train_l, opt, device, dataset_id, term, True, option, weighted)
        sync(); tv = time.time()
        val = run_epoch(model, val_l, opt, device, dataset_id, term, False, option)
        sync(); train_s.append(tv - te); val_s.append(time.time() - tv)
        sched.step()
        print(f"[{option} {dataset}/{term}/{side}/seed_{seed}] epoch {epoch:3d}  val MAE {val:.4f}", flush=True)
        if val < best_val:
            best_val, bad, best_epoch = val, 0, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= T.PATIENCE:
                break

    model.load_state_dict(best_state)
    p, t = predict(model.to(device).eval(), test_l, dataset_id, term, device)
    m = metrics(p, t)
    mg = model.magnet_graph
    gate = mg._causal_gate() if mg is not None else None
    est = mg.causal_estimator if mg is not None else None
    n_used = sum(p.numel() for p in model.parameters() if p.grad is not None)   # params touched by the last step
    out_dir = os.path.join(out_root, f"{term}_{dataset}_{side}", f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(best_state, os.path.join(out_dir, "ckpt.pt"))
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump({"dataset": dataset, "term": term, "side": side, "seed": seed, "option": option, "on_top": None if go == option else go, "causal": CAUSAL,
                   "lag0_nuisance": os.environ.get("MAGNET_LAG0_NUISANCE") == "1",
                   "sig_flags": {k: v for k, v in os.environ.items() if k.startswith(("MAGNET_LAG0", "MAGNET_SIG", "MAGNET_CAUSAL", "MAGNET_GATE", "MAGNET_WIND"))},
                   "hparams": s["hparams"], "training_regime": regime, "best_val_mae": best_val,
                   "stopped_at_epoch": epoch, "test_metrics": m, "ngc_lag": 1,
                   "pretrain_epochs": pre_epochs, "ratio_after_pretrain": ratio,
                   "final_ratio": None if est is None else float(exo_strength_ratio(est)),
                   "n_params_used": n_used,
                   "final_gate": None if gate is None else float(gate),
                   "group_norms_lag_x_var": None if est is None else est.group_norms().tolist(),
                   "contemporaneous_norms_exo": None if est is None or est.contemporaneous_norms() is None
                   else est.contemporaneous_norms().tolist(),
                   "elapsed_s": round(time.time() - t0),
                   "timing": {
                              "gpu": torch.cuda.get_device_name(device) if cuda else "cpu",
                              "train_epoch_s": [round(x, 3) for x in train_s], "val_epoch_s": [round(x, 3) for x in val_s],
                              "mean_train_epoch_s": float(np.mean(train_s)), "epochs_run": len(train_s),
                              "best_epoch": best_epoch, "train_loop_s": float(sum(train_s) + sum(val_s)),
                              "pretrain_s": pre_s,
                              "peak_gpu_mem_mb": torch.cuda.max_memory_allocated(device) / 2**20 if cuda else None,
                              "n_params_total": sum(q.numel() for q in model.parameters()),
                              "n_train_windows": len(train_l.dataset), "batch_size": T.BATCH_SIZE,
                              "load_avg_1m": os.getloadavg()[0]}}, f, indent=2)
    print(f"DONE {option} {dataset}/{term}/{side}/seed_{seed}: val {best_val:.4f}  test MAE {m['mae']:.4f}  "
          f"MSE {m['mse']:.4f}  gate {gate if gate is None else round(float(gate), 3)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--option", required=True, choices=OPTIONS)
    ap.add_argument("--dataset", required=True, choices=list(C.NAME_TO_ID))
    ap.add_argument("--term", required=True, choices=["short", "long"])
    ap.add_argument("--side", default="all")
    ap.add_argument("--seeds", type=int, nargs="+", default=C.SEEDS)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=T.EPOCHS, help="override for smoke tests")
    ap.add_argument("--out-root", default=None, help="default: runs/opt_<option>/")
    ap.add_argument("--hp-rank", type=int, default=None,
                    help="validation re-selection: use hparam_candidates[rank] of the deployment (MAGNET_DEPLOYMENT)")
    ap.add_argument("--fix", default=None, choices=list(FIXES), help="validation re-selection: architecture fix")
    ap.add_argument("--regime", default=None, choices=["default", "warmup"],
                    help="validation re-selection: warmup = the gate/attention-warmup regime")
    args = ap.parse_args()
    T.EPOCHS = args.epochs
    if args.hp_rank is not None or args.fix or args.regime:
        apply_candidate(args.dataset, args.term, args.side, args.hp_rank or 0, args.fix or "none",
                        args.regime or "default")
    out_root = args.out_root or os.path.join(T.ROOT, "runs", f"opt_{args.option}")
    deployed = C.cell_config(args.dataset, args.term)["deployed_sides"]
    sides = deployed if args.side == "all" else [args.side]
    for side in sides:
        sel = C.cell_config(args.dataset, args.term)["sides"][side].get("selected")
        if sel and args.hp_rank is None and not args.fix and not args.regime:   # deployment_valsel.json
            apply_candidate(args.dataset, args.term, side, sel["hp_rank"], sel["fix"], sel["regime"])
        for seed in args.seeds:
            out = os.path.join(out_root, f"{args.term}_{args.dataset}_{side}", f"seed_{seed}", "ckpt.pt")
            if os.path.exists(out):
                print(f"[skip] {out} already exists", flush=True)
                continue
            train_one(args.dataset, args.term, side, seed, args.gpu, out_root, args.option)


if __name__ == "__main__":
    main()
