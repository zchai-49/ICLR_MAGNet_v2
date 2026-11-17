"""Shared evaluation helpers: history-only prediction (+ test-time noise averaging on the noisy datasets, inverse first
differencing for long-term Colbun) and MAE/MSE. Imported by the trainers and every evaluation script; the evaluation
entry point of the final model is scripts/evaluate_magnet.py."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "configs"))
import torch  # noqa: E402
import lag1_config as C  # noqa: E402


@torch.no_grad()
def predict(model, loader, dataset_id, term, device):
    preds, targs = [], []
    for exo_p, endo_p, target, _mean, _std, _tf, last_known in loader:
        exo_p, endo_p = exo_p.to(device), endo_p.to(device)
        # Only the observed history is given to the model; future exogenous values are never used.
        out = model(exo_p, endo_p)
        if out.dim() == 3:
            out = out.squeeze(-1)
        if dataset_id in C.TTA_DATASETS:
            outs = [out]
            for _ in range(2):
                o = model(exo_p + 0.01 * torch.randn_like(exo_p), endo_p + 0.01 * torch.randn_like(endo_p))
                outs.append(o.squeeze(-1) if o.dim() == 3 else o)
            out = torch.stack(outs).mean(0)
        out = out.cpu()
        if dataset_id in C.DIFF_DATASETS and term == "long":
            out = last_known.unsqueeze(1) + out.cumsum(-1)
            target = last_known.unsqueeze(1) + target.cumsum(-1)
        preds.append(out)
        targs.append(target)
    return torch.cat(preds), torch.cat(targs)


def metrics(pred, targ):
    return {"mae": torch.mean(torch.abs(pred - targ)).item(),
            "mse": torch.mean((pred - targ) ** 2).item()}
