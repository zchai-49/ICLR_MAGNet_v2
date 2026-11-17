"""Shared helpers for the experiment scripts: test-set evaluation of a MAGNet+DLinear variant (an option of
train/train_options.py, trained into <ckpt_root>/<term>_<dataset>_<side>/seed_<seed>/ckpt.pt) as the prediction ensemble over
the deployed sides x the given seeds - the same ensemble as the reported numbers."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("configs", "train", "scripts", "data", "models"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import lag1_config as C  # noqa: E402
import train_options as O  # noqa: E402
from MAGNET_data import get_dataloaders  # noqa: E402
from evaluate import predict  # noqa: E402

REAL = ["BE", "DE", "FR", "NP", "PJM", "Energy", "Colbun", "Rapel", "Sdwpfh1", "Sdwpfh2", "Sdwpfm1", "Sdwpfm2"]
SYN = ["Syn5", "Syn20", "Syn50", "Syn20weak", "Syn20null"]
SEEDS = (0, 42, 123, 2024, 3407)
RUNS = os.path.join(ROOT, "runs")
FINAL_CKPTS = os.path.join(ROOT, "checkpoints", "magnet_dlinear")


def ensemble(ds, term, option, ckpt_root, seeds=SEEDS, device="cpu", env=None):
    """Test MAE/MSE of the ensemble over deployed sides x seeds of <option> (model built exactly as in training; `env`
    temporarily sets environment variables read at model construction, e.g. MAGNET_SYN_TEXT).
    Returns (mae, mse, predictions, targets) or None if a deployed side has no checkpoint."""
    old = {k: os.environ.get(k) for k in (env or {})}
    os.environ.update(env or {})
    try:
        c = f"{ds}/{term}"; e = C.cell_config(ds, term); i = C.NAME_TO_ID[ds]; L, H = C.get_seq_horizon(i, term)
        side_preds, t = [], None
        for side in e["deployed_sides"]:
            sel = e["sides"][side]["selected"]; saved = json.loads(json.dumps(C.DEPLOYMENT[c]["sides"][side]))
            O.apply_candidate(ds, term, side, sel["hp_rank"], sel["fix"], sel["regime"])
            P = []
            for sd in seeds:
                ck = os.path.join(ckpt_root, f"{term}_{ds}_{side}", f"seed_{sd}", "ckpt.pt")
                if not os.path.exists(ck):
                    continue
                model, pl = O.build_option_model(ds, term, side, option)
                model.load_state_dict(torch.load(ck, map_location=device), strict=True)
                te = get_dataloaders(npz_path=C.csv_path(i), sensor_id=0, seq_len=L, forecast_horizon=H, patch_len=pl,
                                     batch_size=64, dataset_id=i)[2]
                with torch.no_grad():
                    p, t = predict(model.to(device).eval(), te, i, term, device)
                P.append(np.asarray(p))
            C.DEPLOYMENT[c]["sides"][side] = saved
            if not P:
                return None
            side_preds.append(np.mean(P, 0))
        pred = np.mean(side_preds, 0); t = np.asarray(t); err = pred - t
        return float(np.abs(err).mean()), float((err ** 2).mean()), pred, t
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def md_table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])
