#!/usr/bin/env python3
"""Invariants of the official (final) MAGNet package.

    python3 tests/test_invariants.py

  * the official configuration is active without any environment variable (models/final_defaults.py)
  * every shipped MAGNet+DLinear checkpoint loads strictly into the final model: sparse-input Granger (SIG) graph, lag 0 as a
    nuisance (not in the attention weights), seasonal pre-whitening, detached weights, fixed cross-fit gate
  * no model / script ever receives future exogenous values (DAG / GCGNet: only as a training-time teacher)
  * the data loader emits the history window only; the PnP wrappers obey the same rules; every ablation option builds
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("configs", "train", "scripts", "data", "models"):
    sys.path.insert(0, os.path.join(ROOT, _d))
for _k in [k for k in os.environ if k.startswith("MAGNET_")]:
    os.environ.pop(_k)                      # test the shipped defaults, not the caller's environment
import torch  # noqa: E402
import lag1_config as C  # noqa: E402
import train_options as O  # noqa: E402
import final_defaults  # noqa: E402


def expect_error(fn, msg):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(msg)


def test_final_defaults():
    for k, v in final_defaults.DEFAULTS.items():
        assert os.environ.get(k) == v, k
    assert len(C.DEPLOYMENT) == 34, "12 real + 5 synthetic datasets x short/long"


def test_shipped_checkpoints():
    gg = json.load(open(os.path.join(ROOT, "configs", "granger_gain_crossfit.json")))
    paths = sorted(glob.glob(os.path.join(ROOT, "checkpoints", "magnet_dlinear", "*", "seed_*", "ckpt.pt")))
    assert paths, "no checkpoints found"
    for p in paths:
        term, ds, side = os.path.basename(os.path.dirname(os.path.dirname(p))).split("_", 2)
        c = f"{ds}/{term}"; sel = C.DEPLOYMENT[c]["sides"][side]["selected"]
        saved = json.loads(json.dumps(C.DEPLOYMENT[c]["sides"][side]))
        O.apply_candidate(ds, term, side, sel["hp_rank"], sel["fix"], sel["regime"])
        model, _ = O.build_option_model(ds, term, side, "detach_gg")
        model.load_state_dict(torch.load(p, map_location="cpu"), strict=True)
        C.DEPLOYMENT[c]["sides"][side] = saved
        g = model.magnet_graph; est = g.causal_estimator
        assert type(est).__name__ == "SparseInputGranger" and est.lag0_in_weights is False, p
        assert getattr(g, "detach_causal", False), p
        assert getattr(g, "causal_gate_mode", None) == "fixed", p
        gate = float(g.causal_gate_value)
        assert abs(gate - min(max(gg[c]["ridge"]["gain"], 0.0), 1.0)) < 1e-5, (p, gate)
    return len(paths)


def test_no_future_exogenous():
    model, patch_len = O.build_option_model("Energy", "short", "dual_only", "detach_gg")
    exo = torch.randn(2, 14, patch_len, 5)
    endo = torch.randn(2, 14, patch_len, 1)
    model.eval()
    with torch.no_grad():
        model(exo, endo)  # history only: fine
    expect_error(lambda: model(exo, endo, exog_future=torch.randn(2, 24, 5)),
                 "passing future exogenous values must be rejected")
    # no script may hand future exogenous values to a model (DAG's training-time teacher is checked separately)
    for path in glob.glob(os.path.join(ROOT, "**", "*.py"), recursive=True):
        if os.path.basename(path) in ("test_invariants.py", "hparam_MAGNET_DAG.py", "MAGNET_with_DAG.py",
                                            "hparam_MAGNET_GCGNet.py", "MAGNET_with_GCGNet.py"):
            continue  # DAG: training-time teacher only, covered by test_dag_wrapper / test_gcgnet_wrapper
        if os.sep.join(("backbones", "ts_benchmark")) in path:
            continue  # vendored DAG backbone (its training-time teacher path)
        src = open(path).read()
        assert not re.search(r"exog_future\s*=\s*(?!None)", src.replace("exog_future   : torch.Tensor = None", "")), path


def test_loader_emits_only_history():
    from MAGNET_data import get_dataloaders
    seq_len, horizon = C.get_seq_horizon(10, "short")
    loader = get_dataloaders(npz_path=C.csv_path(10), sensor_id=0, seq_len=seq_len, forecast_horizon=horizon,
                             patch_len=12, batch_size=4, dataset_id=10)[2]
    batch = next(iter(loader))
    assert len(batch) == 7, "batch = (exo_patches, endo_patches, target, mean, std, time_features, last_known)"
    exo_patches = batch[0]
    assert exo_patches.shape[1] * exo_patches.shape[2] == seq_len, "exogenous input must be the history window only"


def test_pnp_wrappers():
    """The plug-and-play wrappers (PatchTST, TimeKAN, TimeXer, TFT) obey the same two rules."""
    import inspect
    for d in ("pnp", "backbones"):
        sys.path.insert(0, os.path.join(ROOT, d))
    import MAGNET_with_PatchTST, MAGNET_with_TimeKAN, MAGNET_with_TimeXer, MAGNET_with_TFT
    import hparam_MAGNET_PatchTST, hparam_MAGNET_TimeKAN, hparam_MAGNET_TFT, hparam_MAGNET_TimeXer
    wrappers = [MAGNET_with_PatchTST.MAGNETWithPatchTST, MAGNET_with_TimeKAN.MAGNETWithTimeKAN,
                MAGNET_with_TFT.MAGNETWithTFT, MAGNET_with_TimeXer.TimeXerWithMAGNET]
    for cls in wrappers:
        fwd, init = inspect.getsource(cls.forward), inspect.getsource(cls.__init__)
        assert "if exog_future is not None" in fwd and "raise ValueError" in fwd, f"{cls.__name__} must reject future exog"
        assert "'ngc_lag', 1" in init, f"{cls.__name__} must default to ngc_lag=1"
    for mod in (hparam_MAGNET_PatchTST, hparam_MAGNET_TimeKAN, hparam_MAGNET_TFT, hparam_MAGNET_TimeXer):
        assert inspect.signature(mod.train_one_config).parameters["ngc_lag"].default == 1, mod.__name__


def test_dag_wrapper():
    """DAG uses future exogenous values only as a training-time teacher; never at evaluation."""
    import inspect
    for d in ("pnp", "backbones"):
        sys.path.insert(0, os.path.join(ROOT, d))
    import MAGNET_with_DAG, hparam_MAGNET_DAG
    fwd, init = inspect.getsource(MAGNET_with_DAG.MAGNETWithDAG.forward), inspect.getsource(MAGNET_with_DAG.MAGNETWithDAG.__init__)
    assert "exog_future is not None and not self.training" in fwd and "raise ValueError" in fwd
    assert "'ngc_lag', 1" in init and "infer_use_future" in init
    assert inspect.signature(hparam_MAGNET_DAG.train_one_config).parameters["ngc_lag"].default == 1
    assert "dag_cfg.infer_use_future = False" in inspect.getsource(hparam_MAGNET_DAG.train_one_config)
    from MAGNET_data import get_dataloaders
    seq_len, horizon = C.get_seq_horizon(10, "short")
    loaders = get_dataloaders(npz_path=C.csv_path(10), sensor_id=0, seq_len=seq_len, forecast_horizon=horizon,
                              patch_len=12, batch_size=4, dataset_id=10)
    assert all(len(next(iter(l))) == 7 for l in loaders), "default loaders carry no future exogenous values"


def test_gcgnet_wrapper():
    """GCGNet (like DAG) uses the true future only as a training-time teacher; eval is history-only and rejects it."""
    import hparam_MAGNET_GCGNet as G
    from MAGNET_with_GCGNet import MAGNETWithGCGNet
    seq_len, horizon, n_exo = 168, 24, 5
    hp = G.GCGNET_HPARAMS["Energy"]

    class Cfg: pass
    cfg = Cfg()
    cfg.seq_len, cfg.pred_len, cfg.patch_len, cfg.enc_in, cfg.series_dim = seq_len, horizon, horizon, n_exo + 1, 1
    cfg.d_model, cfg.d_ff, cfg.rank, cfg.e_layers, cfg.n_heads, cfg.dropout = 16, 32, hp["rank"], 1, 2, 0.0
    cfg.use_norm, cfg.use_future_exog = True, False
    mc = {"n_exo": n_exo, "patch_len": 12, "num_patches": 14, "d_model": 32, "n_heads": 2, "dataset": 10, "term": "short"}
    m = MAGNETWithGCGNet(mc, cfg, mode=2)
    assert m.magnet_graph.ngc_lag == 1
    x, y = torch.randn(2, 14, 12, n_exo), torch.randn(2, 14, 12, 1)
    m.train()
    pred, aux = m(x, y, exog_future=torch.randn(2, horizon, n_exo), endo_future=torch.randn(2, horizon))
    assert pred.shape == (2, horizon)
    m.eval()
    with torch.no_grad():
        assert m(x, y).shape == (2, horizon)
    expect_error(lambda: m(x, y, exog_future=torch.randn(2, horizon, n_exo)), "GCGNet eval must reject future exog")
    assert G.NGC_LAG == 1


def test_ablation_options():
    """Every ablation / control column builds and runs on the final configuration."""
    c = "Energy/short"
    side = C.DEPLOYMENT[c]["deployed_sides"][0]; sel = C.DEPLOYMENT[c]["sides"][side]["selected"]
    O.apply_candidate("Energy", "short", side, sel["hp_rank"], sel["fix"], sel["regime"])
    for opt in ("detach_gg", "no_text", "no_exo", "dlinear_only", "exo_mlp", "xattn_dlinear"):
        model, pl = O.build_option_model("Energy", "short", side, opt)
        model.eval()
        x, y = torch.randn(2, 168 // pl, pl, 5), torch.randn(2, 168 // pl, pl, 1)
        with torch.no_grad():
            out = model(x, y)
        assert out.squeeze(-1).shape == (2, 24), opt


if __name__ == "__main__":
    test_final_defaults()
    print("ok  official configuration active without environment variables (SIG v2, final deployment of 34 cells)")
    n = test_shipped_checkpoints()
    print(f"ok  {n} shipped MAGNet+DLinear checkpoints load strictly (SIG, lag-0 nuisance, detached weights, cross-fit gate)")
    test_no_future_exogenous()
    print("ok  models reject future exogenous input; no script passes it")
    test_loader_emits_only_history()
    print("ok  the data loader emits the history window only (no future exogenous values)")
    test_pnp_wrappers()
    print("ok  PnP wrappers (PatchTST, TimeKAN, TimeXer, TFT) reject future exogenous input")
    test_dag_wrapper()
    print("ok  DAG: future exogenous values only as a training-time teacher; eval is history-only")
    test_gcgnet_wrapper()
    print("ok  GCGNet: future values only as a training-time teacher; eval is history-only")
    test_ablation_options()
    print("ok  ablation / control options build and run")
