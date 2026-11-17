"""Official MAGNet configuration (final, SIG v2). Imported first by configs/lag1_config.py, models/MAGNET_graph_only.py and
models/causal_alternatives.py, so every entry point runs the final model without any environment variable. Each value is a
DEFAULT only (os.environ.setdefault): exporting the variable overrides it, which is how the ablations / controls are run.

  MAGNET_DEPLOYMENT       configs/deployment_final.json  validation-selected MAGNet+DLinear deployment, all 12 real + 5 synthetic
                                                          datasets x short/long (ensemble sides, hyperparameters, enhancers, regime)
  MAGNET_ON_TOP=detach_gg  MAGNet+DLinear uses the sparse-input Granger (SIG) graph with detached weights and the fixed
                          cross-fit Granger gate (configs/granger_gain_crossfit.json)
  MAGNET_PNP_DETACH_GG=1   the same graph inside every plug-and-play backbone wrapper
  MAGNET_LAG0_NUISANCE=1   the lag-0 (same-patch) groups of SIG are fitted and penalised, but only the LAGGED groups set the
                          attention weights (lag 0 is a nuisance control)
  MAGNET_SIG_PREWHITEN=1   SIG inputs are seasonally differenced (s = 24 hourly, 48 for 30-min Sdwpfm, 7 for daily Colbun/Rapel)
  MAGNET_SYN_TEXT=1        synthetic datasets use their (statistics-only) LLM text like the real ones
  MAGNET_VAL_LOOKBACK=1    validation windows may look back into the end of train (standard border); targets stay in val
"""
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULTS = {
    "MAGNET_DEPLOYMENT": os.path.join(_ROOT, "configs", "deployment_final.json"),
    "MAGNET_ON_TOP": "detach_gg",
    "MAGNET_PNP_DETACH_GG": "1",
    "MAGNET_LAG0_NUISANCE": "1",
    "MAGNET_SIG_PREWHITEN": "1",
    "MAGNET_SYN_TEXT": "1",
    "MAGNET_VAL_LOOKBACK": "1",
}
for _k, _v in DEFAULTS.items():
    os.environ.setdefault(_k, _v)
