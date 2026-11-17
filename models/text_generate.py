# text_generate.py
# Generates the precomputed text embeddings MAGNet's causal-graph module
# reads at init time (see MAGNET_graph_only.py's _load_text_embeddings(),
# default node_text_fusion="qk_bias") -- data/text_emb/<dataset>/ ships
# these already-generated for all 12 datasets; run this file only to
# regenerate them or to produce them for a new dataset (its raw CSV +
# exo-variable JSON must be in data/dataset/ -- see README.md).
#
# Two-stage pipeline per variable:
#   1. A local LLM (Llama-3.2-3B-Instruct, 4-bit quantized) answers a fixed
#      set of questions about the variable (its temporal pattern, and for
#      exo variables, its causal relationship to the endo/target variable),
#      conditioned on the variable's JSON description AND numeric statistics
#      computed from the training split (compute_series_stats()).
#   2. BERT (bert-base-uncased) encodes that generated text; the pooled
#      [CLS] embedding is what MAGNET_graph_only.py loads and projects
#      (via TextProjection, models/text_projection.py) into the causal
#      graph's Q/K-bias pathway.
#
# Heavyweight optional dependency: downloading Llama-3.2-3B-Instruct
# requires accepting its license on HuggingFace and a few GB of disk/GPU
# memory (further reduced to ~2GB via 4-bit quantization here). Only needed
# if you're generating text embeddings for a dataset beyond the one this
# repo ships them for -- see requirements.txt's "optional: text_generate.py"
# section for the extra pip installs (transformers, bitsandbytes, accelerate).
#
# Usage:
#   python3 text_generate.py --datasets 5 6 7    # BE, DE, FR
#   python3 text_generate.py --datasets 10       # just Energy (already shipped, will overwrite)
#
# Output (per dataset, per term):
#   data/text_emb/<folder>/<prefix>_new_<term>_endo_vq_emb.pt   [1, 768]
#   data/text_emb/<folder>/<prefix>_new_<term>_exo_vq_emb.pt    [n_exo, 768]
#   data/text_emb/<folder>/<prefix>_new_text_log.json           generated text, for inspection

import argparse
import gc
import json
import time
import warnings
warnings.filterwarnings('ignore')

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file lives in models/, one level down)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "models"))
sys.path.insert(0, os.path.join(ROOT_DIR, "data"))
DATASET_ROOT  = os.path.join(ROOT_DIR, "data", "dataset")
TEXT_EMB_ROOT = os.path.join(ROOT_DIR, "data", "text_emb")
print(f"MAGNet root: {ROOT_DIR}")

import torch
import numpy as np
import pandas as pd
from MAGNET_data import get_dataset_info
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BertTokenizer,
    BertModel,
    BitsAndBytesConfig,
)

D_MODEL     = 64      # must match MAGNet d_model
BERT_DIM    = 768     # bert-base-uncased CLS token size
LLM_MODEL   = 'meta-llama/Llama-3.2-3B-Instruct'
BERT_MODEL  = 'bert-base-uncased'
MAX_NEW_TOKENS = 512

def load_llm():
    """Load Llama 3B in 4-bit quantization for efficiency."""
    print("Loading Llama 3B...")

    # 4-bit quantization — reduces memory from ~6GB to ~2GB
    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_type       = 'nf4',
        bnb_4bit_compute_dtype    = torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model     = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        quantization_config = bnb_config,
        device_map          = 'auto',
    )
    model.eval()
    print("Llama 3B loaded.\n")
    return tokenizer, model

def llm_generate(prompt: str, tokenizer, model) -> str:
    """Run one LLM inference call and return the generated text."""
    messages = [{'role': 'user', 'content': prompt}]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        return_tensors        = 'pt',
        return_dict           = True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **input_ids,
            max_new_tokens = MAX_NEW_TOKENS,
            do_sample      = False,   # greedy for reproducibility
            pad_token_id   = tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_ids["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def compute_series_stats(
    series        : np.ndarray,
    sampling_freq : str = 'hourly',   # 'hourly' or 'daily'
    term          : str = 'short',    # 'short' or 'long'
) -> dict:
    """
    Compute statistics relevant to the forecasting term.

    Short-term stats: capture local persistence and volatility
    Long-term stats:  capture trend direction and seasonal structure

    Args:
        series        : 1D numpy array, training split only
        sampling_freq : 'hourly' or 'daily'
        term          : 'short' or 'long'
    """
    n = len(series)

    base = {
        'mean'    : float(np.mean(series)),
        'std'     : float(np.std(series)),
        'min'     : float(np.min(series)),
        'max'     : float(np.max(series)),
        'range'   : float(np.max(series) - np.min(series)),
        'cv'      : float(np.std(series) / (abs(np.mean(series)) + 1e-8)),
    }

    if term == 'short':
        acf_lag1 = float(np.corrcoef(series[:-1], series[1:])[0, 1]) \
                   if n > 2 else None

        if sampling_freq == 'hourly':
            cycle_lag   = 24
            cycle_label = 'acf_lag24 (daily cycle)'
        else:
            cycle_lag   = 7
            cycle_label = 'acf_lag7 (weekly cycle)'

        acf_cycle = float(np.corrcoef(
            series[:-cycle_lag], series[cycle_lag:])[0, 1]
        ) if n > cycle_lag * 2 else None

        base.update({
            'acf_lag1'    : acf_lag1,
            'acf_cycle'   : acf_cycle,
            'acf_label'   : cycle_label,
        })

    else:
        x = np.arange(n)
        slope = float(np.polyfit(x, series, 1)[0])

        if sampling_freq == 'hourly':
            weekly_lag   = 168    # 7 days × 24 hours
            weekly_label = 'acf_lag168 (weekly cycle)'
        else:
            weekly_lag   = 7
            weekly_label = 'acf_lag7 (weekly cycle)'

        acf_weekly = float(np.corrcoef(
            series[:-weekly_lag], series[weekly_lag:])[0, 1]
        ) if n > weekly_lag * 2 else None

        if sampling_freq == 'hourly':
            monthly_lag   = 720   # ~30 days × 24 hours
            monthly_label = 'acf_lag720 (monthly cycle)'
        else:
            monthly_lag   = 30
            monthly_label = 'acf_lag30 (monthly cycle)'

        acf_monthly = float(np.corrcoef(
            series[:-monthly_lag], series[monthly_lag:])[0, 1]
        ) if n > monthly_lag * 2 else None

        if sampling_freq == 'hourly':
            period = 168   # weekly period for hourly data
        else:
            period = 30    # monthly period for daily data

        if n >= period * 2:
            n_complete = (n // period) * period
            reshaped   = series[:n_complete].reshape(-1, period)
            period_means = reshaped.mean(axis=1)
            seasonality_strength = float(
                np.std(period_means) / (np.std(series) + 1e-8)
            )
        else:
            seasonality_strength = None

        trend_component = slope * x + (np.mean(series) - slope * np.mean(x))
        residual        = series - trend_component
        trend_strength  = float(
            1 - np.var(residual) / (np.var(series) + 1e-8)
        )

        base.update({
            'slope'               : slope,
            'acf_weekly'          : acf_weekly,
            'acf_weekly_label'    : weekly_label,
            'acf_monthly'         : acf_monthly,
            'acf_monthly_label'   : monthly_label,
            'seasonality_strength': seasonality_strength,
            'trend_strength'      : trend_strength,
        })

    return base

def format_stats_str(stats: dict, term: str = 'short') -> str:
    """Render computed series stats into the numeric block spliced into LLM prompts."""
    if term == 'short':
        return f"""
Observed short-term statistics from training data:
- Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}
- Range: [{stats['min']:.4f}, {stats['max']:.4f}]
- Lag-1 autocorrelation (step-to-step persistence): {stats.get('acf_lag1', 'N/A')}
- {stats.get('acf_label', 'Seasonal autocorrelation')}: {stats.get('acf_cycle', 'N/A')}
- Coefficient of variation (relative volatility): {stats['cv']:.4f}
"""
    else:
        slope = stats.get('slope', 0)
        if abs(slope) < 1e-6:
            trend_dir = 'approximately flat (no significant trend)'
        elif slope > 0:
            trend_dir = f'increasing (slope={slope:.6f} per time step)'
        else:
            trend_dir = f'decreasing (slope={slope:.6f} per time step)'

        ss = stats.get('seasonality_strength', None)
        if ss is not None:
            if ss > 0.3:
                seasonal_desc = f'strong ({ss:.3f})'
            elif ss > 0.1:
                seasonal_desc = f'moderate ({ss:.3f})'
            else:
                seasonal_desc = f'weak ({ss:.3f})'
        else:
            seasonal_desc = 'N/A'

        ts = stats.get('trend_strength', None)
        trend_strength_desc = f'{ts:.3f}' if ts is not None else 'N/A'

        return f"""
Observed long-term statistics from training data:
- Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}
- Overall trend: {trend_dir}
- Trend strength (R² of linear fit): {trend_strength_desc}
- {stats.get('acf_weekly_label', 'Weekly autocorrelation')}: {stats.get('acf_weekly', 'N/A')}
- {stats.get('acf_monthly_label', 'Monthly autocorrelation')}: {stats.get('acf_monthly', 'N/A')}
- Seasonality strength (period-to-period variation): {seasonal_desc}
"""

def llm_knowledge(
    var_name       : str,
    var_desc       : str,
    var_type       : str,
    stats          : dict,
    llm_tok,
    llm_model,
    DATASET_CONTEXT: str,
    term           : str = 'short',
    endo_name      : str = None,
    endo_desc      : str = None,
) -> str:

    stats_str = format_stats_str(stats, term)

    if term == 'short':

        if var_type == 'endo':
            prompt = f"""You are an expert in {DATASET_CONTEXT}.

Variable: {var_name} (target variable to forecast)
Description: {var_desc}
{stats_str}
Based on both the description and the observed statistics,
answer concisely (1-2 sentences each):
1. What is the typical short-term temporal pattern — is it
   persistent, mean-reverting, or irregular? Does the lag-1
   autocorrelation support this?
2. How noisy is this variable within a day or week?
   Does the coefficient of variation reflect high volatility?
3. At what sub-daily or daily time scale does it change most?
4. What short-term conditions cause sudden spikes or drops?

Reply with a paragraph containing ONLY the answers. No preamble."""

        else:  # exo, short-term — causal questions FIRST
            prompt = f"""You are an expert in {DATASET_CONTEXT}.

Variable: {var_name} (exogenous input variable)
Description: {var_desc}
Target variable to forecast: {endo_name} — {endo_desc}
{stats_str}
Based on both the description and the observed statistics,
answer concisely (1-2 sentences each):
1. How strongly does {var_name} causally influence {endo_name}?
   Rate the causal strength as strong/moderate/weak and explain
   the physical or economic mechanism behind this relationship.
2. Does an increase in {var_name} typically increase or decrease
   {endo_name}? Is this relationship linear or are there
   nonlinear regimes or threshold effects?
3. Are there specific threshold values of {var_name} beyond which
   its effect on {endo_name} changes dramatically?
4. Is {var_name} a leading indicator (changes before {endo_name})
   or a coincident indicator (changes simultaneously)?
   What is the typical lag in hours or days?
5. What is the short-term temporal pattern of {var_name} itself —
   persistent, volatile, or irregular? Does lag-1 autocorrelation
   support this?
6. How predictable is {var_name} step-to-step, and at what
   sub-daily or daily time scale does it change most?

Reply with a paragraph containing ONLY the answers. No preamble."""

    else:  # long-term

        if var_type == 'endo':
            prompt = f"""You are an expert in {DATASET_CONTEXT}.

Variable: {var_name} (target variable to forecast over long horizons)
Description: {var_desc}
{stats_str}
Based on both the description and the observed long-term statistics,
answer concisely (1-2 sentences each):
1. What is the long-term trend of this variable — increasing,
   decreasing, or flat? Does the trend strength support a
   meaningful directional movement over months?
2. How strong is the seasonal pattern — does this variable
   repeat reliably week-over-week or month-over-month?
   Do the weekly and monthly autocorrelations support this?
3. Over a horizon of weeks to months, what drives the
   largest changes in this variable?
4. What long-term conditions or structural changes cause
   this variable to deviate from its seasonal pattern?

Reply with a paragraph containing ONLY the answers. No preamble."""

        else:  # exo, long-term — causal questions FIRST
            prompt = f"""You are an expert in {DATASET_CONTEXT}.

Variable: {var_name} (exogenous input variable for long-horizon forecasting)
Description: {var_desc}
Target variable to forecast: {endo_name} — {endo_desc}
{stats_str}
Based on both the description and the observed long-term statistics,
answer concisely (1-2 sentences each):
1. How strongly does {var_name} causally influence {endo_name}
   over long horizons of weeks to months? Rate as strong/moderate/weak
   and explain the physical or economic mechanism.
2. Does an increase in {var_name} typically increase or decrease
   {endo_name} over the long term? Is this relationship linear
   or are there nonlinear regimes or threshold effects?
3. Are there threshold values of {var_name} beyond which its
   long-term effect on {endo_name} changes dramatically?
4. Is the long-term causal influence of {var_name} on {endo_name}
   stronger or weaker than its short-term influence?
   Under what long-term conditions does it become a strong
   predictor vs a weak predictor?
5. What is the long-term trend of {var_name} itself — increasing,
   decreasing, or stable over months? Does it repeat reliably
   at weekly or monthly cycles?
6. How does the seasonal pattern of {var_name} align with or
   drive the seasonal pattern of {endo_name}?

Reply with a paragraph containing ONLY the answers. No preamble."""

    return llm_generate(prompt, llm_tok, llm_model)

def retrieve_llm_knowledge(
    var_name      : str,
    var_desc      : str,
    var_type,
    stats,
    llm_tok,
    llm_model,
    DATASET_CONTEXT,
    term,
    endo_name=None,
    endo_desc=None,
    ) -> str:

    return (llm_knowledge(var_name, var_desc, var_type, stats, llm_tok, llm_model, DATASET_CONTEXT, term, endo_name=endo_name, endo_desc=endo_desc), var_name)

def retrieve_text(var_name: str, var_desc: str, var_type, stats,
                  llm_tok, llm_model, DATASET_CONTEXT, term, endo_name=None, endo_desc=None) -> dict:
    text, matched = retrieve_llm_knowledge(
        var_name, var_desc, var_type, stats, llm_tok, llm_model, DATASET_CONTEXT, term, endo_name=endo_name, endo_desc=endo_desc,
    )
    return {
        'variable' : matched,
        'text'  : text
    }

def load_bert():
    """Load BERT tokenizer and model."""
    print("Loading BERT...")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    model     = BertModel.from_pretrained(BERT_MODEL)
    model.eval()
    print("BERT loaded.\n")
    return tokenizer, model


def bert_encode(text: str, bert_tok, bert_model) -> torch.Tensor:
    """
    Encode text with BERT and return the [CLS] token embedding.

    Returns: [768] tensor
    """
    inputs = bert_tok(
        text,
        return_tensors  = 'pt',
        truncation      = True,
        max_length      = 512,
        padding         = True,
    )
    with torch.no_grad():
        outputs = bert_model(**inputs)

    cls_emb = outputs.last_hidden_state[:, 0, :]   # [1, 768]
    return cls_emb.squeeze(0)                      # [768]


# dataset_id -> (folder under data/text_emb/, filename prefix, CSV path under
# data/dataset/, columns to select after pivoting). Matches
# MAGNET_graph_only.py's own folder/prefix mapping exactly (must stay in
# sync -- that's the only other place this mapping is duplicated).
_DATASET_SPECS = {
    5:  ("BE",           "be",       f"{DATASET_ROOT}/EPF/BE.csv",
         ["OT", "System load forecast", "Generation forecast"]),
    6:  ("DE",           "de",       f"{DATASET_ROOT}/EPF/DE.csv",
         ["OT", "Ampirion zonal load forecast", "Wind power forecast"]),
    7:  ("FR",           "fr",       f"{DATASET_ROOT}/EPF/FR.csv",
         ["OT", "System load forecast", "Generation forecast"]),
    8:  ("NP",           "np",       f"{DATASET_ROOT}/EPF/NP.csv",
         ["OT", "Grid load forecast", "Wind power forecast"]),
    9:  ("PJM",          "pjm",      f"{DATASET_ROOT}/EPF/PJM.csv",
         ["OT", "System load forecast", "Zonal COMED load foecast"]),
    10: ("energy",       "Energy",   f"{DATASET_ROOT}/energy/Energy.csv",
         ["Thermoelectric", "Battery", "Geothermal", "Hydroelectric", "Solar", "Wind"]),
    11: ("Colbun",       "Colbun",   f"{DATASET_ROOT}/hydro/Colbun.csv",
         ["Water_Level_daily_avg", "Afluente (m3/seg)_daily_avg", "Precipitation_daily_sum"]),
    12: ("Rapel",        "Rapel",    f"{DATASET_ROOT}/hydro/Rapel.csv",
         ["Water_Level_daily_avg", "Afluente (m3/seg)_daily_avg", "Precipitation_daily_sum"]),
    13: ("wind/Sdwpfh1", "Sdwpfh1",  f"{DATASET_ROOT}/wind/Sdwpfh1.csv",
         ["Patv", "T2m", "Sp", "RelH", "Wspd_w", "Wdir_w", "Tp"]),
    14: ("wind/Sdwpfh2", "Sdwpfh2",  f"{DATASET_ROOT}/wind/Sdwpfh2.csv",
         ["Patv", "T2m", "Sp", "RelH", "Wspd_w", "Wdir_w", "Tp"]),
    15: ("wind/Sdwpfm1", "Sdwpfm1",  f"{DATASET_ROOT}/wind/Sdwpfm1.csv",
         ["Patv", "T2m", "Sp", "RelH", "Wspd_w", "Wdir_w", "Tp"]),
    16: ("wind/Sdwpfm2", "Sdwpfm2",  f"{DATASET_ROOT}/wind/Sdwpfm2.csv",
         ["Patv", "T2m", "Sp", "RelH", "Wspd_w", "Wdir_w", "Tp"]),
}

_DAILY_DATASETS = {11, 12}  # Colbun, Rapel


def generate_text_embeddings(index, llm_tok, llm_model, bert_tok, bert_model):
    """Full text generation pipeline for one dataset -- generates and saves
    the endo/exo pooled text embeddings for both terms."""
    if index not in _DATASET_SPECS:
        print(f"Unknown dataset id: {index}")
        return

    folder, base_name, csv_path, columns = _DATASET_SPECS[index]
    save_prefix = f"{TEXT_EMB_ROOT}/{folder}/{base_name}"
    os.makedirs(f"{TEXT_EMB_ROOT}/{folder}", exist_ok=True)

    # dataset id 13 covers the shared wind exo-variable JSON config
    # (data/dataset/wind/Sdwpf.json) -- see MAGNET_data.py's get_dataset_info().
    info_index = 13 if index in (13, 14, 15, 16) else index
    DATASET_INFO = get_dataset_info(info_index)
    DATASET_CONTEXT = DATASET_INFO['context']
    ALL_VARS        = DATASET_INFO['all_vars']

    data = pd.read_csv(csv_path)
    data["idx"] = data.groupby("cols").cumcount()
    sensor_data = data.pivot(index='idx', columns='cols', values='data').rename_axis(None).rename_axis(None, axis=1)
    sensor_data = sensor_data[columns]

    sampling_freq = 'daily' if index in _DAILY_DATASETS else 'hourly'
    train_end  = int(len(sensor_data) * 0.7)
    train_data = sensor_data[:train_end]

    text_log = {
        'dataset'  : DATASET_CONTEXT,
        'variables': [v['name'] for v in ALL_VARS],
        'short'    : [],   # variable-level text
        'long'     : [],   # relationship-level text
    }

    for term in ['short', 'long']:
        endo_vq_embeddings = []
        exo_vq_embeddings  = []

        for var in ALL_VARS:
            series = train_data.iloc[:, var['index']].values
            stats  = compute_series_stats(series, sampling_freq=sampling_freq, term=term)

            endo_var = ALL_VARS[0]
            retrieved = retrieve_text(
                var['name'], var['description'], var['type'],
                stats,
                llm_tok, llm_model, DATASET_CONTEXT, term,
                endo_name=endo_var['name'] if var['type'] == 'exo' else None,
                endo_desc=endo_var['description'] if var['type'] == 'exo' else None,
            )

            bert_vq_emb = bert_encode(retrieved['text'], bert_tok, bert_model)

            if var['type'] == 'endo':
                endo_vq_embeddings.append(bert_vq_emb)
            else:
                exo_vq_embeddings.append(bert_vq_emb)

            text_log[term].append({
                'variable': var['name'],
                'type'    : var['type'],
                'text'    : retrieved['text'][:1500],
            })

            time.sleep(1)

        endo_vq_emb = torch.stack(endo_vq_embeddings, dim=0)
        exo_vq_emb  = torch.stack(exo_vq_embeddings,  dim=0)

        torch.save(endo_vq_emb, f'{save_prefix}_new_{term}_endo_vq_emb.pt')
        torch.save(exo_vq_emb,  f'{save_prefix}_new_{term}_exo_vq_emb.pt')

    with open(f'{save_prefix}_new_text_log.json', 'w') as f:
        json.dump(text_log, f, indent=2)

    print(f"[{base_name}] done — endo {endo_vq_emb.shape}, exo {exo_vq_emb.shape}, "
          f"saved to {TEXT_EMB_ROOT}/{folder}/")


def cleanup():
    """Release GPU memory between datasets."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', type=int, nargs='+', default=list(range(5, 17)),
                         help='Dataset IDs (5-16, see README.md\'s "Dataset IDs"). '
                              'Requires that dataset\'s raw CSV to already be in data/dataset/.')
    args = parser.parse_args()

    llm_tok, llm_model   = load_llm()
    bert_tok, bert_model = load_bert()

    for i in args.datasets:
        generate_text_embeddings(i, llm_tok, llm_model, bert_tok, bert_model)
        cleanup()

    print("\nText generation complete.")
