# text_generate_variants.py
# Text-ablation embeddings for the rebuttal (kaYf Q1, text vs. statistics). Same pipeline as text_generate.py
# (LLM answers fixed questions per variable -> encoder -> pooled vector), with one thing changed per variant:
#
#   regen       identical to the shipped "new" text (same LLM, prompts, BERT); checks reproducibility on this machine
#   anon        variable names, descriptions and domain replaced by neutral labels ("Variable 1", "the target
#               variable", "time-series forecasting"); the computed statistics are kept -> does the text carry
#               external (named) knowledge, or only what the statistics already say?
#   names_only  names, descriptions and domain kept; the statistics block removed from every prompt
#   qwen        Qwen2.5-3B-Instruct instead of Llama-3.2-3B-Instruct (same prompts, same BERT)
#   minilm      the shipped "new" text re-encoded with all-MiniLM-L6-v2 (mean-pooled, 384-d, zero-padded to 768 so
#               the model's 768-input text projection is unchanged; zero padding = a 384-input linear map)
#
# Output: data/text_emb/<folder>/<prefix>_<variant>_<term>_{endo,exo}_vq_emb.pt and <prefix>_<variant>_text_log.json,
# loadable with MAGNETGraphOnly(text_variant=<variant>).
#
#   python3 models/text_generate_variants.py --variant anon --datasets 10 11 16

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import text_generate as TG  # noqa: E402

import pandas as pd  # noqa: E402
import torch  # noqa: E402

VARIANTS = ("regen", "anon", "names_only", "qwen", "minilm", "llama8b", "qwen7b", "deepseek", "gpt35", "llama1b")
# LLM comparison (rebuttal, 2026-11-12): same prompts, same BERT [CLS] encoder, same 4-bit NF4 loading and greedy
# decoding (max 512 new tokens); only the generator changes. API models use temperature 0 and max_tokens 512.
HF_LLM = {"qwen": "Qwen/Qwen2.5-3B-Instruct", "llama8b": os.environ.get("LLAMA8B_REPO", "meta-llama/Llama-3.1-8B-Instruct"),
          "qwen7b": "Qwen/Qwen2.5-7B-Instruct", "llama1b": "meta-llama/Llama-3.2-1B-Instruct", "deepseek": os.environ.get("DEEPSEEK_REPO", "deepseek-ai/deepseek-llm-7b-chat")}
API_LLM = {"gpt35": ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-3.5-turbo"),
           "deepseek_api": ("https://api.deepseek.com", "DEEPSEEK_API_KEY", "deepseek-chat")}


def api_generate_factory(variant):
    from openai import OpenAI
    base, key_env, model = API_LLM[variant]
    client = OpenAI(base_url=base, api_key=os.environ[key_env])

    def gen(prompt, tokenizer=None, model_=None):
        r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                           temperature=0, max_tokens=TG.MAX_NEW_TOKENS)
        return r.choices[0].message.content.strip()
    return gen
QWEN_MODEL = "Qwen/Qwen2.5-3B-Instruct"
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def anonymize(info):
    """Neutral labels: no names, descriptions or domain; the statistics still enter through the prompt."""
    info = dict(info)
    endo, exo = info["endo"], info["exo"]
    info["endo"] = [dict(v, name="the target variable", description="A time series to forecast.") for v in endo]
    info["exo"] = [dict(v, name=f"Variable {i + 1}", description="A time series observed alongside the target.")
                   for i, v in enumerate(exo)]
    info["all_vars"] = info["endo"] + info["exo"]
    info["context"] = "time-series forecasting"
    return info


def load_minilm():
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MINILM_MODEL)
    model = AutoModel.from_pretrained(MINILM_MODEL).eval()
    return tok, model


def minilm_encode(text, tok, model):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    with torch.no_grad():
        h = model(**inputs).last_hidden_state                          # [1, T, 384]
    m = inputs["attention_mask"].unsqueeze(-1).float()
    emb = (h * m).sum(1) / m.sum(1)                                    # mean pooling, as sentence-transformers
    return torch.cat([emb.squeeze(0), emb.new_zeros(TG.BERT_DIM - emb.shape[-1])])


def generate(index, variant, llm_tok, llm_model, enc_tok, enc_model):
    folder, base_name, csv_path, columns = TG._DATASET_SPECS[index]
    save_prefix = f"{TG.TEXT_EMB_ROOT}/{folder}/{base_name}"
    info = TG.get_dataset_info(13 if index in (13, 14, 15, 16) else index)
    if variant == "anon":
        info = anonymize(info)
    all_vars, context = info["all_vars"], info["context"]

    shipped_log = None
    if variant == "minilm":
        shipped_log = json.load(open(f"{save_prefix}_new_text_log.json"))

    data = pd.read_csv(csv_path)
    data["idx"] = data.groupby("cols").cumcount()
    sensor = data.pivot(index="idx", columns="cols", values="data").rename_axis(None).rename_axis(None, axis=1)[columns]
    freq = "daily" if index in TG._DAILY_DATASETS else "hourly"
    train = sensor[:int(len(sensor) * 0.7)]

    log = {"dataset": context, "variant": variant, "variables": [v["name"] for v in all_vars], "short": [], "long": []}
    for term in ("short", "long"):
        endo_e, exo_e = [], []
        for k, var in enumerate(all_vars):
            if variant == "minilm":
                text = shipped_log[term][k]["text"]
                emb = minilm_encode(text, enc_tok, enc_model)
            else:
                stats = TG.compute_series_stats(train.iloc[:, var["index"]].values, sampling_freq=freq, term=term)
                endo = all_vars[0]
                text = TG.retrieve_text(var["name"], var["description"], var["type"], stats, llm_tok, llm_model,
                                        context, term,
                                        endo_name=endo["name"] if var["type"] == "exo" else None,
                                        endo_desc=endo["description"] if var["type"] == "exo" else None)["text"]
                emb = TG.bert_encode(text, enc_tok, enc_model)
            (endo_e if var["type"] == "endo" else exo_e).append(emb)
            log[term].append({"variable": var["name"], "type": var["type"], "text": text[:1500]})
        torch.save(torch.stack(endo_e), f"{save_prefix}_{variant}_{term}_endo_vq_emb.pt")
        torch.save(torch.stack(exo_e), f"{save_prefix}_{variant}_{term}_exo_vq_emb.pt")
    json.dump(log, open(f"{save_prefix}_{variant}_text_log.json", "w"), indent=2)
    print(f"[{base_name}/{variant}] done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    from text_prompt_variants import PROMPT_VARIANTS, install as install_prompt   # prompt ablation (2026-11-16)
    ap.add_argument("--variant", required=True, choices=VARIANTS + ("deepseek_api",) + PROMPT_VARIANTS)
    ap.add_argument("--datasets", type=int, nargs="+", default=[10, 11, 16])
    a = ap.parse_args()
    if a.variant == "names_only":
        TG.format_stats_str = lambda stats, term="short": "\n"
    if a.variant in HF_LLM:
        TG.LLM_MODEL = HF_LLM[a.variant]
    llm_tok = llm_model = None
    if a.variant in PROMPT_VARIANTS:
        if install_prompt(a.variant, TG):
            llm_tok, llm_model = TG.load_llm()
        enc_tok, enc_model = TG.load_bert()
    elif a.variant in API_LLM:
        TG.llm_generate = api_generate_factory(a.variant)            # retrieve_text calls llm_generate by name
        enc_tok, enc_model = TG.load_bert()
    elif a.variant != "minilm":
        llm_tok, llm_model = TG.load_llm()
        enc_tok, enc_model = TG.load_bert()
    else:
        enc_tok, enc_model = load_minilm()
    for idx in a.datasets:
        generate(idx, a.variant, llm_tok, llm_model, enc_tok, enc_model)
        TG.cleanup()
