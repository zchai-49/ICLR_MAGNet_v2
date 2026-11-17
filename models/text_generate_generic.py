# text_generate_generic.py (rebuttal, 2026-11-12): the text_generate.py pipeline (same LLM, prompts, statistics and
# BERT [CLS]) for the generic wide-CSV datasets (data/dataset/generic/<name>.csv + .json). 10-minute data (Weather) is
# averaged to hourly before the statistics, so the 'hourly' lags of compute_series_stats (24/168/720) keep their meaning.
# Unlike text_generate.py, the FULL text is logged (no 1500-character cut).
#   python3 models/text_generate_generic.py --name Weather --folder weather --steps-per-hour 6
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import text_generate as TG  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--folder", required=True)
ap.add_argument("--steps-per-hour", type=int, default=1)
# 2026-11-16 text ablation (synthetic): --variant <llm> swaps only the generator (text_generate_variants.HF_LLM; same
# prompts, 4-bit NF4, greedy, BERT [CLS]) and writes <name>_<variant>_* instead of <name>_new_*
ap.add_argument("--variant", default="new"); a = ap.parse_args()
from text_prompt_variants import PROMPT_VARIANTS, install as install_prompt   # prompt ablation (2026-11-16)
need_llm = True
if a.variant in PROMPT_VARIANTS:
    need_llm = install_prompt(a.variant, TG)
elif a.variant != "new":
    from text_generate_variants import HF_LLM
    TG.LLM_MODEL = HF_LLM[a.variant]
root = os.path.join(os.path.dirname(TG.TEXT_EMB_ROOT), "dataset", "generic")
info = json.load(open(os.path.join(root, f"{a.name}.json")))
all_vars = sorted(info["endo"] + info["exo"], key=lambda v: v["index"])
data = pd.read_csv(os.path.join(root, f"{a.name}.csv"))[[v["name"] for v in all_vars]]
train = data[:int(len(data) * 0.7)]
if a.steps_per_hour > 1:
    train = train.groupby(train.index // a.steps_per_hour).mean()
llm_tok, llm_model = TG.load_llm() if need_llm else (None, None); bert_tok, bert_model = TG.load_bert()
save = os.path.join(TG.TEXT_EMB_ROOT, a.folder); os.makedirs(save, exist_ok=True)
log = {"dataset": info["context"], "variables": [v["name"] for v in all_vars], "short": [], "long": []}
endo_v = all_vars[0]
for term in ("short", "long"):
    E, X = [], []
    for k, v in enumerate(all_vars):
        stats = TG.compute_series_stats(train.iloc[:, k].values, sampling_freq="hourly", term=term)
        r = TG.retrieve_text(v["name"], v["description"], v["type"], stats, llm_tok, llm_model, info["context"], term,
                             endo_name=endo_v["name"] if v["type"] == "exo" else None,
                             endo_desc=endo_v["description"] if v["type"] == "exo" else None)
        e = TG.bert_encode(r["text"], bert_tok, bert_model)
        (E if v["type"] == "endo" else X).append(e)
        log[term].append({"variable": v["name"], "type": v["type"], "text": r["text"]})
        print(f"[{a.name}/{term}] {k + 1}/{len(all_vars)} {v['name']}", flush=True)
    torch.save(torch.stack(E), f"{save}/{a.name}_{a.variant}_{term}_endo_vq_emb.pt")
    torch.save(torch.stack(X), f"{save}/{a.name}_{a.variant}_{term}_exo_vq_emb.pt")
json.dump(log, open(f"{save}/{a.name}_{a.variant}_text_log.json", "w"), indent=2)
print(f"[{a.name}/{a.variant}] done", flush=True)
