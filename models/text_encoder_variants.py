# text_encoder_variants.py  (rebuttal, 2026-11-12: "replace BERT with other language models")
# Re-encodes the SAME per-variable text with 5 encoders, so only the encoder changes. The text is the shipped
# *_new_text_log.json text (logged up to 1500 characters; the full shipped text of longer answers was not kept and
# greedy re-generation is not bit-identical on this machine), so BERT is re-encoded on it too (variant encbert).
#   encbert     bert-base-uncased, [CLS]                    (the shipped encoder)
#   encminilm   sentence-transformers/all-MiniLM-L6-v2, mean pooling, 384-d zero-padded to 768
#   encroberta  roberta-base, mean pooling
#   encmpnet    sentence-transformers/all-mpnet-base-v2, mean pooling
#   encbge      BAAI/bge-base-en-v1.5, [CLS] + L2 normalisation (its documented pooling)
# All: max 512 tokens. Output data/text_emb/<folder>/<prefix>_<variant>_<term>_{endo,exo}_vq_emb.pt
#   python3 models/text_encoder_variants.py --datasets 5 7 8 9 10 12
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import text_generate as TG  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ENCODERS = {"encbert": ("bert-base-uncased", "cls"), "encminilm": ("sentence-transformers/all-MiniLM-L6-v2", "mean"),
            "encroberta": ("roberta-base", "mean"), "encmpnet": ("sentence-transformers/all-mpnet-base-v2", "mean"),
            "encbge": ("BAAI/bge-base-en-v1.5", "cls_norm")}


def encode(text, tok, model, pooling):
    x = tok(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        h = model(**x).last_hidden_state
    if pooling == "mean":
        m = x["attention_mask"].unsqueeze(-1).float(); e = (h * m).sum(1) / m.sum(1)
    else:
        e = h[:, 0]
        if pooling == "cls_norm":
            e = F.normalize(e, dim=-1)
    e = e.squeeze(0)
    return torch.cat([e, e.new_zeros(TG.BERT_DIM - e.shape[-1])]) if e.shape[-1] < TG.BERT_DIM else e


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", type=int, nargs="*", default=[])
    ap.add_argument("--generic", nargs="*", default=[], help="synthetic datasets (data/text_emb/synthetic/<name>, full logged text)")
    ap.add_argument("--variants", nargs="+", default=list(ENCODERS)); a = ap.parse_args()
    from transformers import AutoModel, AutoTokenizer
    for v in a.variants:
        name, pooling = ENCODERS[v]
        tok, model = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).eval()
        for idx in a.datasets + a.generic:
            if isinstance(idx, str):
                folder, base = f"synthetic/{idx}", idx
            else:
                folder, base, _, _ = TG._DATASET_SPECS[idx]
            p = f"{TG.TEXT_EMB_ROOT}/{folder}/{base}"
            log = json.load(open(f"{p}_new_text_log.json"))
            for term in ("short", "long"):
                endo = [encode(z["text"], tok, model, pooling) for z in log[term] if z["type"] == "endo"]
                exo = [encode(z["text"], tok, model, pooling) for z in log[term] if z["type"] == "exo"]
                torch.save(torch.stack(endo), f"{p}_{v}_{term}_endo_vq_emb.pt")
                torch.save(torch.stack(exo), f"{p}_{v}_{term}_exo_vq_emb.pt")
            print(f"[{base}/{v}] done", flush=True)
