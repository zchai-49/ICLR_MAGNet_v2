# text_prompt_variants.py (2026-11-16, prompt ablation): only the PROMPT changes; LLM (Llama-3.2-3B-Instruct, 4-bit NF4,
# greedy), BERT [CLS] encoder, statistics and model are the deployed ones. install(variant) patches text_generate (TG)
# so that TG.retrieve_text -> TG.llm_knowledge builds the variant's text; returns whether an LLM must be loaded.
#
#   psimple    P1  plain request, no statistics, no question list:
#                  "Describe the time series <name> (<desc>) [and how it relates to <target>] in 2-3 sentences."
#   pnocausal  P2  the deployed prompt without the exogenous->target causal questions (statistics + the variable's own
#                  dynamics questions kept; target-variable prompts unchanged)
#   pnostats   P3  the deployed prompt with the statistics block removed (names, descriptions, questions kept)
#   pstats     P4  no LLM: the statistics block itself (as rendered into the deployed prompt) is the text
#   pnamedesc  P5  no LLM: "<name>: <description>"
#
# Used by models/text_generate_variants.py (real datasets) and models/text_generate_generic.py (synthetic).
PROMPT_VARIANTS = ("psimple", "pnocausal", "pnostats", "pstats", "pnamedesc")


def install(variant, TG):
    if variant not in PROMPT_VARIANTS:
        raise ValueError(variant)
    orig = TG.llm_knowledge

    def knowledge(var_name, var_desc, var_type, stats, llm_tok, llm_model, DATASET_CONTEXT, term="short",
                  endo_name=None, endo_desc=None):
        if variant == "pnamedesc":
            return f"{var_name}: {var_desc}"
        if variant == "pstats":
            return TG.format_stats_str(stats, term).strip()
        if variant == "pnostats":
            return orig(var_name, var_desc, var_type, {**stats}, llm_tok, llm_model, DATASET_CONTEXT, term,
                        endo_name=endo_name, endo_desc=endo_desc)
        if variant == "psimple":
            rel = f" and how it relates to {endo_name}" if var_type != "endo" and endo_name else ""
            prompt = (f"Describe the time series {var_name} ({var_desc}){rel} in 2-3 sentences.\n\n"
                      "Reply with ONLY the description. No preamble.")
            return TG.llm_generate(prompt, llm_tok, llm_model)
        # pnocausal: target prompts unchanged; exogenous prompts keep only the variable's own dynamics questions
        if var_type == "endo":
            return orig(var_name, var_desc, var_type, stats, llm_tok, llm_model, DATASET_CONTEXT, term)
        stats_str = TG.format_stats_str(stats, term)
        if term == "short":
            prompt = f"""You are an expert in {DATASET_CONTEXT}.

Variable: {var_name} (exogenous input variable)
Description: {var_desc}
Target variable to forecast: {endo_name} — {endo_desc}
{stats_str}
Based on both the description and the observed statistics,
answer concisely (1-2 sentences each):
1. What is the short-term temporal pattern of {var_name} itself —
   persistent, volatile, or irregular? Does lag-1 autocorrelation
   support this?
2. How predictable is {var_name} step-to-step, and at what
   sub-daily or daily time scale does it change most?

Reply with a paragraph containing ONLY the answers. No preamble."""
        else:
            prompt = f"""You are an expert in {DATASET_CONTEXT}.

Variable: {var_name} (exogenous input variable for long-horizon forecasting)
Description: {var_desc}
Target variable to forecast: {endo_name} — {endo_desc}
{stats_str}
Based on both the description and the observed long-term statistics,
answer concisely (1-2 sentences each):
1. What is the long-term trend of {var_name} itself — increasing,
   decreasing, or stable over months? Does it repeat reliably
   at weekly or monthly cycles?

Reply with a paragraph containing ONLY the answers. No preamble."""
        return TG.llm_generate(prompt, llm_tok, llm_model)

    TG.llm_knowledge = knowledge
    if variant == "pnostats":
        TG.format_stats_str = lambda stats, term="short": "\n"     # same mechanism as the earlier names_only variant
    return variant not in ("pstats", "pnamedesc")
