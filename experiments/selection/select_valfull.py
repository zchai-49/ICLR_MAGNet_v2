#!/usr/bin/env python3
"""Implements PREREG_valfull.md step by step (validation only; test data are never loaded here).
    python3 experiments/selection/select_valfull.py <step>
steps: screen | mech | hp | hppick | fix | fixpick | seed | final
Each step reads the results of the previous one, writes experiments/selection/valfull_jobs_<next>.json and updates
experiments/selection/valfull/state.json. Prints one summary line last.
"""
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
DS = ["BE", "DE", "FR", "NP", "PJM", "Energy", "Colbun", "Rapel", "Sdwpfh1", "Sdwpfh2", "Sdwpfm1", "Sdwpfm2"]
SHORTLIST = {"short": ["crossattn_mq", "dual_crossattn", "dual_gated_scale_crossattn", "dual_gated_scale", "dual_only"],
             "long": ["dual_only", "gated_mixing_only", "postlinear_film_only", "crossattn_mq"]}
DEF_HP, N_HP = 8, 18
# Amendment 2026-11-10 17:30 (before any screen score was examined): the mechanism choice uses the 6 representative
# datasets of the original screen (all four domains), not all 12, to save compute.
REP = ["NP", "PJM", "Energy", "Colbun", "Sdwpfh1", "Sdwpfm2"]
# Amendment 2026-11-10 21:30 (before any hp result was ranked): the hparam grid is cut to 9 configs = the default (8)
# + the 8 configs most often in the validation top-3 of the earlier hparam search (configs/valsel_hparam_provenance.json).
HP_SET = [0, 1, 2, 4, 6, 7, 8, 12, 13]
FOLDS = ("A", "B")
SEEDS_FINAL = (0, 42, 123, 2024, 3407)
CELLS = [f"{d}/{t}" for t in ("short", "long") for d in DS]
STF = os.path.join(HERE, "valfull", "state.json")
# Rebuttal datasets (2026-11-12): the same protocol on other cells, with its own state / job files / final deployment.
# MAGNET_VF_CELLS="Weather/short,..."  MAGNET_VF_TAG=rebuttal. The injection mechanisms per term are the ones the
# representative-dataset screen chose (copied from valfull/state.json), so these runs start at step "mech" (hp jobs).
TAG = os.environ.get("MAGNET_VF_TAG")
if TAG:
    CELLS = os.environ["MAGNET_VF_CELLS"].split(",")
    STF = os.path.join(HERE, "valfull", f"state_{TAG}.json")
os.makedirs(os.path.dirname(STF), exist_ok=True)
state = json.load(open(STF)) if os.path.exists(STF) else {}
if TAG and "sides" not in state:
    state["sides"] = json.load(open(os.path.join(HERE, "valfull", "state.json")))["sides"]
    state["mech_source"] = "valfull/state.json (representative-dataset screen)"


def path(fold, cell, side, r, fix, regime, seed):
    ds, term = cell.split("/")
    sub = "valfull" if fold == "A" else f"valfull_fold{fold}"
    return os.path.join(HERE, sub, f"hp{r}_{fix}_{regime}", f"{term}_{ds}_{side}", f"seed_{seed}")


def val(fold, cell, side, cfg, seed=0):
    f = os.path.join(path(fold, cell, side, *cfg, seed), "result.json")
    if not os.path.exists(f):
        raise FileNotFoundError(f)
    return json.load(open(f))["best_val_mae"]


def norm(scores):                               # {key: val} -> {key: val / median}
    m = float(np.median(list(scores.values())))
    return {k: v / m for k, v in scores.items()}


def jobs_out(name, jobs):
    p = os.path.join(HERE, f"valfull_jobs_{TAG + '_' if TAG else ''}{name}.json")
    json.dump(jobs, open(p, "w"))
    return f"{len(jobs)} jobs -> {os.path.relpath(p, ROOT)}"


def save():
    json.dump(state, open(STF, "w"), indent=1)


def fixes_for(side):
    f = ["none", "patch8", "mlp_backbone", "residual_head"]
    return f + (["deep_film"] if "crossattn" in side else [])


def cs_list():
    return [(c, s) for c in CELLS for s in state["sides"][c.split("/")[1]]]


step = sys.argv[1]
if step == "screen":
    jobs = [[c, m, DEF_HP, "none", "default", 0, f] for c in CELLS for m in SHORTLIST[c.split("/")[1]] for f in FOLDS
            if c.split("/")[0] in REP]
    print(jobs_out("screen", jobs))

elif step == "mech" and TAG:
    save()
    jobs = [[c, s, r, "none", "default", 0, "A"] for c, s in cs_list() for r in HP_SET
            if not os.path.exists(os.path.join(path("A", c, s, r, "none", "default", 0), "result.json"))]
    print(f"sides (from the screen) short={state['sides']['short']} long={state['sides']['long']}; " + jobs_out("hp", jobs))

elif step == "mech":
    state["mech_scores"], state["sides"] = {}, {}
    for term in ("short", "long"):
        per = {m: [] for m in SHORTLIST[term]}
        for d in REP:
            for f in FOLDS:
                n = norm({m: val(f, f"{d}/{term}", m, (DEF_HP, "none", "default")) for m in SHORTLIST[term]})
                for m in per:
                    per[m].append(n[m])
        sc = {m: float(np.mean(v)) for m, v in per.items()}
        state["mech_scores"][term] = dict(sorted(sc.items(), key=lambda x: x[1]))
        state["sides"][term] = sorted(sc, key=sc.get)[:2]
    save()
    jobs = [[c, s, r, "none", "default", 0, "A"] for c, s in cs_list() for r in HP_SET
            if not os.path.exists(os.path.join(path("A", c, s, r, "none", "default", 0), "result.json"))]
    print(f"sides short={state['sides']['short']} long={state['sides']['long']}; " + jobs_out("hp", jobs))

elif step == "hp":
    state["hp_top3"] = {}
    jobs = []
    for c, s in cs_list():
        v = {r: val("A", c, s, (r, "none", "default")) for r in HP_SET}
        top = sorted(v, key=v.get)[:3]
        state["hp_top3"][f"{c}|{s}"] = top
        jobs += [[c, s, r, "none", "default", 0, "B"] for r in top
                 if not os.path.exists(os.path.join(path("B", c, s, r, "none", "default", 0), "result.json"))]
    save()
    print(jobs_out("hpB", jobs))

elif step == "hppick":
    state["hp"] = {}
    jobs = []
    for c, s in cs_list():
        top = state["hp_top3"][f"{c}|{s}"]
        nA = norm({r: val("A", c, s, (r, "none", "default")) for r in top})
        nB = norm({r: val("B", c, s, (r, "none", "default")) for r in top})
        r = min(top, key=lambda k: (nA[k] + nB[k]) / 2)
        state["hp"][f"{c}|{s}"] = r
        jobs += [[c, s, r, fx, rg, 0, "A"] for fx in fixes_for(s) for rg in ("default", "warmup")
                 if (fx, rg) != ("none", "default")]
    save()
    print(jobs_out("fix", jobs))

elif step == "fix":
    state["fix_top3"] = {}
    jobs = []
    for c, s in cs_list():
        r = state["hp"][f"{c}|{s}"]
        v = {(fx, rg): val("A", c, s, (r, fx, rg)) for fx in fixes_for(s) for rg in ("default", "warmup")}
        top = sorted(v, key=v.get)[:3]
        state["fix_top3"][f"{c}|{s}"] = [list(t) for t in top]
        jobs += [[c, s, r, fx, rg, 0, "B"] for fx, rg in top
                 if not os.path.exists(os.path.join(path("B", c, s, r, fx, rg, 0), "result.json"))]
    save()
    print(jobs_out("fixB", jobs))

elif step == "fixpick":
    jobs = []
    for c, s in cs_list():
        r = state["hp"][f"{c}|{s}"]
        top = [tuple(t) for t in state["fix_top3"][f"{c}|{s}"]]
        jobs += [[c, s, r, fx, rg, 42, f] for fx, rg in top for f in FOLDS]
    print(jobs_out("seed", jobs))

elif step == "seed":
    state["chosen"], state["chosen_scores"] = {}, {}
    for c, s in cs_list():
        r = state["hp"][f"{c}|{s}"]
        top = [tuple(t) for t in state["fix_top3"][f"{c}|{s}"]]
        sc = {}
        mA = float(np.median([np.mean([val("A", c, s, (r, *t), sd) for sd in (0, 42)]) for t in top]))
        mB = float(np.median([np.mean([val("B", c, s, (r, *t), sd) for sd in (0, 42)]) for t in top]))
        for t in top:
            a = [val("A", c, s, (r, *t), sd) / mA for sd in (0, 42)]
            b = [val("B", c, s, (r, *t), sd) / mB for sd in (0, 42)]
            sc[t] = (np.mean(a) + np.mean(b)) / 2 + float(np.std(a, ddof=1))
        best = min(sc, key=sc.get)
        state["chosen"][f"{c}|{s}"] = [r, *best]
        state["chosen_scores"][f"{c}|{s}"] = {"/".join(k): round(float(v), 4) for k, v in sc.items()}
    save()
    print(f"configs chosen for {len(state['chosen'])} cell-sides; next: final (side choice + 5-seed jobs)")

elif step == "final":
    os.environ["MAGNET_DEPLOYMENT"] = os.path.join(ROOT, "configs", "deployment_valfull.json")
    os.environ["MAGNET_VAL_LOOKBACK"] = "1"
    for d in ("configs", "train", "scripts", "data"):
        sys.path.insert(0, os.path.join(ROOT, d))
    import torch
    import lag1_config as C
    import train_options as O
    from MAGNET_data import get_dataloaders
    from evaluate import predict
    torch.set_num_threads(4)

    def preds(fold, c, s, cfg, seed):
        ds, term = c.split("/"); r, fx, rg = cfg
        if fold == "B":
            os.environ["MAGNET_FOLD"] = "B"
        else:
            os.environ.pop("MAGNET_FOLD", None)
        saved = json.loads(json.dumps(C.DEPLOYMENT[c]["sides"][s]))
        O.apply_candidate(ds, term, s, r, fx, rg)
        model, patch_len = O.build_option_model(ds, term, s, "detach_gg")
        model.load_state_dict(torch.load(os.path.join(path(fold, c, s, r, fx, rg, seed), "ckpt.pt"), map_location="cpu"))
        i = C.NAME_TO_ID[ds]; L, H = C.get_seq_horizon(i, term)
        va = get_dataloaders(npz_path=C.csv_path(i), sensor_id=0, seq_len=L, forecast_horizon=H, patch_len=patch_len,
                             batch_size=64, dataset_id=i)[1]
        with torch.no_grad():
            p, t = predict(model.eval(), va, i, term, torch.device("cpu"))
        C.DEPLOYMENT[c]["sides"][s] = saved
        return np.asarray(p), np.asarray(t)

    dep = json.load(open(os.path.join(ROOT, "configs", "deployment_valfull.json")))
    state["deployed"], state["side_scores"] = {}, {}
    jobs = []
    for c in CELLS:
        sides = state["sides"][c.split("/")[1]]
        e = {}
        for f in FOLDS:
            pm, t = {}, None
            for s in sides:
                ps = []
                for sd in (0, 42):
                    p, t = preds(f, c, s, tuple(state["chosen"][f"{c}|{s}"]), sd); ps.append(p)
                pm[s] = np.mean(ps, 0)
            opts = [tuple(x) for k in (1, 2) for x in itertools.combinations(sides, k)]
            err = {o: float(np.abs(np.mean([pm[s] for s in o], 0) - t).mean()) for o in opts}
            mu = float(np.mean(list(err.values())))
            for o in opts:
                e.setdefault(o, []).append(err[o] / mu)
        sc = {o: float(np.mean(v)) for o, v in e.items()}
        pick = min(sc, key=sc.get)
        state["deployed"][c] = list(pick)
        state["side_scores"][c] = {"+".join(o): round(v, 4) for o, v in sc.items()}
        dc = dep[c]; dc["deployed_sides"] = list(pick); dc["selection"] = "PREREG_valfull.md"
        for s in sides:
            r, fx, rg = state["chosen"][f"{c}|{s}"]
            dc["sides"][s]["hparams"] = dc["sides"][s]["hparam_candidates"][r]
            dc["sides"][s]["selected"] = {"hp_rank": r, "fix": fx, "regime": rg}
        dc["sides"] = {s: dc["sides"][s] for s in sides}
        for s in pick:
            r, fx, rg = state["chosen"][f"{c}|{s}"]
            jobs += [[c, s, r, fx, rg, sd, "A"] for sd in SEEDS_FINAL
                     if not os.path.exists(os.path.join(path("A", c, s, r, fx, rg, sd), "result.json"))]
        print(f"{c:14s} deployed {'+'.join(pick):45s} " +
              "  ".join(f"{s}=hp{state['chosen'][f'{c}|{s}'][0]}/{state['chosen'][f'{c}|{s}'][1]}/{state['chosen'][f'{c}|{s}'][2]}"
                        for s in sides), flush=True)
    save()
    if TAG:     # only this run's cells go to its own final deployment file
        dep = {c: dep[c] for c in CELLS}
    json.dump(dep, open(os.path.join(ROOT, "configs", f"deployment_{TAG}_final.json" if TAG else
                                     "deployment_valfull_final.json"), "w"), indent=1)
    print(jobs_out("final", jobs))
else:
    sys.exit(f"unknown step {step}")
