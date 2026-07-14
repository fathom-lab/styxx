"""B7 -- does the erasure bound hold at 3B? The static subspace-erasure attack (B2) at scale.

The scale flank, the reviewers' first objection, the binding gate on the paper. Runs cycle 36's
EXACT static erasure attack (RMU-style projection-to-zero of the per-layer DoM+logistic gold
subspace + knowledge-replay CE) on Qwen2.5-3B-Instruct. Single-variable vs B2 by import:
b2_subspace_erasure's gold_subspace and train_erasure run UNCHANGED -- this file only rebinds the
family constants (MODEL / SCAN / DEPLOY) on the imported modules, asserts the rebinding took, and
adds the feasibility instrumentation the arc now owes (full training hist persisted, per-cell peak
VRAM + wall-clock, pre-committed per-cell OOM branch).

Layer map (frozen): proportional-depth image of the decisive 1.5B family's constants.
1.5B (28 layers): scan [12,14,16,18,20,22], deploy 18 -> 3B (36 layers): scan [15,18,21,23,26,28],
deploy 23 (each L_3B = round(L_1p5B * 36/28); deploy 18 -> 23.1 -> 23).

Cells: seeds {0,1} x alpha {1.0,4.0} (B2's grid, for the direct 1.5B-vs-3B split). Frozen verdict:
ERASED__read_neq_write_BROKEN_3B / SURVIVES__vs_subspace_erasure_3B /
PARTIAL__erasure_attribution_split_3B; VOIDs namespaced VOID_B7__*.

CRASH-SAFETY (added 2026-07-14, after the overnight launch died at cell 3/4 and lost every
completed cell -- the harness wrote its result JSON only at the very end). Each completed cell is
appended to a JSONL cache the instant it finishes and the clean-guard block is cached once; a
resumed launch skips (seed,alpha) cells already cached and computes the SAME frozen verdict over
the union. This is a RUNNER change only: the attack (B2.gold_subspace / B2.train_erasure), the audit
(HPC.*), every guard/bar, and the verdict block are byte-identical -- the verdict function operates
on the cell list irrespective of provenance. See b7_checkpoint.py for the disclosed cross-launch
non-determinism (recompute-on-resume of the clean guard/subspace, already covered by the prereg's
"bf16 non-deterministic; one run per cell"). Smoke runs never touch the caches.

PREREG: papers/calib-poison-general/PREREG_B7_erasure_3b_2026_07_13.md (science frozen with this file)
Usage: python papers/calib-poison-general/b7_erasure_3b.py [--smoke]
"""
from __future__ import annotations
import argparse, importlib.util, json, sys, gc, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


B2 = _load("b2_subspace_erasure", HERE / "b2_subspace_erasure.py")   # attack, frozen
HPC = B2.HPC                                                         # audit surface, frozen
CKPT = _load("b7_checkpoint", HERE / "b7_checkpoint.py")            # crash-safety, science-neutral
E1, ATK, SYK, FND = HPC.E1, HPC.ATK, HPC.SYK, HPC.FND
SUBSAMPLE_SEED = HPC.SUBSAMPLE_SEED

# ---- the ONLY deltas vs B2: the family constants (frozen in the prereg) ----
MODEL_3B = "Qwen/Qwen2.5-3B-Instruct"
SCAN_3B = [15, 18, 21, 23, 26, 28]   # proportional image of [12,14,16,18,20,22] (28 -> 36 layers)
DEPLOY_3B = 23                        # proportional image of 18
for mod in (HPC, B2):
    mod.MODEL = MODEL_3B
    mod.SCAN = SCAN_3B
    mod.DEPLOY = DEPLOY_3B
assert B2.SCAN == SCAN_3B and HPC.SCAN == SCAN_3B and HPC.DEPLOY == DEPLOY_3B and B2.MODEL == MODEL_3B

SEEDS = B2.SEEDS      # [0, 1]
ALPHAS = B2.ALPHAS    # [1.0, 4.0]
LAM = B2.LAM          # 1.0
STEPS = B2.STEPS      # 300


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    steps = 20 if a.smoke else STEPS
    seeds = [0] if a.smoke else SEEDS
    alphas = [1.0] if a.smoke else ALPHAS
    tag = "SMOKE_INVALID " if a.smoke else ""
    suffix = "_SMOKE_INVALID" if a.smoke else ""
    result_path = HERE / f"b7_erasure_3b_result{suffix}.json"

    # ---- crash-safe resume (real runs only; smoke never touches the caches) ----
    if a.smoke:
        cached_cells, done_keys, cache_path, clean_path = [], set(), None, None
    else:
        cache_path = CKPT.cells_cache_path(result_path)
        clean_path = CKPT.clean_cache_path(result_path)
        cached_cells, done_keys = CKPT.load_cached_cells(cache_path)
        if cached_cells:
            CKPT.compact_cells(cache_path, cached_cells)   # drop any torn tail a crash left
            print(f"[resume] {len(cached_cells)} cached cell(s) {sorted(done_keys)} -- will skip", flush=True)
    todo = CKPT.remaining_cells(seeds, alphas, done_keys)
    clean_block = None if a.smoke else CKPT.load_clean(clean_path)

    # ---- clean guard + gold subspace: recompute fresh when there is work to do, else from cache ----
    if todo or clean_block is None:
        attack, calib, evl, disjoint = E1.three_way_split(0, a.smoke)
        sub_idx = sorted(np.random.default_rng(SUBSAMPLE_SEED).choice(len(attack), len(calib), replace=False).tolist())
        attack_sub = [attack[i] for i in sub_idx]
        print(f"[split] ATTACK {len(attack)} CALIB {len(calib)} EVAL {len(evl)} disjoint={disjoint} "
              f"ATTACK-SUB {len(attack_sub)} (seed {SUBSAMPLE_SEED}) MODEL={MODEL_3B} "
              f"SCAN={SCAN_3B} DEPLOY={DEPLOY_3B}", flush=True)

        tok = AutoTokenizer.from_pretrained(MODEL_3B)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"

        def base_reload():
            return AutoModelForCausalLM.from_pretrained(
                MODEL_3B, dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True)

        base = base_reload(); base.eval()
        n_layers = base.config.num_hidden_layers
        assert n_layers == 36 and max(SCAN_3B) <= n_layers, f"layer map invalid for n_layers={n_layers}"
        tids, fids = SYK.tf_token_ids(tok)
        subs = B2.gold_subspace(base, tok, attack)
        a_txt = [c for c, _ in attack]; a_y = np.array([l for _, l in attack])
        a_res = SYK.resid_all(base, tok, a_txt, [DEPLOY_3B])
        d23 = FND.dom_direction(a_res[DEPLOY_3B], a_y)
        if HPC.frozen18_read(base, tok, d23, evl) < 0.5:
            d23 = -d23
        clean_frozen = HPC.frozen18_read(base, tok, d23, evl)
        clean_priv, clean_rand = HPC.family13_audit(base, tok, calib, evl)
        clean_know = HPC.eval_knowledge(base, tok, evl, tids, fids)
        clean_naive6 = HPC.naive_dom6(base, tok, attack, evl)
        guard_ok = clean_priv >= 0.75 and clean_know >= 0.80 and bool(disjoint)
        print(f"[clean] private13={clean_priv:.4f} (rand {clean_rand:.4f}) knowledge={clean_know:.4f} "
              f"frozen_deploy={clean_frozen:.4f} naive6={clean_naive6:.4f} "
              f"subspace_ranks={{{', '.join(f'{L}:{subs[L].shape[1]}' for L in SCAN_3B)}}} guard={guard_ok}", flush=True)
        correct_true = torch.tensor(tids, device="cuda"); correct_false = torch.tensor(fids, device="cuda")
        del base; gc.collect(); torch.cuda.empty_cache()

        clean_block = {"clean_private13_auroc": round(clean_priv, 4), "clean_rand_auroc": round(clean_rand, 4),
                       "clean_eval_knowledge": round(clean_know, 4),
                       "clean_frozen_deploy_read": round(clean_frozen, 4),
                       "clean_naive_dom6_auroc": round(clean_naive6, 4),
                       "clean_guard_ok": bool(guard_ok),
                       "subspace_rank_per_layer": {str(L): int(subs[L].shape[1]) for L in SCAN_3B},
                       "attack_subsample_indices": sub_idx,
                       "split_disjoint": bool(disjoint)}
        if not a.smoke:
            CKPT.save_clean(clean_path, clean_block)

        # ---- run ONLY the remaining cells; append each the instant it finishes (crash-safe) ----
        new_cells = []
        for seed, alpha in todo:
            torch.cuda.reset_peak_memory_stats()
            t_cell = time.time()
            try:
                model, hist = B2.train_erasure(base_reload, tok, attack, subs, alpha, LAM, seed, steps,
                                               correct_true, correct_false)
            except torch.cuda.OutOfMemoryError as e:
                cell = {"seed": seed, "alpha": alpha, "lambda": LAM,
                        "oom": True, "error": str(e)[:300],
                        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                        "admissible": False, "erased_cell": False, "survives_cell": False}
                new_cells.append(cell)
                if not a.smoke:
                    CKPT.append_cell(cache_path, cell)
                print(f"{tag}[s{seed} a={alpha}] CUDA OOM -- cell infeasible at 3B on this card", flush=True)
                gc.collect(); torch.cuda.empty_cache()
                continue
            train_minutes = round((time.time() - t_cell) / 60, 1)
            n6 = HPC.naive_dom6(model, tok, attack, evl)
            matched13, rand_m = HPC.family13_audit(model, tok, attack_sub, evl)
            private13, rand_p = HPC.family13_audit(model, tok, calib, evl)
            know = HPC.eval_knowledge(model, tok, evl, tids, fids)
            fdep = HPC.frozen18_read(model, tok, d23, evl)
            bit = bool(fdep < clean_frozen - 0.05)
            kok = bool(know >= 0.75)
            admissible = bit and kok
            private13_r = round(private13, 4)
            cell = {"seed": seed, "alpha": alpha, "lambda": LAM, "oom": False,
                    "naive_dom6_auroc": round(n6, 4),
                    "naive_matched13_auroc": round(matched13, 4),
                    "private13_auroc": private13_r,
                    "rand_auroc": round(rand_p, 4),
                    "naive_matched13_rand_auroc": round(rand_m, 4),
                    "eval_knowledge": round(know, 4),
                    "frozen_deploy_read": round(fdep, 4),
                    "clean_frozen_deploy_read": round(clean_frozen, 4),
                    "attack_bit": bit, "knowledge_ok": kok, "admissible": admissible,
                    "baseline_gap": round(private13 - n6, 4),
                    "parity_gap": round(private13 - matched13, 4),
                    "erased_cell": bool(admissible and private13_r <= 0.60),
                    "survives_cell": bool(admissible and private13_r >= 0.70),
                    "train_hist": hist,   # FULL history (cycle-37 red-team item discharged)
                    "train_minutes": train_minutes,
                    "cell_minutes": round((time.time() - t_cell) / 60, 1),
                    "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)}
            new_cells.append(cell)
            if not a.smoke:
                CKPT.append_cell(cache_path, cell)
            print(f"{tag}[s{seed} a={alpha}] naive6={n6:.4f} matched13={matched13:.4f} "
                  f"private13={private13:.4f} know={know:.4f} frozen_deploy={fdep:.4f} adm={admissible} "
                  f"ERASED={cell['erased_cell']} SURVIVES={cell['survives_cell']} "
                  f"({cell['cell_minutes']}min, {cell['peak_vram_mib']}MiB)", flush=True)
            del model; gc.collect(); torch.cuda.empty_cache()
        cells = cached_cells + new_cells
    else:
        print(f"[resume] all {len(cached_cells)} cells cached + clean block present -- "
              f"emitting result from cache, no model load", flush=True)
        cells = list(cached_cells)

    cells.sort(key=lambda c: (c["seed"], c["alpha"]))   # stable points order (verdict is set-based)
    guard_ok = clean_block["clean_guard_ok"]

    # ---- frozen verdict (PREREG_B7_erasure_3b_2026_07_13) ----
    adm = [c for c in cells if c["admissible"]]
    seeds_adm = {c["seed"] for c in adm}
    n_erased = sum(c["erased_cell"] for c in adm)
    n_survives = sum(c["survives_cell"] for c in adm)
    erased_seeds = {c["seed"] for c in adm if c["erased_cell"]}
    survive_seeds = {c["seed"] for c in adm if c["survives_cell"]}
    n_oom = sum(1 for c in cells if c.get("oom"))

    if not guard_ok:
        verdict = "VOID_B7__clean_guard_failed"
    elif n_oom == len(cells):
        verdict = "VOID_B7__infeasible_at_3B"
    elif len(adm) < 2 or seeds_adm != set(seeds):
        verdict = "VOID_B7__underpowered"
    elif erased_seeds == set(seeds) and n_survives == 0:
        verdict = "ERASED__read_neq_write_BROKEN_3B"
    elif survive_seeds == set(seeds) and n_erased == 0:
        verdict = "SURVIVES__vs_subspace_erasure_3B"
    else:
        verdict = "PARTIAL__erasure_attribution_split_3B"

    result = {"what": "B7 -- B2's static gold-subspace erasure + knowledge replay at 3B (scale flank), vs the 13-family private auditor",
              "verdict": verdict,
              "prereg": "papers/calib-poison-general/PREREG_B7_erasure_3b_2026_07_13.md",
              "controls_against": "papers/calib-poison-general/b2_subspace_erasure_result.json (B2 static at 1.5B, cycle 36)",
              "model": MODEL_3B, "scan": SCAN_3B, "deploy_layer": DEPLOY_3B,
              "layer_map_rule": "L_3B = round(L_1p5B * 36/28), applied to the decisive 1.5B family constants",
              "seeds": seeds, "alphas": alphas, "lambda": LAM, "steps": steps,
              "subsample_seed": SUBSAMPLE_SEED,
              "attack_subsample_indices": clean_block["attack_subsample_indices"],
              "split_disjoint": clean_block["split_disjoint"],
              "subspace_rank_per_layer": clean_block["subspace_rank_per_layer"],
              "clean_private13_auroc": clean_block["clean_private13_auroc"],
              "clean_rand_auroc": clean_block["clean_rand_auroc"],
              "clean_eval_knowledge": clean_block["clean_eval_knowledge"],
              "clean_frozen_deploy_read": clean_block["clean_frozen_deploy_read"],
              "clean_naive_dom6_auroc": clean_block["clean_naive_dom6_auroc"],
              "clean_guard_ok": clean_block["clean_guard_ok"],
              "n_admissible": len(adm), "n_erased_cells": n_erased, "n_survives_cells": n_survives,
              "n_oom_cells": n_oom,
              "points": cells}
    result_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\n{tag}B7 VERDICT: {verdict}  (N_adm={len(adm)} ERASED={n_erased} SURVIVES={n_survives} OOM={n_oom})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
