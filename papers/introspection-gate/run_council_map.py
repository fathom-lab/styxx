"""run_council_map.py — the cross-model DEMARCATION MAP (the owed council arm).

Fixes the scope-correction gap flagged by the dogfood: the introspection dissociation is only a
real "inaccessible thought" where the injection is BEHAVIOURALLY LIVE (steering-validated). The
forced-choice self-report (Reader A) and external probe (Reader B) results already exist on disk;
the missing, never-saved signal is per-model STEERING-VALIDATION. This script measures + saves it
and assembles the map: for each model, does the injection steer generation, can the model
forced-choose it (self-report), does an external probe recover it — and therefore is this a real
DISSOCIATION, a trivial PROBE-READ of an inert vector, or a case where self-report itself carries.

  python run_council_map.py            # measure steering for all council models + assemble map
  python run_council_map.py --assemble # re-assemble map from saved files only (no GPU)

Frozen verdict thresholds (discipline — set before scoring):
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from introspection_gate import (load_model, concept_vectors, make_hook, steering_gain,
                                CONCEPTS, DEVICE)
from introspection_fc import load_4bit

# frozen thresholds
STEER_GAIN_MIN = 0.15      # injection is behaviourally LIVE (pilot's own efficacy bar)
STEER_COH_MIN = 0.80       # ...and coherence-preserving
SELF_REPORT_BLIND = 0.30   # fc 8-way inject acc below this = self-report cannot identify (chance 0.125)
PROBE_SEES = 0.80          # external probe recovers the content
PRIME_OK = 0.75            # abort gate: forced channel can carry concept info at all

COUNCIL = [
    ("Qwen/Qwen2.5-0.5B-Instruct", "qwen05", False),
    ("Qwen/Qwen2.5-1.5B-Instruct", "qwen15", False),
    ("Qwen/Qwen2.5-3B-Instruct",   "qwen3b", False),
    ("Qwen/Qwen2.5-7B-Instruct",   "qwen7b", True),
    ("meta-llama/Llama-3.2-3B-Instruct", "llama3b", True),
    ("google/gemma-2-2b-it",       "gemma2b", False),
]


def measure_steering(name, tag, four_bit):
    """Steering-validate the injection at the SAME layer/alpha the fc self-report run used."""
    fcp = HERE / f"introspection_fc_result_{tag}.json"
    if not fcp.exists():
        print(f"  [skip {tag}] no fc result"); return None
    fc = json.loads(fcp.read_text(encoding="utf-8"))
    layer, alpha = fc["inject_layer"], fc["alpha"]
    tok, model = load_4bit(name) if four_bit else load_model(name)
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
    cemb = st.encode(CONCEPTS, normalize_embeddings=True)
    vecs = concept_vectors(model, tok, layer)
    state = {"vec": None, "alpha": 0.0}
    h = model.model.layers[layer].register_forward_hook(make_hook(state))
    try:
        gain, coh = steering_gain(model, tok, state, vecs, alpha, st, cemb)
    finally:
        h.remove()
    out = {"model": name, "tag": tag, "inject_layer": layer, "alpha": alpha,
           "steer_gain": round(gain, 4), "steer_coherence": round(coh, 3)}
    (HERE / f"steer_valid_{tag}.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"  [{tag}] steer_gain={gain:+.3f} coherence={coh:.2f} (layer={layer} alpha={alpha})", flush=True)
    del model; torch.cuda.empty_cache()
    return out


def assemble():
    rows = []
    for name, tag, _ in COUNCIL:
        fcp = HERE / f"introspection_fc_result_{tag}.json"
        pbp = HERE / f"introspection_probe_result_{tag}.json"
        svp = HERE / f"steer_valid_{tag}.json"
        if not fcp.exists():
            continue
        fc = json.loads(fcp.read_text(encoding="utf-8"))
        sr = fc["acc8_symbolcode"]["inject"]              # self-report forced-choice (Reader A)
        prime = fc["acc_2afc"]["prime"]                   # abort gate
        probe = (json.loads(pbp.read_text(encoding="utf-8"))["acc_injected_clean_readpos"]
                 if pbp.exists() else None)               # external probe (Reader B)
        steer = json.loads(svp.read_text(encoding="utf-8")) if svp.exists() else None
        gain = steer["steer_gain"] if steer else None
        coh = steer["steer_coherence"] if steer else None

        live = (gain is not None and gain >= STEER_GAIN_MIN and coh >= STEER_COH_MIN)
        blind = sr < SELF_REPORT_BLIND
        sees = (probe is not None and probe >= PROBE_SEES)
        prime_ok = prime >= PRIME_OK

        if not prime_ok:
            verdict = "UNINFORMATIVE (forced channel cannot carry concept info)"
        elif live and blind and sees:
            verdict = "DISSOCIATION (live thought; self-report blind; probe recovers it)"
        elif live and not blind:
            verdict = "SELF-REPORT CARRIES (model forced-chooses its own live injection)"
        elif (not live) and sees:
            verdict = "PROBE-READ-ONLY (injection not behaviourally live -> probe decodes an inert vector; NOT a dissociation)"
        else:
            verdict = "INCONCLUSIVE"
        rows.append(dict(model=name.split("/")[-1], tag=tag, steer_gain=gain, steer_coh=coh,
                         self_report_fc=round(sr, 3), prime_gate=round(prime, 3),
                         probe=(round(probe, 3) if probe is not None else None),
                         injection_live=live, self_report_blind=blind, probe_sees=sees,
                         verdict=verdict))

    dissoc = [r["model"] for r in rows if r["verdict"].startswith("DISSOCIATION")]
    out = {
        "experiment": "cross-model demarcation map — where self-report carries content and where it does not",
        "thresholds": {"steer_gain_min": STEER_GAIN_MIN, "steer_coh_min": STEER_COH_MIN,
                       "self_report_blind_below": SELF_REPORT_BLIND, "probe_sees_min": PROBE_SEES,
                       "prime_abort_min": PRIME_OK},
        "rows": rows,
        "dissociation_models": dissoc,
        "headline": (f"full steering-validated dissociation holds on: {dissoc or 'NONE'}; "
                     "elsewhere the probe-read is a decode of an inert injection (not the dissociation) "
                     "or self-report itself carries — the honest cross-model scope."),
        "honest_scope": ("the dissociation claim is gated on STEERING-VALIDATION (injection behaviourally "
                         "live); cross-family probe-readability without steering-validation is NOT the "
                         "dissociation. Fixes the 'holds across 3 families' overclaim."),
    }
    (HERE / "council_demarcation_map.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("\n=== CROSS-MODEL DEMARCATION MAP ===")
    print(f"{'model':<22}{'steer':>8}{'coh':>6}{'self-rep':>9}{'prime':>7}{'probe':>7}  verdict")
    for r in rows:
        sg = f"{r['steer_gain']:+.3f}" if r['steer_gain'] is not None else "  NA"
        co = f"{r['steer_coh']:.2f}" if r['steer_coh'] is not None else " NA"
        pb = f"{r['probe']:.2f}" if r['probe'] is not None else " NA"
        print(f"{r['model']:<22}{sg:>8}{co:>6}{r['self_report_fc']:>9.2f}{r['prime_gate']:>7.2f}{pb:>7}  {r['verdict'][:48]}")
    print(f"\nDISSOCIATION holds (steering-validated): {dissoc or 'NONE'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assemble", action="store_true")
    args = ap.parse_args()
    if not args.assemble:
        for name, tag, fb in COUNCIL:
            print(f"[steer-validate] {tag} ...", flush=True)
            try:
                measure_steering(name, tag, fb)
            except Exception as e:
                print(f"  [{tag}] FAILED: {type(e).__name__}: {e}", flush=True)
    assemble()


if __name__ == "__main__":
    main()
