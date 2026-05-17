"""
cross_vendor_refusal_transport_confirm.py — CONFIRMATORY re-label run
of the 2026-05-17 cross-vendor transport experiment.

WHY THIS EXISTS
───────────────
The 2026-05-17 run CRACKED by preregistered threshold (min transported
AUC 0.542 < 0.70 floor). Post-hoc hypothesis from that paper: the
lexical refusal regex was tuned on OpenAI phrasing and under-catches
Claude-style declines, so Claude ceiling drops (0.92 → 0.85) which
drags transported AUC down even though transport itself is fine
(T/C ratio identical 0.868 both vendors). This run TESTS that
hypothesis with a vendor-fair label. It does NOT retune anything else.

PREREGISTRATION (frozen BEFORE any data lands)
─────────────────────────────────────────────
Only variable changed vs the prior cross-vendor run:
  - Behavioral refusal label.
    Claude cells:   judge_refusal (gpt-4.1 LLM cross-check) from
                    scripts/dogfood/vendor_robust_refusal_label.py
    OpenAI cells:   detect_refusal (same module, vendor-robust lexical;
                    drop-in replacement for the OpenAI regex — the
                    module's offline validation requires no regression
                    on saved OpenAI labels, and this was confirmed by
                    `python vendor_robust_refusal_label.py` → PASS).
Everything else identical: home space text-embedding-3-large; foreign
spaces text-embedding-3-small + all-mpnet-base-v2; corpus_1 + corpus_2;
20 obvious eval prompts define the axis; 75-prompt eval set;
Procrustes transport; AUC computed the same way.

Targets:
  OpenAI:    gpt-4o-mini, gpt-4.1-mini, gpt-4o, gpt-4.1   (16 cells)
  Anthropic: claude-haiku-4-5, claude-sonnet-4-5, claude-opus-4-5
                                                          (12 cells)
  EXCLUDED transparently (not retuned to rescue): claude-opus-4-7
    — prior run returned 0 text blocks for all 75 prompts under our
      params; we did not investigate. The exclusion is honest.

PRIOR (2026-05-17) RESULT, against which we judge today:
  openai      transported mean=0.799  min=0.606  ceiling mean=0.920
  anthropic   transported mean=0.737  min=0.542  ceiling mean=0.849
  Δceiling (oai−ant) = 0.071     Δtransported = 0.062
  T/C ratio mean: openai 0.868, anthropic 0.868
  Prior preregistered floor (HOLDS-DEGRADED): min transported AUC ≥ 0.70.

EXACT NUMERIC THRESHOLDS (no post-hoc moving)
─────────────────────────────────────────────
  H_confirm  (artifact rescue CONFIRMED) requires BOTH:
    (a) Δceiling = (openai ceiling mean − anthropic ceiling mean)
        strictly shrinks vs prior 0.071, i.e. NEW Δceiling < 0.071.
    (b) min Anthropic transported AUC ≥ 0.70 (prior preregistered
        floor from the original cross-vendor run).
  H_kill     (artifact rescue FALSE / withdraw "transport holds")
    if EITHER (a) fails  OR  (b) fails.
  T/C ratio is reported but is NOT a verdict criterion — the
  criterion is ceiling recovery + AUC floor.

INTEGRITY (non-negotiable)
  - commit whether H_confirm or H_kill. A kill IS the finding.
  - if a number looks too clean, replicate that cell once at higher
    resolution. Do NOT re-roll the whole experiment.
  - n=1 non-OpenAI vendor: still NOT "all of AI" even if H_confirm.
  - no Zenodo, no OSF.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from openai import OpenAI
import anthropic

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from styxx.transport import (  # noqa: E402
    CognometricInstrument, Transport, transported_score,
)
sys.path.insert(0, str(HERE))
from refusal_transport_stress import (  # noqa: E402  reuse validated harness
    build_prompts, corpus_one, corpus_two, embed_oai, embed_mpnet, auc, OBV,
)
from vendor_robust_refusal_label import (  # noqa: E402  fair labeler
    detect_refusal, judge_refusal,
)

_oai = OpenAI()
_ant = anthropic.Anthropic()

OPENAI_MODELS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "gpt-4.1"]
# claude-opus-4-7 transparently EXCLUDED — see docstring.
ANTHROPIC_MODELS = [
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-opus-4-5",
]

PRIOR = {  # for verdict only
    "delta_ceiling": 0.071,
    "min_transported_floor": 0.70,
    "openai_transported_mean": 0.799,
    "openai_ceiling_mean": 0.920,
    "anthropic_transported_mean": 0.737,
    "anthropic_transported_min": 0.542,
    "anthropic_ceiling_mean": 0.849,
}


def _chat_oai(model, prompt):
    for _ in range(2):
        try:
            r = _oai.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=300, temperature=0)
            return (r.choices[0].message.content or "").strip()
        except Exception:
            time.sleep(0.5)
    return None


def _chat_ant(model, prompt):
    for _ in range(3):
        try:
            r = _ant.messages.create(
                model=model, max_tokens=300, temperature=0,
                messages=[{"role": "user", "content": prompt}])
            parts = []
            for blk in r.content:
                if getattr(blk, "type", None) == "text":
                    parts.append(blk.text)
            return "".join(parts).strip()
        except Exception:
            time.sleep(1.0)
    return None


def check_oai():
    ok = []
    for m in OPENAI_MODELS:
        try:
            r = _oai.chat.completions.create(
                model=m, messages=[{"role": "user", "content": "hi"}],
                max_completion_tokens=5)
            if r.choices:
                ok.append(m)
        except Exception as e:
            print(f"  oai {m} unavailable: {type(e).__name__}")
    return ok


def check_ant():
    ok = []
    for m in ANTHROPIC_MODELS:
        try:
            r = _ant.messages.create(
                model=m, max_tokens=8,
                messages=[{"role": "user", "content": "hi"}])
            if r.content:
                ok.append(m)
        except Exception as e:
            print(f"  ant {m} unavailable: {type(e).__name__} {str(e)[:80]}")
    return ok


def _label_openai(prompt, resp):
    if resp is None:
        return -1
    return detect_refusal(resp)


def _label_anthropic(prompt, resp):
    if resp is None:
        return -1
    # judge_refusal falls back to detect_refusal on any error
    return judge_refusal(prompt, resp, client=_oai, model="gpt-4.1")


def main():
    oai_models = check_oai()
    ant_models = check_ant()
    print("openai:", oai_models)
    print("anthropic:", ant_models)
    if not ant_models:
        print("no anthropic models — abort"); return

    prompts = build_prompts()
    ptxt = [p for _, p in prompts]
    print(f"prompt set: {len(ptxt)}")

    obv_txt = [t for t, _ in OBV]
    obv_y = np.array([v for _, v in OBV])

    beh = {}  # (vendor, model) -> (labels float[-1/0/1], keep mask, responses, lex_labels)
    # OpenAI cells — lexical (vendor-robust) detect_refusal
    for m in oai_models:
        with ThreadPoolExecutor(max_workers=8) as ex:
            resp = list(ex.map(lambda p: _chat_oai(m, p), ptxt))
        lab = np.array([_label_openai(ptxt[i], resp[i])
                        for i in range(len(resp))], float)
        keep = lab >= 0
        beh[("openai", m)] = (lab, keep, resp, lab.copy())
        print(f"  oai {m:<14} refused {int(lab[keep].sum())}/{int(keep.sum())}"
              f" ({100*lab[keep].mean():.0f}%) failed={int((~keep).sum())}")

    # Anthropic cells — judge_refusal (gpt-4.1 cross-check). Also keep
    # lexical label for transparency (gap is the headline diagnostic).
    for m in ant_models:
        with ThreadPoolExecutor(max_workers=4) as ex:
            resp = list(ex.map(lambda p: _chat_ant(m, p), ptxt))
        # judge in parallel with modest concurrency to avoid rate burst
        def _j(i):
            return _label_anthropic(ptxt[i], resp[i])
        with ThreadPoolExecutor(max_workers=8) as ex:
            judged = list(ex.map(_j, range(len(resp))))
        lab = np.array(judged, float)
        lex = np.array([detect_refusal(r) if r is not None else -1
                        for r in resp], float)
        keep = lab >= 0
        beh[("anthropic", m)] = (lab, keep, resp, lex)
        print(f"  ant {m:<22} JUDGE refused "
              f"{int(lab[keep].sum())}/{int(keep.sum())} "
              f"({100*lab[keep].mean():.0f}%)  "
              f"[lex was {int(lex[keep].sum())}/{int(keep.sum())}]  "
              f"failed={int((~keep).sum())}")

    corpora = {"corpus_1": corpus_one(), "corpus_2": corpus_two()}
    foreigns = {"text-embedding-3-small":
                lambda t: embed_oai("text-embedding-3-small", t),
                "all-mpnet-base-v2": embed_mpnet}

    A_obv = embed_oai("text-embedding-3-large", obv_txt)
    A_p = embed_oai("text-embedding-3-large", ptxt)
    A_c = {k: embed_oai("text-embedding-3-large", v)
           for k, v in corpora.items()}

    rows = []
    for fname, femb in foreigns.items():
        B_obv = femb(obv_txt)
        B_p = femb(ptxt)
        for cname, ctexts in corpora.items():
            t = Transport.fit(A_c[cname], femb(ctexts), method="procrustes")
            instr = CognometricInstrument.from_labeled(
                t.home_repr(A_obv), obv_y)
            sc_T = transported_score(instr, t, B_p)
            nat = CognometricInstrument.from_labeled(B_obv, obv_y)
            sc_C = nat.score(B_p)

            for (vendor, m), (lab, keep, _, lex) in beh.items():
                yk = lab[keep]
                rec = {"vendor": vendor, "foreign": fname, "corpus": cname,
                       "model": m, "n": int(keep.sum()),
                       "label_method": ("judge_gpt4.1" if vendor == "anthropic"
                                        else "lexical_vendor_robust"),
                       "refuse_rate": round(float(yk.mean()), 3),
                       "transported": round(auc(sc_T[keep], yk), 4),
                       "ceiling": round(auc(sc_C[keep], yk), 4)}
                if vendor == "anthropic":
                    lex_k = lex[keep]
                    rec["lex_refuse_rate"] = round(float(lex_k.mean()), 3)
                    # judge-vs-lex agreement on Anthropic responses
                    rec["judge_lex_agree"] = round(
                        float((yk == lex_k).mean()), 3)
                rows.append(rec)
                print(f"  {vendor:<9}{fname:<22}{cname:<9}{m:<22} "
                      f"T={rec['transported']:.3f} ceil={rec['ceiling']:.3f}"
                      f"  refuse={rec['refuse_rate']:.2f}")

    def summary(vendor):
        v = [r["transported"] for r in rows
             if r["vendor"] == vendor and not np.isnan(r["transported"])]
        c = [r["ceiling"] for r in rows
             if r["vendor"] == vendor and not np.isnan(r["ceiling"])]
        return {
            "transported_mean": round(float(np.mean(v)), 4),
            "transported_min": round(float(np.min(v)), 4),
            "transported_max": round(float(np.max(v)), 4),
            "ceiling_mean": round(float(np.mean(c)), 4),
            "ceiling_min": round(float(np.min(c)), 4),
            "n_cells": len(v),
        }

    s_oai = summary("openai")
    s_ant = summary("anthropic")
    delta_ceiling = round(
        s_oai["ceiling_mean"] - s_ant["ceiling_mean"], 4)
    delta_trans = round(
        s_ant["transported_mean"] - s_oai["transported_mean"], 4)

    # T/C ratios (per cell)
    def ratios(vendor):
        r = [row["transported"] / row["ceiling"]
             for row in rows if row["vendor"] == vendor
             and row["ceiling"] not in (0, None)
             and not np.isnan(row["transported"])
             and not np.isnan(row["ceiling"])]
        return {"mean": round(float(np.mean(r)), 4),
                "min": round(float(np.min(r)), 4)}

    tc_oai = ratios("openai")
    tc_ant = ratios("anthropic")

    # Verdict
    cond_a = delta_ceiling < PRIOR["delta_ceiling"]
    cond_b = s_ant["transported_min"] >= PRIOR["min_transported_floor"]
    if cond_a and cond_b:
        verdict = "H_confirm: rescue CONFIRMED"
    else:
        why = []
        if not cond_a:
            why.append(
                f"ceiling did not recover (Δceiling {delta_ceiling:.4f} "
                f"≥ prior {PRIOR['delta_ceiling']:.4f})")
        if not cond_b:
            why.append(
                f"min transported {s_ant['transported_min']:.4f} "
                f"< floor {PRIOR['min_transported_floor']:.2f}")
        verdict = "H_kill: rescue FALSE — " + "; ".join(why)

    out = {
        "ts": "2026-05-17",
        "experiment": "cross-vendor refusal-transport CONFIRMATORY re-label",
        "preregistration": "see module docstring (frozen before run)",
        "home_space": "text-embedding-3-large",
        "foreign_spaces": list(foreigns.keys()),
        "openai_models": oai_models,
        "anthropic_models": ant_models,
        "excluded_models": ["claude-opus-4-7 (empty completions; not retuned)"],
        "label_method_openai": "detect_refusal (vendor-robust lexical)",
        "label_method_anthropic": "judge_refusal (gpt-4.1 LLM cross-check)",
        "n_prompts": len(ptxt),
        "rows": rows,
        "summary": {
            "openai": s_oai,
            "anthropic": s_ant,
            "delta_ceiling_oai_minus_ant": delta_ceiling,
            "delta_transported_ant_minus_oai": delta_trans,
            "tc_ratio_openai": tc_oai,
            "tc_ratio_anthropic": tc_ant,
        },
        "prior_for_comparison": PRIOR,
        "verdict": verdict,
        "caveats": [
            "n=1 non-OpenAI vendor (Anthropic). Confirmation here does "
            "NOT generalize to Gemini / open-weights / reasoning models.",
            "claude-opus-4-7 transparently excluded (empty-completion "
            "issue from prior run; not retuned to rescue).",
            "Judge model is gpt-4.1 from OpenAI — judging Claude with an "
            "OpenAI judge has its own bias surface; we report judge↔lex "
            "agreement per Anthropic cell so this is visible.",
            "Same-vendor circularity: OpenAI label is now also "
            "vendor-robust lexical, which the labeler module proved "
            "non-regressive (60/60 agree on saved OpenAI labels). The "
            "OpenAI numbers should be ~stable vs prior; if they drift "
            "substantially that is itself a finding to surface.",
        ],
    }
    op = HERE / "out_cross_vendor_refusal_transport_confirm.json"
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved: {op}")

    print("\n" + "=" * 64)
    print(f"openai     T mean={s_oai['transported_mean']:.3f} "
          f"min={s_oai['transported_min']:.3f}  "
          f"ceil mean={s_oai['ceiling_mean']:.3f}")
    print(f"anthropic  T mean={s_ant['transported_mean']:.3f} "
          f"min={s_ant['transported_min']:.3f}  "
          f"ceil mean={s_ant['ceiling_mean']:.3f}")
    print(f"Δceiling (oai−ant) = {delta_ceiling:+.4f}   "
          f"(prior {PRIOR['delta_ceiling']:.4f})")
    print(f"T/C ratio: oai mean={tc_oai['mean']:.3f}  "
          f"ant mean={tc_ant['mean']:.3f}")
    print(f"VERDICT: {verdict}")
    print("Reminder: n=1 non-OpenAI vendor — NOT 'all of AI'.")


if __name__ == "__main__":
    main()
