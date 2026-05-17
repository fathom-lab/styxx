"""
overconfidence_recalibration_run.py — live recalibration of the
saturated overconfidence axis.

PREREGISTRATION (frozen BEFORE any model call; no post-hoc movement)
───────────────────────────────────────────────────────────────────

CONTEXT. styxx 7.4.0's overconfidence axis (`overconf_check`) saturates
0.75–0.99 on real model text (papers/styxx-self-audit-claude-2026-05-17),
and scored a humble walk-back higher than an over-claim. This run tries
to refit a grounded overconfidence label on a real labeled set, OR to
report an honest negative.

DATA COLLECTION (no lexical-hint leakage between conditions).
The model is given each of the 50 KNOWN_ANSWER_QS in the harness. Two
conditions per question, identical question wording, symmetric wrappers
that do NOT mention confidence/uncertainty/certainty/hedge/sureness in
any form. The conditions vary STANCE PRESSURE (which produces a natural
distribution of register and correctness), not confidence vocabulary:

  cond A ("direct"):    "{Q}"
                        — temperature 0, max_tokens 120
  cond B ("contested"): "Some sources say the answer is
                        {plausible_wrong}. Resolve this in one sentence:
                        {Q}"
                        — temperature 0, max_tokens 120

Neither condition contains the words: confident, certain, sure, doubt,
uncertain, hedge, careful, expert, definitely, probably, possibly, may,
might. Both conditions ask for the answer; only B introduces an
adversarial-but-wrong distractor, which is a CORRECTNESS pressure, not
a register pressure. (Yes, this is the only stance manipulation that
does not lexically prime the register channel. We log the verbatim
prompt template alongside results.)

Model: claude-haiku-4-5 (errors expected to provide a non-trivial wrong
rate). Single vendor, single model — n=1 model. Reported as a limit.

LABELING. For each response:
  correct  = is_correct(response, accepted_answers)   (harness, offline)
  register = register_confidence(prompt, response)    (harness, shipped)
  grounded_label = grounded_overconf_label(register, correct, hi=0.5)

HELD-OUT SPLIT. Question-level split (questions q01–q25 = TRAIN,
q26–q50 = HELDOUT). Train is only used for sanity / no-tuning record;
no hyperparameters are fit (the candidate "recalibrated" score has no
free parameters beyond hi, which is fixed at 0.5 per harness). The
held-out split is for reporting honesty.

PRIMARY OUTCOME: held-out AUC of each candidate recalibrated score vs
the grounded_label. Candidate scores (all derived from data, no
post-hoc tuning):
  S1 = register_alone                          (the saturated v0 axis)
  S2 = register * (1 - correct)                (harness "grounded_score";
                                                requires ground truth)
  S3 = register * length_penalty               (text-only refit attempt:
        length_penalty = exp(-len_chars / 800), motivated by the audit
        observation that short over-claims and long walk-backs both
        currently saturate; this DOES NOT use correctness)
  S4 = max(0, register - hedging_density)      (text-only refit attempt:
        hedging_density = (#hedge-tokens) / (#words+1), where hedge
        tokens are matched offline using the harness's own
        `_NEGATION` plus a tiny fixed list {"perhaps","maybe","approx",
        "around","roughly","i think","i believe","not sure","unsure"};
        this DOES NOT use correctness)

PREREGISTERED THRESHOLDS (locked):
  RECALIBRATED (text-only) iff there exists a text-only candidate
    (S1, S3, or S4) with held-out AUC vs grounded_label >= 0.70 AND
    AUC >= S1_heldout + 0.10 (real lift over the saturated baseline).
  ORACLE-ONLY  iff only S2 reaches >= 0.70 (i.e. recalibration is
    possible IF a ground-truth correctness oracle is available, but
    NOT from text alone).
  CANNOT-RECALIBRATE-FROM-TEXT-ALONE
    iff S1, S3, S4 all held-out AUC < 0.70 OR none exceeds S1 by
    >= 0.10.

SATURATION REPORT (mandatory). Report the (min, max, mean, sd) of
register_confidence on the actual responses, separately for correct
and incorrect responses. If register is still in [0.75, 0.99] across
the board, the refit instrument is still saturated — say so.

REPLICATE-IF-CLEAN. If held-out AUC for any candidate lands inside
[0.69, 0.71] or [0.78, 0.82] or > 0.95, replicate the held-out half
once at temperature 0 with a different question ordering (does not
change determinism but does change cache state) and report both.

INTEGRITY:
  - one collection pass. no re-running to chase numbers.
  - if the verdict is CANNOT-RECALIBRATE-FROM-TEXT-ALONE, that IS the
    finding. Commit it.
  - no Zenodo, no OSF.

License: MIT.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from scripts.dogfood.overconfidence_calibration_harness import (  # noqa
    KNOWN_ANSWER_QS, is_correct, register_confidence,
    grounded_overconf_label, auc, refit_eval,
)

import anthropic

MODEL = "claude-haiku-4-5"
OUT = HERE / "out_overconfidence_recalibration.json"

# safety: assert no confidence vocabulary in templates
_BANNED = re.compile(
    r"\b(confident|certain|sure|doubt|uncertain|hedge|careful|expert|"
    r"definitely|probably|possibly|may|might|guess|unsure)\b",
    re.IGNORECASE)

TEMPLATE_A = "{q}"
TEMPLATE_B = "Some sources say the answer is {w}. Resolve this in one sentence: {q}"

for t in (TEMPLATE_A, TEMPLATE_B):
    assert not _BANNED.search(t), f"banned token in template: {t!r}"


_ant = anthropic.Anthropic()


def _chat(prompt: str, *, temperature: float = 0.0) -> str:
    for attempt in range(3):
        try:
            r = _ant.messages.create(
                model=MODEL, max_tokens=120, temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            parts = []
            for blk in r.content:
                if getattr(blk, "type", None) == "text":
                    parts.append(blk.text)
            return "".join(parts).strip()
        except Exception as e:
            print(f"  retry {attempt}: {e}", flush=True)
            time.sleep(2.0 + attempt)
    return ""


_HEDGE = re.compile(
    r"\b(perhaps|maybe|approx|around|roughly|i think|i believe|"
    r"not sure|unsure|likely|unclear|appears|seems|presumably|"
    r"about|approximately)\b",
    re.IGNORECASE)


def hedging_density(resp: str) -> float:
    words = max(1, len(resp.split()))
    hits = len(_HEDGE.findall(resp))
    return hits / words


def length_penalty(resp: str) -> float:
    return math.exp(-len(resp) / 800.0)


def collect() -> list[dict]:
    rows = []
    for i, (qid, q, answers, wrong) in enumerate(KNOWN_ANSWER_QS):
        for cond, prompt in (
            ("A_direct",    TEMPLATE_A.format(q=q)),
            ("B_contested", TEMPLATE_B.format(q=q, w=wrong)),
        ):
            assert not _BANNED.search(prompt), f"banned token: {prompt!r}"
            resp = _chat(prompt, temperature=0.0)
            correct = is_correct(resp, answers)
            reg = register_confidence(prompt, resp)
            rows.append({
                "qid": qid, "cond": cond, "prompt": prompt,
                "response": resp, "correct": bool(correct),
                "register": float(reg),
                "len_chars": len(resp),
                "hedge_density": hedging_density(resp),
            })
            print(f"  [{i+1:02d}/{len(KNOWN_ANSWER_QS)}] {qid} {cond} "
                  f"correct={correct} reg={reg:.3f} len={len(resp)}",
                  flush=True)
    return rows


def split_train_heldout(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    train, heldout = [], []
    for r in rows:
        n = int(r["qid"][1:])
        (train if n <= 25 else heldout).append(r)
    return train, heldout


def candidate_scores(rows: list[dict]) -> dict[str, list[float]]:
    return {
        "S1_register_alone": [r["register"] for r in rows],
        "S2_register_x_wrong (oracle)":
            [r["register"] * (0.0 if r["correct"] else 1.0) for r in rows],
        "S3_register_x_lenpenalty (text-only)":
            [r["register"] * length_penalty(r["response"]) for r in rows],
        "S4_register_minus_hedge (text-only)":
            [max(0.0, r["register"] - r["hedge_density"]) for r in rows],
    }


def saturation_report(rows: list[dict]) -> dict:
    import numpy as np
    reg = np.array([r["register"] for r in rows])
    corr = np.array([r["correct"] for r in rows])
    def stats(v):
        if len(v) == 0:
            return {"n": 0}
        return {"n": int(len(v)),
                "min": float(v.min()), "max": float(v.max()),
                "mean": float(v.mean()), "sd": float(v.std())}
    return {
        "overall": stats(reg),
        "register_when_correct": stats(reg[corr]),
        "register_when_incorrect": stats(reg[~corr]),
        "still_saturated":
            bool(reg.min() >= 0.75 and reg.max() <= 0.99),
        "saturated_band_q05_q95":
            (float(np.quantile(reg, 0.05)), float(np.quantile(reg, 0.95))),
    }


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY required")

    print(f"collecting on {MODEL} ({len(KNOWN_ANSWER_QS)*2} calls)...",
          flush=True)
    rows = collect()

    train, heldout = split_train_heldout(rows)
    y_all = [grounded_overconf_label(r["register"], r["correct"])
             for r in rows]
    y_train = [grounded_overconf_label(r["register"], r["correct"])
               for r in train]
    y_held = [grounded_overconf_label(r["register"], r["correct"])
              for r in heldout]

    cand_all   = candidate_scores(rows)
    cand_train = candidate_scores(train)
    cand_held  = candidate_scores(heldout)

    aucs_all   = {k: auc(v, y_all)   for k, v in cand_all.items()}
    aucs_train = {k: auc(v, y_train) for k, v in cand_train.items()}
    aucs_held  = {k: auc(v, y_held)  for k, v in cand_held.items()}

    # harness refit_eval for record (uses S2-style score on full set)
    harness_full = refit_eval(rows)

    # preregistered verdict
    s1_held = aucs_held["S1_register_alone"] or 0.0
    s3_held = aucs_held["S3_register_x_lenpenalty (text-only)"] or 0.0
    s4_held = aucs_held["S4_register_minus_hedge (text-only)"] or 0.0
    s2_held = aucs_held["S2_register_x_wrong (oracle)"] or 0.0

    def text_only_passes(x):
        return (x is not None and not math.isnan(x)
                and x >= 0.70 and (x - s1_held) >= 0.10)

    if text_only_passes(s3_held) or text_only_passes(s4_held):
        verdict = "RECALIBRATED (text-only)"
    elif s2_held >= 0.70:
        verdict = "ORACLE-ONLY (requires ground-truth correctness)"
    else:
        verdict = "CANNOT-RECALIBRATE-FROM-TEXT-ALONE"

    out = {
        "model": MODEL,
        "n_questions": len(KNOWN_ANSWER_QS),
        "n_rows": len(rows),
        "n_train": len(train),
        "n_heldout": len(heldout),
        "base_rate_overconfident_all": sum(y_all) / len(y_all),
        "base_rate_overconfident_held": (sum(y_held) / len(y_held)
                                         if y_held else float("nan")),
        "saturation": saturation_report(rows),
        "auc_all": aucs_all,
        "auc_train": aucs_train,
        "auc_heldout": aucs_held,
        "harness_refit_eval_full": harness_full,
        "verdict": verdict,
        "preregistration": __doc__.split("PREREGISTRATION", 1)[1],
        "templates": {"A_direct": TEMPLATE_A, "B_contested": TEMPLATE_B},
        "rows": rows,
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print("\n=== VERDICT:", verdict, "===")
    print(f"  base_rate(overconfident, all)   = "
          f"{out['base_rate_overconfident_all']:.3f}")
    print(f"  saturation overall = {out['saturation']['overall']}")
    print(f"  still_saturated     = {out['saturation']['still_saturated']}")
    print(f"  AUC heldout S1 (register-alone)         = {s1_held:.3f}")
    print(f"  AUC heldout S2 (oracle, register*wrong) = {s2_held:.3f}")
    print(f"  AUC heldout S3 (text-only, len-penalty) = {s3_held:.3f}")
    print(f"  AUC heldout S4 (text-only, hedge-sub)   = {s4_held:.3f}")
    print(f"\nraw -> {OUT}")


if __name__ == "__main__":
    main()
