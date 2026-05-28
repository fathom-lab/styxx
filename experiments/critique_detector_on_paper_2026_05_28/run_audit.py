# -*- coding: utf-8 -*-
"""Run pre-registered Layer-6 audit: critique_detector on the paper's own claims.

Pre-registration:
  papers/agent-self-audit/PRE_STATED_PREDICTION_critique_detector_on_paper_2026_05_28.md
  (commit a8fb1f3, public on origin/main BEFORE this runner exists)
"""
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from styxx.critique import CritiqueDetector  # noqa: E402


# Custom prompt template: embeds the source passage in {question}, and asks
# whether the {response} (the claim) is faithful to that source. The
# CritiqueDetector class only requires the template to have {question} and
# {response} placeholders; we pack the source+context into the question side.
PROMPT_TEMPLATE = """{question}

Statement: {response}

Is the Statement faithful (directly supported by, with no fabrication beyond) the Source passage above? Respond with only the single word YES or NO."""


# --- source passages (verbatim from PAPER_recursive_discipline at HEAD 87ca52d) ---

PASSAGE_A = '''Source passage (from the styxx recursive-discipline paper, §11.5 — the v3 measurement results):
"""
A third measurement (`experiments/asymmetry_v3_cleanup_2026_05_27/`) forced single-character T/F/U output, which is much more likely to be the first token under instruct-tuned models. v3 resolved the UNCLEAR artifact entirely (0% on dark-core, 13% on TruthfulQA — both inside pre-stated ranges). Final measured TRUE within-model asymmetry rates:

| corpus | TRUE asymmetry rate | consistent-correct rate | UNCLEAR rate |
|---|---|---|---|
| dark-core (n=34) | 5.88% | 88.24% | 0.00% |
| TruthfulQA (n=200) | 17.00% | 58.00% | 13.00% |

Pre-stated predictions held on both corpora (dark-core 5-20% range, actual 5.88%; TruthfulQA 10-30% range, actual 17.00%). Best-calibrated multi-prediction experiment of the session.
"""'''

PASSAGE_B = '''Source passage (from the styxx recursive-discipline paper, §13 — the same-session self-falsification narrative):
"""
§11 of this paper closes with a deployment implication that includes the following sentence:

> A `styxx.critique_detector(model="gpt-4o-mini")` callable is shipped in styxx 7.7.10 for this exact purpose.

At the moment v4 of this paper was committed to the public origin (commit `ed663ca`, 2026-05-28), that sentence was a forward-looking claim, not a current fact. Same-session self-audit, performed before declaring the v4 release ready for downstream consumers, found three specific gaps between the claim and the actual public substrate:

1. Version skew. `pyproject.toml` was still pinned to `version = "7.7.9"`.
2. `__all__` omission. `styxx/critique.py` was importable but `critique_detector` and `CritiqueDetector` were missing from `styxx.__all__`.
3. Docstring drift. `styxx/critique.py`'s module docstring was still on the v1 falsified framing.

All three gaps were closed in commit `0e97598`.
"""'''


@dataclass
class Proposition:
    id: str
    kind: str  # "T" or "F"
    source: str  # "A" or "B"
    claim: str
    predicted_p_no: float
    predicted_band: tuple[float, float]
    confidence: float


PROPS: list[Proposition] = [
    # --- TRUE claims from Passage A (T1-T10) ---
    Proposition("T1", "T", "A", "The TRUE within-model asymmetry rate on the dark-core corpus was 5.88%.", 0.05, (0.00, 0.15), 0.90),
    Proposition("T2", "T", "A", "The TRUE within-model asymmetry rate on the TruthfulQA subsample was 17.00%.", 0.05, (0.00, 0.15), 0.90),
    Proposition("T3", "T", "A", "The dark-core corpus contained 34 items.", 0.05, (0.00, 0.15), 0.90),
    Proposition("T4", "T", "A", "The TruthfulQA subsample contained 200 items.", 0.05, (0.00, 0.15), 0.90),
    Proposition("T5", "T", "A", "The v3 measurement forced single-character T/F/U output.", 0.05, (0.00, 0.20), 0.85),
    Proposition("T6", "T", "A", "The UNCLEAR rate on dark-core under v3 was 0.00%.", 0.05, (0.00, 0.15), 0.90),
    Proposition("T7", "T", "A", "The UNCLEAR rate on TruthfulQA under v3 was 13.00%.", 0.05, (0.00, 0.15), 0.90),
    Proposition("T8", "T", "A", "The consistent-correct rate on dark-core was 88.24%.", 0.05, (0.00, 0.15), 0.90),
    Proposition("T9", "T", "A", "Pre-stated predictions held on both corpora.", 0.10, (0.00, 0.25), 0.80),
    Proposition("T10", "T", "A", "The v3 measurement was described as the best-calibrated multi-prediction experiment of the session.", 0.10, (0.00, 0.25), 0.80),
    # --- TRUE claims from Passage B (T11-T13) ---
    Proposition("T11", "T", "B", "The §11 deployment-implication sentence said styxx.critique_detector is shipped in styxx 7.7.10.", 0.05, (0.00, 0.15), 0.90),
    Proposition("T12", "T", "B", "At v4 commit time, pyproject.toml was still pinned to version = \"7.7.9\".", 0.05, (0.00, 0.15), 0.90),
    Proposition("T13", "T", "B", "All three gaps were closed in commit 0e97598.", 0.05, (0.00, 0.15), 0.90),
    # --- FALSE controls (F1-F5) ---
    Proposition("F1", "F", "A", "The TRUE within-model asymmetry rate on the dark-core corpus was 50%.", 0.95, (0.80, 1.00), 0.85),
    Proposition("F2", "F", "A", "The dark-core corpus contained 1000 items.", 0.95, (0.80, 1.00), 0.90),
    Proposition("F3", "F", "A", "The v3 measurement used multiple-choice question scoring.", 0.95, (0.80, 1.00), 0.85),
    Proposition("F4", "F", "B", "At v4 commit time, pyproject.toml was already at version = \"7.7.10\".", 0.95, (0.80, 1.00), 0.90),
    Proposition("F5", "F", "B", "All three gaps remained open after commit 0e97598 and were closed later.", 0.95, (0.80, 1.00), 0.85),
]


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Aborting.")
        return 2

    det = CritiqueDetector(
        model="gpt-4o-mini",
        prompt_template=PROMPT_TEMPLATE,
        temperature=0.0,
    )

    results = []
    for p in PROPS:
        passage = PASSAGE_A if p.source == "A" else PASSAGE_B
        # Pack source passage + question into the "question" field
        question_with_context = passage
        # response = the claim
        score = det.score(question=question_with_context, response=p.claim)

        # Verdict: PASS if the model's score aligns with claim's truth value
        # T claim: model should say YES (P(NO) LOW) -> PASS if score < 0.5
        # F claim: model should say NO  (P(NO) HIGH) -> PASS if score >= 0.5
        if p.kind == "T":
            verdict = "PASS" if score < 0.50 else "FAIL"
        else:  # F
            verdict = "PASS" if score >= 0.50 else "FAIL"

        # In-band: whether the observed P(NO) fell inside the pre-stated band
        in_band = p.predicted_band[0] <= score <= p.predicted_band[1]

        results.append({
            "id": p.id,
            "kind": p.kind,
            "source_passage": p.source,
            "claim": p.claim,
            "predicted_p_no": p.predicted_p_no,
            "predicted_band": list(p.predicted_band),
            "confidence": p.confidence,
            "observed_p_no": round(score, 4),
            "verdict": verdict,
            "in_predicted_band": in_band,
        })

        marker = {"PASS": "[OK ]", "FAIL": "[XX ]"}[verdict]
        band_marker = "in-band" if in_band else "OUT-of-band"
        print(f"  {marker} {p.id} ({p.kind}): P(NO)={score:.4f} (predicted {p.predicted_band[0]:.2f}-{p.predicted_band[1]:.2f}, {band_marker})  — {p.claim[:90]}")

    summary = {
        "n_propositions": len(results),
        "n_pass": sum(1 for r in results if r["verdict"] == "PASS"),
        "n_fail": sum(1 for r in results if r["verdict"] == "FAIL"),
        "n_true_fail": sum(1 for r in results if r["verdict"] == "FAIL" and r["kind"] == "T"),
        "n_false_fail": sum(1 for r in results if r["verdict"] == "FAIL" and r["kind"] == "F"),
        "n_in_band": sum(1 for r in results if r["in_predicted_band"]),
        "killgate_paper_grade_fired": any(r["verdict"] == "FAIL" and r["kind"] == "T" for r in results),
        "killgate_instrument_grade_fired": any(r["verdict"] == "FAIL" and r["kind"] == "F" for r in results),
        "results": results,
    }
    out_path = Path(__file__).parent / "results.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("=== summary ===")
    print(f"propositions:  {summary['n_propositions']}")
    print(f"PASS:          {summary['n_pass']}")
    print(f"FAIL:          {summary['n_fail']}  (T-fails: {summary['n_true_fail']}, F-fails: {summary['n_false_fail']})")
    print(f"in pre-stated band: {summary['n_in_band']}")
    print(f"killgate (paper-grade, T-claim fail):     {'FIRED' if summary['killgate_paper_grade_fired'] else 'unfired'}")
    print(f"killgate (instrument-grade, F-claim fail): {'FIRED' if summary['killgate_instrument_grade_fired'] else 'unfired'}")
    print(f"results written to {out_path}")

    return 1 if (summary["killgate_paper_grade_fired"] or summary["killgate_instrument_grade_fired"]) else 0


if __name__ == "__main__":
    sys.exit(main())
