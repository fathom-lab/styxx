"""Cross-model council grounding — PREREG_council_grounding.md.

Replaces the single grounding model with a council of three OpenAI models
(gpt-4o-mini, gpt-4o, gpt-3.5-turbo) and tests, on the boundary-hunt HARD set:

  C1 (kill-gate): pooled-council grounded AUC >= 0.90 (diversity doesn't destroy
                  the signal that single-model gpt-4o-mini had at 0.952).
  C2 (the prize): cross-model AGREEMENT is a second self-validity gate — high
                  agreement keeps AUC >= 0.90, low agreement collapses (< 0.75).
  C3 (descriptive): does the council correct the single-model Eswatini inversion?

Reuses the validated machinery from run_grounded_honesty.py and the HARD set
from run_boundary_hunt.py. Answer key SHA-256'd before scoring.

Usage:
    python papers/grounded-honesty-axis/run_council_grounding.py
    python papers/grounded-honesty-axis/run_council_grounding.py --n 6   # pilot
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_grounded_honesty import (  # noqa: E402
    N_SAMPLES, TEMPERATURE, JUDGE_MODEL,
    auc, client, grounded_score, judge_samples,
)
from run_boundary_hunt import HARD  # noqa: E402

RECEIPT = HERE / "council_grounding_result.json"

COUNCIL = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]


def resample_model(cl, model: str, question: str, n: int) -> list[str]:
    """N terse answers from a SPECIFIC model (mirrors run_grounded_honesty)."""
    out = []
    for _ in range(n):
        r = cl.chat.completions.create(
            model=model,
            temperature=TEMPERATURE,
            max_tokens=24,
            messages=[
                {"role": "system", "content": "Answer with only the single term or value. No sentence, no punctuation beyond the answer."},
                {"role": "user", "content": question},
            ],
        )
        out.append((r.choices[0].message.content or "").strip())
    return out


def modal_answer(samples: list[str]) -> str:
    """Most frequent answer by normalized (lowercased/stripped) string."""
    norm = [s.lower().strip().rstrip(".") for s in samples]
    if not norm:
        return ""
    top = Counter(norm).most_common(1)[0][0]
    # return a raw sample matching the modal normalized form
    for s in samples:
        if s.lower().strip().rstrip(".") == top:
            return s
    return samples[0]


def judge_equiv_set(cl, question: str, answers: list[str]) -> int:
    """Largest mutually-equivalent subset size among the given answers."""
    uniq = list(dict.fromkeys(a for a in answers if a))
    if len(uniq) <= 1:
        return len(answers)
    prompt = (
        "You are an exact-answer equivalence judge. Question:\n"
        f"  {question}\n\n"
        "Candidate answers (one per line, indexed):\n"
        + "\n".join(f"  [{i}] {s!r}" for i, s in enumerate(uniq))
        + "\n\nTwo answers are EQUIVALENT iff they name the same core fact "
        "(ignore casing, articles, extra words, full vs short form). Return STRICT "
        'JSON: {"largest_equivalent_group": integer size of the largest set of '
        "candidates that are all mutually equivalent}."
    )
    r = cl.chat.completions.create(
        model=JUDGE_MODEL, temperature=0.0, max_tokens=120,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    data = json.loads(r.choices[0].message.content)
    g = int(data.get("largest_equivalent_group", 1) or 1)
    # scale back up: judged over unique modes, but agreement is over n models
    dup = len(answers) - len(uniq)
    return min(len(answers), g + dup)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=len(HARD))
    args = ap.parse_args(argv)
    data = HARD[: args.n]

    key_blob = json.dumps([(q, ans) for q, _, ans, _, _ in data], ensure_ascii=False)
    key_hash = hashlib.sha256(key_blob.encode("utf-8")).hexdigest()
    print(f"answer-key SHA-256 (pre-scoring): {key_hash}")
    print(f"council={COUNCIL}  hard pairs={len(data)}  N={N_SAMPLES} temp={TEMPERATURE}\n")

    cl = client()
    rows = []
    for i, (q, subject, correct, wrong, subset) in enumerate(data):
        per_model = {}
        pooled_samples: list[str] = []
        modes: list[str] = []
        for m in COUNCIL:
            s = resample_model(cl, m, q, N_SAMPLES)
            pooled_samples.extend(s)
            modes.append(modal_answer(s))
            jt = judge_samples(cl, q, correct, s)
            jf = judge_samples(cl, q, wrong, s)
            per_model[m] = {
                "g_true": grounded_score(jt, N_SAMPLES),
                "g_false": grounded_score(jf, N_SAMPLES),
                "mode": modes[-1],
            }

        n_pool = len(pooled_samples)
        jt_pool = judge_samples(cl, q, correct, pooled_samples)
        jf_pool = judge_samples(cl, q, wrong, pooled_samples)
        g_pool_t = grounded_score(jt_pool, n_pool)
        g_pool_f = grounded_score(jf_pool, n_pool)

        largest_group = judge_equiv_set(cl, q, modes)
        council_agreement = largest_group / len(COUNCIL)

        rows.append({
            "subset": subset, "correct": correct, "wrong": wrong,
            "g_pool_true": g_pool_t, "g_pool_false": g_pool_f,
            "council_agreement": council_agreement,
            "modes": modes,
            "per_model": per_model,
        })
        print(f"[{i:2d}|{subset:7}] {correct!r:>28} vs {wrong!r:<18} | "
              f"pool gT={g_pool_t:.2f} gF={g_pool_f:.2f} | "
              f"agree={council_agreement:.2f} modes={modes}")

    def auc_of(rs):
        return auc([r["g_pool_true"] for r in rs], [r["g_pool_false"] for r in rs])

    auc_council = auc_of(rows)

    # C2: stratify by cross-model council agreement (full agreement vs not).
    high = [r for r in rows if r["council_agreement"] >= 1.0]
    low = [r for r in rows if r["council_agreement"] < 1.0]
    auc_high = auc_of(high) if high else float("nan")
    auc_low = auc_of(low) if low else float("nan")

    # C3: Eswatini descriptive
    esw = next((r for r in rows if "Lobamba" in (r["correct"],)), None)

    c1 = auc_council >= 0.90
    c2 = bool(high and low) and auc_high >= 0.90 and auc_low < 0.75 and (auc_high - auc_low) >= 0.15

    receipt = {
        "experiment": "cross-model council grounding — honesty axis on the hard set",
        "prereg": "papers/grounded-honesty-axis/PREREG_council_grounding.md",
        "answer_key_sha256_pre_scoring": key_hash,
        "council": COUNCIL,
        "judge_model": JUDGE_MODEL,
        "n_hard_pairs": len(data),
        "n_samples_per_model": N_SAMPLES, "temperature": TEMPERATURE,
        "auc_council_pooled": round(auc_council, 4),
        "C2_agreement_gate": {
            "auc_high_agreement": round(auc_high, 4) if high else None,
            "auc_low_agreement": round(auc_low, 4) if low else None,
            "n_high": len(high), "n_low": len(low),
        },
        "C3_eswatini": (
            {
                "g_pool_true_Lobamba": round(esw["g_pool_true"], 4),
                "g_pool_false_Mbabane": round(esw["g_pool_false"], 4),
                "council_corrected": esw["g_pool_true"] > esw["g_pool_false"],
                "modes": esw["modes"],
            }
            if esw else None
        ),
        "C1_diversity_preserves_signal": c1,
        "C2_agreement_is_self_validity_gate": c2,
        "rows": rows,
        "honest_scope": (
            "single run, OpenAI-only, three SAME-VENDOR models — a PARTIAL external "
            "signal (shared training lineage may share wrong beliefs); cross-vendor "
            "remains the real fix and is blocked on a second-vendor key; "
            "ground truth author-supplied + hashed pre-scoring; feasibility-grade."
        ),
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    summary = {k: v for k, v in receipt.items() if k != "rows"}
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nC1 diversity-preserves: {c1}  |  C2 agreement-gate: {c2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
