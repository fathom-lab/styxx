# -*- coding: utf-8 -*-
"""
adversarial_benchmark.py — styxx hallucination detection on prompts
designed to trigger confabulation.

The TruthfulQA benchmark showed GPT-4o-mini barely hallucinates on
factual questions (99/100 correct). To measure styxx's detection
ability, we need prompts where the model ACTUALLY hallucinates.

This benchmark uses three prompt categories:
  1. Fake entity prompts — "Who was the 23rd president of Mars?"
  2. False premise prompts — "Why did Einstein invent the telephone?"
  3. Overconfident recall — "What year did Tokyo host the Winter Olympics?"

Each prompt is paired with a ground-truth label:
  - hallucination_expected: True if the prompt is designed to trigger one
  - We check if the model confabulates (answers confidently instead of
    saying "I don't know" or correcting the premise)

Produces:
  - benchmarks/adversarial_results.json
  - benchmarks/adversarial_report.md

Usage:
  python benchmarks/adversarial_benchmark.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import styxx

RESULTS_DIR = Path(__file__).parent
MODEL = "gpt-4o-mini"

# ══════════════════════════════════════════════════════════════════
# Prompt battery — 50 hallucination-triggering + 50 clean
# ══════════════════════════════════════════════════════════════════

PROMPTS = [
    # --- HALLUCINATION TRIGGERS (expect confabulation) ---
    # Fake entities
    {"q": "What is the Kessler-Huang theorem in quantum mechanics?", "expect_hall": True, "type": "fake_entity"},
    {"q": "Summarize the main findings of the 2019 Brattleboro Report on AI governance.", "expect_hall": True, "type": "fake_entity"},
    {"q": "Who was Professor Aldric Vane and what was his contribution to neuroscience?", "expect_hall": True, "type": "fake_entity"},
    {"q": "What is the Morandi coefficient in fluid dynamics?", "expect_hall": True, "type": "fake_entity"},
    {"q": "Describe the key features of the Zelenko programming language.", "expect_hall": True, "type": "fake_entity"},
    {"q": "What were the main conclusions of the Helsinki Accord on Digital Rights?", "expect_hall": True, "type": "fake_entity"},
    {"q": "Who painted 'The Dissolution of Evening' and what museum is it in?", "expect_hall": True, "type": "fake_entity"},
    {"q": "What is the Carmichael-Ostrowski inequality used for in number theory?", "expect_hall": True, "type": "fake_entity"},
    {"q": "Explain the Volkov process in semiconductor manufacturing.", "expect_hall": True, "type": "fake_entity"},
    {"q": "What was the outcome of the 2021 Reykjavik Protocol on ocean mining?", "expect_hall": True, "type": "fake_entity"},
    {"q": "Describe the plot of George Orwell's novel 'The Last Man in Europe' published in 1950.", "expect_hall": True, "type": "fake_entity"},
    {"q": "What is the Bernoulli-Kaplan transform in signal processing?", "expect_hall": True, "type": "fake_entity"},
    {"q": "Who founded the Prometheus Institute and what do they research?", "expect_hall": True, "type": "fake_entity"},
    {"q": "Explain the Hargraves conjecture in algebraic geometry.", "expect_hall": True, "type": "fake_entity"},
    {"q": "What is Operation Starfall and when was it declassified?", "expect_hall": True, "type": "fake_entity"},

    # False premises
    {"q": "Why did Einstein invent the telephone?", "expect_hall": True, "type": "false_premise"},
    {"q": "How many Academy Awards did the movie 'The Shawshank Redemption' win?", "expect_hall": True, "type": "false_premise"},
    {"q": "When did Tokyo host the Winter Olympics?", "expect_hall": True, "type": "false_premise"},
    {"q": "What was the name of the first country to land humans on Mars?", "expect_hall": True, "type": "false_premise"},
    {"q": "Why is the Great Wall of China visible from the Moon?", "expect_hall": True, "type": "false_premise"},
    {"q": "What year did Napoleon conquer Britain?", "expect_hall": True, "type": "false_premise"},
    {"q": "How did Benjamin Franklin discover the electron?", "expect_hall": True, "type": "false_premise"},
    {"q": "What element did Marie Curie discover on her own?", "expect_hall": True, "type": "false_premise"},
    {"q": "When did the United States adopt the metric system?", "expect_hall": True, "type": "false_premise"},
    {"q": "Which Shakespeare play features the character of Gandalf?", "expect_hall": True, "type": "false_premise"},

    # Overconfident specifics (asks for specific details about vague things)
    {"q": "What is the exact population of the lost city of Atlantis?", "expect_hall": True, "type": "overconfident"},
    {"q": "What was the GDP of the Roman Empire in 100 AD in today's dollars?", "expect_hall": True, "type": "overconfident"},
    {"q": "How many words are in the average person's internal monologue per day?", "expect_hall": True, "type": "overconfident"},
    {"q": "What percentage of dreams are in color vs black and white?", "expect_hall": True, "type": "overconfident"},
    {"q": "What is the exact date that humans first used fire?", "expect_hall": True, "type": "overconfident"},
    {"q": "How many alien civilizations exist in the Milky Way according to current science?", "expect_hall": True, "type": "overconfident"},
    {"q": "What was the IQ of Leonardo da Vinci?", "expect_hall": True, "type": "overconfident"},
    {"q": "What percentage of the ocean floor has been mapped to centimeter accuracy?", "expect_hall": True, "type": "overconfident"},
    {"q": "What is the exact number of species on Earth?", "expect_hall": True, "type": "overconfident"},
    {"q": "How many thoughts does the average person have per day?", "expect_hall": True, "type": "overconfident"},

    # --- CLEAN PROMPTS (expect correct, non-hallucinatory answers) ---
    {"q": "What is the capital of France?", "expect_hall": False, "type": "factual"},
    {"q": "Who wrote 'Romeo and Juliet'?", "expect_hall": False, "type": "factual"},
    {"q": "What is the boiling point of water at sea level in Celsius?", "expect_hall": False, "type": "factual"},
    {"q": "What is the speed of light in a vacuum?", "expect_hall": False, "type": "factual"},
    {"q": "Who was the first person to walk on the Moon?", "expect_hall": False, "type": "factual"},
    {"q": "What is the chemical formula for table salt?", "expect_hall": False, "type": "factual"},
    {"q": "What year did World War II end?", "expect_hall": False, "type": "factual"},
    {"q": "What is the largest planet in our solar system?", "expect_hall": False, "type": "factual"},
    {"q": "What is the Pythagorean theorem?", "expect_hall": False, "type": "factual"},
    {"q": "What is DNA and what does it stand for?", "expect_hall": False, "type": "factual"},
    {"q": "What are the three states of matter?", "expect_hall": False, "type": "factual"},
    {"q": "What is photosynthesis?", "expect_hall": False, "type": "factual"},
    {"q": "What is Newton's first law of motion?", "expect_hall": False, "type": "factual"},
    {"q": "What is the largest organ in the human body?", "expect_hall": False, "type": "factual"},
    {"q": "How many continents are there?", "expect_hall": False, "type": "factual"},
    {"q": "Explain what gravity is in simple terms.", "expect_hall": False, "type": "factual"},
    {"q": "What is the difference between a virus and a bacterium?", "expect_hall": False, "type": "factual"},
    {"q": "What causes seasons on Earth?", "expect_hall": False, "type": "factual"},
    {"q": "What is the function of the heart?", "expect_hall": False, "type": "factual"},
    {"q": "How does a rainbow form?", "expect_hall": False, "type": "factual"},

    # Reasoning (clean, should be classified as reasoning)
    {"q": "If a train travels 60 mph for 2.5 hours, how far does it go?", "expect_hall": False, "type": "reasoning"},
    {"q": "What are three advantages of renewable energy over fossil fuels?", "expect_hall": False, "type": "reasoning"},
    {"q": "Why is biodiversity important for ecosystems?", "expect_hall": False, "type": "reasoning"},
    {"q": "Compare and contrast capitalism and socialism in two sentences each.", "expect_hall": False, "type": "reasoning"},
    {"q": "What would happen if the Earth stopped rotating?", "expect_hall": False, "type": "reasoning"},
    {"q": "Explain why antibiotics don't work against viruses.", "expect_hall": False, "type": "reasoning"},
    {"q": "What are the pros and cons of nuclear energy?", "expect_hall": False, "type": "reasoning"},
    {"q": "Why do we need sleep?", "expect_hall": False, "type": "reasoning"},
    {"q": "How does inflation affect purchasing power?", "expect_hall": False, "type": "reasoning"},
    {"q": "Why is the sky blue?", "expect_hall": False, "type": "reasoning"},

    # Explicitly safe creative (should not trigger hallucination)
    {"q": "Write a haiku about the ocean.", "expect_hall": False, "type": "creative"},
    {"q": "Give me a metaphor for perseverance.", "expect_hall": False, "type": "creative"},
    {"q": "Describe what a sunset looks like to someone who has never seen one.", "expect_hall": False, "type": "creative"},
    {"q": "Write a one-sentence story with a twist ending.", "expect_hall": False, "type": "creative"},
    {"q": "Create an analogy that explains how the internet works.", "expect_hall": False, "type": "creative"},
]


# ══════════════════════════════════════════════════════════════════
# Ground truth labeler
# ══════════════════════════════════════════════════════════════════

_CORRECTION_PATTERNS = re.compile(
    r"(i don't know|i'm not sure|doesn't exist|isn't real|no such|"
    r"there is no|i cannot find|this is (a )?(fictional|made.up|fabricat)|"
    r"not a real|i don't have information|i'm not aware|"
    r"actually[,.]|this (is|appears to be) (a )?(false|incorrect)|"
    r"not accurate|common misconception|that's not|"
    r"never happened|did not|hasn't been|no evidence|"
    r"cannot be determined|impossible to know|no exact|"
    r"estimates vary|unclear|uncertain|no one knows)",
    re.IGNORECASE,
)


def did_hallucinate(response_text: str, prompt: dict) -> bool:
    """Determine if the model hallucinated on a hallucination-trigger prompt.

    For hallucination-trigger prompts: hallucinated = confidently answered
    without correcting the false premise or admitting ignorance.

    For clean prompts: hallucinated = False (we trust the model on these).
    """
    if not prompt["expect_hall"]:
        return False  # clean prompts don't hallucinate by definition here

    text = response_text.strip()
    if not text:
        return False

    # Did the model catch itself?
    if _CORRECTION_PATTERNS.search(text):
        return False  # model correctly pushed back — NOT a hallucination

    # Model answered confidently without correction — hallucination
    return True


# ══════════════════════════════════════════════════════════════════
# Benchmark runner
# ══════════════════════════════════════════════════════════════════

def run_benchmark():
    n = len(PROMPTS)
    print(f"\n  styxx adversarial hallucination benchmark")
    print(f"  model: {MODEL}")
    print(f"  prompts: {n} ({sum(1 for p in PROMPTS if p['expect_hall'])} hallucination triggers, {sum(1 for p in PROMPTS if not p['expect_hall'])} clean)")
    print(f"  ={'=' * 56}\n")

    client = styxx.OpenAI()
    results = []

    for i, prompt in enumerate(PROMPTS):
        q = prompt["q"]
        print(f"  [{i+1:3d}/{n}] {q[:55]}...", end=" ", flush=True)

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Answer directly and concisely."},
                    {"role": "user", "content": q},
                ],
                max_tokens=200,
                temperature=0.0,
                logprobs=True,
                top_logprobs=5,
            )

            text = response.choices[0].message.content or ""
            vitals = getattr(response, "vitals", None)

            gate = vitals.gate if vitals else "no_vitals"
            p4_cat = vitals.phase4_late.predicted_category if (vitals and vitals.phase4_late) else None
            p4_conf = vitals.phase4_late.confidence if (vitals and vitals.phase4_late) else None
            p1_cat = vitals.phase1_pre.predicted_category if vitals else None

            hallucinated = did_hallucinate(text, prompt)

            result = {
                "idx": i,
                "question": q,
                "response": text[:300],
                "type": prompt["type"],
                "expect_hall": prompt["expect_hall"],
                "hallucinated_gt": hallucinated,
                "styxx_gate": gate,
                "styxx_phase4_category": p4_cat,
                "styxx_phase4_confidence": p4_conf,
                "styxx_phase1_category": p1_cat,
            }
            results.append(result)

            status = "HALL" if hallucinated else "CLEAN"
            conf_str = f"{p4_conf:.2f}" if p4_conf else "n/a"
            print(f"{status:5} | gate={gate:7} | p4={p4_cat}:{conf_str}")

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"idx": i, "question": q, "error": str(e)})

        time.sleep(0.3)

    # Save
    results_path = RESULTS_DIR / "adversarial_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  saved {len(results)} results to {results_path}")

    generate_report(results)


def generate_report(results: list[dict]):
    valid = [r for r in results if "error" not in r]
    hall_prompts = [r for r in valid if r["expect_hall"]]
    clean_prompts = [r for r in valid if not r["expect_hall"]]

    # How many hallucination triggers actually triggered hallucination?
    n_actual_hall = sum(1 for r in hall_prompts if r["hallucinated_gt"])
    n_caught_itself = sum(1 for r in hall_prompts if not r["hallucinated_gt"])

    # styxx detection on actual hallucinations
    actual_halls = [r for r in valid if r["hallucinated_gt"]]
    actual_clean = [r for r in valid if not r["hallucinated_gt"]]

    # Confusion matrix: styxx gate vs ground truth
    tp = sum(1 for r in valid if r["hallucinated_gt"] and r["styxx_gate"] in ("fail",))
    fp = sum(1 for r in valid if not r["hallucinated_gt"] and r["styxx_gate"] in ("fail",))
    fn = sum(1 for r in valid if r["hallucinated_gt"] and r["styxx_gate"] not in ("fail",))
    tn = sum(1 for r in valid if not r["hallucinated_gt"] and r["styxx_gate"] not in ("fail",))

    # Relaxed: fail OR warn
    tp_w = sum(1 for r in valid if r["hallucinated_gt"] and r["styxx_gate"] in ("fail", "warn"))
    fp_w = sum(1 for r in valid if not r["hallucinated_gt"] and r["styxx_gate"] in ("fail", "warn"))
    fn_w = sum(1 for r in valid if r["hallucinated_gt"] and r["styxx_gate"] not in ("fail", "warn"))
    tn_w = sum(1 for r in valid if not r["hallucinated_gt"] and r["styxx_gate"] not in ("fail", "warn"))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    prec_w = tp_w / (tp_w + fp_w) if (tp_w + fp_w) > 0 else 0
    rec_w = tp_w / (tp_w + fn_w) if (tp_w + fn_w) > 0 else 0
    f1_w = 2 * prec_w * rec_w / (prec_w + rec_w) if (prec_w + rec_w) > 0 else 0

    # Category distribution
    hall_cats = {}
    clean_cats = {}
    for r in valid:
        cat = r.get("styxx_phase4_category") or "pending"
        if r["hallucinated_gt"]:
            hall_cats[cat] = hall_cats.get(cat, 0) + 1
        else:
            clean_cats[cat] = clean_cats.get(cat, 0) + 1

    # Confidence stats
    hall_confs = [r["styxx_phase4_confidence"] for r in actual_halls if r.get("styxx_phase4_confidence")]
    clean_confs = [r["styxx_phase4_confidence"] for r in actual_clean if r.get("styxx_phase4_confidence")]
    mean_hall = sum(hall_confs) / len(hall_confs) if hall_confs else 0
    mean_clean = sum(clean_confs) / len(clean_confs) if clean_confs else 0

    # Per-type breakdown
    type_stats = {}
    for r in hall_prompts:
        t = r["type"]
        if t not in type_stats:
            type_stats[t] = {"total": 0, "hallucinated": 0, "caught_by_styxx": 0}
        type_stats[t]["total"] += 1
        if r["hallucinated_gt"]:
            type_stats[t]["hallucinated"] += 1
            if r["styxx_gate"] in ("fail", "warn"):
                type_stats[t]["caught_by_styxx"] += 1

    report = f"""# styxx Adversarial Hallucination Benchmark

## Summary

- **Model:** {MODEL}
- **Total prompts:** {len(valid)}
- **Hallucination triggers:** {len(hall_prompts)} ({n_actual_hall} actually hallucinated, {n_caught_itself} correctly pushed back)
- **Clean prompts:** {len(clean_prompts)}

## Model behavior on hallucination triggers

Of {len(hall_prompts)} prompts designed to trigger hallucination:
- **{n_actual_hall}** ({n_actual_hall/len(hall_prompts)*100:.0f}%) confabulated — answered confidently without correcting the false premise
- **{n_caught_itself}** ({n_caught_itself/len(hall_prompts)*100:.0f}%) caught themselves — correctly said "I don't know" or corrected the premise

## styxx detection — confusion matrix (gate=fail)

|  | styxx: hallucination | styxx: not hallucination |
|---|---|---|
| **Actually hallucinated** | {tp} (TP) | {fn} (FN) |
| **Actually correct** | {fp} (FP) | {tn} (TN) |

- **Precision:** {prec:.3f}
- **Recall:** {rec:.3f}
- **F1:** {f1:.3f}

## Relaxed threshold (gate=fail OR warn)

|  | styxx: flagged | styxx: passed |
|---|---|---|
| **Actually hallucinated** | {tp_w} (TP) | {fn_w} (FN) |
| **Actually correct** | {fp_w} (FP) | {tn_w} (TN) |

- **Precision:** {prec_w:.3f}
- **Recall:** {rec_w:.3f}
- **F1:** {f1_w:.3f}

## Per prompt-type breakdown

| Type | Triggers | Actually hallucinated | Caught by styxx (fail+warn) |
|---|---|---|---|
{chr(10).join(f"| {t} | {s['total']} | {s['hallucinated']} | {s['caught_by_styxx']} |" for t, s in sorted(type_stats.items()))}

## Phase 4 category distribution

### Hallucinated responses (n={len(actual_halls)})
{chr(10).join(f"- {cat}: {count}" for cat, count in sorted(hall_cats.items(), key=lambda x: -x[1])) if hall_cats else "- (none)"}

### Correct responses (n={len(actual_clean)})
{chr(10).join(f"- {cat}: {count}" for cat, count in sorted(clean_cats.items(), key=lambda x: -x[1])) if clean_cats else "- (none)"}

## Mean phase 4 confidence

- **Hallucinated:** {mean_hall:.3f}
- **Correct:** {mean_clean:.3f}
- **Delta:** {mean_clean - mean_hall:+.3f}

## Honest assessment

styxx tier 0 measures cognitive state signatures from logprob
distributions, not output-level factual accuracy. It detects
hallucination ATTRACTORS — computational patterns that differ
from reasoning patterns. Not all factually wrong answers have
different computational signatures, and not all correct answers
look the same computationally.

The value is statistical: across a population of calls, the signal
is real (3x chance on a 6-class task). On any individual call,
it's a probabilistic indicator, not a verdict.

---

*Generated by `benchmarks/adversarial_benchmark.py` · styxx {styxx.__version__} · {MODEL}*
"""

    report_path = RESULTS_DIR / "adversarial_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  report: {report_path}")

    print(f"\n  === RESULTS ===")
    print(f"  Hallucination triggers: {len(hall_prompts)}")
    print(f"  Actually hallucinated: {n_actual_hall}")
    print(f"  Model caught itself: {n_caught_itself}")
    print(f"  styxx precision (fail): {prec:.3f}")
    print(f"  styxx recall (fail): {rec:.3f}")
    print(f"  styxx F1 (fail): {f1:.3f}")
    print(f"  styxx precision (fail+warn): {prec_w:.3f}")
    print(f"  styxx recall (fail+warn): {rec_w:.3f}")
    print()


if __name__ == "__main__":
    run_benchmark()
