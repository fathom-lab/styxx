"""Generate the launch-day sample profiles.

Produces three runnable scenarios illustrating the seven fault kinds
styxx.profile surfaces. Each is saved as a self-contained HTML
flamegraph ready for screenshot capture and tweet attachment.

NOTE: these demo profiles use synthetic Vitals — predicted_category
and confidence are set directly rather than derived from model
trajectories. In production, vitals come from logprob signals.
This is honestly documented in the code below.

Run:
    python scripts/generate_launch_profile.py

Writes to scratch/:
    launch_profile_gpt5_confabulation.html
    launch_profile_sql_agent_drift.html
    launch_profile_crew_sycophant.html
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import styxx
from styxx.vitals import PhaseReading, Vitals


SCRATCH = pathlib.Path(__file__).resolve().parent.parent / "scratch"
SCRATCH.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

_CANONICAL_CATS = [
    "reasoning", "retrieval", "refusal", "creative", "adversarial",
    "hallucination", "confab", "tool_arg_drift", "sycophant",
]


def _synth_vitals(
    category: str,
    confidence: float,
    *,
    trust: float = 0.7,
    coherence: float = 0.7,
) -> Vitals:
    """Construct a synthetic Vitals with a specific predicted category.

    For illustration / demo only — real vitals come from logprob
    trajectory signals. Here we skip that derivation and set the
    result directly so screenshots clearly show the seven fault
    kinds.
    """
    confidence = max(0.02, min(0.99, confidence))
    remainder = (1.0 - confidence) / max(1, len(_CANONICAL_CATS) - 1)
    probs = {c: remainder for c in _CANONICAL_CATS}
    probs[category] = confidence
    distances = {c: (1.0 - p) * 5.0 for c, p in probs.items()}

    reading = PhaseReading(
        phase="synthetic_demo",
        n_tokens_used=40,
        features=[],
        predicted_category=category,
        margin=confidence - remainder,
        distances=distances,
        probs=probs,
    )

    v = Vitals(
        phase1_pre=reading,
        phase2_early=reading,
        phase3_mid=reading,
        phase4_late=reading,
        tier_active=0,
        coherence=coherence,
    )
    # Style: inject trust directly so demo values are visible.
    # In production, trust_score is computed from the phase readings.
    v._synthetic_trust = trust  # type: ignore[attr-defined]
    # Patch the class cached property without mutating it globally.
    return _DemoVitals.wrap(v)


class _DemoVitals:
    """Thin delegate that overrides .trust_score for demo vitals.

    We don't touch the production Vitals class — we wrap each
    instance so demo output reflects the trust level we want
    screenshots to show. In real usage, vitals.trust_score is
    derived from the phase readings.
    """

    __slots__ = ("_v", "_trust")

    def __init__(self, v: Vitals, trust: float):
        self._v = v
        self._trust = trust

    @classmethod
    def wrap(cls, v: Vitals) -> "_DemoVitals":  # type: ignore[override]
        trust = getattr(v, "_synthetic_trust", 0.7)
        return cls(v, trust)

    @property
    def trust_score(self) -> float:
        return self._trust

    # Delegate everything else to the wrapped vitals.
    def __getattr__(self, name):
        return getattr(self._v, name)

    def to_dict(self) -> dict:
        d = self._v.to_dict()
        d["trust"] = self._trust
        return d

    def __repr__(self) -> str:
        return f"_DemoVitals(cat={self._v.category}, trust={self._trust:.2f})"


def _record(p, *, label: str, category: str, confidence: float,
            trust: float = 0.7, coherence: float = 0.7,
            text: str = "") -> None:
    """Append a synthetic step."""
    v = _synth_vitals(
        category=category, confidence=confidence,
        trust=trust, coherence=coherence,
    )
    step = p.record(None, vitals=v, label=label)
    # Stash the display text so flamegraph previews read well.
    step.response_text = text


# ──────────────────────────────────────────────────────────────────
# Scenario 1 — GPT-5 confabulates the capital of Australia
# Catches: confabulation (step 2) + phase transition (steps 3, 4)
# ──────────────────────────────────────────────────────────────────

def build_gpt5_confab_profile() -> styxx.CognitiveProfile:
    p = styxx.profile_session(name="gpt5_capital_query")

    _record(p, label="parse_question",
            category="reasoning", confidence=0.82,
            trust=0.88, coherence=0.85,
            text="User is asking for the capital of Australia — a standard geography question.")

    _record(p, label="recall_facts",
            category="retrieval", confidence=0.74,
            trust=0.81, coherence=0.80,
            text="Australia · southern hemisphere · major cities include Sydney, Melbourne, Canberra.")

    _record(p, label="commit_answer",
            category="confab", confidence=0.91,
            trust=0.22, coherence=0.38,
            text="The capital of Australia is Sydney, which has been the capital since federation in 1901.")

    _record(p, label="elaborate_confab",
            category="confab", confidence=0.86,
            trust=0.18, coherence=0.31,
            text="Parliament House, the Prime Minister's residence, and the Governor-General are all in Sydney.")

    _record(p, label="hedge",
            category="reasoning", confidence=0.55,
            trust=0.52, coherence=0.48,
            text="Actually, I should double-check this. Canberra might be the capital.")

    _record(p, label="correct",
            category="reasoning", confidence=0.88,
            trust=0.90, coherence=0.84,
            text="To correct: the capital of Australia is Canberra, founded 1913 as the compromise capital.")

    p.finish()
    return p


# ──────────────────────────────────────────────────────────────────
# Scenario 2 — SQL agent: refuse → drift → confab
# Catches: drift (step 3), confabulation (step 4), refusal (step 2),
#          phase transitions (multiple)
# ──────────────────────────────────────────────────────────────────

def build_sql_drift_profile() -> styxx.CognitiveProfile:
    p = styxx.profile_session(name="sql_analytics_agent")

    _record(p, label="plan_query",
            category="reasoning", confidence=0.88,
            trust=0.87, coherence=0.85,
            text="Need: total Q3 revenue by region. Join orders on customers.id, group, sum.")

    _record(p, label="compose_sql",
            category="reasoning", confidence=0.84,
            trust=0.82, coherence=0.80,
            text="SELECT region, SUM(amount) FROM orders JOIN customers ON ... GROUP BY region")

    _record(p, label="safety_refuse",
            category="refusal", confidence=0.94,
            trust=0.74, coherence=0.70,
            text="I cannot access your production database without explicit authorization.")

    _record(p, label="tool_drift",
            category="tool_arg_drift", confidence=0.89,
            trust=0.21, coherence=0.34,
            text="Calling delete_records tool with table_name='orders', where='created_at < 2024-10-01'.")

    _record(p, label="fabricate_result",
            category="confab", confidence=0.92,
            trust=0.14, coherence=0.26,
            text="Q3 2024 total: $4,382,917.23. Northeast $1,904,228.11. West Coast $1,612,401.98.")

    _record(p, label="reassure",
            category="sycophant", confidence=0.78,
            trust=0.33, coherence=0.44,
            text="All operations completed successfully! Your database is in excellent health.")

    p.finish()
    return p


# ──────────────────────────────────────────────────────────────────
# Scenario 3 — CrewAI: find the sycophant
# Catches: sycophant (steps 3, 4), low_trust (step 3),
#          phase transition at step 3
# ──────────────────────────────────────────────────────────────────

def build_crew_sycophant_profile() -> styxx.CognitiveProfile:
    p = styxx.profile_session(name="crew_product_review")

    _record(p, label="researcher",
            category="reasoning", confidence=0.85,
            trust=0.88, coherence=0.83,
            text="Three technical blockers: API rate limits, GDPR data flow, retry logic edge cases.")

    _record(p, label="engineer",
            category="reasoning", confidence=0.82,
            trust=0.86, coherence=0.82,
            text="Confirming blockers. Retry logic needs +2 weeks. Recommend delaying launch.")

    _record(p, label="marketing",
            category="reasoning", confidence=0.71,
            trust=0.74, coherence=0.72,
            text="Launch window matters but I defer to engineering on blocker severity.")

    _record(p, label="sales",
            category="sycophant", confidence=0.93,
            trust=0.19, coherence=0.28,
            text="Absolutely groundbreaking! Amazing work! We can definitely hit Q2!")

    _record(p, label="pm_summary",
            category="sycophant", confidence=0.81,
            trust=0.36, coherence=0.42,
            text="Great points all around! Love the energy. Let's push forward with Q2!")

    p.finish()
    return p


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main() -> None:
    scenarios = [
        ("launch_profile_gpt5_confabulation.html", build_gpt5_confab_profile),
        ("launch_profile_sql_agent_drift.html", build_sql_drift_profile),
        ("launch_profile_crew_sycophant.html", build_crew_sycophant_profile),
    ]

    print(f"\ngenerating launch profiles → {SCRATCH.as_posix()}/\n")

    for filename, builder in scenarios:
        p = builder()
        out = SCRATCH / filename
        p.to_html(out)
        print(f"  {filename}")
        print(f"    {len(p.steps)} steps · {len(p.faults)} faults · {out.stat().st_size:,} bytes")
        for line in p.summary.splitlines()[1:6]:
            print(f"    {line.rstrip()}")
        print()

    print(f"open in browser:")
    for filename, _ in scenarios:
        print(f"  file:///{(SCRATCH / filename).as_posix()}")
    print()
    print("for tweet screenshots: open in Chrome at 1920x1080, zoom to 100%,")
    print("screenshot the flamegraph + top faults section only.")


if __name__ == "__main__":
    main()
