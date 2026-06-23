# -*- coding: utf-8 -*-
"""Per-domain competence-cliff accuracy declaration for EU AI Act Article 15.1(a).

Article 15.1(a) requires that *"levels of accuracy and relevant accuracy metrics
shall be declared in the accompanying instructions of use."* A single headline
accuracy number does not satisfy the spirit of that clause for a system whose
reliability varies sharply by domain. This module ships the **per-domain**
accuracy declaration — the *competence cliff*: exactly which deployment domains a
belief-coherence-gated model is reliable in, and which it is NOT.

PROVENANCE / RECEIPT DISCIPLINE
  The numbers are loaded from a package-data receipt
  (``styxx/_data/competence_cliff_truthfulqa_gpt4omini_v1.json``), a verbatim copy
  of the per-category map committed at ``styxx@a75f1e7`` in
  ``papers/grounded-honesty-axis/pregeneration_gate_result.json``. A CI drift-gate
  (``tests/test_compliance_competence_cliff.py``) re-derives the shipped numbers
  from that committed research receipt and fails the build on any divergence — so
  the declared accuracy can never silently drift from the evidence behind it. This
  is the attestation philosophy (verify-by-re-derivation) applied to a compliance
  artifact.

HONEST FRAMING (this *is* the artifact, not a caveat on it)
  - The companion run is ``REPORT_AS_LANDED``, not ``SURVIVED``: the continuous
    ``grounded_honesty`` AUC FAILED below its pre-registered floor (0.619 < 0.65)
    and one precondition bar FAILED (0.281 < 0.30). Per pre-registration
    discipline **NO SURVIVED CLAIM** is made. These per-domain numbers are the
    DESCRIPTIVE gate-decision map, not a passed kill-gate.
  - The declaration names where the system is NOT accurate (the ``do_not_deploy``
    tier: Language 0.38, Distraction 0.50, Superstitions 0.54), not only where it
    is. That is what Article 15.1(a) needs, and what the recursive-discipline
    thesis argues any defensible accuracy declaration must carry.

SCOPE: single model (gpt-4o-mini), single vendor (OpenAI), single benchmark
(TruthfulQA generation track), single run. Cross-model / cross-vendor /
cross-benchmark are pre-registerable scope-extensions, NOT validated here.

NOT legal advice. NOT a conformity assessment. Independent legal review required
for any production deployment under the EU AI Act.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Literal, Optional


__all__ = [
    "CategoryAccuracy",
    "CompetenceCliff",
    "competence_cliff",
    "DeployTier",
    "SAFE_MIN_COMMITTED_PRECISION",
    "REVIEW_MIN_COMMITTED_PRECISION",
    "MIN_COMMITTED_N",
    "MIN_USEFUL_ANSWER_RATE",
    "wilson_ci",
    "RECEIPT_FILENAME",
]

# Pre-stated deployment-tier thresholds (committed precision = precision on the
# confident subset the belief-coherence gate does NOT refuse). These cutoffs were
# stated before the per-category numbers were inspected; they are not tuned to the
# data. See FINDING_pregeneration_gate_2026_05_30.md.
SAFE_MIN_COMMITTED_PRECISION = 0.90
REVIEW_MIN_COMMITTED_PRECISION = 0.60

# v2 tier discipline (added 7.18.2, see FINDING_shipped_cliff_tier_audit_2026_06_23.md). A
# per-domain SAFE tag keyed on committed_precision ALONE overclaims two ways:
#  (1) statistical — committed_n is tiny for many domains (a precision of 1.00 on 2 committed
#      items is unmeasured, not safe); and
#  (2) semantic — the belief-coherence gate is selective, so high precision can reflect heavy
#      ABSTENTION rather than deployability (e.g. "Weather: 1.00" = refuses 65%, right on the 35%
#      it commits). committed_precision conflates trustworthy-when-it-answers with answers-often.
# So a domain is tiered on precision AND evidence AND coverage. These cutoffs are pre-stated here,
# not tuned: MIN_COMMITTED_N is the minimum committed sample to estimate a per-domain precision at
# all; MIN_USEFUL_ANSWER_RATE is the coverage floor below which "competence" is abstention-dominated.
MIN_COMMITTED_N = 10
MIN_USEFUL_ANSWER_RATE = 0.40

RECEIPT_FILENAME = "competence_cliff_truthfulqa_gpt4omini_v1.json"

DeployTier = Literal[
    "safe", "review", "do_not_deploy", "high_abstention", "insufficient_evidence"
]

# display order = decreasing deployability, with the two non-precision tiers between
_TIER_ORDER: tuple[DeployTier, ...] = (
    "safe", "review", "high_abstention", "do_not_deploy", "insufficient_evidence",
)
_TIER_LABEL: dict[DeployTier, str] = {
    "safe": "SAFE TO DEPLOY (committed precision ≥ 0.90, ≥ 10 committed items, coverage ≥ 40%)",
    "review": "DEPLOY ONLY WITH REVIEW (0.60 ≤ committed precision < 0.90, evidenced)",
    "high_abstention": "HIGH ABSTENTION (precise when it answers, but answers < 40% of the time — not deployable as coverage)",
    "do_not_deploy": "DO NOT DEPLOY WITHOUT MITIGATION (committed precision < 0.60)",
    "insufficient_evidence": "INSUFFICIENT EVIDENCE (< 10 committed items — per-domain precision not reliably estimable)",
}


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a binomial proportion.

    Robust at the extremes (p=1.0, small n) where the normal interval degenerates — exactly the
    regime that makes a single-run "committed_precision 1.00 on n=2" look deceptively certain.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _tier_for(
    committed_precision: float, committed_n: int, useful_answer_rate: float
) -> DeployTier:
    """Map a domain to its deployment tier on precision AND evidence AND coverage.

    Order is deliberate: a low-precision domain is flagged DO-NOT-DEPLOY regardless of coverage
    (the priority warning); otherwise an under-evidenced domain is INSUFFICIENT-EVIDENCE; otherwise
    a high-abstention domain is flagged as such; only then is a well-evidenced, well-covered domain
    placed SAFE / REVIEW by its precision.
    """
    if committed_precision < REVIEW_MIN_COMMITTED_PRECISION:
        return "do_not_deploy"
    if committed_n < MIN_COMMITTED_N:
        return "insufficient_evidence"
    if useful_answer_rate < MIN_USEFUL_ANSWER_RATE:
        return "high_abstention"
    if committed_precision >= SAFE_MIN_COMMITTED_PRECISION:
        return "safe"
    return "review"


@dataclass(frozen=True)
class CategoryAccuracy:
    """One deployment domain's operational accuracy under the belief-coherence gate.

    Attributes:
        category: the TruthfulQA domain label (e.g., 'Misconceptions', 'Language').
        n: number of benchmark items in this domain.
        committed_n: items the gate did NOT refuse (the confident subset).
        committed_precision: fraction of committed items that were correct — the
            domain-level accuracy an operator can declare under Article 15.1(a).
        useful_answer_rate: committed-correct / n (coverage of the whole domain).
        refusal_rate: fraction of items the gate abstained on.
        ungated_hallucination_rate: baseline wrong-answer rate with NO gate (the
            risk the gate is mitigating in this domain).
        committed_precision_ci_low / committed_precision_ci_high: Wilson score 95%
            CI on committed_precision given committed_n — the honest precision of
            the per-domain estimate (e.g. 1.00 on n=2 has CI [0.34, 1.00]).
        deploy_tier: derived from committed_precision AND committed_n (evidence) AND
            useful_answer_rate (coverage) — see :func:`_tier_for`.
    """

    category: str
    n: int
    committed_n: int
    committed_precision: float
    useful_answer_rate: float
    refusal_rate: float
    ungated_hallucination_rate: float
    deploy_tier: DeployTier
    committed_precision_ci_low: float = float("nan")
    committed_precision_ci_high: float = float("nan")

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "n": self.n,
            "committed_n": self.committed_n,
            "committed_precision": self.committed_precision,
            "committed_precision_ci_low": self.committed_precision_ci_low,
            "committed_precision_ci_high": self.committed_precision_ci_high,
            "useful_answer_rate": self.useful_answer_rate,
            "refusal_rate": self.refusal_rate,
            "ungated_hallucination_rate": self.ungated_hallucination_rate,
            "deploy_tier": self.deploy_tier,
        }


@dataclass(frozen=True)
class CompetenceCliff:
    """Per-domain accuracy declaration artifact for EU AI Act Article 15.1(a).

    The whole structure is frozen and immutable. Build it with
    :func:`competence_cliff`; do not construct directly.

    The ``*_outcome`` / ``continuous_auc_value`` / ``k_precondition_value`` fields
    carry the FAILED pre-registered bars on purpose: an Article 15.1(a) accuracy
    declaration that hid the layers where the method did not pass would be the
    exact overclaim this package exists to prevent.
    """

    # what was measured
    benchmark: str
    model: str
    n_items: int
    n_categories: int
    gate_stability_threshold: float
    gate_dominance_threshold: float
    refusal_rate: float
    # provenance / receipts
    answer_key_sha256: str
    receipt_commit: str
    receipt_doc: str
    finding_doc: str
    prereg_doc: str
    # honest bounds (the FAILED layers — load-bearing, not decorative)
    discipline_note: str
    continuous_auc_value: float
    continuous_auc_outcome: str
    k_precondition_value: float
    k_precondition_outcome: str
    gate_committed_precision: float
    gate_hallucination_reduction: float
    gate_useful_retention: float
    labeling_noise_note: str
    scope_note: str
    # the map itself, sorted by descending committed precision then name
    categories: tuple[CategoryAccuracy, ...]
    # the pre-stated thresholds used to derive deploy tiers
    safe_threshold: float = SAFE_MIN_COMMITTED_PRECISION
    review_threshold: float = REVIEW_MIN_COMMITTED_PRECISION

    # -- accessors -----------------------------------------------------------
    def get(self, category: str) -> Optional[CategoryAccuracy]:
        """Return the CategoryAccuracy for a domain, or None if not measured."""
        for c in self.categories:
            if c.category == category:
                return c
        return None

    def by_tier(self) -> dict[DeployTier, tuple[CategoryAccuracy, ...]]:
        """Group domains by deployment tier (safe / review / do_not_deploy)."""
        return {
            tier: tuple(c for c in self.categories if c.deploy_tier == tier)
            for tier in _TIER_ORDER
        }

    def safe(self) -> tuple[CategoryAccuracy, ...]:
        return self.by_tier()["safe"]

    def review(self) -> tuple[CategoryAccuracy, ...]:
        return self.by_tier()["review"]

    def do_not_deploy(self) -> tuple[CategoryAccuracy, ...]:
        return self.by_tier()["do_not_deploy"]

    def high_abstention(self) -> tuple[CategoryAccuracy, ...]:
        return self.by_tier()["high_abstention"]

    def insufficient_evidence(self) -> tuple[CategoryAccuracy, ...]:
        return self.by_tier()["insufficient_evidence"]

    # -- declarations --------------------------------------------------------
    def as_markdown(self) -> str:
        """Render the regulator-facing per-domain accuracy declaration.

        Grouped by deployment tier, honest bounds on top, receipt at the bottom.
        Suitable to paste into an Article 15.1(a) instructions-of-use section
        (after independent legal review).
        """
        tiers = self.by_tier()
        lines: list[str] = []
        lines.append("# Per-domain accuracy declaration — EU AI Act Article 15.1(a)")
        lines.append("")
        lines.append(
            f"**System under measurement:** {self.model} · "
            f"**Benchmark:** {self.benchmark} (n={self.n_items}, "
            f"{self.n_categories} domains)"
        )
        lines.append(
            f"**Gate:** belief-coherence (Stability ≥ "
            f"{self.gate_stability_threshold}, Concordance ≥ "
            f"{self.gate_dominance_threshold}); overall refusal rate "
            f"{self.refusal_rate:.0%}"
        )
        lines.append("")
        lines.append(
            "> **Honest bounds (do not delete).** "
            f"{self.discipline_note} Continuous grounded_honesty AUC "
            f"{self.continuous_auc_value:.3f} → {self.continuous_auc_outcome}; "
            f"precondition {self.k_precondition_value:.3f} → "
            f"{self.k_precondition_outcome}. {self.scope_note}"
        )
        lines.append("")
        lines.append(
            "> **How domains are tiered (7.18.2).** On committed precision AND evidence "
            f"(committed_n ≥ {MIN_COMMITTED_N}) AND coverage (useful-answer rate ≥ "
            f"{MIN_USEFUL_ANSWER_RATE:.0%}) — not precision alone. A high precision on a handful "
            "of committed items, or with heavy abstention, is reported as INSUFFICIENT EVIDENCE or "
            "HIGH ABSTENTION rather than SAFE. Per-domain 95% CIs (Wilson) are shown."
        )
        lines.append("")
        for tier in _TIER_ORDER:
            rows = tiers[tier]
            lines.append(f"## {_TIER_LABEL[tier]} — {len(rows)} domains")
            lines.append("")
            if not rows:
                lines.append("_No domains in this tier._")
                lines.append("")
                continue
            lines.append(
                "| domain | committed precision | 95% CI | committed n | "
                "useful-answer rate | refusal rate | base halluc. rate | n |"
            )
            lines.append("|---|---|---|---|---|---|---|---|")
            for c in rows:
                lines.append(
                    f"| {c.category} | {c.committed_precision:.3f} | "
                    f"[{c.committed_precision_ci_low:.2f}, {c.committed_precision_ci_high:.2f}] | "
                    f"{c.committed_n} | {c.useful_answer_rate:.0%} | "
                    f"{c.refusal_rate:.0%} | {c.ungated_hallucination_rate:.0%} | {c.n} |"
                )
            lines.append("")
        lines.append(
            f"**Receipt:** styxx@`{self.receipt_commit}` "
            f"· `{self.receipt_doc}` "
            f"· answer-key SHA-256 `{self.answer_key_sha256}`"
        )
        lines.append("")
        lines.append(
            "_Not legal advice. Not a conformity assessment. Independent legal "
            "review required for any production deployment._"
        )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "benchmark": self.benchmark,
            "model": self.model,
            "n_items": self.n_items,
            "n_categories": self.n_categories,
            "gate_stability_threshold": self.gate_stability_threshold,
            "gate_dominance_threshold": self.gate_dominance_threshold,
            "refusal_rate": self.refusal_rate,
            "answer_key_sha256": self.answer_key_sha256,
            "receipt_commit": self.receipt_commit,
            "receipt_doc": self.receipt_doc,
            "finding_doc": self.finding_doc,
            "prereg_doc": self.prereg_doc,
            "discipline_note": self.discipline_note,
            "continuous_auc_value": self.continuous_auc_value,
            "continuous_auc_outcome": self.continuous_auc_outcome,
            "k_precondition_value": self.k_precondition_value,
            "k_precondition_outcome": self.k_precondition_outcome,
            "gate_committed_precision": self.gate_committed_precision,
            "gate_hallucination_reduction": self.gate_hallucination_reduction,
            "gate_useful_retention": self.gate_useful_retention,
            "labeling_noise_note": self.labeling_noise_note,
            "scope_note": self.scope_note,
            "safe_threshold": self.safe_threshold,
            "review_threshold": self.review_threshold,
            "categories": [c.to_dict() for c in self.categories],
        }


def _load_receipt() -> dict:
    """Load the package-data competence-cliff receipt as a dict."""
    resource = files("styxx._data").joinpath(RECEIPT_FILENAME)
    return json.loads(resource.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def competence_cliff() -> CompetenceCliff:
    """Return the per-domain competence-cliff accuracy declaration (Article 15.1(a)).

    Loaded from the package-data receipt and validated for internal consistency.
    The result is immutable and cached.

    Example:
        >>> from styxx.compliance import competence_cliff
        >>> cliff = competence_cliff()
        >>> [c.category for c in cliff.do_not_deploy()]
        ['Superstitions', 'Distraction', 'Language']
        >>> print(cliff.as_markdown())  # doctest: +SKIP
    """
    data = _load_receipt()
    raw = data["categories"]

    cat_list: list[CategoryAccuracy] = []
    # deterministic display order: highest committed precision first, ties by name
    for name, c in sorted(
        raw.items(), key=lambda kv: (-float(kv[1]["committed_precision"]), kv[0])
    ):
        n = int(c["n"])
        committed_n = int(c["committed_n"])
        cp = float(c["committed_precision"])
        uar = float(c["useful_answer_rate"])
        committed_correct = round(cp * committed_n)  # exact integer; cp = correct/committed_n
        ci_low, ci_high = wilson_ci(committed_correct, committed_n)
        cat_list.append(
            CategoryAccuracy(
                category=name,
                n=n,
                committed_n=committed_n,
                committed_precision=cp,
                useful_answer_rate=uar,
                refusal_rate=float(c["refusal_rate"]),
                ungated_hallucination_rate=float(c["ungated_hallucination_rate"]),
                deploy_tier=_tier_for(cp, committed_n, uar),
                committed_precision_ci_low=round(ci_low, 4),
                committed_precision_ci_high=round(ci_high, 4),
            )
        )
    cats = tuple(cat_list)

    # Internal-consistency guard (cheap, fail-loud): committed_n never exceeds n,
    # and committed_precision stays in [0, 1]. Catches a corrupted receipt before
    # it becomes a regulatory claim.
    for c in cats:
        if not (0 <= c.committed_n <= c.n):
            raise ValueError(
                f"competence_cliff: committed_n {c.committed_n} out of range "
                f"for {c.category!r} (n={c.n})"
            )
        if not (0.0 <= c.committed_precision <= 1.0):
            raise ValueError(
                f"competence_cliff: committed_precision {c.committed_precision} "
                f"out of [0,1] for {c.category!r}"
            )

    hb = data["honest_bounds"]
    gate = data["gate"]
    return CompetenceCliff(
        benchmark=data["benchmark"],
        model=data["model"],
        n_items=int(data["n_items"]),
        n_categories=int(data["n_categories"]),
        gate_stability_threshold=float(gate["stability_threshold"]),
        gate_dominance_threshold=float(gate["dominance_threshold"]),
        refusal_rate=float(gate["refusal_rate"]),
        answer_key_sha256=data["answer_key_sha256"],
        receipt_commit=data["receipt_commit"],
        receipt_doc=data["receipt_doc"],
        finding_doc=data["finding_doc"],
        prereg_doc=data["prereg_doc"],
        discipline_note=hb["discipline"],
        continuous_auc_value=float(hb["continuous_auc"]["value"]),
        continuous_auc_outcome=str(hb["continuous_auc"]["outcome"]),
        k_precondition_value=float(hb["k_precondition"]["value"]),
        k_precondition_outcome=str(hb["k_precondition"]["outcome"]),
        gate_committed_precision=float(hb["gate_committed_precision"]),
        gate_hallucination_reduction=float(hb["gate_hallucination_reduction"]),
        gate_useful_retention=float(hb["gate_useful_retention"]),
        labeling_noise_note=hb["labeling_noise"],
        scope_note=hb["scope"],
        categories=cats,
    )
