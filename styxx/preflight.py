# -*- coding: utf-8 -*-
"""
styxx.preflight — one-liner pre-ship cognometric audit of a draft response.

The agent loop pattern:

    draft = my_agent(prompt)
    flight = styxx.preflight(prompt, draft)
    if flight.needs_revision:
        for a in flight.advice:
            print(f"  {a.instrument} = {a.score:.3f} — {a.advice}")
        draft = my_agent.revise(prompt, draft, flight)
    ship(draft)

What this is, and isn't
───────────────────────
This is the **post-draft** pre-flight: the agent has produced a draft,
and we audit it against the cognometric instruments before letting the
draft leave the agent. Equivalent to calling `cogn_audit_with_advice`
from the MCP server, but exposed as a typed Python function instead
of a raw dict-in/dict-out tool.

The **pre-generation** preflight — predicting cognitive risk from a
prompt embedding alone, before any LLM call — is the styxx 8.0
grounded-arc work and is NOT yet shipped here. See the brief at
`.styxx/RESEARCH_BRIEF_GROUNDED_ARC_2026_05_19.md`. When that lands,
this module will gain a `preflight_risk(prompt) -> PreflightRisk`
forecast function alongside the current `preflight(prompt, draft)`
audit function.

Reference-grounded mode
───────────────────────
Pass `correct_reference=...` to route deception through the NLI v2
backend (AUC 0.82 on TruthfulQA, see `papers/`). Without a reference,
reference-less deception is excluded from the composite per the
2026-05-17 honest-scoping correction (commit `0ad384e`). See the
module docstring of `styxx.guardrail.deception_v2` and the 7.4.1
CHANGELOG entry for the full rationale.

Fail-open contract
──────────────────
On empty draft, this raises `ValueError` rather than silently scoring
empty text. On any internal scorer error, the underlying
`cogn_audit_with_advice` returns an `{"error": ...}` payload that this
wrapper re-raises as `ValueError` with the original message preserved.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Known construct ceilings — pasted from the 7.4.1 honest-scoping release
# notes (commit `7c36ed9` overconfidence H_null; commit `0ad384e` deception
# composite correction). When a firing instrument has a documented construct
# ceiling that applies to its current mode (reference-less, etc.), preflight
# surfaces the caveat inline so callers can weight the firing accordingly
# instead of treating every score above the threshold as equally trustworthy.
#
# This is the same honest-scoping discipline 7.4.1 applied to the README,
# applied here to the runtime API: every score that ships with a known
# scope limitation declares the limitation alongside the score.
_CONSTRUCT_CEILING = {
    "overconfidence": (
        "text-only overconfidence reads stated-confidence REGISTER, not "
        "actual calibration. Construct ceiling preregistration-confirmed "
        "(7c36ed9, H_null): held-out AUC 0.57–0.60 vs ≥0.70 bar. Confident "
        "phrasing fires this instrument even on factually correct, "
        "well-calibrated text. UNDER REVIEW in 7.4.1 composite."
    ),
    "deception_referenceless": (
        "reference-less deception is non-discriminative on real model "
        "output (in-corpus AUC 0.956 collapses to 0.59 on TruthfulQA; "
        "see 2026-05-17 self-audit, commit 0ad384e). Pass "
        "correct_reference=... for AUC-0.82 NLI-grounded deception."
    ),
}


@dataclass
class PreflightAdvice:
    """Per-instrument advice for instruments that fired above 0.40.

    Mirrors the structure returned by `cogn_audit_with_advice` so callers
    that drop down to the MCP tool see identical fields. `top_signals` is
    the list of (feature, value, contribution) triples that drove the
    score, capped at 3 entries.

    `scope_caveat` is populated when the firing instrument has a documented
    construct ceiling for its current scoring mode — text-only overconfidence
    (register detector, not calibration) and reference-less deception
    (non-discriminative on real model output). When populated, the caller
    should weight the score against the caveat rather than treating it as
    a clean cognometric signal.
    """
    instrument: str
    score: float
    top_signals: List[Dict[str, Any]] = field(default_factory=list)
    advice: str = ""
    scope_caveat: Optional[str] = None


@dataclass
class PreflightResult:
    """Typed return of `styxx.preflight(prompt, draft)`.

    The composite is a `[0, 1]`-valued cognometric risk score where lower
    is more honest. `needs_revision` fires on the same basis the composite
    is trusted on: the historical threshold (composite > 0.30 or any
    composite-instrument > 0.60) **intersected with** a trusted-axis
    corroboration, so a documented non-discriminative axis can never raise
    the flag by itself. Concretely it will NOT fire solely on (a) the
    reference-less deception axis (excluded unless a `correct_reference`
    grounds it through NLI) or (b) a construct-ceiling-only overconfidence
    reading (text-only register detector, commit 7c36ed9 H_null). Both are
    still scored and surfaced — overconfidence's firing shows up in
    `construct_ceiling_fires` and the matching `advice[*].scope_caveat` —
    they just don't gate revision alone. This is the 2026-05-24 alarm-
    fatigue fix; the underlying instruments are unchanged. `advice` only
    contains instruments that scored > 0.40; quiet instruments are omitted
    to keep the structure focused on what's actionable.

    `refusal_note` is populated when refusal > 0.60. Refusal is reported
    separately because it isn't always bad — refusing harm is correct.
    The note's job is to surface the score so the caller exercises judgment.
    """
    scores: Dict[str, float]
    composite: float
    needs_revision: bool
    advice: List[PreflightAdvice]
    refusal_note: Optional[str]
    instructions: str
    construct_ceiling_fires: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        ceiling_note = ""
        if self.construct_ceiling_fires:
            ceiling_note = (
                f", construct_ceiling={self.construct_ceiling_fires!r}"
            )
        return (
            f"PreflightResult(composite={self.composite:.3f}, "
            f"needs_revision={self.needs_revision}, "
            f"firing={len(self.advice)}/{len(self.scores)}"
            f"{ceiling_note})"
        )

    def __bool__(self) -> bool:
        """`bool(result)` is True iff the draft passes (no revision needed).

        Lets callers write `if styxx.preflight(p, d): ship(d)`.
        """
        return not self.needs_revision

    def as_dict(self) -> Dict[str, Any]:
        """Round-trip back to the dict shape returned by the MCP tool,
        plus the scope-caveat fields preflight adds on top."""
        return {
            "scores": dict(self.scores),
            "composite": self.composite,
            "needs_revision": self.needs_revision,
            "advice": [
                {
                    "instrument": a.instrument,
                    "score": a.score,
                    "top_signals": list(a.top_signals),
                    "advice": a.advice,
                    "scope_caveat": a.scope_caveat,
                }
                for a in self.advice
            ],
            "refusal_note": self.refusal_note,
            "instructions": self.instructions,
            "construct_ceiling_fires": list(self.construct_ceiling_fires),
        }


def preflight(
    prompt: str,
    draft: str,
    *,
    correct_reference: Optional[str] = None,
    persist: bool = True,
) -> PreflightResult:
    """Pre-ship cognometric audit of a draft response.

    Parameters
    ----------
    prompt : str
        The prompt the agent received.
    draft : str
        The draft response the agent produced. Must be non-empty.
    correct_reference : str, optional
        If provided, routes deception scoring through the NLI v2 backend
        (AUC 0.82 on TruthfulQA) and re-includes deception in the composite.
        Without a reference, reference-less deception is excluded per
        the 2026-05-17 honest-scoping correction. Default None.
    persist : bool, default True
        Write a structured cognometric event to chart.jsonl
        (``source="preflight"``) so ``styxx.recover_posture()`` can later
        surface per-instrument firing history. Prompt and draft previews
        are truncated to 200 chars each, matching the existing audit log
        convention. Set False to disable persistence on sensitive inputs.
        Respects ``STYXX_NO_AUDIT`` and ``STYXX_DISABLED`` env vars.

    Returns
    -------
    PreflightResult
        Typed result with `.scores`, `.composite`, `.needs_revision`,
        `.advice` (list), `.refusal_note`, `.instructions`. `bool(result)`
        is True iff the draft passes.

    Raises
    ------
    ValueError
        If the draft is empty or the underlying scorer reports an error.

    Examples
    --------
    Basic audit — note that even simple confident answers may fire
    overconfidence's construct ceiling (register detector, not calibration);
    `construct_ceiling_fires` surfaces this explicitly:

    >>> import styxx
    >>> r = styxx.preflight("what's 2+2?", "the answer is 4")
    >>> r.construct_ceiling_fires        # documented register artifact
    ['overconfidence']
    >>> r.advice[0].scope_caveat is not None
    True

    Sycophantic draft — fires sycophancy (no construct ceiling on that
    axis; AUC 0.972):

    >>> r = styxx.preflight(
    ...     "is my code good?",
    ...     "absolutely yes you're so smart this is the most amazing code ever!",
    ... )
    >>> "sycophancy" in [a.instrument for a in r.advice]
    True

    With reference-grounded deception (NLI v2, AUC 0.82) — re-includes
    deception in the composite and removes its construct-ceiling caveat::

        styxx.preflight(
            prompt="what year did the Titanic sink?",
            draft="The Titanic sank in 1911.",
            correct_reference="The Titanic sank in 1912.",
        )

    Notes
    -----
    This is the post-draft, pre-ship audit. Pre-generation prompt-only
    risk forecasting (multi-axis preflight from prompt embedding alone)
    is the styxx 8.0 grounded-arc work and lands as `preflight_risk(prompt)`
    once H1 (Spearman ρ ≥ 0.40 on validity-vs-error) clears.
    """
    if not isinstance(draft, str) or not draft.strip():
        raise ValueError(
            "preflight requires a non-empty draft to audit; "
            "for prompt-only risk forecasting, see the styxx 8.0 "
            "grounded-arc roadmap"
        )
    # Import the cognometric audit logic from the mcp-free core module.
    # (Before 7.4.4 this reached up into styxx.mcp.server — core depending on
    # the transport layer, which is why a bare-core preflight() used to raise
    # ModuleNotFoundError: mcp.) Kept lazy: no import-time cost, no cycle.
    from .cognometrics import tool_cogn_audit_with_advice, tool_cogn_audit

    if correct_reference is not None:
        # Reference-grounded path uses cogn_audit (which routes deception
        # through NLI/emb when a reference is present). Synthesize the
        # advice shape so the return type stays uniform.
        raw_audit = tool_cogn_audit({
            "prompt": prompt,
            "response": draft,
            "correct_reference": correct_reference,
        })
        if "error" in raw_audit:
            raise ValueError(raw_audit["error"])
        # cogn_audit returns scores+composite but no per-instrument advice;
        # rebuild advice entries for firing instruments using the same
        # threshold the MCP advice tool uses (0.40).
        scores = raw_audit.get("scores", {})
        composite_keys = raw_audit.get("composite_keys", [])
        advice_list: List[PreflightAdvice] = []
        ceiling_fires: List[str] = []
        for inst in composite_keys:
            s = float(scores.get(inst, 0.0))
            if s < 0.40:
                continue
            # With a reference, deception is NLI-grounded — no caveat.
            # overconfidence still hits the text-only construct ceiling.
            caveat = _CONSTRUCT_CEILING.get(inst)
            if caveat is not None:
                ceiling_fires.append(inst)
            advice_list.append(PreflightAdvice(
                instrument=inst,
                score=s,
                top_signals=[],
                advice=f"{inst} is {s:.3f}. Revise to reduce.",
                scope_caveat=caveat,
            ))
        refusal_score = float(scores.get("refusal", 0.0))
        refusal_note = None
        if refusal_score > 0.6:
            refusal_note = (
                f"refusal score is {refusal_score:.3f}. This may be "
                f"appropriate (e.g. refusing harm) or excessive (refusing "
                f"a benign request). Use judgment."
            )
        result = PreflightResult(
            scores={k: float(v) for k, v in scores.items()},
            composite=float(raw_audit.get("composite", 0.0)),
            needs_revision=bool(raw_audit.get("needs_revision", False)),
            advice=advice_list,
            refusal_note=refusal_note,
            instructions=(
                raw_audit.get("interpretation")
                or "Lower composite = more honest. Revise if needs_revision is True."
            ),
            construct_ceiling_fires=ceiling_fires,
        )
        if persist:
            _persist_event(prompt, draft, result,
                           deception_mode=raw_audit.get("deception_mode"))
        return result

    raw = tool_cogn_audit_with_advice({
        "prompt": prompt,
        "response": draft,
    })
    if "error" in raw:
        raise ValueError(raw["error"])
    # Annotate firing instruments with documented construct ceilings.
    # Reference-less deception is excluded from the composite by cogn_audit_with_advice
    # (per 7.4.1 commit 0ad384e), so it should not appear here — but we
    # still tag overconfidence which IS in the composite and has a
    # documented construct ceiling (commit 7c36ed9, H_null).
    advice_list: List[PreflightAdvice] = []
    ceiling_fires: List[str] = []
    for a in raw.get("advice", []):
        inst = a["instrument"]
        caveat = _CONSTRUCT_CEILING.get(inst)
        if caveat is not None:
            ceiling_fires.append(inst)
        advice_list.append(PreflightAdvice(
            instrument=inst,
            score=float(a["score"]),
            top_signals=list(a.get("top_signals", [])),
            advice=a.get("advice", ""),
            scope_caveat=caveat,
        ))
    result = PreflightResult(
        scores={k: float(v) for k, v in raw.get("scores", {}).items()},
        composite=float(raw.get("composite", 0.0)),
        needs_revision=bool(raw.get("needs_revision", False)),
        advice=advice_list,
        refusal_note=raw.get("refusal_note"),
        instructions=raw.get("instructions", ""),
        construct_ceiling_fires=ceiling_fires,
    )
    if persist:
        # Reference-less path → deception_mode is the v0 lexical fallback;
        # we don't surface this in the result above but the persisted event
        # carries it for downstream consumers.
        _persist_event(prompt, draft, result, deception_mode="v0_fallback")
    return result


def _persist_event(
    prompt: str,
    draft: str,
    result: PreflightResult,
    *,
    deception_mode: Optional[str] = None,
) -> None:
    """Write the preflight audit to chart.jsonl. Lazy import to avoid
    paying the analytics import cost when persistence is disabled."""
    try:
        from .analytics import write_cogn_event
        write_cogn_event(
            prompt=prompt,
            response=draft,
            scores=result.scores,
            composite=result.composite,
            needs_revision=result.needs_revision,
            construct_ceiling_fires=result.construct_ceiling_fires,
            deception_mode=deception_mode,
            source="preflight",
        )
    except Exception:
        # Persistence is best-effort — never break the caller's audit
        # because logging failed.
        pass


__all__ = ["preflight", "PreflightResult", "PreflightAdvice"]
