# -*- coding: utf-8 -*-
"""
Top-level guardrail entry point.

Usage:

    from styxx.guardrail import check

    verdict = check(
        prompt="Who wrote Hamlet?",
        response="Hamlet was written by William Shakespeare in 1601.",
        use_entity_verify=True,
    )
"""
from __future__ import annotations

from typing import List, Optional

from .claim_decomposer import decompose
from .entity_verify import verify_entities_batch
from .text_signals import compute_text_signal, claim_risk_text_only
from .fusion import fuse_signals, calibrate_piecewise_linear
from .policy import decide_action, ActionPolicy
from .types import Verdict, Span, SignalReading


def check(
    prompt: str,
    response: str,
    *,
    model: Optional[str] = None,
    reference: Optional[str] = None,    # knowledge grounding passage
    use_entity_verify: bool = True,
    use_grounding: bool = True,
    use_probe: bool = False,       # requires HF model loaded
    use_consensus: bool = False,   # requires API credits
    policy: Optional[ActionPolicy] = None,
    probe_scorer=None,             # preferred: ProbeScorer instance
    probe_model=None,              # fallback: loaded HF model
    probe_tokenizer=None,
    consensus_sampler=None,        # callable(prompt) -> str
    consensus_n_samples: int = 5,
    consensus_reference_samples: Optional[list] = None,
) -> Verdict:
    """Run the full guardrail pipeline on a (prompt, response) pair.

    Parameters
    ----------
    prompt : str
        The original user prompt (or empty if post-hoc).
    response : str
        The AI-generated response to evaluate.
    model : str, optional
        The model that produced the response (used to select a
        matching probe if use_probe=True).
    use_entity_verify : bool
        Query Wikipedia for every named entity in the response.
        Adds latency (~0.1s per entity) but provides the strongest
        grounding signal. Default True.
    use_probe : bool
        Run the residual-level confab probe. Requires probe_model
        + probe_tokenizer. Default False.
    use_consensus : bool
        Resample N times and measure disagreement. Requires API.
        Default False.
    policy : ActionPolicy, optional
        Thresholds for halt/retry/annotate decisions.

    Returns
    -------
    Verdict
    """
    # 1. Decompose
    claims = decompose(response)

    # 2. Text-level signal per claim
    text_response_signal = compute_text_signal(response, prompt)
    per_claim_text_risk = [
        claim_risk_text_only(c, text_response_signal)
        for c in claims
    ]

    # 3. Entity verification (response-level)
    all_entities = []
    for c in claims:
        all_entities.extend(c.entities)
    all_entities = list(dict.fromkeys(all_entities))  # dedupe, keep order

    entity_results = {}
    unverified_frac = 0.0
    unverified_entities = []
    if use_entity_verify and all_entities:
        entity_results = verify_entities_batch(all_entities)
        n_unv = sum(1 for r in entity_results.values() if not r["verified"])
        unverified_frac = n_unv / len(all_entities)
        unverified_entities = [
            e for e, r in entity_results.items() if not r["verified"]
        ]

    # 4. Probe signal (optional)
    probe_score = None
    if use_probe:
        # Priority: caller supplies a ProbeScorer instance (amortized
        # across many calls). Fallback: try the old
        # probe_model + probe_tokenizer path.
        if probe_scorer is not None:
            try:
                probe_score = probe_scorer.score(prompt, response)
            except Exception:
                probe_score = None
        elif probe_model is not None and probe_tokenizer is not None:
            try:
                from ..hallucination import hallucination_verdict
                hv = hallucination_verdict(
                    model=probe_model, tokenizer=probe_tokenizer,
                    prompt=prompt, response_text=response,
                    probe_task="confab_behavioral",
                )
                probe_score = hv.risk_score
            except Exception:
                probe_score = None

    # 5. Consensus signal (optional)
    consensus_score = None
    if use_consensus and (consensus_sampler is not None
                           or consensus_reference_samples is not None):
        try:
            from .consensus_signal import consensus_disagreement
            consensus_score = consensus_disagreement(
                prompt, response,
                sampler=consensus_sampler,
                n_samples=consensus_n_samples,
                reference_samples=consensus_reference_samples,
            )
        except Exception:
            consensus_score = None

    # 5b. Knowledge-grounding signal (optional)
    grounding_score = None
    if use_grounding and reference:
        from .knowledge_grounding import response_grounding_risk
        grounding_score = response_grounding_risk(claims, reference)

    # 6. Fuse signals
    # Response-level text risk: mean of per-claim text risks
    if per_claim_text_risk:
        response_text_risk = sum(per_claim_text_risk) / len(per_claim_text_risk)
    else:
        response_text_risk = 0.0

    signals_dict = {
        "text_claim_risk": response_text_risk,
    }
    if use_entity_verify:
        signals_dict["entity_unverified_frac"] = unverified_frac
    if probe_score is not None:
        signals_dict["probe_confab"] = probe_score
    if consensus_score is not None:
        signals_dict["consensus_disagreement"] = consensus_score
    if grounding_score is not None:
        signals_dict["knowledge_grounding"] = grounding_score

    raw_risk = fuse_signals(signals_dict)
    calibrated_risk = calibrate_piecewise_linear(raw_risk)

    # 7. Per-span output
    spans: List[Span] = []
    for c, text_risk in zip(claims, per_claim_text_risk):
        reasons = []
        if c.claim_type == "factual":
            reasons.append("concrete factual claim")
        # Entity-specific: any unverified entity in this claim?
        unverified_in_claim = [e for e in c.entities
                                if e in unverified_entities]
        if unverified_in_claim:
            reasons.append(
                f"unverified entities: {', '.join(unverified_in_claim[:3])}"
            )
        if c.has_year:
            reasons.append("contains year")
        if c.has_number:
            reasons.append("contains numeric claim")
        if c.has_quote:
            reasons.append("contains direct quote")
        if c.has_identifier:
            reasons.append("contains identifier (DOI/ISBN/arXiv)")

        # Elevate span risk if it has unverified entities
        span_risk = text_risk
        if unverified_in_claim:
            span_risk = min(1.0, span_risk + 0.3)

        spans.append(Span(
            text=c.text,
            start=c.start,
            end=c.end,
            risk=span_risk,
            reasons=reasons,
            claim_type=c.claim_type,
        ))

    # 8. Policy decision
    action = decide_action(calibrated_risk, policy)

    # 9. Build verdict
    signal_readings = [
        SignalReading(
            name="text_claim_risk",
            value=response_text_risk,
            details={"n_claims": len(claims)},
        ),
    ]
    if use_entity_verify:
        signal_readings.append(SignalReading(
            name="entity_unverified_frac",
            value=unverified_frac,
            details={
                "n_entities": len(all_entities),
                "unverified": unverified_entities,
            },
        ))
    if probe_score is not None:
        signal_readings.append(SignalReading(
            name="probe_confab",
            value=probe_score,
        ))

    return Verdict(
        prompt=prompt,
        response=response,
        risk=calibrated_risk,
        action=action,
        spans=spans,
        signals=signal_readings,
        model=model,
        threshold=(policy or ActionPolicy()).annotate_threshold,
    )


__all__ = ["check"]
