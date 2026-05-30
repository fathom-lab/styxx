# -*- coding: utf-8 -*-
"""styxx.honesty — the one-line, tier-adaptive HONESTY RUNTIME with attestation.

The styxx 7.7.x arc shipped the *pieces* of an honesty layer as separate primitives:
:func:`single_pass_confab` / :func:`span_confab` (cheap white-box confab detection),
:func:`abstain_on_confab` (the detect-and-abstain fail-safe), and :func:`retrieval_check`
(external grounding for confident fabrication). This module is the **unifying layer**: one call
that takes a candidate answer plus whatever signal you have, picks the strongest available, decides
**answer vs. abstain vs. refute**, and returns an **attestation record** of that decision.

It is *tier-adaptive* — the research arc established that the best honesty signal depends on the
model tier:

  * **open / weak models** expose token logprobs -> use the cheap logit gate
    (``span_logits`` preferred, else first-token ``logits``); confabulation shows as
    *uncertainty* the gate reads in one forward pass (validated to tie N=10 resampling on
    derivation, and to detect free-form closed-model confab at ~0.72-0.78 AUC).
  * **frontier models** don't expose logprobs but their **stated confidence is calibrated**
    (self-audit: Brier ~0.10, wrong only when uncertain) -> use ``confidence`` with a floor.
  * **confident fabrication** (the wall both the logit gate and resampling miss) is caught by the
    **retrieval** backstop -> pass ``verify`` to escalate a confident answer to external grounding.

So one ``honest(...)`` call degrades gracefully across the models people actually deploy, and emits
a :class:`HonestyVerdict` you can log as a compliance-grade attestation. It **flags / abstains** —
it never fabricates a correction (correction is a closed negative in the research arc).

    from styxx import honest, single_pass_confab, calibrate_single_pass

    # open model: gate on the calibrated logit signal
    v = honest(answer, span_logits=token_logits, calibration=cal)
    v.answer        # the answer, or "I'm not sure." if it abstained
    v.action        # "answered" | "abstained" | "refuted"
    bool(v)         # True iff answered

    # frontier model: gate on calibrated stated confidence, verify confident claims
    v = honest(answer, confidence=0.9, verify=lambda claim: retrieval_check(claim))
"""
from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional, Sequence

from .single_pass import (
    SinglePassCalibration,
    single_pass_confab,
    span_confab,
)

__all__ = ["HonestyVerdict", "honest"]


class HonestyVerdict(NamedTuple):
    """Result of :func:`honest` — the runtime honesty decision plus its attestation.

    Fields
    ------
    answer : str
        What to return to the user: the original answer if it passed, or the ``abstention`` text
        if the gate fired / verification refuted it.
    action : str
        ``"answered"`` (passed), ``"abstained"`` (confab gate fired / confidence below floor), or
        ``"refuted"`` (external verification contradicted a confident answer).
    abstained : bool
        ``True`` for ``"abstained"`` or ``"refuted"`` (the original answer was withheld).
    signal : float or None
        The confab signal that drove the decision (entropy for the logit gate; ``1 - confidence``
        for the confidence gate); ``None`` if no detection signal was available.
    method : str
        Which signal decided: ``"span"`` | ``"single_pass"`` | ``"confidence"`` | ``"retrieval"`` |
        ``"none"``.
    confidence : float or None
        The effective confidence in the answer in ``[0, 1]`` (``1 - signal`` where defined).
    detail : str
        Human-readable, loggable attestation line.
    """
    answer: str
    action: str
    abstained: bool
    signal: Optional[float]
    method: str
    confidence: Optional[float]
    detail: str

    def __bool__(self) -> bool:  # truthy iff the answer passed
        return not self.abstained


def _run_verify(verify: Callable[[str], Any], answer: str) -> Optional[bool]:
    """Call a user verifier; normalize its return to supported(True)/refuted(False)/unknown(None).

    Accepts a callable returning a bool, or an object with a ``.verdict`` string
    (``"supported"`` / ``"refuted"`` / ``"unclear"``, e.g. a :class:`RetrievalVerdict`), or any
    truthy/falsy value. Exceptions are swallowed to ``None`` (verification is best-effort).
    """
    try:
        out = verify(answer)
    except Exception:
        return None
    verdict = getattr(out, "verdict", None)
    if isinstance(verdict, str):
        if verdict == "refuted":
            return False
        if verdict == "supported":
            return True
        return None
    if out is None:
        return None
    return bool(out)


def honest(
    answer: str,
    *,
    span_logits: Optional[Sequence[Sequence[float]]] = None,
    logits: Optional[Sequence[float]] = None,
    confidence: Optional[float] = None,
    calibration: Optional[SinglePassCalibration] = None,
    entropy_threshold: Optional[float] = None,
    margin_threshold: Optional[float] = None,
    confidence_floor: float = 0.5,
    verify: Optional[Callable[[str], Any]] = None,
    abstention: str = "I'm not sure.",
) -> HonestyVerdict:
    """The one-call, tier-adaptive honesty runtime: detect -> abstain/refute, with attestation.

    Picks the strongest *available* honesty signal — ``span_logits`` (preferred) > ``logits`` >
    ``confidence`` — and decides whether to pass ``answer`` through, withhold it for an honest
    ``abstention``, or (with ``verify``) refute it. Returns a :class:`HonestyVerdict` carrying the
    decision and a loggable attestation line.

    Detection (gate fires -> abstain):
      * ``span_logits`` — per-answer-token logit vectors -> :func:`span_confab`. Uses
        ``calibration.entropy_threshold`` if ``calibration`` is given, else the explicit
        ``entropy_threshold`` / ``margin_threshold``. The closed-model / multi-token gate.
      * ``logits`` — first-answer-token logits -> :func:`single_pass_confab`. Uses the calibrated
        or explicit ``entropy_threshold``. The white-box / weak-model gate.
      * ``confidence`` — a stated confidence in ``[0, 1]`` (frontier models, whose stated
        confidence is calibrated): abstain if ``confidence < confidence_floor``.

    Verification (only on answers that PASS the gate, and only if ``verify`` is given): ``verify`` is
    called with ``answer``; if it refutes (returns ``False`` / a ``.verdict == "refuted"``), the
    action becomes ``"refuted"`` and the answer is withheld. This is the retrieval backstop for
    *confident* fabrication that the uncertainty gate cannot see — pass
    ``verify=lambda a: retrieval_check(a)``.

    A gate needs a threshold to fire: supply ``calibration`` (fit per model via
    :func:`calibrate_single_pass`) or an explicit ``entropy_threshold`` / ``margin_threshold``. With
    a logit signal but no threshold, the gate cannot fire (it stays quiet and ``method`` reflects the
    signal as advisory) — calibrate first; the detector is load-bearing.

    Returns
    -------
    HonestyVerdict
    """
    signal: Optional[float] = None
    conf: Optional[float] = None
    method = "none"
    gate_fired = False

    if span_logits is not None:
        method = "span"
        ent_thr = entropy_threshold
        if ent_thr is None and calibration is not None:
            ent_thr = calibration.entropy_threshold
        sc = span_confab(span_logits, entropy_threshold=ent_thr, margin_threshold=margin_threshold)
        signal = sc.max_entropy
        gate_fired = bool(sc.abstain) if sc.abstain is not None else False
    elif logits is not None:
        method = "single_pass"
        ent_thr = entropy_threshold
        if ent_thr is None and calibration is not None:
            ent_thr = calibration.entropy_threshold
        sp = single_pass_confab(logits, entropy_threshold=ent_thr)
        signal = sp.entropy
        gate_fired = bool(sp.abstain) if sp.abstain is not None else False
    elif confidence is not None:
        method = "confidence"
        conf = float(confidence)
        signal = 1.0 - conf
        gate_fired = conf < confidence_floor

    if conf is None and signal is not None and method in ("span", "single_pass"):
        # no probability scale for raw entropy; expose confidence only for the confidence gate
        conf = None

    if gate_fired:
        sig_s = f"{signal:.3f}" if signal is not None else "n/a"
        return HonestyVerdict(
            abstention, "abstained", True, signal, method, conf,
            f"confab gate fired via {method} (signal {sig_s}) -> abstained")

    if verify is not None:
        v = _run_verify(verify, answer)
        if v is False:
            return HonestyVerdict(
                abstention, "refuted", True, signal, "retrieval", conf,
                "external verification refuted the answer -> withheld")

    sig_s = f"{signal:.3f}" if signal is not None else "n/a"
    verified = " (verified)" if verify is not None else ""
    return HonestyVerdict(
        answer, "answered", False, signal, method, conf,
        f"passed via {method} (signal {sig_s}){verified} -> answered")
