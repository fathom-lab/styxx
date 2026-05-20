# -*- coding: utf-8 -*-
"""
styxx.streaming_preflight — runtime cognometric audit during generation.

Post-draft preflight (commit 12bd7fd) scores a complete draft before the
agent ships it. **Streaming preflight** moves that audit into the hot path:
as the response grows, the partial text is scored periodically, so the
agent can short-circuit on `needs_revision` before generation completes.

This is NOT per-token SDK integration (that requires hooking into each LLM
provider's streaming API, which lives in `styxx.reflex` for the openai
adapter). This module ships the **vendor-neutral primitive**: a stateful
session the agent feeds chunks to, with audit triggered when accumulated
text crosses an interval threshold.

Usage
─────

    session = styxx.streaming_preflight(
        prompt="is my code good?",
        audit_interval_chars=50,
    )

    for chunk in llm.stream(prompt=...):
        audit = session.append(chunk)              # returns None or audit
        if audit is not None and audit.needs_revision:
            # cognometric circuit-breaker: early termination signal
            print(f"early abort at {len(session.text)} chars: "
                  f"composite={audit.composite:.2f}")
            llm.abort()
            break

    # always run a final audit, regardless of mid-stream early-stop
    final = session.finalize()
    if final.needs_revision:
        ship_revised_version()

Design notes
────────────
- audits during accumulation use ``persist=False`` — streaming generates
  many intermediate audits that would pollute chart.jsonl with partial-
  text noise. Only the ``finalize()`` audit honors the ``persist`` flag.
- ``audit_history`` retains every interval-triggered audit (with character
  position) so the caller can inspect the trajectory after generation.
- short partial responses score noisily (a 5-char fragment fires the
  same instruments as a 5-page essay). ``min_chars_before_first_audit``
  defers the first audit until there's enough text to score meaningfully.
- ``correct_reference`` is honored on every audit when supplied — useful
  for grounded deception scoring during reference-validated generation.

Construct-ceiling self-disclosure carries through unchanged: every
``StreamingPreflightSession.last_audit`` is a normal PreflightResult,
with ``.construct_ceiling_fires`` populated when the firing instruments
are subject to documented register-detector artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .preflight import PreflightResult, preflight


@dataclass
class StreamingPreflightSession:
    """Stateful wrapper around `preflight()` for streaming generation.

    Maintains the partial response as it grows; runs `preflight()` every
    `audit_interval_chars` characters; exposes the latest audit so callers
    can short-circuit on `needs_revision` mid-stream.

    Audits during streaming pass `persist=False` to avoid polluting
    chart.jsonl with partial-text noise. The final audit (via `finalize()`)
    honors the `persist` field — default False for streaming sessions since
    a finalized audit is typically followed by a separate post-draft
    `preflight()` call.
    """
    prompt: str
    audit_interval_chars: int = 50
    min_chars_before_first_audit: int = 30
    correct_reference: Optional[str] = None
    persist: bool = False

    # ── internal state ─────────────────────────────────────────────
    _accumulated: str = field(default="", init=False, repr=False)
    _last_audited_at: int = field(default=0, init=False, repr=False)
    last_audit: Optional[PreflightResult] = field(
        default=None, init=False, repr=False,
    )
    # (char_position, audit) per interval-triggered audit
    audit_history: List[Tuple[int, PreflightResult]] = field(
        default_factory=list, init=False, repr=False,
    )
    finalized: bool = field(default=False, init=False, repr=False)

    def append(self, chunk: str) -> Optional[PreflightResult]:
        """Append a chunk to the accumulated response. If the accumulated
        text has grown past the next audit interval, run preflight on the
        current accumulated text and return the result. Otherwise return
        None.

        Calling this after `finalize()` raises RuntimeError.
        """
        if self.finalized:
            raise RuntimeError(
                "StreamingPreflightSession is finalized; further appends "
                "would invalidate the recorded final audit"
            )
        if not isinstance(chunk, str):
            chunk = str(chunk)
        self._accumulated += chunk

        # Don't audit until we have enough text to score meaningfully —
        # a 5-character fragment scores noisily on every instrument.
        if len(self._accumulated) < self.min_chars_before_first_audit:
            return None

        # Audit at every interval crossing past the first-audit threshold.
        crossed = (
            len(self._accumulated) - self._last_audited_at
            >= self.audit_interval_chars
        )
        if not crossed:
            return None

        audit = preflight(
            self.prompt,
            self._accumulated,
            correct_reference=self.correct_reference,
            persist=False,
        )
        self._last_audited_at = len(self._accumulated)
        self.last_audit = audit
        self.audit_history.append((len(self._accumulated), audit))
        return audit

    def finalize(self) -> PreflightResult:
        """Run a final audit on the full accumulated response. Honors the
        `persist` field on the session: when True, the final audit is
        written to chart.jsonl with source='preflight' so recover_posture
        can see it. Subsequent calls to append() raise RuntimeError.
        """
        final = preflight(
            self.prompt,
            self._accumulated,
            correct_reference=self.correct_reference,
            persist=self.persist,
        )
        self.last_audit = final
        self.audit_history.append((len(self._accumulated), final))
        self.finalized = True
        return final

    @property
    def text(self) -> str:
        """The accumulated response text so far."""
        return self._accumulated

    @property
    def n_audits(self) -> int:
        """How many audits have been recorded (including any finalize)."""
        return len(self.audit_history)

    def composite_trajectory(self) -> List[Tuple[int, float]]:
        """Per-audit (char_position, composite) trajectory — useful for
        plotting drift during streaming."""
        return [(pos, a.composite) for pos, a in self.audit_history]

    def __repr__(self) -> str:
        return (
            f"StreamingPreflightSession(chars={len(self._accumulated)}, "
            f"audits={len(self.audit_history)}, "
            f"finalized={self.finalized})"
        )


def streaming_preflight(
    prompt: str,
    *,
    audit_interval_chars: int = 50,
    min_chars_before_first_audit: int = 30,
    correct_reference: Optional[str] = None,
    persist: bool = False,
) -> StreamingPreflightSession:
    """Create a streaming preflight session for runtime cognometric audit.

    Returns a `StreamingPreflightSession` the caller feeds chunks to via
    `.append(chunk)`. When accumulated text crosses an audit interval, the
    session runs `preflight()` on the partial response and returns the
    result. The session's `.last_audit` is always the most recent audit.

    See module docstring for the streaming-loop pattern. The session
    integrates with any provider's streaming API — the caller drives the
    chunk loop; this module is vendor-neutral.

    Parameters
    ----------
    prompt : str
        The user's prompt, fixed for the lifetime of the session.
    audit_interval_chars : int, default 50
        How many additional characters must accumulate before triggering
        the next audit. Smaller = more audits, more overhead; larger = less
        frequent signal updates.
    min_chars_before_first_audit : int, default 30
        Defer the first audit until accumulated text reaches this length.
        Below this, fragments score noisily and the audit isn't meaningful.
    correct_reference : str, optional
        Routes deception scoring through NLI v2 (AUC 0.82) on every audit
        when supplied. Useful for reference-grounded generation.
    persist : bool, default False
        Write the `finalize()` audit to chart.jsonl. Intermediate
        interval-triggered audits never persist (would generate partial-
        text noise). Set True when the streamed response should appear
        in `recover_posture()` history alongside non-streaming preflights.
    """
    return StreamingPreflightSession(
        prompt=prompt,
        audit_interval_chars=audit_interval_chars,
        min_chars_before_first_audit=min_chars_before_first_audit,
        correct_reference=correct_reference,
        persist=persist,
    )


__all__ = ["streaming_preflight", "StreamingPreflightSession"]
