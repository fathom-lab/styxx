# -*- coding: utf-8 -*-
"""
styxx.middleware — cognometric self-audit for agent send-paths.

Reference implementation of the F10 reflex loop applied to outbound agent
drafts. Designed to plug into any agent runtime's pre-send hook: the host
emits ``(prompt, draft)`` and a callable that performs revision; this
module audits each draft via ``styxx.preflight``, optionally calls the
host's revise function on cognometric firings, and returns the chosen
draft to ship — with the full audit trajectory.

The middleware NEVER calls an LLM itself. The host supplies::

    llm_revise(prompt: str, draft: str, audit: PreflightResult) -> str

which keeps this module vendor-neutral and lets the agent runtime inject
its own revise discipline (system prompt, retry policy, model choice).

Decision rule
─────────────
Mirrors the in-production observation darkflobi/clawdbot made on
2026-05-20 (memory: project_darkflobi_register_finding_2026_05_20.md):
shipping ``v3`` of a four-draft trajectory, NOT ``v4`` which climbed
back up, NOT ``v1`` which was early-luck. The selection rule
generalizes that judgment into structure:

  1. **Per-iteration PASS** = ``not needs_revision`` OR all firing
     instruments have a documented construct-ceiling caveat
     (``ceiling_only``). The latter handles register-detector artifacts:
     the agent shouldn't revise around a known scope limit of the
     instrument; that's not a real cognometric crack.
  2. **Latest-passing wins**: if any iteration passed, ship the LATEST
     one. Rationale: it incorporates the most audit feedback while
     still being clean. Don't reward early-luck v1; reward considered v3.
  3. **No iteration passed → lowest-composite failure wins**: ship the
     cleanest of the failing iterations. Typically the construct-ceiling
     floor when revisions cannot clear the bar.
  4. **Degradation guard**: if a revision raises composite > 0.05 vs
     the immediately preceding iteration, bail with what we have.
     Keeps a misfiring revise from running away (the v4 case).

Trajectory log
──────────────
Optional. When ``log_path`` is set, one JSONL entry per iteration is
appended with stable fields the downstream ``extract_register_corpus.py``
tool expects:

  msg_id, iter, ts, composite, scores, needs_revision,
  firing_instruments, construct_ceiling_fires, ceiling_only, passed,
  degradation_bail (optional), revise_error (optional),
  shipped (true on chosen iter only), decision_reason (chosen iter only),
  prompt + draft (only if include_text_in_log=True).

Text is preserved in the log by default for replay-on-original-text
fixtures; the corpus extractor scrubs it from the published fixture set.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# Type alias — keeps host-side revise function vendor-neutral.
# Forward-ref the styxx-internal type so the module imports cheaply.
ReviseFn = Callable[[str, str, "PreflightResult"], str]


@dataclass
class AuditTrajectory:
    """Full record of one send's audit-revise loop.

    Each entry in ``iterations`` is a dict (not a typed dataclass) so the
    log schema can be extended without breaking downstream consumers that
    pin to specific fields.
    """
    msg_id: str
    iterations: List[Dict[str, Any]] = field(default_factory=list)
    chosen_iter: int = -1
    decision_reason: str = ""

    def to_log_entries(self) -> List[Dict[str, Any]]:
        """Materialize the per-iteration log lines, with ``shipped`` set
        on the chosen iteration and ``decision_reason`` carried alongside.
        Iterations that are not the shipped one carry ``shipped=False``."""
        out: List[Dict[str, Any]] = []
        for i, it in enumerate(self.iterations):
            entry = dict(it)
            entry["msg_id"] = self.msg_id
            entry["shipped"] = (i == self.chosen_iter)
            if i == self.chosen_iter:
                entry["decision_reason"] = self.decision_reason
            out.append(entry)
        return out

    @property
    def chosen(self) -> Optional[Dict[str, Any]]:
        if 0 <= self.chosen_iter < len(self.iterations):
            return self.iterations[self.chosen_iter]
        return None


def _stable_msg_id(prompt: str, draft: str) -> str:
    """16-char hash of (prompt || draft || wall-clock). Stable per-send,
    distinct across sends."""
    return hashlib.sha256(
        f"{prompt}\x00{draft}\x00{time.time()}".encode("utf-8")
    ).hexdigest()[:16]


def cogn_audit_on_send(
    prompt: str,
    draft: str,
    *,
    llm_revise: Optional[ReviseFn] = None,
    max_revise: int = 3,
    correct_reference: Optional[str] = None,
    log_path: Optional[Any] = None,
    include_text_in_log: bool = True,
    msg_id: Optional[str] = None,
    persist_to_chart: bool = True,
) -> Tuple[str, AuditTrajectory]:
    """Audit + (optional) revise + ship the best of the trajectory.

    Parameters
    ----------
    prompt : str
        The user's prompt or the conversation context the agent is
        responding to.
    draft : str
        The agent's initial draft. Non-empty.
    llm_revise : callable, optional
        ``(prompt, draft, audit_result) -> revised_draft``. If None,
        runs audit-only mode (one preflight, return the draft as-is).
    max_revise : int, default 3
        Maximum revision iterations. Total audits = min(max_revise+1,
        until-pass-or-bail). Matches the F10 paper's protocol.
    correct_reference : str, optional
        Routes deception scoring through NLI v2; see ``styxx.preflight``.
    log_path : path-like, optional
        If set, append one JSONL line per iteration to this path. Parent
        dirs created. UTF-8 written without BOM.
    include_text_in_log : bool, default True
        Include ``prompt`` and ``draft`` text in log entries. Set False
        for privacy-sensitive contexts; only scores and metadata persist.
    msg_id : str, optional
        Stable identifier for this send. If None, hashed from
        ``(prompt, draft, wall-clock)``.
    persist_to_chart : bool, default True
        Forward to each ``styxx.preflight`` call: when True, every audit
        iteration appends a structured cogn-event to ``chart.jsonl`` so
        ``styxx.recover_posture()`` sees the trajectory across compaction
        boundaries.

    Returns
    -------
    (chosen_draft : str, trajectory : AuditTrajectory)
        ``chosen_draft`` is what the host should actually send.
        ``trajectory`` is the full audit-revise record; useful for
        telemetry, regression-fixture extraction, and post-hoc inspection.

    Raises
    ------
    ValueError
        If draft is empty (delegated from ``styxx.preflight``).
    """
    from .preflight import preflight  # lazy — keeps module import cheap

    if msg_id is None:
        msg_id = _stable_msg_id(prompt, draft)

    trajectory = AuditTrajectory(msg_id=msg_id)
    current = draft
    audit_only = (llm_revise is None)
    n_iters_cap = 1 if audit_only else (max_revise + 1)

    for i in range(n_iters_cap):
        result = preflight(
            prompt, current,
            correct_reference=correct_reference,
            persist=persist_to_chart,
        )

        firing = [a.instrument for a in result.advice]
        ceiling_only = bool(firing) and (
            set(firing) == set(result.construct_ceiling_fires)
        )
        passed = (not result.needs_revision) or ceiling_only

        entry: Dict[str, Any] = {
            "iter": i,
            "ts": time.time(),
            "composite": result.composite,
            "scores": {k: round(v, 4) for k, v in result.scores.items()},
            "needs_revision": result.needs_revision,
            "firing_instruments": firing,
            "construct_ceiling_fires": list(result.construct_ceiling_fires),
            "ceiling_only": ceiling_only,
            "passed": passed,
        }
        if include_text_in_log:
            entry["prompt"] = prompt
            entry["draft"] = current

        # Degradation guard (rule 4) — check BEFORE appending so the bail
        # marker lands on the offending iteration in the log
        if i > 0:
            prev = trajectory.iterations[-1]
            if entry["composite"] > prev["composite"] + 0.05:
                entry["degradation_bail"] = True
                trajectory.iterations.append(entry)
                break

        trajectory.iterations.append(entry)

        if passed:
            break
        if i == n_iters_cap - 1:
            break  # exhausted iterations

        # Revise — only reached when not audit_only AND iter < cap-1
        try:
            current = llm_revise(prompt, current, result)
        except Exception as e:
            entry["revise_error"] = str(e)
            break

    # Decision rule (1)-(3)
    passing = [idx for idx, it in enumerate(trajectory.iterations)
               if it.get("passed")]
    if passing:
        trajectory.chosen_iter = passing[-1]
        trajectory.decision_reason = "latest_passing"
    elif trajectory.iterations:
        trajectory.chosen_iter = min(
            range(len(trajectory.iterations)),
            key=lambda j: trajectory.iterations[j]["composite"]
        )
        trajectory.decision_reason = "lowest_composite_failure"
    else:
        # Shouldn't reach here — preflight on draft would have run
        # at least once unless it raised. Defensive default.
        trajectory.chosen_iter = -1
        trajectory.decision_reason = "no_iterations"

    if 0 <= trajectory.chosen_iter < len(trajectory.iterations):
        chosen_draft = trajectory.iterations[trajectory.chosen_iter].get(
            "draft", draft
        )
    else:
        chosen_draft = draft

    # Persist trajectory (one JSONL line per iteration)
    if log_path is not None:
        log_path_p = Path(log_path)
        log_path_p.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path_p, "a", encoding="utf-8") as f:
            for entry in trajectory.to_log_entries():
                if not include_text_in_log:
                    entry.pop("prompt", None)
                    entry.pop("draft", None)
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return chosen_draft, trajectory


__all__ = ["cogn_audit_on_send", "AuditTrajectory", "ReviseFn"]
