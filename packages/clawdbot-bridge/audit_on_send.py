"""
clawdbot_bridge.audit_on_send

Tiny stdio bridge between clawdbot's `cognometric-audit-on-send` hook
(JavaScript handler) and the styxx primitive `styxx.cogn_audit_on_send`.

stdin:  JSON of {prompt, draft, channel?, sessionKey?, msg_id?}
stdout: JSON of {shipped_draft, n_iterations, fired, best_v, climbing_guard_fired}
exit:   0 on success, 1 on parse error, 2 on missing styxx, 3 on internal error

The bridge does NOT supply an `llm_revise` callable in this baseline:
revision via the same session model requires clawdbot session-LLM access
that the planned event payload does not yet carry. With `llm_revise=None`
the middleware runs in audit-only mode — the trajectory is logged, the
original draft ships unchanged. This is the conservative baseline; richer
modes ship when the planned event payload includes a session-LLM handle.

Usage (from the JS handler):
    python -m clawdbot_bridge.audit_on_send  < payload.json  > result.json
"""
from __future__ import annotations

import json
import sys
import traceback
from typing import Any, Dict


def _emit(obj: Dict[str, Any]) -> None:
    json.dump(obj, sys.stdout)
    sys.stdout.write("\n")
    sys.stdout.flush()


def _err(code: int, message: str) -> int:
    sys.stderr.write(f"[clawdbot_bridge.audit_on_send] {message}\n")
    return code


def main() -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        return _err(1, "empty stdin")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return _err(1, f"invalid stdin JSON: {exc}")

    prompt = payload.get("prompt")
    draft = payload.get("draft")
    if not isinstance(prompt, str) or not isinstance(draft, str):
        return _err(1, "prompt and draft must be strings")

    msg_id = payload.get("msg_id")
    if msg_id is not None and not isinstance(msg_id, str):
        msg_id = str(msg_id)

    try:
        import styxx  # noqa: F401
        from styxx import cogn_audit_on_send
    except Exception as exc:
        return _err(
            2,
            f"styxx>=7.4.2 not importable: {exc}. "
            f"`pip install styxx` and retry.",
        )

    try:
        shipped_draft, trajectory = cogn_audit_on_send(
            prompt=prompt,
            draft=draft,
            llm_revise=None,             # audit-only baseline; see module docstring
            persist_to_chart=True,       # feed recover_posture()
            msg_id=msg_id,
            include_text_in_log=False,   # privacy: log scores only
        )
    except Exception as exc:
        sys.stderr.write(traceback.format_exc())
        return _err(3, f"cogn_audit_on_send raised: {exc}")

    # Reflect against the styxx.middleware.AuditTrajectory dataclass
    # (msg_id, iterations, chosen_iter, decision_reason).
    iterations = list(getattr(trajectory, "iterations", []) or [])
    n_iterations = len(iterations)
    fired = any(bool(it.get("needs_revision")) for it in iterations)
    chosen_iter = getattr(trajectory, "chosen_iter", None)
    decision_reason = getattr(trajectory, "decision_reason", None)

    _emit(
        {
            "shipped_draft": shipped_draft,
            "n_iterations": n_iterations,
            "fired": fired,
            "chosen_iter": chosen_iter,
            "decision_reason": decision_reason,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
