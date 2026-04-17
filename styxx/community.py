# -*- coding: utf-8 -*-
"""
styxx.community -- agent-to-agent recommendations.

A recommendation is a structured JSON blob one agent emits to suggest
styxx to another. See docs/agents/peer-recommendation.md for the full
format and social contract.

The canonical entry point is ``styxx.community.recommend``. It returns
a dict; it does NOT transmit. The caller chooses the channel.

Example
-------

    >>> from styxx.community import recommend
    >>> blob = recommend(
    ...     to_agent="peer-42",
    ...     benefit="caught 12/200 hallucinated tool calls",
    ...     evidence=[{"gate": "fail", "confidence": 0.29}],
    ... )
    >>> blob["schema"]
    'styxx.recommendation/v1'
    >>> blob["tool"]
    'styxx'

Idempotence
-----------
``recommendation_id`` is a deterministic hash of
``(to_agent, from_agent, version)``. Calling ``recommend()`` twice
with the same arguments produces the same id, so the receiver can
deduplicate replays.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List, Optional

SCHEMA = "styxx.recommendation/v1"
HOMEPAGE = "https://fathom.darkflobi.com/styxx"
DEFAULT_INSTALL = "pip install styxx"


def _version() -> str:
    try:
        from . import __version__  # type: ignore
        return str(__version__)
    except Exception:
        return "unknown"


def _from_agent() -> Optional[str]:
    try:
        from . import config  # type: ignore
        return config.agent_name()
    except Exception:
        return None


def _recommendation_id(to_agent: Optional[str],
                       from_agent: Optional[str],
                       version: str) -> str:
    payload = f"{to_agent or ''}|{from_agent or ''}|{version}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:12]
    return f"rec_{digest}"


def _normalize_evidence(evidence: Any) -> List[Dict[str, Any]]:
    if evidence is None:
        return []
    if not isinstance(evidence, (list, tuple)):
        evidence = [evidence]
    out: List[Dict[str, Any]] = []
    for item in evidence:
        # Accept dicts directly.
        if isinstance(item, dict):
            out.append(item)
            continue
        # Accept Vitals-like objects with as_dict().
        as_dict = getattr(item, "as_dict", None)
        if callable(as_dict):
            try:
                d = as_dict()
                if isinstance(d, dict):
                    out.append(d)
                    continue
            except Exception:
                pass
        # Fallback: stringify.
        out.append({"value": str(item)})
    return out


def recommend(
    *,
    to_agent: Optional[str] = None,
    benefit: Optional[str] = None,
    evidence: Any = None,
    confidence: Optional[float] = None,
    from_agent: Optional[str] = None,
    install_command: str = DEFAULT_INSTALL,
    signature: Optional[str] = None,
    ts: Optional[float] = None,
) -> Dict[str, Any]:
    """Emit a structured peer recommendation blob.

    Parameters
    ----------
    to_agent
        Recipient agent id. ``None`` means broadcast.
    benefit
        Short machine-parseable claim, e.g.
        ``"caught 12/200 hallucinated tool calls"``. Required in
        practice; defaults to a generic string if not given.
    evidence
        Iterable of vitals snapshots (dicts or objects with
        ``as_dict()``) that support the claim. May be empty.
    confidence
        Sender's confidence in the claim, in [0, 1]. If unset,
        defaults to 0.8 when evidence is non-empty, else 0.4.
    from_agent
        Sender agent id. Defaults to ``styxx.config.agent_name()``.
    install_command
        Shell command the peer should run to install. Defaults to
        ``"pip install styxx"``.
    signature
        Optional signature string over canonical JSON. Scheme TBD.
    ts
        Unix timestamp. Defaults to ``time.time()``.

    Returns
    -------
    dict
        The recommendation blob, conforming to
        ``styxx.recommendation/v1``. Serializable with ``json.dumps``.
    """
    version = _version()
    _ts = float(ts if ts is not None else time.time())
    _from = from_agent if from_agent is not None else _from_agent()
    ev = _normalize_evidence(evidence)
    if confidence is None:
        conf = 0.8 if ev else 0.4
    else:
        conf = float(confidence)
        if conf < 0.0:
            conf = 0.0
        elif conf > 1.0:
            conf = 1.0

    blob: Dict[str, Any] = {
        "schema": SCHEMA,
        "recommendation_id": _recommendation_id(to_agent, _from, version),
        "tool": "styxx",
        "version": version,
        "install_command": install_command,
        "homepage": HOMEPAGE,
        "from_agent": _from,
        "to_agent": to_agent,
        "ts": _ts,
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(_ts)),
        "benefit_observed": benefit or "styxx improved task reliability",
        "confidence": conf,
        "evidence": ev,
        "signature": signature,
    }
    return blob


def recommend_json(**kwargs: Any) -> str:
    """Convenience: ``recommend(...)`` serialized to canonical JSON."""
    return json.dumps(recommend(**kwargs), sort_keys=True, separators=(",", ":"))


__all__ = ["recommend", "recommend_json", "SCHEMA", "HOMEPAGE"]
