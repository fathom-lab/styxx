# -*- coding: utf-8 -*-
"""
styxx.errors — structured, agent-friendly exceptions.

StyxxError is the base class for every styxx-raised error. It carries
a stable machine-readable code, a severity, an optional retry hint,
and a reason blob — so agents can make routing / retry decisions
without scraping human-prose messages.

    try:
        ...
    except styxx.StyxxError as e:
        if e.retry:
            ...
        log(e.to_json())
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


VALID_SEVERITIES = ("info", "warn", "error", "fatal")


class StyxxError(Exception):
    """Base class for all structured styxx errors.

    Attributes:
        code:      stable machine-readable identifier (e.g. "styxx.config.missing")
        message:   human-readable description
        retry:     hint to the caller that the operation is safe to retry
        severity:  one of info | warn | error | fatal
        reason:    optional free-form context (dict / str / None)
    """

    default_code: str = "styxx.error"
    default_severity: str = "error"

    def __init__(
        self,
        code: Optional[str] = None,
        message: str = "",
        *,
        retry: bool = False,
        severity: Optional[str] = None,
        reason: Any = None,
    ) -> None:
        self.code = code or self.default_code
        self.message = message or ""
        self.retry = bool(retry)
        sev = severity or self.default_severity
        if sev not in VALID_SEVERITIES:
            sev = "error"
        self.severity = sev
        self.reason = reason
        super().__init__(self.message or self.code)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "retry": self.retry,
            "severity": self.severity,
            "reason": self.reason,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, default=str)

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} code={self.code!r} "
            f"severity={self.severity!r} retry={self.retry}>"
        )


class StyxxConfigError(StyxxError):
    default_code = "styxx.config.error"


class StyxxModelError(StyxxError):
    default_code = "styxx.model.error"


class StyxxVitalsError(StyxxError):
    default_code = "styxx.vitals.error"
