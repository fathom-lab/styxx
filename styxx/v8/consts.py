"""styxx.v8.consts — the constants every v8 module imports (INTERFACES_layer2.md, header).

Values are copied from the frozen contract verbatim. Nothing here is computed.
"""
from __future__ import annotations

__all__ = [
    "SCHEMA_VERSION",
    "TYPES",
    "RESULT_KINDS",
    "PREREG_KINDS",
    "BATTERY_KINDS",
    "REF_ROLES",
    "CERT_TAG",
    "STH_TAG",
    "SEAL_TAG",
    "FAMILIES",
    "CHANNELS",
    "ROUND_PLACES",
    "EXIT",
    "ID_RE",
]

SCHEMA_VERSION = "8.0"

# GATED S0-02: recommendation implemented; operator may reverse.
# The v0.2 type set: `eval` removed, `sublog` added, eight types.
TYPES = ("fingerprint", "battery", "prereg", "result", "promotion", "action", "challenge", "sublog")

RESULT_KINDS = ("confirmatory", "pilot", "robustness", "sensitivity", "verify", "response", "document")
PREREG_KINDS = ("study", "noise-plan")
BATTERY_KINDS = ("pool-v1", "fixed-v1", "canary-v1")
REF_ROLES = (
    "battery",
    "run",
    "selected_against",
    "pool",
    "prereg",
    "result",
    "robustness",
    "target",
    "own",
    "parent",
    "fingerprint",
    "sealed",
    "previous",
    "sensitivity",
    "noise_plan",
)
CERT_TAG, STH_TAG, SEAL_TAG = "styxx.v8/cert/1", "styxx.v8/sth/1", "styxx.v8/seal/1"
FAMILIES = ("recall", "short-reasoning", "instruction-following", "format", "refusal-boundary")
CHANNELS = ("exact", "seqlp", "topk", "resid", "lens")
ROUND_PLACES = 9
EXIT = {
    "same": 0,
    "drift": 1,
    "identity": 1,
    "inconclusive": 2,
    "skew": 2,
    "beyond-floor-coverage": 2,
    "sensitivity-unmeasured": 2,
    "mismatch": 3,
    "invalid": 4,
    "unavailable": 5,
}
ID_RE = r"^sha256:[0-9a-f]{64}$"
