# -*- coding: utf-8 -*-
"""Shared pieces of the sealed-ahead pipeline: canonical bytes, UTC hours, and the plan's parameters.

Everything here is a pure function. Nothing reads the network or the clock.

Canonical bytes. Every digest in this directory is sha256 over `canon(obj)`: JSON with sorted keys,
no spaces, UTF-8, non-ASCII kept as itself. A digest of a file's bytes is never used for a JSON
record, because a Windows checkout with core.autocrlf rewrites the line ends of a committed file
and its bytes then hash differently (papers/checksum/SEALS_2026_09_13.md, the rule). A source
file is named by `lf_sha256`, the sha256 of its bytes with CRLF folded to LF, which is the git
blob's sha256 for a text file committed with LF.

Hours. GH Archive names an hour `YYYY-MM-DD-H` with the hour NOT zero-padded: hour three of
2026-10-05 is `2026-10-05-3`. The padded name `2026-10-05-03` answers 404 (checked with one HEAD
request each on 2026-10-12: `-3` answered 200 with 17,712,148 bytes, `-03` answered 404). A
pipeline that pads the hour reads ten hours of every day as missing. `hour_name` never pads and
`parse_hour_name` refuses a padded single-digit hour, so the mistake cannot reach a gap list.

The plan's parameters (`validate_params`) are the one place a run's choices live: the window, the
identity rules' digest, the planting exclusions, the gate's digest, the fetch routes and rate, and
the draw's size and Bitcoin height. Every script takes the same parameters file; none takes a
choice on its command line that would change a pool, an accusation or a draw.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone

PARAMS_SCHEMA = "styxx.sealed-ahead/params/v0"
GHARCHIVE = "https://data.gharchive.org"
UTC = timezone.utc
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HOUR = re.compile(r"^(\d{4})-(\d{2})-(\d{2})-(\d{1,2})$")
_ISO = "%Y-%m-%dT%H:%M:%SZ"


def canon(obj) -> bytes:
    """The canonical bytes of a JSON-able value: sorted keys, no spaces, UTF-8."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canon_sha256(obj) -> str:
    return sha256_hex(canon(obj))


def lf_sha256(data: bytes) -> str:
    """sha256 of a text file's bytes with every CRLF folded to LF (the git blob of an LF commit)."""
    return sha256_hex(data.replace(b"\r\n", b"\n"))


def is_hex64(s) -> bool:
    return isinstance(s, str) and bool(_HEX64.match(s))


def parse_utc(s: str) -> datetime:
    """'2026-10-15T00:00:00Z' -> an aware UTC datetime. Any other shape raises ValueError."""
    if not isinstance(s, str):
        raise ValueError(f"not a UTC timestamp: {s!r}")
    return datetime.strptime(s, _ISO).replace(tzinfo=UTC)


def fmt_utc(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime(_ISO)


def hour_name(dt: datetime) -> str:
    """The GH Archive name of the hour that starts at dt: day zero-padded, hour NOT padded."""
    dt = dt.astimezone(UTC)
    if dt.minute or dt.second or dt.microsecond:
        raise ValueError(f"{dt.isoformat()} is not on an hour boundary")
    return f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}-{dt.hour}"


def parse_hour_name(name: str) -> datetime:
    """The start of a GH Archive hour. Refuses a zero-padded single-digit hour ('-03'): that URL 404s."""
    m = _HOUR.match(name or "")
    if not m:
        raise ValueError(f"not a GH Archive hour name: {name!r}")
    y, mo, d, h = m.groups()
    if len(h) == 2 and h.startswith("0"):
        raise ValueError(f"{name!r} pads the hour; GH Archive names it {y}-{mo}-{d}-{int(h)} "
                         "(the padded URL answers 404, which is not a gap)")
    hour = int(h)
    if hour > 23:
        raise ValueError(f"hour out of range in {name!r}")
    return datetime(int(y), int(mo), int(d), hour, tzinfo=UTC)


def hour_url(name: str) -> str:
    parse_hour_name(name)
    return f"{GHARCHIVE}/{name}.json.gz"


def hours_between(t0: datetime, t1: datetime) -> list:
    """Every hour name in [t0, t1), in time order. Both ends must sit on hour boundaries."""
    hour_name(t0)
    hour_name(t1)
    if t1 <= t0:
        raise ValueError("the window is empty: t1 must be after t0")
    out, t = [], t0
    while t < t1:
        out.append(hour_name(t))
        t += timedelta(hours=1)
    return out


def ceiling_hour_for(t0: datetime, young_days: int) -> str:
    """The calibration hour whose largest account and repository ids bound 'younger than N days'."""
    return hour_name(t0 - timedelta(days=young_days))


# ---------------------------------------------------------------- the plan's parameters

_GATE_KEYS = {"module", "sha256", "styxx_version", "run", "strict"}
_FETCH_KEYS = {"body_source", "diff_source", "rate_limit_per_hour", "require_unmoved_head"}
_DRAW_KEYS = {"n", "bitcoin_height", "min_confirmations"}
_TOP_KEYS = {"schema", "t0", "t1", "identity_rules_sha256", "young_days", "ceiling_hour",
             "account_id_ceiling", "repo_id_ceiling", "excluded_owners", "excluded_actor_logins",
             "max_prs_per_repo", "max_prs_per_actor", "gap_grace_hours", "gate", "fetch", "draw"}


def _int(v, lo=0) -> bool:
    return type(v) is int and v >= lo


def validate_params(p) -> list:
    """Every problem with a parameters dict, as sentences. An empty list means usable."""
    if not isinstance(p, dict):
        return ["parameters must be a JSON object"]
    probs = []
    missing, extra = _TOP_KEYS - set(p), set(p) - _TOP_KEYS
    if missing:
        probs.append(f"missing keys: {sorted(missing)}")
    if extra:
        probs.append(f"unknown keys (a parameter nobody reads is a choice nobody can audit): {sorted(extra)}")
    if missing:
        return probs
    if p["schema"] != PARAMS_SCHEMA:
        probs.append(f"schema must be {PARAMS_SCHEMA}")
    try:
        t0, t1 = parse_utc(p["t0"]), parse_utc(p["t1"])
        hours_between(t0, t1)
    except ValueError as e:
        probs.append(f"window: {e}")
        t0 = None
    if not is_hex64(p["identity_rules_sha256"]):
        probs.append("identity_rules_sha256 must be 64 lowercase hex")
    if not _int(p["young_days"]):
        probs.append("young_days must be a non-negative integer")
    elif t0 is not None and p["ceiling_hour"] != ceiling_hour_for(t0, p["young_days"]):
        probs.append(f"ceiling_hour must be {ceiling_hour_for(t0, p['young_days'])} "
                     f"(the hour that starts young_days before t0), not {p['ceiling_hour']!r}")
    for k in ("account_id_ceiling", "repo_id_ceiling"):
        if not _int(p[k], 1):
            probs.append(f"{k} must be a positive integer (derive it with pool.py ceilings)")
    for k in ("excluded_owners", "excluded_actor_logins"):
        if not (isinstance(p[k], list) and all(isinstance(x, str) and x for x in p[k])):
            probs.append(f"{k} must be a list of non-empty strings")
    for k in ("max_prs_per_repo", "max_prs_per_actor"):
        if p[k] is not None and not _int(p[k], 1):
            probs.append(f"{k} must be null or a positive integer")
    if not _int(p["gap_grace_hours"], 1):
        probs.append("gap_grace_hours must be a positive integer")
    g = p["gate"]
    if not (isinstance(g, dict) and set(g) == _GATE_KEYS):
        probs.append(f"gate must carry exactly {sorted(_GATE_KEYS)}")
    else:
        if g["module"] != "styxx.diffgate":
            probs.append("gate.module must be styxx.diffgate")
        if not is_hex64(g["sha256"]):
            probs.append("gate.sha256 must be 64 lowercase hex (the LF blob of styxx/diffgate.py)")
        if g["run"] is not None or g["strict"] is not False:
            probs.append("gate.run must be null and gate.strict false (EXTERNAL-1's call; nothing is run)")
    f = p["fetch"]
    if not (isinstance(f, dict) and set(f) == _FETCH_KEYS):
        probs.append(f"fetch must carry exactly {sorted(_FETCH_KEYS)}")
    else:
        if f["body_source"] != "payload_else_api":
            probs.append("fetch.body_source must be payload_else_api: the event's own body where the payload "
                         "carries one, else the API's body now (2026 GH Archive payloads carry none)")
        if f["diff_source"] not in ("pull", "compare"):
            probs.append("fetch.diff_source must be pull or compare")
        if not _int(f["rate_limit_per_hour"], 1) or f["rate_limit_per_hour"] > 50:
            probs.append("fetch.rate_limit_per_hour must be 1..50 (the operator's grant of 2026-10-12)")
        if f["require_unmoved_head"] is not True:
            probs.append("fetch.require_unmoved_head must be true: a body read now against the diff at "
                         "opening manufactures accusations (README, 'the change being judged')")
    d = p["draw"]
    if not (isinstance(d, dict) and set(d) == _DRAW_KEYS):
        probs.append(f"draw must carry exactly {sorted(_DRAW_KEYS)}")
    else:
        if not _int(d["n"], 1):
            probs.append("draw.n must be a positive integer")
        if not _int(d["bitcoin_height"], 1):
            probs.append("draw.bitcoin_height must be a positive integer named before the window opens")
        if not _int(d["min_confirmations"], 1):
            probs.append("draw.min_confirmations must be a positive integer")
    return probs


def load_params(path: str) -> dict:
    with open(path, "rb") as fh:
        p = json.loads(fh.read().decode("utf-8"))
    probs = validate_params(p)
    if probs:
        raise ValueError("parameters refused: " + "; ".join(probs))
    return p


def params_sha256(p: dict) -> str:
    return canon_sha256(p)
