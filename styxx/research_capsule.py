# -*- coding: utf-8 -*-
"""styxx.research_capsule — one HTML file that carries the chain for one preregistered experiment.

Spec id ``styxx-oath/capsule/research/v0.1``. Designed in
``next/capsule-chain/DESIGN.md`` and narrowed by its red team (``ATTACK.md``, 2026-10-11); this
is Phase A: the generic builder and verifier, no specimen. The profile knows nothing about any
particular experiment.

WHAT IT CARRIES. The PREREG bytes (frozen text with exactly one ```gates block); the anchor
line(s) in ``styxx.clock``'s ``anchors.jsonl`` format; the receipt JSON(s) the run wrote and
the gates read from; the runner source(s); the RESULT with its sidecar and sworn receipt
(``styxx.sworn``); a snapshot of every ``path:`` target the RESULT swears through, so the spans
resolve with no git; the ``styxx.charon`` entry WITH its seq; a committed run note; and the list
of independent public clocks that bind the PREREG's existence. Every artifact travels as exact
bytes (base64) beside its sha256. ``created`` sits outside the payload digest (sworn R9, charon
rule).

THREE READINGS PER LINK, never two. Each link prints an OFFLINE reading — HOLDS or FAILS, a pure
function of the bytes the file carries, re-derived by the page in a browser and by
``python -m styxx.research_capsule verify`` — an ONLINE reading — CHECKABLE-ONLINE, with the exact
public read a stranger would make (a ``git show … | sha256sum`` at a named commit, a Solana
``getTransaction``, a server timestamp), which ``verify --repo CLONE`` turns into MATCHES / DIFFERS
/ NOT-CHECKED and ``verify --network`` into CONFIRMED-ON-CHAIN / CONTRADICTED / UNAVAILABLE-NOW
with the clock's own status word beside it — and a STATED reading — STATED-BY-THE-LAB, for what
nothing here checks (when the run started and ended, on which device, what the builder saw the
chain say). The summary line is ``INTERNALLY CONSISTENT (offline)`` and never the word that
``styxx.sworn`` uses for receipt re-derivation: nothing on the page is checked against the world
unless the reader runs the public reads.

THE LABEL, verbatim on the page and in every terminal report: re-derives the chain's links
offline from the bytes it carries; a forger controlling the whole file passes every offline link;
the chain is identified by the public repository at the named commits and by the anchor on the
public chain, and the run is vouched for by nothing here — only a re-run checks it.

THREAT MODEL — what a dishonest lab can still do, and what the capsule says about each:

  * forge the whole file: fabricate a PREREG, a receipt, a RESULT, a sworn receipt, a charon line
    and an anchor line that agree with each other. Every OFFLINE link reads HOLDS. The page says
    so in its label; the file points out of itself at L0b/L2b/L4b/L5b (public reads against
    the named repository at the named commits) and L1 (the public chain, where the forger's anchor
    is the forger's wallet's — the wallet the page names is a constant inside the file, so a
    stranger confirms it at github.com/fathom-lab/styxx and in the release metadata, not here);
  * fabricate the receipt without running: five numbers re-score correctly whatever produced
    them. INVISIBLE to every link; the capsule carries the runner bytes so a re-run is
    well-defined, and prints that only a re-run checks it;
  * run many times, keep the passing receipt: INVISIBLE. The PREREG froze the seeds; a discarded
    run leaves no trace in bytes;
  * finish the run before placing the seal: INVISIBLE to the chain. A seal bounds the digest's
    existence from above and the run's from nowhere (``styxx.clock`` docstring). The page prints
    ``block_time − run_end`` with its sign and the sentence that the chain orders the PREREG
    before the receipt's READING, not before the run; when the anchor postdates the run (the
    recorded case for the pMNIST untied run) the page says so and lists third-party clocks that
    are weaker than a seal and why. The phrase "sealed before the run" never appears here;
  * seal several preregistrations, publish one (the file drawer): INVISIBLE offline. With
    ``--network`` the wallet's listed ``sealed-prereg`` memos through the anchor's slot are
    counted and listed, a lower bound a stranger can hold against the SEALS table; a lab with two
    wallets defeats the count;
  * edit a bar after the seal, report another row's verdict, type a number the receipt does not
    hold, swap the receipt after the RESULT swore to it, rebuild the charon line: CAUGHT offline,
    by L0, L3, L4, L2/L4/L5 and L5 respectively — each flips exactly its own reading. L4 holds
    only when every sworn span re-derives HELD over the carried snapshot: a typed number reads
    FAILS there whether the carried sworn receipt lies about it (says HELD) or says SWORN-FAILED
    honestly, and a span read against no leaf (UNRESOLVED, which ``styxx.sworn``'s document rule
    still calls SWORN-HELD) or a RESULT that swears nothing reads FAILS too, so ``build`` refuses
    all of them by name;
  * regenerate a committed receipt in place: VISIBLE only by public read (L2b/L4b) or by charon
    re-verify; offline the capsule carries whichever bytes were built in;
  * rebuild the ferry log with this line and a different past: the capsule pins INCLUSION at seq
    (the line at that seq has that entry_id and the chain re-derives through it), not the head,
    because the head moves with every honest ingest; the current head is printed, never compared;
  * a lying RPC endpoint, or one that no longer serves an old block: ``clock_check.rpc`` names
    the endpoint the builder trusted; UNAVAILABLE-NOW is availability, not contradiction, and
    is printed as such; ask another endpoint.

What the chain makes impossible is narrow and exact: changing what was frozen, what was measured,
what was sworn or what was logged after the fact WITHOUT THE CHANGE SHOWING. It cannot make a lab
honest about what it ran. The capsule prints both halves.

DISCLOSURE (operator decision O9, 2026-10-11: no ``sys.version`` in a sealed record or a charon
line): the payload carries no ``sys.version``, no absolute path, no drive letter, no home
directory, no worktree name, no host name, no secret and no signature, and an interpreter version
string (``Python 3.x.y``) is refused anywhere in it. ``verifier`` names no interpreter: it names
``styxx_version`` and the module digests only, each the sha256 of the module's file with CRLF read
as LF (``styxx.charon._module_pair``, the rule of charon's verifier scheme v1). Carried history is
carried as is, and that includes what a charon line names: a line of verifier scheme v1
(``papers/charon/AMENDMENT_charon_verifier_names_its_runtime_2026_10_12.md``) holds
``verifier.runtime``, the implementation, version and Unicode database of the interpreter it was
derived under, as three separate fields inside the core its ``entry_id`` digests, so a capsule
that dropped them would read FAILS at L5; a legacy line names none. A payload holding a
token-shaped string is refused at build, by name. The scan reads decoded text: every string of
the payload and of each carried JSON artifact as its parsed value, never its JSON escape (an
escaped line feed after a one-letter word is not a drive letter), every pair of a repeated key
included (``json.loads`` keeps only the last), and every other artifact as its bytes. The words
the lab never prints are scanned over what the lab writes for the capsule (the page outside its
inlined verifier, the payload's own fields, the stated facts, the public clocks, the RESULT and
the run note) and not over carried history (the frozen PREREG, receipts, runners, the snapshot,
the sworn receipt, the charon line, the anchor lines): a receipt is history and is carried as
is, and a sealed PREREG cannot be reworded. A hit is named by its place in the list,
never by echoing the word.

CLI::

    python -m styxx.research_capsule build  --repo R --at COMMIT --prereg P.md --prereg-commit C
                                            --anchors anchors.jsonl --receipt R.json [--receipt ...]
                                            [--runner F.py ...] [--run-note N.md] --result RESULT.md
                                            --charon-log LOG.jsonl [--public-clocks C.json]
                                            [--stated S.json] [--clock-check CHECK.json]
                                            [--allow-unanchored] --out X.research.html
    python -m styxx.research_capsule verify X.research.html [--repo CLONE] [--network] [--json OUT]
    python -m styxx.research_capsule datasheet X.research.html

Prior art the spec names so the word "capsule" invites no priority reading: Code Ocean's compute
capsules (2017→) and ReproZip for executable bundles; Research Objects (Bechhofer et al., 2013)
and RO-Crate (2021) for packaged research artefacts with a manifest; in-toto (2016→) for a signed
layout with per-step link metadata; OpenTimestamps and Irving & Holden (2016) for hash anchoring
of protocols; Rekor (2021) for inclusion proofs against a published checkpoint; C2PA for
provenance embedded in the asset with a verifier. This profile composes a frozen gates table that
re-scores by a pure function, a sworn-span document bound to receipt leaves, a hash-chained log and
a public-chain seal into one unsigned file with the verifier inside, and prints what each link
does not prove at the same prominence. That is a reading of READMEs, not a survey, and licenses
no claim.
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import json
import math
import numbers
import platform
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from styxx.attestation import jcs
from styxx import charon as _charon
from styxx import clock as _clock
from styxx import protocol as _protocol
from styxx import sworn as _sworn
from styxx.capsule import (_BEGIN, _END, _b64, _bare_name, _payload_depth_problem, _say, _sha256,
                           _sworn_core_sha256)
from styxx.protocol import GateSpecError

__all__ = ["SPEC", "LABEL", "SIZE_CAP", "LINKS", "compose", "build", "verify", "check_payload",
           "render_html", "datasheet", "score_frozen", "check_anchor_offline",
           "receipt_tally_problem", "disclosure_hits", "main"]

SPEC = "styxx-oath/capsule/research/v0.1"
SIZE_CAP = 1_500_000

LABEL = ("re-derives the chain's links offline from the bytes it carries; a forger controlling the "
         "whole file passes every offline link; the chain is identified by the public repository at "
         "the named commits and by the anchor on the public chain, and the run is vouched for by "
         "nothing here — only a re-run checks it")

# the three readings
HOLDS, FAILS = "HOLDS", "FAILS"
ONLINE = "CHECKABLE-ONLINE"
MATCHES, DIFFERS, NOT_CHECKED = "MATCHES", "DIFFERS", "NOT-CHECKED"
CONFIRMED, CONTRADICTED, UNAVAILABLE = "CONFIRMED-ON-CHAIN", "CONTRADICTED", "UNAVAILABLE-NOW"
STATED = "STATED-BY-THE-LAB"
NOTHING_STATED = "nothing stated"
NOTHING_ONLINE = "nothing to read online"
SUMMARY_OK = "INTERNALLY CONSISTENT (offline)"
SUMMARY_FAILS = "FAILS OFFLINE"
SUMMARY_REPO_OK = "MATCHES THE PUBLIC REPOSITORY AT"

# clock statuses, mapped to the network reading (ATTACK M2): availability is not contradiction
_CLOCK_CONTRADICTED = ("MALFORMED_LINE", "NOT_FOUND", "FAILED_TX", "MEMO_MISMATCH", "NOT_CREATOR",
                       "NO_TRANSFER", "SLOT_MISMATCH", "TIME_MISMATCH", "EARLIER_MEMO_EXISTS")
_CLOCK_UNAVAILABLE = ("RPC_ERROR", "TIME_UNAVAILABLE", "BEACON_UNAVAILABLE", "BEACON_MALFORMED",
                      "EARLIEST_UNKNOWN")

_SCORE_DEFAULT = {"receipt": 0, "metrics": "/metrics", "verdict": "/verdict", "gates": "/gates",
                  "smoke": "/config/smoke"}
_MODULES = ("styxx.research_capsule", "styxx.protocol", "styxx.sworn", "styxx.charon", "styxx.clock",
            "styxx.attestation")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
# ASCII digits only: Python's \d also matches a fullwidth digit, which strptime then reads, and
# the page's regex does not
_ISO = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
_ISO_WHAT = "an ISO-8601 UTC time that exists (YYYY-MM-DDTHH:MM:SSZ)"
_JS_BEGIN = '<script id="research-verify">'
_WALLET_NOTE = ("the wallet the anchor is checked against is a constant inside this file; confirm "
                "the address at github.com/fathom-lab/styxx (styxx/clock.py CREATOR, the README) and "
                "in the release metadata, not here")
_ORDER_NOTE = ("the chain orders the PREREG's digest before the receipt's reading, not before the "
               "run; the PREREG-then-run order rests on git author dates and public server "
               "timestamps, not on the chain")

# words the lab never prints, stored reversed so this file does not print them either; the eighth
# is the ordinal, held reversed as tests/test_unpriced_priority_sentences_withdrawn.py holds it
_HOUSE_WORDS = tuple(w[::-1] for w in ("levon", "yranoitulover", "gnikaerbdnuorg", "hguorhtkaerb",
                                       "foorp-repmat", "gniyfirev-fles", "rotceted noitanicullah",
                                       "tsrif"))
# carried history: scanned for leaks, never for the house words (a receipt is carried as is)
_HISTORY_ARTIFACTS = ("prereg", "anchors", "clock_check", "receipts", "runners", "snapshot",
                      "sworn_receipt", "charon")
_HOUSE_RX = re.compile(r"(?<![A-Za-z0-9-])(?:" + "|".join(re.escape(w) for w in _HOUSE_WORDS)
                       + r")(?![A-Za-z0-9-])", re.IGNORECASE)
_TOKEN_SHAPES = (
    (re.compile(r"\bghp_[A-Za-z0-9]{20,}"), "a GitHub personal access token"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"), "a GitHub fine-grained token"),
    (re.compile(r"\bgh[ousr]_[A-Za-z0-9]{20,}"), "a GitHub app or OAuth token"),
    (re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}"), "a GitLab token"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"), "an API key"),
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}"), "a Hugging Face token"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "an AWS access key id"),
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"), "a Slack token"),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), "a Google API key"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "a private key"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9_\-.=]{20,}"), "a bearer token"),
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"), "a JWT"),
    (re.compile(r"\b(?:api[_-]?key|secret|token|password)\b\s*[:=]\s*['\"]?[A-Za-z0-9_\-/+=]{24,}",
                re.IGNORECASE), "a key=value secret"),
)
_DISCLOSURE_SHAPES = (
    (re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]"), "a drive-letter path"),
    (re.compile(r"(?<![A-Za-z0-9])/(?:home|Users|root)/[A-Za-z0-9._\-]+"), "a home directory path"),
    (re.compile(r"\\\\\?\\"), "an extended-length Windows path"),
    (re.compile(r"\b(?:C)?Python\s+3\.\d+\.\d+\b"), "an interpreter version string"),
    (re.compile(r"(?<![A-Za-z0-9_])(?:_frequency_2026_10_11|pmnist-untied|clawd)(?![A-Za-z0-9])"),
     "a worktree or workspace name"),
)

# id, name, proves, does_not_prove
LINKS = (
    ("L0", "the preregistration, frozen",
     "the text on the page is the text whose digest the anchor names, and it freezes exactly one "
     "gates block",
     "that this text sits at the named commit (L0b, public read)"),
    ("L1", "the chain anchor",
     "offline: the line's memo is the memo its kind and digest imply and the digest is L0's; online: "
     "the digest existed no later than the block time of the recorded slot",
     "that the run began after that time; that this is the earliest seal (the browser cannot scan); "
     "that the endpoint told the truth; that the wallet is the lab's (confirm it out of band)"),
    ("L1b", "public clocks",
     "nothing offline; each is a third-party server timestamp that binds the PREREG's existence, "
     "read by the stranger",
     "a server's clock, operated by a third party, is not re-derivable; it orders nothing about "
     "the run; it is weaker than a seal and listed because for this chain the seal may postdate "
     "the run"),
    ("L2", "the run: receipt, runner, run note",
     "the receipt on the page is the receipt the RESULT's spans read (same bytes in the snapshot), "
     "it is not a smoke run, and the runner and run note are the bytes the digests name",
     "that the numbers came from a run; that the run was the frozen one; anything the run note says "
     "(stated); that the committed bytes are these (L2b, public read)"),
    ("L3", "the verdict re-scored",
     "the verdict string is the frozen table's reading of the recorded numbers and the table is "
     "total; the runner did not choose it",
     "that the bars were sensible (the PREREG's power_basis, printed beside each gate, is the lab's "
     "statement)"),
    ("L4", "the RESULT sworn to the receipt",
     "every number inside a sworn span equals the receipt leaf it names: every span reads HELD, none "
     "FAILED, MALFORMED or UNRESOLVED, re-derived over the carried snapshot; the author could not "
     "choose what the receipt says",
     "that the narrative outside the spans is true; that the right sentences were bound; that the "
     "snapshot is the commit's tree (L4b, public read)"),
    ("L5", "the ferry-log line",
     "the lab's own log carries this verdict at this seq, re-derived under the verifier build the "
     "line names, over a receipt set that includes this receipt",
     "that the log was not rebuilt: only inclusion at seq against the public log identifies the "
     "line (L5b, public read); the current head is printed, never compared"),
    ("L6", "the limit",
     "the page and the terminal print the label verbatim",
     "a forger controlling the whole file passes every offline link; L0b, L2b, L4b, L5b and L1 are "
     "where the file points out of itself"),
)


# ------------------------------------------------------------------------------------ utilities

def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _refuse(name: str, why: str):
    raise SystemExit(f"REFUSED {name}: {why}")


def _b64d(s) -> Optional[bytes]:
    try:
        return base64.b64decode(s, validate=True)
    except (ValueError, TypeError):
        return None


def _is_hex(s, n: int) -> bool:
    return isinstance(s, str) and len(s) == n and all(c in "0123456789abcdef" for c in s)


def _posix_path_problem(path) -> Optional[str]:
    """Why a path is not a repository-relative POSIX path with a portable bare file name."""
    if not isinstance(path, str) or not path:
        return "is empty"
    if path.startswith("/") or "\\" in path or ":" in path:
        return "is not repository-relative POSIX (a leading slash, a backslash or a drive)"
    if any(ord(c) < 32 for c in path):
        return "holds a control character"
    segs = path.split("/")
    if any(s in ("", ".", "..") for s in segs):
        return "has an empty, '.' or '..' segment"
    if not _bare_name(segs[-1]):
        return "ends in a file name that is not a bare portable name"
    return None


def _strict_json(data):
    """JSON with duplicate and non-ASCII keys refused (the protocol.py hook, portable)."""
    def hook(pairs):
        seen: Dict[str, Any] = {}
        for k, v in pairs:
            if not k.isascii():
                raise ValueError(f"non-ASCII key {k!r}")
            if k in seen:
                raise ValueError(f"duplicate key {k!r}")
            seen[k] = v
        return seen
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return json.loads(data, object_pairs_hook=hook)


def _pointer(obj, ptr: str) -> Tuple[Any, Optional[str]]:
    """RFC 6901 pointer into a parsed JSON value; ('' is the whole value)."""
    if ptr == "":
        return obj, None
    if not isinstance(ptr, str) or not ptr.startswith("/"):
        return None, "pointer must start with '/'"
    cur = obj
    for tok in ptr.split("/")[1:]:
        tok = tok.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, dict):
            if tok not in cur:
                return None, f"no key {tok!r}"
            cur = cur[tok]
        elif isinstance(cur, list):
            if not tok.isdigit() or int(tok) >= len(cur):
                return None, f"no index {tok!r}"
            cur = cur[int(tok)]
        else:
            return None, f"cannot step into a {type(cur).__name__} at {tok!r}"
    return cur, None


def _parse_iso(s) -> Optional[_dt.datetime]:
    """The time a ``YYYY-MM-DDTHH:MM:SSZ`` string names, or None: a string that fits the format
    but names no time (February 30, month 13, hour 24, second 60, year 0) is None too, so it ends
    in a named problem and never in a traceback; the page applies the same rule."""
    if not isinstance(s, str) or not _ISO.fullmatch(s):
        return None
    try:
        return _dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=_dt.timezone.utc)
    except ValueError:
        return None


def _int_as_written(v) -> bool:
    """An integer as Python's ``json`` reads it: an int literal, not a float literal (``2.0``,
    ``2e0``) and not a bool. The page reads the same through its typed parse (``intLit``)."""
    return isinstance(v, int) and not isinstance(v, bool)


def _signed_delta(later: _dt.datetime, earlier: _dt.datetime) -> str:
    secs = int((later - earlier).total_seconds())
    sign = "+" if secs >= 0 else "−"
    secs = abs(secs)
    h, rem = divmod(secs, 3600)
    m = rem // 60
    return f"{sign}{h} h {m:02d} min"


def _verifier_record() -> dict:
    """The capsule's own verifier: the version string and the module digests, and no interpreter
    (the DISCLOSURE). Each sha is charon's ``_module_pair``, over the module's file with CRLF read as
    LF since charon's verifier scheme v1; ``verify`` prints the version and never re-derives a sha."""
    from styxx._version import __version__
    modules = [list(_charon._module_pair(m)) for m in _MODULES]
    return {"styxx_version": __version__, "modules": modules,
            "digest": _sha256(jcs(modules).encode("utf-8"))}


def _artifact(path: str, data: bytes, **extra) -> dict:
    out = {"name": path.rsplit("/", 1)[-1], "path": path, "sha256": _sha256(data), "b64": _b64(data)}
    out.update(extra)
    return out


# ------------------------------------------------------------------------------ pure re-score

def _check_composition(name: str, g: dict, quoted, result: dict) -> None:
    """The v4 composition check of styxx.protocol, over a parsed gates block, with no git."""
    agg, over, exc = g.get("agg"), g.get("over"), g.get("excluding")
    if agg not in ("min", "max") or not isinstance(over, str) or not over:
        raise GateSpecError(f"gate {name!r}: a composition declaration needs 'agg' min|max and a "
                            f"non-empty 'over' path")
    pop = _protocol._resolve(result, over)
    if not isinstance(pop, dict) or not pop:
        raise GateSpecError(f"gate {name!r}: 'over' path {over!r} must resolve to a non-empty dict")
    excluded: set = set()
    if exc is not None:
        if not isinstance(exc, str) or not exc:
            raise GateSpecError(f"gate {name!r}: 'excluding' must be a non-empty result path")
        ex = _protocol._resolve(result, exc)
        if not isinstance(ex, list) or any(not isinstance(e, str) for e in ex):
            raise GateSpecError(f"gate {name!r}: 'excluding' must resolve to a list of member names")
        unknown = sorted(set(ex) - set(pop))
        if unknown:
            raise GateSpecError(f"gate {name!r}: 'excluding' names members absent from 'over': {unknown}")
        excluded = set(ex)
    vals = []
    for k, v in pop.items():
        if k in excluded:
            continue
        if isinstance(v, bool) or not isinstance(v, numbers.Real) or not math.isfinite(float(v)):
            raise GateSpecError(f"gate {name!r}: member {k!r} of {over!r} is {v!r}, which cannot be "
                                f"aggregated")
        vals.append(float(v))
    if not vals:
        raise GateSpecError(f"gate {name!r}: every member of {over!r} is excluded")
    recomputed = min(vals) if agg == "min" else max(vals)
    if not math.isclose(float(quoted), recomputed, rel_tol=0.0, abs_tol=1e-12):
        raise GateSpecError(f"gate {name!r}: COMPOSITION VIOLATION — metric quotes {quoted!r} but the "
                            f"declared {agg} over {over!r} recomputes to {recomputed!r}")


def _table_total(gate_names: List[str], outcomes: list) -> Optional[bool]:
    """True when every assignment of the gates reaches a row; None when there are too many gates
    to enumerate (over 12)."""
    if len(gate_names) > 12:
        return None
    for bits in range(2 ** len(gate_names)):
        fired = {n: bool(bits >> i & 1) for i, n in enumerate(gate_names)}
        if not any(all(fired.get(k) == v for k, v in row["when"].items()) for row in outcomes):
            return False
    return True


def score_frozen(gates_text: str, result: dict) -> dict:
    """The body of ``styxx.protocol.Experiment.score`` as a pure function of the gates block text
    and a result dict: no disk, no git. Returns ``{verdict, gates, gates_sha256, total,
    smoke_verdict, power_basis}``; raises ``GateSpecError`` as the Experiment does."""
    try:
        spec = _strict_json(gates_text)
    except (ValueError, UnicodeDecodeError) as e:
        raise GateSpecError(f"gates block is not valid strict JSON: {e}") from e
    if not isinstance(spec, dict):
        raise GateSpecError("gates block is not a JSON object")
    for key in ("gates", "outcomes", "smoke_verdict"):
        if key not in spec:
            raise GateSpecError(f"gates block missing {key!r}")
    gates, outcomes = spec["gates"], spec["outcomes"]
    if not isinstance(gates, dict) or not gates:
        raise GateSpecError("gates block 'gates' must be a non-empty object")
    if not isinstance(outcomes, list) or not outcomes:
        raise GateSpecError("gates block 'outcomes' must be a non-empty list")
    fired: Dict[str, bool] = {}
    power: Dict[str, Any] = {}
    for name, g in gates.items():
        if not isinstance(g, dict):
            raise GateSpecError(f"gate {name!r} is not an object")
        op = _protocol._OPS.get(g.get("op"))
        if op is None:
            raise GateSpecError(f"gate {name!r}: unknown op {g.get('op')!r}")
        if not isinstance(g.get("metric"), str) or not g["metric"]:
            raise GateSpecError(f"gate {name!r}: missing or non-string 'metric' path")
        v = _protocol._resolve(result, g["metric"])
        if isinstance(v, bool) or not isinstance(v, numbers.Real) or not math.isfinite(float(v)):
            raise GateSpecError(f"gate {name!r}: metric {g['metric']!r} resolves to {v!r}, which "
                                f"cannot be compared")
        bar = g.get("value")
        if isinstance(bar, bool) or not isinstance(bar, numbers.Real) or not math.isfinite(float(bar)):
            raise GateSpecError(f"gate {name!r}: 'value' {bar!r} is not a finite number")
        if any(k in g for k in ("agg", "over", "excluding")):
            _check_composition(name, g, v, result)
        fired[name] = bool(op(v, bar))
        power[name] = g.get("power_basis")
    verdict = None
    for row in outcomes:
        if (not isinstance(row, dict) or not isinstance(row.get("when"), dict)
                or not isinstance(row.get("verdict"), str)):
            raise GateSpecError("an outcome row is not {when: {...}, verdict: str}")
        unknown = sorted(k for k in row["when"] if k not in gates)
        if unknown:
            raise GateSpecError(f"an outcome row names gates the block does not declare: {unknown}")
        if verdict is None and all(fired.get(k) == v for k, v in row["when"].items()):
            verdict = row["verdict"]
    if verdict is None:
        raise GateSpecError(f"no outcome row matches gates {fired} — the frozen table is not total")
    return {"verdict": verdict, "gates": fired, "gates_sha256": _sha256(gates_text.encode("utf-8")),
            "total": _table_total(list(gates), outcomes), "smoke_verdict": spec["smoke_verdict"],
            "power_basis": power}


# ------------------------------------------------------------------------------ anchor, offline

def check_anchor_offline(line: dict) -> dict:
    """The pure half of ``styxx.clock.check_line``: MALFORMED_LINE or SHAPE_OK, from the line
    alone. Nothing here touches the chain."""
    out: dict = {"n": line.get("n") if isinstance(line, dict) else None}
    if not isinstance(line, dict):
        out.update(status="MALFORMED_LINE", detail="not an object")
        return out
    out.update(kind=line.get("kind"), digest=line.get("digest"), tx=line.get("tx"))
    for k in ("kind", "digest", "tx"):
        if not isinstance(line.get(k), str) or not line.get(k):
            out.update(status="MALFORMED_LINE", detail=f"missing or empty {k!r}")
            return out
    if line["kind"] not in _clock.KINDS:
        out.update(status="MALFORMED_LINE", detail=f"unknown kind {line['kind']!r}")
        return out
    try:
        expected = _clock.memo(line["kind"], line["digest"])
    except ValueError as e:
        out.update(status="MALFORMED_LINE", detail=str(e))
        return out
    out["memo_expected"] = expected
    if "memo" in line and line["memo"] != expected:
        out.update(status="MALFORMED_LINE",
                   detail="the line's recorded memo is not the memo its kind and digest imply")
        return out
    slot = line.get("slot")
    if "slot" in line and (not _int_as_written(slot) or slot < 0):
        out.update(status="MALFORMED_LINE", detail="slot is not a non-negative integer as written")
        return out
    if "block_time" in line and _parse_iso(line["block_time"]) is None:
        out.update(status="MALFORMED_LINE", detail=f"block_time is not {_ISO_WHAT}")
        return out
    out["status"] = "SHAPE_OK"
    return out


def _clock_reading(status: str) -> str:
    if status == "ANCHORED":
        return CONFIRMED
    if status in _CLOCK_CONTRADICTED:
        return CONTRADICTED
    if status in _CLOCK_UNAVAILABLE:
        return UNAVAILABLE
    return UNAVAILABLE


# ------------------------------------------------------------------- the sworn receipt, alone

def _tally_verdict(tally: dict, malformed) -> str:
    """``styxx.sworn.verify``'s document rule over a tally of span verdicts."""
    total = sum(v for v in tally.values() if _int_as_written(v))
    if malformed:
        return "SWORN-FAILED"
    if total == 0:
        return "UNSWORN"
    if tally.get("FAILED", 0) == 0 and tally.get("MALFORMED", 0) == 0:
        return "SWORN-HELD"
    return "SWORN-FAILED"


def receipt_tally_problem(srec) -> Optional[str]:
    """Why a sworn receipt's ``document_verdict``, ``counts``, ``sworn_total`` and ``unresolved``
    are not its own spans' tally under ``styxx.sworn``'s document rule, or None. A receipt whose
    spans all read HELD cannot say SWORN-FAILED. Every count, ``sworn_total`` and ``unresolved``
    is an integer as written (a ``2.0`` is not a count; JCS writes it ``2``, so no digest shows
    it), and the verdict is derived from the spans' tally. Read from the receipt alone, so the
    browser layer and this one apply the same rule; a test reads every sworn receipt committed
    under ``papers/``, chosen by its schema field and not by its file name."""
    if not isinstance(srec, dict):
        return "the sworn receipt is not an object"
    spans, counts = srec.get("spans"), srec.get("counts")
    if not isinstance(spans, list) or not isinstance(counts, dict):
        return "the sworn receipt carries no spans list or no counts object"
    # the value is not echoed: a float literal and an integer print alike once parsed
    not_int = sorted(k for k, v in counts.items() if not _int_as_written(v))
    if not_int:
        return f"its counts hold a value that is not an integer as written ({', '.join(not_int)})"
    for k in ("sworn_total", "unresolved"):
        if k in srec and not _int_as_written(srec[k]):
            return f"its {k} is not an integer as written"
    tally: Dict[str, int] = {}
    for s in spans:
        v = s.get("verdict") if isinstance(s, dict) else None
        if not isinstance(v, str):
            return "a span of the sworn receipt carries no verdict"
        tally[v] = tally.get(v, 0) + 1
    if any(k not in counts for k in tally) or any(counts[k] != tally.get(k, 0) for k in counts):
        return f"its counts {jcs(counts)} are not its spans' tally {jcs(tally)}"
    total = sum(tally.values())
    if "sworn_total" in srec and srec["sworn_total"] != total:
        return f"its sworn_total {srec['sworn_total']!r} is not its {total} spans"
    if "unresolved" in srec and srec["unresolved"] != counts.get("UNRESOLVED", 0):
        return f"its unresolved {srec['unresolved']!r} is not its UNRESOLVED count"
    want = _tally_verdict(tally, srec.get("document_malformed"))
    if srec.get("document_verdict") != want:
        return (f"it says {srec.get('document_verdict')!r} and its own spans tally {want} "
                f"{jcs(tally)}")
    return None


def _all_held_problem(core: dict) -> Optional[str]:
    """Why a re-derived sworn core is not one whose every span reads HELD, or None. L4 proves that
    every number inside a sworn span equals the leaf it names, which is more than the document
    rule: SWORN-HELD reports only that none FAILED (an UNRESOLVED span was read against no leaf),
    SWORN-FAILED is an honest receipt saying a span did not hold, and UNSWORN swears nothing. The
    page applies the same rule to the core it re-derives."""
    counts = core.get("counts") if isinstance(core.get("counts"), dict) else {}
    total = sum(v for v in counts.values() if isinstance(v, int) and not isinstance(v, bool))
    held = counts.get("HELD", 0)
    verdict = core.get("document_verdict")
    if verdict == "SWORN-HELD" and total > 0 and held == total:
        return None
    if total == 0:
        return f"the RESULT swears nothing ({verdict}): no span is read against a leaf"
    parts = [f"{k} {v}" for k, v in counts.items() if k != "HELD" and v]
    parts.append(f"the document reads {verdict}")
    return (f"{held} of {total} sworn spans read HELD ({'; '.join(parts)}): a FAILED span prints "
            f"what its leaf does not hold, an UNRESOLVED span was read against no leaf, and a capsule "
            f"carries a RESULT whose every span holds")


# ------------------------------------------------------------------------------------ compose

def _public_clock_problem(c) -> Optional[str]:
    """Why a public clock entry cannot be printed, or None. The page's ``clockProblem`` is the same
    rule, word for word: blank is what ``str.strip`` removes (``str.isspace``), which the page
    holds as an explicit class because ``String.prototype.trim`` strips another set."""
    if not isinstance(c, dict):
        return "is not an object"
    for k in ("name", "binds", "read"):
        if not isinstance(c.get(k), str) or not c[k].strip():
            return f"has no non-blank string {k!r}"
    if c.get("stated") is not None and _parse_iso(c.get("stated")) is None:
        return f"has a stated time that is not {_ISO_WHAT}"
    if "does_not_prove" in c and not isinstance(c["does_not_prove"], str):
        return "has a does_not_prove that is not text"
    return None


def compose(inputs: dict, created: Optional[str] = None) -> dict:
    """The payload, from bytes and objects. Pure: no disk, no git, no network. Refuses, by name,
    shapes it cannot carry; the chain's consistency is ``check_payload``'s to judge.

    ``inputs`` keys: ``experiment`` {prereg_name, repo, at_commit, score}, ``prereg`` {path, bytes,
    blob_commit}, ``anchors`` [line...], ``anchors_path``, ``clock_check`` (dict or None),
    ``receipts`` [{path, bytes}...], ``runners`` [...], ``run_note`` ({path, bytes} or None),
    ``result`` {path, bytes}, ``sidecar``, ``sworn_receipt``, ``snapshot`` {commit, entries:
    {path: {mode, size, sha256, bytes}}}, ``charon`` {header, entry, log_path} or None,
    ``public_clocks`` [...], ``stated`` {name: scalar}.
    """
    exp = dict(inputs.get("experiment") or {})
    score = dict(_SCORE_DEFAULT)
    score.update(exp.get("score") or {})
    exp["score"] = score
    if not isinstance(exp.get("prereg_name"), str) or not exp["prereg_name"]:
        _refuse("experiment", "experiment.prereg_name is required")
    if not isinstance(exp.get("repo"), str) or not exp["repo"] or "://" in exp["repo"]:
        _refuse("experiment", "experiment.repo must name the public repository (host/owner/name)")
    if exp.get("at_commit") is not None and not _COMMIT.fullmatch(str(exp["at_commit"])):
        _refuse("experiment", "experiment.at_commit must be a full lowercase hex object id or null")

    def art(entry: dict, what: str, **extra) -> dict:
        if not isinstance(entry, dict) or not isinstance(entry.get("bytes"), (bytes, bytearray)):
            _refuse(what, "needs {path, bytes}")
        why = _posix_path_problem(entry.get("path"))
        if why:
            _refuse(what, f"path {entry.get('path')!r} {why}")
        return _artifact(entry["path"], bytes(entry["bytes"]), **extra)

    pr = inputs.get("prereg") or {}
    blob_commit = pr.get("blob_commit")
    if blob_commit is not None and not _COMMIT.fullmatch(str(blob_commit)):
        _refuse("prereg", "blob_commit must be a full lowercase hex object id or null")
    prereg_bytes = bytes(pr.get("bytes") or b"")
    try:
        gates_text = _protocol._select_gates_block(prereg_bytes.decode("utf-8"))
    except (GateSpecError, UnicodeDecodeError) as e:
        _refuse("prereg_gates", f"the PREREG does not freeze exactly one gates block: {e}")
    prereg = art(pr, "prereg", blob_commit=blob_commit,
                 gates_sha256=_sha256(gates_text.encode("utf-8")))
    if prereg["name"] != exp["prereg_name"]:
        _refuse("prereg", f"experiment.prereg_name {exp['prereg_name']!r} is not the file's name "
                          f"{prereg['name']!r}")

    anchors = list(inputs.get("anchors") or [])
    for ln in anchors:
        if not isinstance(ln, dict):
            _refuse("anchor", "an anchor line is not an object")
    anchors_path = inputs.get("anchors_path") or "papers/charon/anchors.jsonl"
    if _posix_path_problem(anchors_path):
        _refuse("anchors_path", f"{anchors_path!r} is not a repository-relative POSIX path")
    clock_check = inputs.get("clock_check")
    if clock_check is not None and not isinstance(clock_check, dict):
        _refuse("clock_check", "must be the dict styxx.clock.check_line returned, or null")

    receipts = [art(r, "receipt") for r in (inputs.get("receipts") or [])]
    if not receipts:
        _refuse("receipt", "at least one receipt is required")
    runners = [art(r, "runner") for r in (inputs.get("runners") or [])]
    run_note = art(inputs["run_note"], "run_note") if inputs.get("run_note") else None
    result = art(inputs.get("result") or {}, "result")
    sidecar, sworn_receipt = inputs.get("sidecar"), inputs.get("sworn_receipt")
    if not isinstance(sidecar, dict) or not isinstance(sworn_receipt, dict):
        _refuse("sworn", "sidecar and sworn_receipt must be objects")

    snap = inputs.get("snapshot") or {}
    if not _COMMIT.fullmatch(str(snap.get("commit"))):
        _refuse("snapshot", "snapshot.commit must be a full lowercase hex object id")
    entries = {}
    for p, e in sorted((snap.get("entries") or {}).items()):
        why = _posix_path_problem(p)
        if why or not isinstance(e, dict) or not isinstance(e.get("bytes"), (bytes, bytearray)):
            _refuse("snapshot", f"entry {p!r} {why or 'needs {mode, size, sha256, bytes}'}")
        data = bytes(e["bytes"])
        entries[p] = {"mode": str(e.get("mode") or "100644"), "size": len(data),
                      "sha256": _sha256(data), "b64": _b64(data)}
    snapshot = {"commit": snap["commit"], "entries": entries}

    ch = inputs.get("charon")
    charon = None
    if ch is not None:
        if not isinstance(ch, dict) or not isinstance(ch.get("entry"), dict):
            _refuse("charon", "needs {header, entry, log_path}")
        entry = ch["entry"]
        log_path = ch.get("log_path") or "papers/charon/charon.log.jsonl"
        if _posix_path_problem(log_path):
            _refuse("charon", f"log_path {log_path!r} is not a repository-relative POSIX path")
        seq = entry.get("seq")
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 1:
            _refuse("charon", "the entry has no positive integer seq")
        if not _is_hex(entry.get("entry_id"), 64):
            _refuse("charon", "the entry has no 64-hex entry_id")
        header = ch.get("header")
        if header is not None and not isinstance(header, dict):
            _refuse("charon", "header must be the log's header object, or null")
        charon = {"header": header, "entry": entry, "seq": seq, "entry_id": entry["entry_id"],
                  "head_at_seq": entry["entry_id"], "log_path": log_path}

    clocks = [dict(c) if isinstance(c, dict) else c for c in (inputs.get("public_clocks") or [])]
    for c in clocks:
        why = _public_clock_problem(c)
        if why:
            _refuse("public_clock", f"a public clock {why}")
        c.setdefault("does_not_prove", "a third party's server clock as it was read; not re-derivable "
                                       "from bytes; it orders nothing about the run")
    stated = dict(inputs.get("stated") or {})
    for k, v in stated.items():
        if not isinstance(k, str) or not k.isascii() or isinstance(v, (dict, list)):
            _refuse("stated", f"stated.{k!r} must be an ASCII key with a scalar value")
    if "run_ended_utc" in stated and _parse_iso(stated["run_ended_utc"]) is None:
        # the one stated fact the capsule computes with (block_time − run_end)
        _refuse("stated", f"stated.run_ended_utc {stated['run_ended_utc']!r} is not {_ISO_WHAT}")

    payload = {
        "spec": SPEC,
        "created": created or _now(),
        "label": LABEL,
        "experiment": exp,
        "artifacts": {"prereg": prereg, "anchors": anchors, "anchors_path": anchors_path,
                      "clock_check": clock_check, "receipts": receipts, "runners": runners,
                      "run_note": run_note, "result": result, "sidecar": sidecar,
                      "sworn_receipt": sworn_receipt, "snapshot": snapshot, "charon": charon,
                      "public_clocks": clocks},
        "stated": stated,
        "verifier": _verifier_record(),
        "verifier_js": {"sha256": _sha256(_JS.encode("utf-8"))},
    }
    payload["core_sha256"] = _core_sha256(payload)
    return payload


def _core_sha256(payload: dict) -> str:
    core = {k: v for k, v in payload.items() if k not in ("created", "core_sha256")}
    return _sha256(jcs(core).encode("utf-8"))


# ----------------------------------------------------------------------------- the link checks

class _Link:
    def __init__(self, lid: str):
        self.id = lid
        _, self.name, self.proves, self.does_not_prove = next(x for x in LINKS if x[0] == lid)
        self.checked: List[str] = []
        self.problems: List[str] = []
        self.reads: List[str] = []
        self.online_detail: List[str] = []
        self.online_reading = ONLINE
        self.facts: List[str] = []

    def ok(self, text: str) -> None:
        self.checked.append(text)

    def bad(self, text: str) -> None:
        self.problems.append(text)

    def check(self, cond: bool, text: str, why: str = "") -> bool:
        if cond:
            self.ok(text)
        else:
            self.bad(text + (f" — {why}" if why else ""))
        return bool(cond)

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name,
                "offline": FAILS if self.problems else HOLDS,
                "checked": list(self.checked), "problems": list(self.problems),
                "online": {"reading": self.online_reading if self.reads else NOTHING_ONLINE,
                           "reads": list(self.reads), "detail": list(self.online_detail)},
                "stated": {"reading": STATED if self.facts else NOTHING_STATED,
                           "facts": list(self.facts)},
                "proves": self.proves, "does_not_prove": self.does_not_prove}


def _decoded(entry, what: str, link: _Link) -> Optional[bytes]:
    if not isinstance(entry, dict):
        link.bad(f"{what} is not an embedded artifact")
        return None
    data = _b64d(entry.get("b64"))
    if data is None:
        link.bad(f"{what} bytes are not canonical base64")
        return None
    link.check(_sha256(data) == entry.get("sha256"), f"{what} ({entry.get('name')}) hashes to its "
               f"recorded sha256", "the carried bytes are not the bytes the digest names")
    return data


def check_payload(payload: dict, html: Optional[str] = None) -> dict:
    """Every OFFLINE link, re-derived from the payload alone; the public reads and the stated
    facts printed beside each. ``html`` (the page) adds the page-level checks: the inlined
    verifier hashes to its sealed digest, the page is the one this styxx renders for the payload,
    the label is printed, the size is under the cap."""
    links = {lid: _Link(lid) for lid, *_ in LINKS}
    L0, L1, L1b, L2, L3, L4, L5, L6 = (links[k] for k in ("L0", "L1", "L1b", "L2", "L3", "L4", "L5", "L6"))
    advisory: List[str] = []
    payload_checks: Dict[str, Any] = {"core_sha256_ok": None, "size_ok": None, "verifier_js_ok": None,
                                      "page_ok": None, "disclosure_hits": []}
    art = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    exp = payload.get("experiment") if isinstance(payload.get("experiment"), dict) else {}
    score = dict(_SCORE_DEFAULT)
    score.update(exp.get("score") if isinstance(exp.get("score"), dict) else {})
    repo = exp.get("repo") if isinstance(exp.get("repo"), str) else "?"
    at_commit = exp.get("at_commit")

    # payload-level
    try:
        payload_checks["core_sha256_ok"] = (_core_sha256(payload) == payload.get("core_sha256"))
    except (TypeError, ValueError):
        payload_checks["core_sha256_ok"] = False
    if payload.get("spec") != SPEC:
        advisory.append(f"payload spec is {payload.get('spec')!r}, this verifier knows {SPEC}")

    # ---------------------------------------------------------------- L0 the preregistration
    pr = art.get("prereg") if isinstance(art.get("prereg"), dict) else {}
    prereg_bytes = _decoded(pr, "the PREREG", L0)
    gates_text: Optional[str] = None
    if prereg_bytes is not None:
        try:
            gates_text = _protocol._select_gates_block(prereg_bytes.decode("utf-8"))
            L0.ok("the PREREG freezes exactly one plain ```gates block")
            L0.check(_sha256(gates_text.encode("utf-8")) == pr.get("gates_sha256"),
                     "the gates block hashes to the recorded gates_sha256",
                     "the block in the text is not the block the payload names")
        except (GateSpecError, UnicodeDecodeError) as e:
            L0.bad(f"the PREREG does not freeze exactly one gates block — {str(e)[:200]}")
    if pr.get("blob_commit"):
        L0.reads.append(f"git show {pr['blob_commit']}:{pr.get('path')} | sha256sum   # expect "
                        f"{pr.get('sha256')}")
        L0.facts.append(f"frozen at commit {pr['blob_commit']} of {repo} (the builder's statement; "
                        f"L0b is the check)")
    else:
        L0.facts.append("no commit is named for the PREREG blob")

    # ---------------------------------------------------------------- L1 the chain anchor
    anchors = art.get("anchors") if isinstance(art.get("anchors"), list) else []
    sealed = [a for a in anchors if isinstance(a, dict) and a.get("kind") == "sealed-prereg"]
    L1.check(bool(sealed), "the capsule carries a sealed-prereg anchor line",
             "no sealed-prereg line is carried (UNANCHORED)")
    for a in anchors:
        shape = check_anchor_offline(a)
        L1.check(shape["status"] == "SHAPE_OK",
                 f"anchor #{shape.get('n')} ({shape.get('kind')}) memo = kind + digest, slot and "
                 f"block_time well-formed", shape.get("detail", "MALFORMED_LINE"))
        if shape["status"] == "SHAPE_OK" and a.get("kind") == "sealed-prereg":
            L1.check(a.get("digest") == pr.get("sha256"),
                     f"anchor #{a.get('n')} digest is the PREREG's sha256 (L0)",
                     "the sealed digest is not the carried PREREG's")
        if isinstance(a, dict) and isinstance(a.get("tx"), str):
            L1.reads.append(f"getTransaction {a['tx']} at {' or '.join(_clock.RPCS)} (memo, fee payer, "
                            f"slot {a.get('slot')}, blockTime {a.get('block_time')})")
    L1.reads.append(f"python -m styxx.clock verify {art.get('anchors_path')}   # the lab's checker, "
                    f"with the earliest-memo scan")
    L1.reads.append("python -m styxx.research_capsule verify THIS_FILE --network")
    cc = art.get("clock_check")
    if isinstance(cc, dict):
        L1.facts.append("the builder saw the chain say: status %s, slot %s, block_time %s, rpc %s%s"
                        % (cc.get("status"), cc.get("slot"), cc.get("block_time"), cc.get("rpc"),
                           (", beacon %s" % cc.get("beacon")) if cc.get("beacon") else ""))
    else:
        L1.facts.append("the builder recorded no clock check (the anchor was not asked of the chain "
                        "at build)")
    stated = payload.get("stated") if isinstance(payload.get("stated"), dict) else {}
    run_end = _parse_iso(stated.get("run_ended_utc"))
    for a in sealed:
        bt = _parse_iso(a.get("block_time"))
        if bt is not None and run_end is not None:
            delta = _signed_delta(bt, run_end)
            if bt >= run_end:
                L1.facts.append(f"anchor #{a.get('n')} block_time − run_end = {delta}: the anchor is "
                                f"after the run ended; {_ORDER_NOTE}")
            else:
                L1.facts.append(f"anchor #{a.get('n')} block_time − run_end = {delta}: the anchor "
                                f"precedes the run's stated end; a seal bounds the digest's existence "
                                f"from above and orders nothing else; {_ORDER_NOTE}")
        elif bt is not None:
            why = ("no run end is stated" if "run_ended_utc" not in stated else
                   f"the stated run end {stated['run_ended_utc']!r} is not {_ISO_WHAT}")
            L1.facts.append(f"anchor #{a.get('n')} block_time {a.get('block_time')}; {why}, so "
                            f"block_time − run_end cannot be printed; {_ORDER_NOTE}")
    L1.facts.append(_WALLET_NOTE)

    # ---------------------------------------------------------------- L1b public clocks
    clocks = art.get("public_clocks") if isinstance(art.get("public_clocks"), list) else []
    for i, c in enumerate(clocks):
        why = _public_clock_problem(c)
        if not L1b.check(why is None, f"public clock {i + 1} is well-formed", why or ""):
            continue
        L1b.reads.append(f"{c['name']}: {c['read']}   # binds {c['binds']}")
        if c.get("stated"):
            L1b.facts.append(f"{c['name']} read {c['stated']} when the builder looked; "
                             f"{c.get('does_not_prove')}")
    if not clocks:
        L1b.ok("no public clock is listed (nothing to check offline)")
        L1b.facts.append("the lab lists no third-party clock for this PREREG")

    # ---------------------------------------------------------------- L2 the run
    snap = art.get("snapshot") if isinstance(art.get("snapshot"), dict) else {}
    entries = snap.get("entries") if isinstance(snap.get("entries"), dict) else {}
    receipts = art.get("receipts") if isinstance(art.get("receipts"), list) else []
    run_idx = score.get("receipt")
    if not _int_as_written(run_idx):
        # named, never read as 0: a JS array reads an index written 1.0 or "1" as 1, and Python's json
        # reads neither as an integer, so a confused index would split the layers
        L2.bad(f"experiment.score.receipt is {run_idx!r}, which is not an integer as written")
        run_idx = None
    run_receipt_obj = None
    run_receipt = (receipts[run_idx] if run_idx is not None and 0 <= run_idx < len(receipts)
                   and isinstance(receipts[run_idx], dict) else None)
    L2.check(run_receipt is not None, "the run's receipt is carried", "experiment.score.receipt names no "
                                                                      "carried receipt")
    for i, r in enumerate(receipts):
        data = _decoded(r, f"receipt {i}", L2)
        if data is None:
            continue
        try:
            obj = _strict_json(data)
        except (ValueError, UnicodeDecodeError) as e:
            L2.bad(f"receipt {r.get('name')} is not strict JSON — {str(e)[:120]}")
            continue
        if r is run_receipt:
            run_receipt_obj = obj
            smoke, _why = _pointer(obj, score.get("smoke") or "/config/smoke")
            L2.check(not smoke, f"the run's receipt is not a smoke run ({score.get('smoke')} is not "
                                f"true)", "a smoke receipt scores INVALID by type and licenses nothing")
            if isinstance(obj, dict) and isinstance(obj.get("prereg"), str):
                L2.check(obj["prereg"] == pr.get("name"), "the receipt names the carried PREREG",
                         f"it names {obj['prereg']!r}")
            else:
                L2.ok("the receipt names no PREREG (nothing to compare)")
            ent = entries.get(r.get("path")) if isinstance(r.get("path"), str) else None
            L2.check(isinstance(ent, dict) and ent.get("sha256") == r.get("sha256"),
                     "the run's receipt is the receipt the RESULT's spans read (same sha256 in the "
                     "snapshot at the sworn commit)",
                     "the snapshot holds other bytes at that path, or none")
        else:
            ent = entries.get(r.get("path")) if isinstance(r.get("path"), str) else None
            if isinstance(ent, dict):
                L2.check(ent.get("sha256") == r.get("sha256"),
                         f"receipt {r.get('name')} is the one the snapshot holds",
                         "the snapshot holds other bytes at that path")
        if at_commit:
            L2.reads.append(f"git show {at_commit}:{r.get('path')} | sha256sum   # expect {r.get('sha256')}")
    for i, rn in enumerate(art.get("runners") if isinstance(art.get("runners"), list) else []):
        if _decoded(rn, f"runner {i}", L2) is not None and at_commit:
            L2.reads.append(f"git show {at_commit}:{rn.get('path')} | sha256sum   # expect {rn.get('sha256')}")
    note = art.get("run_note")
    if note is not None:
        if _decoded(note, "the run note", L2) is not None and at_commit:
            L2.reads.append(f"git show {at_commit}:{note.get('path')} | sha256sum   # expect {note.get('sha256')}")
        L2.facts.append(f"the run note {note.get('name') if isinstance(note, dict) else '?'} is the lab's "
                        f"account of the run; its digest re-derives, its content is stated")
    for k in sorted(stated):
        L2.facts.append(f"{k} = {stated[k]!r}")
    if not stated and note is None:
        L2.facts.append("the lab states nothing about the run beyond the receipt's own fields")
    if run_receipt is not None:
        L2.facts.append("the receipt's own config fields (device, library versions) are what the "
                        "runner wrote; a receipt is history and is carried as is")

    # ---------------------------------------------------------------- L3 the verdict re-scored
    if gates_text is None:
        L3.bad("no gates block could be read from the PREREG (L0)")
    elif run_receipt_obj is None:
        L3.bad("the run's receipt could not be read (L2)")
    else:
        metrics, why = _pointer(run_receipt_obj, score.get("metrics") or "")
        rec_verdict, why_v = _pointer(run_receipt_obj, score.get("verdict") or "/verdict")
        rec_gates, why_g = _pointer(run_receipt_obj, score.get("gates") or "/gates")
        if why or not isinstance(metrics, dict):
            L3.bad(f"the receipt has no metrics object at {score.get('metrics')!r} ({why or 'not an object'})")
        else:
            try:
                sc = score_frozen(gates_text, metrics)
                L3.check(sc["verdict"] == rec_verdict,
                         f"re-scored verdict {sc['verdict']} equals the recorded {rec_verdict}",
                         "the recorded verdict is not the frozen table's reading of the recorded numbers")
                L3.check(jcs(sc["gates"]) == jcs(rec_gates) if isinstance(rec_gates, dict) else False,
                         "re-scored gates equal the recorded gates map",
                         f"recorded {rec_gates!r}, re-scored {sc['gates']!r}")
                L3.check(sc["total"] is not False, "the frozen outcome table is total (every assignment "
                                                   "of the gates reaches a row)",
                         "an assignment reaches no row")
                for g, pb in sc["power_basis"].items():
                    L3.facts.append(f"{g} power_basis: {pb if isinstance(pb, str) else 'undeclared'}")
            except GateSpecError as e:
                L3.bad(f"re-scoring refused — {str(e)[:300]}")
    L3.facts.append("the recorded verdict field is what the runner wrote; the re-score reads the frozen "
                    "table (L0) over the recorded numbers (L2) and does not trust the field")

    # ---------------------------------------------------------------- L4 the RESULT sworn
    res = art.get("result") if isinstance(art.get("result"), dict) else {}
    side = art.get("sidecar") if isinstance(art.get("sidecar"), dict) else None
    srec = art.get("sworn_receipt") if isinstance(art.get("sworn_receipt"), dict) else None
    result_bytes = _decoded(res, "the RESULT", L4)
    snap_commit = snap.get("commit")
    tree_entries: Dict[str, dict] = {}
    for p, e in entries.items():
        data = _b64d(e.get("b64")) if isinstance(e, dict) else None
        if data is None or _sha256(data) != e.get("sha256"):
            L4.bad(f"snapshot entry {p} does not decode to the bytes its sha256 names")
            continue
        size = e.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size != len(data):
            # an integer as written, and the byte count: the page reads the same (a 102.0 or a "102"
            # is not a size), so neither layer hands a confused type to the resolver
            L4.bad(f"snapshot entry {p} size {size!r} is not the integer byte count {len(data)}")
            continue
        tree_entries[p] = {"mode": e.get("mode"), "size": size, "sha256": e.get("sha256"),
                           "bytes": data}
    if result_bytes is not None and side is not None and srec is not None:
        doc = srec.get("document") if isinstance(srec.get("document"), dict) else {}
        L4.check(_sha256(result_bytes) == doc.get("inline_sha256"),
                 "the RESULT bytes hash to the sworn receipt's document.inline_sha256",
                 "the carried RESULT is not the document the receipt was issued over")
        tally_why = receipt_tally_problem(srec)
        L4.check(tally_why is None, "the sworn receipt's document_verdict and counts are its own "
                                    "spans' tally", tally_why or "")
        L4.check(side.get("commit") == snap_commit,
                 f"the sidecar's commit is the snapshot's ({str(snap_commit)[:12]})",
                 f"sidecar names {side.get('commit')!r}")
        raw_spans = side.get("spans") if isinstance(side.get("spans"), list) else []
        off_bad = [i for i, s in enumerate(raw_spans) if isinstance(s, dict) and not all(
            isinstance(s.get(k), int) and not isinstance(s.get(k), bool) for k in ("start", "end"))]
        L4.check(not off_bad, "the sidecar's span offsets are integers as written",
                 f"span(s) {off_bad} carry an offset that is not an integer")
        try:
            loaded = _sworn.load_sidecar(side)
            L4.check(_sworn.render(loaded) == result_bytes, "the sidecar renders to the RESULT bytes "
                                                            "byte for byte", "it renders other bytes")
            seen_targets: set = set()
            for s in loaded["spans"]:
                parsed, _m = _sworn._parse_receipt(s.get("receipt"))
                if parsed and parsed["form"] == "path" and parsed["target"] not in seen_targets:
                    seen_targets.add(parsed["target"])
                    L4.check(parsed["target"] in tree_entries,
                             f"span target {parsed['target']} is in the snapshot",
                             "no snapshot entry; the span could only be UNRESOLVED")
            tree = _sworn.SnapshotTree(tree_entries, snap_commit)
            live = _sworn.verify(sidecar=loaded, tree=tree, commit=side.get("commit"))
            live_rec = _sworn.issue_receipt(live)
            same_build = (live["verifier"]["sworn_sha256"]
                          == (srec.get("verifier") or {}).get("sworn_sha256"))
            if not same_build:
                advisory.append("INSTRUMENT SKEW: the sworn receipt names styxx.sworn "
                                f"{str((srec.get('verifier') or {}).get('sworn_sha256'))[:12]} and this "
                                f"is {live['verifier']['sworn_sha256'][:12]}")
            # the live core projected onto the carried receipt's own schema: a /v1 receipt re-derives
            # as it did under a /v1 verifier (papers/sworn/NOTE_certifies_correction_2026_10_24.md);
            # a schema with no projection is a ValueError, read below as a failed link
            live_rec = _sworn._core_as(live_rec, srec.get("schema"))
            L4.check(_sworn_core_sha256(live_rec) == _sworn_core_sha256(srec),
                     f"the sworn core re-derives over the snapshot: {live['document_verdict']} "
                     f"{live['counts']}",
                     ("the live core differs" + (" — and the instrument moved, so this is SKEW, not "
                                                 "tamper" if not same_build else " under the same build")
                      + f" (live {live['document_verdict']} {live['counts']} vs carried "
                        f"{srec.get('document_verdict')} {srec.get('counts')})"))
            held_why = _all_held_problem(live)
            L4.check(held_why is None, f"every sworn span reads HELD against the leaf it names "
                                       f"({live['counts'].get('HELD', 0)} of {live.get('sworn_total')})",
                     held_why or "")
            chk = _sworn.verify_receipt(srec, sidecar=loaded, tree=tree)
            L4.check(chk.get("status") == "VERIFIED" and chk.get("digest_match"),
                     "the sworn receipt's digest and verdict re-derive (styxx.sworn verify_receipt)",
                     f"{chk.get('status')}: digest_match={chk.get('digest_match')} "
                     f"verdict_reproduces={chk.get('verdict_reproduces')} {chk.get('note', '')[:120]}")
            cov = live.get("coverage") or {}
            L4.facts.append("sworn spans: " + " ".join(f"{k.lower()}={v}" for k, v in live["counts"].items())
                            + f"; coverage sentence_share {cov.get('sentence_share')} (a floor; the "
                              f"narrative outside the spans is the author's and nothing here checks it)")
        except SystemExit as e:
            L4.bad(f"styxx.sworn refused the carried sidecar — {str(e.code)[:200]}")
        except (TypeError, ValueError, KeyError, RecursionError) as e:
            L4.bad(f"the sworn objects could not be read — {type(e).__name__}: {str(e)[:120]}")
    else:
        if side is None:
            L4.bad("no sidecar is carried")
        if srec is None:
            L4.bad("no sworn receipt is carried")
    for p, e in sorted(entries.items()):
        if isinstance(e, dict) and snap_commit:
            L4.reads.append(f"git show {snap_commit}:{p} | sha256sum   # expect {e.get('sha256')}")
    if res.get("name") and at_commit:
        stem = str(res["name"])[:-3] if str(res["name"]).endswith(".md") else str(res["name"])
        folder = str(res.get("path", "")).rsplit("/", 1)[0] if "/" in str(res.get("path", "")) else "."
        L4.reads.append(f"python -m styxx.sworn check {folder}/{stem}.sworn-receipt.json "
                        f"{folder}/{stem}.sworn.json --repo CLONE")
    L4.facts.append("the receipt's own certifies sentence travels inside the sworn receipt (payload."
                    "artifacts.sworn_receipt.certifies)")

    # ---------------------------------------------------------------- L5 the ferry-log line
    ch = art.get("charon") if isinstance(art.get("charon"), dict) else None
    if ch is None:
        L5.bad("no charon entry is carried")
    else:
        entry = ch.get("entry") if isinstance(ch.get("entry"), dict) else {}
        core = _charon._core_of(entry)
        try:
            want = _sha256(jcs(core).encode("utf-8"))
        except (TypeError, ValueError):
            want = None
        L5.check(entry.get("schema") == _charon.ENTRY_SCHEMA, f"entry schema is {_charon.ENTRY_SCHEMA}",
                 f"it is {entry.get('schema')!r}")
        L5.check(want is not None and entry.get("entry_id") == want,
                 "entry_id = sha256(jcs(core)) with timestamp and note outside",
                 "the line's entry_id does not re-derive from its core")
        L5.check(entry.get("seq") == ch.get("seq") and ch.get("entry_id") == entry.get("entry_id")
                 and ch.get("head_at_seq") == entry.get("entry_id"),
                 f"the carried seq {ch.get('seq')}, entry_id and head_at_seq are the line's",
                 "the pin disagrees with the line")
        L5.check(entry.get("kind") == "sworn", "the line is a sworn line", f"kind is {entry.get('kind')!r}")
        subj = entry.get("subject") if isinstance(entry.get("subject"), dict) else {}
        at = entry.get("at") if isinstance(entry.get("at"), dict) else {}
        doc = (srec or {}).get("document") if isinstance((srec or {}).get("document"), dict) else {}
        L5.check(subj.get("sha256") == doc.get("inline_sha256"),
                 "subject.sha256 is the sworn RESULT's inline_sha256", "the line is about another document")
        L5.check(subj.get("path") == res.get("path"), f"subject.path is {res.get('path')}",
                 f"it is {subj.get('path')!r}")
        L5.check(at.get("commit") == (side or {}).get("commit"), "at.commit is the sidecar's commit",
                 f"it is {at.get('commit')!r}")
        recs = entry.get("receipts") if isinstance(entry.get("receipts"), dict) else {}
        shas = recs.get("sha256") if isinstance(recs.get("sha256"), list) else []
        L5.check(run_receipt is not None and run_receipt.get("sha256") in shas,
                 f"the run's receipt is in the line's resolved receipt set (n={recs.get('n')})",
                 "the line's receipt set does not include it")
        L5.check(entry.get("verdict") == (srec or {}).get("document_verdict"),
                 f"the line's verdict {entry.get('verdict')} is the sworn receipt's",
                 f"the receipt says {(srec or {}).get('document_verdict')!r}")
        L5.check(entry.get("reproduced") is True, "the line says the committed receipt reproduced at "
                                                  "ingest", f"reproduced is {entry.get('reproduced')!r}")
        header = ch.get("header")
        if entry.get("seq") == 1:
            L5.check(isinstance(header, dict) and entry.get("prev") == _charon.header_digest(header),
                     "prev names the carried header's domain-separated digest", "it does not")
        else:
            L5.check(_is_hex(entry.get("prev"), 64), "prev is a 64-hex entry_id (the line before, read "
                                                      "online)", f"it is {entry.get('prev')!r}")
        ver = entry.get("verifier") if isinstance(entry.get("verifier"), dict) else {}
        L5.facts.append(f"the line names verifier build {str(ver.get('digest'))[:12]} (styxx "
                        f"{ver.get('styxx_version')}); the log's current head is not carried — print it "
                        f"with --repo")
        L5.reads.append(f"python -m styxx.charon status --log {ch.get('log_path')}   # the current head")
        L5.reads.append(f"python -m styxx.research_capsule verify THIS_FILE --repo CLONE   # inclusion: "
                        f"line at seq {ch.get('seq')} has entry_id {str(ch.get('entry_id'))[:12]}… and the "
                        f"chain re-derives through it")

    # ---------------------------------------------------------------- L6 the limit
    L6.check(payload.get("label") == LABEL, "the payload carries the label verbatim",
             "the label was edited or dropped")
    if html is not None:
        L6.check(LABEL in html, "the page prints the label verbatim", "the page does not print it")
    L6.facts.append(LABEL)
    for r in (L0, L2, L4, L5):
        L6.reads.extend(x for x in r.reads if x.startswith("git show"))
    L6.online_reading = ONLINE

    # ---------------------------------------------------------------- page-level
    if html is not None:
        payload_checks["size_ok"] = len(html.encode("utf-8")) <= SIZE_CAP
        js = _inlined_js(html)
        sealed_js = (payload.get("verifier_js") or {}).get("sha256") if isinstance(payload.get("verifier_js"), dict) else None
        payload_checks["verifier_js_ok"] = js is not None and _sha256(js.encode("utf-8")) == sealed_js
        if js is not None and js != _JS:
            advisory.append("INSTRUMENT SKEW: the inlined browser verifier differs from the installed "
                            "one; the sealed bytes are what layer 1 ran")
        try:
            expected = render_html(payload, js if js is not None else _JS)
            payload_checks["page_ok"] = (_outside_payload(expected).replace("\r\n", "\n")
                                         == _outside_payload(html).replace("\r\n", "\n"))
        except (TypeError, ValueError, KeyError):
            payload_checks["page_ok"] = False
        payload_checks["disclosure_hits"] = disclosure_hits(payload, html)

    out_links = [links[k].to_dict() for k, *_ in LINKS]
    ok = all(l["offline"] == HOLDS for l in out_links) and bool(payload_checks["core_sha256_ok"]) and \
        payload_checks["size_ok"] is not False and payload_checks["verifier_js_ok"] is not False and \
        payload_checks["page_ok"] is not False and not payload_checks["disclosure_hits"]
    return {"spec": payload.get("spec"), "name": pr.get("name"),
            "styxx_version_stated": (payload.get("verifier") or {}).get("styxx_version")
            if isinstance(payload.get("verifier"), dict) else None,
            "label": LABEL, "payload": payload_checks, "links": out_links, "advisory": advisory,
            "ok": ok, "summary": SUMMARY_OK if ok else SUMMARY_FAILS}


# -------------------------------------------------------------------- disclosure and secrets

def _outside_payload(html: str) -> str:
    """The page with the payload body cut out: what a key-order change of the payload leaves."""
    i = html.find(_BEGIN)
    j = html.find(_END, i) if i >= 0 else -1
    return html if i < 0 or j < 0 else html[:i + len(_BEGIN)] + html[j:]


def _without_verifier(html: str) -> str:
    """The page outside the payload with the inlined verifier cut out: the text the page prints.
    The verifier is the installed package's code, carried verbatim and sealed by digest."""
    page = _outside_payload(html)
    i = page.find(_JS_BEGIN)
    j = page.find(_END, i) if i >= 0 else -1
    return page if i < 0 or j < 0 else page[:i + len(_JS_BEGIN)] + page[j:]


class _Pairs(list):
    """A JSON object as every (key, value) pair it was written with, a repeated key included, so a
    value that a later duplicate hides from ``json.loads`` is still scanned."""


def _items(x) -> List[Tuple[Any, Any]]:
    if isinstance(x, _Pairs):
        return list(x)
    return list(x.items()) if isinstance(x, dict) else []


def _all_pairs(text: str):
    """Parsed JSON with every object as ``_Pairs``; None when the text is not JSON."""
    try:
        return json.loads(text, object_pairs_hook=_Pairs)
    except (ValueError, RecursionError):
        return None


def _decoded_lines(x, out: List[str], skip_b64: bool = True) -> None:
    """Every key and string value of a parsed JSON value as decoded text, one line each, a string
    value beside its key (``key: value``), every pair of a repeated key included; base64 bodies
    are left to the artifact walk, which scans them decoded."""
    if isinstance(x, (dict, _Pairs)):
        for k, v in _items(x):
            if skip_b64 and k == "b64":
                out.append(str(k))
            elif isinstance(v, str):
                out.append(f"{k}: {v}")
            else:
                out.append(str(k))
                _decoded_lines(v, out, skip_b64)
    elif isinstance(x, list):
        for v in x:
            _decoded_lines(v, out, skip_b64)
    elif isinstance(x, str):
        out.append(x)


def _decoded_text(x) -> str:
    lines: List[str] = []
    _decoded_lines(x, lines)
    return "\n".join(lines)


def _artifact_text(data: bytes) -> str:
    """A carried artifact as text to scan: a JSON artifact as its decoded keys and strings, every
    pair of a repeated key included (an escape is read as the character it stands for), anything
    else as its bytes."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", "replace")
    obj = _all_pairs(text)
    if obj is None:
        return text
    lines: List[str] = []
    _decoded_lines(obj, lines, skip_b64=False)
    return "\n".join(lines)


def _payload_view(payload: dict, html: Optional[str]):
    """The payload as the page writes it, every pair of a repeated key included, when the page is
    given and its payload parses; else the parsed payload."""
    if html is not None:
        i = html.find(_BEGIN)
        j = html.find(_END, i + len(_BEGIN)) if i >= 0 else -1
        if i >= 0 and j >= 0:
            view = _all_pairs(html[i + len(_BEGIN):j])
            if isinstance(view, _Pairs):
                return view
    return payload


def _scan_text(payload: dict, html: Optional[str]) -> List[Tuple[str, str, bool, bool]]:
    """(where, text, scan for leaks, scan for the house words): the page outside the payload
    (leaks over all of it, the words over what it prints), the payload's decoded strings without
    their base64, and every decoded artifact, a value behind a repeated key included. Carried
    history is scanned for leaks only."""
    out: List[Tuple[str, str, bool, bool]] = []
    if html is not None:
        out.append(("the page", _outside_payload(html), True, False))
        out.append(("the page", _without_verifier(html), False, True))
    top = _items(_payload_view(payload, html))
    art = [kv for k, v in top if k == "artifacts" for kv in _items(v)]
    own = _Pairs([kv for kv in top if kv[0] != "artifacts"]
                 + [("artifacts", _Pairs(kv for kv in art if kv[0] not in _HISTORY_ARTIFACTS))])
    out.append(("the payload", _decoded_text(own), True, True))
    out.append(("the payload", _decoded_text(_Pairs(kv for kv in art if kv[0] in _HISTORY_ARTIFACTS)),
                True, False))

    def walk(x, where, house):
        if isinstance(x, (dict, _Pairs)):
            kv = _items(x)
            names = [v for k, v in kv if k == "name"]
            for k, v in kv:
                if k == "b64" and isinstance(v, str):
                    data = _b64d(v)
                    if data is not None:
                        out.append((f"{where}/{names[-1] if names else '?'}", _artifact_text(data),
                                    True, house))
            for k, v in kv:
                walk(v, f"{where}/{k}", house)
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{where}[{i}]", house)
    for k, v in art:
        walk(v, f"artifacts/{k}", k not in _HISTORY_ARTIFACTS)
    return out


def disclosure_hits(payload: dict, html: Optional[str] = None, host: Optional[str] = None) -> List[str]:
    """Zero hits or the list: drive-letter and home paths, interpreter version strings, the lab's
    worktree names, token-shaped strings and the building host's name when one is given, over
    everything; the words the lab never prints over what the lab writes for the capsule. A word
    hit is named by its place in the list and never echoed."""
    hits: List[str] = []
    host_rx = re.compile(r"(?<![A-Za-z0-9])" + re.escape(host) + r"(?![A-Za-z0-9])", re.I) \
        if host and len(host) >= 4 else None
    for where, text, leaks, house in _scan_text(payload, html):
        if leaks:
            for rx, what in _DISCLOSURE_SHAPES:
                m = rx.search(text)
                if m:
                    hits.append(f"{where}: {what} ({m.group(0)[:40]!r})")
            for rx, what in _TOKEN_SHAPES:
                m = rx.search(text)
                if m:
                    hits.append(f"{where}: {what} ({m.group(0)[:12]!r}…)")
            if host_rx and host_rx.search(text):
                hits.append(f"{where}: the building host's name")
        if house:
            m = _HOUSE_RX.search(text)
            if m:
                n = next(i for i, w in enumerate(_HOUSE_WORDS, 1) if w.lower() == m.group(0).lower())
                hits.append(f"{where}: a word the lab never prints (entry {n} of {len(_HOUSE_WORDS)} "
                            f"on its list, at character {m.start()})")
    if re.search(r"\bsys\.version\b", json.dumps(payload.get("verifier"))):
        hits.append("verifier: sys.version (O9: never in a sealed record)")
    return hits


# ------------------------------------------------------------------------------------- build

def build(inputs: dict, out: Path, created: Optional[str] = None, allow_unanchored: bool = False) -> Path:
    """Compose, check every link, render, write, verify the written file. Refuses by name on any
    link that FAILS, on a disclosure or token hit, on an oversize page; a file that does not verify
    is not kept (the capsule rule)."""
    payload = compose(inputs, created=created)
    rep = check_payload(payload)
    bad = [l for l in rep["links"] if l["offline"] == FAILS]
    if bad:
        if allow_unanchored and len(bad) == 1 and bad[0]["id"] == "L1" and \
                all("UNANCHORED" in p for p in bad[0]["problems"]):
            pass
        else:
            lines = [f"{l['id']} {l['name']}: " + "; ".join(l["problems"]) for l in bad]
            _refuse("inconsistent_chain", "the chain does not hold offline, so no capsule is minted\n  - "
                                          + "\n  - ".join(lines))
    html = render_html(payload, _JS)
    hits = disclosure_hits(payload, html, host=platform.node())
    if hits:
        _refuse("disclosure", "the minted bytes would carry\n  - " + "\n  - ".join(hits))
    raw = html.encode("utf-8")
    if len(raw) > SIZE_CAP:
        _refuse("oversize", f"the page is {len(raw)} bytes and the cap is {SIZE_CAP}")
    out = Path(out)
    out.write_bytes(raw)
    rep = verify(out)
    if not rep["ok"] and not (allow_unanchored and _only_unanchored(rep)):
        out.unlink(missing_ok=True)
        probs = [f"{l['id']}: {p}" for l in rep["links"] for p in l["problems"]]
        _refuse("does_not_verify", "the written capsule does not verify, so it was not kept\n  - "
                                   + "\n  - ".join(probs or [json.dumps(rep["payload"])]))
    return out


def _only_unanchored(rep: dict) -> bool:
    bad = [l for l in rep["links"] if l["offline"] == FAILS]
    return (len(bad) == 1 and bad[0]["id"] == "L1" and all("UNANCHORED" in p for p in bad[0]["problems"])
            and bool(rep["payload"]["core_sha256_ok"]) and rep["payload"]["page_ok"] is not False
            and rep["payload"]["verifier_js_ok"] is not False and rep["payload"]["size_ok"] is not False
            and not rep["payload"]["disclosure_hits"])


# ------------------------------------------------------------------------------------ verify

def _read_payload(path: Path) -> Tuple[Optional[str], Optional[dict], Optional[str]]:
    try:
        html = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return None, None, f"the file could not be read as UTF-8 text ({type(e).__name__})"
    try:
        i = html.index(_BEGIN) + len(_BEGIN)
        j = html.index(_END, i)
        payload = json.loads(html[i:j])
    except (ValueError, RecursionError) as e:
        return html, None, f"no capsule payload: {type(e).__name__}: {str(e)[:120]}"
    if not isinstance(payload, dict):
        return html, None, "the capsule payload is not an object"
    deep = _payload_depth_problem(payload)
    if deep:
        return html, None, deep
    return html, payload, None


def _inlined_js(html: str) -> Optional[str]:
    i = html.find(_JS_BEGIN)
    if i < 0:
        return None
    i += len(_JS_BEGIN)
    j = html.find(_END, i)
    return None if j < 0 else html[i:j].replace("\r\n", "\n")


def verify(path: Path, repo: Optional[Path] = None, network: bool = False, fetch=None) -> dict:
    """Layer 2. Offline always; ``repo`` turns the public reads into MATCHES / DIFFERS / NOT-CHECKED
    against a clone; ``network`` asks the chain through ``styxx.clock.check_line``."""
    path = Path(path)
    html, payload, problem = _read_payload(path)
    if payload is None:
        return {"ok": False, "spec": None, "name": path.name, "summary": SUMMARY_FAILS, "links": [],
                "payload": {"problem": problem}, "advisory": [], "label": LABEL, "repo": None,
                "network": None, "bytes": len(html.encode("utf-8")) if html else None}
    rep = check_payload(payload, html)
    rep["bytes"] = len(html.encode("utf-8"))
    rep["file"] = path.name
    rep["repo"] = _repo_reads(payload, Path(repo)) if repo is not None else None
    rep["network"] = _network_reads(payload, fetch) if network else None
    if rep["repo"] is not None and rep["repo"]["differs"]:
        rep["ok"] = False
    if rep["network"] is not None and any(r["reading"] == CONTRADICTED for r in rep["network"]["anchors"]):
        rep["ok"] = False
    return rep


def _repo_blob(repo: Path, commit: str, rel: str) -> Tuple[Optional[bytes], str, str]:
    """(bytes, reading, detail) for one public read against a clone. A commit the clone does not
    have is NOT-CHECKED (a shallow clone, another repository, a commit never pushed: fetch it and
    read again); a commit the clone has with no blob at the path is DIFFERS, because the public
    repository holds no such object there."""
    tree = _sworn.GitTree(repo, commit)
    data, why = tree.blob(rel)
    if data is not None:
        return data, MATCHES, ""
    if why == "commit_absent":
        return None, NOT_CHECKED, ("commit_absent: the clone has no commit with this id (a shallow "
                                   "clone, another repository, or a commit never pushed); fetch it "
                                   "and read again")
    if why in ("path_absent", "not_a_blob"):
        rc, _out, _err = tree._git("ls-tree", "--full-tree", commit, "--", rel)
        if rc == 0:
            return None, DIFFERS, (f"{why}: the clone has commit {commit[:12]} and it holds no blob "
                                   f"at this path")
        return None, NOT_CHECKED, f"{why}: git could not list the tree at {commit[:12]} (exit {rc})"
    return None, NOT_CHECKED, why


def _repo_reads(payload: dict, repo: Path) -> dict:
    """L0b, L2b, L4b, L5b against a clone: every carried digest against the blob at the named
    commit, the anchors file at the build commit, and charon inclusion at seq."""
    art = payload.get("artifacts") or {}
    exp = payload.get("experiment") or {}
    rows: List[dict] = []

    def read(link: str, commit, rel, want) -> None:
        row = {"link": link, "commit": commit, "path": rel, "expect": want, "reading": NOT_CHECKED,
               "detail": None}
        rows.append(row)
        if not commit or not rel:
            row["detail"] = "no commit or path carried"
            return
        data, reading, detail = _repo_blob(repo, str(commit), str(rel))
        if data is None:
            row["reading"], row["detail"] = reading, detail
            return
        row["reading"] = MATCHES if _sha256(data) == want else DIFFERS
        row["detail"] = None if row["reading"] == MATCHES else f"the blob hashes to {_sha256(data)}"

    pr = art.get("prereg") or {}
    read("L0", pr.get("blob_commit"), pr.get("path"), pr.get("sha256"))
    at = exp.get("at_commit")
    for r in list(art.get("receipts") or []) + list(art.get("runners") or []) + \
            ([art["run_note"]] if art.get("run_note") else []):
        read("L2", at, r.get("path"), r.get("sha256"))
    read("L4", at, (art.get("result") or {}).get("path"), (art.get("result") or {}).get("sha256"))
    snap = art.get("snapshot") or {}
    for p, e in sorted((snap.get("entries") or {}).items()):
        read("L4", snap.get("commit"), p, e.get("sha256"))
    # the anchors file at the build commit must carry each anchor line
    anchors_row = {"link": "L1", "commit": at, "path": art.get("anchors_path"), "expect": "each carried "
                   "anchor line", "reading": NOT_CHECKED, "detail": None}
    rows.append(anchors_row)
    if at and art.get("anchors_path"):
        data, reading, detail = _repo_blob(repo, str(at), str(art["anchors_path"]))
        if data is None:
            anchors_row["reading"], anchors_row["detail"] = reading, detail
        else:
            have = []
            for raw in data.decode("utf-8", "replace").splitlines():
                try:
                    have.append(jcs(json.loads(raw)))
                except (ValueError, TypeError):
                    continue
            missing = [a.get("n") for a in art.get("anchors") or [] if jcs(a) not in have]
            anchors_row["reading"] = MATCHES if not missing else DIFFERS
            anchors_row["detail"] = None if not missing else f"anchor line(s) {missing} are not in the file"
    # charon inclusion at seq
    ch = art.get("charon")
    head_now = None
    incl = {"link": "L5", "commit": None, "path": (ch or {}).get("log_path"), "expect": f"seq "
            f"{(ch or {}).get('seq')} = {str((ch or {}).get('entry_id'))[:12]}…", "reading": NOT_CHECKED,
            "detail": None}
    rows.append(incl)
    if ch:
        log = repo / str(ch.get("log_path"))
        if not log.exists():
            incl["detail"] = "the log is not in the clone"
        else:
            header, entries, problems, _facts = _charon.read_log(log)
            chain = _charon.check_chain(header, entries)
            head_now = _charon.head_of(entries)
            target = [e for e in entries if e.get("seq") == ch.get("seq")]
            if not target:
                incl["reading"], incl["detail"] = DIFFERS, f"the log has no line at seq {ch.get('seq')}"
            else:
                line_no = target[0].get("_line")
                broken_before = [p for p in problems + chain if p.get("line") is not None
                                 and p["line"] <= line_no]
                if target[0].get("entry_id") != ch.get("entry_id"):
                    incl["reading"] = DIFFERS
                    incl["detail"] = f"the line at seq {ch.get('seq')} has entry_id {str(target[0].get('entry_id'))[:12]}…"
                elif broken_before:
                    incl["reading"] = DIFFERS
                    incl["detail"] = f"the chain does not re-derive through seq {ch.get('seq')}: {broken_before[0]['problem']}"
                else:
                    incl["reading"] = MATCHES
                    incl["detail"] = f"the chain re-derives through seq {ch.get('seq')}; {len(entries)} lines"
    return {"clone": str(repo.name), "rows": rows, "head_now": head_now,
            "matches": sum(1 for r in rows if r["reading"] == MATCHES),
            "differs": sum(1 for r in rows if r["reading"] == DIFFERS),
            "not_checked": sum(1 for r in rows if r["reading"] == NOT_CHECKED),
            "commits": sorted({str(r["commit"]) for r in rows if r["commit"] and r["reading"] == MATCHES})}


def _network_reads(payload: dict, fetch=None) -> dict:
    """L1 against the chain: ``styxx.clock.check_line`` per anchor, the status verbatim beside the
    reading; the wallet's listed sealed-prereg memos through the anchor's slot (a lower bound)."""
    fetch = fetch or _clock._rpc
    art = payload.get("artifacts") or {}
    rows = []
    for a in art.get("anchors") or []:
        try:
            r = _clock.check_line(a, fetch=fetch)
        except Exception as e:                                   # noqa: BLE001 - a status, never a crash
            r = {"n": a.get("n"), "status": "RPC_ERROR", "detail": f"{type(e).__name__}: {str(e)[:120]}"}
        rows.append({"n": a.get("n"), "kind": a.get("kind"), "status": r.get("status"),
                     "reading": _clock_reading(str(r.get("status"))), "rpc": r.get("rpc"),
                     "slot": r.get("slot"), "block_time": r.get("block_time"), "beacon": r.get("beacon"),
                     "detail": r.get("detail"), "checks": r.get("checks"),
                     "earliest_tx": r.get("earliest_tx")})
    count = None
    sealed = [a for a in art.get("anchors") or [] if a.get("kind") == "sealed-prereg"
              and isinstance(a.get("slot"), int)]
    if sealed:
        try:
            listing = _clock.wallet_memos(fetch=fetch)
            slot = max(a["slot"] for a in sealed)
            memos = [e for e in listing if not e.get("_truncated") and isinstance(e.get("memo"), str)
                     and "styxx sealed-prereg " in e["memo"] and isinstance(e.get("slot"), int)
                     and e["slot"] <= slot and e.get("err") is None]
            digests = sorted({m.group(1) for e in memos
                              for m in [re.search(r"styxx sealed-prereg ([0-9a-f]{64})", e["memo"])] if m})
            count = {"listed_through_slot": len(memos), "distinct_digests": digests,
                     "complete": not any(e.get("_truncated") for e in listing),
                     "note": "listed memos, not resolved: a transfer another key sent the wallet is in "
                             "the listing too; a lower bound to hold against the SEALS table"}
        except Exception as e:                                   # noqa: BLE001
            count = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    return {"anchors": rows, "wallet": _clock.CREATOR, "wallet_is_installed_constant": True,
            "sealed_prereg_memos": count, "endpoints": list(_clock.RPCS)}


# --------------------------------------------------------------------------------- datasheet

def datasheet(rep: dict) -> List[str]:
    """The terminal table: three readings per link, the public reads and the stated facts under
    each, the summary last. The same words the page prints."""
    lines = [f"research capsule  {rep.get('file') or rep.get('name')}   spec {rep.get('spec')}   "
             f"styxx {rep.get('styxx_version_stated')} (stated)   bytes {rep.get('bytes')}"]
    if not rep.get("links"):
        lines.append(f"{SUMMARY_FAILS}: {rep.get('payload', {}).get('problem')}")
        return lines
    repo_rows = {}
    if rep.get("repo"):
        for r in rep["repo"]["rows"]:
            repo_rows.setdefault(r["link"], []).append(r)
    net = rep.get("network")
    for l in rep["links"]:
        lines.append(f"{l['id']:4s} {l['name']}")
        lines.append(f"     offline  {l['offline']:18s} " + ("; ".join(l["problems"]) if l["problems"]
                                                           else "; ".join(l["checked"])))
        online = l["online"]["reading"]
        detail = ""
        if l["id"] in repo_rows:
            rr = repo_rows[l["id"]]
            d = sum(1 for r in rr if r["reading"] == DIFFERS)
            m = sum(1 for r in rr if r["reading"] == MATCHES)
            nc = sum(1 for r in rr if r["reading"] == NOT_CHECKED)
            online = DIFFERS if d else (MATCHES if m and not nc else NOT_CHECKED)
            detail = f"{m} match, {d} differ, {nc} not checked against the clone"
        if l["id"] == "L1" and net:
            parts = [f"#{r['n']} {r['reading']} (clock says {r['status']}" + (f", {r['detail']}" if r.get("detail") else "") + ")"
                     for r in net["anchors"]]
            online = ", ".join(parts) if parts else UNAVAILABLE
        lines.append(f"     online   {online:18s} {detail}")
        for rd in l["online"]["reads"]:
            lines.append(f"              read: {rd}")
        if l["id"] in repo_rows:
            for r in repo_rows[l["id"]]:
                if r["reading"] != MATCHES:
                    lines.append(f"              {r['reading']}: {r['path']} at {str(r['commit'])[:12]} — {r['detail']}")
        lines.append(f"     stated   {l['stated']['reading']:18s} " + (l["stated"]["facts"][0] if l["stated"]["facts"] else ""))
        for f in l["stated"]["facts"][1:]:
            lines.append(f"              {f}")
        lines.append(f"     proves   {l['proves']}")
        lines.append(f"     does not prove   {l['does_not_prove']}")
    for adv in rep.get("advisory") or []:
        lines.append(f"advisory: {adv}")
    pc = rep.get("payload") or {}
    lines.append(f"payload   core_sha256 {'re-derives' if pc.get('core_sha256_ok') else 'DOES NOT re-derive'}; "
                 f"page {'is the one this styxx renders for it' if pc.get('page_ok') else 'DIFFERS from the one this styxx renders' if pc.get('page_ok') is False else 'not compared'}; "
                 f"inlined verifier {'sealed' if pc.get('verifier_js_ok') else 'NOT the sealed one' if pc.get('verifier_js_ok') is False else 'not compared'}; "
                 f"size {'under' if pc.get('size_ok') else 'OVER' if pc.get('size_ok') is False else '?'} the {SIZE_CAP} cap; "
                 f"disclosure hits {len(pc.get('disclosure_hits') or [])}")
    for h in pc.get("disclosure_hits") or []:
        lines.append(f"          {h}")
    if net:
        wm = net.get("sealed_prereg_memos") or {}
        lines.append(f"network   wallet {net['wallet']} (the installed constant; confirm it out of band); "
                     f"sealed-prereg memos listed through the anchor's slot: {wm.get('listed_through_slot', wm.get('error', 'n/a'))}"
                     + (f", distinct digests {len(wm.get('distinct_digests') or [])}" if wm.get("distinct_digests") is not None else ""))
    offline_ok = all(l["offline"] == HOLDS for l in rep["links"]) and bool(pc.get("core_sha256_ok")) and \
        pc.get("page_ok") is not False and pc.get("verifier_js_ok") is not False and pc.get("size_ok") is not False \
        and not pc.get("disclosure_hits")
    n_reads = sum(len(l["online"]["reads"]) for l in rep["links"])
    n_stated = sum(1 for l in rep["links"] if l["stated"]["reading"] == STATED)
    if offline_ok:
        lines.append(f"{SUMMARY_OK}: every carried digest re-derives and every link holds; nothing here is "
                     f"checked against the world. {n_reads} public reads printed; {n_stated} links carry "
                     f"facts stated by the lab.")
    else:
        bad = [l["id"] for l in rep["links"] if l["offline"] == FAILS]
        lines.append(f"{SUMMARY_FAILS}" + (f" at {', '.join(bad)}" if bad else " (payload)") + ".")
    if rep.get("repo"):
        rr = rep["repo"]
        if rr["differs"]:
            lines.append(f"DIFFERS FROM THE REPOSITORY in {rr['differs']} read(s); current charon head "
                         f"{rr.get('head_now')}")
        elif rr["matches"] and not rr["not_checked"]:
            lines.append(f"{SUMMARY_REPO_OK} {', '.join(c[:12] for c in rr['commits'])}: {rr['matches']} public "
                         f"reads match; current charon head {rr.get('head_now')} (printed, not compared)")
        else:
            lines.append(f"REPOSITORY READS: {rr['matches']} match, {rr['not_checked']} not checked; current "
                         f"charon head {rr.get('head_now')}")
    lines.append(f"  {LABEL}")
    lines.append("Nothing above vouches for the run; the re-run command is the embedded runner.")
    return lines


# ------------------------------------------------------------------------------------- page

def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))


def render_html(payload: dict, js: Optional[str] = None) -> str:
    """The page for a payload. Deterministic: the same payload and verifier give the same bytes,
    so layer 2 can re-render and compare. Every ``<`` in the payload is written ``\\u003c``."""
    js = _JS if js is None else js
    rep = check_payload(payload)
    cards = []
    for l in rep["links"]:
        reads = "".join(f"<div class=rd>read: {_esc(r)}</div>" for r in l["online"]["reads"]) or \
            f"<div class=rd>{_esc(NOTHING_ONLINE)}</div>"
        facts = "".join(f"<div class=rd>{_esc(f)}</div>" for f in l["stated"]["facts"]) or \
            f"<div class=rd>{_esc(NOTHING_STATED)}</div>"
        cards.append(
            f'<section class=link id="{_esc(l["id"])}"><h3>{_esc(l["id"])} {_esc(l["name"])}</h3>'
            f'<div class=row><span class=k>offline</span> <span class="st notrun" data-offline="{_esc(l["id"])}">'
            f'NOT RUN</span> <span class=what data-checked="{_esc(l["id"])}">layer 1 has not run</span></div>'
            f'<div class=row><span class=k>online</span> <span class="st online">{_esc(l["online"]["reading"])}</span>'
            f'{reads}</div>'
            f'<div class=row><span class=k>stated</span> <span class="st stated">{_esc(l["stated"]["reading"])}</span>'
            f'{facts}</div>'
            f'<div class=row><span class=k>proves</span> {_esc(l["proves"])}</div>'
            f'<div class=row><span class=k>does not prove</span> {_esc(l["does_not_prove"])}</div></section>')
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True).replace("<", "\\u003c")
    exp = payload.get("experiment") or {}
    ver = payload.get("verifier") or {}
    fill = {"TITLE": _esc(f"research capsule — {exp.get('prereg_name')}"),
            "LABEL": _esc(LABEL), "SPEC": _esc(SPEC), "CARDS": "".join(cards), "PAYLOAD": body,
            "JS": js, "VERSION": _esc(ver.get("styxx_version")), "REPO": _esc(exp.get("repo")),
            "SUMMARY_OK": _esc(SUMMARY_OK), "SUMMARY_FAILS": _esc(SUMMARY_FAILS)}
    return re.sub(r"__([A-Z_]+)__", lambda m: fill[m.group(1)], _TEMPLATE)


_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title>
<style>
 body{background:#000;color:#c8ffc8;font:13px/1.55 ui-monospace,Consolas,Menlo,monospace;margin:0;padding:1.5rem}
 main{max-width:68rem;margin:0 auto} h1{font-size:1.05rem;color:#00ff00;margin:0 0 .3rem}
 h2{font-size:.85rem;color:#00ffff;letter-spacing:.1em;text-transform:uppercase;margin:1.8rem 0 .5rem}
 h3{font-size:.95rem;color:#00ff00;margin:1rem 0 .3rem}
 .label{border:1px solid #0a4;padding:.6rem .8rem;margin:.6rem 0;color:#9f9}
 .summary{padding:.6rem .8rem;margin:.6rem 0;border:1px solid #444;color:#aaa}
 .summary.ok{border-color:#00ff00;color:#00ff00} .summary.bad{border-color:#f33;color:#f66}
 .link{border-top:1px solid #133;padding:.4rem 0}
 .row{margin:.15rem 0 .15rem 0;padding-left:9.5rem;position:relative}
 .k{position:absolute;left:0;color:#6a6;width:9rem}
 .st{font-weight:700} .st.notrun{color:#aaa} .st.holds{color:#00ff00} .st.fails{color:#f33}
 .st.online,.st.stated{color:#00ffff} .what{color:#9c9} .rd{color:#8cc;word-break:break-all}
 pre{white-space:pre-wrap;word-break:break-word;background:#050a05;border:1px solid #133;padding:.8rem;color:#bdb}
 mark{background:#063;color:#dfd} .meta{color:#6a6} details summary{cursor:pointer;color:#8cc}
 .bad{color:#f66}
</style></head><body><main>
<h1>__TITLE__</h1>
<div class=meta>spec __SPEC__ &middot; repository __REPO__ &middot; built with styxx __VERSION__ (stated by the builder) &middot; <span id=created></span></div>
<div class=label><b>What this file is.</b> __LABEL__</div>
<div id=summary class=summary>layer 1 has not run: nothing on this page is checked until the inline verifier runs in this browser (scripts on, no network needed). Layer 2 is <code>python -m styxx.research_capsule verify THIS_FILE</code> at a styxx build at or after the one named above; <code>--repo CLONE</code> turns the public reads below into MATCHES / DIFFERS / NOT-CHECKED and <code>--network</code> asks the chain.</div>
<h2>the links — three readings each</h2>
<div class=meta>offline: HOLDS / FAILS, re-derived here from the carried bytes &middot; online: CHECKABLE-ONLINE, the public read a stranger makes (this page makes none) &middot; stated: STATED-BY-THE-LAB, what nothing checks</div>
__CARDS__
<h2>the anchor</h2><pre id=anchor></pre>
<h2>the preregistration</h2><pre id=prereg></pre>
<h2>the receipt, re-scored</h2><pre id=score></pre><pre id=receipt></pre>
<h2>the RESULT, with its sworn spans</h2><pre id=result></pre><pre id=sworn></pre>
<h2>the ferry-log line</h2><pre id=charon></pre>
<h2>the run note</h2><pre id=runnote></pre>
<h2>the runner</h2><div id=runners></div>
<div class=label>__LABEL__<br>Nothing above vouches for the run; the re-run command is the embedded runner.</div>
<script type="application/json" id="oath-capsule">__PAYLOAD__</script>
<script id="research-verify">__JS__</script>
</main></body></html>
"""

# The browser verifier (layer 1). It re-derives every digest, the memo, the gates re-score, the
# sidecar render, the sworn receipt's own digest and tally, the sworn verdict core over the carried
# snapshot (every span HELD, as _all_held_problem rules), the charon entry_id and the payload core
# from the carried bytes. Where an integer is required (a span offset, a snapshot size) it reads
# the payload through a typed parse that keeps a float literal apart, as Python's json does,
# because JSON.parse writes 102.0 as 102. The sworn core is
# re-derived by styxx/_data/sworn_verify.js, the second implementation of styxx.sworn held to the
# sworn conformance vectors, inlined verbatim; that file resolves no path: span (it has no tree),
# so the page gives it the carried snapshot through a resolver that reads entries as
# styxx.sworn.SnapshotTree does. sworn_verify.js is not edited. It never fetches anything.
_JS_RESEARCH = r"""(function(SW){
"use strict";
var SUMMARY_OK="INTERNALLY CONSISTENT (offline)", SUMMARY_FAILS="FAILS OFFLINE";
function sha256(bytes){
  var K=[0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2];
  var H=[0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19];
  var l=bytes.length, padLen=((l+9+63)>>6)<<6, p=new Uint8Array(padLen); p.set(bytes); p[l]=0x80;
  var hi=Math.floor(l/0x20000000), lo=(l<<3)>>>0;
  p[padLen-8]=(hi>>>24)&255;p[padLen-7]=(hi>>>16)&255;p[padLen-6]=(hi>>>8)&255;p[padLen-5]=hi&255;
  p[padLen-4]=(lo>>>24)&255;p[padLen-3]=(lo>>>16)&255;p[padLen-2]=(lo>>>8)&255;p[padLen-1]=lo&255;
  var w=new Int32Array(64);
  for(var off=0;off<padLen;off+=64){
    for(var i=0;i<16;i++)w[i]=(p[off+4*i]<<24)|(p[off+4*i+1]<<16)|(p[off+4*i+2]<<8)|p[off+4*i+3];
    for(i=16;i<64;i++){var x=w[i-15],y=w[i-2];
      var s0=((x>>>7)|(x<<25))^((x>>>18)|(x<<14))^(x>>>3), s1=((y>>>17)|(y<<15))^((y>>>19)|(y<<13))^(y>>>10);
      w[i]=(w[i-16]+s0+w[i-7]+s1)|0;}
    var a=H[0],b=H[1],c=H[2],d=H[3],e=H[4],f=H[5],g=H[6],h=H[7];
    for(i=0;i<64;i++){var S1=((e>>>6)|(e<<26))^((e>>>11)|(e<<21))^((e>>>25)|(e<<7));
      var ch=(e&f)^(~e&g), t1=(h+S1+ch+K[i]+w[i])|0;
      var S0=((a>>>2)|(a<<30))^((a>>>13)|(a<<19))^((a>>>22)|(a<<10)), mj=(a&b)^(a&c)^(b&c), t2=(S0+mj)|0;
      h=g;g=f;f=e;e=(d+t1)|0;d=c;c=b;b=a;a=(t1+t2)|0;}
    H[0]=(H[0]+a)|0;H[1]=(H[1]+b)|0;H[2]=(H[2]+c)|0;H[3]=(H[3]+d)|0;H[4]=(H[4]+e)|0;H[5]=(H[5]+f)|0;H[6]=(H[6]+g)|0;H[7]=(H[7]+h)|0;}
  var out="";for(i=0;i<8;i++)out+=("00000000"+(H[i]>>>0).toString(16)).slice(-8);return out;}
var TE=new TextEncoder(), TD=new TextDecoder("utf-8",{fatal:false});
function enc(s){return TE.encode(s);} function dec(b){return TD.decode(b);}
function b64(s){var bin=atob(s),out=new Uint8Array(bin.length);for(var i=0;i<bin.length;i++)out[i]=bin.charCodeAt(i);return out;}
function shaText(s){return sha256(enc(s));}
function concat(a,b){var o=new Uint8Array(a.length+b.length);o.set(a);o.set(b,a.length);return o;}
function jcs(x){
  if(x===null||x===true||x===false)return JSON.stringify(x);
  if(typeof x==="number"){if(!isFinite(x))throw new Error("non-finite number");return JSON.stringify(x);}
  if(typeof x==="string")return JSON.stringify(x);
  if(Array.isArray(x))return "["+x.map(jcs).join(",")+"]";
  if(typeof x==="object"){var ks=Object.keys(x).sort();return "{"+ks.map(function(k){return JSON.stringify(k)+":"+jcs(x[k]);}).join(",")+"}";}
  throw new Error("not JCS-serializable");}
// strict by default (duplicate and non-ASCII keys refused); with opts.typed it reads any JSON as
// JSON.parse does (a repeated key's last value wins, into an object with no prototype) and hands
// each number's literal to opts.num, so a 102.0 can be told from a 102 as Python's json tells them
function parseStrict(text,opts){
  var i=0,n=text.length,typed=!!(opts&&opts.typed),num=opts&&opts.num;
  function ws(){while(i<n&&(text[i]===" "||text[i]==="\t"||text[i]==="\n"||text[i]==="\r"))i++;}
  function fail(m){throw new Error((typed?"the payload":"gates block")+" is not valid strict JSON: "+m+" at "+i);}
  function str(){if(text[i]!=='"')fail("expected string");i++;var out="";
    while(i<n){var j=i;while(j<n){var cc=text.charCodeAt(j);if(cc===34||cc===92||cc<32)break;j++;}
      if(j>i){out+=text.slice(i,j);i=j;continue;}
      var c=text[i++];if(c==='"')return out;
      if(c==="\\"){var e=text[i++];if(e==="u"){var h=text.substr(i,4);if(!/^[0-9a-fA-F]{4}$/.test(h))fail("bad escape");out+=String.fromCharCode(parseInt(h,16));i+=4;}
        else{var map={'"':'"',"\\":"\\","/":"/",b:"\b",f:"\f",n:"\n",r:"\r",t:"\t"};if(!(e in map))fail("bad escape");out+=map[e];}}
      else{if(c.charCodeAt(0)<32)fail("control character in string");out+=c;}}
    fail("unterminated string");}
  function val(){ws();var c=text[i];
    if(c==='"')return str();
    if(c==="{"){i++;var o=typed?Object.create(null):{};ws();if(text[i]==="}"){i++;return o;}
      for(;;){ws();var k=str();if(!typed&&!/^[\x00-\x7f]*$/.test(k))fail("non-ASCII key "+JSON.stringify(k));
        if(!typed&&Object.prototype.hasOwnProperty.call(o,k))fail("duplicate key "+JSON.stringify(k));
        ws();if(text[i]!==":")fail("expected :");i++;o[k]=val();ws();
        if(text[i]===","){i++;continue;}if(text[i]==="}"){i++;return o;}fail("expected , or }");}}
    if(c==="["){i++;var a=[];ws();if(text[i]==="]"){i++;return a;}
      for(;;){a.push(val());ws();if(text[i]===","){i++;continue;}if(text[i]==="]"){i++;return a;}fail("expected , or ]");}}
    if(text.substr(i,4)==="true"){i+=4;return true;}if(text.substr(i,5)==="false"){i+=5;return false;}if(text.substr(i,4)==="null"){i+=4;return null;}
    var m=/^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?/.exec(text.substr(i));if(!m)fail("unexpected token");i+=m[0].length;return num?num(m[0]):Number(m[0]);}
  var v=val();ws();if(i!==n)fail("trailing data");return v;}
function selectGates(text){
  var lines=text.split("\n"),spans=[],pos=0;
  for(;;){var s=text.indexOf("<!--",pos);if(s<0)break;var e=text.indexOf("-->",s+4);spans.push([s,e<0?text.length:e+3]);pos=e<0?s+4:e+3;}
  var offsets=[],off=0;for(var i=0;i<lines.length;i++){offsets.push(off);off+=lines[i].length+1;}
  var FENCE=/^([ \t]*)(`{3,}|~{3,})[ \t]*(\S*)[ \t]*$/,CLOSER=/^ {0,3}`{3,}[ \t]*$/,like=[];
  for(i=0;i<lines.length;i++){var m=FENCE.exec(lines[i]);if(!m)continue;var indent=m[1],marker=m[2],info=m[3];
    if(!/^[\x00-\x7f]*$/.test(info))throw new Error("fence info string contains non-ASCII characters");
    var cols=0;for(var j=0;j<indent.length;j++)cols=indent[j]==="\t"?cols+4-cols%4:cols+1;
    if(cols<=3&&info.toLowerCase()==="gates")like.push([i,indent,marker,info]);}
  if(like.length===0)throw new Error("prereg has no gates block");
  if(like.length>1)throw new Error("prereg contains "+like.length+" gates-like fenced blocks");
  var L=like[0];if(L[1]||L[2]!=="```"||L[3]!=="gates")throw new Error("the single gates-like block is not a plain unindented ```gates fence");
  for(j=0;j<spans.length;j++)if(spans[j][0]<=offsets[L[0]]&&offsets[L[0]]<spans[j][1])throw new Error("the gates fence sits inside an HTML comment");
  for(j=L[0]+1;j<lines.length;j++)if(CLOSER.test(lines[j]))return lines.slice(L[0]+1,j).join("\n");
  throw new Error("the gates fence is never closed");}
var OPS={">=":function(a,b){return a>=b;},"<=":function(a,b){return a<=b;},">":function(a,b){return a>b;},"<":function(a,b){return a<b;},"==":function(a,b){return a===b;}};
function resolve(result,dotted){var o=result,ks=dotted.split(".");for(var i=0;i<ks.length;i++){if(o===null||typeof o!=="object"||Array.isArray(o)||!(ks[i] in o))throw new Error("metric "+JSON.stringify(dotted)+" not present in result");o=o[ks[i]];}return o;}
function isNum(v){return typeof v==="number"&&isFinite(v);}
function scoreFrozen(gatesText,result){
  var spec=parseStrict(gatesText);if(spec===null||typeof spec!=="object"||Array.isArray(spec))throw new Error("gates block is not an object");
  ["gates","outcomes","smoke_verdict"].forEach(function(k){if(!(k in spec))throw new Error("gates block missing "+k);});
  var gates=spec.gates,outcomes=spec.outcomes,names=Object.keys(gates||{});
  if(!gates||typeof gates!=="object"||names.length===0)throw new Error("'gates' must be a non-empty object");
  if(!Array.isArray(outcomes)||outcomes.length===0)throw new Error("'outcomes' must be a non-empty list");
  var fired={},power={};
  names.forEach(function(name){var g=gates[name];if(!g||typeof g!=="object")throw new Error("gate "+name+" is not an object");
    var op=OPS[g.op];if(!op)throw new Error("gate "+name+": unknown op");if(typeof g.metric!=="string"||!g.metric)throw new Error("gate "+name+": missing metric");
    var v=resolve(result,g.metric);if(!isNum(v))throw new Error("gate "+name+": metric cannot be compared");
    if(!isNum(g.value))throw new Error("gate "+name+": value is not a finite number");
    if("agg" in g||"over" in g||"excluding" in g){
      if((g.agg!=="min"&&g.agg!=="max")||typeof g.over!=="string"||!g.over)throw new Error("gate "+name+": bad composition declaration");
      var pop=resolve(result,g.over);if(!pop||typeof pop!=="object"||Array.isArray(pop)||Object.keys(pop).length===0)throw new Error("gate "+name+": 'over' is not a non-empty dict");
      var ex={};if("excluding" in g){if(typeof g.excluding!=="string"||!g.excluding)throw new Error("gate "+name+": bad excluding");var el=resolve(result,g.excluding);if(!Array.isArray(el))throw new Error("gate "+name+": excluding is not a list");
        el.forEach(function(e){if(typeof e!=="string"||!(e in pop))throw new Error("gate "+name+": excluding names an unknown member");ex[e]=true;});}
      var vals=[];Object.keys(pop).forEach(function(k){if(ex[k])return;if(!isNum(pop[k]))throw new Error("gate "+name+": member cannot be aggregated");vals.push(pop[k]);});
      if(!vals.length)throw new Error("gate "+name+": every member excluded");
      var rec=g.agg==="min"?Math.min.apply(null,vals):Math.max.apply(null,vals);if(Math.abs(v-rec)>1e-12)throw new Error("gate "+name+": COMPOSITION VIOLATION");}
    fired[name]=!!op(v,g.value);power[name]=g.power_basis;});
  var verdict=null;
  outcomes.forEach(function(row){if(!row||typeof row!=="object"||!row.when||typeof row.when!=="object"||typeof row.verdict!=="string")throw new Error("an outcome row is malformed");
    Object.keys(row.when).forEach(function(k){if(!(k in gates))throw new Error("an outcome row names an undeclared gate "+k);});
    if(verdict===null&&Object.keys(row.when).every(function(k){return fired[k]===row.when[k];}))verdict=row.verdict;});
  if(verdict===null)throw new Error("no outcome row matches — the frozen table is not total");
  var total=null;if(names.length<=12){total=true;for(var bits=0;bits<(1<<names.length)&&total;bits++){var f={};names.forEach(function(nm,i){f[nm]=!!((bits>>i)&1);});
    if(!outcomes.some(function(row){return Object.keys(row.when).every(function(k){return f[k]===row.when[k];});}))total=false;}}
  return {verdict:verdict,gates:fired,total:total,power_basis:power};}
function renderSidecar(side){var text=enc(side.text),events=[];
  side.spans.forEach(function(s){events.push([s.start,enc('<sworn r="'+s.receipt+'" k="'+s.kind+'">')]);events.push([s.end,enc("</sworn>")]);});
  var out=text;for(var i=events.length-1;i>=0;i--){var o=events[i][0],t=events[i][1];var n=new Uint8Array(out.length+t.length);n.set(out.subarray(0,o));n.set(t,o);n.set(out.subarray(o),o+t.length);out=n;}return out;}
function pointer(obj,ptr){if(ptr==="")return obj;if(ptr[0]!=="/")return undefined;var cur=obj,toks=ptr.split("/").slice(1);
  for(var i=0;i<toks.length;i++){var t=toks[i].replace(/~1/g,"/").replace(/~0/g,"~");if(cur===null||typeof cur!=="object")return undefined;
    if(Array.isArray(cur)){if(!/^\d+$/.test(t)||+t>=cur.length)return undefined;cur=cur[+t];}else{if(!(t in cur))return undefined;cur=cur[t];}}return cur;}
function esc(s){return String(s).replace(/[&<>"]/g,function(c){return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c];});}
// research_capsule.receipt_tally_problem; tr is the same receipt from the typed parse, where every
// count, sworn_total and unresolved must be an integer as written (a 2.0 is not a count)
function tallyProblem(r,tr){var spans=r.spans,counts=r.counts;
  if(!Array.isArray(spans)||!counts||typeof counts!=="object"||Array.isArray(counts))return "the sworn receipt carries no spans list or no counts object";
  var t=(tr&&typeof tr==="object")?tr:{},tc=(t.counts&&typeof t.counts==="object")?t.counts:{};
  var notInt=Object.keys(counts).filter(function(k){return !(Object.prototype.hasOwnProperty.call(tc,k)&&intLit(tc[k]));}).sort();
  if(notInt.length)return "its counts hold a value that is not an integer as written ("+notInt.join(", ")+")";
  var ik=["sworn_total","unresolved"];for(var j=0;j<ik.length;j++)if(Object.prototype.hasOwnProperty.call(r,ik[j])&&!intLit(t[ik[j]]))return "its "+ik[j]+" is not an integer as written";
  var tally=Object.create(null);for(var i=0;i<spans.length;i++){var v=spans[i]&&spans[i].verdict;if(typeof v!=="string")return "a span of the sworn receipt carries no verdict";tally[v]=(tally[v]||0)+1;}
  var ck=Object.keys(counts),tk=Object.keys(tally);
  if(tk.some(function(k){return !Object.prototype.hasOwnProperty.call(counts,k);})||ck.some(function(k){return counts[k]!==(tally[k]||0);}))return "its counts "+jcs(counts)+" are not its spans' tally "+jcs(tally);
  var total=0;tk.forEach(function(k){total+=tally[k];});
  if(Object.prototype.hasOwnProperty.call(r,"sworn_total")&&r.sworn_total!==total)return "its sworn_total "+JSON.stringify(r.sworn_total)+" is not its "+total+" spans";
  if(Object.prototype.hasOwnProperty.call(r,"unresolved")&&r.unresolved!==(counts.UNRESOLVED||0))return "its unresolved "+JSON.stringify(r.unresolved)+" is not its UNRESOLVED count";
  var want=r.document_malformed?"SWORN-FAILED":total===0?"UNSWORN":((counts.FAILED||0)===0&&(counts.MALFORMED||0)===0)?"SWORN-HELD":"SWORN-FAILED";
  if(r.document_verdict!==want)return "it says "+JSON.stringify(r.document_verdict)+" and its own spans tally "+want+" "+jcs(tally);
  return null;}
// research_capsule._parse_iso: the format in ASCII digits and a time that exists (no February 30,
// month 13, hour 24, second 60 or year 0); setUTCFullYear, because Date.UTC reads a year 0-99 as 19xx
function isoOk(s){if(typeof s!=="string"||!/^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$/.test(s))return false;
  var y=+s.slice(0,4),mo=+s.slice(5,7),d=+s.slice(8,10),h=+s.slice(11,13),mi=+s.slice(14,16),se=+s.slice(17,19),t=new Date(0);
  t.setUTCFullYear(y,mo-1,d);t.setUTCHours(h,mi,se,0);
  return y>=1&&t.getUTCFullYear()===y&&t.getUTCMonth()===mo-1&&t.getUTCDate()===d&&t.getUTCHours()===h&&t.getUTCMinutes()===mi&&t.getUTCSeconds()===se;}
// research_capsule._public_clock_problem, word for word. PYWS is str.isspace's set, what Python's
// str.strip removes; String.prototype.trim strips another (U+FEFF in, U+001C-U+001F and U+0085 out)
var PYWS=/^[\t\n\x0b\x0c\r\x1c-\x1f \x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]*$/;
function clockProblem(c){if(!c||typeof c!=="object"||Array.isArray(c))return "is not an object";
  var ks=["name","binds","read"];for(var i=0;i<ks.length;i++){var v=c[ks[i]];if(typeof v!=="string"||PYWS.test(v))return "has no non-blank string '"+ks[i]+"'";}
  if(c.stated!==null&&c.stated!==undefined&&!isoOk(c.stated))return "has a stated time that is not an ISO-8601 UTC time that exists (YYYY-MM-DDTHH:MM:SSZ)";
  if(Object.prototype.hasOwnProperty.call(c,"does_not_prove")&&typeof c.does_not_prove!=="string")return "has a does_not_prove that is not text";
  return null;}
// research_capsule._all_held_problem: L4 holds only when every sworn span reads HELD
function allHeldProblem(core){var counts=(core.counts&&typeof core.counts==="object")?core.counts:{},total=0;
  Object.keys(counts).forEach(function(k){if(typeof counts[k]==="number"&&Number.isInteger(counts[k]))total+=counts[k];});
  var held=counts.HELD||0,verdict=core.document_verdict;
  if(verdict==="SWORN-HELD"&&total>0&&held===total)return null;
  if(total===0)return "the RESULT swears nothing ("+verdict+"): no span is read against a leaf";
  var parts=Object.keys(counts).filter(function(k){return k!=="HELD"&&counts[k];}).map(function(k){return k+" "+counts[k];});
  parts.push("the document reads "+verdict);
  return held+" of "+total+" sworn spans read HELD ("+parts.join("; ")+"): a FAILED span prints what its leaf does not hold, an UNRESOLVED span was read against no leaf, and a capsule carries a RESULT whose every span holds";}
function portableCore(r){var o={};Object.keys(r).forEach(function(k){if(["digest","timestamp","coverage","coverage_sha256","verifier"].indexOf(k)<0)o[k]=r[k];});return o;}
function truthy(x){if(x===null||x===undefined||x===false||x===0||x==="")return false;if(Array.isArray(x))return x.length>0;if(typeof x==="object")return Object.keys(x).length>0;return true;}
// styxx.sworn.verify(sidecar=..., tree=SnapshotTree(...)) in this browser: the sidecar's own checks
// (load_sidecar's shape, the text digest, the re-scan reproducing the spans), then the core
function swornCoreHere(side,tree,snapCommit){
  var KEYS=["commit","document","manifest","spans","spec","text"];
  if(!side||typeof side!=="object"||side.spec!==SW.spec)throw new Error("the sidecar's spec is not "+SW.spec);
  if(jcs(Object.keys(side).sort())!==jcs(KEYS))throw new Error("the sidecar's keys are not "+KEYS.join(", "));
  if(side.commit!==null&&!(typeof side.commit==="string"&&/^([0-9a-f]{40}|[0-9a-f]{64})$/.test(side.commit)))throw new Error("the sidecar's commit is not a full lowercase hex object id or null");
  if(typeof side.text!=="string"||!side.document||typeof side.document.name!=="string"||typeof side.document.sha256!=="string")throw new Error("the sidecar carries no text or no document name and sha256");
  var text=enc(side.text);if(sha256(text)!==side.document.sha256)throw new Error("the sidecar text does not hash to document.sha256");
  if(!Array.isArray(side.spans))throw new Error("the sidecar spans are not a list");
  var last=0;side.spans.forEach(function(s,i){
    if(!s||typeof s!=="object"||jcs(Object.keys(s).sort())!==jcs(["end","kind","receipt","start"]))throw new Error("span "+i+" is not start/end/receipt/kind");
    if(!Number.isInteger(s.start)||!Number.isInteger(s.end)||!(0<=s.start&&s.start<s.end&&s.end<=text.length)||s.start<last)throw new Error("span "+i+" offsets are not an ordered non-empty range in the text");
    [s.start,s.end].forEach(function(o){if(o<text.length&&(text[o]&0xC0)===0x80)throw new Error("span "+i+" offset "+o+" is not on a UTF-8 character boundary");});
    if(typeof s.receipt!=="string"||typeof s.kind!=="string")throw new Error("span "+i+" receipt or kind is not a string");
    last=s.end;});
  function isObj(x){return !!x&&typeof x==="object"&&!Array.isArray(x);}
  var m=side.manifest;
  if(!isObj(m)||SW.manifestSpecs.indexOf(m.spec)<0||!isObj(m.receipts)||Object.keys(m.receipts).some(function(k){return !isObj(m.receipts[k]);})
     ||(Object.prototype.hasOwnProperty.call(m,"authored_sha256")&&(!Array.isArray(m.authored_sha256)||m.authored_sha256.some(function(x){return typeof x!=="string";}))))
    throw new Error("the sidecar manifest is not a "+SW.manifestSpecs.join(" / ")+" object");
  var raw=renderSidecar(side),sc=SW.api.scan(raw);
  if(sc.document_malformed)throw new Error("the rendered sidecar has no canonical form ("+sc.document_malformed.reason+")");
  if(!sc.lexical_ok||sc.declarations.some(function(d){return d.malformed==="empty_span";}))throw new Error("the rendered sidecar carries a declaration the sidecar form cannot represent");
  function cmp(a,b){for(var i=0;i<4;i++){if(a[i]<b[i])return -1;if(a[i]>b[i])return 1;}return 0;}
  var seen=sc.declarations.map(function(d){return [d.start,d.end,d.receipt,d.kind];}).sort(cmp);
  var want=side.spans.map(function(s){return [s.start,s.end,s.receipt,s.kind];});
  if(jcs(seen)!==jcs(want))throw new Error("re-scanning the rendered sidecar does not reproduce its spans");
  var emb=side.manifest||{},man=(truthy(emb.receipts)||truthy(emb.authored_sha256))?SW.api.jsonPlain(JSON.stringify(emb)):null;
  return SW.overSnapshot(tree,snapCommit,side.commit,function(){return SW.api.swornVerify(raw,man,{name:side.document.name,commit:side.commit});});}
var PTEXT=document.getElementById("oath-capsule").textContent,P=JSON.parse(PTEXT);
// the same payload with each number kept as Python's json reads it: a literal with a fraction or an
// exponent is a float there and FLOAT here, never an integer; read only where an integer is required
var FLOAT={},PT=null;try{PT=parseStrict(PTEXT,{typed:true,num:function(lit){return /[.eE]/.test(lit)?FLOAT:Number(lit);}});}catch(e){PT=null;}
function typedAt(path){var o=PT;for(var i=0;i<path.length;i++){if(o===null||typeof o!=="object"||!Object.prototype.hasOwnProperty.call(o,path[i]))return undefined;o=o[path[i]];}return o;}
function intLit(x){return typeof x==="number"&&Number.isInteger(x);}
var A=P.artifacts||{},X=P.experiment||{},S=Object.assign({receipt:0,metrics:"/metrics",verdict:"/verdict",gates:"/gates",smoke:"/config/smoke"},X.score||{});
var links={};["L0","L1","L1b","L2","L3","L4","L5","L6"].forEach(function(k){links[k]={ok:[],bad:[]};});
function chk(L,cond,text,why){if(cond)links[L].ok.push(text);else links[L].bad.push(text+(why?" — "+why:""));return !!cond;}
function art(L,entry,what){if(!entry||typeof entry!=="object"){links[L].bad.push(what+" is not an embedded artifact");return null;}
  var data;try{data=b64(entry.b64);}catch(e){links[L].bad.push(what+" bytes are not base64");return null;}
  chk(L,sha256(data)===entry.sha256,what+" ("+entry.name+") hashes to its recorded sha256","the carried bytes are not the bytes the digest names");return data;}
document.getElementById("created").textContent="created "+P.created+" (outside the digest)";
var corePayload={};Object.keys(P).forEach(function(k){if(k!=="created"&&k!=="core_sha256")corePayload[k]=P[k];});
var coreOk=false;try{coreOk=shaText(jcs(corePayload))===P.core_sha256;}catch(e){coreOk=false;}
var jsEl=document.getElementById("research-verify"),jsOk=!!(jsEl&&P.verifier_js&&shaText(jsEl.textContent.replace(/\r\n/g,"\n"))===P.verifier_js.sha256);
// L0
var pr=A.prereg||{},prBytes=art("L0",pr,"the PREREG"),gatesText=null,prText="";
if(prBytes){prText=dec(prBytes);try{gatesText=selectGates(prText);links.L0.ok.push("the PREREG freezes exactly one plain ```gates block");
    chk("L0",shaText(gatesText)===pr.gates_sha256,"the gates block hashes to the recorded gates_sha256","the block in the text is not the block the payload names");}
  catch(e){links.L0.bad.push("the PREREG does not freeze exactly one gates block — "+e.message);}}
// L1
var anchors=Array.isArray(A.anchors)?A.anchors:[],KINDS=["sworn-receipt","sealed-prereg","sealed-canaries","charon-head"];
var sealed=anchors.filter(function(a){return a&&a.kind==="sealed-prereg";});
chk("L1",sealed.length>0,"the capsule carries a sealed-prereg anchor line","no sealed-prereg line is carried (UNANCHORED)");
anchors.forEach(function(a,i){var obj=!!a&&typeof a==="object"&&!Array.isArray(a);
  var okShape=obj&&typeof a.kind==="string"&&KINDS.indexOf(a.kind)>=0&&typeof a.digest==="string"&&/^[0-9a-f]{64}$/.test(a.digest)&&typeof a.tx==="string"&&a.tx;
  var memoOk=okShape&&(!("memo" in a)||a.memo==="styxx "+a.kind+" "+a.digest);
  var ts=typedAt(["artifacts","anchors",i,"slot"]),slotOk=!obj||!("slot" in a)||(intLit(ts)&&ts>=0);
  var timeOk=!obj||!("block_time" in a)||isoOk(a.block_time);
  chk("L1",okShape&&memoOk&&slotOk&&timeOk,"anchor #"+(obj?a.n:undefined)+" ("+(obj?a.kind:undefined)+") memo = kind + digest, slot and block_time well-formed","MALFORMED_LINE");
  if(okShape&&memoOk&&a.kind==="sealed-prereg")chk("L1",a.digest===pr.sha256,"anchor #"+a.n+" digest is the PREREG's sha256 (L0)","the sealed digest is not the carried PREREG's");});
// L1b
var clocks=Array.isArray(A.public_clocks)?A.public_clocks:[];
clocks.forEach(function(c,i){var why=clockProblem(c);chk("L1b",why===null,"public clock "+(i+1)+" is well-formed",why||"");});
if(!clocks.length)links.L1b.ok.push("no public clock is listed (nothing to check offline)");
// L2
var snap=A.snapshot||{},entries=snap.entries||{},receipts=Array.isArray(A.receipts)?A.receipts:[];
// experiment.score.receipt as research_capsule reads it: an integer as written, else named (never
// an index of 1.0, "1" or true, which JS reads and Python does not)
var TSC=typedAt(["experiment","score"]),runIdx=(TSC&&typeof TSC==="object"&&!Array.isArray(TSC)&&Object.prototype.hasOwnProperty.call(TSC,"receipt"))?TSC.receipt:0;
if(!intLit(runIdx)){links.L2.bad.push("experiment.score.receipt is "+JSON.stringify(S.receipt)+", which is not an integer as written");runIdx=null;}
var rr=(runIdx!==null&&runIdx>=0&&runIdx<receipts.length)?receipts[runIdx]:null,runReceipt=(rr&&typeof rr==="object"&&!Array.isArray(rr))?rr:null,runObj=null;
chk("L2",!!runReceipt,"the run's receipt is carried","experiment.score.receipt names no carried receipt");
receipts.forEach(function(r,i){var data=art("L2",r,"receipt "+i);if(!data)return;var obj;try{obj=parseStrict(dec(data));}catch(e){links.L2.bad.push("receipt "+r.name+" is not strict JSON — "+e.message);return;}
  var ent=entries[r.path];
  if(r===runReceipt){runObj=obj;var smoke=pointer(obj,S.smoke);chk("L2",!smoke,"the run's receipt is not a smoke run ("+S.smoke+" is not true)","a smoke receipt scores INVALID by type and licenses nothing");
    if(obj&&typeof obj.prereg==="string")chk("L2",obj.prereg===pr.name,"the receipt names the carried PREREG","it names "+obj.prereg);else links.L2.ok.push("the receipt names no PREREG (nothing to compare)");
    chk("L2",!!ent&&ent.sha256===r.sha256,"the run's receipt is the receipt the RESULT's spans read (same sha256 in the snapshot at the sworn commit)","the snapshot holds other bytes at that path, or none");}
  else if(ent)chk("L2",ent.sha256===r.sha256,"receipt "+r.name+" is the one the snapshot holds","the snapshot holds other bytes at that path");});
(Array.isArray(A.runners)?A.runners:[]).forEach(function(r,i){art("L2",r,"runner "+i);});
if(A.run_note)art("L2",A.run_note,"the run note");
// L3
var scoreOut=null;
if(gatesText===null)links.L3.bad.push("no gates block could be read from the PREREG (L0)");
else if(runObj===null)links.L3.bad.push("the run's receipt could not be read (L2)");
else{var metrics=pointer(runObj,S.metrics),recV=pointer(runObj,S.verdict),recG=pointer(runObj,S.gates);
  if(!metrics||typeof metrics!=="object")links.L3.bad.push("the receipt has no metrics object at "+S.metrics);
  else{try{scoreOut=scoreFrozen(gatesText,metrics);chk("L3",scoreOut.verdict===recV,"re-scored verdict "+scoreOut.verdict+" equals the recorded "+recV,"the recorded verdict is not the frozen table's reading of the recorded numbers");
      chk("L3",recG&&typeof recG==="object"&&jcs(scoreOut.gates)===jcs(recG),"re-scored gates equal the recorded gates map","recorded "+JSON.stringify(recG)+", re-scored "+JSON.stringify(scoreOut.gates));
      chk("L3",scoreOut.total!==false,"the frozen outcome table is total (every assignment of the gates reaches a row)","an assignment reaches no row");}
    catch(e){links.L3.bad.push("re-scoring refused — "+e.message);}}}
// L4
var res=A.result||{},side=A.sidecar,srec=A.sworn_receipt,resBytes=art("L4",res,"the RESULT");
var snapTree={};Object.keys(entries).forEach(function(p){var e=entries[p];var d;try{d=b64(e.b64);}catch(x){d=null;}if(!d||sha256(d)!==e.sha256){links.L4.bad.push("snapshot entry "+p+" does not decode to the bytes its sha256 names");return;}
  var sz=typedAt(["artifacts","snapshot","entries",p,"size"]);
  if(!intLit(sz)||sz!==d.length){links.L4.bad.push("snapshot entry "+p+" size "+JSON.stringify(e.size)+" is not the integer byte count "+d.length);return;}
  snapTree[p]={mode:e.mode,size:sz,bytes:d};});
if(resBytes&&side&&srec){var doc=srec.document||{};
  chk("L4",sha256(resBytes)===doc.inline_sha256,"the RESULT bytes hash to the sworn receipt's document.inline_sha256","the carried RESULT is not the document the receipt was issued over");
  var tallyWhy=tallyProblem(srec,typedAt(["artifacts","sworn_receipt"]));
  chk("L4",tallyWhy===null,"the sworn receipt's document_verdict and counts are its own spans' tally",tallyWhy||"");
  chk("L4",side.commit===snap.commit,"the sidecar's commit is the snapshot's ("+String(snap.commit).slice(0,12)+")","sidecar names "+side.commit);
  var tsp=typedAt(["artifacts","sidecar","spans"]),offBad=[];
  (Array.isArray(side.spans)?side.spans:[]).forEach(function(s,i){var t=Array.isArray(tsp)?tsp[i]:undefined;
    if(s&&typeof s==="object"&&!Array.isArray(s)&&!(t&&typeof t==="object"&&intLit(t.start)&&intLit(t.end)))offBad.push(i);});
  chk("L4",offBad.length===0,"the sidecar's span offsets are integers as written","span(s) "+JSON.stringify(offBad)+" carry an offset that is not an integer");
  try{chk("L4",sha256(renderSidecar(side))===sha256(resBytes),"the sidecar renders to the RESULT bytes byte for byte","it renders other bytes");}catch(e){links.L4.bad.push("the sidecar could not be rendered — "+e.message);}
  (side.spans||[]).forEach(function(s){var r=String(s.receipt||"");if(r.slice(0,5)==="path:"){var target=r.slice(5).split("#")[0];chk("L4",target in entries,"span target "+target+" is in the snapshot","no snapshot entry; the span could only be UNRESOLVED");}});
  var body={};Object.keys(srec).forEach(function(k){if(["digest","timestamp","coverage","coverage_sha256"].indexOf(k)<0)body[k]=srec[k];});
  var dOk=false;try{dOk=shaText(jcs(body))===srec.digest;}catch(e){dOk=false;}
  chk("L4",dOk,"the sworn receipt's digest re-derives over its core (sworn R9: coverage and timestamp outside)","the receipt's digest is not its core's");
  (srec.spans||[]).forEach(function(s){var r=String(s.receipt||"");if(r.slice(0,5)==="path:"&&s.verdict!=="UNRESOLVED"){var target=r.slice(5).split("#")[0],e=entries[target];
    chk("L4",!!e&&e.sha256===s.resolved_sha256,"span @"+s.at+" resolved_sha256 is the snapshot entry's for "+target,"the receipt resolved other bytes");}});
  try{var swCore=swornCoreHere(side,snapTree,snap.commit),swWant=shaText(jcs(portableCore(srec)));
    chk("L4",SW.api.coreDigest(SW.api.coreAs(swCore,srec.schema))===swWant,"the sworn core re-derives here over the carried snapshot: "+swCore.document_verdict+" "+JSON.stringify(swCore.counts)+" (each span adjudicated again by the inlined sworn_verify.js)",
        "the live core differs (live "+swCore.document_verdict+" "+JSON.stringify(swCore.counts)+" vs carried "+srec.document_verdict+" "+JSON.stringify(srec.counts)+")");
    var heldWhy=allHeldProblem(swCore);
    chk("L4",heldWhy===null,"every sworn span reads HELD against the leaf it names ("+(swCore.counts.HELD||0)+" of "+swCore.sworn_total+")",heldWhy||"");}
  catch(e){links.L4.bad.push("the sworn core could not be re-derived here — "+e.message);}
  links.L4.ok.push("not here: whether the sworn build the receipt names is the installed one (layer 2 prints INSTRUMENT SKEW)");}
else{if(!side)links.L4.bad.push("no sidecar is carried");if(!srec)links.L4.bad.push("no sworn receipt is carried");}
// L5
var ch=A.charon;
if(!ch)links.L5.bad.push("no charon entry is carried");
else{var entry=ch.entry||{},core={};Object.keys(entry).forEach(function(k){if(["entry_id","timestamp","note"].indexOf(k)<0)core[k]=entry[k];});
  var want=null;try{want=shaText(jcs(core));}catch(e){want=null;}
  chk("L5",entry.schema==="styxx.charon/entry/v1","entry schema is styxx.charon/entry/v1","it is "+entry.schema);
  chk("L5",want!==null&&entry.entry_id===want,"entry_id = sha256(jcs(core)) with timestamp and note outside","the line's entry_id does not re-derive from its core");
  chk("L5",entry.seq===ch.seq&&ch.entry_id===entry.entry_id&&ch.head_at_seq===entry.entry_id,"the carried seq "+ch.seq+", entry_id and head_at_seq are the line's","the pin disagrees with the line");
  chk("L5",entry.kind==="sworn","the line is a sworn line","kind is "+entry.kind);
  var subj=entry.subject||{},at=entry.at||{},sdoc=(srec&&srec.document)||{};
  chk("L5",subj.sha256===sdoc.inline_sha256,"subject.sha256 is the sworn RESULT's inline_sha256","the line is about another document");
  chk("L5",subj.path===res.path,"subject.path is "+res.path,"it is "+subj.path);
  chk("L5",at.commit===(side||{}).commit,"at.commit is the sidecar's commit","it is "+at.commit);
  var shas=(entry.receipts&&Array.isArray(entry.receipts.sha256))?entry.receipts.sha256:[];
  chk("L5",!!runReceipt&&shas.indexOf(runReceipt.sha256)>=0,"the run's receipt is in the line's resolved receipt set (n="+(entry.receipts||{}).n+")","the line's receipt set does not include it");
  chk("L5",entry.verdict===(srec||{}).document_verdict,"the line's verdict "+entry.verdict+" is the sworn receipt's","the receipt says "+(srec||{}).document_verdict);
  chk("L5",entry.reproduced===true,"the line says the committed receipt reproduced at ingest","reproduced is "+entry.reproduced);
  if(entry.seq===1){var hd=null;try{hd=ch.header?sha256(concat(enc("styxx.charon/log/v1\n"),enc(jcs(ch.header)))):null;}catch(e){hd=null;}
    chk("L5",hd!==null&&entry.prev===hd,"prev names the carried header's domain-separated digest","it does not");}
  else chk("L5",typeof entry.prev==="string"&&/^[0-9a-f]{64}$/.test(entry.prev),"prev is a 64-hex entry_id (the line before, read online)","it is "+entry.prev);}
// L6
var labelOnPage=document.body.textContent.indexOf(P.label)>=0&&typeof P.label==="string"&&P.label.indexOf("a forger controlling the whole file passes every offline link")>=0;
chk("L6",labelOnPage,"the page prints the label verbatim","the label was edited or dropped");
// paint
var allOk=coreOk&&jsOk;
Object.keys(links).forEach(function(k){var L=links[k],fails=L.bad.length>0;allOk=allOk&&!fails;
  var st=document.querySelector('[data-offline="'+k+'"]'),wh=document.querySelector('[data-checked="'+k+'"]');
  if(st){st.textContent=fails?"FAILS":"HOLDS";st.className="st "+(fails?"fails":"holds");}
  if(wh)wh.textContent=(fails?L.bad:L.ok).join("; ");});
var sum=document.getElementById("summary"),failed=Object.keys(links).filter(function(k){return links[k].bad.length;});
sum.className="summary "+(allOk?"ok":"bad");
sum.textContent=allOk?(SUMMARY_OK+": every carried digest re-derives in this browser, the sworn spans re-derive over the carried snapshot, and every link holds; nothing here is checked against the world. Layer 2 (python -m styxx.research_capsule verify THIS_FILE) re-derives every link again under the installed styxx and names the sworn build; --repo CLONE and --network are the public reads.")
  :(SUMMARY_FAILS+(failed.length?" at "+failed.join(", "):"")+(coreOk?"":"; the payload core_sha256 does not re-derive")+(jsOk?"":"; the inlined verifier is not the sealed one")+". A forger controlling this whole file controls this layer too.");
// artifacts
function pre(id,text){var el=document.getElementById(id);if(el)el.textContent=text;}
pre("anchor",anchors.map(function(a){return JSON.stringify(a);}).join("\n")+(A.clock_check?"\n\nthe builder saw the chain say (stated): "+JSON.stringify(A.clock_check,null,1):"\n\n(no clock check recorded at build)")+(clocks.length?"\n\npublic clocks (each a third-party read):\n"+clocks.map(function(c){return "- "+c.name+": "+c.read+(c.stated?"  read "+c.stated:"")+"\n  binds "+c.binds+"; does not prove: "+c.does_not_prove;}).join("\n"):""));
pre("prereg",prText);
pre("score",scoreOut?("re-scored: "+scoreOut.verdict+"  gates "+JSON.stringify(scoreOut.gates)+"\nrecorded:  "+(runObj?pointer(runObj,S.verdict):"?")+"  gates "+JSON.stringify(runObj?pointer(runObj,S.gates):null)+"\noutcome table total: "+scoreOut.total+"\n"+Object.keys(scoreOut.power_basis).map(function(g){return g+" power_basis: "+scoreOut.power_basis[g];}).join("\n")):"(not re-scored)");
pre("receipt",receipts.map(function(r){var d;try{d=dec(b64(r.b64));}catch(e){d="(undecodable)";}return "# "+r.path+"  sha256 "+r.sha256+"\n"+d;}).join("\n\n"));
if(resBytes){var el=document.getElementById("result");if(el){var h=esc(dec(resBytes));h=h.replace(/&lt;sworn r=&quot;([^&]*)&quot; k=&quot;([^&]*)&quot;&gt;([\s\S]*?)&lt;\/sworn&gt;/g,function(m,r,k,inner){return '<mark title="'+r+' ('+k+')">'+inner+"</mark>";});el.innerHTML=h;}}
pre("sworn",srec?("sworn receipt: "+srec.document_verdict+" "+JSON.stringify(srec.counts)+"  commit "+srec.commit+"\ncoverage sentence_share "+((srec.coverage||{}).sentence_share)+" (a floor; the narrative outside the spans is the author's)\ndigest "+srec.digest+"\ncertifies: "+srec.certifies+(srec.schema==="styxx.sworn.verdict-receipt/v1"?" (the /v1 sentence, withdrawn: papers/sworn/NOTE_certifies_correction_2026_10_24.md)":"")):"(none)");
pre("charon",ch?("seq "+ch.seq+"  entry_id "+ch.entry_id+"  head_at_seq "+ch.head_at_seq+"  log "+ch.log_path+"\n"+JSON.stringify(ch.entry,null,1)):"(none)");
pre("runnote",A.run_note?("# "+A.run_note.path+"  sha256 "+A.run_note.sha256+"  (stated by the lab)\n"+dec(b64(A.run_note.b64))):"(no run note carried)");
var rn=document.getElementById("runners");if(rn)rn.innerHTML=(A.runners||[]).map(function(r){var d;try{d=dec(b64(r.b64));}catch(e){d="(undecodable)";}return "<details><summary>"+esc(r.path)+"  sha256 "+esc(r.sha256)+"</summary><pre>"+esc(d)+"</pre></details>";}).join("")||"(no runner carried)";
})(SW);
"""

# The snapshot resolver, appended inside sworn_verify.js's own scope so the file stays byte for byte
# what the package ships: `resolve` is reassigned (a function binding), every other function is the
# file's. It reads entries as styxx.sworn.SnapshotTree.blob/find_sha256 and builds what
# styxx.sworn._resolve builds on its tree branch (complete, the committed-object provenance note,
# the authored-receipt refusal, then the fragment).
_JS_SNAPSHOT = r"""
;var _snap=null,_snapCommit=null,_treeCommit=null,_noTree=resolve,_BLOB_MODES=["100644","100755"],_MAX_BLOB=__MAX_BLOB__;
function _snapReady(){if(_treeCommit===null||_treeCommit===undefined)return "no_commit";
  if(typeof _treeCommit!=="string"||!/^([0-9a-f]{40}|[0-9a-f]{64})$/.test(_treeCommit))return "commit_absent";
  if(_treeCommit!==_snapCommit)return "commit_absent";return null;}
function _snapBlob(path){var why=_snapReady();if(why)return [null,why];
  if(!Object.prototype.hasOwnProperty.call(_snap,path))return [null,"path_absent"];
  var e=_snap[path];if(_BLOB_MODES.indexOf(e.mode)<0)return [null,"not_a_blob"];
  var data=e.bytes,size=e.size;if((size===null||size===undefined)&&data)size=data.length;
  if(!data||(size!==null&&size!==undefined&&size>_MAX_BLOB))return [null,"receipt_too_large"];
  return [data,"ok"];}
function _snapFind(digest){var why=_snapReady();if(why)return [null,why];
  var ks=Object.keys(_snap).sort();for(var i=0;i<ks.length;i++){var e=_snap[ks[i]];
    if(_BLOB_MODES.indexOf(e.mode)>=0&&e.bytes&&sha256Bytes(e.bytes)===digest)return [e.bytes,"ok"];}
  return [null,"prereg_not_in_tree"];}
resolve=function(parsed,kind,manifest){
  if(parsed.form==="rn"||_snap===null)return _noTree(parsed,kind,manifest);
  var r=parsed.form==="path"?_snapBlob(parsed.target):_snapFind(parsed.target);
  if(r[0]===null)return {status:"unresolved",reason:r[1]};
  var sha=sha256Bytes(r[0]);
  if(manifest!==null&&manifest.authored_sha256.indexOf(sha)>=0)return {status:"malformed",reason:"receipt_author_minted"};
  return finishResolve({status:"ok",reason:null,bytes:r[0],sha256:sha,complete:true,leaf:null,has_leaf:false,slice:null,
    provenance:{form:parsed.form,note:"committed object at "+(_treeCommit||"?")+"; authorship unchecked"}},parsed);};
return {api:_api,spec:SPEC,manifestSpecs:MANIFEST_SPECS.slice(),
  overSnapshot:function(entries,snapCommit,treeCommit,fn){_snap=entries;_snapCommit=snapCommit;_treeCommit=treeCommit;
    try{return fn();}finally{_snap=null;_snapCommit=null;_treeCommit=null;}}};
"""


def _compose_js() -> str:
    """The inlined verifier: sworn_verify.js verbatim in its own scope with the snapshot resolver,
    then the research page's checks. Line endings are LF whatever the checkout wrote."""
    sworn_js = (Path(__file__).parent / "_data" / "sworn_verify.js").read_bytes().decode("utf-8")
    snapshot = _JS_SNAPSHOT.replace("__MAX_BLOB__", str(int(_sworn.MAX_RECEIPT_BYTES)))
    return ("(function(){\nvar SW=(function(){\n" + sworn_js + "\n" + snapshot + "})();\n" + _JS_RESEARCH
            + "})();\n").replace("\r\n", "\n")


_JS = _compose_js()
assert "</script" not in _JS.lower()  # the verifier is inlined in a script element


# --------------------------------------------------------------------------------------- CLI

def _read_json_file(p: Optional[str]):
    return json.loads(Path(p).read_text(encoding="utf-8")) if p else None


def _blob_at(repo: Path, commit: str, rel: str, what: str) -> bytes:
    data, why = _sworn.GitTree(repo, commit).blob(rel)
    if data is None:
        _refuse("read", f"{what} {rel} at {commit[:12]}: {why} (the build reads the repository at named "
                        f"commits, never the working tree)")
    return data


def _snapshot_from_git(repo: Path, commit: str, sidecar: dict) -> dict:
    tree = _sworn.GitTree(repo, commit)
    entries: Dict[str, dict] = {}
    for s in sidecar.get("spans") or []:
        parsed, _m = _sworn._parse_receipt(s.get("receipt"))
        if not parsed or parsed["form"] != "path" or parsed["target"] in entries:
            continue
        rel = parsed["target"]
        rc, out, _err = tree._git("ls-tree", "-z", "-l", "--full-tree", commit, "--", rel)
        meta = out.split(b"\0")[0].split(b"\t")[0].split() if rc == 0 and out.strip(b"\0") else []
        mode = meta[0].decode("ascii") if meta else "100644"
        data, why = tree.blob(rel)
        if data is None:
            _refuse("snapshot", f"span target {rel} at {commit[:12]}: {why}")
        entries[rel] = {"mode": mode, "size": len(data), "sha256": _sha256(data), "bytes": data}
    return {"commit": commit, "entries": entries}


def _charon_from_git(repo: Path, commit: str, log_rel: str, result_rel: str) -> dict:
    raw = _blob_at(repo, commit, log_rel, "the charon log")
    header, entry = None, None
    for i, line in enumerate(raw.decode("utf-8", "replace").split("\n")):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if i == 0 and isinstance(obj, dict) and obj.get("schema") == _charon.LOG_SCHEMA:
            header = obj
            continue
        if isinstance(obj, dict) and obj.get("kind") == "sworn" and \
                (obj.get("subject") or {}).get("path") == result_rel:
            entry = obj
    if entry is None:
        _refuse("charon", f"no sworn line for {result_rel} in {log_rel} at {commit[:12]}")
    return {"header": header, "entry": entry, "log_path": log_rel}


def _cli_build(a) -> int:
    repo = Path(a.repo)
    at = a.at
    if not _COMMIT.fullmatch(at) or not _COMMIT.fullmatch(a.prereg_commit):
        _refuse("commit", "--at and --prereg-commit must be full lowercase hex object ids")
    prereg_bytes = _blob_at(repo, a.prereg_commit, a.prereg, "the PREREG")
    anchors_raw = _blob_at(repo, at, a.anchors, "the anchors file")
    prereg_sha = _sha256(prereg_bytes)
    wanted = set(a.anchor_n or [])
    anchors = []
    for line in anchors_raw.decode("utf-8", "replace").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict) and ((obj.get("kind") == "sealed-prereg" and obj.get("digest") == prereg_sha)
                                      or obj.get("n") in wanted):
            anchors.append(obj)
    result_rel = a.result
    if not result_rel.endswith(".md"):
        _refuse("result", "--result must be the RESULT .md (its .sworn.json and .sworn-receipt.json sit beside it)")
    stem = result_rel[:-3]
    sidecar = json.loads(_blob_at(repo, at, stem + ".sworn.json", "the sidecar").decode("utf-8"))
    sworn_receipt = json.loads(_blob_at(repo, at, stem + ".sworn-receipt.json", "the sworn receipt").decode("utf-8"))
    if not _COMMIT.fullmatch(str(sidecar.get("commit"))):
        _refuse("sidecar", "the sidecar names no commit; a research capsule needs path: spans at a commit")
    inputs = {
        "experiment": {"prereg_name": a.prereg.rsplit("/", 1)[-1], "repo": a.public_repo, "at_commit": at,
                       "score": {"receipt": 0, "metrics": a.metrics_pointer, "verdict": a.verdict_pointer,
                                 "gates": a.gates_pointer, "smoke": a.smoke_pointer}},
        "prereg": {"path": a.prereg, "bytes": prereg_bytes, "blob_commit": a.prereg_commit},
        "anchors": anchors, "anchors_path": a.anchors,
        "clock_check": _read_json_file(a.clock_check),
        "receipts": [{"path": r, "bytes": _blob_at(repo, at, r, "a receipt")} for r in a.receipt],
        "runners": [{"path": r, "bytes": _blob_at(repo, at, r, "a runner")} for r in (a.runner or [])],
        "run_note": {"path": a.run_note, "bytes": _blob_at(repo, at, a.run_note, "the run note")} if a.run_note else None,
        "result": {"path": result_rel, "bytes": _blob_at(repo, at, result_rel, "the RESULT")},
        "sidecar": sidecar, "sworn_receipt": sworn_receipt,
        "snapshot": _snapshot_from_git(repo, sidecar["commit"], sidecar),
        "charon": _charon_from_git(repo, at, a.charon_log, result_rel) if a.charon_log else None,
        "public_clocks": _read_json_file(a.public_clocks) or [],
        "stated": _read_json_file(a.stated) or {},
    }
    if not anchors and not a.allow_unanchored:
        _refuse("unanchored", f"no sealed-prereg line for {prereg_sha[:12]}… in {a.anchors} at {at[:12]}; "
                              f"pass --allow-unanchored to mint a capsule whose L1 reads FAILS (UNANCHORED)")
    out = build(inputs, Path(a.out), allow_unanchored=a.allow_unanchored)
    rep = verify(out)
    for line in datasheet(rep):
        _say(line)
    _say(f"research capsule built -> {out}")
    return 0


def _cli_verify(a, table_only: bool = False) -> int:
    rep = verify(Path(a.capsule), repo=Path(a.repo) if getattr(a, "repo", None) else None,
                 network=bool(getattr(a, "network", False)))
    for line in datasheet(rep):
        _say(line)
    if getattr(a, "json", None):
        Path(a.json).write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if rep["ok"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="styxx.research_capsule",
                                 description="the research capsule: one file carries the chain for one "
                                             "preregistered experiment; it does not vouch for the run")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="compose from the repository at named commits; refuses by name")
    b.add_argument("--repo", required=True)
    b.add_argument("--at", required=True, help="the commit holding the RESULT trio, receipts, runners, log")
    b.add_argument("--prereg", required=True, help="repository-relative path of the PREREG")
    b.add_argument("--prereg-commit", required=True, help="the commit whose blob the seal digests")
    b.add_argument("--anchors", default="papers/charon/anchors.jsonl")
    b.add_argument("--anchor-n", type=int, action="append", help="also carry anchor line n (e.g. a charon-head)")
    b.add_argument("--clock-check", default=None, help="JSON of styxx.clock.check_line as the builder saw it")
    b.add_argument("--receipt", action="append", required=True, help="the run's receipt, then every other "
                                                                      "receipt a gate reads from")
    b.add_argument("--runner", action="append")
    b.add_argument("--run-note", default=None)
    b.add_argument("--result", required=True, help="the sworn RESULT .md")
    b.add_argument("--charon-log", default="papers/charon/charon.log.jsonl")
    b.add_argument("--public-clocks", default=None, help="JSON list of {name, binds, read, stated, does_not_prove}")
    b.add_argument("--stated", default=None, help="JSON object of scalar facts the lab states (run_ended_utc, ...)")
    b.add_argument("--public-repo", default="github.com/fathom-lab/styxx")
    b.add_argument("--metrics-pointer", default="/metrics")
    b.add_argument("--verdict-pointer", default="/verdict")
    b.add_argument("--gates-pointer", default="/gates")
    b.add_argument("--smoke-pointer", default="/config/smoke")
    b.add_argument("--allow-unanchored", action="store_true")
    b.add_argument("--out", required=True)
    b.set_defaults(func=_cli_build)
    v = sub.add_parser("verify", help="layer 2: offline always; --repo and --network are the public reads")
    v.add_argument("capsule")
    v.add_argument("--repo", default=None, help="a clone: turns the public reads into MATCHES/DIFFERS/NOT-CHECKED")
    v.add_argument("--network", action="store_true", help="ask the chain through styxx.clock")
    v.add_argument("--json", default=None, help="write the report here")
    v.set_defaults(func=_cli_verify)
    d = sub.add_parser("datasheet", help="the offline table alone")
    d.add_argument("capsule")
    d.set_defaults(func=lambda a: _cli_verify(a, table_only=True))
    a = ap.parse_args(argv)
    try:
        return a.func(a)
    except SystemExit as e:
        if isinstance(e.code, str):
            _say(e.code)
            return 2
        raise


if __name__ == "__main__":
    sys.exit(main())
