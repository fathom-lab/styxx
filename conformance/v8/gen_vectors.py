# -*- coding: utf-8 -*-
"""Generate the styxx.v8 conformance set: run the sources under the recorder, fold, replay, write.

Built to `styxx/v8/INTERFACES_layer2.md` section 10.

    python conformance/v8/gen_vectors.py            # regenerate in place; refuses a moved core
    python conformance/v8/gen_vectors.py --check    # regenerate in memory; exit 1 if the set moved
    python conformance/v8/gen_vectors.py --replay   # replay the committed set (see replay.py)

The generator refuses to write in three situations, and each refusal is a finding about the
implementation rather than a reason to rewrite the set:

* **A failing source.** The vectors are the outcomes the tests chose; recording them from a run
  where a test failed would pin an outcome nobody vouched for.
* **A moved core.** A vector already in the committed set whose `expect` differs from what this
  run produced means `styxx.v8` changed its answer on an input the set already addresses. The
  generator prints the id, both outcomes and the sources, and writes nothing. A receipt is history:
  the fix is a commit that says what changed and why, never a silent regeneration.
* **A vector that will not replay.** Every vector is replayed through `replay.py` before anything
  is written, so a set that ships is a set that reproduces on the machine that made it.

Vectors the run no longer produces are classified rather than counted, because "dropped" hides two
different events. Every dropped vector is replayed against the tree that dropped it:

* it still reproduces -> **input churn**. The sources stopped making that call; the address went
  away with the call. This is an ordinary retirement and needs no permission.
* it does not reproduce -> **a behaviour change**, wearing a drop's clothes. The tree answers
  differently on inputs the set had pinned; the address changed only because the repair that moved
  the behaviour also touched the fixture. This is a moved core and is refused like one.
* it cannot be replayed at all -> **undiagnosed**, and it is refused rather than assumed benign.

## Retiring a moved core, with a reason

A behaviour change can be deliberate. When it is, the operator resolves it in the tool by naming
the ids -- never a flag that retires whatever moved:

    python conformance/v8/gen_vectors.py \
        --retire <id> --reason "subject.environment became required (ENV-ABSENT)" \
        --retire <id> --reason "body.runs became required on a noise plan (A-NORUNS)"

`--retire` and `--reason` are positional pairs: an id with no reason is refused, a reason with no
id is refused, an id that did not move in this run is refused, and an id that could not be
diagnosed at all is refused. Each retirement is written into `index.retired.with_reason` beside the
old outcome, the new one, the sources that produced it and the operator's reason, and that ledger
is carried forward by later runs -- a retirement is a record, not a switch. Ordinary retirements
sit in `retired.input_churn` in the same block, so the two are never confused for one another when
the set is diffed.

## Where the ledger lives, and why it moved

The ledger used to sit in `index.provenance`, which is written **after** `set_sha256` is computed.
That put the record of every retirement outside the only digest in the file: deleting all six
`with_reason` rows and all 149 `input_churn` rows left `set_sha256` unchanged, so a set with its
whole history removed still verified. The ledger is `index.retired` now, inside the digested core,
and a deleted row moves `set_sha256`.

The rest of `provenance` stays outside the digest and the reason is not the same reason. Those
entries -- the file hashes of the implementation, the tooling and the sources, and the per-family
replay counts -- describe **the tree that produced the set**, not the set. A comment added to
`floor.py` changes them without changing a single vector, and a set identity that moved on that
would make `--check` a test of the working tree. The ledger is not in that category: it is content,
it is history about these exact addresses, and nothing else in the repository witnesses it.

The cost is stated rather than hidden. `set_sha256` is no longer a function of the sources and
`styxx.v8` alone; it is a function of those **and of the ledger the committed set carries forward**,
so generating this set from scratch in an empty directory produces a different digest than
regenerating it here. That is the correct reading -- a set that has retired six answers is not the
same artifact as one that has retired none -- but it means `--check` compares a set against its own
history as well as against its sources, and a reader who wants the source-only identity must strip
`retired` themselves.

What this does **not** do is make the ledger unforgeable. A digest inside a file cannot protect that
file from someone editing the file: delete a row, recompute `set_sha256` over the edited core, and
the index is self-consistent again. Two things outside the index detect that, and they are the whole
of the protection: `mutation_coverage.json` pins the `set_sha256` it measured, and
`tests/test_v8_conformance.py` fails when the two disagree; and the previous bytes are in git. The
in-file digest catches the careless edit. Only the commit history catches the careful one.

Nothing here has a clock, so two runs on two machines with the same committed ledger produce the
same bytes and `--check` is a real comparison rather than a timestamp.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from styxx.v8.consts import CHANNELS, EXIT  # noqa: E402
from styxx.v8.floor import CHANNEL_VERDICTS  # noqa: E402
from styxx.v8.jcs import canonical_bytes  # noqa: E402

from conformance.v8 import (  # noqa: E402
    ENTRYPOINTS,
    FAMILIES,
    REASON_KINDS,
    RECORDER_OUT,
    SCHEMA,
    SOURCES,
)
from conformance.v8 import replay as R  # noqa: E402

CONTRACT = "styxx/v8/INTERFACES_layer2.md section 10"

#: Modules the set is a function of. Their digests are provenance, outside `set_sha256`.
IMPLEMENTATION = (
    "styxx/v8/cert.py",
    "styxx/v8/distances.py",
    "styxx/v8/floor.py",
    "styxx/v8/jcs.py",
    "styxx/v8/keys.py",
    "styxx/v8/log.py",
    "styxx/v8/merkle.py",
    "styxx/v8/verify.py",
)

TOOLING = ("conformance/v8/__init__.py", "conformance/v8/recorder.py", "conformance/v8/replay.py")


def _sha_file(path: Path) -> str:
    """Content identity modulo newlines: this box has core.autocrlf on."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _dump(obj: Any) -> bytes:
    """The set's file format: sorted keys, one-space indent, ASCII, LF, one trailing newline."""
    text = json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False)
    return (text + "\n").encode("utf-8")


def _canonical(value: Any) -> Any:
    """`value` as it survives RFC 8785. Storing the canonical form is what makes a vector's
    stored inputs re-derive its own id; `-1.0` becomes `-1` here and stays that way."""
    return json.loads(canonical_bytes(value).decode("utf-8"))


# --------------------------------------------------------------------------- recording


def record(sources: Tuple[str, ...], out: Path) -> List[dict]:
    """Run the sources under the recorder and return the records, or raise on a failing source."""
    env = dict(os.environ)
    env[RECORDER_OUT] = str(out)
    env["PYTHONIOENCODING"] = "utf-8"
    argv = [sys.executable, "-m", "pytest", *sources, "-q", "-p", "no:cacheprovider",
            "-p", "conformance.v8.recorder"]
    print("recording: %s" % " ".join(sources), flush=True)
    proc = subprocess.run(argv, cwd=str(ROOT), env=env, capture_output=True, text=True)
    tail = (proc.stdout or "").strip().splitlines()[-1:] or ["(no output)"]
    print("  %s" % tail[0], flush=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout or "")
        sys.stderr.write(proc.stderr or "")
        raise SystemExit(
            "REFUSED: the sources do not pass (pytest exited %d); the set records the outcomes "
            "the tests chose, and a failing run has none to record" % proc.returncode
        )
    records = []
    with open(out, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# --------------------------------------------------------------------------- folding


class Moved(Exception):
    """A vector already in the set now carries a different outcome."""


def fold(records: List[dict]) -> Tuple[Dict[str, dict], Dict[str, str], List[dict], List[dict]]:
    """Records -> (vectors by id, blob store, skipped calls, unvectored sources).

    One call is one vector; the same call recorded from several tests is one vector naming
    several sources. Two records with the same address and different outcomes are a defect in the
    input model -- the record is not carrying something that changes the answer -- and the fold
    refuses rather than picking one.
    """
    vectors: Dict[str, dict] = {}
    blobs: Dict[str, str] = {}
    skipped: List[dict] = []
    unvectored_sources: List[dict] = []
    for rec in records:
        if "skip" in rec:
            row = {"source": rec.get("source"), "entrypoint": rec.get("entrypoint"),
                   "reason": rec.get("skip")}
            (unvectored_sources if rec.get("entrypoint") == "*" else skipped).append(row)
            continue
        family = rec["family"]
        inputs = _canonical(rec["inputs"])
        expect = _canonical(rec["expect"])
        vid = R.vector_id(family, inputs)
        existing = vectors.get(vid)
        if existing is None:
            vectors[vid] = {
                "id": vid,
                "family": family,
                "entrypoint": rec["entrypoint"],
                "inputs": inputs,
                "expect": expect,
                "sources": [rec["source"]],
                "_blobs": dict(rec["blobs"]),
            }
        else:
            if existing["entrypoint"] != rec["entrypoint"]:
                raise SystemExit(
                    "REFUSED: address %s is shared by %s and %s; the family and inputs do not "
                    "identify the entrypoint" % (vid, existing["entrypoint"], rec["entrypoint"])
                )
            if canonical_bytes(existing["expect"]) != canonical_bytes(expect):
                raise SystemExit(
                    "REFUSED: address %s carries two outcomes.\n"
                    "  %s -> %s\n  %s -> %s\n"
                    "The record is not carrying something the entrypoint reads; fix the recorder's "
                    "input model, do not pick a winner."
                    % (vid, existing["sources"][0], json.dumps(existing["expect"], sort_keys=True),
                       rec["source"], json.dumps(expect, sort_keys=True))
                )
            if rec["source"] not in existing["sources"]:
                existing["sources"].append(rec["source"])
            existing["_blobs"].update(rec["blobs"])
        blobs.update(rec["blobs"])
    for vector in vectors.values():
        vector["sources"] = sorted(vector["sources"])
    return vectors, blobs, skipped, unvectored_sources


def reachable_blobs(vectors: Dict[str, dict], blobs: Dict[str, str]) -> Dict[str, str]:
    """The blobs the surviving vectors actually name. A blob nothing references is dropped."""
    keep: Dict[str, str] = {}
    for vector in vectors.values():
        # The replay's own account of what the vector names, so the store is exactly what a
        # consumer resolves -- never what the recorder happened to collect along the way.
        for sha in R.blob_refs(vector):
            if sha not in blobs:
                raise SystemExit("REFUSED: vector %s names a blob the run did not store: %s"
                                 % (vector["id"], sha))
            keep[sha] = blobs[sha]
    for sha, payload in keep.items():
        raw = base64.b64decode(payload, validate=True)
        if hashlib.sha256(raw).hexdigest() != sha:
            raise SystemExit("REFUSED: blob %s does not hash to its key" % sha)
    return dict(sorted(keep.items()))


# --------------------------------------------------------------------------- the honest remainder


#: Why a miss is a miss. A number in a list says nothing; a gap that is unreachable by
#: construction and a gap the sources simply never drive are different findings, and the set is
#: the wrong place to blur them. A miss with no note here is an unexplained gap, and says so.
MISS_NOTES = {
    "exit_code:1": "drift. A --diff runs no confirmation, so section 5.2 cannot reach drift from "
                   "this entrypoint; verify.diff asserts it (GATED S5-02, recommendation A).",
    "exit_code:5": "unavailable. Produced when a runner refuses, and verify.ref is not a recorded "
                   "entrypoint of this set.",
    "exit_code:4": "invalid. REACHABLE from verify.diff and reached by no recorded call: every "
                   "exit-4 test in the sources drives verify.ref instead. A real gap.",
    "reason_kind:number": "cert.check's NaN/infinity/past-2**53 reasons. The sources reach them; "
                          "the record cannot carry the cert, because no canonical bytes exist for "
                          "it. See unvectored.skipped.",
    "reason_kind:id-uncomputable": "same cause: the cert whose id cannot be computed is the cert "
                                   "whose bytes the record cannot carry. See unvectored.skipped.",
    "reason_kind:not-an-object": "cert.check on something that is not a JSON object. The sources "
                                 "reach it with a str; the record carries certs, not strings. "
                                 "See unvectored.skipped.",
}


def unvectored(vectors: Dict[str, dict]) -> dict:
    """What the set does not reach: named in full, never inferred from a count."""
    reasons: set = set()
    verdicts: set = set()
    channels: set = set()
    codes: set = set()
    for vector in vectors.values():
        expect = vector["expect"]
        if vector["entrypoint"] == "cert.check":
            reasons.update(expect.get("reason_kinds") or [])
        if vector["entrypoint"] == "floor.decide" and expect.get("outcome") == "verdict":
            verdicts.add(expect["verdict"])
        if vector["entrypoint"] == "floor.pairwise" and expect.get("block") is not None:
            channels.add(vector["inputs"]["channel"])
        if vector["entrypoint"] == "verify.diff" and expect.get("outcome") == "outcome":
            codes.add(expect["exit_code"])
    covered = Counter(v["entrypoint"] for v in vectors.values())
    out = {
        "entrypoints": sorted(e for e in ENTRYPOINTS if not covered.get(e)),
        "reason_kinds": sorted(k for k in REASON_KINDS if k not in reasons),
        "channel_verdicts": sorted(v for v in CHANNEL_VERDICTS if v not in verdicts),
        "channels_with_a_floor": sorted(c for c in CHANNELS if c not in channels),
        "exit_codes": sorted(set(EXIT.values()) - codes),
    }
    notes = {}
    for kind, missing in (("reason_kind", out["reason_kinds"]),
                          ("exit_code", out["exit_codes"]),
                          ("channel_verdict", out["channel_verdicts"]),
                          ("entrypoint", out["entrypoints"])):
        for item in missing:
            key = "%s:%s" % (kind, item)
            notes[key] = MISS_NOTES.get(key, "no note: an unexplained gap")
    out["notes"] = notes
    return out


# --------------------------------------------------------------------------- writing


def assemble(records: List[dict]) -> dict:
    """Records -> everything the index is made of except the retirement ledger. Replays.

    The fold and the replay happen here, once. `finalize` turns the result into an index, and the
    split exists because the ledger is inside `set_sha256` now: the ledger is not known until this
    run's vectors have been compared against the committed ones, and the digest cannot be computed
    before the ledger without being patched afterwards -- which is the defect this repair closes.
    """
    vectors, all_blobs, skipped, unvectored_sources = fold(records)
    blobs = reachable_blobs(vectors, all_blobs)

    by_family: Dict[str, List[dict]] = {family: [] for family in FAMILIES}
    for vector in sorted(vectors.values(), key=lambda v: v["id"]):
        family = vector["family"]
        if family not in by_family:
            raise SystemExit("REFUSED: unknown family %r" % family)
        by_family[family].append({k: v for k, v in vector.items() if not k.startswith("_")})

    counts, failures = R.replay_set(by_family, blobs)
    if failures:
        for family, vid, why in failures[:20]:
            print("  WILL NOT REPLAY %s %s: %s" % (family, vid, why), file=sys.stderr)
        raise SystemExit(
            "REFUSED: %d of %d vectors do not reproduce on the machine that recorded them"
            % (len(failures), len(vectors))
        )
    return {
        "vectors": vectors,
        "blobs": blobs,
        "by_family": by_family,
        "counts": counts,
        "skipped": skipped,
        "unvectored_sources": unvectored_sources,
    }


def finalize(parts: dict, retired: Optional[dict] = None) -> Tuple[dict, Dict[str, bytes]]:
    """`assemble`'s parts plus a ledger -> (index, {relative path: bytes}).

    `retired` is the retirement ledger and it goes into `index.retired`, INSIDE `set_sha256`:
    deleting a retirement moves the digest. See the module docstring for why the rest of
    `provenance` stays outside it and what that does and does not protect.
    """
    vectors = parts["vectors"]
    blobs = parts["blobs"]
    by_family = parts["by_family"]
    counts = parts["counts"]
    skipped = parts["skipped"]
    unvectored_sources = parts["unvectored_sources"]

    files: Dict[str, bytes] = {}
    families_index: Dict[str, dict] = {}
    for family in FAMILIES:
        rel = "vectors/%s.json" % family
        data = _dump(by_family[family])
        files[rel] = data
        families_index[family] = {
            "file": rel,
            "count": len(by_family[family]),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    blobs_bytes = _dump(blobs)
    files["blobs.json"] = blobs_bytes

    entrypoints = Counter(v["entrypoint"] for v in vectors.values())
    index = {
        "schema": SCHEMA,
        "contract": CONTRACT,
        "address": "id = sha256(canonical_bytes({\"family\": family, \"inputs\": inputs}))",
        "families": families_index,
        "blobs": {
            "file": "blobs.json",
            "count": len(blobs),
            "sha256": hashlib.sha256(blobs_bytes).hexdigest(),
            "encoding": "base64 of the value's RFC 8785 canonical bytes, keyed by sha256",
        },
        "entrypoints": {
            name: {"family": ENTRYPOINTS[name], "vectors": entrypoints.get(name, 0)}
            for name in sorted(ENTRYPOINTS)
        },
        "sources": list(SOURCES),
        "vectors": len(vectors),
        # Inside the digest on purpose: a history the digest does not cover can be deleted
        # without moving the digest, and this one was.
        "retired": retired if retired is not None else empty_ledger(),
        "unvectored": dict(
            unvectored(vectors),
            sources=sorted(unvectored_sources, key=lambda r: (r["source"] or "")),
            skipped=sorted(skipped, key=lambda r: (r["source"] or "", r["entrypoint"] or "",
                                                   r["reason"] or "")),
        ),
        "reading": (
            "agreement on these vectors makes an implementation agree with styxx.v8 on the inputs "
            "five test files chose. It does not make it correct, and it does not mean the contract "
            "is covered: mutation_coverage.json measures what a disagreement would have to be for "
            "this set to see it, and its miss list is part of the answer."
        ),
    }
    digest_over = {k: v for k, v in index.items()}
    index["set_sha256"] = hashlib.sha256(canonical_bytes(digest_over)).hexdigest()
    index["provenance"] = {
        "note": "outside set_sha256: this block describes the TREE that produced the set, not the "
                "set. A comment added to floor.py moves these hashes without moving a vector, and "
                "a set identity that moved on that would make --check a test of the working tree. "
                "The retirement ledger used to sit here and does not any more: it is index.retired "
                "now, inside the digest, because a history the digest does not cover was deleted "
                "without moving the digest.",
        "implementation": {p: _sha_file(ROOT / p) for p in IMPLEMENTATION},
        "tooling": {p: _sha_file(ROOT / p) for p in TOOLING},
        "sources": {p: _sha_file(ROOT / p) for p in SOURCES},
        "replay": {f: counts.get(f, {}) for f in FAMILIES},
    }
    files["index.json"] = _dump(index)
    return index, files


def build(records: List[dict], retired: Optional[dict] = None
          ) -> Tuple[dict, Dict[str, bytes], Dict[str, dict]]:
    """Records -> (index, {relative path: bytes}, vectors by id). `assemble` then `finalize`."""
    parts = assemble(records)
    index, files = finalize(parts, retired)
    return index, files, parts["vectors"]


def committed(directory: Path) -> Dict[str, dict]:
    """The vectors already on disk, by id. Empty when the set has never been generated."""
    out: Dict[str, dict] = {}
    for family, family_vectors in R.load_vectors(directory).items():
        for vector in family_vectors:
            out[vector["id"]] = vector
    return out


def moved_cores(old: Dict[str, dict], new: Dict[str, dict]) -> List[dict]:
    """Ids in both sets whose outcome changed: the same address, a different answer."""
    rows: List[dict] = []
    for vid, vector in sorted(new.items()):
        was = old.get(vid)
        if was is None:
            continue
        if canonical_bytes(was.get("expect")) != canonical_bytes(vector["expect"]):
            rows.append({
                "id": vid,
                "kind": "moved-core",
                "entrypoint": vector["entrypoint"],
                "was": was.get("expect"),
                "now": vector["expect"],
                "sources": list(vector["sources"]),
            })
    return rows


def classify_drops(
    old: Dict[str, dict], new: Dict[str, dict], blobs: Dict[str, str]
) -> List[dict]:
    """Every id the run no longer produces, replayed against the tree that dropped it.

    "Dropped" is one word for two events. A vector whose call a test simply stopped making still
    reproduces -- the sources moved, `styxx.v8` did not -- and that is input churn. A vector that
    no longer reproduces is a behaviour change that happens to have taken its address with it,
    because the repair that moved the answer also touched the fixture the address is computed
    from; printing that as a drop would lose exactly the finding a moved core is refused for.
    """
    rows: List[dict] = []
    for vid in sorted(set(old) - set(new)):
        vector = old[vid]
        row = {
            "id": vid,
            "entrypoint": vector.get("entrypoint"),
            "was": vector.get("expect"),
            "sources": list(vector.get("sources") or []),
        }
        try:
            got = R.replay(vector, blobs)
        except R.ReplayError as exc:
            row["kind"] = "undiagnosed"
            row["now"] = None
            row["detail"] = "the dropped vector cannot be replayed: %s" % exc
        else:
            if R.agrees(vector.get("expect"), got):
                row["kind"] = "input-churn"
                row["now"] = got
                row["detail"] = "it still reproduces: the sources stopped making this call"
            else:
                row["kind"] = "behaviour-change"
                row["now"] = got
                row["detail"] = R.difference(vector.get("expect"), got)
        rows.append(row)
    return rows


#: What each drop kind means, printed once per run above the ids so the terminal says which
#: event happened rather than one word for all three.
DROP_KINDS = {
    "input-churn": "they still reproduce; the sources stopped making the call",
    "behaviour-change": "they no longer reproduce; styxx.v8 answers differently and the address "
                        "left with the fixture",
    "undiagnosed": "they cannot be replayed at all, so neither reading is available",
}


def format_move(row: dict) -> str:
    """One moved address, both outcomes and the tests that reach it."""
    return (
        "  %s (%s, %s)\n    was      %s\n    this run %s\n    sources  %s"
        % (row["id"], row["entrypoint"], row["kind"],
           json.dumps(row["was"], sort_keys=True)[:300],
           json.dumps(row["now"], sort_keys=True)[:300],
           ", ".join(row["sources"]) or "(none)")
    )


def compare_to_committed(old: Dict[str, dict], new: Dict[str, dict]) -> List[str]:
    """The moved cores as printable lines. Kept as the narrow question -- did an address that
    survived change its answer -- for callers that do not have the blob store to classify drops."""
    return [format_move(row) for row in moved_cores(old, new)]


# --------------------------------------------------------------------------- retirement


def retirement_plan(retire: List[str], reason: List[str]) -> Dict[str, str]:
    """`--retire`/`--reason` as `{id: reason}`, or a refusal.

    The pairs are positional and both halves are required. There is deliberately no flag that
    means "retire whatever moved": an operator who cannot name the address has not looked at it.
    """
    retire = list(retire or [])
    reason = list(reason or [])
    if len(retire) != len(reason):
        raise SystemExit(
            "REFUSED: %d --retire and %d --reason. They are positional pairs, and a retirement "
            "with no reason is the override this path exists to avoid."
            % (len(retire), len(reason))
        )
    plan: Dict[str, str] = {}
    for vid, why in zip(retire, reason):
        vid = vid.strip().lower()
        if not vid:
            raise SystemExit("REFUSED: --retire needs the id of a vector, not an empty string")
        if not (why or "").strip():
            raise SystemExit("REFUSED: --retire %s carries an empty reason; say what changed "
                             "and why the change is deliberate" % vid)
        if vid in plan:
            raise SystemExit("REFUSED: --retire %s given twice, with two reasons; one address "
                             "moved once in this run" % vid)
        plan[vid] = why.strip()
    return plan


#: The drop kinds a `--retire` may name. `undiagnosed` is deliberately absent: see
#: `apply_retirements`.
RETIRABLE_KINDS = ("moved-core", "behaviour-change")


def apply_retirements(
    plan: Dict[str, str], moved: List[dict], drops: List[dict]
) -> Tuple[List[dict], List[str], List[str], List[str]]:
    """`(rows to record, moved-and-unnamed, reasons to refuse a name, undiagnosed drops)`.

    Every address whose answer changed must be named, and every named address must have changed.
    A drop that still reproduces is an ordinary retirement and cannot be laundered through this
    path: naming one is refused, so the reason recorded beside a retired core is always a reason
    about `styxx.v8` and never about a test that moved.

    An **undiagnosed** drop -- one that cannot be replayed at all, so neither the answer it used to
    give nor the answer this tree gives is available -- is not retirable, and that is the repair
    this signature exists for. It used to sit in `changed` alongside a measured behaviour change,
    which meant `--retire <id> --reason "..."` wrote it into the ledger with a reason about a move
    nobody had measured: the one drop that must not be quietly retired was the one the path
    accepted most easily. It comes back separately now. Naming it is refused, and `main` refuses
    the run on it, because the repair is to make it replayable -- or to find the defect in the
    vector -- not to write a sentence beside it.
    """
    undiagnosed = {row["id"]: row for row in drops if row["kind"] == "undiagnosed"}
    changed = {row["id"]: row for row in moved if row["kind"] in RETIRABLE_KINDS}
    for row in drops:
        if row["kind"] in RETIRABLE_KINDS:
            changed[row["id"]] = row
    churn = {row["id"] for row in drops if row["kind"] == "input-churn"}

    refusals: List[str] = []
    for vid in sorted(plan):
        if vid in changed:
            continue
        if vid in undiagnosed:
            refusals.append(
                "  %s cannot be replayed at all, so neither the answer it used to give nor the "
                "answer this tree gives is available, and there is nothing for a reason to be "
                "about. An undiagnosed drop is neither an ordinary retirement nor a deliberate "
                "one; make it replayable, or find the defect in the vector, and run again." % vid
            )
        elif vid in churn:
            refusals.append(
                "  %s is an ordinary retirement: it still reproduces against this tree, so the "
                "sources stopped making the call and nothing about styxx.v8 moved. It needs no "
                "reason and cannot carry one." % vid
            )
        else:
            refusals.append(
                "  %s did not move in this run: no committed vector at that address changed its "
                "answer or disappeared." % vid
            )
    unnamed = sorted(set(changed) - set(plan))

    recorded = [
        dict(changed[vid], reason=plan[vid], detail=changed[vid].get("detail"))
        for vid in sorted(plan)
        if vid in changed
    ]
    return recorded, unnamed, refusals, sorted(undiagnosed)


LEDGER_NOTE = (
    "Both halves are ledgers: every run carries the previous one forward, because a run that "
    "recorded only its own diff would erase the record on the very next regeneration. "
    "with_reason: an address whose ANSWER changed and whose old answer an operator retired on "
    "purpose, naming the id and giving the reason; the row carries the old outcome, the new one, "
    "the sources that produced it and the reason. input_churn: an address the sources stopped "
    "producing whose old vector still reproduced against the tree that dropped it -- an ordinary "
    "retirement, no permission asked and none given, so the row carries the address alone. An id "
    "the sources produce again leaves input_churn; nothing leaves with_reason. The two are never "
    "merged: one word for both would hide a behaviour change behind a rewritten test. "
    "A drop that cannot be replayed at all reaches NEITHER half: it is refused, not retired, "
    "because a reason written beside a move nobody measured is the quietest retirement there is. "
    "This block sits inside set_sha256, so deleting a row from it moves the identity of the set."
)

#: What `input_churn` costs, decided rather than left to grow and be noticed later.
#:
#: It is a ledger, not a leak, and the distinction is arithmetic rather than taste. The half is
#: built as a dict keyed by vector id, so an address enters it at most once however many times it
#: is dropped and re-added, and it is bounded by the number of DISTINCT addresses the sources have
#: ever produced -- not by the number of runs and not by elapsed time. At 149 rows of
#: `{id, entrypoint}` it is about 18KB against a 1.4MB set. Pruning it would buy that back and
#: cost the only question it answers, which is "was this address ever in the set" -- the question
#: a reader asks when a vector they remember is not there, and the question `classify_drops`
#: would otherwise be unfalsifiable about. The honest limit, stated because it bounds the value:
#: a churn row carries the address and the entrypoint and nothing else, so it can confirm that an
#: address was retired and cannot say what it tested. It is a receipt for a deletion, not a copy
#: of what was deleted.
CHURN_IS_BOUNDED = (
    "input_churn is keyed by id, so an address enters at most once and the half is bounded by the "
    "distinct addresses the sources have ever produced, not by the number of runs. It is kept in "
    "full: a retirement nobody had to ask permission for is still a retirement."
)

#: The fields of a with-reason row, so a carried-forward row and a new one have the same shape.
LEDGER_FIELDS = ("id", "kind", "entrypoint", "was", "now", "sources", "reason", "detail")


def _ledger_key(row: Mapping) -> str:
    return hashlib.sha256(canonical_bytes(
        {"id": row.get("id"), "was": row.get("was"), "now": row.get("now")}
    )).hexdigest()


def empty_ledger() -> dict:
    """The ledger of a set that has retired nothing. Not a stand-in for one that went missing."""
    return {"note": LEDGER_NOTE, "growth": CHURN_IS_BOUNDED, "with_reason": [], "input_churn": [],
            "counts": {"with_reason": 0, "input_churn": 0}}


def retirement_ledger(previous: Optional[dict], recorded: List[dict], drops: List[dict],
                      current: Optional[Iterable[str]] = None) -> dict:
    """`index.retired`: both ledgers, this run's rows appended to the committed set's.

    `current` is the addresses this run produced. An id in it is not retired, whatever an earlier
    run recorded, so a call a test brings back leaves `input_churn` rather than sitting in the
    ledger contradicting the vectors beside it.

    An `undiagnosed` row is refused on the way in, wherever it came from. `apply_retirements` will
    not produce one, so reaching this refusal means a committed ledger already carried one and is
    being carried forward -- which is the state a hand-edited index would be in.
    """
    previous = previous or {}
    live = set(current or ())

    rows: List[dict] = []
    seen: set = set()
    for row in list(previous.get("with_reason") or []) + list(recorded):
        clean = {field: row.get(field) for field in LEDGER_FIELDS}
        if clean.get("kind") not in RETIRABLE_KINDS:
            raise SystemExit(
                "REFUSED: the ledger carries a retirement of kind %r at %s. Only %s can be "
                "retired: a drop nobody could diagnose has no old answer and no new one, so the "
                "reason beside it is about nothing."
                % (clean.get("kind"), clean.get("id"), " and ".join(RETIRABLE_KINDS))
            )
        key = _ledger_key(clean)
        if key in seen:
            continue
        seen.add(key)
        rows.append(clean)

    churn: Dict[str, dict] = {}
    for row in list(previous.get("input_churn") or []) + [
        r for r in drops if r["kind"] == "input-churn"
    ]:
        vid = row.get("id")
        if vid in live:
            continue
        churn[vid] = {"id": vid, "entrypoint": row.get("entrypoint")}
    churn_rows = [churn[vid] for vid in sorted(churn)]

    return {
        "note": LEDGER_NOTE,
        "growth": CHURN_IS_BOUNDED,
        "with_reason": rows,
        "input_churn": churn_rows,
        "counts": {"with_reason": len(rows), "input_churn": len(churn_rows)},
    }


def _set_files(directory: Path) -> List[str]:
    """The pieces of a generated set present in `directory`, other than the index."""
    present = [p for p in [directory / "blobs.json"] if p.exists()]
    present += sorted((directory / "vectors").glob("*.json"))
    return [p.name for p in present]


def read_ledger(directory: Path) -> Tuple[dict, List[str]]:
    """`(the committed retirement ledger, reasons this directory must not be regenerated)`.

    The repair for the second attack on the ledger. `main` used to compute the ledger only when
    the directory already held vectors, and to read it out of the index only then, so deleting
    `index.json` -- or the `vectors/` directory -- made the next run fall through to the default
    empty ledger and write it, with no refusal and nothing in the output to say a history had just
    been dropped. The ledger is read here on its own, before anything is recorded, and a directory
    that still holds pieces of a set but has lost the index that carried its history is refused.

    The one case that is not a refusal is a directory with nothing in it: that is a new set, and a
    new set has retired nothing. The distinction is the presence of the vectors, not the absence
    of the index, because an attacker who deletes the whole directory has deleted the set as well
    and there is nothing left here to make a false claim about. What catches that is the published
    `set_sha256`, and only that.
    """
    index_path = directory / "index.json"
    remains = _set_files(directory)
    if not index_path.exists():
        if remains:
            return {}, [
                "  %s has no index.json and still holds %s. The retirement ledger lived in that "
                "index; it cannot be recovered from the vectors, and regenerating here would "
                "write an empty one over the record of every retired address. Restore index.json "
                "from git, or empty the directory to declare this a new set."
                % (directory, ", ".join(remains))
            ]
        return {}, []
    try:
        index = R.load_index(directory)
    except (R.ReplayError, ValueError) as exc:
        return {}, ["  %s does not parse and its ledger cannot be read: %s" % (index_path, exc)]
    ledger = index.get("retired")
    legacy = (index.get("provenance") or {}).get("retired")
    if ledger is not None and legacy is not None:
        return {}, [
            "  %s carries a ledger under BOTH `retired` and `provenance.retired`. One of them is "
            "inside `set_sha256` and one is not, so the two can disagree and a reader cannot tell "
            "which is the record. Restore the index from git." % index_path
        ]
    if ledger is None and legacy is not None:
        # THE DOWNGRADE, refused rather than accepted. This branch used to READ the legacy key,
        # for sets written before the ledger moved inside the digest. A seventh adversarial pass
        # pointed out what that buys an attacker: move the ledger back under `provenance`,
        # recompute `set_sha256` over the now-smaller core, and the index is self-consistent while
        # this reader still finds the history -- outside the digest again, exactly where the
        # repair took it from. The compatibility path undid the repair.
        #
        # It is refused instead of removed silently because the refusal names the migration. And
        # it costs nothing: the migration is finished. The only v8 set in this tree carries
        # `retired` at the top level and no `provenance.retired`, so the branch had no remaining
        # honest consumer at the moment it was closed. A genuinely old set restored from git is
        # now migrated deliberately rather than accepted quietly, which is the right way round.
        #
        # What this does NOT close, stated so nobody reads it as more than it is: an attacker who
        # edits the ledger IN PLACE under `retired` and recomputes `set_sha256` leaves a
        # self-consistent index, and nothing inside the file catches that. What catches it is the
        # `set_sha256` pinned in `mutation_coverage.json` and the previous bytes in git, both of
        # which are outside the artifact a stranger receives.
        return {}, [
            "  %s carries its retirement ledger under `provenance.retired`, where it sits OUTSIDE "
            "`set_sha256`, and carries none under `retired`. Either this set predates the ledger "
            "moving inside the digest, in which case migrate it deliberately by moving the key and "
            "regenerating, or somebody moved it back out. This reader will not accept a ledger "
            "the digest does not cover." % index_path
        ]
    if ledger is None:
        if remains:
            return {}, [
                "  %s carries no ledger under `retired` or `provenance.retired`, and the "
                "directory holds %s. An index with vectors beside it and no ledger is an index "
                "somebody edited; restore it from git rather than letting this run write a fresh "
                "one." % (index_path, ", ".join(remains))
            ]
        return {}, []
    return ledger, []


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="generate the styxx.v8 conformance set")
    parser.add_argument("--check", action="store_true",
                        help="regenerate in memory; exit 1 if set_sha256 differs")
    parser.add_argument("--replay", action="store_true",
                        help="replay the committed set instead of regenerating it")
    parser.add_argument("--dir", default=str(HERE))
    parser.add_argument("--retire", action="append", default=[], metavar="ID",
                        help="retire the committed answer at this address; pair it with --reason")
    parser.add_argument("--reason", action="append", default=[], metavar="TEXT",
                        help="why the change at the preceding --retire is deliberate")
    args = parser.parse_args(argv)

    directory = Path(args.dir).resolve()
    if args.replay:
        return R.main(["--dir", str(directory)])

    plan = retirement_plan(args.retire, args.reason)

    # Read before recording: a directory that has lost the index its history lived in is refused
    # before an hour of pytest, not after.
    previous, cannot = read_ledger(directory)
    if cannot:
        print("REFUSED: the retirement ledger cannot be carried forward:\n%s" % "\n".join(cannot),
              file=sys.stderr)
        print("Nothing was written.", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as td:
        records = record(SOURCES, Path(td) / "records.jsonl")
    parts = assemble(records)
    vectors = parts["vectors"]

    old = committed(directory)
    moved: List[dict] = []
    drops: List[dict] = []
    if old:
        moved = moved_cores(old, vectors)
        try:
            on_disk = R.load_blobs(directory)
        except (R.ReplayError, ValueError):
            on_disk = {}
        drops = classify_drops(old, vectors, on_disk)
        added = sorted(set(vectors) - set(old))
        if added:
            print("  %d vectors added" % len(added))
        by_kind = Counter(row["kind"] for row in drops)
        for kind in ("input-churn", "behaviour-change", "undiagnosed"):
            if by_kind.get(kind):
                print("  %d dropped: %s" % (by_kind[kind], DROP_KINDS[kind]))
        for row in drops:
            print("  DROPPED %s (%s) -- %s: %s"
                  % (row["id"], row["entrypoint"], row["kind"], row["detail"]))

    recorded, unnamed, refusals, undiagnosed = apply_retirements(plan, moved, drops)
    if refusals:
        print("REFUSED: --retire names %d address(es) that this run cannot retire:\n%s"
              % (len(refusals), "\n".join(refusals)), file=sys.stderr)
        print("Nothing was written.", file=sys.stderr)
        return 1
    if undiagnosed:
        rows = {row["id"]: row for row in drops}
        print("REFUSED: %d dropped vector(s) cannot be replayed at all:\n%s"
              % (len(undiagnosed),
                 "\n".join("  %s (%s) -- %s"
                           % (vid, rows[vid]["entrypoint"], rows[vid]["detail"])
                           for vid in undiagnosed)),
              file=sys.stderr)
        print("An undiagnosed drop is the one drop no reason can be written about: neither the "
              "answer it used to give nor the answer this tree gives is available, so a "
              "retirement here would record a decision nobody was in a position to make. "
              "--retire refuses these by name. Make it replayable, or find the defect in the "
              "vector. Nothing was written.", file=sys.stderr)
        return 1
    if unnamed:
        rows = {row["id"]: row for row in moved + drops}
        print("REFUSED: %d committed vector(s) now carry a different outcome:\n%s"
              % (len(unnamed), "\n".join(format_move(rows[vid]) for vid in unnamed)),
              file=sys.stderr)
        print("A moved core is a finding about styxx.v8. Nothing was written.\n"
              "If the change is deliberate, retire each address by name and say why:\n"
              "  --retire %s --reason \"...\"" % unnamed[0], file=sys.stderr)
        return 1
    for row in recorded:
        print("  RETIRED %s (%s) -- %s" % (row["id"], row["kind"], row["reason"]))

    # The ledger is inside `set_sha256`, so it is known before the digest rather than patched in
    # after it. Carried forward whether or not this directory still holds vectors: the half of the
    # attack that deleted `vectors/` used to reach the default through the old `if old:`.
    ledger = retirement_ledger(previous, recorded, drops, current=vectors)
    index, files = finalize(parts, ledger)
    print("%d vectors, %d blobs, %d retired with a reason, %d churned, set_sha256 %s"
          % (index["vectors"], index["blobs"]["count"],
             ledger["counts"]["with_reason"], ledger["counts"]["input_churn"],
             index["set_sha256"]))

    if args.check:
        was = None
        try:
            was = R.load_index(directory).get("set_sha256")
        except R.ReplayError:
            pass
        if was == index["set_sha256"]:
            print("set_sha256 unchanged")
            return 0
        print("set_sha256 differs: committed %s, this run %s" % (was, index["set_sha256"]),
              file=sys.stderr)
        return 1

    (directory / "vectors").mkdir(parents=True, exist_ok=True)
    for rel, data in sorted(files.items()):
        path = directory / rel
        path.write_bytes(data)
        back = path.read_bytes()
        if back != data:
            raise SystemExit("REFUSED: %s did not read back as written" % rel)
    print("wrote %d files under %s" % (len(files), directory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
