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

Vectors that the run no longer produces are NOT a refusal -- a test may legitimately be rewritten --
but every dropped id is printed, so the loss is visible in the terminal and in the diff.

Nothing here has a clock. The set is a function of the sources and of `styxx.v8`, so two runs on
two machines produce the same bytes, and `--check` is a real comparison rather than a timestamp.
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
from typing import Any, Dict, List, Optional, Tuple

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


def build(records: List[dict]) -> Tuple[dict, Dict[str, bytes], Dict[str, dict]]:
    """Records -> (index, {relative path: bytes}, vectors by id). Replays before returning."""
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
        "note": "outside set_sha256: the set is a function of these bytes, not of this block",
        "implementation": {p: _sha_file(ROOT / p) for p in IMPLEMENTATION},
        "tooling": {p: _sha_file(ROOT / p) for p in TOOLING},
        "sources": {p: _sha_file(ROOT / p) for p in SOURCES},
        "replay": {f: counts.get(f, {}) for f in FAMILIES},
    }
    files["index.json"] = _dump(index)
    return index, files, vectors


def committed(directory: Path) -> Dict[str, dict]:
    """The vectors already on disk, by id. Empty when the set has never been generated."""
    out: Dict[str, dict] = {}
    for family, family_vectors in R.load_vectors(directory).items():
        for vector in family_vectors:
            out[vector["id"]] = vector
    return out


def compare_to_committed(old: Dict[str, dict], new: Dict[str, dict]) -> List[str]:
    """The moved cores: ids in both whose outcome changed. Prints the additions and the drops."""
    moved: List[str] = []
    for vid, vector in sorted(new.items()):
        was = old.get(vid)
        if was is None:
            continue
        if canonical_bytes(was.get("expect")) != canonical_bytes(vector["expect"]):
            moved.append(
                "  %s (%s)\n    was      %s\n    this run %s\n    sources  %s"
                % (vid, vector["entrypoint"],
                   json.dumps(was.get("expect"), sort_keys=True)[:300],
                   json.dumps(vector["expect"], sort_keys=True)[:300],
                   ", ".join(vector["sources"]))
            )
    added = sorted(set(new) - set(old))
    dropped = sorted(set(old) - set(new))
    if added:
        print("  %d vectors added" % len(added))
    for vid in dropped:
        print("  DROPPED %s (%s) -- the run no longer produces it"
              % (vid, old[vid].get("entrypoint")))
    return moved


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="generate the styxx.v8 conformance set")
    parser.add_argument("--check", action="store_true",
                        help="regenerate in memory; exit 1 if set_sha256 differs")
    parser.add_argument("--replay", action="store_true",
                        help="replay the committed set instead of regenerating it")
    parser.add_argument("--dir", default=str(HERE))
    args = parser.parse_args(argv)

    directory = Path(args.dir).resolve()
    if args.replay:
        return R.main(["--dir", str(directory)])

    with tempfile.TemporaryDirectory() as td:
        records = record(SOURCES, Path(td) / "records.jsonl")
    index, files, vectors = build(records)
    print("%d vectors, %d blobs, set_sha256 %s"
          % (index["vectors"], index["blobs"]["count"], index["set_sha256"]))

    old = committed(directory)
    if old:
        moved = compare_to_committed(old, vectors)
        if moved:
            print("REFUSED: %d committed vector(s) now carry a different outcome:\n%s"
                  % (len(moved), "\n".join(moved)), file=sys.stderr)
            print("A moved core is a finding about styxx.v8. Nothing was written.", file=sys.stderr)
            return 1

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
