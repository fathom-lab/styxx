# -*- coding: utf-8 -*-
"""Generate locks/LOCKS_checksum_deploy_quant_reconstructed.md: the checksum scorer's deploy-quant path
written as a locks decision, a RECONSTRUCTION for step B4 of the prereg-ci build.

  python -B conformance/prereg_ci/gen_locks_checksum.py           write the file where none is committed
  python -B conformance/prereg_ci/gen_locks_checksum.py --check   regenerate in memory; exit 1 if a byte differs

What it writes and why (DESIGN 3.2(a) and (f), PLAN K2b). The two checksum PREREGs froze their reading
rules as prose and are sealed (``papers/checksum/SEALS_2026_09_13.md`` rows 3 and 6), so neither can gain
a locks fence. The rules were executed later by ``papers/checksum/score.py``. The file this writes is a
locks block over that scorer as it stands at ``SCORER_COMMIT`` (the commit that last wrote it and the v2
dry-run card), labelled a reconstruction the way the design labels its reconstructed EXTERNAL-1 table
(RV4): written after the data and after the scorer, never sealed, not a catch.

The block's ``outcomes`` are the readings the deploy-quant path can print, enumerated from the scorer's
code at that blob. The generator reads the blob from git, refuses unless its sha256 is ``SCORER_SHA256``,
and refuses unless every line the grammar is read from (``ANCHORS``) is in it, in order; it prints each
anchor's line. The grammar (``outcomes()``):
- K1 fired: ``INCONCLUSIVE (K1)``, behind ``INSTRUMENT CHECK; `` when the certs are not the experiment;
- else ``gates fired: G, ...`` over the gates that fired among provenance, K2, K3, K4 in the card's
  order (K1 sits ahead of them and did not fire on this branch), or ``no gate fired``; then
  ``H1 S1, H2 S2, H3 S3, H4 S4``, each status HELD, FAILED or NOT_EVALUABLE, and H1's also the
  CORRECTION's "HELD on the floor, INCONCLUSIVE on the pair by construction". ``INSTRUMENT CHECK; ``
  leads exactly when the provenance gate fired, since a run that is not the experiment fires it.
That is 2 + 16 x 108 = 1730 strings, a superset of what can occur (K2 fires only when A vs Q4 reads
SAME, which fails H2's DRIFT clause). ``tests/test_prereg_ci_scorer_run.py`` restates the grammar apart
from this file and checks 4000 varied certs through the scorer, each reading one of them.

The committed file is never written again (a receipt is history too): a run that regenerates it byte
for byte exits 0, any other prints DIFFERS and writes nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from styxx.prereg_ci import core  # noqa: E402
from styxx.prereg_ci import git as G  # noqa: E402

OUT = ROOT / "conformance" / "prereg_ci" / "locks" / "LOCKS_checksum_deploy_quant_reconstructed.md"
SCORE = "papers/checksum/score.py"
CERTS = "papers/checksum/deploy_quant_certs_dryrun_qwen0.5b.json"
CARD = "papers/checksum/deploy_quant_scorecard_v2_dryrun_qwen0.5b.json"
SCORER_COMMIT = "f0ec56056fc6c903e7cfbf70a1b17c9f5ba1574e"
SCORER_SHA256 = "da9c5de490b590da2a931d631fdc29932be987e175fcbb39788be55bce85bec9"
CERTS_COMMIT = "2b4c3dcd"
OLDEST_SCORER_COMMIT = "383b4155cb92f9234434f17ef23ce52a59b08211"
# the lines of score.py the grammar is read from, in the order they stand in the blob
ANCHORS = (
    'def _status(clauses):',
    'return "NOT_EVALUABLE"',
    'return "HELD" if all(hs) else "FAILED"',
    'card["gates"]["K1"] = {"fired": k1_fired if floor is not None else None',
    'card["gates"]["provenance"] = {"fired": bool(problems)',
    'valid_experiment = card["is_the_experiment"] and not invalid',
    '("" if valid_experiment else "INSTRUMENT CHECK; ")',
    'card["run_reading"] = prefix + "INCONCLUSIVE (K1)"',
    'st = "HELD on the floor, INCONCLUSIVE on the pair by construction"   # CORRECTION rule 2',
    'card["hypotheses"]["H2"] = {"status": _status(cl), "clauses": cl}',
    'card["hypotheses"]["H3"] = {"status": _status(cl), "clauses": cl}',
    'card["hypotheses"]["H4"] = {"status": _status(cl), "clauses": cl}',
    'card["gates"]["K2"] = {"fired": dq4.get("verdict") == "SAME" if dq4 else None',
    'card["gates"]["K3"] = {"fired": bool(k3)',
    'card["gates"]["K4"] = {"fired": bool(k4)',
    'fired = [g for g, v in card["gates"].items() if v.get("fired")]',
    """card["run_reading"] = prefix + (f"gates fired: {', '.join(fired)}" if fired else "no gate fired") + "; " +""",
    """", ".join(f"{h} {v['status']}" for h, v in card["hypotheses"].items())""",
)
STATUS = ("HELD", "FAILED", "NOT_EVALUABLE")
H1_STATUS = STATUS + ("HELD on the floor, INCONCLUSIVE on the pair by construction",)
LATER_GATES = ("K2", "K3", "K4")
PREFIX = "INSTRUMENT CHECK; "


class Refused(Exception):
    pass


def outcomes() -> List[str]:
    """Every reading of the deploy-quant path, sorted."""
    out = {"INCONCLUSIVE (K1)", PREFIX + "INCONCLUSIVE (K1)"}
    for provenance in (False, True):
        for k in range(len(LATER_GATES) + 1):
            for later in itertools.combinations(LATER_GATES, k):
                fired = (["provenance"] if provenance else []) + list(later)
                gates = f"gates fired: {', '.join(fired)}" if fired else "no gate fired"
                for h1, h2, h3, h4 in itertools.product(H1_STATUS, STATUS, STATUS, STATUS):
                    out.add((PREFIX if provenance else "") + f"{gates}; H1 {h1}, H2 {h2}, H3 {h3}, H4 {h4}")
    return sorted(out)


def _anchors(src: str) -> List[int]:
    """The 1-based line of each anchor, refused unless each stands in the blob once and in order."""
    lines, at, pos = src.split("\n"), [], -1
    for a in ANCHORS:
        hits = [i for i, ln in enumerate(lines) if a in ln]
        if len(hits) != 1:
            raise Refused(f"the anchor {a!r} stands {len(hits)} times in {SCORE}; the grammar reads it once")
        if hits[0] <= pos:
            raise Refused(f"the anchor {a!r} (line {hits[0] + 1}) is not below the one above it")
        pos = hits[0]
        at.append(hits[0] + 1)
    return at


PROSE = """# LOCKS: the checksum deploy-quant scorer as a locks decision (a reconstruction)

status: RECONSTRUCTION, written 2026-10-25 for step B4 of the prereg-ci build, after the data and after the
scorer. It is not a preregistration: it was never sealed, nothing was registered with it before any run, and
a reading under it is not a catch. It is labelled as the design labels its reconstructed EXTERNAL-1 table.
Generated by `conformance/prereg_ci/gen_locks_checksum.py`; never edited by hand.

## what it locks

`papers/checksum/PREREG_checksum_deploy_quant_2026_09_13.md` froze its reading rules as prose and holds no
decision fence, so prereg-ci reads it as NOT_GATED; it is sealed (`papers/checksum/SEALS_2026_09_13.md`
row 3), so it cannot gain one. Its rules were executed later by `papers/checksum/score.py`. The block below
locks that scorer as it stands at `{commit}` (blob sha256 `{sha}`), the commit that last wrote it
and the v2 dry-run card, and runs it as the lab ran it:

- `scorer` is `papers/checksum/score.py`; `reads` is empty, since the deploy-quant path imports the
  standard library alone;
- `argv` is the certs, `--prereg deploy_quant`, and `--out` with the card's path;
- the verdict is the card's `run_reading` (`verdict_from` `/run_reading`), and `withdraws` is empty: no
  sentence's licence is reconstructed here.

prereg-ci's scorer run writes the scorer's blob and the certs into an empty directory and runs them under
`python -I -S -B -X utf8`. Over `{certs}` it writes the committed card byte for byte, at each of the
four commits that wrote a deploy-quant dry-run card, each run with the scorer's blob of that commit; a
moved number in the certs moves the reading, and a card recorded with another reading is VERDICT_DIFFERS.

## the outcomes

`outcomes` is every reading the deploy-quant path can print, enumerated from the scorer's code at that
blob: `INCONCLUSIVE (K1)` when K1 fired; else the gates that fired among provenance, K2, K3 and K4 in the
card's order, or `no gate fired`, then the four hypotheses' statuses (HELD, FAILED or NOT_EVALUABLE, and
for H1 also the CORRECTION's `HELD on the floor, INCONCLUSIVE on the pair by construction`).
`INSTRUMENT CHECK; ` leads exactly when the provenance gate fired. That is {n} strings, a superset of
what can occur: K2 fires only when A vs Q4 reads SAME, which fails H2's DRIFT clause.

## what it cannot show

- Order. The certs were written at `{certs_commit}` (2026-09-13), before the scorer's oldest blob at
  `{oldest}` (2026-09-14). Sealed, a block holding the scorer's blob could only follow the receipt it
  scores, and the order check would read it NOT_ORDERED. The run re-derives committed bytes; it orders
  nothing.
- A result. The dry-run certs are not the experiment: another model, a tag, and no canary set named. The
  card's own reading begins INSTRUMENT CHECK and its `counts_as_result` is false. Re-deriving it is an
  instrument check, not a result.
- The beacon-draw path. `--prereg beacon_draw` imports `styxx.beacon`, which imports `styxx.checksum`,
  which imports numpy at module level. `-S` leaves site packages off the path, so the run records the
  failed import and reads SCORER_READS_UNLOCKED; no block is written for that path. The scorer itself
  catches the failure and prints its committed reading anyway, with one more line in K5's detail, which
  is why the run watches imports and does not read that agreement as a re-derivation.
- The interpreter. The run locks the scorer's bytes, not the Python that runs them; it prints the
  runtime's implementation, version and Unicode database beside the reading.

"""


def build(repo: G._Repo) -> dict:
    blob = repo.read(SCORER_COMMIT, SCORE)
    if blob is None:
        raise Refused(f"{SCORE} is not a file at {SCORER_COMMIT[:12]}")
    got = hashlib.sha256(blob).hexdigest()
    if got != SCORER_SHA256:
        raise Refused(f"{SCORE} at {SCORER_COMMIT[:12]} hashes to {got}, not {SCORER_SHA256}")
    lines = _anchors(blob.decode("utf-8"))
    block = {"scorer": SCORE, "reads": [], "argv": ["{receipt}", "--prereg", "deploy_quant", "--out", "{out}"],
             "verdict_from": "/run_reading", "withdraws": {}, "outcomes": outcomes()}
    fence = json.dumps(block, indent=1, ensure_ascii=True)
    text = PROSE.format(commit=SCORER_COMMIT, sha=SCORER_SHA256, certs=CERTS, n=len(block["outcomes"]),
                        certs_commit=CERTS_COMMIT, oldest=OLDEST_SCORER_COMMIT[:8])
    text += "```locks\n" + fence + "\n```\n"
    raw = text.encode("utf-8")
    d = core._decision(raw)
    if d["form"] != "locks" or core._locks(d["block"]) != {**block, "locked": [SCORE]}:
        raise Refused(f"the written block does not read back through core._decision and core._locks: {d}")
    reading = core._result_reading(raw)
    if reading["code"] is not None or reading["declarations"]:
        raise Refused("the reconstruction reads as a RESULT; it declares nothing")
    if b"\r" in raw or raw.startswith(b"\xef\xbb\xbf"):
        raise Refused("the reconstruction holds a CR or a BOM")
    return {"bytes": raw, "anchors": lines, "outcomes": len(block["outcomes"]), "digest": d["gates_sha256"]}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="regenerate in memory; exit 1 if a byte differs")
    args = ap.parse_args(argv)
    try:
        with G._Repo(ROOT) as repo:
            got = build(repo)
    except (Refused, G._Refusal) as e:
        print(f"REFUSED {e}")
        return 1
    raw = got["bytes"]
    name = f"{OUT.parent.name}/{OUT.name}"
    if args.check or OUT.exists():
        if OUT.exists() and OUT.read_bytes() == raw:
            print(f"SAME {name}={hashlib.sha256(raw).hexdigest()[:16]}; decision digest {got['digest']}; "
                  f"{got['outcomes']} outcomes; nothing written")
        else:
            print(f"DIFFERS {name}: the run does not regenerate the committed bytes; nothing written. A new "
                  f"reconstruction goes in a new file.")
            return 1
    else:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_bytes(raw)
        if OUT.read_bytes() != raw:
            print(f"REFUSED {name} did not read back as written")
            return 1
        print(f"WROTE {name} ({len(raw)} bytes); decision digest {got['digest']}; {got['outcomes']} outcomes")
    for a, line in zip(ANCHORS, got["anchors"]):
        print(f"ANCHOR {SCORE}:{line} {a}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
