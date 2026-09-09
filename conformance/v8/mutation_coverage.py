# -*- coding: utf-8 -*-
"""Mutation coverage for the styxx.v8 conformance set: what disagreement could it actually see?

Built to `styxx/v8/INTERFACES_layer2.md` section 10, under the lab's calibration rule: an agreement
number is worth nothing without a measurement of the instrument's detection power.

A set replaying green means one of two things -- the implementation agrees with the set, or
the set cannot reach where an implementation would differ -- and the replay cannot tell them apart.
This tells them apart, for a catalogue of specific behaviours. One localised edit is applied to a
scratch copy of one module, the copy is loaded and monkeypatched over the real one for the length
of the run, the committed set is replayed, and the question is whether any vector noticed.

**caught** means a second implementation that got that behaviour wrong would fail this set.
**missed** means it would pass, and the miss list is the deliverable: a real difference could sit
in any of those places today with every replay in this repository looking exactly as it does now.

Nothing here writes into the tree. Every mutant is a file in a temporary directory, every patch is
undone in a `finally`, and the run refuses to overwrite a receipt git already tracks.

    python conformance/v8/mutation_coverage.py
    python conformance/v8/mutation_coverage.py --catalogue <in.json> --out <receipt.json>

The catalogue is a JSON object with a `mutations` list of
`{name, module, old, new, why, region, control}`. `old` must occur EXACTLY once in the named
module, or the mutation is recorded as `anchor_missing` / `anchor_ambiguous` and excluded from the
denominator, as are mutants that will not load.

Two limits are stated rather than worked around. A module is patched by rebinding, in every loaded
`styxx.*` and `conformance.*` module, each attribute that holds the same object under the same
name -- so `from styxx.v8 import merkle` and `from styxx.v8.distances import ABSENT_LP` both
follow the mutant, but an import that RENAMES what it binds (`from styxx.v8.jcs import digest as
_jcs_digest`) does not, and `styxx/v8/jcs.py` is therefore not a mutable module here. And a
mutation of a module the set does not call through at all is not a measurement of the set: it is
listed, it is missed, and the miss says which entrypoint is absent rather than pretending the
vectors looked.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conformance.v8 import FAMILIES  # noqa: E402
from conformance.v8 import replay as R  # noqa: E402

CATALOGUE = HERE / "mutation_catalogue.json"
RECEIPT = HERE / "mutation_coverage.json"
SCHEMA = "styxx.v8.mutation-coverage/v1"

#: Modules a mutation may name. `styxx/v8/jcs.py` is absent on purpose: see the module docstring.
MUTABLE = (
    "styxx/v8/cert.py",
    "styxx/v8/distances.py",
    "styxx/v8/floor.py",
    "styxx/v8/keys.py",
    "styxx/v8/log.py",
    "styxx/v8/merkle.py",
    "styxx/v8/verify.py",
)

#: Gate bars, fixed here rather than argued after the numbers are in.
BAR_VIABLE = 12
BAR_CONTROLS = 3
BAR_REGIONS = 5


def _source(rel: str) -> str:
    """The module's text with LF endings: this box has core.autocrlf on and an anchor is text."""
    return (ROOT / rel).read_bytes().decode("utf-8").replace("\r\n", "\n")


def _sha(rel: str) -> str:
    return hashlib.sha256(_source(rel).encode("utf-8")).hexdigest()


def module_name(rel: str) -> str:
    return rel[: -len(".py")].replace("/", ".")


# --------------------------------------------------------------------------- loading a mutant


def load_mutant(rel: str, text: str, scratch: Path) -> ModuleType:
    """Load `text` as a stand-in for `rel`, under the real module's dotted name.

    The scratch directory gets `styxx/v8/schema/` alongside the copy, because `cert.py` resolves
    its schemas from `__file__`; loading a copy from a bare temp directory would otherwise change
    a behaviour the mutation did not ask to change.
    """
    name = module_name(rel)
    target = scratch / Path(rel).name
    target.write_bytes(text.encode("utf-8"))
    schema_src = ROOT / "styxx" / "v8" / "schema"
    schema_dst = scratch / "schema"
    if schema_src.is_dir() and not schema_dst.exists():
        shutil.copytree(schema_src, schema_dst)
    spec = importlib.util.spec_from_file_location(name, str(target))
    if spec is None or spec.loader is None:
        raise ImportError("no loader for %s" % target)
    mutant = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mutant)
    return mutant


def _patch_targets() -> List[ModuleType]:
    return [
        mod
        for name, mod in list(sys.modules.items())
        if mod is not None and (name.startswith("styxx") or name.startswith("conformance"))
    ]


def install(rel: str, mutant: ModuleType) -> List[Tuple[Any, str, Any]]:
    """Put `mutant` in the real module's place. Returns the undo list."""
    name = module_name(rel)
    real = sys.modules[name]
    undo: List[Tuple[Any, str, Any]] = []
    for mod in _patch_targets():
        if mod is real:
            continue
        for attr, value in list(vars(mod).items()):
            if attr.startswith("__"):
                continue
            if value is real:
                undo.append((mod, attr, value))
                setattr(mod, attr, mutant)
                continue
            # `from styxx.v8.distances import ABSENT_LP` binds the object, not the module.
            if hasattr(real, attr) and getattr(real, attr) is value and hasattr(mutant, attr):
                replacement = getattr(mutant, attr)
                if replacement is not value:
                    undo.append((mod, attr, value))
                    setattr(mod, attr, replacement)
    undo.append((sys.modules, name, real))
    sys.modules[name] = mutant
    return undo


def uninstall(undo: List[Tuple[Any, str, Any]]) -> None:
    for target, attr, value in reversed(undo):
        if isinstance(target, dict):
            target[attr] = value
        else:
            setattr(target, attr, value)


# --------------------------------------------------------------------------- one mutation


def run_mutation(mutation: dict, vectors, blobs, scratch: Path) -> dict:
    """Apply one mutation, replay the committed set, and report what the set saw."""
    out = dict(mutation)
    rel = mutation.get("module")
    if rel not in MUTABLE:
        out["verdict"] = "not_mutable"
        out["detail"] = "%r is not in MUTABLE; see the module docstring" % (rel,)
        return out
    if mutation["old"] == mutation["new"]:
        out["verdict"] = "no_op"
        return out

    text = _source(rel)
    hits = text.count(mutation["old"])
    out["anchor_occurrences"] = hits
    if hits == 0:
        out["verdict"] = "anchor_missing"
        return out
    if hits > 1:
        out["verdict"] = "anchor_ambiguous"
        out["detail"] = "the anchor occurs %d times; an edit must name one place" % hits
        return out

    work = scratch / mutation["name"]
    work.mkdir(parents=True, exist_ok=True)
    try:
        mutant = load_mutant(rel, text.replace(mutation["old"], mutation["new"], 1), work)
    except Exception as exc:  # noqa: BLE001
        out["verdict"] = "non_viable"
        out["detail"] = "the mutant will not load: %s: %s" % (type(exc).__name__, str(exc)[:200])
        return out

    undo = install(rel, mutant)
    try:
        counts, failures = R.replay_set(vectors, blobs)
    except Exception as exc:  # noqa: BLE001
        uninstall(undo)
        out["verdict"] = "non_viable"
        out["detail"] = "the replay could not run: %s: %s" % (type(exc).__name__, str(exc)[:200])
        return out
    finally:
        uninstall(undo)

    total = sum(row["vectors"] for row in counts.values())
    unreplayable = sum(1 for _f, _i, why in failures if why.startswith("the vector cannot"))
    out["vectors"] = total
    out["noticed"] = len(failures)
    out["noticed_by_family"] = {
        family: sum(1 for f, _i, _w in failures if f == family) for family in FAMILIES
    }
    out["unreplayable"] = unreplayable
    out["example"] = None if not failures else {
        "family": failures[0][0], "id": failures[0][1], "why": failures[0][2][:240]
    }
    if unreplayable == total and total:
        out["verdict"] = "non_viable"
        out["detail"] = "every vector became unreplayable; this measures nothing"
        return out
    out["verdict"] = "caught" if failures else "missed"
    return out


# --------------------------------------------------------------------------- the run


def _tracked(path: Path) -> bool:
    try:
        proc = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "--error-unmatch", str(path)],
            capture_output=True,
        )
    except OSError:
        return False
    return proc.returncode == 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="measure what the styxx.v8 vectors can see")
    parser.add_argument("--catalogue", default=str(CATALOGUE))
    parser.add_argument("--out", default=str(RECEIPT))
    parser.add_argument("--dir", default=str(HERE), help="the set to replay")
    args = parser.parse_args(argv)

    out_path = Path(args.out).resolve()
    if out_path.exists() and _tracked(out_path):
        print(
            "REFUSED: %s is tracked; a run is history -- write a new file rather than "
            "regenerating a committed receipt in place" % out_path.name,
            file=sys.stderr,
        )
        return 2

    directory = Path(args.dir).resolve()
    blobs = R.load_blobs(directory)
    vectors = R.load_vectors(directory)
    index = R.load_index(directory)

    # The unmutated set first. A detection rate measured over a set that does not already
    # reproduce would be a number about the baseline, not about the mutations.
    counts, failures = R.replay_set(vectors, blobs)
    total = sum(row["vectors"] for row in counts.values())
    print("baseline: %d vectors, %d do not reproduce" % (total, len(failures)), flush=True)
    if failures:
        for family, vid, why in failures[:10]:
            print("  %s %s: %s" % (family, vid, why), file=sys.stderr)
        print("REFUSED: the committed set does not replay; nothing below would mean anything",
              file=sys.stderr)
        return 2

    catalogue = json.loads(Path(args.catalogue).read_text(encoding="utf-8"))
    mutations = catalogue["mutations"] if isinstance(catalogue, dict) else catalogue

    results: List[dict] = []
    with tempfile.TemporaryDirectory() as td:
        scratch = Path(td)
        for i, mutation in enumerate(mutations, 1):
            row = run_mutation(mutation, vectors, blobs, scratch)
            results.append(row)
            print(
                "  [%2d/%2d] %-16s %-34s %s"
                % (i, len(mutations), row["verdict"], row.get("name", "")[:34],
                   ("%d of %d vectors noticed" % (row["noticed"], row["vectors"]))
                   if "noticed" in row else row.get("detail", "")[:60]),
                flush=True,
            )

    verdicts = Counter(row["verdict"] for row in results)
    controls = [r for r in results if r.get("control")]
    mutants = [r for r in results if not r.get("control")]
    viable = [r for r in mutants if r["verdict"] in ("caught", "missed")]
    caught = [r for r in viable if r["verdict"] == "caught"]
    missed = [r for r in viable if r["verdict"] == "missed"]
    controls_viable = [r for r in controls if r["verdict"] in ("caught", "missed")]
    controls_caught = [r for r in controls_viable if r["verdict"] == "caught"]

    by_region: Dict[str, dict] = {}
    by_module: Dict[str, dict] = {}
    for row in viable:
        for bucket, key in ((by_region, row.get("region", "?")), (by_module, row["module"])):
            cell = bucket.setdefault(key, {"viable": 0, "caught": 0, "missed": 0})
            cell["viable"] += 1
            cell["caught" if row["verdict"] == "caught" else "missed"] += 1

    gates = {
        "G-M": {"quantity": "viable mutants measured", "value": len(viable),
                "bar": ">= %d" % BAR_VIABLE, "pass": len(viable) >= BAR_VIABLE},
        "G-C": {"quantity": "viable semantics-preserving controls", "value": len(controls_viable),
                "bar": ">= %d" % BAR_CONTROLS, "pass": len(controls_viable) >= BAR_CONTROLS},
        "G-K": {"quantity": "controls caught", "value": len(controls_caught), "bar": "== 0",
                "pass": len(controls_caught) == 0,
                "note": "a caught control VOIDS the run: the set would be detecting editing "
                        "rather than a change of behaviour"},
        "G-R": {"quantity": "regions with at least one viable mutant", "value": len(by_region),
                "bar": ">= %d" % BAR_REGIONS, "pass": len(by_region) >= BAR_REGIONS},
        "G-D": {"quantity": "detection rate, caught / viable",
                "value": {"caught": len(caught), "viable": len(viable),
                          "rate": round(len(caught) / len(viable), 4) if viable else None},
                "bar": "none -- reported, never passed or failed", "pass": None},
    }
    void = not gates["G-K"]["pass"]

    receipt = {
        "schema": SCHEMA,
        "contract": "styxx/v8/INTERFACES_layer2.md section 10",
        "void": void,
        "set": {
            "dir": "conformance/v8",
            "set_sha256": index.get("set_sha256"),
            "vectors": total,
            "families": {family: counts.get(family, {}) for family in FAMILIES},
        },
        "catalogue": {
            "file": str(Path(args.catalogue).name),
            "sha256": hashlib.sha256(
                Path(args.catalogue).read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest(),
        },
        "implementation": {rel: _sha(rel) for rel in MUTABLE},
        "not_mutable": {
            "styxx/v8/jcs.py": "its consumers rename what they import, so a rebinding patch does "
                               "not reach them; a mutation here would measure the patcher"
        },
        "counts": {
            "proposed": len(mutations),
            "controls": len(controls),
            "controls_viable": len(controls_viable),
            "controls_caught": len(controls_caught),
            "viable": len(viable),
            "caught": len(caught),
            "missed": len(missed),
            "anchor_missing": verdicts.get("anchor_missing", 0),
            "anchor_ambiguous": verdicts.get("anchor_ambiguous", 0),
            "non_viable": verdicts.get("non_viable", 0),
            "no_op": verdicts.get("no_op", 0),
            "not_mutable": verdicts.get("not_mutable", 0),
        },
        "gates": gates,
        "by_region": by_region,
        "by_module": by_module,
        # The misses ARE the result. Named in full, never summarised into a rate.
        "missed": [
            {"name": r["name"], "module": r["module"], "region": r.get("region"),
             "old": r["old"], "new": r["new"], "why": r.get("why")}
            for r in missed
        ],
        "mutations": results,
        "reading": (
            "caught means a second implementation that got this behaviour wrong would fail the "
            "committed set. missed means it would pass, and a real difference could sit there "
            "today with the replay looking exactly as it does now. The rate is a property of the "
            "SET and of this catalogue together; a catalogue that avoids the thin places would "
            "report a higher one and mean less."
        ),
    }
    out_path.write_bytes(
        (json.dumps(receipt, indent=1, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")
    )
    print(
        "\nviable %d  caught %d  missed %d  (controls %d, caught %d)  -> %s"
        % (len(viable), len(caught), len(missed), len(controls), len(controls_caught),
           out_path.name)
    )
    for row in missed:
        print("  MISSED %-30s %s" % (row["name"], (row.get("why") or "")[:90]))
    if void:
        print("VOID: a control was caught", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
