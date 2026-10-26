# -*- coding: utf-8 -*-
"""The reference legs the prereg-ci vectors are replayed through, and the brute-force cover.

Nothing here imports ``styxx.prereg_ci``: these are the two scorers the lab already ships, run the
way their callers run them, so a vector compares prereg-ci's reading against them and never
against itself.

* ``frozen_leg``: ``research_capsule.score_frozen`` over ``protocol._select_gates_block`` of the
  PREREG bytes decoded as UTF-8, as the design's census (R1) and ``research_capsule`` call it. No
  newline translation and no smoke branch: score_frozen has none.
* ``experiment_leg``: ``protocol.Experiment(path).score(receipt, smoke=bool(receipt.get("smoke")))``
  over a PREREG committed to a git repository, which is how ``styxx.seal`` scores a pair. The
  Experiment reads the working copy with universal newlines and asks git whether the file was
  added, so the PREREG is written to disk and committed by ``build_experiment_repo``.
* ``brute_force``: every assignment of the gates walked through the outcome rows with the
  scorers' own matching expression, ``all(fired.get(k) == v for k, v in when.items())``; the
  earliest matching row lands.
"""
from __future__ import annotations

import base64
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from styxx.protocol import Experiment, GateSpecError, _select_gates_block
from styxx.research_capsule import score_frozen


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_id(name: str, inputs: dict) -> str:
    """A vector's id: sha256 of the sorted, compact UTF-8 JSON of its name and inputs."""
    text = json.dumps({"name": name, "inputs": inputs}, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)
    return sha256(text.encode("utf-8"))


def blob_entry(data: bytes) -> dict:
    """``{"utf8": text}`` when the bytes are UTF-8 (readable in review), else ``{"b64": ...}``."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return {"b64": base64.b64encode(data).decode("ascii")}
    return {"utf8": text}


def blob_bytes(entry: dict) -> bytes:
    if set(entry) == {"utf8"}:
        return entry["utf8"].encode("utf-8")
    if set(entry) == {"b64"}:
        return base64.b64decode(entry["b64"], validate=True)
    raise ValueError(f"a blob entry holds exactly one of 'utf8' or 'b64', not {sorted(entry)}")


def parse_receipt(raw: bytes) -> Any:
    """The receipt rule of the set: ``json.loads`` over the bytes decoded as UTF-8."""
    return json.loads(raw.decode("utf-8"))


def normal(value: Any) -> Any:
    """A live outcome as the vector file stores it (a JSON round trip)."""
    return json.loads(json.dumps(value))


def _raised(exc: BaseException, at: str) -> dict:
    out = {"raises": type(exc).__name__, "at": at}
    if isinstance(exc, GateSpecError):
        # the message is pinned only for the scorers' own refusals; a Python error's text may move
        # between interpreter versions, its class does not
        out["message"] = str(exc)
    return out


def frozen_leg(prereg: bytes, receipt: Any) -> dict:
    try:
        text = prereg.decode("utf-8")
    except UnicodeDecodeError as e:
        return _raised(e, "decode")
    try:
        block = _select_gates_block(text)
    except Exception as e:                                  # noqa: BLE001 - the class is the outcome
        return _raised(e, "select")
    try:
        s = score_frozen(block, receipt)
    except Exception as e:                                  # noqa: BLE001
        return _raised(e, "score")
    return normal({"verdict": s["verdict"], "gates": s["gates"], "gates_sha256": s["gates_sha256"]})


def experiment_leg(path: Path, receipt: Any) -> dict:
    try:
        exp = Experiment(path)
    except Exception as e:                                  # noqa: BLE001
        return _raised(e, "init")
    try:
        v = exp.score(receipt, smoke=bool(receipt.get("smoke")))
    except Exception as e:                                  # noqa: BLE001
        return _raised(e, "score")
    return normal({"verdict": v.verdict, "gates": v.gates, "gates_sha256": v.gates_sha256})


def agree(a: dict, b: dict) -> bool:
    """Two leg outcomes agree when both refuse (any class, any stage) or both score the same
    verdict, gates and gates digest."""
    if "raises" in a or "raises" in b:
        return "raises" in a and "raises" in b
    return all(a[k] == b[k] for k in ("verdict", "gates", "gates_sha256"))


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True)


def build_experiment_repo(root: Path, files: Sequence[Tuple[str, bytes]]) -> List[Path]:
    """Write each (file name, bytes) into its own directory under *root*, commit them all in one
    commit of a new repository, and return the paths in order."""
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    paths = []
    for i, (name, data) in enumerate(files):
        d = root / f"v{i:03d}"
        d.mkdir()
        p = d / name
        p.write_bytes(data)
        paths.append(p)
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "the PREREGs of the vector set")
    return paths


def brute_force(gates: Sequence[str], rows: Sequence[Dict[str, Any]]) -> dict:
    """Every assignment of *gates* walked through *rows*: counts of holes and of landings per row."""
    landings = [0] * len(rows)
    holes = 0
    for bits in range(1 << len(gates)):
        fired = {g: bool(bits >> i & 1) for i, g in enumerate(gates)}
        for i, when in enumerate(rows):
            if all(fired.get(k) == v for k, v in when.items()):
                landings[i] += 1
                break
        else:
            holes += 1
    return {"total": holes == 0, "holes": holes, "landings": landings,
            "unreachable_rows": [i for i, c in enumerate(landings) if c == 0]}


def matches(assignment: Dict[str, bool], when: Dict[str, Any]) -> bool:
    return all(assignment.get(k) == v for k, v in when.items())
