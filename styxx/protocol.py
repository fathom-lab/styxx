# -*- coding: utf-8 -*-
"""styxx.protocol — the research loop as enforceable machinery.

The witness harnesses the program's *instruments*; this harnesses its *process*. The
prereg → frozen-gates → scored-run → verdict discipline that styxx cycles perform by
convention becomes machinery an agent cannot quietly bend:

  1. **Prereg-before-data.** ``Experiment(prereg=...)`` refuses to score unless the prereg
     file is committed in git history (not merely on disk) — the freeze is checked against
     the repository, not the agent's word.
  2. **Gates parse from the frozen document.** The prereg embeds a fenced ```gates json
     block; the scorer reads bars from the committed text. There is no API to pass a bar at
     scoring time — a bar that isn't in the frozen document does not exist.
  3. **Verdicts are mechanical.** The outcome table is part of the gates block; ``score()``
     evaluates gate expressions against the result dict and walks the table. The agent
     reports the verdict; it does not choose it.
  4. **Smoke is INVALID-only.** ``score(smoke=True)`` always returns the INVALID smoke
     verdict regardless of numbers — a smoke that looks good licenses nothing, by type.
  5. **Tamper check.** The gates block's sha256 is returned with every verdict; re-scoring
     against an edited prereg produces a hash mismatch against the recorded one.

Format — a prereg embeds one fenced block::

    ```gates
    {"gates": {"G0": {"metric": "llama_top1", "op": ">=", "value": 0.29},
               "G1": {"metric": "gemma_top1", "op": ">=", "value": 0.143}},
     "outcomes": [{"when": {"G0": false}, "verdict": "INVALID__pipeline_broken"},
                  {"when": {"G0": true, "G1": true}, "verdict": "DOOR_OPENS"},
                  {"when": {"G0": true, "G1": false}, "verdict": "CLOSED_NEGATIVE"}],
     "smoke_verdict": "INVALID__smoke_plumbing_only"}
    ```

Metrics are keys of the result dict (dots traverse nesting). Ops: >=, <=, >, <, ==.
Outcomes are evaluated in order; the first row whose ``when`` matches wins; no row matching
is itself an error (the frozen table must be total — a partial table is a design bug the
harness surfaces instead of guessing).
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Experiment", "Verdict", "PrologueError", "GateSpecError"]

_GATES_RE = re.compile(r"```gates\s*\n(.*?)\n```", re.S)
_OPS = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b, "<": lambda a, b: a < b,
        "==": lambda a, b: a == b}


class PrologueError(RuntimeError):
    """The prereg is not committed / not found — scoring is refused."""


class GateSpecError(ValueError):
    """The frozen gates block is missing, malformed, or not total."""


@dataclass
class Verdict:
    verdict: str
    gates: dict                 # gate name -> bool
    gates_sha256: str           # hash of the frozen gates block text
    prereg_commit: str          # the earliest commit containing the prereg
    smoke: bool = False


def _resolve(result: dict, dotted: str):
    obj = result
    for k in dotted.split("."):
        if not isinstance(obj, dict) or k not in obj:
            raise GateSpecError(f"metric {dotted!r} not present in result")
        obj = obj[k]
    return obj


class Experiment:
    """One preregistered experiment, scored only on the frozen document's terms."""

    def __init__(self, prereg: str | Path, repo_root: str | Path | None = None):
        self.prereg = Path(prereg)
        self.repo_root = Path(repo_root) if repo_root else self.prereg.resolve().parent
        if not self.prereg.exists():
            raise PrologueError(f"prereg not found: {self.prereg}")
        self.prereg_commit = self._committed_at()
        text = self.prereg.read_text(encoding="utf-8")
        m = _GATES_RE.search(text)
        if not m:
            raise GateSpecError("prereg has no ```gates block — nothing frozen to score against")
        self._gates_text = m.group(1)
        self.gates_sha256 = hashlib.sha256(self._gates_text.encode("utf-8")).hexdigest()
        try:
            spec = json.loads(self._gates_text)
        except json.JSONDecodeError as e:
            raise GateSpecError(f"gates block is not valid JSON: {e}") from e
        for key in ("gates", "outcomes", "smoke_verdict"):
            if key not in spec:
                raise GateSpecError(f"gates block missing {key!r}")
        self.spec = spec

    # -- the freeze check --------------------------------------------------

    def _committed_at(self) -> str:
        """Earliest commit hash containing the prereg file; refuses if none."""
        try:
            r = subprocess.run(
                ["git", "log", "--diff-filter=A", "--format=%H", "--follow", "--",
                 str(self.prereg.name)],
                cwd=self.prereg.resolve().parent, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise PrologueError(f"cannot verify the freeze against git: {e}") from e
        hashes = [h for h in r.stdout.split() if h]
        if r.returncode != 0 or not hashes:
            raise PrologueError(
                f"{self.prereg.name} is not committed — a prereg on disk is a draft, "
                "not a freeze; scoring is refused")
        return hashes[-1]

    # -- scoring -----------------------------------------------------------

    def score(self, result: dict, smoke: bool = False) -> Verdict:
        """Evaluate the frozen gates against a result dict; walk the frozen outcome table."""
        if smoke:
            return Verdict(verdict=self.spec["smoke_verdict"], gates={},
                           gates_sha256=self.gates_sha256,
                           prereg_commit=self.prereg_commit, smoke=True)
        fired: dict[str, bool] = {}
        for name, g in self.spec["gates"].items():
            op = _OPS.get(g.get("op"))
            if op is None:
                raise GateSpecError(f"gate {name!r}: unknown op {g.get('op')!r}")
            fired[name] = bool(op(_resolve(result, g["metric"]), g["value"]))
        for row in self.spec["outcomes"]:
            if all(fired.get(k) == v for k, v in row["when"].items()):
                return Verdict(verdict=row["verdict"], gates=fired,
                               gates_sha256=self.gates_sha256,
                               prereg_commit=self.prereg_commit)
        raise GateSpecError(
            f"no outcome row matches gates {fired} — the frozen table is not total; "
            "this is a prereg design bug, surfaced instead of guessed around")
