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
import math
import json
import numbers
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Experiment", "Verdict", "PrologueError", "GateSpecError",
           "undeclared_power_gates"]


def undeclared_power_gates(prereg) -> list:
    """Gate names in *prereg* carrying no usable ``power_basis`` — corpus-auditable.

    Frozen as a deliverable in ``PREREG_protocol_power_basis_2026_08_07.md`` item 4 and silently
    dropped from the implementation; the red team caught the omission, not the exam.
    """
    return Experiment(prereg).undeclared_power_gates

_GATES_RE = re.compile(r"```gates\s*\n(.*?)\n```", re.S)
# Any fence-opener line whose info string could read as "gates" to a HUMAN: backtick or tilde
# fences, any indent. The info string is normalized (Unicode format chars stripped, casefolded)
# before comparison, so ```GATES, ~~~gates and a zero-width-corrupted ```ga​tes all count.
_FENCE_OPENER_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})[ \t]*(\S*)", re.M)   # >3 spaces = not a
# fence to a CommonMark renderer, so not to this count either; a 4-space-indented decoy that the
# strict extractor might still match is caught by the matches-count check below instead.


def _gates_like_fences(text: str) -> list[str]:
    """Every fence-opener whose normalized info string is 'gates' — the human's view, not the
    regex's. Verification pass F1 (2026-08-09): the multiplicity guard originally counted only
    exact-lowercase-backtick fences, so a ~~~gates block could show a reader the honest
    declaration while the machine's single ```gates match was a decoy hidden in an HTML comment.
    The guard must be over blocks a human could read as gates, not over exact matches."""
    import unicodedata
    out = []
    for m in _FENCE_OPENER_RE.finditer(text):
        info = "".join(ch for ch in m.group(2) if unicodedata.category(ch) != "Cf").casefold()
        if info == "gates":
            out.append(m.group(0).strip())
    return out
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
    power_basis: dict = field(default_factory=dict)      # gate -> how its bar was derived
    undeclared_power_gates: list = field(default_factory=list)
    metric_paths: dict = field(default_factory=dict)     # gate -> the dotted path it read


def _resolve(result: dict, dotted: str):
    obj = result
    for k in dotted.split("."):
        if not isinstance(obj, dict) or k not in obj:
            raise GateSpecError(f"metric {dotted!r} not present in result")
        obj = obj[k]
    return obj


class Experiment:
    """One preregistered experiment, scored only on the frozen document's terms."""

    def __init__(self, prereg: str | Path, repo_root: str | Path | None = None,
                 require_power_basis: bool = False):
        self.require_power_basis = require_power_basis
        self.prereg = Path(prereg)
        self.repo_root = Path(repo_root) if repo_root else self.prereg.resolve().parent
        if not self.prereg.exists():
            raise PrologueError(f"prereg not found: {self.prereg}")
        self.prereg_commit = self._committed_at()
        text = self.prereg.read_text(encoding="utf-8")
        # The authority question, settled before any parsing: how many blocks could a HUMAN read
        # as the gates block? Counted on the ORIGINAL text (an HTML comment hides a fence from a
        # renderer but not from this count) and normalized (tilde fences, any case, zero-width
        # chars — verification pass F1 broke the exact-match version of this guard with all
        # three). Anything other than exactly one, in the one supported form, refuses.
        gates_like = _gates_like_fences(text)
        if not gates_like:
            raise GateSpecError("prereg has no ```gates block — nothing frozen to score against")
        if len(gates_like) > 1:
            raise GateSpecError(
                f"prereg contains {len(gates_like)} gates-like fenced blocks ({gates_like}) — "
                f"the frozen block must be unambiguous. A document that can show a reader one "
                f"block and score against another is not frozen; remove or rename all but one.")
        stripped = re.sub(r"<!--.*?-->", "", text, flags=re.S)
        matches = _GATES_RE.findall(stripped)
        if len(matches) != 1 or gates_like[0] != "```gates":
            raise GateSpecError(
                f"the single gates-like block ({gates_like[0]!r}) is not a plain unindented "
                f"```gates fence outside HTML comments — the one form both a renderer and this "
                f"parser read identically. A block only one of them can see is a shadowing "
                f"channel, not a style choice.")
        self._gates_text = matches[0]
        self.gates_sha256 = hashlib.sha256(self._gates_text.encode("utf-8")).hexdigest()

        def _no_dup_keys(pairs):
            # Red team 2026-08-09 (v4 audit, D2): json.loads silently keeps the LAST duplicate
            # key, so '"excluding": "real", "excluding": "decoy"' displayed both while the
            # machine honoured only the decoy. Corpus measured clean; refusal rewrites nothing.
            seen = {}
            for k, v in pairs:
                if not k.isascii():
                    # Verification pass F2 (2026-08-09): "exсluding" with a Cyrillic с is a
                    # DIFFERENT key to json and the same word to a human — two live
                    # declarations, machine honours one. Byte-equality dup detection cannot see
                    # it; an ASCII allowlist can. No committed gates block has non-ASCII keys.
                    raise GateSpecError(
                        f"non-ASCII key {k!r} in the gates block — a key a human cannot "
                        f"distinguish from an ASCII one is a shadowing channel. Keys must be "
                        f"ASCII.")
                if k in seen:
                    raise GateSpecError(
                        f"duplicate key {k!r} in the gates block — a block that declares the "
                        f"same key twice shows a reader both and honours only the last, which "
                        f"is a shadowing channel, not a typo to forgive")
                seen[k] = v
            return seen

        try:
            spec = json.loads(self._gates_text, object_pairs_hook=_no_dup_keys)
        except json.JSONDecodeError as e:
            raise GateSpecError(f"gates block is not valid JSON: {e}") from e
        for key in ("gates", "outcomes", "smoke_verdict"):
            if key not in spec:
                raise GateSpecError(f"gates block missing {key!r}")
        self.spec = spec
        self.power_basis = {n: g.get("power_basis") for n, g in spec["gates"].items()}
        self.metric_paths = {n: g.get("metric") for n, g in spec["gates"].items()}
        _bad = sorted(n for n, m in self.metric_paths.items() if not isinstance(m, str) or not m)
        if _bad:
            raise GateSpecError(
                f"gates with a missing or non-string 'metric' path: {_bad}. Left unchecked this "
                f"put None into metric_paths and made check_metrics() raise AttributeError — the "
                f"pre-run safety tool crashing on the most mis-specified gate there is.")
        self.metric_means = {n: g.get("metric_means") for n, g in spec["gates"].items()}

        # -- v4: declared gate composition ------------------------------------------------------
        # E1 (cycle 159): G1 judged the minimum over ALL candidates while G2 disqualified one of
        # them in the same run. Every component was individually correct; the composition was
        # wrong, and nothing here looked at relationships between gates. A gate whose metric is
        # an aggregate over a set may now declare that set:
        #   "agg": "min"|"max"  — which extremum the metric claims to be
        #   "over": path        — a dict of per-member values in the result
        #   "excluding": path   — optional; a list of member names to exclude first
        # score() recomputes the aggregate over the declared population minus the declared
        # exclusions and REFUSES if the quoted metric does not equal the recomputation. This
        # checks declared composition only — a ratchet, not a proof.
        self.composition = {}
        for n, g in spec["gates"].items():
            keys = {k: g.get(k) for k in ("agg", "over", "excluding") if k in g}
            if not keys:
                continue
            if "agg" not in keys or "over" not in keys:
                raise GateSpecError(
                    f"gate {n!r}: a composition declaration needs both 'agg' and 'over' "
                    f"(got {sorted(keys)}). Half a declaration checks nothing while looking "
                    f"like it checks something, which is worse than no declaration.")
            if keys["agg"] not in ("min", "max"):
                raise GateSpecError(
                    f"gate {n!r}: 'agg' must be \"min\" or \"max\", got {keys['agg']!r}")
            if not isinstance(keys["over"], str) or not keys["over"]:
                raise GateSpecError(f"gate {n!r}: 'over' must be a non-empty result path")
            if "excluding" in keys and (not isinstance(keys["excluding"], str)
                                        or not keys["excluding"]):
                raise GateSpecError(
                    f"gate {n!r}: 'excluding' must be a non-empty result path when present")
            self.composition[n] = keys
        self.undeclared_power_gates = sorted(
            n for n, v in self.power_basis.items()
            if not (isinstance(v, str) and v.strip()))   # " " and true are NOT declarations
        if self.require_power_basis and self.undeclared_power_gates:
            raise GateSpecError(
                f"gates without a declared power basis: {self.undeclared_power_gates}. "
                f"Each gate must carry \"power_basis\": how its bar was derived, or the literal "
                f"\"none — exploratory\". Three bars in this program were set without checking "
                f"whether any instrument could clear them (b37 G2, b48 G2, C5 G1); each was "
                f"written up and each recurred. An undeclared bar is now a refusal, not a note.")

    def check_metrics(self, result: dict) -> dict:
        """Resolve every gate's metric path against a candidate result WITHOUT scoring.

        Call this before launching a run. ``_resolve`` already raises on a missing path, but only
        at scoring time — after the compute is spent. B49 lost a whole re-analysis to a gate whose
        path resolved to a real but *wrong* field; this catches the cheaper cousin (a path that is
        simply absent) for the cost of one call.

        **It cannot catch B49's actual error.** A path that resolves to an existing field the
        author did not intend is indistinguishable from a correct one, because the machinery has
        no access to intent. ``metric_means`` records that intent for a human reader; it is not a
        check and must not be read as one.
        """
        import math
        out = {}
        smoke = bool(result.get("smoke"))
        for name, path in self.metric_paths.items():
            try:
                val = _resolve(result, path)
            except GateSpecError:
                out[name] = {"path": path, "present": False, "usable": False,
                             "note": ("result is a smoke run; smoke scores by type and never "
                                      "reads gate metrics" if smoke else "path not in result")}
                continue
            usable = isinstance(val, (int, float)) and not isinstance(val, bool)                 and math.isfinite(val)
            out[name] = {"path": path, "present": True, "usable": usable,
                         "note": None if usable else
                         f"resolves to {type(val).__name__}"
                         f"{' (NaN/inf)' if isinstance(val, float) else ''} — score() cannot "
                         f"compare it"}
        # v4 composition paths get the same pre-run resolution (red team D4: the pre-run safety
        # tool previously passed a result whose 'over' path was absent, and score() then refused
        # after the compute was spent — the exact failure mode this method exists to prevent).
        for name, c in self.composition.items():
            for kind in ("over", "excluding"):
                if kind not in c:
                    continue
                key = f"{name}:{kind}"
                try:
                    val = _resolve(result, c[kind])
                except GateSpecError:
                    out[key] = {"path": c[kind], "present": False, "usable": False,
                                "note": ("smoke run" if smoke else
                                         f"composition path ({kind}) not in result")}
                    continue
                want = dict if kind == "over" else list
                ok = isinstance(val, want) and (bool(val) if kind == "over" else True)
                out[key] = {"path": c[kind], "present": True, "usable": ok,
                            "note": None if ok else
                            f"composition path ({kind}) resolves to {type(val).__name__}; "
                            f"score() will refuse it"}
        return out

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

    def _check_composition(self, name: str, quoted: float, result: dict) -> None:
        """Recompute a declared aggregate and refuse if the quoted metric disagrees.

        The E1 defect made concrete: a metric quoting the unrestricted minimum cannot equal the
        eligible-restricted recomputation when the two differ, so declaring the population turns
        that mistake from a silent pass into a refusal. Everything ill-formed refuses — a
        composition check that guesses is a composition check that can be gamed.
        """
        c = self.composition[name]
        pop = _resolve(result, c["over"])
        if not isinstance(pop, dict) or not pop:
            raise GateSpecError(
                f"gate {name!r}: 'over' path {c['over']!r} must resolve to a non-empty dict of "
                f"per-member values, got {type(pop).__name__}")
        excluded: set = set()
        if "excluding" in c:
            exc = _resolve(result, c["excluding"])
            if not isinstance(exc, list):
                raise GateSpecError(
                    f"gate {name!r}: 'excluding' path {c['excluding']!r} must resolve to a "
                    f"list of member names, got {type(exc).__name__}")
            bad_names = [e for e in exc if not isinstance(e, str)]
            if bad_names:
                # A mixed-type list previously crashed the sorted(set-difference) below with a
                # raw TypeError, which a harness catching only GateSpecError records as a crash
                # rather than a refusal (red team D8).
                raise GateSpecError(
                    f"gate {name!r}: 'excluding' entries must be strings naming members of "
                    f"'over'; got {bad_names!r}")
            unknown = sorted(set(exc) - set(pop))
            if unknown:
                raise GateSpecError(
                    f"gate {name!r}: 'excluding' names members absent from 'over': {unknown}. "
                    f"An exclusion that excludes nothing hides a key mismatch between the two "
                    f"fields, and the check would silently pass on the full population.")
            excluded = set(exc)
        eligible = {k: v for k, v in pop.items() if k not in excluded}
        if not eligible:
            raise GateSpecError(
                f"gate {name!r}: every member of {c['over']!r} is excluded — an aggregate over "
                f"an empty population is not a measurement")
        vals = []
        for k, v in eligible.items():
            # numbers.Real admits numpy float32/int64 scalars in in-process receipts (red team
            # D9: they are finite numbers and were refused with a wrong diagnosis); bool is a
            # Real and stays banned.
            if isinstance(v, bool) or not isinstance(v, numbers.Real) \
                    or not math.isfinite(float(v)):
                raise GateSpecError(
                    f"gate {name!r}: member {k!r} of {c['over']!r} is {v!r}, which cannot be "
                    f"aggregated")
            vals.append(float(v))
        recomputed = min(vals) if c["agg"] == "min" else max(vals)
        if not math.isclose(float(quoted), recomputed, rel_tol=0.0, abs_tol=1e-12):
            # The comparison is exact (1e-12) by design: rounding forgiveness would readmit
            # near-miss shadowing. The convention this imposes (red team D5): store per-member
            # values at the same precision you quote the metric — round both or neither. A
            # runner quoting round(min, 4) over full-precision members refuses HERE, loudly,
            # before any verdict; that is the fail-safe direction and it is intentional.
            members = sorted(k for k, v in eligible.items()
                             if math.isclose(float(v), recomputed, rel_tol=0.0, abs_tol=1e-12))
            raise GateSpecError(
                f"gate {name!r}: COMPOSITION VIOLATION — metric quotes {quoted!r} but the "
                f"declared {c['agg']} over {c['over']!r}"
                + (f" excluding {sorted(excluded)}" if excluded else "")
                + f" recomputes to {recomputed!r} (attained by {members}). The quoted value "
                f"belongs to a population this prereg's own declarations rule out. This is the "
                f"E1 defect (cycle 159): a gate passing on a candidate another gate disqualified.")

    def score(self, result: dict, smoke: bool = False) -> Verdict:
        # power_basis rides on the Verdict as metadata; verdict STRINGS are untouched so every
        # committed seal keeps verifying byte-identically.
        """Evaluate the frozen gates against a result dict; walk the frozen outcome table."""
        if smoke:
            return Verdict(verdict=self.spec["smoke_verdict"], gates={},
                           gates_sha256=self.gates_sha256,
                           prereg_commit=self.prereg_commit, smoke=True,
                           power_basis=dict(self.power_basis),
                           undeclared_power_gates=list(self.undeclared_power_gates),
                           metric_paths=dict(self.metric_paths))
        fired: dict[str, bool] = {}
        for name, g in self.spec["gates"].items():
            op = _OPS.get(g.get("op"))
            if op is None:
                raise GateSpecError(f"gate {name!r}: unknown op {g.get('op')!r}")
            _v = _resolve(result, g["metric"])
            # numbers.Real for consistency with the composition member guard (verification pass
            # F4: the two guards disagreed on what a number is — np.float32 aggregated as a
            # member but refused as a quoted metric). bool stays banned; both refuse safely.
            if isinstance(_v, bool) or not isinstance(_v, numbers.Real) \
                    or not math.isfinite(float(_v)):
                raise GateSpecError(
                    f"gate {name!r}: metric {g['metric']!r} resolves to {_v!r}, which cannot be "
                    f"compared. A NaN previously made every comparison False and returned the "
                    f"frozen table's false branch as a SEALED verdict, with no refusal anywhere.")
            if name in self.composition:
                self._check_composition(name, _v, result)
            fired[name] = bool(op(_v, g["value"]))
        for row in self.spec["outcomes"]:
            if all(fired.get(k) == v for k, v in row["when"].items()):
                return Verdict(verdict=row["verdict"], gates=fired,
                               gates_sha256=self.gates_sha256,
                               prereg_commit=self.prereg_commit,
                               power_basis=dict(self.power_basis),
                               undeclared_power_gates=list(self.undeclared_power_gates),
                               metric_paths=dict(self.metric_paths))
        raise GateSpecError(
            f"no outcome row matches gates {fired} — the frozen table is not total; "
            "this is a prereg design bug, surfaced instead of guessed around")
