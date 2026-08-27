"""CENSUS — what a VACUOUS_PASS detector would REACH, and what it would DESTROY.

**A census. It licenses no claim and proposes no detector.** It measures candidate syntactic
patterns over the population a rule would actually see, BEFORE anything is designed — which is the
one thing the v0.12 cycle died for not doing. That cycle froze a bar against a line-level census
and then specified a span-level clause; the populations differed and the clause under-reached.

## The class being sized

`RECON_vacuous_pass_2026_08_27.md` catalogues five instances of SUCCESS BY EMPTY POPULATION in
this lab's own verification machinery, none of which `styxx.absence` or `styxx.loops` detects. The
shape: the thing to be checked is filtered away upstream, and a downstream check then passes over
nothing, reporting the same green it would report on a full population.

## Why the DESTROY column decides this

`all()` over a possibly-empty collection is not a bug. `if not xs: return` is not a bug. Both are
overwhelmingly correct, everywhere, in every codebase. A rule that flags them all would bury five
real defects under hundreds of correct guards, which is the broad-detector catastrophe the v0.11
census killed three designs with and the v0.12 census killed four more.

So each candidate below is scored on two numbers, and the second is the one that kills:

    reaches   how many of the known vacuous-pass sites it flags
    sites     how many times it fires across the repository at all

A candidate that reaches 3 and fires 400 times is not a detector, it is a highlighter.

## The honest ceiling, stated before the numbers

Only three of the five catalogued instances have a *syntactic* defect site at all. VP-C is a
classification decision (a mismatch recorded as absence) and VP-E is a data fact (a receipt whose
content changed). No AST pattern reaches either, and a census that quietly scored itself out of
5 rather than out of 3 would be flattering by choice of denominator.

  python papers/closed-model-frontier/vacuous_pass_population_census.py
"""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

OUT = HERE / "vacuous_pass_population_census.json"

# The catalogued instances, with the line their defect actually sits on. `syntactic` says whether
# the defect HAS a code site a pattern could reach — declared here so the denominator is honest.
KNOWN = [
    {"id": "VP-A", "commit": "HEAD", "path": "tests/test_ledger.py", "line": 25, "syntactic": True,
     "shape": "pytest.skip() on a shallow clone; a skipped test reads green in CI"},
    {"id": "VP-B", "commit": "cbd2864", "path": "tests/test_certificate_reproduces.py", "line": 56, "syntactic": True,
     "shape": "generator yields only fully-resolvable documents; the rest are dropped"},
    {"id": "VP-C", "commit": "cbd2864", "path": "styxx/corpus_audit.py", "line": 85, "syntactic": False,
     "shape": "a sha mismatch recorded as absence — a classification, not a syntactic guard"},
    {"id": "VP-D", "commit": "9d09ef3",
     "path": "papers/closed-model-frontier/run_oath_v11_battery.py", "line": 383,
     "syntactic": True,
     "shape": "all() over a population that can be empty -- `bar_ii = all(... for c in fresh)`"},
    {"id": "VP-E", "commit": None, "path": None, "line": None, "syntactic": False,
     "shape": "a receipt present in the tree whose content changed — a data fact, not code"},
]
WINDOW = 6


class Candidate(ast.NodeVisitor):
    """Each rule is deliberately naive. The point is to measure what naive costs."""

    def __init__(self, rule: str):
        self.rule = rule
        self.hits: list = []

    # C1 — all()/any() whose argument is a comprehension: true over an empty population.
    def visit_Call(self, node):
        f = node.func
        name = getattr(f, "id", None) or getattr(f, "attr", None)
        if self.rule == "C1_all_any_over_comprehension":
            if name in ("all", "any") and node.args and isinstance(
                    node.args[0], (ast.GeneratorExp, ast.ListComp, ast.SetComp)):
                self.hits.append(node.lineno)
        elif self.rule == "C4_swallowing_skip_call":
            if name == "skip":
                self.hits.append(node.lineno)
        self.generic_visit(node)

    # C2 — `if not <name>:` whose body only exits. The guard-drop shape.
    def visit_If(self, node):
        if self.rule == "C2_if_not_x_then_exit":
            t = node.test
            if isinstance(t, ast.UnaryOp) and isinstance(t.op, ast.Not):
                if all(isinstance(s, (ast.Return, ast.Continue, ast.Break, ast.Pass))
                       for s in node.body):
                    self.hits.append(node.lineno)
        self.generic_visit(node)

    # C3 — a comprehension carrying an `if` filter: silently drops what it does not match.
    def visit_comprehension(self, node):
        if self.rule == "C3_filtered_comprehension" and node.ifs:
            self.hits.append(getattr(node.iter, "lineno", 0))
        self.generic_visit(node)

    # C5 — `except: pass|continue`: a failure absorbed with no record.
    def visit_ExceptHandler(self, node):
        if self.rule == "C5_except_pass_or_continue":
            if all(isinstance(s, (ast.Pass, ast.Continue)) for s in node.body):
                self.hits.append(node.lineno)
        self.generic_visit(node)


RULES = ["C1_all_any_over_comprehension", "C2_if_not_x_then_exit",
         "C3_filtered_comprehension", "C4_swallowing_skip_call",
         "C5_except_pass_or_continue"]


def scan(src: str, rule: str) -> list:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    v = Candidate(rule)
    v.visit(tree)
    return sorted(set(v.hits))


def source_at(commit: str, path: str) -> str:
    """The defect's source AS IT WAS, not as it is now.

    The first run of this census scanned the CURRENT tree for lines recorded from PRE-FIX code.
    Three of the four defects have since been repaired, so their lines no longer hold what the
    line numbers point at, and the reach column read as zeroes. That is measuring against the
    wrong population — the exact error v0.12 died of, committed inside the census written to
    stop it happening again. Reach is now read from the commit each defect was catalogued at;
    the DESTROY column still comes from the current tree, because that is the population a rule
    deployed today would actually see.
    """
    r = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.stdout


def python_files() -> list:
    out = []
    for p in sorted(ROOT.rglob("*.py")):
        parts = p.parts
        if any(x in parts for x in (".claude", ".venv-71-smoke", "build", "dist",
                                    "__pycache__", "anc")):
            continue
        out.append(p)
    return out


def main() -> int:
    files = python_files()
    sources = {}
    for p in files:
        try:
            sources[p] = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

    syntactic = [k for k in KNOWN if k["syntactic"]]
    rows = []
    for rule in RULES:
        sites = 0
        for p, src in sources.items():
            sites += len(scan(src, rule))
        reached = []
        for k in syntactic:
            src = source_at(k["commit"], k["path"])
            if not src:
                continue
            if any(abs(ln - k["line"]) <= WINDOW for ln in scan(src, rule)):
                reached.append(k["id"])
        rows.append({"rule": rule, "sites_repo_wide": sites,
                     "reaches": reached, "n_reached": len(reached),
                     "cost_per_reach": (round(sites / len(reached), 1) if reached else None)})

    payload = {
        "census": "vacuous pass — what a detector would reach and what it would destroy",
        "status": "CENSUS. Licenses no claim, proposes no detector. Measures the population a "
                  "rule would actually see, BEFORE any design — the step v0.12 died for skipping.",
        "verifier_sha256":
            hashlib.sha256((ROOT / "styxx" / "certify.py").read_bytes()).hexdigest(),
        "python_files_scanned": len(sources),
        "window_lines": WINDOW,
        "line_correction": "VP-D was first recorded at line 329, taken from the audit finding's "
                           "header without checking it. At the catalogued commit line 329 is "
                           "`if not PANEL.exists():`; the vacuous bars are at 382-384. Corrected "
                           "to 383 after reading the source. A census that trusts a cited line "
                           "number is measuring the citation, not the code.",
        "measurement_note": "REACH is measured on the pre-fix source at each defect's catalogued "
                            "commit; DESTROY is measured on the current tree. The first run of "
                            "this census read both from the current tree and scored zeroes, "
                            "because three of the four defects are repaired there — measuring "
                            "against the wrong population, inside the census written to stop "
                            "exactly that.",
        "denominator": {
            "catalogued_instances": len(KNOWN),
            "with_a_syntactic_defect_site": len(syntactic),
            "note": "Scored out of the SYNTACTIC instances only. VP-C is a classification "
                    "decision and VP-E is a data fact; no AST pattern reaches either, and "
                    "scoring out of five would flatter by choice of denominator.",
        },
        "candidates": rows,
        "how_to_read_this": "`sites_repo_wide` is the destroy surface: every one of those is a "
                            "guard a rule would flag, and the overwhelming majority are correct "
                            "code. `cost_per_reach` is sites divided by instances reached. A "
                            "candidate in the hundreds per reach is a highlighter, not a "
                            "detector, and no threshold rescues it — that is the finding v0.11 "
                            "and v0.12 both paid for.",
        "what_this_does_not_show": [
            "That any candidate is good. None is proposed and none is recommended.",
            "That a hand-adjudicated subset of the sites would be mostly false positives — "
            "nobody has read them. The destroy column is a COUNT, not a judgement.",
            "That the class is five. Five is five, found in two days by people looking for "
            "other things.",
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"scanned {len(sources)} python files; scoring out of {len(syntactic)} syntactic "
          f"instances of {len(KNOWN)} catalogued\n")
    print(f"{'candidate':<34}{'sites':>8}{'reaches':>9}   {'cost/reach':>10}")
    for r in rows:
        print(f"{r['rule']:<34}{r['sites_repo_wide']:>8}{r['n_reached']:>9}   "
              f"{str(r['cost_per_reach']):>10}   {','.join(r['reaches'])}")
    print(f"\n-> {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
