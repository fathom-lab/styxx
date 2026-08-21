# -*- coding: utf-8 -*-
"""Where does the edge screen stop finding things? Instrumented funnel.

Not a result. A diagnostic, run because the screen fires 3/3 on hand-written
cases and 0 on 227 real files, and exactly one of those two facts is about the
world. Prints the count surviving each of the five prereg requirements so the
binding constraint is visible instead of guessed at.
"""
from __future__ import annotations

import ast
import sys
from collections import Counter
from pathlib import Path

from styxx import edges as E


def main(root: str = "styxx") -> int:
    root = Path(root)
    files = E._files(root, skip_tests=True)
    trees = {}
    for p in files:
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
            trees[p] = (ast.parse(src), src.splitlines())
        except Exception:
            pass

    seen, dupes, defined = {}, set(), {}
    for p, (tree, _) in trees.items():
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defined[n.name] = defined.get(n.name, 0) + 1
        v = E._ProducerVisitor(str(p))
        v.visit(tree)
        for prod in v.found:
            if prod.name in seen:
                dupes.add(prod.name)
            seen[prod.name] = prod
    producers = {k: v for k, v in seen.items() if k not in dupes}

    print(f"files                                  {len(trees)}")
    print(f"functions defined                      {len(defined)}")
    print(f"PRODUCERS (absence -> constant)        {len(producers)}"
          f"   ({len(dupes)} dropped as ambiguous)")
    print(f"  of those, with a computed return too {sum(1 for p in producers.values() if p.n_computed_returns)}")

    # walk consumers with counters at each requirement
    stage = Counter()
    examples = {"decision": [], "contrast": [], "defended": [], "polarity": []}

    class Probe(E._ConsumerScan):
        def _decide(self, test, body, orelse, binds):
            for target, p in self._targets(test, binds):
                stage["1_decision_on_producer"] += 1
                if len(examples["decision"]) < 6:
                    examples["decision"].append(
                        f"{Path(self.path).name}:{getattr(test,'lineno',0)} {p.name}")
                if p.n_computed_returns == 0:
                    stage["2a_killed_by_CONTRAST"] += 1
                    continue
                stage["2b_passed_contrast"] += 1
                for line, k, why in p.absence_returns:
                    if E._is_defended(k):
                        stage["3a_killed_by_DEFENDED"] += 1
                        continue
                    stage["3b_passed_indistinguishable"] += 1
                    r = E._resolves_to_quiet(k, test, target, body, orelse)
                    if r is None:
                        stage["4a_killed_by_POLARITY"] += 1
                        if len(examples["polarity"]) < 8:
                            lb = E._loud_evidence(body) or "-"
                            le = E._loud_evidence(orelse) or "-"
                            examples["polarity"].append(
                                f"{Path(self.path).name}:{getattr(test,'lineno',0)} "
                                f"{p.name}()->{k!r}  if-branch[{lb}] else-branch[{le}]")
                        continue
                    stage["5_FLAGGED"] += 1

    for p, (tree, lines) in trees.items():
        Probe(str(p), producers, lines, defined).run(tree)

    print("\nFUNNEL")
    for k in sorted(stage):
        print(f"  {k:34} {stage[k]}")

    print("\nsample decisions reached:")
    for e in examples["decision"]:
        print("   ", e)
    print("\nsample kills at requirement 4 (polarity from the consumer):")
    for e in examples["polarity"]:
        print("   ", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "styxx"))
