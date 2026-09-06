"""Read the suite-power receipt and report the two things a rate cannot say.

A kill rate answers "how much of this layer is defended". It cannot answer either of the questions
somebody about to change the code actually has:

  1. WHERE is it undefended — which specific behaviours would ship broken and green;
  2. HOW THINLY is the rest defended — a mutant killed by ONE assertion in ONE file is one deleted
     test away from joining the survivors, and that fragility is invisible in a percentage.

S5 of the frozen spec records the failing test ids for every kill precisely so the second question
can be asked. This asks it.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def main(argv=None):
    receipt = Path(argv[0]).resolve() if argv else ROOT / "conformance/sworn/suite_power.json"
    R = json.loads(receipt.read_text(encoding="utf-8"))

    print("=" * 88)
    print("receipt: %s   VOID=%s" % (receipt.name, R["void"]))
    print("oracle : %s" % ", ".join(Path(t).name for t in R["oracle"]["tests"]))
    print()
    for k, g in sorted(R["gates"].items()):
        print("  %-4s %-46s bar=%-12s pass=%s"
              % (k, str(g["value"])[:46], str(g.get("bar", ""))[:12], g.get("pass")))
    print()
    print("counts:", json.dumps(R["counts"]))
    print()

    muts = R["mutations"]
    viable = [m for m in muts if m["verdict"] in ("killed", "survived") and not m.get("control")]
    killed = [m for m in viable if m["verdict"] == "killed"]
    survived = [m for m in viable if m["verdict"] == "survived"]

    print("kill rate: %d/%d = %.1f%%"
          % (len(killed), len(viable), 100.0 * len(killed) / max(1, len(viable))))
    print()
    print("by layer:")
    for layer in sorted({m.get("layer") for m in viable}):
        v = [m for m in viable if m.get("layer") == layer]
        k = [m for m in v if m["verdict"] == "killed"]
        print("  %-10s %2d/%2d  %5.1f%%" % (layer, len(k), len(v), 100.0 * len(k) / len(v)))

    # ---- how thin is the defence where it exists? -------------------------------------------
    print()
    thin = Counter()
    for m in killed:
        n = m.get("killed_by_count", 0)
        thin["1 test" if n == 1 else
             "2-5 tests" if n <= 5 else
             "6-25 tests" if n <= 25 else
             "over 25 tests"] += 1
    print("how many tests caught each kill:")
    for k in ("1 test", "2-5 tests", "6-25 tests", "over 25 tests"):
        if thin[k]:
            print("   %-14s %d" % (k, thin[k]))

    single = sorted((m for m in killed if m.get("killed_by_count", 0) == 1),
                    key=lambda m: m.get("layer", ""))
    if single:
        print()
        print("KILLED BY EXACTLY ONE TEST — one deleted assertion from becoming a survivor:")
        for m in single:
            print("   [%s] %s" % (m.get("layer"), m["name"]))
            print("        %s" % (m.get("killed_by") or ["?"])[0])

    # ---- who does the defending? --------------------------------------------------------------
    files = Counter()
    for m in killed:
        for t in m.get("killed_by", []):
            files[t.split("::")[0]] += 1
    print()
    print("which test files did the killing (a mutant may be counted in several):")
    for f, n in files.most_common():
        print("   %3d  %s" % (n, f))

    # ---- the result --------------------------------------------------------------------------
    print()
    print("=" * 88)
    print("THE SURVIVOR LIST (%d) — every one is a place a defect would ship green" % len(survived))
    for m in sorted(survived, key=lambda m: m.get("layer", "")):
        print()
        print("  [%s] %s" % (m.get("layer"), m["name"]))
        print("        why: %s" % str(m.get("why", ""))[:300])

    excluded = [m for m in muts
                if m["verdict"] not in ("killed", "survived") and not m.get("control")]
    if excluded:
        print()
        print("excluded from the denominator (%d):" % len(excluded))
        for m in excluded:
            print("   %-14s %s" % (m["verdict"], m["name"][:64]))

    controls = [m for m in muts if m.get("control")]
    bad = [m for m in controls if m["verdict"] == "killed"]
    print()
    print("controls: %d, killed %d (must be 0)" % (len(controls), len(bad)))
    for m in bad:
        print("   KILLED CONTROL:", m["name"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
