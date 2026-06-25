"""rigor_gate — the discipline made structural. Scans committed result JSONs for OVERCLAIMS: a verdict that
asserts a strong positive result ("robust / significant / real / proven / confirmed / undeniable / wins")
WITHOUT any attached uncertainty quantification (CI, bootstrap, permutation p) or an explicit disclosure
(corrigendum / hedged verdict). If a finding claims a win, it must show error bars.

This makes "the lab that doesn't overclaim" true BY CONSTRUCTION, not by vigilance. Today (2026-06-24) two
overclaims (genmatch_xvendor 'RESIDUAL ROBUST' no-CI, crossfamily post-hoc floor) had shipped to the public
record and were caught only by a hand-run adversarial pass. This gate would have blocked them.

  python scripts/rigor_gate.py            # report flags (exit 1 if any)
  python scripts/rigor_gate.py --list     # also list files that passed
Used by tests/test_rigor_gate.py to BLOCK the build on any overclaim.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = [ROOT / "papers", ROOT / "benchmarks" / "data"]

# A verdict using any of these asserts a strong positive result...
STRONG = re.compile(r"\b(robust|significant|real vendor|proven|confirmed|undeniable|definitive|"
                    r"established|generalizes|holds across|is a (?:real|genuine))\b", re.I)
# ...UNLESS it is already hedged (these make a 'robust' read non-assertive, e.g. 'NOT robust', 'robust? no')...
HEDGED = re.compile(r"\b(inconclusive|underpowered|null|partial|within noise|not above|surface-artifact|"
                    r"weak|fail(?:ed|s)?|cannot|does not|did not|no transfer|exploratory|caveat|over(?:claim|stat))\b", re.I)
# ...and the FILE must carry uncertainty quantification OR an explicit disclosure to back/exempt the claim.
EVIDENCE = re.compile(r"(\bci\b|confidence interval|bootstrap|permutation|perm[_ -]?p|p\s*=|p_value|pvalue|"
                      r"_ci\b|interval|std[_ ]?err|stderr|wilson|corrigendum|post_hoc|sanity_ok|rigor_gate_ok)", re.I)

# keys whose string values are treated as the finding's OWN verdict (not dataset/item content).
# Deliberately narrow: "claim"/"summary"/"text" often hold corpus items, not conclusions.
CLAIM_KEYS = re.compile(r"^verdict$|_verdict$|conclusion|final_conclusion", re.I)


def _claim_strings(obj, parent_key=""):
    """Yield (key, string) for string values under claim-ish keys (recursive)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and CLAIM_KEYS.search(str(k)):
                yield str(k), v
            else:
                yield from _claim_strings(v, str(k))
    elif isinstance(obj, list):
        for v in obj:
            yield from _claim_strings(v, parent_key)


def scan_file(p: Path):
    """Return list of (key, verdict_snippet) that overclaim, or [] if clean / unparseable."""
    try:
        raw = p.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception:
        return []
    file_has_evidence = bool(EVIDENCE.search(raw))   # uncertainty/disclosure anywhere in the file
    flags = []
    for k, s in _claim_strings(data):
        if STRONG.search(s) and not HEDGED.search(s) and not file_has_evidence:
            flags.append((k, s[:140]))
    return flags


def main(argv):
    show_pass = "--list" in argv
    result_files = sorted({f for d in SCAN_DIRS if d.exists()
                           for f in d.rglob("*.json") if "result" in f.name.lower()})
    flagged, passed = [], 0
    for f in result_files:
        fl = scan_file(f)
        if fl:
            flagged.append((f, fl))
        else:
            passed += 1
    print(f"rigor-gate: scanned {len(result_files)} result JSON(s) — {passed} clean, {len(flagged)} FLAGGED\n")
    for f, fl in flagged:
        print(f"  OVERCLAIM  {f.relative_to(ROOT)}")
        for k, s in fl:
            print(f"      [{k}] {s}")
        print(f"      -> add a CI / permutation-p / bootstrap, or hedge the verdict, or a corrigendum.")
    if show_pass and not flagged:
        print("  all clear — every strong-claim verdict carries uncertainty quantification or disclosure.")
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
