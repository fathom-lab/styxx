"""EXTERNAL-2 census: what the gate still accuses after EXTERNAL-1 withheld the path accusations.

Reads ``external2_ledger.jsonl`` (written by ``external2_harness.py gate``: the installed wheel
over the EXTERNAL-1 corpus, same reconstruction, same exclusions) and counts, per claim kind and
verdict, then asks two mechanical questions of every accusation the instrument still makes:

  tests_added / symbol_added   Did the diff touch any Python file?  Both templates count ``def``
                               lines, so in a diff with no ``.py``/``.pyi`` file the count is 0
                               before the diff is read, and the stated reason ("diff adds 0 test
                               functions") is not evidence about the sentence.
  only_touches                 Is the captured "prefix" path-shaped, or an English word?  The
                               template takes the token after "only modifies/touches/changes";
                               "only modifies the footer" yields the prefix "the".

Neither question adjudicates the sentence.  "Added 3 tests" in a TypeScript PR may be true or
false; a ``def test_`` count cannot tell, so the honest verdict is UNCHECKABLE and the emitted
CONTRADICTED is unsupported.  The census also records, for the VERIFIED side, the #101 exposure:
``tests_added`` claims whose added ``def test_`` names also appear in removed lines.

Counts only.  No third-party PR is named in the receipt.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "external2_ledger.jsonl"
OUT = HERE / "external2_summary.json"
PY = {".py", ".pyi"}
# English function words the only_touches template captured as a "prefix" on this corpus.  A
# prefix is path-shaped when it is not one of these AND carries a path character (/ \ . - _).
STOP = {"the", "a", "an", "with", "that", "this", "these", "those", "how", "is", "are", "was",
        "were", "when", "what", "which", "to", "in", "on", "at", "by", "for", "of", "one", "two",
        "necessary", "existing", "relevant", "specific", "code", "files", "file", "values",
        "behavior", "per", "excluded", "it", "its", "as", "and", "or", "not", "if", "where"}
TEST_NOUN = re.compile(r"\b(?:add\w+|creat\w+)\s+\d+\s+(?:new\s+)?tests?\b\s*(\w+)?", re.I)


def looks_like_path(prefix: str) -> bool:
    p = prefix.strip("`'\"").rstrip("/.")
    if not p or p.lower() in STOP:
        return False
    return any(ch in p for ch in "/\\.-_")


def main() -> int:
    kv: Counter = Counter()
    per_agent: dict = defaultdict(Counter)
    eligible = covered = prs_with_py = 0
    tests: Counter = Counter({"verified_with_a_redefined_def": 0, "verified_but_net_rule_disagrees": 0})
    symbols: Counter = Counter()
    only: Counter = Counter()
    files: Counter = Counter()
    only_prefixes: Counter = Counter()
    test_nouns: Counter = Counter()
    for line in LEDGER.open(encoding="utf-8"):
        r = json.loads(line)
        eligible += 1
        has_py = any(e in PY for e in r["exts"])
        prs_with_py += has_py
        if r["claims"]:
            covered += 1
        for c in r["claims"]:
            k, v = c["kind"], c["verdict"]
            kv[(k, v)] += 1
            per_agent[r["agent"]][(k, v)] += 1
            if k == "tests_added":
                tests[v] += 1
                if v == "CONTRADICTED":
                    got = int(re.search(r"diff adds (\d+)", c["why"]).group(1))
                    tests["contradicted_no_python_in_diff" if not has_py else "contradicted_python_in_diff"] += 1
                    if got == 0:
                        tests["contradicted_count_was_zero"] += 1
                    m = TEST_NOUN.search(c["text"])
                    nxt = (m.group(1) or "").lower() if m else ""
                    test_nouns[nxt if nxt in ("cases", "case", "files", "file", "scenarios", "functions",
                                              "methods", "suites", "classes") else "(none)"] += 1
                elif v == "VERIFIED":
                    tests["verified_python_in_diff" if has_py else "verified_no_python_in_diff"] += 1
                    added = r.get("def_test_added", [])
                    removed = set(r.get("def_test_removed", []))
                    n = int(c["detail"].get("n", 0))
                    net = [x for x in added if x not in removed]
                    if any(x in removed for x in added):
                        tests["verified_with_a_redefined_def"] += 1
                    if len(net) != n:
                        tests["verified_but_net_rule_disagrees"] += 1
            elif k == "symbol_added":
                symbols[v] += 1
                if v == "CONTRADICTED":
                    symbols["contradicted_no_python_in_diff" if not has_py else "contradicted_python_in_diff"] += 1
                    if c["detail"].get("name", "").lower() in STOP:
                        symbols["contradicted_name_is_a_function_word"] += 1
            elif k == "only_touches":
                only[v] += 1
                if v == "CONTRADICTED":
                    pref = c["detail"].get("prefix", "")
                    only_prefixes[pref.lower()] += 1
                    only["contradicted_prefix_is_not_a_path" if not looks_like_path(pref)
                         else "contradicted_prefix_looks_like_a_path"] += 1
            elif k == "files_changed_count":
                files[v] += 1
                if v == "CONTRADICTED" and re.search(r"insertions?\(\+\)|deletions?\(-\)", c["text"]):
                    files["contradicted_sentence_is_a_git_stat_line"] += 1
    accusations = sum(n for (_k, v), n in kv.items() if v == "CONTRADICTED")
    unsupported = (tests["contradicted_no_python_in_diff"] + symbols["contradicted_no_python_in_diff"]
                   + only["contradicted_prefix_is_not_a_path"])
    kinds = sorted({k for k, _ in kv})
    payload = {
        "what": "census of the accusations the installed wheel still makes on the EXTERNAL-1 corpus",
        "eligible_prs": eligible, "covered_prs": covered, "prs_with_a_python_file": prs_with_py,
        "claims_by_kind_and_verdict": {k: {v: kv[(k, v)] for v in ("VERIFIED", "CONTRADICTED", "UNCHECKABLE")
                                           if kv[(k, v)]} for k in kinds},
        "accusations_total": accusations,
        "accusations_by_kind": {k: kv[(k, "CONTRADICTED")] for k in kinds if kv[(k, "CONTRADICTED")]},
        "tests_added": dict(tests),
        "tests_added_contradicted_by_counted_noun": dict(test_nouns),
        "symbol_added": dict(symbols),
        "only_touches": dict(only),
        "only_touches_top_contradicted_prefixes": only_prefixes.most_common(20),
        "files_changed_count": dict(files),
        "accusations_unsupported_by_construction": unsupported,
        "accusations_unsupported_share": round(unsupported / accusations, 4) if accusations else None,
        "per_agent_accusations": {a: {k: n for (k, v), n in cnt.items() if v == "CONTRADICTED"}
                                  for a, cnt in sorted(per_agent.items())},
        "stop_words": sorted(STOP),
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
