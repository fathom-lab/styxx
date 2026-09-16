"""The fuzzed half of the differential corpus: 3,000 synthetic (summary, diff) pairs.

    python fuzz_corpus.py            # writes corpus_fuzz.json next to this file

Deterministic: random.seed(20260916), so every run yields the same 3,000 pairs (CPython's
`random` has kept these methods stable since 3.2). The generator leans on the template set's
soft spots on purpose — quoted and bare paths, non-file nouns (Node.js), "the same way x.py
was" references, bullets, counts of zero, "only touches" with and without a trailing slash,
diffs that are empty or not diffs — because a port that agrees on easy input proves little.
"""
from __future__ import annotations

import json
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
random.seed(20260916)
paths = ["src/app.py", "src/retry.py", "docs/readme.md", "README.md", "integrations/git/README.md", "config/settings.yml", "tests/test_retry.py",
         "lib/node.js", "node.js", "Express.js", "styxx/diffgate.py", "web/index.html", "a/b/c.rs", "sla.py", "tsconfig.json", "test.yml", "glob.ts", "src/node/glob.ts"]
verbs_touch = ["Modified", "updated", "Edited", "changed", "Refactored", "fixed", "Added", "extended", "hardened", "wired", "patched", "fixes", "Fixing"]
verbs_create = ["Created", "creates", "new file", "Creating", "Create the module", "new"]
verbs_del = ["Deleted", "removed the file", "Removes", "deleting"]
refs = ["the same way", "as in", "similar to", "will be", "staged", "follow-up", "TODO", "avoids modifying", "unchanged", "not in this", "preserves", ""]
cont = ["from", "in", "inside", "within", "out of", "of", ""]
names = ["backoff", "retry_once", "Retry", "go", "_helper", "x1"]
def sentence():
    r = random.random()
    p = random.choice(paths)
    if r < 0.2: return f"{random.choice(verbs_touch)} {random.choice(['', 'the parser in ', 'stuff in '])}{random.choice(['`','',chr(34)])}{p}{random.choice(['`','',chr(34)])}."
    if r < 0.32: return f"{random.choice(verbs_create)} {random.choice(['', 'file ', 'script '])}{p}."
    if r < 0.42: return f"{random.choice(verbs_del)} {random.choice(cont)} {random.choice(['the ','its ',''])}{p}."
    if r < 0.5: return f"- {p} — {random.choice(['new', 'created', 'the renderer', 'touched'])}."
    if r < 0.58: return f"{random.randint(0,6)} files {random.choice(['', 'were '])}changed."
    if r < 0.66: return f"{random.choice(['Added','creates','add'])} {random.randint(0,4)} {random.choice(['', 'new '])}tests."
    if r < 0.74: return f"{random.choice(['adds','introduces','Added'])} {random.choice(['', 'a ', 'the '])}{random.choice(['function','class','method'])} {random.choice(names)}."
    if r < 0.82: return f"only {random.choice(['touches','modifies','changes'])} {random.choice(['', 'files under ', 'files in '])}{random.choice(['src/', 'src', 'docs/.', './src', 'tests', 'a/b', 'config'])}."
    if r < 0.9: return random.choice(["All tests pass.", "tests are passing", "Tests green!", "the suite is fine"])
    ref = random.choice(refs)
    return f"{random.choice(verbs_touch)} {p} {ref} {random.choice(paths)} {random.choice(['', 'later', 'in a later commit'])}."
def diff():
    out = []
    for p in random.sample(paths, random.randint(0, 5)):
        st = random.choice(["A", "M", "D"])
        if st == "A": out.append(f"--- /dev/null\n+++ b/{p}\n@@ -0,0 +1,2 @@\n+def test_{random.choice(names)}():\n+    pass\n")
        elif st == "D": out.append(f"--- a/{p}\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n")
        else: out.append(f"--- a/{p}\n+++ b/{p}\n@@ -1,2 +1,3 @@\n x\n+{random.choice(['def '+random.choice(names)+'(n):', 'class '+random.choice(names)+':', '    def test_'+random.choice(names)+'():', 'y = 1'])}\n")
    if random.random() < 0.05: return ""
    if random.random() < 0.03: return "not a diff\n"
    return "".join(out)
items = []
for i in range(3000):
    n = random.randint(1, 6)
    sep = random.choice([" ", "\n", "  ", "\n\n"])
    summary = sep.join(sentence() for _ in range(n))
    items.append({"id": f"fuzz:{i}", "summary": summary, "diff": diff()})
(HERE / "corpus_fuzz.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
print(f"{len(items)} fuzzed pairs -> corpus_fuzz.json")
