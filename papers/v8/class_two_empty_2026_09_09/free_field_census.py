"""Which certificate fields does nothing read? A screening tool for the error made four times.

THE ERROR. THE_BOUNDARY placed four defects in a class called "reachable by nothing" and all four
were reachable. Every time, the shape was identical: a field was written into a certificate, some
other logged byte constrained it, and no code compared them. The author checked one certificate,
found nothing that contradicted the field, and concluded nothing could.

A field that no code reads can hold anything. That is not a proof of unreachability -- today's
lesson is precisely that it is not -- but it is where candidates come from, and it is mechanical,
which is the point. Four times a human missed this by inspection.

WHAT THIS IS NOT. Text search cannot see a field read through a variable (`cert["body"][key]`),
through a schema walk, or by a canonicaliser that digests whole objects. So:

    read found      -> something reads it. Reliable.
    no read found   -> CANDIDATE. It may still be read dynamically. Not a conclusion.

The honest version of this measurement is mutation: change the field, run the suite, see if
anything fails. That is recommended at the end and is not done here, because a mutation census
needs a stable tree and ten agents are editing this one.

Everything is read from git HEAD, not the working tree, so the result names a commit.
"""
import json
import re
import subprocess
import sys

REPO = r"C:\Users\heyzo\clawd\wt\v8"


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout


HEAD = git("rev-parse", "--short", "HEAD").strip()
print(f"census at commit {HEAD}\n")

schema_files = [p for p in git("ls-tree", "-r", "--name-only", "HEAD").splitlines()
                if p.startswith("styxx/v8/schema/") and p.endswith(".json")]
code_files = [p for p in git("ls-tree", "-r", "--name-only", "HEAD").splitlines()
              if p.startswith("styxx/") and p.endswith((".py", ".js"))]
test_files = [p for p in git("ls-tree", "-r", "--name-only", "HEAD").splitlines()
              if p.startswith("tests/") and p.endswith((".py", ".js"))]

if not schema_files:
    print("no schema files at HEAD; nothing to census")
    sys.exit(0)

code = {p: git("show", f"HEAD:{p}") for p in code_files}
tests = {p: git("show", f"HEAD:{p}") for p in test_files}
print(f"{len(schema_files)} schema files, {len(code_files)} source files, "
      f"{len(test_files)} test files\n")


def fields(node, prefix=""):
    """Every property name a schema declares, with its path."""
    out = []
    if not isinstance(node, dict):
        return out
    props = node.get("properties")
    if isinstance(props, dict):
        for k, v in props.items():
            path = f"{prefix}.{k}" if prefix else k
            out.append(path)
            out.extend(fields(v, path))
    for key in ("items", "additionalProperties"):
        if isinstance(node.get(key), dict):
            out.extend(fields(node[key], prefix))
    for key in ("oneOf", "anyOf", "allOf"):
        for sub in node.get(key) or []:
            out.extend(fields(sub, prefix))
    if isinstance(node.get("$defs"), dict):
        # Keep the definition's name in the path. Dropping it reported `$defs.alias.observed_at`
        # as a top-level `observed_at`, which sent a reader looking in the wrong place -- a third
        # failure mode of this tool, found by checking its output by hand.
        for k, v in node["$defs"].items():
            out.extend(fields(v, f"{prefix}.$defs.{k}" if prefix else f"$defs.{k}"))
    return out


def reads(name, corpus):
    """Files that appear to read this field name, excluding the schema files themselves."""
    pat = re.compile(r"""["']%s["']|\.%s\b""" % (re.escape(name), re.escape(name)))
    return [p for p, text in corpus.items() if pat.search(text)]


rows = []
for sf in schema_files:
    try:
        schema = json.loads(git("show", f"HEAD:{sf}"))
    except json.JSONDecodeError:
        print(f"  ! {sf} does not parse at HEAD")
        continue
    for path in sorted(set(fields(schema))):
        leaf = path.split(".")[-1]
        rows.append({
            "schema": sf.split("/")[-1],
            "path": path,
            "leaf": leaf,
            "code": reads(leaf, code),
            "tests": reads(leaf, tests),
        })

seen, uniq = set(), []
for r in rows:
    key = (r["schema"], r["path"])
    if key not in seen:
        seen.add(key)
        uniq.append(r)

# SECOND PASS, and it exists because the first pass is wrong often enough to matter. The screen
# above looks for "name" or .name, which misses a field read as a bare identifier -- a keyword
# argument, a local, a destructured key. Re-check every candidate with a word-boundary match over
# source files only, and report the disagreement as this tool's own precision. Publishing an
# uncalibrated detector is the thing this lab refuses to do.
src = {p: t for p, t in code.items() if p.endswith((".py", ".js"))}


def word_read(name):
    pat = re.compile(r"\b" + re.escape(name) + r"\b")
    return [p for p, t in src.items() if pat.search(t)]


screened = [r for r in uniq if not r["code"]]
for r in screened:
    r["recheck"] = word_read(r["leaf"])

free = [r for r in screened if not r["recheck"]]
missed = [r for r in screened if r["recheck"]]
untested = [r for r in uniq if r["code"] and not r["tests"]]

print(f"{len(uniq)} distinct declared fields across {len(schema_files)} schemas")
print(f"  read somewhere in styxx/          : {len(uniq) - len(free)}")
print(f"  no read found by the first screen : {len(screened)}")
print(f"  of those, a word-boundary recheck does find a read for {len(missed)}, so this screen's")
print(f"  precision on its own candidate list is {len(free)}/{len(screened)} = "
      f"{len(free) / max(1, len(screened)):.2f}")
print(f"  SURVIVING CANDIDATES              : {len(free)}")
print(f"  read by code but by no test       : {len(untested)}")

if missed:
    print("\nthe first screen was wrong about these; they are read: " +
          ", ".join(sorted(r["leaf"] for r in missed)))

if free:
    print("\nCANDIDATES -- declared, and no read found by either pass. Each is a field that could")
    print("hold anything, until someone shows a read neither search can see:")
    by_schema = {}
    for r in free:
        by_schema.setdefault(r["schema"], []).append(r["path"])
    for s in sorted(by_schema):
        print(f"\n  {s}")
        for path in sorted(by_schema[s]):
            print(f"    {path}")

if untested:
    print("\nREAD BUT NEVER ASSERTED ON -- code consults these, no test mentions them. A read")
    print("with no test behind it is a read that can be deleted without anything going red:")
    for r in sorted(untested, key=lambda r: (r["schema"], r["path"]))[:40]:
        print(f"    {r['schema']:<28} {r['path']}")
    if len(untested) > 40:
        print(f"    ... and {len(untested) - 40} more")

print(f"""
HOW TO READ THIS
  The candidate list is where the next misclassification will come from, and it is generated
  rather than argued. Four times today an author decided by inspection that nothing could
  constrain a field. This list is that judgement made mechanical and cheap to redo.

  It is a screening tool with a known false-negative mode: a field read dynamically looks free
  here and is not. So a name on the list is a question -- "what, if anything, would contradict a
  forged value here?" -- and today's record says the answer is usually something.

  The measurement this approximates is mutation: change each field, run the suite, and record
  what goes red. That gives detection power instead of a text match, and this lab already holds
  that a coverage number without detection power is not a number. Run it on a stable tree.

  Commit: {HEAD}""")
