"""DECLARE-1: the claim stops being guessed and starts being declared.

PREREG_declare1_the_toll_2026_09_18.md (sha256 `7ffd0ba1…`), frozen before this file existed.

Eleven preregistered cycles went into reading English well enough to catch a lying pull request.
DECIDE-1 measured the result: 71% of claims in the corpus are decidable from the diff, the gate
returns a verdict on 5.7% of `only_touches`, and 9 of the 11 accusations it did make were wrong.
Every remaining failure is extraction. "only modifies CHANGELOG.md" and "only changed mods/submods
are serialized" are the same shape and the difference is meaning; two independent oracles died
proving no surface test separates them.

So an agent may declare its claims instead, in one fenced block:

    ```styxx
    files_changed: 38
    only_touches: web/gate
    adds_symbol: gate_diff_text
    tests_added: 22
    ```

**A declaration is normalised into the canonical sentence the existing reader already understands,
and then read by exactly that reader.** Nothing here re-implements a verdict. That is deliberate:
a second checking path would drift from the first, and the differential would not see it. It also
means a declared claim and a prose claim of the same content cannot disagree, by construction.

What this file must not do, from the prereg:

* **It must not change prose reading.** It never touches `_TEMPLATES` or the sentence loop; the
  gate runs its normal pass first and this adds a second, separate one.
* **A declaration must not license an accusation about anything undeclared.** Declaring narrowly is
  visible (`undeclared_kinds`) and is not punished.
* **`tests_pass` must stay UNCHECKABLE even when declared.** It is accepted, reported, and never
  turned into a sentence the reader could verify — the one field an agent could most easily use to
  write the verdict it wants is the one this format refuses to help with.
* **A declaration that cannot be read is not a lie.** Unknown keys and unparseable values are
  reported as problems and never become accusations.
"""
from __future__ import annotations

import re

__all__ = ["BLOCK_RE", "DECLARABLE", "parse_declaration", "canonical_sentence", "declaration_pass"]

# Only a fence tagged exactly `styxx` is a declaration. Nothing else in a body is read as one.
BLOCK_RE = re.compile(r"^[ \t]*```[ \t]*styxx[ \t]*\r?\n(?P<body>.*?)^[ \t]*```", re.S | re.M)

_LINE_RE = re.compile(r"^\s*(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*(?P<value>.*?)\s*$")

#: key -> the claim kind it declares. `tests_pass` is accepted and deliberately unverifiable.
DECLARABLE = {
    "files_changed": "files_changed_count",
    "only_touches": "only_touches",
    "adds_symbol": "symbol_added",
    "tests_added": "tests_added",
    "file_touched": "file_touched",
    "file_created": "file_created",
    "file_deleted": "file_deleted",
    "tests_pass": "tests_pass",
}

_INT_RE = re.compile(r"^\d{1,9}$")
_PATHY_RE = re.compile(r"^[\w.\-/\\]+$")
_IDENT_RE = re.compile(r"^[A-Za-z_]\w*$")


def parse_declaration(text: str) -> tuple[dict[str, str] | None, list[str]]:
    """Find the one declaration block and read its `key: value` lines.

    Returns `(mapping, problems)`. `mapping` is None when there is no block at all, which is the
    ordinary case and is not a problem. Two blocks is a hard error and yields no mapping: merging
    them would invent a declaration nobody wrote.
    """
    blocks = BLOCK_RE.findall(text or "")
    if not blocks:
        return None, []
    if len(blocks) > 1:
        return None, [f"{len(blocks)} styxx blocks; a body declares once or not at all"]

    out: dict[str, str] = {}
    problems: list[str] = []
    for raw in blocks[0].splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        m = _LINE_RE.match(raw)
        if not m:
            problems.append(f"MALFORMED line, not `key: value`: {raw.strip()[:60]!r}")
            continue
        key, value = m.group("key").lower(), m.group("value").strip().strip("`\"'")
        if key not in DECLARABLE:
            problems.append(f"unknown key {key!r}; reported, never checked")
            continue
        if key in out:
            problems.append(f"duplicate key {key!r}; the first is kept")
            continue
        out[key] = value
    return out, problems


def canonical_sentence(key: str, value: str) -> tuple[str | None, str | None]:
    """The sentence the existing reader already understands, or a reason there isn't one.

    `tests_pass` returns `(None, reason)` on purpose — see the module docstring.
    """
    if key == "tests_pass":
        return None, ("declared, and deliberately not verifiable: a declaration that tests passed "
                      "is not evidence that they did")
    if key == "files_changed":
        if not _INT_RE.match(value):
            return None, f"MALFORMED: {value!r} is not a count"
        return f"{int(value)} files changed.", None
    if key == "tests_added":
        if not _INT_RE.match(value):
            return None, f"MALFORMED: {value!r} is not a count"
        return f"Added {int(value)} tests.", None
    if key == "adds_symbol":
        if not _IDENT_RE.match(value):
            return None, f"MALFORMED: {value!r} is not an identifier"
        return f"Adds function {value}.", None

    # The path-shaped keys. A trailing glob is how people naturally write a prefix; it is stripped
    # rather than refused, and nothing else glob-like is accepted.
    v = re.sub(r"/\*{1,2}$", "", value)
    if not v or not _PATHY_RE.match(v):
        return None, f"MALFORMED: {value!r} is not a path"
    if key == "only_touches":
        return f"Only touches {v}.", None
    if key == "file_touched":
        return f"Modified {v}.", None
    if key == "file_created":
        return f"Created file {v}.", None
    if key == "file_deleted":
        return f"Deleted {v}.", None
    return None, f"no canonical form for {key!r}"


def declaration_pass(summary_text: str) -> tuple[str, dict]:
    """`(text_to_gate, report)`.

    `text_to_gate` is the synthesized canonical sentences, to be read by the ordinary gate. It is
    empty when there is nothing declarable, and the caller then does no second pass at all.
    """
    mapping, problems = parse_declaration(summary_text)
    report = {"declared": False, "keys": [], "problems": list(problems),
              "unverifiable": [], "undeclared_kinds": []}
    if mapping is None:
        return "", report
    report["declared"] = True
    report["keys"] = sorted(mapping)

    sentences = []
    for key in sorted(mapping):
        sent, why = canonical_sentence(key, mapping[key])
        if sent is None:
            report["unverifiable"].append({"key": key, "value": mapping[key], "why": why})
        else:
            sentences.append(sent)
    return "\n".join(sentences), report
