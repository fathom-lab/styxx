"""Pin the certificate fields that no code reads, so a repair announces itself.

`papers/v8/class_two_empty_2026_09_09/FINDING_free_fields_2026_09_09.md` reports thirteen schema
fields for which no read could be found in `styxx/`. Two clusters were confirmed by hand and are
pinned here.

The convention is this repository's: a test asserts the DESIRED state and carries
``@pytest.mark.xfail(strict=True)`` while the state is not yet reached. When someone implements the
read, the xfail becomes an unexpected pass, the suite goes red, and whoever did the work is told to
delete the marker. Drift that fails silently is drift nobody fixes.

The search here is deliberately generous -- any mention of the name anywhere under ``styxx/``
counts as a read. A generous search that still finds nothing is the finding; a strict one would
invite argument about the regex instead of about the gap.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "styxx"
SCHEMA = SRC / "v8" / "schema"


def _sources() -> dict[pathlib.Path, str]:
    out = {}
    for p in SRC.rglob("*"):
        if p.suffix in (".py", ".js") and "schema" not in p.parts and "_data" not in p.parts:
            try:
                out[p] = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass
    for p in (SRC / "_data").glob("*.js"):
        try:
            out[p] = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
    return out


def _readers(name: str) -> list[str]:
    pat = re.compile(r"\b" + re.escape(name) + r"\b")
    return [str(p.relative_to(ROOT)) for p, text in _sources().items() if pat.search(text)]


def _schema(stem: str) -> dict:
    path = SCHEMA / f"{stem}.json"
    if not path.exists():
        pytest.skip(f"{path.name} is not present; this pin has gone blind rather than passed")
    return json.loads(path.read_text(encoding="utf-8"))


def _declared(node, name: str) -> bool:
    """Is `name` declared anywhere in this schema, at any depth?"""
    if isinstance(node, dict):
        if isinstance(node.get("properties"), dict) and name in node["properties"]:
            return True
        return any(_declared(v, name) for v in node.values())
    if isinstance(node, list):
        return any(_declared(v, name) for v in node)
    return False


# `observed_at` is deliberately NOT pinned for the read, only for the declaration.
#
# It was, and the pin fired within minutes -- not because anything read the alias subject's field,
# but because a new module took `observed_at` as its own parameter name for an unrelated quantity.
# Leaf-name matching cannot tell two fields of the same name in different types apart, which is the
# fourth failure mode this census method has shown (after the two search-precision ones and the
# `$defs` path-attribution one), and it is the worst of them: it reports a field as READ when
# nothing reads it, so it hides gaps rather than inventing them.
#
# `observed_model_id` is distinctive enough to survive the method. It carries the pin; `observed_at`
# shares its fate in the schema and is checked only for being declared.
ALIAS_OBSERVED = ("observed_model_id", "observed_at")
ALIAS_OBSERVED_PINNED = ("observed_model_id",)
SUBLOG_CHAIN = ("sublog_id", "prev_root_hash", "prev_tree_size",
                "entries_sha256", "count_since_prev")


@pytest.mark.parametrize("field", ALIAS_OBSERVED)
def test_alias_observed_identity_is_declared(field: str) -> None:
    """The schema must still declare these, or this pin is testing nothing."""
    subject = _schema("subject")
    assert _declared(subject, field), (
        f"subject.json no longer declares {field!r}. Either the field was removed, which is a "
        f"legitimate resolution of the finding, or this pin has gone blind. Update or delete it "
        f"deliberately; do not leave it asserting nothing."
    )


@pytest.mark.parametrize("field", ALIAS_OBSERVED_PINNED)
@pytest.mark.xfail(
    strict=True,
    reason="FINDING_free_fields_2026_09_09: an alias subject is a hosted model behind an API, so "
           "its identity cannot be established by hashing weights and these two fields are the "
           "only identity evidence there is. No code reads either. This is the dead subject guard "
           "repaired earlier the same day, left unrepaired on the subject kind where it matters "
           "more. Delete this marker when a read exists.",
)
def test_alias_observed_identity_is_read(field: str) -> None:
    readers = _readers(field)
    assert readers, f"nothing under styxx/ mentions {field!r}"


@pytest.mark.parametrize("field", SUBLOG_CHAIN)
@pytest.mark.xfail(
    strict=True,
    reason="FINDING_free_fields_2026_09_09: every field describing a sublog's linkage to its "
           "parent log is declared and read by nothing, so a sublog chains to nothing that is "
           "checked -- the same family as the open truncation attack on the main log. Note the "
           "other resolution: if no one issues sublogs, remove the type from the schema rather "
           "than implementing it, and delete this pin with it.",
)
def test_sublog_chaining_is_read(field: str) -> None:
    readers = _readers(field)
    assert readers, f"nothing under styxx/ mentions {field!r}"


def test_the_census_script_still_runs() -> None:
    """The finding is only as good as the script that produced it."""
    script = (ROOT / "papers" / "v8" / "class_two_empty_2026_09_09" / "free_field_census.py")
    assert script.exists(), (
        "free_field_census.py is gone; FINDING_free_fields_2026_09_09.md cites it as its receipt "
        "and a finding whose receipt has been deleted is an assertion."
    )
    compile(script.read_text(encoding="utf-8"), str(script), "exec")
