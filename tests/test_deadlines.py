"""The hard-deadline guard.

## Why this file exists

`release/formal/patent-clinic-intake.md` records three US provisional patent
applications and a hard twelve-month conversion deadline. Until now that clock existed
in two places: one sentence inside a draft email, and whatever chat session last
mentioned it. A calendar reminder that lives in a chat session dies with the session.

This test reads `docs/DEADLINES.md` and fails the suite once any recorded deadline is
inside its stated lead time. The failure message says what is due, on what date, how
many days remain, and the one action that clears it.

## Two rules that make it a guard rather than a decoration

1. **A malformed row is a failure, never a skip.** A guard that silently drops a row it
   cannot parse is worse than no guard: it reports green while the clock runs out. Every
   parse and validation failure raises `MalformedTable`, and the tests turn that into a
   red suite with the offending line quoted.

2. **The dates are pinned here as well as in the document.** `test_patent_rows_are_pinned`
   asserts the three provisional numbers and their twelve-month dates against constants
   written in this file. Editing `docs/DEADLINES.md` to push a date out or delete a row
   does not quiet the alarm on its own; the diff has to touch a test file too.

The tripping behaviour is proved two ways. `test_guard_trips_at_the_lead_boundary` runs
the real rows against synthetic clocks and asserts silence one day before the boundary
and a breach on it — so the red path is exercised on every run, today, without waiting
for 2027 and without editing the document. `test_no_deadline_is_inside_its_lead_time` is
the live check that actually fires in February 2027.
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DEADLINES_MD = ROOT / "docs" / "DEADLINES.md"

BEGIN_MARKER = "<!-- DEADLINES-TABLE-BEGIN -->"
END_MARKER = "<!-- DEADLINES-TABLE-END -->"

COLUMNS = (
    "id",
    "due",
    "lead_days",
    "what",
    "lost_if_missed",
    "source",
    "owner",
    "action",
)

# Free-text columns that must carry a real answer, not a shrug.
PROSE_COLUMNS = ("what", "lost_if_missed", "owner", "action")
PLACEHOLDERS = {"", "-", "--", "---", "?", "tbd", "t.b.d.", "n/a", "na", "none", "todo", "xxx"}

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_INT_RE = re.compile(r"^\d+$")
_SEPARATOR_CELL_RE = re.compile(r"^:?-{3,}:?$")

MAX_LEAD_DAYS = 3650

# Pinned from release/formal/patent-clinic-intake.md (do not edit without reading it):
# provisional number -> (filing date as stated there, filing date + the 12-month rule
# that same document states). The value is the date the priority lapses.
PINNED_PATENT_ROWS = {
    "patent-64-020-489": ("64/020,489", dt.date(2026, 3, 29), dt.date(2027, 3, 29)),
    "patent-64-021-113": ("64/021,113", dt.date(2026, 3, 30), dt.date(2027, 3, 30)),
    "patent-64-026-964": ("64/026,964", dt.date(2026, 4, 2), dt.date(2027, 4, 2)),
}
PINNED_PATENT_LEAD_DAYS = 60


class MalformedTable(Exception):
    """The deadline table could not be parsed or validated. Always fatal, never skipped."""


def today_utc() -> dt.date:
    """Today in UTC, so the guard does not depend on the machine's timezone."""
    return dt.datetime.now(dt.timezone.utc).date()


def _split_row(line: str) -> list:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        raise MalformedTable("table line is not a pipe-delimited row: {0!r}".format(line))
    return [cell.strip() for cell in stripped[1:-1].split("|")]


def parse_table(text: str) -> list:
    """Parse the machine-read block of docs/DEADLINES.md into a list of row dicts.

    Raises MalformedTable on anything it cannot fully understand.
    """
    if text.count(BEGIN_MARKER) != 1 or text.count(END_MARKER) != 1:
        raise MalformedTable(
            "expected exactly one {0} and one {1} marker; found {2} and {3}".format(
                BEGIN_MARKER, END_MARKER, text.count(BEGIN_MARKER), text.count(END_MARKER)
            )
        )
    start = text.index(BEGIN_MARKER) + len(BEGIN_MARKER)
    end = text.index(END_MARKER)
    if end < start:
        raise MalformedTable("the END marker appears before the BEGIN marker")

    lines = [ln for ln in text[start:end].splitlines() if ln.strip()]
    if len(lines) < 3:
        raise MalformedTable(
            "the deadline table needs a header, a separator and at least one row; "
            "found {0} non-blank line(s). An empty table is a silent pass.".format(len(lines))
        )

    header = _split_row(lines[0])
    if tuple(header) != COLUMNS:
        raise MalformedTable(
            "header columns are {0!r}, expected {1!r}".format(tuple(header), COLUMNS)
        )

    separator = _split_row(lines[1])
    if len(separator) != len(COLUMNS) or not all(
        _SEPARATOR_CELL_RE.match(cell) for cell in separator
    ):
        raise MalformedTable("second table line is not a markdown separator row: {0!r}".format(lines[1]))

    rows = []
    seen_ids = set()
    for lineno, line in enumerate(lines[2:], start=3):
        cells = _split_row(line)
        if len(cells) != len(COLUMNS):
            raise MalformedTable(
                "row {0} has {1} cell(s), expected {2}: {3!r}".format(
                    lineno, len(cells), len(COLUMNS), line
                )
            )
        raw = dict(zip(COLUMNS, cells))

        row_id = raw["id"]
        if not _ID_RE.match(row_id):
            raise MalformedTable(
                "row {0}: id {1!r} is not lowercase alphanumeric with . - _".format(lineno, row_id)
            )
        if row_id in seen_ids:
            raise MalformedTable("row {0}: duplicate id {1!r}".format(lineno, row_id))
        seen_ids.add(row_id)

        if not _DATE_RE.match(raw["due"]):
            raise MalformedTable(
                "row {0} ({1}): due {2!r} is not YYYY-MM-DD".format(lineno, row_id, raw["due"])
            )
        try:
            due = dt.date(*(int(part) for part in raw["due"].split("-")))
        except ValueError as exc:
            raise MalformedTable(
                "row {0} ({1}): due {2!r} is not a real calendar date ({3})".format(
                    lineno, row_id, raw["due"], exc
                )
            )

        if not _INT_RE.match(raw["lead_days"]):
            raise MalformedTable(
                "row {0} ({1}): lead_days {2!r} is not a whole number".format(
                    lineno, row_id, raw["lead_days"]
                )
            )
        lead_days = int(raw["lead_days"])
        if not 1 <= lead_days <= MAX_LEAD_DAYS:
            raise MalformedTable(
                "row {0} ({1}): lead_days {2} is outside 1..{3}".format(
                    lineno, row_id, lead_days, MAX_LEAD_DAYS
                )
            )

        for column in PROSE_COLUMNS:
            if raw[column].strip().lower() in PLACEHOLDERS:
                raise MalformedTable(
                    "row {0} ({1}): column {2!r} is empty or a placeholder ({3!r}). "
                    "A row nobody has filled in is a malformed row.".format(
                        lineno, row_id, column, raw[column]
                    )
                )

        source = raw["source"]
        if source.strip().lower() in PLACEHOLDERS:
            raise MalformedTable("row {0} ({1}): source is empty".format(lineno, row_id))
        source_path = ROOT / source
        if not source_path.is_file():
            raise MalformedTable(
                "row {0} ({1}): source {2!r} does not exist. A deadline whose source "
                "document has moved or been deleted cannot be acted on.".format(
                    lineno, row_id, source
                )
            )

        rows.append(
            {
                "id": row_id,
                "due": due,
                "lead_days": lead_days,
                "what": raw["what"],
                "lost_if_missed": raw["lost_if_missed"],
                "source": source,
                "owner": raw["owner"],
                "action": raw["action"],
            }
        )

    return rows


def breaches(rows, today: dt.date) -> list:
    """Rows whose deadline is at or inside its lead time on `today`, soonest first."""
    hit = [row for row in rows if (row["due"] - today).days <= row["lead_days"]]
    return sorted(hit, key=lambda row: row["due"])


def format_breach(row, today: dt.date) -> str:
    days = (row["due"] - today).days
    if days < 0:
        remaining = "OVERDUE by {0} day(s) - the date has already passed".format(-days)
    elif days == 0:
        remaining = "DUE TODAY - 0 days remaining"
    else:
        remaining = "{0} day(s) remaining".format(days)
    return "\n".join(
        [
            "DEADLINE APPROACHING [{0}]".format(row["id"]),
            "  what is due    : {0}".format(row["what"]),
            "  due date       : {0} (UTC)".format(row["due"].isoformat()),
            "  today          : {0} (UTC)".format(today.isoformat()),
            "  time left      : {0}".format(remaining),
            "  lead time      : {0} day(s) - this guard fires at or inside it".format(
                row["lead_days"]
            ),
            "  if missed      : {0}".format(row["lost_if_missed"]),
            "  who must act   : {0}".format(row["owner"]),
            "  action that clears it: {0}".format(row["action"]),
            "  source document: {0}".format(row["source"]),
        ]
    )


def _load_rows():
    if not DEADLINES_MD.is_file():
        pytest.fail(
            "{0} is missing. It is the only committed record of this project's hard "
            "external deadlines; deleting it deletes the patent conversion clock.".format(
                DEADLINES_MD
            )
        )
    try:
        return parse_table(DEADLINES_MD.read_text(encoding="utf-8"))
    except MalformedTable as exc:
        pytest.fail(
            "docs/DEADLINES.md is malformed and the guard refuses to skip it: {0}".format(exc)
        )


# --------------------------------------------------------------------------------------
# The live guard
# --------------------------------------------------------------------------------------


def test_deadline_table_parses_and_has_rows():
    rows = _load_rows()
    assert rows, "docs/DEADLINES.md parsed to zero rows - an empty guard is a silent pass"


def test_no_deadline_is_inside_its_lead_time():
    """Fails once any recorded deadline is within its lead time. This is the alarm."""
    rows = _load_rows()
    today = today_utc()
    hit = breaches(rows, today)
    if hit:
        pytest.fail(
            "\n\n".join([format_breach(row, today) for row in hit])
            + "\n\nThis test fails until the action above is taken, or until "
            "docs/DEADLINES.md records a deliberate decision to let the deadline lapse."
        )


def test_patent_rows_are_pinned():
    """The alarm cannot be quieted by editing docs/DEADLINES.md alone."""
    rows = {row["id"]: row for row in _load_rows()}
    for row_id, (number, filed, due) in sorted(PINNED_PATENT_ROWS.items()):
        assert row_id in rows, (
            "row {0!r} (US Provisional {1}, filed {2}) is missing from docs/DEADLINES.md. "
            "Its priority lapses {3}; removing the row does not stop the clock.".format(
                row_id, number, filed.isoformat(), due.isoformat()
            )
        )
        row = rows[row_id]
        assert row["due"] == due, (
            "row {0!r} says due {1}, but release/formal/patent-clinic-intake.md gives a "
            "filing date of {2} and a 12-month rule, which is {3}.".format(
                row_id, row["due"].isoformat(), filed.isoformat(), due.isoformat()
            )
        )
        assert row["lead_days"] == PINNED_PATENT_LEAD_DAYS, (
            "row {0!r} has lead_days {1}; the patent rows are pinned at {2}.".format(
                row_id, row["lead_days"], PINNED_PATENT_LEAD_DAYS
            )
        )
        assert number in row["what"], (
            "row {0!r} no longer names provisional {1} in its 'what' column".format(row_id, number)
        )


def test_guard_trips_at_the_lead_boundary():
    """Proves the red path on every run, against the real rows, without waiting for 2027."""
    rows = _load_rows()
    soonest = min(rows, key=lambda row: row["due"])
    boundary = soonest["due"] - dt.timedelta(days=soonest["lead_days"])

    assert not breaches(rows, boundary - dt.timedelta(days=1)), (
        "guard fired one day before the lead boundary ({0}); it is one day too eager".format(
            boundary.isoformat()
        )
    )
    fired = breaches(rows, boundary)
    assert fired and fired[0]["id"] == soonest["id"], (
        "guard did not fire on {0}, which is exactly {1} days before {2} is lost".format(
            boundary.isoformat(), soonest["lead_days"], soonest["id"]
        )
    )
    assert breaches(rows, soonest["due"] + dt.timedelta(days=1)), "guard went quiet after the date passed"

    message = format_breach(fired[0], boundary)
    for required in (soonest["due"].isoformat(), str(soonest["lead_days"]), soonest["action"]):
        assert required in message, "failure message omits {0!r}".format(required)
    assert "day(s) remaining" in message


def test_the_patent_clock_will_ring_in_early_2027():
    """The whole point, stated as an assertion: silence in January, noise by February."""
    rows = _load_rows()
    assert not breaches(rows, dt.date(2027, 1, 27))
    assert breaches(rows, dt.date(2027, 2, 1)), (
        "no deadline fires on 2027-02-01; the patent conversion clock is not being watched"
    )


# --------------------------------------------------------------------------------------
# The parser's own contract, on synthetic tables
# --------------------------------------------------------------------------------------

_GOOD_ROW = (
    "| demo-row | 2099-01-01 | 30 | Do the thing | The thing is lost | "
    "docs/DEADLINES.md | Somebody Named | File the thing. |"
)


def _table(*rows: str) -> str:
    header = "| " + " | ".join(COLUMNS) + " |"
    separator = "| " + " | ".join(["---"] * len(COLUMNS)) + " |"
    body = "\n".join([header, separator] + list(rows))
    return "before\n{0}\n{1}\n{2}\nafter\n".format(BEGIN_MARKER, body, END_MARKER)


def test_parses_a_well_formed_synthetic_table():
    rows = parse_table(_table(_GOOD_ROW))
    assert len(rows) == 1
    assert rows[0]["due"] == dt.date(2099, 1, 1)
    assert rows[0]["lead_days"] == 30


@pytest.mark.parametrize(
    "row, because",
    [
        (_GOOD_ROW.replace("2099-01-01", "2099-02-30"), "date does not exist"),
        (_GOOD_ROW.replace("2099-01-01", "01/01/2099"), "wrong date format"),
        (_GOOD_ROW.replace("2099-01-01", "soon"), "date is not a date"),
        (_GOOD_ROW.replace("| 30 |", "| thirty |"), "lead_days is not a number"),
        (_GOOD_ROW.replace("| 30 |", "| 0 |"), "lead_days of zero never warns"),
        (_GOOD_ROW.replace("| 30 |", "| 99999 |"), "lead_days beyond the cap"),
        (_GOOD_ROW.replace("| Do the thing |", "| TBD |"), "placeholder in what"),
        (_GOOD_ROW.replace("| The thing is lost |", "|  |"), "empty lost_if_missed"),
        (_GOOD_ROW.replace("| Somebody Named |", "| n/a |"), "nobody is named as owner"),
        (_GOOD_ROW.replace("| File the thing. |", "| ? |"), "no action recorded"),
        (_GOOD_ROW.replace("docs/DEADLINES.md", "docs/does-not-exist.md"), "source is gone"),
        (_GOOD_ROW.replace("| demo-row |", "| Demo Row |"), "id is not a slug"),
        (_GOOD_ROW.rstrip("|").rstrip() + " | extra |", "wrong cell count"),
        (_GOOD_ROW.replace("| demo-row |", "").strip(), "row is truncated"),
    ],
)
def test_malformed_rows_are_fatal_not_skipped(row, because):
    with pytest.raises(MalformedTable):
        parse_table(_table(row))


def test_duplicate_ids_are_fatal():
    with pytest.raises(MalformedTable):
        parse_table(_table(_GOOD_ROW, _GOOD_ROW))


def test_empty_table_is_fatal():
    with pytest.raises(MalformedTable):
        parse_table(_table())


def test_missing_or_doubled_markers_are_fatal():
    good = _table(_GOOD_ROW)
    with pytest.raises(MalformedTable):
        parse_table(good.replace(BEGIN_MARKER, ""))
    with pytest.raises(MalformedTable):
        parse_table(good.replace(END_MARKER, ""))
    with pytest.raises(MalformedTable):
        parse_table(good + good)


def test_wrong_header_is_fatal():
    good = _table(_GOOD_ROW)
    with pytest.raises(MalformedTable):
        parse_table(good.replace("| id |", "| identifier |", 1))


def test_breach_boundary_is_inclusive():
    rows = parse_table(_table(_GOOD_ROW))
    due = rows[0]["due"]
    lead = rows[0]["lead_days"]
    assert not breaches(rows, due - dt.timedelta(days=lead + 1))
    assert breaches(rows, due - dt.timedelta(days=lead))
    assert breaches(rows, due)
    assert breaches(rows, due + dt.timedelta(days=365))


def test_overdue_message_says_overdue():
    row = parse_table(_table(_GOOD_ROW))[0]
    message = format_breach(row, row["due"] + dt.timedelta(days=3))
    assert "OVERDUE by 3 day(s)" in message
    message = format_breach(row, row["due"])
    assert "DUE TODAY" in message
