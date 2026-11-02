# -*- coding: utf-8 -*-
"""Constants of the tests-pass-ci collector, each with the PREREG section it comes from.

PREREG = PREREG_tests_pass_ci_2026_11_02.md (draft of 2026-11-02). Part I sections are written
"section N", Part II procedures "PN". Nothing here is tuned on data; every value is the PREREG's.
"""
from __future__ import annotations

import datetime as dt
from fractions import Fraction
from pathlib import Path

UTC = dt.timezone.utc

# ---- the window (P1; section 2) -----------------------------------------------------------------
WINDOW_START = dt.datetime(2026, 11, 2, 0, 0, 0, tzinfo=UTC)
WINDOW_END = dt.datetime(2026, 11, 29, 23, 59, 59, tzinfo=UTC)  # inclusive
WINDOW_DAYS = [dt.date(2026, 11, 2) + dt.timedelta(days=i) for i in range(28)]

# ---- the frame (P2) -----------------------------------------------------------------------------
MIN_STARS = 100  # P1, stargazerCount as returned when the day's frame is enumerated (P2.4)

# P2.1, in the table's order: a pull request matched by two signatures belongs to the earliest
# row that matches it. Agent keys are testreceipt's own (TR:evals/fresh.py AGENTS).
SIGNATURES = (
    ("Copilot", "author:app/copilot-swe-agent", None),
    ("Claude_Code", '"Generated with Claude Code" in:body', "claude code"),
    ("Cursor", '"cursor.com/background-agent" in:body', "cursor.com/"),
    ("Devin", "author:app/devin-ai-integration", None),
    ("Google_Jules", "author:app/google-labs-jules", None),
    ("OpenAI_Codex", "label:codex", None),
)
AGENT_ORDER = tuple(a for a, _, _ in SIGNATURES)
# P2.1 (revised 2026-11-02): a pull request matched by two signatures goes to the earliest of P2.1's rows that
# matches it, Claude Code read last. Its slices see 1/6 of its matches, so a double match given to Claude Code
# would sit in M with probability 1/6 x 0.120 + 5/6 x 0.020 = 0.0367; read last, every double match is in M
# with probability 0.020 like any other kept pull request.
MATCH_PRECEDENCE = tuple(a for a in AGENT_ORDER if a != "Claude_Code") + ("Claude_Code",)
QUALIFIER = {a: q for a, q, _ in SIGNATURES}
POSTFILTER = {a: t for a, _, t in SIGNATURES if t}  # body, lowercased, must contain it (P2.1)
SLICED_AGENT = "Claude_Code"  # P2.3: enumerated in 48 of 288 five-minute slices a day
WHOLE_RANGE_HOURS = 4  # P2.3: other agents, each day whole in four-hour ranges
CC_SLICE_MINUTES = 5
CC_SLICES_PER_DAY = 288
CC_SLICE_EVERY = 6  # slice k enumerated iff (k + o_D) % 6 == 0
SEARCH_CAP = 1000  # a search query returns at most 1,000 results; halve above it
PAGE_SIZE = 100
# P2.4: Bot-authored rows in these strata are excluded at read (P4.4 BOT_AUTHOR)
BOT_EXCLUDED_AGENTS = ("Claude_Code", "Cursor", "OpenAI_Codex")

# P2.2: exactly these fields, nothing else
FRAME_FIELDS = (
    "databaseId number createdAt author { __typename login } "
    "repository { nameWithOwner stargazerCount isFork }"
)
# The dry run counts by the seal's own list only (number, repository, stars, author login,
# created_at); it requests neither databaseId, isFork nor author type, and never the body.
DRYRUN_FIELDS = "number createdAt author { login } repository { nameWithOwner stargazerCount }"

# ---- the draw (P3) ------------------------------------------------------------------------------
P0 = Fraction("0.020")  # main component, every agent (section 3.3)
P_AGENT = {  # P1 P_AGENT; Claude Code's 0.120 applies inside its enumerated slices
    "Claude_Code": Fraction("0.120"),
    "Cursor": Fraction("0.10"),
    "Devin": Fraction("0.35"),
    "OpenAI_Codex": Fraction(1),
    "Copilot": Fraction(1),
    "Google_Jules": Fraction(1),
}
BEACON_HEIGHT = 975000  # section 3.1 (late rule there)
DRAW_TAG = ":tests-pass-ci:"

# ---- reading (P4) -------------------------------------------------------------------------------
READ_LAG = dt.timedelta(hours=72)
READ_WINDOW_END = dt.timedelta(hours=84)  # outside the backlog, read between 72 and 84 hours
LATE_READ_AFTER = dt.timedelta(days=7)
BACKLOG_READ_WITHIN = dt.timedelta(days=3)  # backlog rows: within 3 days of the beacon block
LOG_MAX = 30 * 1024 * 1024  # bytes per job log, gzipped on disk (P1)
ARTIFACT_MAX = 25 * 1024 * 1024  # P1, P18
DISK_FLOOR = 4 * 1024 ** 3  # stop logs and artifacts under 4 GB free on C: (P4.3)
DISK_RESUME = 6 * 1024 ** 3  # resume above 6 GB
DISK_PATH = "C:/"
ANNOTATED_CONCLUSIONS = ("failure", "timed_out", "action_required")  # P4.2
LOGGED_JOB_CONCLUSIONS = ("failure", "timed_out")  # P4.2
CAP = 3  # section 9.2: claimers with a CI test result per repository per analysis set

# ---- packets, decoys, seats, the scorer (P7-P15) --------------------------------------------------
AU_SIZES = {"AU1": 40, "AU2": 30, "AU3": 30, "AU4": 100}  # P1, P7: each the lowest v of its kind in M
PACKET_MAX = 400  # P1, P7: items in P2 with its 30 decoys
DECOYS_PER_PACKET = 30  # section 6, P9: 15 YES, 15 NO
DESC_MAX = 20000  # P1, P8.2: characters of the description, cut with "[cut]"
LOG_EXCERPT_LINES = 150  # P1, P8.3: the last 150 lines of each failing step's group
LOG_EXCERPT_JOB_CHARS = 12000  # P8.3: per job
LOG_EXCERPT_ITEM_CHARS = 30000  # P8.3: per item
LOG_EXCERPT_JOBS = 3  # P8.3: failing test-like jobs per item, in job-id order
ANNOTATIONS_MAX = 20  # P8.3
ANNOTATION_CHARS = 300  # P8.3
QUALIFY_NO = 14  # P13: of the 15 NO-side decoys
QUALIFY_YES = 13  # P13: of the 15 YES-side decoys
SITTING_MAX = 50  # P10.1: at most 50 items a sitting
MAX_OUTPUT_TOKENS = 400  # P10.3
BOOT = 10000  # P1, P15: repository-cluster resamples
CLOSE_P1 = dt.date(2026, 12, 20)  # section 7: an item unanswered by then is NO_ANSWER
CLOSE_P2 = dt.date(2027, 1, 10)
CLAUDE_MODEL = "claude-opus-5-5"  # P10.2
LOCAL_SEATS = ("microsoft/Phi-3.5-mini-instruct", "meta-llama/Llama-3.2-3B-Instruct")  # P10.3, in order

# P18: an artifact counts when its name matches this
REPORT_ARTIFACT_NAME = (
    r"(?i)test|junit|xunit|report|result|surefire|trx|pytest|jest|vitest|mocha|playwright"
    r"|cypress|nunit|ctest"
)
REPORT_CHECK_NAME = r"(?i)test results|junit|test report|unit test results"

# ---- the seal (section 0.1; P21) ----------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DIRECTION_A = Path("C:/Users/heyzo/clawd/wt/_direction_a_2026_11_02")
SEALED_FILE = DIRECTION_A / "SEALED"
STYXX_REPO = Path("C:/Users/heyzo/clawd/styxx")
PREREG_REPO_PATH = "papers/tests-pass-ci/PREREG_tests_pass_ci_2026_11_02.md"

# ---- storage (P2.4, P4.3) -----------------------------------------------------------------------
# The PREREG draft names data/direction_a/frames/ and data/direction_a/a2/; the run brief names
# data/direction_a/raw/. The collector keeps both under one root: <root>/frames, <root>/a2.
DATA_ROOT = Path("C:/Users/heyzo/clawd/data/direction_a/raw")
DEV_ROOT = Path("C:/Users/heyzo/clawd/data/direction_a/raw_dev")
# P9: every salt (the decoy sets' and the packet keys') is written here, outside any git working tree, until its
# release; tpc_packet refuses a salt folder inside one, and a set, key or packet folder inside one that does not
# ignore it (C:/Users/heyzo/clawd, which holds data/direction_a/, is a git working tree)
SALT_DIR = Path("C:/Users/heyzo/direction_a_salts")

USER_AGENT = "styxx-tests-pass-ci-collector/0.1 (research; read-only)"


def iso(t: dt.datetime) -> str:
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(s: str) -> dt.datetime:
    """GitHub's '2026-11-02T00:00:00Z' (and '+00:00' forms) as an aware UTC datetime (py3.9 safe)."""
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    if "." in s:  # drop fractional seconds of any length; fromisoformat on 3.9 wants 3 or 6
        head, _, tail = s.partition(".")
        tz = ""
        for sep in ("+", "-"):
            if sep in tail:
                tz = sep + tail.split(sep, 1)[1]
                break
        s = head + tz
    t = dt.datetime.fromisoformat(s)
    if t.tzinfo is None:
        t = t.replace(tzinfo=UTC)
    return t.astimezone(UTC)


def day_start(d: dt.date) -> dt.datetime:
    return dt.datetime(d.year, d.month, d.day, tzinfo=UTC)
