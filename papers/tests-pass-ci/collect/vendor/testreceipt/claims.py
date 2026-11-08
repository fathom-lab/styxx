"""What an agent says about its test runs, in a pull request description or a commit message.

A claim is a line that says tests passed ("All 101 tests pass", "✅ `pytest -q`", "12 passed in
0.4s") or failed ("`pytest` *(fails: 3 failed)*"). Lines that only ask for tests to pass, such as
an unticked template checkbox, a merge requirement or quoted issue text, are not claims. Precision
matters more than recall: a line is only taken as a claim when it reads as a report of something
done.

A pass claim also has a scope. "All tests pass" and "All 252 existing tests pass" speak for the
whole suite; "All 53 permission endpoint tests pass", "the new tests pass", "all tests pass on
Windows" or "All tests pass:" followed by a list of test files speak for a part of it. A suite that
fails in CI contradicts the first kind; the second kind is a narrower claim that a reviewer may
read as the first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

TEST_WORD = r"(?:tests?|test suites?|test cases?|unit tests?|testing)"
RUNNER = r"(?<![\w-])(?:pytest|py\.test|jest|vitest|mocha|tox|nox|ctest)(?![\w-])"
# "passing" only as an outcome ("are passing", "passing with"), not as in "passing programs"
PASS_VERB = (
    r"(?:pass(?:es|ed)?\b|(?:are|is|now|all|still)\s+passing\b"
    r"|passing(?=\s*(?:[.,;:!()]|$|with\b|locally\b|successfully\b|after\b|on\b|in\b))"
    r"|succeed(?:s|ed)?\b|(?:are|is|now)\s+green\b|continue\s+to\s+pass\b)"
)
TEST_COMMAND = re.compile(
    r"`[^`]*(?<![\w-])(?:pytest|py\.test|tox|nox|unittest|jest|vitest|mocha|ctest)(?![\w-])[^`]*`"
    r"|`[^`]*\b(?:(?:npm|yarn|pnpm|bun)\s+(?:run\s+)?test|go\s+test|cargo\s+test|dotnet\s+test"
    r"|mvn\s+test|gradlew?\s+test|make\s+test|manage\.py\s+test|hatch\s+test)\b[^`]*`",
    re.IGNORECASE,
)
# a command as an agent lists it under "Testing": it starts with a tool and is not a file name
COMMAND_LIKE = re.compile(
    r"`\s*(?:[A-Z_]+=\S+\s+)*(?:uv\s+run\s+|poetry\s+run\s+|python3?\s+-m\s+|npx\s+|corepack\s+)?"
    r"(?:pytest|py\.test|tox|nox|jest|vitest|mocha|ctest|npm|yarn|pnpm|bun|go|cargo|dotnet|mvn|"
    r"gradlew?|make|hatch|python3?\s+manage\.py)\s[^`]*`",
    re.IGNORECASE,
)
PASSED = re.compile(
    # the tests as the subject, at most five words before the verb: "All 101 tests now pass"
    rf"\b{TEST_WORD}\b(?:\s+[^\s.;]+){{0,5}}?\s*:?\s+{PASS_VERB}"
    rf"|\bpass(?:es|ed)?\s+(?:all\s+)?(?:the\s+)?(?:\d+\s+)?(?:existing\s+)?{TEST_WORD}\b"
    r"|\b\d+\s+passed\b|\b(\d+)\s*/\s*\1\s+(?:tests?\s+)?pass"
    r"|\b100\s*%\s+pass(?:ing)?\s+rate\b"
    r"|(?:测试|单元测试|用例)[^。\n]{0,20}(?:全部|均|都)?通过",
    re.IGNORECASE,
)
FAILED = re.compile(
    r"\bfail(?:s|ed|ing|ures?)?\b|\berrors?\b|❌|⛔|\bbroken\b|未通过|失败",
    re.IGNORECASE,
)
ZERO_FAILURES = re.compile(
    r"\b(?:0|no|zero)\s+(?:failures?|failed|errors?|failing)\b", re.IGNORECASE
)
# "772/773 passing": a report that admits failures
SHORT_OF_ALL = re.compile(r"\b(\d+)\s*/\s*(\d+)\b")
PASS_RATE = re.compile(r"\b(\d{1,3}(?:\.\d+)?)\s*%\s+pass(?:ing)?\s+rate\b", re.IGNORECASE)
HEDGE = re.compile(
    r"\b(?:should|ensure|make sure|verify that|needs? to|must|will|would|could|might|please|"
    r"expect(?:ed|s)?|hopefully|once|if|todo|in order to|so that|so (?:it|they)|to make|"
    r"(?<!continue )to pass)\b"
    r"|n't\b|\bnot\b|\bunable\b|\bwithout\b"
    # reported speech: "descriptions that say tests pass", "the agent claimed all tests passed"
    r"|\b(?:say|says|said|saying|claim|claims|claimed|claiming|assert|asserts|asserted|"
    r"report|reports|reported|reporting|stating|states|stated)\s+(?:that\s+)?(?:all\s+)?"
    r"(?:the\s+)?(?:\w+\s+)?tests?\b",
    re.IGNORECASE,
)
UNTICKED = re.compile(r"^\s*(?:[-*+]|\d+\.)\s*\[\s\]")
TICKED = re.compile(r"^\s*(?:[-*+]|\d+\.)\s*\[[xX]\]")
COUNT = re.compile(
    rf"\b(?:all\s+)?(\d{{1,6}})\+?\s+(?:\w+\s+){{0,2}}?{TEST_WORD}\b|\b(\d{{1,6}})\s+passed\b",
    re.IGNORECASE,
)
SECTION = re.compile(r"^\s*#{1,6}\s*(.+?)\s*$")
TESTING_SECTIONS = re.compile(r"\b(?:testing|tests?|checks?|verification|validation)\b", re.I)
COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
QUOTED_BLOCKS = re.compile(
    r"<(issue_title|issue_description|details)>.*?</\1>", re.DOTALL | re.IGNORECASE
)
QUOTED = re.compile(r"\"[^\"\n]{1,80}\"|“[^”\n]{1,80}”|「[^」\n]{1,80}」")
# a lead-in whose list states requirements or plans, not results
REQUIREMENTS = re.compile(
    r"\b(?:until|before\s+merg\w*|requirements?|todo|to\s+do|next\s+steps|acceptance\s+criteria"
    r"|remaining|follow[- ]?ups?)\b.*:\s*$",
    re.IGNORECASE,
)

# --- scope -------------------------------------------------------------------------------------

# words that leave a claim about the whole suite: "all existing unit and integration tests"
GENERIC = frozenset(
    """all the existing other remaining current previous previously unit integration e2e
    end-to-end regression automated and of my our its both every each full entire whole
    test tests suite suites case cases existing, also still now including plus""".split()
)
PLATFORM = re.compile(
    r"\bon\s+(?:windows|macos|mac|linux|ubuntu|darwin|py(?:thon)?\s*3\.\d+|node\s*\d+)\b",
    re.IGNORECASE,
)
NARROWING = re.compile(
    r"(?:new(?:ly)?|added|relevant|related|affected|targeted|specific|modified|changed|selected"
    r"|[\w-]+-related)",
    re.IGNORECASE,
)
# "(5/5 for react-monaco-editor)", "(16/16 across v1 and v2 validation)", "(20 tests in `tests/x/`)"
PARENTHESISED_PART = re.compile(r"\([^)]*\b(?:for|across|in)\s+[`\w][^)]*\)", re.IGNORECASE)
# a path the tests were limited to: backticked, or a bare tests/ path
PATH = re.compile(r"`(?!https?:)[^`\s]*/[^`\s]*`|\b(?:tests?|spec|__tests__)/[\w./-]+")
LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
URL = re.compile(r"https?://\S+")
TEST_FILE = re.compile(r"[\w/.-]*(?:test_[\w-]+\.py|[\w-]+_test\.\w+|\.(?:test|spec)\.\w+)")
# a test command limited to some tests: `--grep "..."`, `-k name`, `file.py::test_x`, `-t name`
FILTERED = re.compile(r"(?:--grep|--testNamePattern|\s-k\s|\s-t\s|::\w)")
# a named group of tests in a list: "CLI flag tests: 5/5 pass", "Integration tests: All pass"
GROUP = re.compile(r"^[-*+]\s*(?:[✅✔]\s*)?[\w /&-]{2,60}\btests?\b\s*[:(-]", re.IGNORECASE)
# a ticked step of a plan: "[x] Verify existing tests pass" says what was to be done
PLAN_STEP = re.compile(
    r"^\s*(?:[-*+]|\d+\.)\s*\[[xX]\]\s*(?:verify|run|ensure|add|update|test|write|check|make"
    r"|fix|implement|create|confirm)\b",
    re.IGNORECASE,
)
# where the qualifier of the tests starts: after "all", "the", "existing" or a verb before it
QUALIFIER_STOPS = frozenset(
    """all the every each both and existing other remaining current confirmed verified
    checked ran run that with""".split()
)


@dataclass(frozen=True)
class Claim:
    kind: str  # "pass" or "fail"
    text: str  # the line, as written
    count: int | None  # how many tests it says passed, when it says
    explicit: bool  # said in words, rather than implied by a command listed without a failure
    scope: str = "full"  # "full": the whole suite; "part": some of it; "count": N tests, either
    why_part: str = ""  # what narrows a "part" claim


def claims(text: str) -> list[Claim]:
    found: list[Claim] = []
    section = ""
    in_requirements = False
    body = QUOTED_BLOCKS.sub("", COMMENT.sub("", text or ""))
    lines = [raw.strip() for raw in body.splitlines()]
    for i, line in enumerate(lines):
        if not line:
            in_requirements = False
            continue
        heading = SECTION.match(line)
        if heading:
            section = heading.group(1)
            in_requirements = bool(REQUIREMENTS.search(section + ":"))
            continue
        if REQUIREMENTS.search(line):
            in_requirements = True
            continue
        if in_requirements or line.startswith(">") or UNTICKED.match(line):
            continue
        claim = _claim(
            line, testing_section=bool(TESTING_SECTIONS.search(section)), following=lines[i + 1 :]
        )
        if claim:
            found.append(claim)
    return found


def _claim(line: str, *, testing_section: bool, following: list[str]) -> Claim | None:
    # a quoted phrase ('the "tests pass" claim') is talked about, not claimed
    line = QUOTED.sub("", line)
    cleared = ZERO_FAILURES.sub("", line)
    command = TEST_COMMAND.search(line)
    tests_mentioned = bool(command) or bool(
        re.search(rf"\b{TEST_WORD}\b|{RUNNER}|\b\d+\s+passed\b|测试|用例", line, re.IGNORECASE)
    )
    if not tests_mentioned:
        return None
    if FAILED.search(cleared) or "⚠" in line or _short_of_all(line):
        return Claim("fail", line, None, explicit=True)
    rate = PASS_RATE.search(line)
    if rate and float(rate.group(1)) < 100:
        return None  # "8-15% pass rate": a measurement of something else, not a test run
    count = _count(line)
    if PLAN_STEP.match(line):
        return None
    if PASSED.search(line) and not HEDGE.search(_after_tick(line)):
        scope, why = scope_of(line, following)
        return Claim("pass", line, count, True, scope, why)
    # a tick next to a test command reports a run; a tick next to "Added tests" reports work done
    if command and any(mark in line for mark in ("✅", "✔", "☑")):
        scope, why = scope_of(line, following)
        return Claim("pass", line, count, True, scope, why)
    if command and testing_section and _listed(line) and COMMAND_LIKE.search(line):
        # agents such as Codex list the commands they ran under "Testing" and note failures
        return Claim("pass", line, count, explicit=False)
    return None


def _short_of_all(line: str) -> bool:
    return any(int(a) < int(b) for a, b in SHORT_OF_ALL.findall(line) if int(b) < 1_000_000)


def scope_of(line: str, following: list[str] | None = None) -> tuple[str, str]:
    """ "full" or "part", and what makes it a part."""
    text = URL.sub("", LINK.sub(r"\1", TICKED.sub("", line)))
    platform = PLATFORM.search(text)
    if platform:
        return "part", platform.group(0)
    part = PARENTHESISED_PART.search(text)
    if part:
        return "part", part.group(0)
    path = PATH.search(text)
    if path:
        return "part", path.group(0)
    qualifier = _subject_words(text)
    if [w for w in qualifier if NARROWING.fullmatch(w)]:
        return "part", " ".join(qualifier)
    if [w for w in qualifier if w.lower() not in GENERIC and not _decoration(w)]:
        return "part", " ".join(qualifier)
    only_new = re.search(r"\b(?:added|new)\s+(?:and\s+)?(?:passing|tests?)\b", text, re.I)
    if only_new and not re.search(r"\bexisting\b|\bincluding\b", text, re.IGNORECASE):
        return "part", "new tests"
    if text.rstrip("*_ ").endswith(":") and following:
        narrowed = _narrowed_by_block(following)
        if narrowed:
            return "part", narrowed
    # "All 68 tests pass", "38/38 passing": the number may be the suite or a part of it; a test
    # run's total settles it. "All tests pass (13/13)" and "All existing tests pass (252)" do not
    # depend on the number.
    numbered = re.search(
        rf"\b\d[\d,]*\+?\s+(?:\w+\s+)?{TEST_WORD}\b|\b(\d+)\s*/\s*\1\b", text, re.I
    )
    whole = re.search(
        rf"\ball\s+(?:the\s+)?(?:existing\s+)?(?:unit\s+)?{TEST_WORD}\b|\b(?:existing|entire|full|whole)\b",
        text,
        re.IGNORECASE,
    )
    if numbered and not whole:
        return "count", numbered.group(0)
    return "full", ""


def _narrowed_by_block(following: list[str]) -> str:
    """What the block right under "All tests pass:" says it covers, if it narrows the claim: test
    files, named groups of tests ("CLI flag tests: 5/5 pass") or one filtered test command."""
    block: list[str] = []
    for line in following:
        if not line.strip():
            if block:
                break
            continue
        block.append(line.strip())
        if len(block) >= 8:
            break
    if not block:
        return ""
    if block[0].startswith("```"):
        commands = [b for b in block[1:] if not b.startswith("```")]
        if any(TEST_FILE.search(c) or FILTERED.search(c) for c in commands):
            return "the test command shown below it"
        return ""
    items = [b for b in block if b.startswith(("-", "*", "+")) or b[:1].isdigit()]
    if items and any(TEST_FILE.search(i) or GROUP.search(i) for i in items):
        return "the tests listed below it"
    return ""


def _subject_words(text: str) -> list[str]:
    """The words that qualify the tests the pass verb is about: "permission endpoint" in
    "All 53 permission endpoint tests pass", nothing in "All existing tests pass"."""
    match = PASSED.search(text)
    if match is None:
        return []
    found = match.group(0)
    if re.match(TEST_WORD, found, re.IGNORECASE):
        # the last mention of tests before the verb is its subject ("npm test # all tests pass")
        last = list(re.finditer(rf"\b{TEST_WORD}\b", found, re.IGNORECASE))[-1]
        before = text[: match.start() + last.start()]
    else:  # "passes all unit tests": the words between the verb and the tests
        inner = re.search(rf"pass(?:es|ed)?\s+(.*?)\b{TEST_WORD}\b", found, re.IGNORECASE)
        before = inner.group(1) if inner else ""
    # a label before a colon ("**Test Results**: 38/38 tests passing") is not part of the subject
    before = before.rsplit(":", 1)[-1]
    words = [w.strip("`*_'.,:()[]") for w in before.split()][-6:]
    words = [w for w in words if w and not re.fullmatch(r"[\d,]+\+?|\d+/\d+", w)]
    for i in range(len(words) - 1, -1, -1):
        if words[i].lower() in QUALIFIER_STOPS:
            words = words[i + 1 :]
            break
    return [w for w in words if not _decoration(w)]


def _decoration(word: str) -> bool:
    return bool(re.fullmatch(r"[✅✔☑🧪\W]+", word)) or word.lower() in {"x", "[x]"}


def _after_tick(line: str) -> str:
    return TICKED.sub("", line)


def _listed(line: str) -> bool:
    return bool(re.match(r"^\s*(?:[-*+]|\d+\.)\s", line)) or line.startswith("`")


def _count(line: str) -> int | None:
    match = COUNT.search(line)
    if not match:
        return None
    return int(match.group(1) or match.group(2))


def overall(found: list[Claim]) -> str:
    """ "pass", "implied pass", "fail", "mixed" or "none" for a whole description."""
    kinds = {c.kind for c in found}
    if kinds == {"pass"}:
        return "pass" if any(c.explicit for c in found) else "implied pass"
    if kinds == {"fail"}:
        return "fail"
    if kinds:
        return "mixed"
    return "none"
