"""spec_claim_coverage — does a rule the specification states have a test that would fail?

WHY THIS EXISTS. Three times on 2026-09-09 the specification was repaired and the implementation
never heard about it: amendment A-29 (the recipe split, still not in `cert.RECIPE_CORE_FIELDS`),
finding S9-01 (the challenge subject check), and the section 6 subject guard. Each was found by
accident — one by a test written for another reason, two by an adversary
(`papers/v8/challenge_and_attack_2026_09_09/`). Nothing systematically asked whether a rule the
spec states is a rule the code keeps.

This is the inventory that asks. It walks `papers/v8/SPEC_v8_v0.2_draft.md`, extracts every
sentence that reads as normative, gives each a stable id, and then asks of each: would any test
under `tests/` fail if the code stopped honouring it?

IT CANNOT ANSWER THAT QUESTION. Nobody can, short of mutating the code once per claim and running
the suite. What it does instead is stated exactly, in `classify()` and in the README beside this
file: it searches the test corpus for the identifiers, verdict literals, exit codes and refusal
strings the claim names, and classifies the claim `has-a-test`, `maybe` or `no-test` on what it
finds. A `has-a-test` means A TEST MENTIONS THE BEHAVIOUR IN A PLACE THAT ASSERTS SOMETHING. It
does not mean the test is correct, and it does not mean the test would fail on a violation.

THE MISS LIST IS THE DELIVERABLE, not the rate. The rate is a property of this program's aperture
as much as of the suite.

WHEN IN DOUBT, CLASSIFY DOWN. Every threshold below is set so that an unclear case lands in
`maybe` rather than `has-a-test`, because an inflated `has-a-test` hides a miss and an inflated
`no-test` only costs a reader time.

Run:  python papers/v8/spec_claim_coverage/inventory.py
Writes spec_claim_coverage.json beside this file and prints the summary and the miss list.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SPEC = ROOT / "papers" / "v8" / "SPEC_v8_v0.2_draft.md"
TESTS = ROOT / "tests"
CODE_DIRS = [ROOT / "styxx" / "v8", ROOT / "styxx" / "_data"]
OUT = HERE / "spec_claim_coverage.json"

# --------------------------------------------------------------------------- parameters
# Every number here is an aperture setting, not a measurement. They are named so a reader can
# change one and re-run rather than guess what a rate depended on.
WINDOW = 60          # lines: the fallback unit of evidence where no function boundary is found
DF_MAX = 60          # a probe found in more than this many test files cannot anchor a search
MIN_SENTENCE = 40    # chars: shorter fragments are merged into the next sentence
MIN_TOKEN = 4        # chars: shorter identifiers are not probes

# --------------------------------------------------------------------------- the aperture
# A sentence is normative iff one of these fires. This is the whole extraction rule.
MARKERS = {
    "MUST": re.compile(r"\bMUST(?:\s+NOT)?\b"),
    "must": re.compile(r"\b(?:must|may not|cannot|can only|shall)\b"),
    "required": re.compile(r"\b(?:required|requires|require|mandatory)\b"),
    "forbidden": re.compile(r"\b(?:forbidden|forbids|forbid)\b"),
    "refuse": re.compile(r"\b(?:refuse|refuses|refused|refusing|refusal|reject|rejects|rejected)\b"),
    "never": re.compile(r"\bnever\b"),
    "always": re.compile(r"\balways\b"),
    "only": re.compile(r"\b(?:iff|only if|only when|only where|the only)\b"),
    "exit": re.compile(r"\bexits?\s+(?:with\s+)?(?:code\s+)?[0-5]\b|\bexit code\b"),
    "valid": re.compile(r"\bis (?:in)?valid\b|\bis well-formed\b|\bis not a\b.{0,40}\bfingerprint\b"),
}

# Verdict, status, kind and role literals. A claim naming one of these names a behaviour that a
# test can print, so they are strong probes even when they are single lowercase words.
VOCAB = {
    "same", "drift", "skew", "inconclusive", "identity", "unavailable", "unmeasured",
    "beyond-floor-coverage", "baseline-gap", "baseline_gap", "coverage-contested", "Disputed",
    "cross-subject", "cross-configuration", "equivocation", "unpublished", "incomplete",
    "TAMPER", "exceeds_floor", "anchor_flips", "skipped_channels", "floor_owner",
    "noise-plan", "sensitivity", "response", "document", "promotion", "challenge", "sublog",
    "fingerprint", "prereg", "battery", "result", "action", "confirmatory", "robustness",
    "pool-v1", "fixed-v1", "canary-v1", "white-box", "black-box", "certified", "weights",
    "alias", "selected_against", "noise_plan", "forced_on", "previous", "canonical",
}
# Strong even alone: a claim naming one of these names a specific decision.
VOCAB_STRONG = {
    "beyond-floor-coverage", "baseline-gap", "baseline_gap", "coverage-contested",
    "cross-subject", "cross-configuration", "equivocation", "exceeds_floor", "anchor_flips",
    "skipped_channels", "floor_owner", "noise-plan", "pool-v1", "fixed-v1", "canary-v1",
    "selected_against", "noise_plan", "forced_on", "TAMPER", "Disputed", "unpublished",
}

# Words that are identifiers in the spec and noise in a test corpus.
STOP = {
    "the", "and", "for", "that", "this", "with", "from", "into", "which", "what", "when",
    "every", "each", "only", "never", "always", "must", "cert", "certs", "test", "tests",
    "spec", "code", "true", "false", "null", "none", "type", "types", "kind", "kinds",
    "body", "name", "names", "value", "values", "list", "lists", "file", "files", "path",
    "line", "lines", "text", "json", "data", "hash", "hashes", "item", "items", "index",
    "size", "case", "form", "rule", "rules", "field", "fields", "here", "there", "over",
    "under", "above", "below", "step", "steps", "section", "sections", "append", "log",
    "logs", "run", "runs", "read", "reads", "same", "state", "half", "part", "one", "two",
    "set", "sets", "key", "keys", "not", "have", "has", "was", "were", "are", "its", "it",
    "self", "true", "print", "prints", "says", "said", "make", "makes", "made", "give",
    "gives", "take", "takes", "does", "did", "can", "may", "will", "would", "should",
}

IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:[.\-][A-Za-z0-9_]+)*")
BACKTICK_RE = re.compile(r"`([^`\n]+)`")
ITALIC_RE = re.compile(r"\*([^*\n]{20,200})\*")
ASSERT_RE = re.compile(
    r"\bassert\b|pytest\.raises|\.raises\(|assert\.|\bthrows\(|\bt\.assert|expect\("
)
# A claim that says something NEVER happens, is FORBIDDEN or is REFUSED is kept by a test only
# where that test asserts an absence or a refusal. A positive assertion about a neighbouring
# field is not evidence for it. This is the rule the A-29 calibration failure bought.
NEGATIVE_RE = re.compile(
    r"pytest\.raises|\.raises\(|assertThrows|\bthrows\(|\bassert not\b|!==?|"
    r"\bnot in\b|refus|reject|invalid|INVALID|Invalid|== \[\]|=== \[\]|is None|"
    r"\bnull\b|EXIT\[|returncode == [1-5]|code == [1-5]"
)
INERT_RE = re.compile(r"^\s*@pytest\.mark\.(xfail|skip|skipif)\b")
DEF_RE = re.compile(r"^(\s*)(?:async\s+)?def\s+(\w+)\s*\(")
JSBLOCK_RE = re.compile(r"^\s*(?:test|it|describe)\s*\(")
V8_TEST_RE = re.compile(r"^tests/(?:test_v8_|v8_fixtures|js/)")
WORD_RE = re.compile(r"[a-z][a-z']+")
NEGATIVE_MARKERS = {"never", "forbidden", "refuse"}
NEGATIVE_PHRASE = re.compile(r"\b(?:may not|must not|cannot|is not|are not|no \w+ is)\b")


# --------------------------------------------------------------------------- spec walking

def section_id(level: int, title: str) -> str:
    t = title.strip().lstrip("#").strip()
    m = re.match(r"^Appendix\s+([A-D])\b", t)
    if m:
        return f"App.{m.group(1)}"
    if t.lower().startswith("amendment ledger"):
        return "ledger"
    m = re.match(r"^(\d+(?:\.\d+)*)\.?\s", t)
    if m:
        return "§" + m.group(1)
    slug = re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")[:32]
    return slug or "preamble"


def split_sentences(paragraph: str) -> list[str]:
    """Split a paragraph into sentences, conservatively.

    A period followed by whitespace and a capital, a backtick, an emphasis marker or a section
    sign ends a sentence, unless the fragment before it is short — `execution`. and App. and
    e.g. all produce short fragments, and merging them into what follows keeps the rule whole.
    """
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z`*§\"(—])", paragraph.strip())
    out: list[str] = []
    carry = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        cand = (carry + " " + p).strip() if carry else p
        if len(cand) < MIN_SENTENCE:
            carry = cand
            continue
        out.append(cand)
        carry = ""
    if carry:
        if out:
            out[-1] = out[-1] + " " + carry
        else:
            out.append(carry)
    return out


def walk_spec(text: str) -> list[dict]:
    """Yield candidate units: prose sentences and table data rows, with their section."""
    units: list[dict] = []
    section = "preamble"
    fence = False
    para: list[str] = []
    table: list[str] = []
    para_start = 0

    def flush_para(end_line: int) -> None:
        nonlocal para
        if not para:
            return
        joined = " ".join(x.strip() for x in para).strip()
        para = []
        if not joined:
            return
        gated = "[OPERATOR-GATED" in joined
        for s in split_sentences(joined):
            units.append(
                {"section": section, "unit": "prose", "line": para_start,
                 "sentence": s, "gated": gated}
            )

    def flush_table(end_line: int) -> None:
        nonlocal table
        rows = table
        table = []
        if len(rows) < 3:
            return
        for r in rows[2:]:                       # row 0 header, row 1 separator
            cells = [c.strip() for c in r.strip().strip("|").split("|")]
            joined = " | ".join(c for c in cells if c)
            if len(joined) < 20:
                continue
            units.append(
                {"section": section, "unit": "table-row", "line": end_line,
                 "sentence": joined, "gated": False}
            )

    for n, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip()
        if line.startswith("```"):
            flush_para(n)
            flush_table(n)
            fence = not fence
            continue
        if fence:
            continue
        if line.startswith("#"):
            flush_para(n)
            flush_table(n)
            m = re.match(r"^(#+)\s*(.*)$", line)
            if m:
                section = section_id(len(m.group(1)), m.group(2))
            continue
        if line.strip().startswith("|"):
            flush_para(n)
            table.append(line)
            continue
        flush_table(n)
        if not line.strip():
            flush_para(n)
            continue
        if re.match(r"^\s*(?:[-*]|\d+\.)\s+", line):   # a bullet is its own unit
            flush_para(n)
            para_start = n
            para.append(re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", line))
            continue
        if not para:
            para_start = n
        para.append(line)
    flush_para(len(text.splitlines()))
    flush_table(len(text.splitlines()))
    return units


def markers_of(sentence: str) -> list[str]:
    return sorted(k for k, rx in MARKERS.items() if rx.search(sentence))


# --------------------------------------------------------------------------- probes

def probes_of(sentence: str) -> dict[str, list[str]]:
    """The searchable names a claim carries.

    strong  — an identifier carrying a dot, an underscore or a hyphen, or a vocabulary literal
              this specification uses as a printed decision.
    weak    — a bare identifier the spec put in backticks.
    message — a refusal string the spec quotes in italics, reduced to its first five words.
    exit    — an exit code the sentence names. Never an anchor: every suite mentions exit codes.
    """
    strong: set[str] = set()
    weak: set[str] = set()
    message: set[str] = set()
    exits: set[str] = set()

    for span in BACKTICK_RE.findall(sentence):
        # A path citation is provenance, not behaviour. `papers/v8/first_log_2026_09_09/` in a
        # sentence about the baseline rule made a test that merely cites the same receipt look
        # like coverage of the rule; it is not, so a span naming a path contributes nothing.
        if re.search(r"[\w.-]+/[\w.-]+", span) or re.search(r"\d{4}_\d{2}_\d{2}", span):
            continue                      # a path, not a formula: `distance / floor` survives
        machine = bool(re.search(r"[\^${}\[\]\\|]", span))     # a regex or a JSON fragment
        toks = IDENT_RE.findall(span)
        for tok in toks:
            if len(tok) < MIN_TOKEN or tok.lower() in STOP:
                continue
            if any(c in tok for c in "._-") or tok in VOCAB_STRONG or tok in VOCAB:
                strong.add(tok)
            else:
                weak.add(tok)                        # promoted to strong below when it is rare
        # `verify --ref`, `log append`, `battery fixed`: a command is two words, and two words
        # co-occurring is a much better probe than either alone. The hyphen makes the lookup
        # resolve it as a conjunction of its parts on neighbouring lines. Not from a regex or a
        # JSON fragment: `^sha256:[0-9a-f]{64}$` yields nothing a test could be searched for.
        parts = [t.lower() for t in toks if len(t) >= 3 and "." not in t]
        if len(parts) >= 2 and not machine:
            strong.add("-".join(parts[:2]))
    for v in VOCAB_STRONG:                        # named outside backticks too
        if re.search(r"(?<![A-Za-z0-9_])" + re.escape(v) + r"(?![A-Za-z0-9_])", sentence):
            strong.add(v)
    # hyphenated terms of art the spec uses unquoted: comparability-gating, append-only,
    # content-addressed, beyond-floor-coverage.
    for term in re.findall(r"(?<![`\w-])([a-z]{4,}(?:-[a-z]{4,})+)(?![\w-])", sentence):
        strong.add(term)
    for phrase in ITALIC_RE.findall(sentence):
        if not re.search(r"refus|reject|is not|does not|:", phrase):
            continue
        words = WORD_RE.findall(phrase.lower())
        if len(words) >= 5:
            message.add(" ".join(words[:5]))
    for m in re.finditer(r"\bexits?\s+(?:with\s+)?(?:code\s+)?([0-5])\b", sentence):
        exits.add("exit:" + m.group(1))
    return {"strong": sorted(strong), "weak": sorted(weak - strong),
            "message": sorted(message), "exit": sorted(exits)}


# --------------------------------------------------------------------------- corpora

def inert_ranges(lines: list[str]) -> list[tuple[int, int]]:
    """Line ranges (1-based, inclusive) of test functions marked xfail/skip.

    A hit inside one of these is not coverage. It is a recorded drift: the assertion exists and
    does not run, which is exactly the state `test_v8_spec_agreement.py` is in for A-29.
    """
    out: list[tuple[int, int]] = []
    i = 0
    while i < len(lines):
        if INERT_RE.match(lines[i]):
            j = i
            while j < len(lines) and not re.match(r"\s*(?:async\s+)?(?:def|class)\s", lines[j]):
                j += 1
            if j >= len(lines):
                break
            indent = len(lines[j]) - len(lines[j].lstrip())
            k = j + 1
            while k < len(lines):
                s = lines[k]
                if s.strip() and (len(s) - len(s.lstrip())) <= indent and not s.lstrip().startswith("#"):
                    break
                k += 1
            out.append((i + 1, k))
            i = k
        else:
            i += 1
    return out


def blocks_of(lines: list[str], is_py: bool) -> list[tuple[str, int, int]]:
    """The unit of evidence: one test function.

    Two probe hits in one function are one place. Two probe hits sixty lines apart that happen to
    straddle a `def` are two different tests, and treating them as one place is how a claim
    collects coverage it does not have — the A-29 case, where a test about `noise_floor.covers`
    sat beside an unrelated test that refused something.
    """
    out: list[tuple[str, int, int]] = []
    if is_py:
        starts: list[tuple[int, int, str]] = []
        for n, s in enumerate(lines, start=1):
            m = DEF_RE.match(s)
            if m:
                starts.append((n, len(m.group(1)), m.group(2)))
        for i, (n, indent, name) in enumerate(starts):
            end = len(lines)
            for n2, indent2, _ in starts[i + 1:]:
                if indent2 <= indent:
                    end = n2 - 1
                    break
            out.append((name, n, end))
    else:
        starts = [(n, 0, "block") for n, s in enumerate(lines, start=1) if JSBLOCK_RE.match(s)]
        for i, (n, _, name) in enumerate(starts):
            end = starts[i + 1][0] - 1 if i + 1 < len(starts) else len(lines)
            out.append((f"{name}@{n}", n, end))
    return out


class Corpus:
    """An inverted index over a set of files: token -> {file: [line numbers]}."""

    def __init__(self, files: list[Path], root: Path):
        self.root = root
        self.index: dict[str, dict[str, list[int]]] = {}
        self.assert_lines: dict[str, set[int]] = {}
        self.negative_lines: dict[str, set[int]] = {}
        self.inert: dict[str, list[tuple[int, int]]] = {}
        self.blocks: dict[str, list[tuple[str, int, int]]] = {}
        self.text: dict[str, str] = {}
        self.files: list[str] = []
        for f in files:
            try:
                raw = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = f.relative_to(root).as_posix()
            self.files.append(rel)
            self.text[rel] = raw.lower()
            lines = raw.splitlines()
            self.assert_lines[rel] = {
                n for n, s in enumerate(lines, start=1) if ASSERT_RE.search(s)
            }
            # A negative assertion, not merely a negative word: the line must both assert and
            # express an absence or a refusal. `assert covered & not_covered == set()` in a
            # test about something else is not evidence that a refusal is tested.
            self.negative_lines[rel] = {
                n for n, s in enumerate(lines, start=1)
                if NEGATIVE_RE.search(s) and ASSERT_RE.search(s)
            }
            self.inert[rel] = inert_ranges(lines) if f.suffix == ".py" else []
            self.blocks[rel] = blocks_of(lines, f.suffix == ".py")
            for n, s in enumerate(lines, start=1):
                for tok in IDENT_RE.findall(s):
                    self._add(tok, rel, n)
                    if any(c in tok for c in ".-"):
                        for part in re.split(r"[.\-]", tok):
                            if len(part) >= MIN_TOKEN:
                                self._add(part, rel, n)
        self.df = {t: len(v) for t, v in self.index.items()}

    def block_at(self, rel: str, line: int) -> tuple[str, int, int]:
        """The test function containing a line.

        The innermost containing block whose name starts with `test`, so a helper nested inside a
        test still reports the test; failing that the innermost block of any kind; failing that a
        60-line window. Only a `test`-named block is ever accepted as evidence (see classify).
        """
        best: tuple[str, int, int] | None = None
        best_test: tuple[str, int, int] | None = None
        for name, a, b in self.blocks.get(rel, []):
            if not (a <= line <= b):
                continue
            if best is None or (b - a) < (best[2] - best[1]):
                best = (name, a, b)
            if name.startswith("test") and (
                    best_test is None or (b - a) < (best_test[2] - best_test[1])):
                best_test = (name, a, b)
        return best_test or best or ("<module>", line, line + WINDOW)

    def _add(self, tok: str, rel: str, line: int) -> None:
        self.index.setdefault(tok, {}).setdefault(rel, []).append(line)

    def lookup(self, probe: str) -> dict[str, list[int]]:
        """Hits for one probe. A dotted probe that is not a literal token in the corpus is
        resolved as a conjunction of its parts on the same line, which is how a test writes
        `body["noise_floor"]["covers"]`."""
        if probe.startswith("exit:"):
            hits: dict[str, list[int]] = {}
            for name in ("EXIT", "returncode", "exit_code", "code"):
                for rel, lns in self.index.get(name, {}).items():
                    hits.setdefault(rel, []).extend(lns)
            return {rel: sorted(set(lns)) for rel, lns in hits.items()}
        if probe in self.index:
            return self.index[probe]
        if any(c in probe for c in ".-"):
            parts = [p for p in re.split(r"[.\-]", probe) if len(p) >= 3]
            if len(parts) < 2:
                return {}
            sets = [self.index.get(p, {}) for p in parts]
            if not all(sets):
                return {}
            common = set(sets[0])
            for s in sets[1:]:
                common &= set(s)
            out: dict[str, list[int]] = {}
            for rel in common:
                lines_by_part = [set(s[rel]) for s in sets]
                near: set[int] = set()
                for ln in lines_by_part[0]:
                    if all(any(abs(ln - o) <= 2 for o in s) for s in lines_by_part[1:]):
                        near.add(ln)
                if near:
                    out[rel] = sorted(near)
            return out
        return {}

    def phrase(self, text: str) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        needle = text.lower()
        for rel, body in self.text.items():
            if needle in body:
                pos = body.index(needle)
                out[rel] = [body.count("\n", 0, pos) + 1]
        return out

    def is_inert(self, rel: str, line: int) -> bool:
        return any(a <= line <= b for a, b in self.inert.get(rel, []))


# --------------------------------------------------------------------------- classification

def classify(claim: dict, tests: Corpus, code: Corpus) -> dict:
    """The whole of the method, in one function, so it can be argued with.

    A claim's names are sorted into ANCHORS — the strong ones (a dotted or underscored
    identifier, a decision literal, a quoted refusal) that occur in at most DF_MAX test files —
    and COMPANIONS: bare words and exit codes, which corroborate a place and can never point at
    one. Only an anchor is evidence that a test is about this claim at all.

    no-test      reason `no-probe`   the sentence carries no strong name: no dotted identifier,
                                     no vocabulary literal, no quoted refusal. This is a hole in
                                     the instrument, not a measured miss, and it is listed
                                     separately for that reason.
                 reason `generic-only` its strong names are corpus-wide and point at nothing.
                 reason `no-hit`     it names something specific and nothing in tests/ mentions it.
                 reason `xfail-only` the only mentions are inside xfail/skip-marked functions.
                                     An assertion that does not run is a recorded drift, not
                                     coverage — this is the A-29 state exactly.
    maybe        `mentioned-not-pinned` an anchor is mentioned in a live test, but no single test
                                     function gathers TWO of this claim's anchors beside an
                                     assertion of the right polarity.
    has-a-test   one `test`-named function, in a v8 test file, mentions two or more distinct
                 anchors of this claim — a compound and its own parts counting once — and asserts
                 something there; and, when the claim says something is never done, is forbidden
                 or is refused, that same function also carries a line that both asserts and
                 expresses an absence or a refusal.

    `has-a-test` therefore means A TEST NAMES THE BEHAVIOUR IN A PLACE THAT ASSERTS SOMETHING.
    It is not a claim that the test is correct, that it is the right test, or that it would fail
    on a violation.
    """
    # A bare backticked word the spec uses is a name too, when the corpus does not use it
    # everywhere: `refs` in 13 test files discriminates, `same` in 110 does not. The df cut is
    # what separates them, and it is measured on this corpus, not assumed.
    strong_all = list(claim["probes_strong"]) + [
        p for p in claim["probes_weak"] if 0 < tests.df.get(p, 0) <= DF_MAX
    ]
    weak_all = [p for p in claim["probes_weak"] if p not in strong_all]
    msgs = claim["probes_message"]
    exits = claim["probes_exit"]

    # An anchor is a strong name that is not corpus-wide. A name with NO hits at all is still an
    # anchor: it is the search that came back empty, which is the finding.
    anchors = [p for p in strong_all if tests.df.get(p, 0) <= DF_MAX]
    anchors += ["msg:" + m for m in msgs]
    companions = [p for p in strong_all + weak_all if p not in anchors] + exits
    anchor_set = set(anchors)

    if not strong_all and not msgs:
        return {"classification": "no-test", "reason": "no-probe", "evidence": [],
                "hit_files": [], "inert_hits": [], "anchors": [], "companions": companions}
    if not anchors:
        return {"classification": "no-test", "reason": "generic-only", "evidence": [],
                "hit_files": [], "inert_hits": [], "anchors": [], "companions": companions}

    hits: dict[str, list[tuple[int, str]]] = {}
    anchor_hits: dict[str, set[int]] = {}
    inert: list[str] = []
    inert_anchor: list[str] = []
    for p in [a for a in anchors if not a.startswith("msg:")] + companions:
        for rel, lns in tests.lookup(p).items():
            for ln in lns:
                if tests.is_inert(rel, ln):
                    inert.append(f"{rel}:{ln}:{p}")
                    if p in anchor_set:
                        inert_anchor.append(f"{rel}:{ln}:{p}")
                    continue
                hits.setdefault(rel, []).append((ln, p))
                if p in anchor_set:
                    anchor_hits.setdefault(rel, set()).add(ln)
    for m in msgs:
        for rel, lns in tests.phrase(m).items():
            for ln in lns:
                if tests.is_inert(rel, ln):
                    inert.append(f"{rel}:{ln}:msg")
                    inert_anchor.append(f"{rel}:{ln}:msg")
                    continue
                hits.setdefault(rel, []).append((ln, "msg:" + m))
                anchor_hits.setdefault(rel, set()).add(ln)

    if not anchor_hits:
        reason = "xfail-only" if inert_anchor else "no-hit"
        return {"classification": "no-test", "reason": reason, "evidence": [],
                "hit_files": [], "inert_hits": sorted(set(inert_anchor or inert))[:6],
                "anchors": anchors, "companions": companions}

    negative = bool(set(claim["markers"]) & NEGATIVE_MARKERS) or bool(
        NEGATIVE_PHRASE.search(claim["sentence"]))
    best: list[dict] = []
    for rel, alines in anchor_hits.items():
        # Evidence for a v8 spec claim must live in a v8 test. A match in a 7.x test file is a
        # word collision far more often than it is coverage — `refusal` and `response` in
        # tests/test_attack_v0.py are about a different system — and this whole instrument is
        # built to not accept that.
        if not V8_TEST_RE.match(rel):
            continue
        seen_blocks: set[tuple[int, int]] = set()
        for aline in sorted(alines):
            name, a, b = tests.block_at(rel, aline)
            if not (name.startswith("test") or name.startswith("block@")):
                continue                      # a fixture or a helper is not a test
            if (a, b) in seen_blocks:
                continue
            seen_blocks.add((a, b))
            names = {p for ln, p in hits[rel] if a <= ln <= b}
            # Two anchors, and a compound does not count twice with its own parts:
            # `verify --diff` and `diff` are one observation, not two.
            fams: list[frozenset] = []
            for n in sorted(names & anchor_set, key=len, reverse=True):
                f = frozenset(x for x in re.split(r"[.\-:]", n) if x)
                if not any(f <= g for g in fams):
                    fams.append(f)
            if len(fams) < 2:
                continue
            if not any(a <= x <= b for x in tests.assert_lines.get(rel, set())):
                continue
            if negative and not any(a <= x <= b for x in tests.negative_lines.get(rel, set())):
                continue
            best.append({"file": rel, "function": name, "lines": [a, b],
                         "probes": sorted(names)})
    if best:
        best.sort(key=lambda e: (-len(e["probes"]), e["file"], e["lines"][0]))
        return {"classification": "has-a-test",
                "reason": "one-test-function-names-it-and-asserts",
                "evidence": best[:3], "hit_files": sorted(hits)[:6],
                "inert_hits": sorted(set(inert))[:6],
                "anchors": anchors, "companions": companions}
    return {"classification": "maybe", "reason": "mentioned-not-pinned", "evidence": [],
            "hit_files": sorted(hits)[:6], "inert_hits": sorted(set(inert))[:6],
            "anchors": anchors, "companions": companions}


def implemented(claim: dict, code: Corpus) -> str:
    """Does the v8 implementation mention what this claim names? `no` on a claim with strong
    probes is the S9-01 shape: a rule in the spec text and in no code path."""
    strong = list(claim["probes_strong"])
    if not strong:
        return "unknown"
    found = sum(1 for p in strong if code.lookup(p))
    if found == 0:
        return "no"
    if found < len(strong):
        return "partial"
    return "yes"


LOAD_BEARING = re.compile(r"^§(?:2|3|5|6|8|9)(?:\.|$)")
PROCESS = re.compile(r"^(?:§(?:0|11|12|13|14|15)(?:\.|$)|ledger|preamble)")

SEVERITY = [
    (3, "append-refusal", re.compile(r"\brefus|\breject|at append|is rejected|is refused")),
    (3, "verdict-or-exit", re.compile(r"\bexits?\s+[0-5]\b|\bverdict\b|`same`|`drift`|`identity`|`skew`|inconclusive")),
    (3, "integrity", re.compile(r"signature|canonical|Merkle|leaf hash|STH|root_hash|log_id|roster|TAMPER|JCS|sha256|preimage")),
    (2, "floor-or-coverage", re.compile(r"floor|coverage|sensitivity|alpha_|nuisance|noise_floor|noise plan")),
    (1, "must", re.compile(r"\bMUST\b|\bmust\b")),
]


def severity(claim: dict, impl: str) -> tuple[int, list[str]]:
    score, why = 0, []
    for pts, name, rx in SEVERITY:
        if rx.search(claim["sentence"]):
            score += pts
            why.append(name)
    if LOAD_BEARING.match(claim["section"]):
        score += 2
        why.append("load-bearing-section")
    if PROCESS.match(claim["section"]):
        score -= 2
        why.append("process-section")
    if impl == "no" and claim["probes_strong"]:
        score += 2
        why.append("no-code-trace")
    return score, why


# --------------------------------------------------------------------------- calibration
# The three drifts of 2026-09-09, each pinned by a substring of the sentence in the spec that
# states the rule. Their ground truth is NOT all the same today, and that is the calibration:
# one is still open and two were repaired hours ago, so the set contains both a negative and a
# positive control. See the README.
CALIBRATION = [
    {
        "id": "A-29",
        "what": "the recipe split: `execution` is nuisance, never comparability-gating",
        "section": "§2.3",
        "needle": "never comparability-gating",
        "state_2026_09_09": "OPEN — cert.RECIPE_CORE_FIELDS still gates on the whole `decoding` block",
        "expect": ["no-test", "maybe"],
        "why": (
            "the code does not honour it, so no test can fail when it stops honouring it. "
            "tests/test_v8_spec_agreement.py holds the assertion under a strict xfail, which is a "
            "recorded drift and not coverage — a method that counts it as coverage is wrong."
        ),
    },
    {
        "id": "S9-01",
        "what": "a challenge is valid only on equal subject identity",
        "section": "§9",
        "needle": "A challenge is valid iff the challenger's fingerprint is comparable",
        "state_2026_09_09": (
            "REPAIRED SINCE the attack pass: cert.challenge_validity exists and is asserted in "
            "tests/test_v8_cert.py; it was `no-test` when the attack ran"
        ),
        "expect": ["has-a-test", "maybe"],
        "why": (
            "a positive control. It was in the spec and in no code path at 2026-09-09 18:00 and it "
            "is in both now, so a method that still calls it `no-test` is blind to real coverage."
        ),
    },
    {
        "id": "S6-subject-guard",
        "what": "`identity` is emitted when a subject identity field differs from the run",
        "section": "§5.2",
        "needle": "is a subject verdict, not a channel verdict",
        "state_2026_09_09": (
            "REPAIRED SINCE the attack pass: TransformersRunner.subject exists and "
            "tests/test_v8_subject_guard.py pins C2; it was dead code when the attack ran"
        ),
        "expect": ["has-a-test", "maybe"],
        "why": "the second positive control, same reasoning as S9-01.",
    },
]


def run_calibration(claims: list[dict]) -> list[dict]:
    out = []
    for spec in CALIBRATION:
        found = [
            c for c in claims
            if c["section"] == spec["section"] and spec["needle"] in c["sentence"]
        ]
        row = dict(spec)
        row["matched_claim_ids"] = [c["id"] for c in found]
        row["actual"] = sorted({c["classification"] for c in found})
        if not found:
            row["verdict"] = "NOT EXTRACTED — the aperture does not see this rule"
        elif all(c["classification"] in spec["expect"] for c in found):
            row["verdict"] = "pass"
        else:
            row["verdict"] = "FAIL"
        out.append(row)
    return out


# --------------------------------------------------------------------------- main

def build() -> dict:
    if not SPEC.exists():
        sys.exit(f"specification not found at {SPEC}")
    text = SPEC.read_text(encoding="utf-8")
    units = walk_spec(text)

    test_files = sorted(
        [p for p in TESTS.rglob("*.py") if "__pycache__" not in p.parts]
        + [p for p in TESTS.rglob("*.js") if "node_modules" not in p.parts]
    )
    code_files: list[Path] = []
    for d in CODE_DIRS:
        if d.exists():
            code_files += [p for p in d.rglob("*.py") if "__pycache__" not in p.parts]
            code_files += list(d.rglob("*.js"))
    tests = Corpus(test_files, ROOT)
    code = Corpus(sorted(code_files), ROOT)

    claims: list[dict] = []
    gated = 0
    seen: set[str] = set()
    for u in units:
        marks = markers_of(u["sentence"])
        if not marks:
            continue
        if u["section"] == "ledger":
            continue                                  # amendment history is not a rule
        if u["gated"]:
            gated += 1
            continue                                  # an undecided block states no rule
        p = probes_of(u["sentence"])
        cid = u["section"] + "#" + hashlib.sha256(
            u["sentence"].encode("utf-8")).hexdigest()[:10]
        if cid in seen:
            continue
        seen.add(cid)
        claims.append({
            "id": cid,
            "section": u["section"],
            "unit": u["unit"],
            "spec_line": u["line"],
            "markers": marks,
            "sentence": u["sentence"],
            "probes_strong": p["strong"],
            "probes_weak": p["weak"],
            "probes_message": p["message"],
            "probes_exit": p["exit"],
        })

    for c in claims:
        c.update(classify(c, tests, code))
        c["implemented"] = implemented(c, code)
        c["severity"], c["severity_why"] = severity(c, c["implemented"])

    counts = {"has-a-test": 0, "maybe": 0, "no-test": 0}
    for c in claims:
        counts[c["classification"]] += 1

    def row(c: dict) -> dict:
        return {k: c[k] for k in
                ("id", "section", "spec_line", "severity", "severity_why", "reason",
                 "implemented", "probes_strong", "anchors", "inert_hits", "sentence")}

    order = lambda c: (-c["severity"], c["section"], c["id"])          # noqa: E731
    measured = sorted(
        [c for c in claims
         if c["classification"] == "no-test"
         and c["reason"] in ("no-hit", "xfail-only", "generic-only")],
        key=order,
    )
    blind = sorted(
        [c for c in claims if c["classification"] == "no-test" and c["reason"] == "no-probe"],
        key=order,
    )
    unpinned = sorted([c for c in claims if c["classification"] == "maybe"], key=order)
    # The shape of all three drifts of 2026-09-09: a rule stated in the spec whose names appear
    # nowhere in styxx/v8 and which no test pins. This is the list to read first.
    spec_only = sorted(
        [c for c in claims
         if c["implemented"] == "no" and c["classification"] != "has-a-test"],
        key=order,
    )

    return {
        "generated_by": "papers/v8/spec_claim_coverage/inventory.py",
        "spec": SPEC.relative_to(ROOT).as_posix(),
        "spec_sha256": hashlib.sha256(SPEC.read_bytes()).hexdigest(),
        "test_corpus": {
            "root": TESTS.relative_to(ROOT).as_posix(),
            "files": len(tests.files),
        },
        "code_corpus": {
            "dirs": [d.relative_to(ROOT).as_posix() for d in CODE_DIRS if d.exists()],
            "files": len(code.files),
        },
        "parameters": {
            "window_lines": WINDOW, "df_max": DF_MAX,
            "min_sentence_chars": MIN_SENTENCE, "min_token_chars": MIN_TOKEN,
            "markers": sorted(MARKERS),
        },
        "excluded": {
            "operator_gated_sentences": gated,
            "amendment_ledger": "not extracted; it is history, not a rule",
            "fenced_code_blocks": "not extracted; a schema is not a sentence",
        },
        "counts": {
            "claims_total": len(claims),
            "has_a_test": counts["has-a-test"],
            "maybe": counts["maybe"],
            "no_test": counts["no-test"],
            "no_test_measured": len(measured),
            "no_test_invisible_to_this_method": len(blind),
        },
        "calibration": run_calibration(claims),
        "miss_list": [row(c) for c in measured],
        "spec_only_rules": [row(c) for c in spec_only],
        "not_pinned_maybe": [row(c) for c in unpinned],
        "invisible_to_this_method": [row(c) for c in blind],
        "claims": claims,
    }


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                    # a console that cannot be reconfigured
        pass
    report = build()
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8", newline="\n")
    c = report["counts"]
    print(f"claims extracted : {c['claims_total']}")
    print(f"  has-a-test     : {c['has_a_test']}")
    print(f"  maybe          : {c['maybe']}")
    print(f"  no-test        : {c['no_test']}"
          f"  ({c['no_test_measured']} searched and not found,"
          f" {c['no_test_invisible_to_this_method']} unsearchable)")
    print(f"operator-gated sentences skipped: {report['excluded']['operator_gated_sentences']}")
    print()
    print("calibration")
    for row in report["calibration"]:
        print(f"  {row['id']:<18} expect {row['expect']} got {row['actual']} -> {row['verdict']}")
    print()
    print("MISS LIST — no test names it, most damaging first")
    for m in report["miss_list"]:
        s = m["sentence"]
        print(f"  [{m['severity']:>2}] {m['id']}  ({m['reason']}, code:{m['implemented']})")
        print(f"        {s[:160]}{'...' if len(s) > 160 else ''}")
    print()
    print("SPEC-ONLY RULES — named nowhere in styxx/v8 and pinned by no test "
          f"({len(report['spec_only_rules'])})")
    for m in report["spec_only_rules"][:15]:
        print(f"  [{m['severity']:>2}] {m['id']}  ({m['reason']}) {m['anchors'][:3]}")
        print(f"        {m['sentence'][:150]}{'...' if len(m['sentence']) > 150 else ''}")
    print()
    print(f"NOT PINNED (maybe) — a related test exists, none pins the claim "
          f"({len(report['not_pinned_maybe'])}); top 12")
    for m in report["not_pinned_maybe"][:12]:
        print(f"  [{m['severity']:>2}] {m['id']}  {m['sentence'][:130]}")
    print()
    print(f"invisible to this method (no searchable name): "
          f"{c['no_test_invisible_to_this_method']}; listed in the json")
    print(f"written: {OUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
