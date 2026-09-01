# -*- coding: utf-8 -*-
"""EXTRACTION census for the `tests_pass` template: given a sentence the regex
matched, was a claim actually being made?

Every precision this lab has published measures ADJUDICATION — given that a claim
was extracted, was the verdict right (EXTERNAL-1 0.23, V14 held-out 0.16).
EXTRACTION is different: we know of no measurement of it, ours or anyone's. A
false extraction produces a false accusation exactly as surely as a false
adjudication does, and it is invisible to every panel we have run, because a panel
is only ever shown items the extractor already produced.

Target, verbatim from `styxx/diffgate.py` (`_TEMPLATES`, near line 76):

    ("tests_pass", re.compile(r"\\b(?:all\\s+)?tests\\s+(?:pass|are\\s+passing|green)\\b", re.I))

Population, deliberately parasitic on `flag_rate.py` so it reconciles exactly with
the published 71,016:

  * same SQL (`body IS NOT NULL AND body != ''`, PRs with at least one file row);
  * same reconstruction (`external1_harness.reconstruct`, imported, never
    reimplemented) and the same `parsed != implied` round-trip skip;
  * same DEVELOPMENT / HELD-OUT split (`v14_gates.bucket` on the first five URL
    segments, `< 3` DEVELOPMENT);
  * same summary text (`f"{title}\\n\\n{body}"`) and the same sentence splitter the
    gate uses (`re.split(r"(?<=[.!?])\\s+|\\n+", ...)`), reproduced here with byte
    offsets so each match can be located in the ORIGINAL body. The splitter is
    reproduced rather than imported because `re.split` discards offsets; the
    reproduction is checked against `re.split` on every single PR and any
    divergence is counted and reported.

The gate applies NO filter to `tests_pass` matches: every match in every sentence
becomes a claim (`diffgate.py:328-358`; the `no_evidence` short-circuit at line 357
explicitly exempts this kind). So the match count here must equal `tests_pass_claims`
in `flag_rate.json`. That equality is asserted in the output as a reconciliation
check, not assumed.

TWO GROUPS, KEPT SEPARATE, AND THE SEPARATION IS THE POINT
----------------------------------------------------------
MECHANICAL — a function of the bytes. No judgment, reproducible by anyone who runs
this file. Every rule is stated in `RULES` below in the same words the code uses.

JUDGMENT — a regex approximating a human reading. This lab measured a regex
approximating a human judgment at 0.16 held-out precision. The judgment group here
is the same kind of object and it has had NO panel. It is emitted marked
UNVALIDATED and MUST NOT be quoted as a precision, a rate of false extraction, or
anything else load-bearing. It is a hypothesis generator.

    python papers/closed-model-frontier/extraction_census.py
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import styxx.diffgate as DG                                    # noqa: E402
from external1_harness import reconstruct                      # noqa: E402
from v14_gates import bucket                                   # noqa: E402

DB = HERE / "external1_shelf.sqlite"
OUT = HERE / "extraction_census.json"

EXPECTED_ELIGIBLE = 71016          # v14_gates.json / flag_rate.json, prs_scored
EXPECTED_MATCHES = {               # flag_rate.json, tests_pass_claims
    "development": 1476, "held_out": 4038, "corpus_wide": 5514,
}
EXPECTED_PRS_WITH_MATCH = {        # flag_rate.json, prs_with_tests_pass_claim
    "development": 1236, "held_out": 3402, "corpus_wide": 4638,
}

# The instrument under census. Taken from the shipped module by NAME rather than
# retyped, so this file cannot drift from the gate it is measuring.
TESTS_PASS_RX = dict(DG._TEMPLATES)["tests_pass"]

# The gate's splitter. Reproduced with offsets; verified against re.split per PR.
SPLIT_RX = re.compile(r"(?<=[.!?])\s+|\n+")

EXAMPLES_PER_CATEGORY = 10

# D4. The judgment numbers must carry their status IN SCOPE wherever they can be
# read, not only at the top of the file. A sibling `status` field beside a bare
# integer map is strippable: a consumer who writes
# d["held_out"]["judgment_category"] never touches the sibling and quotes a
# number with no marker attached. So the MARKER IS IN THE KEY NAMES, which a
# consumer must type to reach the integers, and the status string is repeated
# inside the value so that dumping any level of the path still shows it. The
# examples for judgment labels carry the same prefix for the same reason.
JUDGMENT_STATUS = (
    "UNVALIDATED — a regex approximating a human reading, with NO blind panel. "
    "These counts are NOT a precision, NOT a false-extraction rate, and decide "
    "nothing. See .judgment.warning in this file before quoting any of them.")
JUDG_KEY_FLAGS = "judgment_UNVALIDATED_flags_any"
JUDG_KEY_CATEGORY = "judgment_UNVALIDATED_category"
JUDG_COUNTS_KEY = "counts_UNVALIDATED"
JUDG_EXAMPLE_PREFIX = "JUDGMENT_UNVALIDATED:"


# --------------------------------------------------------------------------
# MECHANICAL detectors. Each is a pure function of the summary text.
# --------------------------------------------------------------------------

_FENCE = re.compile(r"^ {0,3}(?P<f>`{3,}|~{3,})")
_BLOCKQUOTE = re.compile(r"^ {0,3}>")
_TASK = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\[(?P<box>[ xX])\]")
_LIST_MARKER = re.compile(r"^ {0,3}(?:[-*+]|\d+[.)])\s")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_MD_LINK = re.compile(r"\[[^\]\n]*\]\([^)\n]*\)")
_AUTOLINK = re.compile(r"<https?://[^>\s]*>")
_BARE_URL = re.compile(r"https?://[^\s)>\]`\"']+")
_BACKTICKS = re.compile(r"`+")

_NEGATION = re.compile(
    r"\b(?:not|cannot|can't|cant|won't|wont|doesn't|doesnt|don't|dont|didn't|didnt"
    r"|isn't|isnt|aren't|arent|wasn't|wasnt|weren't|werent|hasn't|hasnt|haven't"
    r"|havent|couldn't|couldnt|shouldn't|shouldnt|wouldn't|wouldnt|never|no longer"
    r"|fails?\s+to|failed\s+to|failing\s+to|unable\s+to|without)\b", re.I)
NEGATION_WINDOW = 40


def _expand(line: str) -> str:
    return line.expandtabs(4)


def _line_index(text: str):
    """(start, end_exclusive_of_newline, line_text) for every line, in order."""
    out, pos = [], 0
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\n").rstrip("\r")
        out.append((pos, pos + len(body), body))
        pos += len(line)
    return out


def _fenced_and_indented(lines):
    """-> (set of line indices inside a ``` / ~~~ fence,
           set of line indices that are indented code by the rule in RULES)"""
    fenced, opener = set(), None
    for i, (_s, _e, raw) in enumerate(lines):
        m = _FENCE.match(raw)
        if opener is None:
            if m:
                opener = m.group("f")[0], len(m.group("f"))
                fenced.add(i)                    # the fence line itself counts as fence
        else:
            fenced.add(i)
            if m and m.group("f")[0] == opener[0] and len(m.group("f")) >= opener[1]:
                opener = None

    indented = set()
    for i, (_s, _e, raw) in enumerate(lines):
        if i in fenced:
            continue
        exp = _expand(raw)
        if not exp.strip() or len(exp) - len(exp.lstrip(" ")) < 4:
            continue
        # must not interrupt a paragraph: previous line blank, or itself indented
        prev = _expand(lines[i - 1][2]) if i else ""
        prev_blank = (i == 0) or (not prev.strip())
        prev_indented = (i - 1) in indented
        if not (prev_blank or prev_indented):
            continue
        # 4-space indentation inside a list is CONTINUATION, not code. Walk back to
        # the nearest non-blank line at indent < 4 and refuse if it is a list item.
        j, in_list = i - 1, False
        while j >= 0:
            pe = _expand(lines[j][2])
            if not pe.strip():
                j -= 1
                continue
            if len(pe) - len(pe.lstrip(" ")) >= 4:
                j -= 1
                continue
            in_list = bool(_LIST_MARKER.match(pe))
            break
        if in_list:
            continue
        indented.add(i)
    return fenced, indented


def _spans(rx, text):
    return [(m.start(), m.end()) for m in rx.finditer(text)]


def _inline_code_spans(lines):
    """Backtick code spans, paired left-to-right per line by equal run length."""
    spans = []
    for (s, _e, raw) in lines:
        runs = [(m.start(), m.end()) for m in _BACKTICKS.finditer(raw)]
        used = [False] * len(runs)
        for a in range(len(runs)):
            if used[a]:
                continue
            la = runs[a][1] - runs[a][0]
            for b in range(a + 1, len(runs)):
                if used[b] or (runs[b][1] - runs[b][0]) != la:
                    continue
                spans.append((s + runs[a][1], s + runs[b][0]))
                used[a] = used[b] = True
                break
        # unmatched runs open nothing
    return spans


def _in(spans, off):
    return any(a <= off < b for a, b in spans)


def _link_spans(text):
    """(title spans, destination spans) for markdown links, plus autolinks/bare URLs
    which are recorded as destinations."""
    titles, dests = [], []
    for m in _MD_LINK.finditer(text):
        s = m.start()
        close = m.group(0).index("](")
        titles.append((s + 1, s + close))
        dests.append((s + close + 2, m.end() - 1))
    for a, b in _spans(_AUTOLINK, text) + _spans(_BARE_URL, text):
        dests.append((a, b))
    return titles, dests


def mechanical(text, lines, ctx, off, sentence, m_start_in_sent):
    """Every mechanical flag for one match, as a dict of booleans plus box state."""
    li = ctx["line_of"](off)
    raw = lines[li][2]
    flags = {
        "fenced_code_block": li in ctx["fenced"],
        "indented_code_block": li in ctx["indented"],
        "blockquote": bool(_BLOCKQUOTE.match(raw)),
        "html_comment": _in(ctx["comments"], off),
        "link_title": _in(ctx["link_titles"], off),
        "link_url": _in(ctx["link_dests"], off),
        "inline_code_span": _in(ctx["inline_code"], off),
        "task_list_item": False,
        "task_box_unchecked": False,
        "task_box_checked": False,
        "negation_cue_near": False,
        "negation_cue_in_sentence": False,
        "in_title": off < ctx["title_end"],
        # DIAGNOSTICS — never used in the disjoint table. They exist so a reader
        # can tell a detector that found nothing from a detector that is broken.
        "indented_ge4_spaces_naive": (
            li not in ctx["fenced"]
            and bool(_expand(raw).strip())
            and (len(_expand(raw)) - len(_expand(raw).lstrip(" "))) >= 4),
        "backtick_somewhere_on_line": "`" in raw,
        "url_somewhere_on_line": bool(_BARE_URL.search(raw)),
    }
    tm = _TASK.match(raw)
    if tm:
        flags["task_list_item"] = True
        if tm.group("box") in ("x", "X"):
            flags["task_box_checked"] = True
        else:
            flags["task_box_unchecked"] = True
    before = sentence[:m_start_in_sent]
    if _NEGATION.search(before):
        flags["negation_cue_in_sentence"] = True
    win = before[-NEGATION_WINDOW:]
    cues = [c.group(0).lower() for c in _NEGATION.finditer(win)]
    if cues:
        flags["negation_cue_near"] = True
        # DISCLOSED DEFECT, counted rather than quietly patched: "without" is in
        # the cue set for "without modifying X", but it also produces
        # "builds without warnings and all tests pass" — a sentence that
        # ASSERTS. Where "without" is the only cue the flag is unreliable.
        flags["negation_cue_is_without_only"] = set(cues) == {"without"}
    flags["link_title_or_url"] = flags["link_title"] or flags["link_url"]
    return flags


# Disjoint assignment. Containment first (the text is not prose at all), then the
# task box (an explicit refusal to assert), then the rest.
MECH_PRIORITY = [
    "html_comment",
    "fenced_code_block",
    "indented_code_block",
    "blockquote",
    "task_box_unchecked",
    "task_box_checked",
    "link_title_or_url",
    "inline_code_span",
    "negation_cue_near",
]


def mech_category(flags):
    for k in MECH_PRIORITY:
        if flags.get(k):
            return k
    return "no_mechanical_flag"


# --------------------------------------------------------------------------
# JUDGMENT detectors. UNVALIDATED. A regex approximating a human reading.
# --------------------------------------------------------------------------

_SELF_DISCLOSED = re.compile(
    r"\b(?:fail\w*|error\w*|broken|breaks?|breaking|flaky|regress\w*|red\b|"
    r"except\s+for|apart\s+from|other\s+than|still\s+investigating)\b", re.I)
_CONDITIONAL = re.compile(
    r"\b(?:once|after|when|whenever|if|should|shall|will|would|expect\w*|assum\w*|"
    r"pending|until|need\s+to|needs\s+to|make\s+sure|ensure|ensures|ensuring|"
    r"verify|verifies|confirm|confirms|please|hopefully|plan\s+to|going\s+to|"
    r"let'?s|to\s+be\s+sure)\b", re.I)
_LOCAL = re.compile(
    r"\b(?:locally|on\s+my\s+(?:machine|laptop|box|end)|in\s+my\s+environment|"
    r"on\s+my\s+local|local\s+run|локально)\b", re.I)
# Words that may sit immediately before "tests" WITHOUT scoping it. Anything else
# ("auth", "unit", "e2e", "affected", "remaining", ...) is read as a subset scope.
_NOT_A_SCOPE = frozenset({
    "all", "the", "a", "an", "and", "or", "but", "so", "then", "now", "that",
    "these", "those", "my", "our", "its", "their", "this", "as", "if", "when",
    "while", "where", "since", "because", "also", "still", "again",
})
_PREV_WORD = re.compile(r"([A-Za-z][\w-]*)\W*$")


def judgment(sentence, m):
    """UNVALIDATED. Booleans only; the caller assigns the disjoint label."""
    before = sentence[:m.start()]
    pw = _PREV_WORD.search(before)
    scoped = bool(pw and pw.group(1).lower() not in _NOT_A_SCOPE)
    # "all tests pass" — the regex swallowed the "all", so nothing precedes it.
    if m.group(0).lower().startswith("all"):
        scoped = False
    return {
        "self_disclosed_failure": bool(_SELF_DISCLOSED.search(sentence)),
        "conditional_or_future": bool(_CONDITIONAL.search(sentence)),
        "scoped_to_authors_machine": bool(_LOCAL.search(sentence)),
        "scoped_to_a_subset": scoped,
    }


JUDG_PRIORITY = [
    "self_disclosed_failure",
    "conditional_or_future",
    "scoped_to_authors_machine",
    "scoped_to_a_subset",
]


def judg_category(flags):
    for k in JUDG_PRIORITY:
        if flags.get(k):
            return k
    return "bare_unqualified_assertion"


RULES = {
    "fenced_code_block":
        "the match's line lies between a line matching ^ {0,3}(`{3,}|~{3,}) and the "
        "next line closing with the same character at >= the opening length; the "
        "fence lines themselves are counted as inside",
    "indented_code_block":
        "tabs expanded to 4; the line's leading spaces >= 4; it is not inside a "
        "fence; the previous line is blank or is itself indented code by this rule; "
        "and the nearest preceding non-blank line at indent < 4 is not a list-item "
        "marker (^ {0,3}([-*+]|\\d+[.)])\\s) — because 4-space indentation inside a "
        "list is continuation, not code",
    "blockquote": "the match's line matches ^ {0,3}>",
    "html_comment": "the match offset lies inside a <!-- ... --> span (DOTALL, "
                    "non-greedy); an unterminated <!-- is NOT treated as a comment",
    "task_list_item": "the match's line matches ^\\s*([-*+]|\\d+[.)])\\s+\\[( |x|X)\\]; "
                      "the box character decides checked vs unchecked",
    "link_title_or_url":
        "the match offset lies inside the [title] or (destination) span of a "
        "[..](..) link, inside a <http...> autolink, or inside a bare https?:// run",
    "inline_code_span":
        "the match offset lies between two backtick runs of equal length on the "
        "same line, paired left to right",
    "negation_cue_near":
        f"a cue from the closed set (not/cannot/n't forms/never/no longer/fails to/"
        f"unable to/without) occurs in the {NEGATION_WINDOW} characters of the same "
        f"gate-sentence immediately before the match",
    "negation_cue_in_sentence":
        "the same closed cue set, anywhere earlier in the same gate-sentence "
        "(reported alongside as the looser variant; the disjoint table uses the "
        "windowed form)",
}


def blank_counts():
    keys = (MECH_PRIORITY + ["no_mechanical_flag", "task_list_item",
                             "negation_cue_in_sentence", "link_title", "link_url",
                             "in_title", "indented_ge4_spaces_naive",
                             "backtick_somewhere_on_line", "url_somewhere_on_line",
                             "negation_cue_is_without_only"])
    return {
        "matches": 0,
        "prs_with_match": 0,
        "mech_flags_any": {k: 0 for k in keys},
        "mech_category": {k: 0 for k in MECH_PRIORITY + ["no_mechanical_flag"]},
        JUDG_KEY_FLAGS: {
            "status": JUDGMENT_STATUS,
            JUDG_COUNTS_KEY: {k: 0 for k in JUDG_PRIORITY},
        },
        JUDG_KEY_CATEGORY: {
            "status": JUDGMENT_STATUS,
            JUDG_COUNTS_KEY: {
                k: 0 for k in JUDG_PRIORITY + ["bare_unqualified_assertion"]},
        },
        "residual_matches": 0,
    }


def merge(dst, src):
    dst["matches"] += src["matches"]
    dst["prs_with_match"] += src["prs_with_match"]
    dst["residual_matches"] += src["residual_matches"]
    for sec in ("mech_flags_any", "mech_category"):
        for k, v in src[sec].items():
            dst[sec][k] = dst[sec].get(k, 0) + v
    for sec in (JUDG_KEY_FLAGS, JUDG_KEY_CATEGORY):
        assert dst[sec]["status"] == src[sec]["status"] == JUDGMENT_STATUS
        for k, v in src[sec][JUDG_COUNTS_KEY].items():
            d = dst[sec][JUDG_COUNTS_KEY]
            d[k] = d.get(k, 0) + v


def pct(n, d):
    return round(100.0 * n / d, 2) if d else None


def main() -> int:
    con = sqlite3.connect(DB)
    splits = {"development": blank_counts(), "held_out": blank_counts()}
    examples = {}
    skipped_roundtrip = 0
    prs_scored = 0
    splitter_divergences = 0

    for pid, title, body, url in con.execute(
            "SELECT id, title, body, html_url FROM pr "
            "WHERE body IS NOT NULL AND body != ''"):
        rows = con.execute("SELECT filename, status, patch FROM f WHERE pr_id=?",
                           (pid,)).fetchall()
        if not rows:
            continue
        diff, implied = reconstruct(rows)
        parsed, _ = DG.parse_unified_diff(diff)
        if parsed != implied:
            skipped_roundtrip += 1
            continue
        prs_scored += 1

        title = title or ""
        summary = f"{title}\n\n{body}"

        # cheap pre-filter, identical semantics: if the regex fires nowhere in the
        # whole text it cannot fire in any sentence of it
        if not TESTS_PASS_RX.search(summary):
            continue

        # split with offsets, and prove the reproduction matches the gate's re.split
        pieces, pos = [], 0
        for sep in SPLIT_RX.finditer(summary):
            pieces.append((pos, summary[pos:sep.start()]))
            pos = sep.end()
        pieces.append((pos, summary[pos:]))
        if [p[1] for p in pieces] != SPLIT_RX.split(summary):
            splitter_divergences += 1
            continue

        lines = _line_index(summary)
        starts = [s for s, _e, _r in lines]

        def line_of(off, _starts=starts):
            lo, hi = 0, len(_starts) - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if _starts[mid] <= off:
                    lo = mid
                else:
                    hi = mid - 1
            return lo

        fenced, indented = _fenced_and_indented(lines)
        link_titles, link_dests = _link_spans(summary)
        ctx = {
            "fenced": fenced, "indented": indented,
            "comments": _spans(_HTML_COMMENT, summary),
            "link_titles": link_titles, "link_dests": link_dests,
            "inline_code": _inline_code_spans(lines),
            "line_of": line_of,
            "title_end": len(title),
        }

        key = ("development"
               if bucket("/".join((url or "").split("/")[:5])) < 3
               else "held_out")
        s = splits[key]
        hit_this_pr = False

        for pstart, sent in pieces:
            for m in TESTS_PASS_RX.finditer(sent):
                off = pstart + m.start()
                mf = mechanical(summary, lines, ctx, off, sent, m.start())
                mc = mech_category(mf)
                s["matches"] += 1
                hit_this_pr = True
                for k in s["mech_flags_any"]:
                    if mf.get(k):
                        s["mech_flags_any"][k] += 1
                if mc == "no_mechanical_flag":
                    s["mech_flags_any"]["no_mechanical_flag"] += 1
                s["mech_category"][mc] += 1

                cat = mc
                if mc == "no_mechanical_flag":
                    s["residual_matches"] += 1
                    jf = judgment(sent, m)
                    jflags = s[JUDG_KEY_FLAGS][JUDG_COUNTS_KEY]
                    for k in jflags:
                        if jf.get(k):
                            jflags[k] += 1
                    jc = judg_category(jf)
                    s[JUDG_KEY_CATEGORY][JUDG_COUNTS_KEY][jc] += 1
                    cat = JUDG_EXAMPLE_PREFIX + jc

                bucket_ex = examples.setdefault(cat, [])
                if len(bucket_ex) < EXAMPLES_PER_CATEGORY:
                    bucket_ex.append({
                        "split": key,
                        "url": url,
                        "matched_text": m.group(0),
                        "sentence": sent.strip()[:200],
                        "line": lines[line_of(off)][2].strip()[:200],
                    })

        if hit_this_pr:
            s["prs_with_match"] += 1

    con.close()

    both = blank_counts()
    merge(both, splits["development"])
    merge(both, splits["held_out"])

    recon = {
        "expected_eligible": EXPECTED_ELIGIBLE,
        "observed_eligible": prs_scored,
        "eligible_matches_flag_rate": prs_scored == EXPECTED_ELIGIBLE,
        "skipped_reconstruction_roundtrip": skipped_roundtrip,
        "splitter_divergences_from_re_split": splitter_divergences,
        "expected_tests_pass_claims": EXPECTED_MATCHES,
        "observed_matches": {
            "development": splits["development"]["matches"],
            "held_out": splits["held_out"]["matches"],
            "corpus_wide": both["matches"],
        },
        "expected_prs_with_tests_pass_claim": EXPECTED_PRS_WITH_MATCH,
        "observed_prs_with_match": {
            "development": splits["development"]["prs_with_match"],
            "held_out": splits["held_out"]["prs_with_match"],
            "corpus_wide": both["prs_with_match"],
        },
    }
    recon["matches_reconcile_with_flag_rate"] = (
        recon["observed_matches"] == EXPECTED_MATCHES
        and recon["observed_prs_with_match"] == EXPECTED_PRS_WITH_MATCH)

    payload = {
        "what": ("EXTRACTION census of the tests_pass template: of the sentences "
                 "the regex fires on, how many are a claim being made"),
        "target_regex": TESTS_PASS_RX.pattern,
        "target_source": "styxx/diffgate.py _TEMPLATES entry 'tests_pass'",
        "not_measured_here": (
            "adjudication. Every tests_pass verdict in this corpus is UNCHECKABLE "
            "(run=None), so no accusation exists to adjudicate. This file measures "
            "the step BEFORE the verdict."),
        "population": recon,
        "mechanical": {
            "status": "MECHANICAL — a pure function of the summary bytes, "
                      "reproducible by anyone who runs this file",
            "rules": RULES,
            "priority_for_the_disjoint_table": MECH_PRIORITY,
            "note": ("mech_flags_any counts each flag independently and matches may "
                     "carry several; mech_category assigns each match to exactly one "
                     "bucket by the priority above. The headline unchecked-box number "
                     "is the INDEPENDENT count, mech_flags_any.task_box_unchecked."),
            "diagnostics_not_categories": {
                "indented_ge4_spaces_naive":
                    "the NAIVE indented-code rule (>= 4 leading spaces, outside a "
                    "fence) with none of the CommonMark conditions. Reported so the "
                    "cost of the list-continuation refusal is visible: the gap "
                    "between this and indented_code_block is what the strict rule "
                    "declined to call code, and it is almost entirely markdown list "
                    "continuation, not code",
                "backtick_somewhere_on_line":
                    "the match's line contains a backtick at all. If this is large "
                    "while inline_code_span is zero, the code-span detector is "
                    "working and the matches simply sit outside the spans",
                "url_somewhere_on_line":
                    "the match's line contains an http(s) URL at all. Same purpose "
                    "for link_url. Note the regex requires literal whitespace "
                    "between 'tests' and 'pass', which URLs almost never contain — "
                    "a zero here is expected from the pattern, not evidence of a "
                    "bug",
                "negation_cue_is_without_only":
                    "of the negation_cue_near matches, those whose ONLY cue in the "
                    "window is the word 'without'. See disclosed_defects",
            },
            "disclosed_defects": {
                "negation_cue_near_over-fires_on_without":
                    "'without' earns its place in the cue set for 'without "
                    "modifying X', but it also fires on 'the solution builds "
                    "without any warnings and all tests pass' — a sentence that "
                    "ASSERTS. This is a false positive inside the group that is "
                    "supposed to carry weight, so it is counted "
                    "(negation_cue_is_without_only) rather than patched away. "
                    "Subtract it from negation_cue_near for a conservative read",
                "fence_lines_counted_as_inside":
                    "the ``` line itself is counted as fenced. A match can only "
                    "land there via the info string, which is vanishingly rare, "
                    "but the choice is stated rather than hidden",
                "blockquote_lazy_continuation_missed":
                    "a blockquote continued lazily (a following line with no '>') "
                    "is NOT detected. Such matches fall through to the residual "
                    "and are therefore counted against us, not for us",
                "task_item_continuation_lines_missed":
                    "the box state is read from the line containing the match. A "
                    "match on a continuation line under a task item is not "
                    "attributed to that item's box, so the unchecked-box count is "
                    "a LOWER bound",
            },
        },
        "judgment": {
            "status": "UNVALIDATED",
            "warning": (
                "This group is a regex approximating a human judgment. That is "
                "exactly the class of instrument this lab measured at 0.16 held-out "
                "precision (RESULT_v14). It has had NO blind panel. It MUST NOT be "
                "quoted as a precision, as a false-extraction rate, or as any "
                "figure that decides anything. It is a hypothesis generator and "
                "nothing else."),
            "priority_for_the_disjoint_table": JUDG_PRIORITY,
            "applied_to": ("only the matches carrying no mechanical flag "
                           "(mech_category == no_mechanical_flag)"),
            "where_the_numbers_live": {
                "keys": [JUDG_KEY_FLAGS, JUDG_KEY_CATEGORY],
                "counts_under": JUDG_COUNTS_KEY,
                "examples_prefixed": JUDG_EXAMPLE_PREFIX,
                "in_each_of": ["development", "held_out", "corpus_wide"],
            },
            "why_the_keys_are_named_that_way": (
                "An earlier emission of this file marked judgment.status "
                "UNVALIDATED at the top level only, while the per-split counts "
                "sat under a bare `judgment_category` map. A consumer reading "
                "d['held_out']['judgment_category'] saw no marker and could "
                "quote the number with nothing attached to it. A sibling status "
                "field beside the map would not have fixed that, because the "
                "careless read never touches the sibling. So the marker is in "
                "the KEY NAMES, which cannot be reached without being typed, "
                "and the status string is repeated inside the value so that a "
                "dump of any level of the path still carries it. The key names "
                "changed in this emission; that is a deliberate break for any "
                "consumer indexing the old names."),
            "known_defects_found_by_reading_its_own_output": {
                "self_disclosed_failure_fires_on_paths":
                    "the cue 'error\\w*' matched inside a file path in "
                    "'Verified all tests pass with `yarn test "
                    "src/elements/common/error-boundary/__tests__/`'. A path is "
                    "not a disclosure. Left in place, disclosed, because "
                    "hand-patching an unvalidated classifier until its output "
                    "looks right is the failure mode this lab is trying to stop",
                "self_disclosed_failure_fires_on_flaky":
                    "'All tests pass, including the previously flaky test' is "
                    "labelled a self-disclosed failure; a reader might call it an "
                    "assertion. This label is contested and unadjudicated",
                "scoped_to_a_subset_is_the_largest_and_the_weakest":
                    "it is the biggest bucket and it rests entirely on the "
                    "single word before 'tests' against a hand-written stoplist. "
                    "'Verified existing tests pass' and 'All 21 unit tests pass' "
                    "both land here and they are not obviously the same thing. "
                    "This number in particular must not leave the file",
            },
            "stoplist_verbatim": sorted(_NOT_A_SCOPE),
        },
        "development": splits["development"],
        "held_out": splits["held_out"],
        "corpus_wide": both,
        "examples": examples,
        "headline": {},
    }

    uc = both["mech_flags_any"]["task_box_unchecked"]
    ch = both["mech_flags_any"]["task_box_checked"]
    payload["headline"] = {
        "unchecked_task_box_matches_corpus_wide": uc,
        "unchecked_task_box_share_of_all_matches_pct": pct(uc, both["matches"]),
        "checked_task_box_matches_corpus_wide": ch,
        "unchecked_share_of_task_list_matches_pct": pct(
            uc, both["mech_flags_any"]["task_list_item"]),
        "held_out_unchecked": splits["held_out"]["mech_flags_any"]["task_box_unchecked"],
        "development_unchecked": splits["development"]["mech_flags_any"]["task_box_unchecked"],
        "reading": ("an unchecked box is an author explicitly DECLINING to claim "
                    "the tests pass; the extractor reads it as asserting it"),
    }

    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    # ---------------- table ----------------
    def row(label, n, tot):
        return f"  {label:<34}{n:>7}{'':2}{str(pct(n, tot)) + '%':>8}"

    print(f"\nEXTRACTION CENSUS — tests_pass  {TESTS_PASS_RX.pattern}")
    print(f"eligible PRs: {prs_scored} (flag_rate reports {EXPECTED_ELIGIBLE}) "
          f"{'MATCH' if prs_scored == EXPECTED_ELIGIBLE else 'MISMATCH — investigate'}")
    print(f"round-trip skips: {skipped_roundtrip}   "
          f"splitter divergences: {splitter_divergences}")
    print(f"matches: dev {splits['development']['matches']} / "
          f"held-out {splits['held_out']['matches']} / corpus {both['matches']}  "
          f"(flag_rate tests_pass_claims {EXPECTED_MATCHES['corpus_wide']}) "
          f"{'RECONCILES' if recon['matches_reconcile_with_flag_rate'] else 'DOES NOT RECONCILE'}")

    for name, s in (("DEVELOPMENT", splits["development"]),
                    ("HELD-OUT", splits["held_out"]),
                    ("CORPUS-WIDE", both)):
        t = s["matches"]
        print(f"\n{name}  ({t} matches, {s['prs_with_match']} PRs)")
        print("  MECHANICALLY DETERMINABLE — disjoint, by priority")
        for k in MECH_PRIORITY + ["no_mechanical_flag"]:
            print(row(k, s["mech_category"][k], t))
        print("  MECHANICAL — independent flags (a match may carry several)")
        for k in ("task_list_item", "task_box_unchecked", "task_box_checked",
                  "fenced_code_block", "indented_code_block", "blockquote",
                  "html_comment", "link_title", "link_url", "inline_code_span",
                  "negation_cue_near", "negation_cue_in_sentence", "in_title"):
            print(row(k, s["mech_flags_any"][k], t))
        print("  DIAGNOSTIC — not a category; proves the detector saw the material")
        for k in ("indented_ge4_spaces_naive", "backtick_somewhere_on_line",
                  "url_somewhere_on_line", "negation_cue_is_without_only"):
            print(row(k, s["mech_flags_any"][k], t))
        r = s["residual_matches"]
        print(f"  REQUIRES JUDGMENT — UNVALIDATED, NOT A PRECISION "
              f"({r} mechanically-clean matches)")
        jcounts = s[JUDG_KEY_CATEGORY][JUDG_COUNTS_KEY]
        for k in JUDG_PRIORITY + ["bare_unqualified_assertion"]:
            print(row(k, jcounts[k], r))

    print(f"\nHEADLINE  unchecked task boxes read as assertions: "
          f"{uc} of {both['matches']} matches "
          f"({pct(uc, both['matches'])}%), held-out "
          f"{splits['held_out']['mech_flags_any']['task_box_unchecked']}")
    print(f"          checked boxes for comparison: {ch}")
    print(f"\n-> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
