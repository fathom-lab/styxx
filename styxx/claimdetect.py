# -*- coding: utf-8 -*-
"""styxx.claimdetect — STRUCT-1: is this sentence of agent prose a diff-checkable claim?

The agent-gate boundary RESULT measured diffgate reading 6 sentences of an agent's 2,824 and
misreading half of what it read. The baseline cycle then priced the extractor blind: precision
0.3333, corpus recall near one claim in thirty, and a never-read band whose claim density is
0.0204 — real claims, sitting unexamined.

STRUCT-1 is the licensed repair candidate, specified in
``papers/closed-model-frontier/PREREG_claim_detector_2026_08_30.md`` BEFORE this file existed.
Four conjuncts, all required, plus one exception declared in the prereg:

1. ACTION HEAD    — a change-action verb (frozen list) in simple past / present / imperative.
                    Deliberately the WEAK conjunct: the null control N2 measures this list
                    ALONE at weighted precision 0.2061, so structure must earn its keep on
                    top of it or the candidate dies.
2. CONCRETE OBJECT— a file path, a backtick code span, a symbol-shaped token, a tests/files
                    count, or a scope phrase. Something a diff could actually be checked for.
3. NOT STATIVE    — perfect/pluperfect and stative constructions describe STATE, not this
                    commit's act, and the blind panel labelled them C. The declared exception:
                    NEGATIVE-scope statives ("the rung ladder is untouched") assert a
                    diff-checkable property OF this commit and the panel adjudicated exactly
                    that sentence A.
4. NO OTHER ACTOR — a sentence naming another commit, branch, or cycle as the actor is
                    reporting, not claiming.

`RESULT`-shaped sentences (test totals, CI verdicts, measured rates — the panel's B label)
are recognised separately and are never claims: their evidence lies outside any diff.

Nothing here reads or writes a verdict. The detector is an OBSERVER — diffgate's verdict
logic is untouched by this module, exactly as the epistemics annotation was for OATH.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["detect", "ClaimReading", "STRUCT1_VERSION"]

STRUCT1_VERSION = "struct-1/2026-08-30"

# ── conjunct 1: the action head (the weak conjunct; N2 is exactly this list) ──────────────
_ACTION_STEMS = (
    "chang", "modif", "updat", "edit", "rewrit", "rewrot", "rebuil", "rebuilt", "renam",
    "mov", "add", "creat", "delet", "remov", "drop", "extend", "wir", "fold", "split",
    "collaps", "promot", "demot", "retarget", "relabel", "commit", "ship", "bump", "fix",
    "patch", "land", "introduc", "gut", "strip", "restor", "revert", "touch",
)
# The suffix set is morphology, not vocabulary — widening it does not widen the verb LIST
# (which is what the N2 null measures). `ied`/`ies` were missing in the first cut, so the
# single commonest verb in this corpus, "Modified", never matched at all. Caught by the
# conjunct-1 unit test, fixed as licensed implementation detail, and recorded here because
# the DEV numbers before this fix understated the detector.
_ACTION = re.compile(
    r"\b(?:" + "|".join(_ACTION_STEMS) + r")(?:e|es|ed|s|ing|ted|ped|ied|ies)?\b", re.I)

# Progressive and infinitival forms describe intent or ongoing work, not a completed act by
# this commit ("planning to update", "will be adding"). The panel's tense rule is explicit.
_NON_FINITE = re.compile(
    r"\b(?:to|will|would|should|could|may|might|must|plan(?:s|ned|ning)?|going)\s+"
    r"(?:\w+\s+){0,2}?(?:" + "|".join(_ACTION_STEMS) + r")", re.I)

# ── conjunct 2: something a diff could be checked against ────────────────────────────────
_EXT = (r"py|md|json|jsonl|txt|yml|yaml|toml|cfg|ini|js|ts|tsx|jsx|css|html|tex|sh|ps1|"
        r"bat|ipynb|csv|tsv|npz|npy|pdf|png|jpg|svg|gz|zip|lock|xml|rst|c|h|cpp|rs|go|java")
_PATH = re.compile(rf"[\w./\\-]*[A-Za-z_][\w-]*\.(?:{_EXT})\b")
_BACKTICK = re.compile(r"`[^`\n]+`")
_SYMBOL = re.compile(r"\b(?:def|class|function|method|module|package)\s+[A-Za-z_]\w*|"
                     r"\b[a-z_]+(?:_[a-z0-9]+){1,}\b")          # snake_case identifiers
_COUNT = re.compile(r"\b\d+\s+(?:new\s+)?(?:tests?|files?|lines?|cases?|fixtures?)\b", re.I)
_SCOPE = re.compile(r"\bonly\s+(?:touch\w*|modif\w*|chang\w*|affect\w*)\b", re.I)

# ── conjunct 3: state, not act — and the declared negative-scope exception ───────────────
_STATIVE = re.compile(
    r"\b(?:had|has|have)\s+(?:not\s+|never\s+)?been\b|"          # perfect passive
    r"\b(?:had|hadn't)\s+(?:not\s+)?\w+ed\b|"                    # pluperfect active
    r"\b(?:is|are|was|were|remains?|stays?|sits?|lives?|holds?|carries|contains?|"
    r"exists?)\s+(?:present|in the tree|committed|recorded|stored|listed|there)\b",
    re.I)
_NEG_SCOPE = re.compile(
    r"\b(?:is|are|was|were|stays?|remains?)\s+(?:still\s+)?"
    r"(?:untouched|unchanged|unmodified|intact)\b|"
    r"\b(?:not|never)\s+(?:touched|modified|changed|edited)\b|"
    r"\bleft\s+(?:alone|untouched|unchanged)\b", re.I)

# ── conjunct 4: someone else did it ──────────────────────────────────────────────────────
_OTHER_ACTOR = re.compile(
    r"\b(?:the\s+)?(?:prior|previous|earlier|last|next|following|other)\s+"
    r"(?:commit|cycle|branch|run|pass|release|version)\b|"
    r"\bcommit\s+[0-9a-f]{7,}\b|"
    # A bare commit sha ANYWHERE in the sentence names another commit as the frame of
    # reference, whether or not the verb follows it directly. DEV tuning (licensed by the
    # prereg for implementation detail) after the sole DEV false positive:
    # "cbd2864 before styxx/certify.py was touched." -- the act belongs to that sha's
    # timeline, not to this commit. Requires at least one digit AND one letter so ordinary
    # words ("added", "decade") and pure numbers can never match.
    r"(?<![\w/.-])(?=[0-9a-f]{7,40}(?![\w/.-]))(?=[a-f0-9]*\d)(?=[a-f0-9]*[a-f])"
    r"[0-9a-f]{7,40}(?![\w/.-])|"
    r"\bin\s+a\s+(?:later|follow-?up|separate)\s+commit\b", re.I)

# ── the B band: results whose evidence is outside any diff ───────────────────────────────
_RESULT_SHAPE = re.compile(
    r"\b\d[\d,.]*\s*(?:passed|failed|skipped|xfailed|green|errors?)\b|"
    r"\b(?:AUC|auroc|p\s*=|rate|share|precision|recall|accuracy|mean|median)\b\s*[\d.]|"
    r"\b\d*\.\d+\b.*\b(?:AUC|rate|share|precision|recall|accuracy)\b|"
    r"\bCI\b.*\b(?:green|red|pass|fail)\b|\b\d+\s*/\s*\d+\s+(?:green|passing|checks)\b",
    re.I)


@dataclass
class ClaimReading:
    """What STRUCT-1 saw. Observation only — nothing here decides a verdict."""

    is_claim: bool
    band: str                      # CLAIM | RESULT | NEITHER
    conjuncts: dict                # each frozen conjunct, pass/fail, for audit
    evidence: dict                 # the substrings that satisfied them
    version: str = STRUCT1_VERSION

    def to_dict(self) -> dict:
        return {"is_claim": self.is_claim, "band": self.band,
                "conjuncts": self.conjuncts, "evidence": self.evidence,
                "struct1_version": self.version}


def _first(rx: re.Pattern, s: str):
    m = rx.search(s)
    return m.group(0) if m else None


def detect(sentence: str) -> ClaimReading:
    """Read one sentence of agent prose. Returns the four frozen conjuncts and the band.

    A sentence is a CLAIM iff every conjunct holds. Conjunct failures are reported
    individually so a reader can see exactly which structural requirement was missed —
    the extractor's boundary, made legible per sentence rather than aggregated away.
    """
    s = (sentence or "").strip()

    action = _first(_ACTION, s)
    neg_scope = _first(_NEG_SCOPE, s)
    non_finite = _first(_NON_FINITE, s)
    # The declared exception: a negative-scope stative asserts a diff-checkable property of
    # this commit, so it satisfies the action head even with no action verb of its own.
    has_action = bool((action and not non_finite) or neg_scope)

    obj_ev = (_first(_PATH, s) or _first(_BACKTICK, s) or _first(_COUNT, s)
              or _first(_SCOPE, s) or _first(_SYMBOL, s))
    has_object = obj_ev is not None

    stative = _first(_STATIVE, s)
    # conjunct 3 passes when the sentence is not stative, OR is a negative-scope stative
    not_stative = (stative is None) or bool(neg_scope)

    other = _first(_OTHER_ACTOR, s)
    no_other_actor = other is None

    conjuncts = {
        "action_head": has_action,
        "concrete_object": has_object,
        "not_stative": not_stative,
        "no_other_actor": no_other_actor,
    }
    evidence = {"action": action, "negative_scope": neg_scope, "object": obj_ev,
                "stative": stative, "other_actor": other,
                "non_finite": non_finite}

    is_claim = all(conjuncts.values())
    if is_claim:
        band = "CLAIM"
    elif _RESULT_SHAPE.search(s):
        band = "RESULT"
    else:
        band = "NEITHER"
    return ClaimReading(is_claim=is_claim, band=band, conjuncts=conjuncts, evidence=evidence)


# ── the frozen null controls, so the comparison lives beside the candidate ───────────────

def null_n1(sentence: str) -> bool:
    """N1: flag iff a file path appears anywhere. The dumbest possible reader."""
    return bool(_PATH.search(sentence or ""))


def null_n2(sentence: str) -> bool:
    """N2: flag iff any action verb stem appears. STRUCT-1's own conjunct 1, alone.

    This is the bar the prereg set (weighted precision 0.2061). STRUCT-1 must beat the
    list it is built on, or structure bought nothing.
    """
    return bool(_ACTION.search(sentence or ""))
