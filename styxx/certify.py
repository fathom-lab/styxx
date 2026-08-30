"""styxx.certify — OATH v0: the certificate-carrying document.

Wires the demarcation rung (abstain on what cannot be verified) into a deployable artifact: given a
research/report markdown and the receipt JSONs it cites, extract every NUMERIC claim, verify each against
the receipts, and emit a machine-checkable certificate. Anyone can re-run it; trust is a measurement.

Claim classes:
  VERIFIED      the doc number matches a receipt value (rounding-aware, percent/fraction-aware).
  ABSTAIN       not checkable against the provided receipts (the oath says so LOUDLY).
  UNGROUNDED    the doc number sits inside a context that names a receipt-kind quantity, but NO provided
                receipt grounds it -> THE OATH FAILS. Covers both a genuine contradiction (receipt value
                disagrees) and a missing receipt (the number was computed but never persisted) — the
                certificate refuses to swear in either case; repairing the receipt set is the cure for
                the second. [Disclosed pre-validation amendment: the prereg named this class CONTRADICTED
                ("conflicts with the receipt it should match"), but claim->field binding strong enough to
                prove a CONFLICT is beyond v0; the pilot exposed the gap, so the class is renamed and
                broadened BEFORE the frozen D1/D2/D3 run, bars unchanged.]

v0 scope (stated, not hidden): numeric claims only; receipts are the explicit set passed in (the doc's own
cited result JSONs), not discovered; no semantic/prose entailment (that is audit_claim/NLI territory).

Pre-registration (kill-gates D1/D2/D3, frozen before the validation run):
  papers/closed-model-frontier/PREREG_oath_v0_certify_doc_2026_06_09.md

CLI:
  python -m styxx.certify DOC.md receipt1.json [receipt2.json ...] [--out CERT.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

__all__ = ["extract_numbers", "receipt_values", "certify_doc"]

# ---------------------------------------------------------------- numeric extraction (doc side)

# a number token in prose: optional sign, digits, optional decimal part; tolerate thousands commas.
_NUM = re.compile(r"(?<![\w.])[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w.])|(?<![\w.])[-+]?\d+\.\d+(?![\w.])"
                  r"|(?<![\w.])[-+]?\.\d+(?![\w.])|(?<![\w.])[-+]?\d+(?![\w.])")
# numbers we never try to ground (calendar years, semver-ish, sha fragments are filtered by context)
_YEAR = re.compile(r"^(19|20)\d{2}$")
_DATEISH = re.compile(r"\d{4}[-_]\d{2}[-_]\d{2}")
# v0.6.2 (PREREG_oath_v062_signed_extraction_2026_07_31): the scrub requires >=1 letter — an
# all-digit span of >=7 chars is a decimal fraction / count / identifier, not a hash, and the
# digit-only form was eating the fractional part of every full-precision decimal (>=7 fractional
# digits), leaving them invisible to extraction: certified-by-omission, the inverse of the oath.
_SHAISH = re.compile(r"\b(?=[0-9a-f]*[a-f])[0-9a-f]{7,64}\b")
_VERSIONISH = re.compile(r"\bv?\d+\.\d+\.\d+\b")


def _decimals(tok: str) -> int:
    return len(tok.split(".")[1]) if "." in tok else 0


_TABLE_SEP = re.compile(r"^\s*\|[\s:|-]+\|?\s*$")
_FORMULA_AFTER = re.compile(r"^\s?[−–-]\s?[A-Za-z]")   # '1−syc', '1-dec': notation, not a claim
# markdown STRUCTURE lines whose leading small int is an artifact (heading number,
# bullet content marker, blockquote, footnote def) rather than a claim. The line-start
# filter below applies ONLY on these: an unconditional positional drop made every
# line-initial single-digit count invisible to extraction ("9/12 held" led its line ->
# the doctored 9 never entered the ledger -> OATH-HELD by omission, the inverse of the
# oath), while the identical token mid-line was correctly flagged.
_MD_STRUCTURE = re.compile(r"^\s{0,3}(?:#{1,6}\s|[-*+]\s|>\s|\[\^?\d+\]:)")


def _table_rows(lines: list[str]) -> dict[int, int]:
    """Map each markdown table DATA row to its HEADER row, both as 1-based line numbers.

    The single definition of "data row" and "header row" in this module. `extract_numbers`
    reads it to build binding context, and the v0.11 row-ordinal clause reads it to decide
    scope — so clause scope and binding-context scope cannot diverge (PREREG_oath_v11,
    conjunct 1: the clause reads this machinery, it never copies it).

    Lifted verbatim out of `extract_numbers`; the walk is unchanged, only the value stored
    (the header's line number rather than its stripped text) differs, and the caller there
    re-derives the text it used to store.
    """
    rows: dict[int, int] = {}
    for i, line in enumerate(lines):
        if _TABLE_SEP.match(line) and i > 0 and lines[i - 1].lstrip().startswith("|"):
            j = i + 1
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                rows[j + 1] = i   # 1-based: data row j+1, header row (i-1)+1 == i
                j += 1
    return rows


def extract_numbers(text: str) -> list[dict]:
    """All groundable number tokens with line context. Filters dates/SHAs/versions/years/markdown
    artifacts and formula notation; keeps order and position so the ledger is reviewable.

    v0.3: markdown table rows inherit their table's HEADER line as additional binding context —
    the trigger vocabulary of '| regime | AUC-g | margin |' binds the numbers in every data row."""
    out = []
    lines = text.splitlines()
    header_for: dict[int, str] = {ln: lines[h - 1].strip()
                                  for ln, h in _table_rows(lines).items()}
    for ln_no, line in enumerate(lines, 1):
        # v0.6.2 signed extraction: the typographic minus U+2212 becomes ASCII '-' so _NUM reads
        # it as a sign — pre-fix, '−0.0154' extracted as POSITIVE 0.0154 and an accurate negative
        # claim could be accused (the v0.6.1 G3 kill). En-dash ranges (U+2013) are untouched.
        line = line.replace("−", "-")
        # drop fenced/sha/date/version spans from the searchable line.
        # v0.10 (PREREG_oath_v10_token_column): `re.sub(pat, " ", s)` collapses each match to ONE
        # space, so a 40-char sha shifts every token to its right 39 columns left and `m.start()`
        # is NOT the source column. `V10_TOKEN_COLUMN` blanks one space PER MATCHED CHARACTER
        # instead, which is what makes the recorded `col` below an address rather than an
        # approximation. Measured pre-fix over all 1,073 documents under papers/ (48,097 tokens):
        # length preservation changes the extracted token sequence on ZERO lines, so this buys
        # correct columns without moving extraction (battery gate G2).
        _blank = (lambda m: " " * (m.end() - m.start())) if V10_TOKEN_COLUMN else " "
        scrub = _SHAISH.sub(_blank, line)
        scrub = _DATEISH.sub(_blank, scrub)
        scrub = _VERSIONISH.sub(_blank, scrub)
        for m in _NUM.finditer(scrub):
            tok = m.group(0)
            raw = tok.replace(",", "")
            if _YEAR.match(raw.lstrip("+-")):
                continue
            # markdown heading/bullet artifacts: a bare int at the start of a
            # STRUCTURE line only — and never a slash-pair numerator ("7/12").
            if (m.start() <= 2 and "." not in raw and abs(int(raw)) < 10
                    and scrub[m.end():m.end() + 1] != "/"
                    and _MD_STRUCTURE.match(line)):
                continue
            if _FORMULA_AFTER.match(scrub[m.end():]):
                continue   # notation like '1−syc' — not a numeric claim
            if m.start() >= 2 and scrub[m.start() - 1] in "–-−" and scrub[m.start() - 2].isdigit():
                continue   # second half of a numeric range ('L27–31'): notation, not a claim
            if m.start() >= 2 and scrub[m.start() - 1] == "-" and scrub[m.start() - 2].isalpha():
                continue   # compound identifier ('shared-48', 'POS-A29'): a label, not a claim
            try:
                val = float(raw)
            except ValueError:
                continue
            entry = {"line": ln_no, "token": tok, "value": val, "decimals": _decimals(raw),
                     "context": line.strip()[:160]}
            if V10_TOKEN_COLUMN:
                # the column THIS match was found at, in the U+2212-normalized source line.
                # `certify_doc` anchors its windows here; without it, it re-finds the token
                # STRING and lands on the first occurrence, which is a different token 9.6% of
                # the time. Additive: no existing key changes (invariant I1).
                entry["col"] = m.start()
            if ln_no in header_for:
                entry["binding_context"] = (header_for[ln_no] + " " + line.strip())[:320]
            out.append(entry)
    return out


# ---------------------------------------------------------------- receipt flattening (truth side)

# v0.1: bulk per-item arrays are NOT the claimable truth surface — a receipt with 1000+ row leaves
# covers ~80% of all 2-decimal values in [0,1], so any corrupted doc number "verifies" by coincidence
# (the D1 failure mode of the v0 validation). Claims must ground in SUMMARY fields; a doc citing an
# individual row value lands in ABSTAIN (honest) unless the row path is explicitly allowed.
# v0.2: the rule applies to LIST nodes only — a SCALAR summary field that happens to share a bulk
# name (e.g. claim_totals.UNGROUNDED) is claimable truth and must ground (caught dogfooding the
# corpus attestation: its own headline counts failed to verify under v0.1).
_BULK_PATHS = re.compile(r"\b(rows|tier2_rows|ledger|abstained|ungrounded|ramp\w*|charfw\w*|cells)\b\[",
                         re.I)


def receipt_values(obj, prefix="", include_bulk: bool = False) -> list[tuple[str, float]]:
    """Flatten the numeric leaves of a receipt JSON to (path, value). Bulk per-item arrays are
    excluded by default (see _BULK_PATHS)."""
    vals: list[tuple[str, float]] = []
    if isinstance(obj, bool):
        return vals
    if isinstance(obj, (int, float)):
        return [(prefix or "$", float(obj))]
    if isinstance(obj, dict):
        for k, v in obj.items():
            vals.extend(receipt_values(v, f"{prefix}.{k}" if prefix else k, include_bulk))
    elif isinstance(obj, list):
        if not include_bulk and _BULK_PATHS.search(f"{prefix}["):
            return vals
        for i, v in enumerate(obj):
            vals.extend(receipt_values(v, f"{prefix}[{i}]", include_bulk))
    elif isinstance(obj, str):
        # numeric strings inside receipts count too
        try:
            vals.append((prefix or "$", float(obj)))
        except ValueError:
            pass
    return vals


def _match(doc_val: float, doc_dec: int, r_val: float, allow_scaling: bool = True) -> bool:
    """Rounding-aware equality: the doc may print a receipt value at lower precision.

    v0.3: percent<->fraction scaling only when the claim context shows a '%'/'percent' marker
    (allow_scaling) — unconditional scaling tripled the coincidence surface and let mutated
    values 'verify' against unrelated leaves (D1 misses k=0/k=17 of the v0.1 battery)."""
    if doc_val == r_val:
        return True
    if doc_dec > 0:
        # v0.6.2 epsilon hole (PREREG_oath_v062): the flat 1e-12 term is invisible at <=6
        # decimals and fatal at 16 — it verified any mutation in fractional digits >=13 of a
        # full-precision claim. Claims at <=12 decimals keep the historic tolerance
        # byte-for-byte; claims at >=13 decimals get no epsilon subsidy (verbatim quotes still
        # pass via exact equality above; the float64 floor at digit 16+ is disclosed, not hidden).
        eps = 1e-12 if doc_dec <= 12 else 0.0
        tol = 0.5 * 10 ** (-doc_dec) + eps
        if abs(round(r_val, doc_dec) - doc_val) <= eps or abs(r_val - doc_val) <= tol:
            return True
    if allow_scaling:
        # percent <-> fraction (doc says 80%, receipt holds 0.80; or doc 0.8 vs receipt 80)
        for scale in (100.0, 0.01):
            rv = r_val * scale
            # The product is ARITHMETIC, not a parsed decimal: 0.29*100 is
            # 28.999999999999996 in float64, so bare == made integer-percent
            # verdicts value-dependent noise (29% vs {"recall": 0.29} was
            # UNGROUNDED while 13% vs 0.13 verified — which correct claims
            # failed depended on the binary representation of the fraction).
            # rel_tol 1e-9 admits only representation error; a real mutation
            # differs at the first meaningful digit and still fails.
            if math.isclose(doc_val, rv, rel_tol=1e-9, abs_tol=1e-12):
                return True
            if doc_dec > 0 and abs(round(rv, doc_dec) - doc_val) <= 0.5 * 10 ** (-doc_dec) + 1e-12:
                return True
    return False


def _ulp_neighbour(val: float, rvals: list[tuple[str, str, float]], n: int):
    """First receipt leaf within *n* ULP of *val*, else None (v0.7 accusation-only escape).

    Distinct from `_match`: this does NOT ground a claim, it only withholds an accusation. Two
    float64s within a few ULP are the same measurement reached by differently ordered arithmetic
    (a different summation order, a numpy round-trip, a rounded intermediate), and at >=13 decimals
    v0.6.2 deliberately leaves no tolerance to absorb that."""
    if not math.isfinite(val):
        return None
    tol = n * math.ulp(val)
    for rn, pth, rv in rvals:
        if abs(rv - val) <= tol:
            return rn, pth
    return None


# v0.5 precision classes (PREREG_oath_v05_precision_2026_07_13) -- severable feature flags. Each
# gates one false-positive-elimination clause below; the prereg's severability procedure toggles
# these to measure per-class battery deltas and to drop a class that misses a bar. Shipped values
# are the composition the battery admits.
V05_APPROX_NOTATION = False   # class A: token after ≈/~/∼ -> ABSTAIN. DROPPED per the prereg's
                              # severability procedure (cycle38 battery: keeping it cost 3 catches
                              # and added 6 false-verifies -- the tilde abstains approximations that
                              # were being correctly caught -- for one FP eliminated). A refined
                              # ≈-only class A' is a future prereg, not a mid-run redesign.
V05_UNIT_RANGE = True         # class B: "2–3B" unit-suffixed range -> ABSTAIN
V05_ARXIV_ID = True           # class C: dddd.ddddd arXiv id -> ABSTAIN
V05_AT_PARAM = True           # class D: @-glued parameter -> ABSTAIN
V05_DERIVED_PCT = True        # class E: "12.7% (19/150" -> VERIFY iff both operands ground
V05_SELF_SCOPED_N = True      # class F: n= obligates only its own glued token

# v0.6.2 (PREREG_oath_v062_signed_extraction_2026_07_31) -- severable per the same procedure.
V062_FLOAT_STEM_PREF = True   # attribution-only: prefer stem-binding receipt_ref for floats;
                              # may reorder `hits`, never change a status (G5-gated)

# v0.7 (PREREG_oath_v07_precision_obligation_2026_08_22) -- severable, G5-gated.
V07_PRECISION_OBLIGATION = True   # primary: a token printed at >= V07_PRECISION_DIGITS fractional
                                  # digits was copied out of a computation, so it is OBLIGATED
                                  # regardless of line vocabulary. This is the trigger-recall debt
                                  # on this class: 0.5244 of the full-precision pool sits on
                                  # unbound lines, where a MUTATED number loses its value-match,
                                  # never triggers, and falls silently to ABSTAIN while the
                                  # document keeps its OATH-HELD verdict. The predicate reads only
                                  # the token, so it survives the mutation it exists to catch --
                                  # an obligation that consults `hits` evaporates under exactly
                                  # that mutation and therefore cannot gate.
V07_PRECISION_DIGITS = 7          # 7, not 5: every live counterexample the red-team pass produced
                                  # sits at exactly 5-6 digits -- a frozen kill-gate bar in this
                                  # repo's JSON idiom ("op": "<=", "value": 0.00648), the half-ULP
                                  # tolerance definition 0.00005, pi as 3.14159, a Bonferroni alpha
                                  # 0.00714, and the arXiv DOI prefix 10.48550, which neither
                                  # _VERSIONISH nor v0.5 class C reaches. 5 buys two tokens of
                                  # catch surface in the certified corpus and re-arms five boundary
                                  # occupants; cycle 24 died on ONE token sitting on the boundary
                                  # of a numeric guard.
V07_ULP_ESCAPE = True             # severable, accusation-only: an obligation created by the
                                  # precision clause ALONE, with no match but a receipt leaf within
                                  # V07_ULP_N ULP, degrades to ABSTAIN with a named reason. It may
                                  # never produce a VERIFIED and never softens an obligation that
                                  # _TRIGGERS / _TRIGGERS_CORR / n= / range-sanity created.
# v0.8 (PREREG_oath_v08_float_field_binding_2026_08_23) -- severable, G5-gated. The standing
# v0.4 debt: promote the v0.6.2 float stem test from ATTRIBUTION to STATUS, so a float claim must
# ground in a leaf whose PATH relates to the claim's context and not merely in one that happens to
# hold its value. This is the only instrument that attacks FALSE ATTESTATION: of 3951 claims the
# verifier certifies VERIFIED, mutating one significant digit leaves 604 still VERIFIED, matched to
# an unrelated leaf. Obligation cannot reach them -- obligation decides whether a claim MUST match,
# and these already do.
#
# SHIPPED OFF. The battery cleared every mechanical bar -- G2 removed 107 false attestations at the
# gating seed against a bar of 60, G3 cost ratio 1.056 against 1.5, G5 severability 0, and invariant
# I1 held exactly (UNGROUNDED 4->4, HELD 136->136, all 113 transitions VERIFIED->ABSTAIN). It died
# on G4, the gate that asks whether the demotions are RIGHT: of 40 sampled demotions hand-scored
# against the prereg's frozen definition, 30 destroyed a GENUINE binding against a bar of 12.
# Kill token V08_COVERAGE_DESTRUCTIVE.
#
# The structural reason, and why no threshold rescues it: scientific prose names a measurement
# narratively -- "whole-stack r=16: 0.616-0.626", "the loop beats the stubborn baseline 0.57 vs
# 0.47" -- while the receipt field that holds it is structural (points[2].naive_relock_auroc,
# lambdas[1], final.accuracy). Path-stem overlap has no purchase on that, so the honest population
# is not lexically separable from the tampered one. Five design families were swept pre-fix (naked
# stem 3.48 cost per kill, KEEP-widenings 1.18, context window 1.06, window+NAMEABLE 1.02,
# all-hits-array-indexed 7.72) and NONE beats parity: the instrument buys about one honest
# demotion per false attestation removed. An ACCUSING variant is worse still and was rejected
# outright -- every operating point of every family would have put dozens of new UNGROUNDED tokens
# on honest documents (40 at the best point).
#
# Kept in tree behind the flag, as V05_APPROX_NOTATION was after the cycle-38 severability drop, so
# the measurement is re-runnable and the negative is not re-attempted. G5 proves it inert when off.
# DO NOT re-enable without a new prereg: the v0.4 debt is CLOSED_NEGATIVE, not owed.
V08_FLOAT_FIELD_BINDING = False
V08_FIELD_BIND_MAX_DECIMALS = 3   # floats at 1..3 fractional digits. The coincidence surface
                                  # shrinks with printed precision, so binding pays only at low
                                  # width: swept pre-fix, cost per false attestation removed is
                                  # 1.02 at <=3 decimals and 27x at 4 and 76x at 6, where a
                                  # demotion nearly always destroys a correct binding instead. The
                                  # ceiling also keeps this clause disjoint from v0.7's >=7 rule.
V08_FIELD_BIND_PREV_LINES = 1     # the binding window reaches back one line. Prose names a
                                  # measurement a sentence before it prints it, and a line-local
                                  # window scores 3.48 cost per kill against 1.06 at prev1.
                                  # DEMOTE-ONLY: the clause yields ABSTAIN, never UNGROUNDED, so it
                                  # can neither create nor remove an accusation and no certificate
                                  # can flip HELD->FAILED (invariant I1, asserted in the suite and
                                  # deliberately NOT gated -- a leg that cannot fail must not gate).
                                  # An accusing variant was measured and rejected: every operating
                                  # point of all five swept design families would have put dozens of
                                  # new UNGROUNDED tokens on honest documents (40 at the best point).

V07_ULP_N = 8                     # v0.6.2 withdrew the epsilon subsidy at >=13 decimals, so at
                                  # doc_dec=16 the tolerance (5e-17) sits BELOW the float64 ULP
                                  # near 1.0 (1.11e-16). That was safe while such tokens were never
                                  # obligated; the clause above makes it live, and without this
                                  # escape a restatement of the SAME measurement by differently
                                  # ordered arithmetic becomes a loud accusation. Yielding ABSTAIN
                                  # rather than VERIFIED is what keeps the v0.6.2 epsilon hole
                                  # closed, and the countable `ulp-neighbour` reason is what keeps
                                  # the residual enumerable instead of invisible.

# v0.9 (PREREG_oath_v09_is_spec_json_idiom_2026_08_23) -- severable, G5-gated. `is_spec` reads an
# 18-character window before the token, so it recognises a bar only when the operator or the bar
# noun happens to sit there. Two recall extensions were measured pre-fix; ONE ships, and the split
# is the finding.
V09_IS_SPEC_JSON_IDIOM = True     # primary: a token in JSON value position whose object also
                                  # carries a comparison-operator field is a bar written in the
                                  # idiom `{"op": "<=", "value": 0.00648}` -- the operator is in a
                                  # separate field, so the pre-window never sees it. 145 such
                                  # tokens in 42 documents under papers/, 0 rescued before this
                                  # clause. It is live, not hypothetical:
                                  # PREDICTION_h1_human_islands' committed certificate swears two
                                  # DIFFERENT preregistered bars -- a dip-test p-value and an R^2 --
                                  # against one unrelated leaf, b45_result.json:null_expectation_k20,
                                  # whose only qualification is holding the float 0.05.
                                  # This clause is safe to abstain because of a measured fact about
                                  # its class and NOT about abstention in general: 0 of the 145 sit
                                  # on a line the obligation predicate binds, so none of them has a
                                  # catch to destroy. See V09_IS_SPEC_BAR_NOUN for what happens
                                  # where that is not true.
V09_IS_SPEC_BAR_NOUN = False      # control: a bar named AFTER the number ("clears the 0.10 floor").
                                  # DROPPED, and retained here so one flag reproduces the negative.
                                  # 38 such tokens in the 137-document certified corpus, 37 of them
                                  # VERIFIED -- but 36 sit on BOUND lines, and a bound token is one
                                  # the verifier CATCHES when it is doctored. Mutating all 38 at the
                                  # shipped verifier over seeds 1-10: 18.7 caught on average
                                  # (range 16-22) against 17.4 falsely attested (range 14-20). The
                                  # columns overlap and the per-seed net changes sign, so no seed
                                  # settles the trade -- but this clause takes the CATCH column to
                                  # ZERO at every seed, because the predicate reads context and a
                                  # one-digit mutation leaves the context unchanged. It does not
                                  # detect the tamper it is credited with; it stops looking.
                                  # An abstention rule improves every tamper metric by destroying
                                  # coverage, and G6 is what made that visible instead of flattering.
                                  # The 2-character slack is frozen from the pre-fix sweep: at 0 the
                                  # clause never fires, at >=4 it starts taking the MEASUREMENT in
                                  # '0.5362318840579711 (floor 0.50)', and at 8 it takes the 0.954
                                  # out of '(clean read 0.954) floors at ~0.62', where 'floors' is a
                                  # verb.

# The comparison-operator field is REQUIRED, not optional: `"value": 0.75` alone is an ordinary
# key/value pair and says nothing about specification. The operator field on the same object is
# what makes the number a bound ON some other quantity rather than a record OF one, and it is what
# keeps this clause off result receipts transcribed into a document.
_JSON_OP_FIELD = re.compile(r'"(?:op|operator|cmp|comparison|direction|sense)"\s*:\s*"\s*'
                            r'(?:<=|>=|!=|==|<|>|=)\s*"')
_JSON_BAR_KEY = re.compile(r'"(?:value|bar|threshold|floor|ceiling|cutoff|min|max|alpha|target'
                           r'|bound)"\s*:\s*$')
_BAR_NOUN_POST = re.compile(r"^[ \t\-]{0,2}(?:floors?|ceilings?|cutoffs?|caps?|bounds?)\b", re.I)

# v0.10 (PREREG_oath_v10_token_column_2026_08_23) -- severable, gated. Every cycle from v0.1 to
# v0.9 argued about what the context windows should MEAN. This one is about where they ARE.
V10_TOKEN_COLUMN = True           # primary: `extract_numbers` records the column its match was
                                  # found at and `certify_doc` anchors `pre`/`post` there, instead
                                  # of `ctx.find(token)` -- which returns the FIRST occurrence of
                                  # the token STRING and is a DIFFERENT token 4,612 times in the
                                  # 48,097 tokens under papers/ (9.589%, 841 documents). When it
                                  # is, every predicate downstream of the windows is decided
                                  # against text that does not surround the claim: `is_spec`,
                                  # `is_notation`, `is_hist`, the range-sanity unit_kw/sign_kw
                                  # tests, the slash-pair branch of count-binding, the v0.5 class F
                                  # n= self-scope, and the class E derived-percent parse. 95 of the
                                  # 349 misplaced tokens in the certified corpus have a predicate
                                  # that actually disagrees between the two anchors.
                                  # It is live, not hypothetical: PREREG_b49_amplitude_reaudit L23
                                  # holds a preregistered bar in JSON value position at column 98,
                                  # `ctx.find("5")` returns 6 -- the 5 inside `b45` -- and
                                  # V09_IS_SPEC_JSON_IDIOM, shipped for exactly that token class,
                                  # cannot see the "value": key and does not fire.
                                  # Requires the length-preserving scrub in `extract_numbers`; a
                                  # raw m.start() against the shipped collapsing scrub would be a
                                  # NEW wrong column on every line carrying a sha/date/version.
V10_SLASHPAIR_RANGE_GUARD = True  # companion: the v0.3 range-sanity rule does not fire on a
                                  # slash-pair numerator. A value written `a/b` is a count pair,
                                  # never a value of the bounded quantity named to its left.
                                  # This exists for exactly one token that the primary un-masks:
                                  # FINDING_mapped_whitening L31 `(stability 5/5)`, where correct
                                  # anchoring puts `stability ` in `pre`, 5 leaves [0,1], and the
                                  # rule accuses a document whose receipt DOES hold the count
                                  # (mapped_whitening_result.json:stability_count_under_ceiling).
                                  # Without it the repair flips one committed OATH-HELD to
                                  # OATH-FAILED on a false accusation (battery gate G3).
                                  # Measured: with V10_TOKEN_COLUMN OFF this clause changes 0
                                  # ledger rows, and on the tamper collision channel it changes
                                  # nothing at all (caught 43->43, false-attested 271->271 over
                                  # seeds 1-10) -- it carries no behaviour of its own.
#
# NAMED RESIDUAL, owed to a successor prereg and expressly NOT owed by this cycle:
# V10_EQUALS_SPEC_OVERREACH. `is_spec` reads a bare `=` at the end of `pre` as a comparison
# operator. Pointed at the wrong text it rarely fired; pointed correctly it fires on the ASSIGNMENT
# idiom, which in this corpus is a MEASUREMENT idiom -- `n = 1`, `n_refits=5`, `n_admissible=5`,
# `P(>=0.15)=1.0`, `0.0854 = 0.0854`, `95th percentile = 1.000`. All 9 destructive abstentions this
# cycle measured are that one shape (gate G4c checks it mechanically). Not fixed here for v0.8's
# reason: `V07_PRECISION_DIGITS = 7` is a spec and `AUROC(S_frame) = 0.75` is not, and the two are
# identical in form, so the populations are not lexically separable and any narrowing is a doctrine
# change to `is_spec` with its own battery.

# v0.11 (PREREG_oath_v11_row_ordinal_retraction_2026_08_25) -- flag-gated, gated by a nine-gate
# battery. Every cycle before this one asked whether a token was GROUNDED. This one asks whether
# it is a CLAIM.
#
# A markdown table's first column is where this corpus writes its row numbers. `extract_numbers`
# extracts them like any other token, and on rows whose text carries trigger vocabulary the
# OBLIGATION predicate binds them -- so a row number must ground in a receipt leaf or be accused.
# A row number has no receipt, because it asserts nothing. The certified frame's ENTIRE standing
# accusation surface was four of them: PROSPECTUS_knowsay_2026_07_27.md L27 `3`, L28 `4`, L29 `5`,
# L32 `8`. The VERIFIED half of the same column is worse than the accused half -- L26 `2` swears
# against `scale_test_result.json:per_item[2].i`, an index leaf equal to its own subscript, so the
# oath is taken on a coincidence. Exhaustive substitution over the 11-token class (117 mutants)
# answers UNGROUNDED 46 / VERIFIED 50 / ABSTAIN 21: a 0.427 false-attestation rate under tamper,
# on tokens that assert nothing.
#
# UNGROUNDED asserts "this token is a claim whose truth condition was never met." A hand panel
# (`oath_v10_panel_isclaim.json`, re-checked blind at the shipping verifier by
# `oath_v11_panel_recheck.json`) found the accused tokens are LABELs -- they have no truth
# condition -- so neither VERIFIED nor UNGROUNDED is meaningful and ABSTAIN is the only defensible
# status. An accusation is itself a claim, and these four accusations have no receipts.
#
# THE RETRACTION PREDICATE, doctrine: a status may be withdrawn only when what is shown false is
# the accusation's PRESUPPOSITION (claimhood), never its verdict (groundedness). v0.9's G4 -- zero
# accusations silenced, zero FAILED->HELD flips -- protected accusations that are MEASUREMENTS. It
# never contemplated accusations that fail to be claims. The whitelist is non-precedential as a
# mechanism: the next retraction runs the full protocol again, with its own panel and its own
# prereg.
#
# NEVER NON-EXTRACTION. A fix that stops accusing by stopping extracting is not a fix. Every
# silenced token stays countable by coordinate: the certificate's `abstained` array carries its
# line and token, and its ledger row carries the reason `row_ordinal_label`. Silence loud, never
# omission.
V11_ORDINAL_LABEL = True

# The frozen vocabulary, written here and nowhere else. THIS LIST CAN ONLY SHRINK.
# Named exclusions, each with its measured reason -- exclusion is the safe direction, because an
# excluded header leaves its tokens OBLIGATED (a disclosed false-accusation surface) while an
# admitted one silences them:
#   ''  and '-'  the unlabeled-parameter convention. Admitting them was "luck rather than design";
#                closing them costs zero -- all 11 in-frame firings and all 128 corpus-wide
#                firings carry a literal '#'.
#   'rank'       27 corpus-wide firings, hand-labeled by the red team as ordinal rankings -- a
#                label, same class as a row number. Excluded NOT because they are claims but
#                because retracting a class needs its own panel and prereg. They stay obligated.
#                ('rank k' is a different population entirely: its sweep values ground in
#                `ranks[j]` under a NON-identity mapping, i.e. genuine claims.)
#   'n', 'no', 'num', 'id', 'item', 'line', 'claim', 'seed', 'k', 'run', 'attempt'
#                each a live or plausible claim header. `seed` alone is 63 of the 150 first-cell
#                tokens in frame, 61 of them VERIFIED; silencing it replays the broad-detector
#                catastrophe (115 of 150 tokens falsely silenced, 28.1 reader-visible catches
#                destroyed per seed).
# The three vocabulary variants in the v0.10 receipts disagree with each other; that discrepancy is
# a disclosed defect of the measurement cycle, resolved by freezing this narrower list BEFORE data.
_V11_ORDINAL_HEADERS = frozenset({"#", "#.", "no.", "nr", "idx", "index", "row", "row #", "№"})

# The cell must be ENTIRELY a bare non-negative integer of value <= 100. In-frame this conjunct and
# this cap do no discriminative work -- the header does all of it, and the largest of the 11
# firings is 11 -- so both are anti-gaming defense-in-depth, bounding what an author could hide
# under a renamed column. Receipt-bound variance disclosed rather than averaged: the detectors
# receipt says `|value| <= 100`, the red-team receipt says `|value| < 100`; frozen here as
# non-negative and <= 100 (no negative first-cell integer exists under any variant).
# EDGE DISCLOSED: a 1..N column longer than 100 rows flips behaviour at row 101, re-manufacturing
# the accused class on the rows past the cap. No in-frame table is within a factor of two of it.
_V11_MAX_VALUE = 100
_V11_BARE_INT = re.compile(r"[0-9]+")
_V11_EMPHASIS = " \t*_"


def _first_cell(line: str):
    """(start, end) column span of a markdown table row's FIRST cell, or None.

    Columns are offsets into the U+2212-normalized line, which is the frame `col` is recorded in
    (the replacement is one character for one, so offsets are preserved)."""
    a = line.find("|")
    if a < 0:
        return None
    b = line.find("|", a + 1)
    if b < 0:
        return None
    return a + 1, b


def _v11_header_ok(cell: str) -> bool:
    """Header first cell, backticks and emphasis stripped, trimmed, case-folded, EXACT match."""
    return cell.replace("`", "").strip(_V11_EMPHASIS).strip().casefold() in _V11_ORDINAL_HEADERS


def _v11_sole_int(cell: str) -> bool:
    """Cell, emphasis stripped, is entirely a bare non-negative integer <= _V11_MAX_VALUE.

    Backticks are deliberately NOT stripped here (the prereg strips them for the HEADER only):
    a cell written `` `3` `` fails this conjunct and its token stays obligated, which is the safe
    direction. `_V11_BARE_INT` is ASCII-only on purpose -- `str.isdigit()` accepts superscripts
    and non-ASCII digit forms that `int()` and `_NUM` do not agree about."""
    s = cell.strip(_V11_EMPHASIS).strip()
    return bool(_V11_BARE_INT.fullmatch(s)) and int(s) <= _V11_MAX_VALUE


def _v11_row_ordinal_label(num: dict, lines: list[str], table_rows: dict[int, int]) -> bool:
    """True iff token *num* is a markdown table row ordinal under a frozen ordinal header.

    Value-blind by construction: it reads the token's ADDRESS and the table's STRUCTURE, never
    the token's value beyond the sole-content bound, and never `hits`. That is what stops it
    being a fuse -- doctor the digit and the clause still fires, which is the property v0.7 and
    the rejected value-reading designs failed (`override_missed_mutant` 22/22).
    """
    if not V11_ORDINAL_LABEL:
        return False
    # V10_TOKEN_COLUMN is a DECLARED, NON-SEVERABLE prerequisite: position must be an address,
    # not a re-found string. Without `col` this clause has no scope and must not fire.
    if not V10_TOKEN_COLUMN or "col" not in num:
        return False
    hdr_ln = table_rows.get(num["line"])
    if hdr_ln is None:
        return False                       # not a table data row
    row = lines[num["line"] - 1].replace("−", "-")
    span = _first_cell(row)
    if span is None or not span[0] <= num["col"] < span[1]:
        return False                       # not in the first cell
    hdr_span = _first_cell(lines[hdr_ln - 1])
    if hdr_span is None:
        return False
    return (_v11_header_ok(lines[hdr_ln - 1][hdr_span[0]:hdr_span[1]])
            and _v11_sole_int(row[span[0]:span[1]]))


# v0.12 (PREREG_oath_v12_formula_constant_2026_08_26) -- SHIPPED OFF, killed by its own G2.
#
# The defect it aimed at is real and is still open: `extract_numbers` takes numerals out of
# rendered mathematics, and `\Delta` is trigger vocabulary because `delta` is in `_TRIGGERS`, so
# the literal `1` in `\left(1 \pm \frac{\Delta \sigma^2}{\sigma^2}\right)` gets accused of being
# a claim whose truth condition was never met. It is a mathematical constant. It has no truth
# condition at all -- the same category error v0.11 spent a cycle retracting for row ordinals.
#
# THE KILL. The prereg froze G2 on an 11-token roster (3 UNGROUNDED + 8 VERIFIED) taken from the
# census's LINE-level marker, then specified a SPAN-level clause. Those are different
# populations and the gap is the whole defect: the clause reaches 6 tokens, not 11, so G2
# under-fires and the pre-committed outcome is `V12_UNDERREACH` -- revert and publish.
#
# What it misses is the part worth remembering: **the prereg's own motivating specimen.** That
# formula is written as an indented code block, so there is no inline-code span and no `$`
# delimiter, and conjunct 1 never fires. The prereg pre-committed that if its own certificate
# failed to flip to OATH-HELD the cycle had under-reached regardless of the other gates. It did
# not flip. A clause that cannot reach the example its own preregistration quotes has not earned
# a corpus.
#
# Kept in tree behind the flag -- as V05_APPROX_NOTATION and V08_FLOAT_FIELD_BINDING were after
# their kills -- so the measurement is re-runnable and the negative is not re-attempted. DO NOT
# widen conjunct 1 to catch indented code blocks and re-run: the prereg says the clause is atomic
# and forbids post-freeze narrowing or widening, and "no second attempt inside this cycle". A
# successor needs its own preregistration, frozen against a SPAN-level census rather than a
# line-level one.
V12_FORMULA_CONSTANT = False

_V12_BACKSLASH_CMD = re.compile(r"\\[A-Za-z]+")
_V12_BARE_NUM = re.compile(r"[0-9]+(?:\.[0-9]+)?")


def _delimited_spans(line: str, delim: str) -> list:
    """(start, end) content spans between successive occurrences of *delim*."""
    out, i = [], 0
    while True:
        a = line.find(delim, i)
        if a < 0:
            return out
        b = line.find(delim, a + len(delim))
        if b < 0:
            return out
        out.append((a + len(delim), b))
        i = b + len(delim)


def _v12_formula_constant(num: dict, lines: list[str]) -> bool:
    """True iff token *num* sits inside a delimited mathematical span (frozen v0.12 conjuncts).

    Conjunct 1: the recorded column lies inside a `$...$` / `$$...$$` span, or inside an
    inline-code span whose content carries a backslash command. Conjunct 2: that span contains a
    backslash command -- a `$...$` span without one is a dollar amount or a shell prompt, not
    rendered mathematics. Conjunct 3: the token is a bare integer or decimal with no thousands
    comma, because a formula does not contain `100,000`.
    """
    if not V12_FORMULA_CONSTANT:
        return False
    if not V10_TOKEN_COLUMN or "col" not in num:
        return False
    if not _V12_BARE_NUM.fullmatch(num["token"]):
        return False
    line = lines[num["line"] - 1].replace("−", "-") if num["line"] - 1 < len(lines) else ""
    col = num["col"]
    for delim in ("$$", "$", "`"):
        for a, b in _delimited_spans(line, delim):
            if a <= col < b and _V12_BACKSLASH_CMD.search(line[a:b]):
                return True
    return False


def _ctx_stems(text: str) -> set[str]:
    """The v0.3/v0.6.2 binding-context stem set, lifted to module level for the v0.8 clause.

    Identical vocabulary to the inline copies in `certify_doc` (which are left byte-identical):
    4-char prefixes of every word, plus 4-char prefixes of each hyphen/underscore segment."""
    words = {w.lower().strip("'’") for w in re.findall(r"[A-Za-z][A-Za-z_-]{2,}", text)}
    return {w[:4] for w in words} | {s[:4] for w in words
                                     for s in re.split(r"[-_]", w) if len(s) >= 3}


def _path_stems(path: str) -> set[str]:
    """The v0.3/v0.6.2 receipt-path stem set (`path_ok` / `_stem_ok`), lifted to module level."""
    segs = {s.lower() for seg in re.split(r"[.\[\]]", path) for s in re.split(r"[-_]", seg) if s}
    return {s[:4] for s in segs if len(s) >= 3}


# ---------------------------------------------------------------- contradiction triggers

# context keywords that bind a doc number to receipt quantities (the v0 trigger vocabulary).
# a number whose line mentions one of these AND whose receipts carry a same-kind quantity is
# OBLIGATED to match some receipt value, else UNGROUNDED.
_TRIGGERS = re.compile(
    r"\b(aurocs?|aucs?|margins?|cis?\b|boot(strap)?|perm(utation)?(_p\d+)?|p9\d|recall|precision|"
    r"fpr|fnr|accuracy|rate|median|mean|elevation|floor|delta|n_held|n_caved|held|caved|"
    r"gated|dropped|grounded|sycophancy|deception|surface|lens|firewall|collapse|wilson|"
    r"concordance|stability|score[sd]?)\b", re.I)
# v0.5 class F (self-scoped n=, PREREG_oath_v05_precision): `n\s*=` no longer sits in the
# LINE-WIDE alternation above — an "N=4" was obligating every other bare integer sharing its line
# (the dominant measured false-positive class). It now obligates ONLY the token it directly
# prefixes, via the n_self check in certify_doc.

# v0.4 trigger-recall: the correlation/similarity register the AUROC-centric _TRIGGERS above misses
# (the 182/269 abstain-degrade bucket of the cycle-19 mutant battery lives here). A BLUNT add
# over-triggers (cycle 23: 6 artifacts) and a bare value-range guard leaves the integer ordinal that
# collides with the correlation boundary 1.0 (cycle 24: "drift, stage 1"). The shipped rule: this
# register obligates a number only when it is a FRACTIONAL correlation — decimals > 0 AND value in
# [−1, 1]. No ordinal/index/count/API-constant/whole-percent (all decimals == 0) can bind; the range
# spares out-of-range decimals. See PREREG_oath_v04_recall_decimalguard_2026_07_04.md.
_TRIGGERS_CORR = re.compile(
    r"\b(rsa|rdm|spearman|pearson|correlations?|rho|consistency|reliability|ceiling|agreement|"
    r"convergence|drift|entropy|similarity|variance)\b", re.I)


# The ten labels this verifier version can emit, in execution order: two pre-ladder demotions,
# then the ladder. A new branch or clause is a NEW schema string, never a mutation of v1.
_EPISTEMICS_BRANCHES = ("row-ordinal-label", "formula-constant", "spec-or-historical",
                        "notation", "derived", "unbound-field", "value-match",
                        "ulp-neighbour", "obligated-accusation", "silent")
_EPISTEMICS_SOURCES = ("vocabulary", "n-glued", "range-correlation", "precision", "range-sanity")


def _epistemics_summary(ledger: list, counts: dict) -> dict:
    """Fold the per-token epistemics into a machine-consumable block.

    Frozen shape: styxx-oath/epistemics-summary/v1, designed and red-teamed 2026-08-30
    (`papers/closed-model-frontier/DESIGN_epistemics_summary_schema_2026_08_30.md`). Counts only,
    no rates; every key always present, zeros included; a pure deterministic function of the
    certificate's own ledger and of nothing else. It counts which door each token came through and
    says nothing about whether any token is a true claim or a good one.

    The four value-match cells name the MECHANISM, not a virtue: `integer_filter_ran` means the
    v0.3 count-binding filter executed (decimals == 0); `integer_filter_na` means the token is a
    float and receives no status-level binding at this verifier version (v0.8 CLOSED_NEGATIVE).
    `unobligated_integer_filter_na` is the weakest attestation the instrument produces;
    `obligated_integer_filter_na` is the larger obligated exposure the first draft of this schema
    hid. Invariants are asserted loudly at issuance -- a certificate must fail to issue rather
    than carry a self-inconsistent summary.
    """
    by_branch = {b: 0 for b in _EPISTEMICS_BRANCHES}
    sources = {k: 0 for k in _EPISTEMICS_SOURCES}
    vm = {"obligated_integer_filter_ran": 0, "obligated_integer_filter_na": 0,
          "unobligated_integer_filter_ran": 0, "unobligated_integer_filter_na": 0}
    derived = {"obligated": 0, "unobligated": 0}
    obligated_total = 0
    for e in ledger:
        ep = e["epistemics"]
        by_branch[ep["branch"]] += 1
        if ep["obligated"]:
            obligated_total += 1
            sources[ep["obligation_source"]] += 1
        if e["status"] == "VERIFIED":
            if ep["branch"] == "derived":
                derived["obligated" if ep["obligated"] else "unobligated"] += 1
            elif ep["branch"] == "value-match":
                key = ("obligated" if ep["obligated"] else "unobligated") +                       ("_integer_filter_ran" if ep["path_checked"] else "_integer_filter_na")
                vm[key] += 1
    total_tokens = counts["VERIFIED"] + counts["ABSTAIN"] + counts["UNGROUNDED"]
    assert sum(by_branch.values()) == total_tokens, "epistemics_summary: branch sum drifted"
    assert by_branch["obligated-accusation"] == counts["UNGROUNDED"],         "epistemics_summary: accusation branch != UNGROUNDED count"
    assert counts["VERIFIED"] == sum(derived.values()) + sum(vm.values()),         "epistemics_summary: verified partition drifted"
    assert obligated_total == sum(sources.values()), "epistemics_summary: source sum drifted"
    return {
        "schema": "styxx-oath/epistemics-summary/v1",
        "note": ("attestation composition folded from this certificate's own ledger; counts "
                 "which door each token came through; says nothing about whether any token is a "
                 "true claim or a good one"),
        "by_branch": by_branch,
        "verified": {"total": counts["VERIFIED"], "derived": derived, "value_match": vm},
        "obligated_total": obligated_total,
        "obligation_sources": sources,
    }


def certify_doc(doc_path: Path, receipt_paths: list[Path]) -> dict:
    text = doc_path.read_text(encoding="utf-8")
    receipts = {}
    rvals: list[tuple[str, str, float]] = []   # (receipt, path, value)  — summary surface only
    for rp in receipt_paths:
        j = json.loads(rp.read_text(encoding="utf-8"))
        receipts[rp.name] = hashlib.sha256(rp.read_bytes()).hexdigest()
        for path, v in receipt_values(j):
            rvals.append((rp.name, path, v))

    # v0.8: every path stem present anywhere in the cited receipt set, computed ONCE per document.
    # The NAMEABLE test below is per-claim and would otherwise re-scan every leaf for every token.
    all_path_stems: set[str] = set()
    for _rn, _pth, _rv in rvals:
        all_path_stems |= _path_stems(_pth)

    ledger = []
    doc_lines = text.splitlines()
    table_rows = _table_rows(doc_lines)   # v0.11: the SAME machinery extract_numbers binds by
    for num in extract_numbers(text):
        # v0.11 ROW-ORDINAL LABEL: a status-level demotion to ABSTAIN with the machine-readable
        # reason `row_ordinal_label`, at the `is_spec` tier — literally BEFORE any obligation or
        # match is consulted, which is what keeps the clause value-blind and idempotent. A row
        # number is not a claim whose truth condition was unmet; it has no truth condition, so
        # neither VERIFIED nor UNGROUNDED is meaningful for it.
        if _v11_row_ordinal_label(num, doc_lines, table_rows):
            ledger.append({**num, "status": "ABSTAIN", "receipt_ref": "row_ordinal_label",
                           "epistemics": {"branch": "row-ordinal-label", "obligated": False,
                                          "obligation_source": None}})
            continue
        # v0.12 FORMULA CONSTANT: same tier, same shape — and SHIPPED OFF, killed by its own G2
        # for under-reaching. Live only so the negative stays re-runnable.
        if _v12_formula_constant(num, doc_lines):
            ledger.append({**num, "status": "ABSTAIN", "receipt_ref": "formula_constant",
                           "epistemics": {"branch": "formula-constant", "obligated": False,
                                          "obligation_source": None}})
            continue
        # v0.1 SPEC-CONSTANT rule: a number that is a pre-registered bar/threshold, a CI confidence
        # level, or a comparison bound is SPEC, not a measurement -> ABSTAIN (it has no receipt by
        # design; its receipt is the PREREG document).
        # v0.3: rules test the FULL line — the display context truncates at 160 chars and a
        # disclosure note past that boundary was invisible to is_hist (caught in the D2 hand-check).
        # v0.6.2: same U+2212 normalization as extraction, so signed tokens are findable in ctx
        ctx = doc_lines[num["line"] - 1].strip().replace("−", "-")
        bctx = num.get("binding_context", ctx)   # v0.3: table rows bind via their header too
        # v0.10: anchor the windows at the column the token was EXTRACTED at. `ctx.find` returns
        # the first occurrence of the token STRING, which is a different token on 9.6% of the
        # corpus ("10 neutral + 10 in-frame", "0.0854 = 0.0854", a bar whose digits also appear in
        # an identifier earlier on the line). `ctx` is `.strip()`ed and `col` is a raw-line offset,
        # so the leading whitespace comes back off; `.strip()` and the U+2212 replace commute
        # because U+2212 is not whitespace, so the two spellings of `ctx` agree.
        if V10_TOKEN_COLUMN and "col" in num:
            _raw = doc_lines[num["line"] - 1].replace("−", "-")
            tok_at = num["col"] - (len(_raw) - len(_raw.lstrip()))
        else:
            tok_at = ctx.find(num["token"])
        pre = ctx[max(0, tok_at - 18):tok_at] if tok_at >= 0 else ""
        # v0.3: a token at line start inherits the tail of the previous line as pre-context —
        # 'subclass AUC\n1.0)' wraps mid-sentence and the unit keyword must still bind.
        if 0 <= tok_at < 18 and num["line"] >= 2:
            pre = (doc_lines[num["line"] - 2].strip().replace("−", "-")[-(18 - tok_at):]
                   + " " + pre).strip()[-24:]
        post = ctx[tok_at + len(num["token"]):] if tok_at >= 0 else ""
        # v0.5 class A (approx-notation): ≈/~/∼ join the comparison class — a value the doc itself
        # marks approximate is not an exact-oath-swearable claim (PREREG_oath_v05_precision).
        _spec_ops = "≥≤<>=≈~∼" if V05_APPROX_NOTATION else "≥≤<>="
        is_spec = bool(re.search(r"[" + _spec_ops + r"]\s*\+?$|\b(bar|gate|threshold|requires?|must|pre-?registered)"
                                 r"\b[^.]{0,16}$", pre)) \
            or bool(re.match(r"\s*%?\s*(CI|confidence)", post)) \
            or bool(re.match(r"[^.\d]{0,12}\b(bar|threshold|gate)\b", post))
        # v0.9 is_spec recall (PREREG_oath_v09_is_spec_json_idiom). Both clauses only ADD to the
        # spec set and both yield ABSTAIN, so neither can create an accusation and no certificate
        # can flip HELD->FAILED (invariant I2, asserted in the suite and deliberately NOT gated --
        # a leg that cannot fail must not gate). The JSON clause tests the FULL line for the
        # operator field, as the v0.3 amendment requires: the operator sits in a sibling field
        # that the 18-character pre-window structurally cannot see.
        if V09_IS_SPEC_JSON_IDIOM and not is_spec:
            is_spec = bool(_JSON_BAR_KEY.search(pre)) and bool(_JSON_OP_FIELD.search(ctx))
        if V09_IS_SPEC_BAR_NOUN and not is_spec:
            is_spec = bool(_BAR_NOUN_POST.match(post))
        # v0.5 classes B/C/D (notation-level non-measurements -> ABSTAIN, PREREG_oath_v05_precision):
        # B unit-suffixed numeric range ("2–3B"), C arXiv identifier, D @-glued parameter.
        is_notation = (
            (V05_UNIT_RANGE and (
                bool(re.match(r"\s*[–-]\s*\d+(\.\d+)?\s*[BMK]\b", post))
                or (bool(re.search(r"\d\s*[–-]\s*$", pre)) and bool(re.match(r"\s*[BMK]\b", post)))))
            or (V05_ARXIV_ID and bool(re.fullmatch(r"\d{4}\.\d{4,5}", num["token"])))
            or (V05_AT_PARAM and bool(re.search(r"@\s*$", pre)))
        )
        # v0.1 QUOTED-HISTORICAL rule: corrected-away values quoted inside a disclosure note are
        # historical quotations, not live claims. v0.3: prior-run narrative counts, and on a MIXED
        # line the rule covers only tokens at/after the disclosure phrase (live values stay live).
        hist_m = re.search(r"originally printed|caught by OATH|superseded|was printed|"
                           r"\b(first|earlier|prior)\s+(scored\s+)?run\b", ctx, re.I)
        is_hist = bool(hist_m) and (tok_at < 0 or tok_at >= hist_m.start() - 24)
        # v0.10: hoisted out of the count-binding block below (same expression, same inputs — `pre`
        # and `post` are final by here) so the range-sanity guard can read it too.
        slash_pair = bool(re.search(r"/\s*$", pre)) or bool(re.match(r"\s*/", post))
        allow_scaling = "%" in ctx or re.search(r"\bpercent", ctx, re.I) is not None
        hits = [(rn, pth) for rn, pth, rv in rvals
                if _match(num["value"], num["decimals"], rv, allow_scaling)]
        # v0.3 COUNT-BINDING rule: an integer claim only grounds in a leaf whose PATH shares a word
        # stem with the claim's line (or an n=/n_ pairing) — bare counts coincide with unrelated
        # count fields far too easily (the k=14-class D1 misses: 27->37 'verified' because a shared
        # addendum carries another experiment's n_held=37). Floats are bound at STATUS level by the
        # v0.8 clause below (the debt this comment used to name as owed to v0.4); the filter here
        # stays integer-only because the two populations need different treatment -- an integer's
        # hits are FILTERED and may fall through to UNGROUNDED, a float's are DEMOTED to ABSTAIN.
        if num["decimals"] == 0 and hits:
            words = {w.lower().strip("'’") for w in re.findall(r"[A-Za-z][A-Za-z_-]{2,}", bctx)}
            stems = {w[:4] for w in words} | {s[:4] for w in words for s in re.split(r"[-_]", w) if len(s) >= 3}
            is_n_eq = bool(re.search(r"\bn\s*=\s*$", pre, re.I))
            # slash-pair counts ('72/37', '13/16') carry their semantics jointly — bind on the pair's
            # line vocabulary, and accept digits glued to path segments ('shared48').
            stems |= {d for d in re.findall(r"\d{2,}", bctx)}
            def path_ok(p):
                segs = {s.lower() for seg in re.split(r"[.\[\]]", p) for s in re.split(r"[-_]", seg) if s}
                pst = {s[:4] for s in segs if len(s) >= 3} | {m for s in segs for m in re.findall(r"\d{2,}", s)}
                return bool(pst & stems) or (is_n_eq and any(s == "n" or s.startswith("n_") for s in segs))
            if not slash_pair:
                hits = [(rn, pth) for rn, pth in hits if path_ok(pth)]
            elif not any(path_ok(p) for _, p in hits):
                # a slash-pair still needs SOME plausible home: keep value-matching but only against
                # count-like fields (n_*/counts), else drop
                hits = [(rn, pth) for rn, pth in hits
                        if re.search(r"(^|[._\[])n_|n_held|n_caved|^n(\.|$)|count", pth, re.I)]
        # v0.5 class F: the n= register obligates only its OWN glued token (self-scoped). When the
        # class is OFF, n= falls back to the v0.4 line-wide trigger behavior.
        #
        # EPISTEMICS (2026-08-28, annotation only): `_ob_src` records the FIRST clause that set
        # `bound`, and the ladder below records which arm produced the status. Both land in the
        # ledger entry so a certificate can say, per token, which epistemic path it took — most
        # consequentially whether a VERIFIED token was ever obligated at all, which the ladder
        # RECON showed it need not be. The invariant frozen in
        # INVARIANT_epistemics_annotation_2026_08_28.md: this may move NOTHING.
        _voc = bool(_TRIGGERS.search(bctx))
        if V05_SELF_SCOPED_N:
            _ngl = bool(re.search(r"\bn\s*=\s*$", pre, re.I))
        else:
            _ngl = bool(re.search(r"\bn\s*=", bctx, re.I))
        bound = _voc or _ngl
        _ob_src = "vocabulary" if _voc else ("n-glued" if _ngl else None)
        # v0.4 decimal+range-guarded recall: the correlation/similarity register obligates a number
        # only when it is a fractional correlation (decimals > 0 and in [−1, 1]) — spares ordinals /
        # counts / API caps / whole-percents, binds RSA 0.264 / reliability 0.735.
        if not bound and num["decimals"] > 0 and -1.0 <= num["value"] <= 1.0 \
                and _TRIGGERS_CORR.search(bctx):
            bound = True
            _ob_src = "range-correlation"
        # v0.7 precision obligation: printed precision IS the binding signal, because a number at
        # this width was copied out of a computation rather than typed by a person. `precision_only`
        # records that THIS clause is the sole source of the obligation, which is what scopes the
        # ULP escape in the status ladder below — an obligation from the trigger registers, from
        # n=, or from range-sanity is never softened.
        precision_only = False
        if not bound and V07_PRECISION_OBLIGATION and num["decimals"] >= V07_PRECISION_DIGITS:
            bound, precision_only = True, True
            _ob_src = "precision"
        # v0.3 RANGE-SANITY rule: a value sitting directly after bounded-quantity vocabulary cannot
        # leave its possible range — an 'AUC 4.0' is UNGROUNDED no matter what leaf it happens to
        # match (kills the coincidence-verification class of the v0.1 battery misses).
        unit_kw = re.search(r"\b(aurocs?|aucs?|recall|precision|accuracy|fpr|fnr|concordance|"
                            r"stability|rates?|p)\s*[(=:≈~\s]*$", pre, re.I)
        sign_kw = re.search(r"\b(margins?|deltas?|elevation)\s*[(=:≈~\s]*$", pre, re.I)
        out_of_range = (unit_kw and not 0.0 <= num["value"] <= 1.0) or \
                       (sign_kw and not -1.0 <= num["value"] <= 1.0)
        # v0.10 companion: a slash-pair numerator is a COUNT, not a value of the bounded quantity
        # named to its left. '(stability 5/5)' is five of five, not a stability of 5.0.
        if V10_SLASHPAIR_RANGE_GUARD and slash_pair:
            out_of_range = False
        if out_of_range:
            hits, bound = [], True
            precision_only = False   # a range-sanity obligation is never ULP-escapable
            if _ob_src is None:      # first-writer, as the annotation contract documents --
                _ob_src = "range-sanity"   # range-sanity FORCES the accusation but does not
                                           # rewrite who obligated the token (caught in schema
                                           # red-team: this line used to clobber unconditionally)
        # v0.5 class E (derived-percent VERIFY, PREREG_oath_v05_precision): "12.7% (19/150" — a
        # percent restated by its OWN parenthetical operands verifies iff BOTH operands ground as
        # receipt values AND 100·a/b rounds to the token at the token's decimals.
        derived_ref = None
        if V05_DERIVED_PCT and not hits and not is_spec and not is_hist and not is_notation:
            dm = re.match(r"\s*%\s*\(\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", post)
            if dm:
                a_v, b_v = float(dm.group(1)), float(dm.group(2))
                a_ok = [(rn, pth) for rn, pth, rv in rvals if _match(a_v, _decimals(dm.group(1)), rv, False)]
                b_ok = [(rn, pth) for rn, pth, rv in rvals if _match(b_v, _decimals(dm.group(2)), rv, False)]
                if a_ok and b_ok and b_v != 0 and \
                        abs(round(100.0 * a_v / b_v, num["decimals"]) - num["value"]) < 1e-9:
                    derived_ref = (f"derived:{dm.group(1)}/{dm.group(2)}@"
                                   f"{a_ok[0][0]}:{a_ok[0][1]}|{b_ok[0][0]}:{b_ok[0][1]}")
        # v0.6.2 float stem-preference (attribution-only, severable): when a float claim
        # value-matches several leaves, record the hit whose path shares a word stem with the
        # claim's binding context (the v0.3 count-binding stem test) over a coincidental first
        # hit. STATUS untouched — a stable sort reorders `hits`, never empties it. Full
        # claim->field binding for floats (status-level) remains the named v0.4 debt.
        if V062_FLOAT_STEM_PREF and num["decimals"] > 0 and len(hits) > 1:
            words = {w.lower().strip("'’") for w in re.findall(r"[A-Za-z][A-Za-z_-]{2,}", bctx)}
            stems = {w[:4] for w in words} | {s[:4] for w in words for s in re.split(r"[-_]", w) if len(s) >= 3}
            def _stem_ok(p):
                segs = {s.lower() for seg in re.split(r"[.\[\]]", p) for s in re.split(r"[-_]", seg) if s}
                return bool({s[:4] for s in segs if len(s) >= 3} & stems)
            hits = sorted(hits, key=lambda h: (not _stem_ok(h[1]),))
        # v0.8 STATUS-LEVEL claim->field binding for FLOAT claims -- the standing v0.4 debt that
        # the v0.3 count-binding comment above names. A float that value-matches ONLY leaves whose
        # PATH is unrelated to its context is not grounded, it is coincident: that is exactly how a
        # mutated number keeps a VERIFIED status by landing on some other leaf. Unlike an
        # obligation predicate this survives the mutation it exists to catch, because it reads
        # receipt PATHS and doc CONTEXT, never the claim's value.
        #
        # DEMOTE-ONLY. It intercepts claims that would otherwise be VERIFIED and sends them to
        # ABSTAIN; it can never produce or remove an UNGROUNDED, so no certificate can flip
        # HELD->FAILED. That is invariant I1 -- asserted in the suite, deliberately not gated.
        field_unbound_ref = None
        if (V08_FLOAT_FIELD_BINDING and hits
                and 0 < num["decimals"] <= V08_FIELD_BIND_MAX_DECIMALS):
            i0 = max(0, num["line"] - 1 - V08_FIELD_BIND_PREV_LINES)
            prev = [doc_lines[k].strip().replace("−", "-") for k in range(i0, num["line"] - 1)]
            wstems = _ctx_stems(" ".join(prev + [bctx])[:800])
            if not any(_path_stems(pth) & wstems for _, pth in hits):
                # NAMEABLE: withhold the oath only where binding was POSSIBLE and failed. If the
                # cited receipts carry NO path the sentence names, the claim is unbindable in
                # principle (an acronym field like frozen_gates.CG1_SEP under a line reading
                # "floor 0.10") and demoting it would be a pure coverage loss buying nothing.
                if all_path_stems & wstems:
                    field_unbound_ref = f"unbound-field:{hits[0][0]}:{hits[0][1]}"
        if is_spec or is_hist:
            status, ref = "ABSTAIN", "spec-or-historical"
            _branch = "spec-or-historical"
        elif is_notation:
            status, ref = "ABSTAIN", "v05-notation"
            _branch = "notation"
        elif derived_ref:
            status, ref = "VERIFIED", derived_ref
            _branch = "derived"
        elif field_unbound_ref:
            # v0.8: the claim's value is in the receipts, but not in any field its context names.
            # The oath is withheld, not inverted -- ABSTAIN names the gap and stays countable.
            status, ref = "ABSTAIN", field_unbound_ref
            _branch = "unbound-field"
        elif hits:
            status = "VERIFIED"
            ref = f"{hits[0][0]}:{hits[0][1]}"
            _branch = "value-match"
        elif bound:
            # NOTE (v0.3): a bulk-row match deliberately does NOT soften this to ABSTAIN — letting
            # claims ground in per-item arrays let 13/20 seeded mutants hide in row noise when it
            # was tried. The cure for a legitimate grid-cell cite is persisting it as a summary
            # field in an addendum receipt (the repair loop), not weakening the oath.
            # v0.7: the ONE softening the oath admits, and only for an obligation the precision
            # clause created by itself — a leaf within V07_ULP_N ULP means the receipt holds this
            # measurement, reached by differently ordered arithmetic. Abstaining names it; it is
            # never upgraded to VERIFIED, so the claim is still not sworn to.
            nb = (_ulp_neighbour(num["value"], rvals, V07_ULP_N)
                  if (V07_ULP_ESCAPE and precision_only) else None)
            if nb:
                status, ref = "ABSTAIN", f"ulp-neighbour:{nb[0]}:{nb[1]}"
                _branch = "ulp-neighbour"
            else:
                status, ref = "UNGROUNDED", None
                _branch = "obligated-accusation"
        else:
            status = "ABSTAIN"
            ref = None
            _branch = "silent"
        # The epistemic path, recorded rather than discarded. `obligated` is `bound` at ladder
        # time; a VERIFIED entry with obligated=False is an UNOBLIGATED OATH -- the verifier swore
        # to a value nothing required it to examine. `path_checked` says whether the v0.3 integer
        # count-binding filter ran; for decimals it never does (v0.8 CLOSED_NEGATIVE), which is
        # the gpu_memory_fraction class of binding. Annotation only; see the frozen invariant.
        _ep = {"branch": _branch, "obligated": bool(bound), "obligation_source": _ob_src}
        if _branch == "value-match":
            _ep["path_checked"] = num["decimals"] == 0
        ledger.append({**num, "status": status, "receipt_ref": ref, "epistemics": _ep})

    counts = {s: sum(1 for c in ledger if c["status"] == s) for s in ("VERIFIED", "ABSTAIN", "UNGROUNDED")}
    summary = _epistemics_summary(ledger, counts)
    cert = {
        "oath": "styxx OATH v0 (numeric-claim certificate)",
        "prereg": "papers/closed-model-frontier/PREREG_oath_v0_certify_doc_2026_06_09.md",
        "document": doc_path.name,
        "document_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "receipts_sha256": receipts,
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "counts": counts,
        "epistemics_summary": summary,
        "verdict": "OATH-HELD" if counts["UNGROUNDED"] == 0 else "OATH-FAILED",
        "ungrounded": [c for c in ledger if c["status"] == "UNGROUNDED"],
        "abstained": [{"line": c["line"], "token": c["token"]} for c in ledger if c["status"] == "ABSTAIN"],
        "ledger": ledger,
    }
    return cert


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="styxx.certify")
    ap.add_argument("doc")
    ap.add_argument("receipts", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cert = certify_doc(Path(a.doc), [Path(r) for r in a.receipts])
    out = Path(a.out) if a.out else Path(a.doc).with_suffix(".certificate.json")
    out.write_text(json.dumps(cert, indent=2) + "\n", encoding="utf-8")
    c = cert["counts"]
    print(f"{cert['verdict']}  verified={c['VERIFIED']} abstained={c['ABSTAIN']} "
          f"contradicted={c['UNGROUNDED']}  -> {out.name}")
    # A verified count alone is the green-checkmark half-truth this instrument exists to reject:
    # it says nothing about how much of what was sworn was ever obligated. Print the split so the
    # boundary is visible without opening the JSON.
    es = cert["epistemics_summary"]["verified"]
    obl = (es["value_match"]["obligated_integer_filter_ran"]
           + es["value_match"]["obligated_integer_filter_na"] + es["derived"]["obligated"])
    if es["total"]:
        unobl = es["total"] - obl
        weakest = es["value_match"]["unobligated_integer_filter_na"]
        print(f"  of {es['total']} verified: {obl} obligated, {unobl} volunteered "
              f"({round(unobl / es['total'] * 100)}%) — {weakest} by value match alone")
    for bad in cert["ungrounded"]:
        print(f"  UNGROUNDED L{bad['line']}: {bad['token']}  | {bad['context'][:100]}")
    return 0 if cert["verdict"] == "OATH-HELD" else 1


if __name__ == "__main__":
    sys.exit(main())
