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


def extract_numbers(text: str) -> list[dict]:
    """All groundable number tokens with line context. Filters dates/SHAs/versions/years/markdown
    artifacts and formula notation; keeps order and position so the ledger is reviewable.

    v0.3: markdown table rows inherit their table's HEADER line as additional binding context —
    the trigger vocabulary of '| regime | AUC-g | margin |' binds the numbers in every data row."""
    out = []
    lines = text.splitlines()
    header_for: dict[int, str] = {}
    for i, line in enumerate(lines):
        if _TABLE_SEP.match(line) and i > 0 and lines[i - 1].lstrip().startswith("|"):
            hdr = lines[i - 1].strip()
            j = i + 1
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                header_for[j + 1] = hdr   # 1-based line numbers
                j += 1
    for ln_no, line in enumerate(lines, 1):
        # v0.6.2 signed extraction: the typographic minus U+2212 becomes ASCII '-' so _NUM reads
        # it as a sign — pre-fix, '−0.0154' extracted as POSITIVE 0.0154 and an accurate negative
        # claim could be accused (the v0.6.1 G3 kill). En-dash ranges (U+2013) are untouched.
        line = line.replace("−", "-")
        # drop fenced/sha/date/version spans from the searchable line
        scrub = _SHAISH.sub(" ", line)
        scrub = _DATEISH.sub(" ", scrub)
        scrub = _VERSIONISH.sub(" ", scrub)
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
V07_ULP_N = 8                     # v0.6.2 withdrew the epsilon subsidy at >=13 decimals, so at
                                  # doc_dec=16 the tolerance (5e-17) sits BELOW the float64 ULP
                                  # near 1.0 (1.11e-16). That was safe while such tokens were never
                                  # obligated; the clause above makes it live, and without this
                                  # escape a restatement of the SAME measurement by differently
                                  # ordered arithmetic becomes a loud accusation. Yielding ABSTAIN
                                  # rather than VERIFIED is what keeps the v0.6.2 epsilon hole
                                  # closed, and the countable `ulp-neighbour` reason is what keeps
                                  # the residual enumerable instead of invisible.


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


def certify_doc(doc_path: Path, receipt_paths: list[Path]) -> dict:
    text = doc_path.read_text(encoding="utf-8")
    receipts = {}
    rvals: list[tuple[str, str, float]] = []   # (receipt, path, value)  — summary surface only
    for rp in receipt_paths:
        j = json.loads(rp.read_text(encoding="utf-8"))
        receipts[rp.name] = hashlib.sha256(rp.read_bytes()).hexdigest()
        for path, v in receipt_values(j):
            rvals.append((rp.name, path, v))

    ledger = []
    doc_lines = text.splitlines()
    for num in extract_numbers(text):
        # v0.1 SPEC-CONSTANT rule: a number that is a pre-registered bar/threshold, a CI confidence
        # level, or a comparison bound is SPEC, not a measurement -> ABSTAIN (it has no receipt by
        # design; its receipt is the PREREG document).
        # v0.3: rules test the FULL line — the display context truncates at 160 chars and a
        # disclosure note past that boundary was invisible to is_hist (caught in the D2 hand-check).
        # v0.6.2: same U+2212 normalization as extraction, so signed tokens are findable in ctx
        ctx = doc_lines[num["line"] - 1].strip().replace("−", "-")
        bctx = num.get("binding_context", ctx)   # v0.3: table rows bind via their header too
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
        allow_scaling = "%" in ctx or re.search(r"\bpercent", ctx, re.I) is not None
        hits = [(rn, pth) for rn, pth, rv in rvals
                if _match(num["value"], num["decimals"], rv, allow_scaling)]
        # v0.3 COUNT-BINDING rule: an integer claim only grounds in a leaf whose PATH shares a word
        # stem with the claim's line (or an n=/n_ pairing) — bare counts coincide with unrelated
        # count fields far too easily (the k=14-class D1 misses: 27->37 'verified' because a shared
        # addendum carries another experiment's n_held=37). Floats keep value-only matching (v0.4
        # owes them full claim->field binding).
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
            slash_pair = bool(re.search(r"/\s*$", pre)) or bool(re.match(r"\s*/", post))
            if not slash_pair:
                hits = [(rn, pth) for rn, pth in hits if path_ok(pth)]
            elif not any(path_ok(p) for _, p in hits):
                # a slash-pair still needs SOME plausible home: keep value-matching but only against
                # count-like fields (n_*/counts), else drop
                hits = [(rn, pth) for rn, pth in hits
                        if re.search(r"(^|[._\[])n_|n_held|n_caved|^n(\.|$)|count", pth, re.I)]
        # v0.5 class F: the n= register obligates only its OWN glued token (self-scoped). When the
        # class is OFF, n= falls back to the v0.4 line-wide trigger behavior.
        if V05_SELF_SCOPED_N:
            bound = bool(_TRIGGERS.search(bctx)) or bool(re.search(r"\bn\s*=\s*$", pre, re.I))
        else:
            bound = bool(_TRIGGERS.search(bctx)) or bool(re.search(r"\bn\s*=", bctx, re.I))
        # v0.4 decimal+range-guarded recall: the correlation/similarity register obligates a number
        # only when it is a fractional correlation (decimals > 0 and in [−1, 1]) — spares ordinals /
        # counts / API caps / whole-percents, binds RSA 0.264 / reliability 0.735.
        if not bound and num["decimals"] > 0 and -1.0 <= num["value"] <= 1.0 \
                and _TRIGGERS_CORR.search(bctx):
            bound = True
        # v0.7 precision obligation: printed precision IS the binding signal, because a number at
        # this width was copied out of a computation rather than typed by a person. `precision_only`
        # records that THIS clause is the sole source of the obligation, which is what scopes the
        # ULP escape in the status ladder below — an obligation from the trigger registers, from
        # n=, or from range-sanity is never softened.
        precision_only = False
        if not bound and V07_PRECISION_OBLIGATION and num["decimals"] >= V07_PRECISION_DIGITS:
            bound, precision_only = True, True
        # v0.3 RANGE-SANITY rule: a value sitting directly after bounded-quantity vocabulary cannot
        # leave its possible range — an 'AUC 4.0' is UNGROUNDED no matter what leaf it happens to
        # match (kills the coincidence-verification class of the v0.1 battery misses).
        unit_kw = re.search(r"\b(aurocs?|aucs?|recall|precision|accuracy|fpr|fnr|concordance|"
                            r"stability|rates?|p)\s*[(=:≈~\s]*$", pre, re.I)
        sign_kw = re.search(r"\b(margins?|deltas?|elevation)\s*[(=:≈~\s]*$", pre, re.I)
        out_of_range = (unit_kw and not 0.0 <= num["value"] <= 1.0) or \
                       (sign_kw and not -1.0 <= num["value"] <= 1.0)
        if out_of_range:
            hits, bound = [], True
            precision_only = False   # a range-sanity obligation is never ULP-escapable
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
        if is_spec or is_hist:
            status, ref = "ABSTAIN", "spec-or-historical"
        elif is_notation:
            status, ref = "ABSTAIN", "v05-notation"
        elif derived_ref:
            status, ref = "VERIFIED", derived_ref
        elif hits:
            status = "VERIFIED"
            ref = f"{hits[0][0]}:{hits[0][1]}"
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
            else:
                status, ref = "UNGROUNDED", None
        else:
            status = "ABSTAIN"
            ref = None
        ledger.append({**num, "status": status, "receipt_ref": ref})

    counts = {s: sum(1 for c in ledger if c["status"] == s) for s in ("VERIFIED", "ABSTAIN", "UNGROUNDED")}
    cert = {
        "oath": "styxx OATH v0 (numeric-claim certificate)",
        "prereg": "papers/closed-model-frontier/PREREG_oath_v0_certify_doc_2026_06_09.md",
        "document": doc_path.name,
        "document_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "receipts_sha256": receipts,
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "counts": counts,
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
    for bad in cert["ungrounded"]:
        print(f"  UNGROUNDED L{bad['line']}: {bad['token']}  | {bad['context'][:100]}")
    return 0 if cert["verdict"] == "OATH-HELD" else 1


if __name__ == "__main__":
    sys.exit(main())
