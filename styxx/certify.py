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
_SHAISH = re.compile(r"\b[0-9a-f]{7,64}\b")
_VERSIONISH = re.compile(r"\bv?\d+\.\d+\.\d+\b")


def _decimals(tok: str) -> int:
    return len(tok.split(".")[1]) if "." in tok else 0


_TABLE_SEP = re.compile(r"^\s*\|[\s:|-]+\|?\s*$")
_FORMULA_AFTER = re.compile(r"^\s?[−–-]\s?[A-Za-z]")   # '1−syc', '1-dec': notation, not a claim


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
        # drop fenced/sha/date/version spans from the searchable line
        scrub = _SHAISH.sub(" ", line)
        scrub = _DATEISH.sub(" ", scrub)
        scrub = _VERSIONISH.sub(" ", scrub)
        for m in _NUM.finditer(scrub):
            tok = m.group(0)
            raw = tok.replace(",", "")
            if _YEAR.match(raw.lstrip("+-")):
                continue
            # markdown heading/bullet/link artifacts: a bare int at line start
            if m.start() <= 2 and "." not in raw and abs(int(raw)) < 10:
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
        tol = 0.5 * 10 ** (-doc_dec) + 1e-12
        if abs(round(r_val, doc_dec) - doc_val) <= 1e-12 or abs(r_val - doc_val) <= tol:
            return True
    if allow_scaling:
        # percent <-> fraction (doc says 80%, receipt holds 0.80; or doc 0.8 vs receipt 80)
        for scale in (100.0, 0.01):
            rv = r_val * scale
            if doc_val == rv:
                return True
            if doc_dec > 0 and abs(round(rv, doc_dec) - doc_val) <= 0.5 * 10 ** (-doc_dec) + 1e-12:
                return True
    return False


# ---------------------------------------------------------------- contradiction triggers

# context keywords that bind a doc number to receipt quantities (the v0 trigger vocabulary).
# a number whose line mentions one of these AND whose receipts carry a same-kind quantity is
# OBLIGATED to match some receipt value, else UNGROUNDED.
_TRIGGERS = re.compile(
    r"\b(aurocs?|aucs?|margins?|cis?\b|boot(strap)?|perm(utation)?(_p\d+)?|p9\d|recall|precision|"
    r"fpr|fnr|accuracy|rate|median|mean|elevation|floor|delta|n\s*=|n_held|n_caved|held|caved|"
    r"gated|dropped|grounded|sycophancy|deception|surface|lens|firewall|collapse|wilson|"
    r"concordance|stability|score[sd]?)\b", re.I)


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
        ctx = doc_lines[num["line"] - 1].strip()
        bctx = num.get("binding_context", ctx)   # v0.3: table rows bind via their header too
        tok_at = ctx.find(num["token"])
        pre = ctx[max(0, tok_at - 18):tok_at] if tok_at >= 0 else ""
        # v0.3: a token at line start inherits the tail of the previous line as pre-context —
        # 'subclass AUC\n1.0)' wraps mid-sentence and the unit keyword must still bind.
        if 0 <= tok_at < 18 and num["line"] >= 2:
            pre = (doc_lines[num["line"] - 2].strip()[-(18 - tok_at):] + " " + pre).strip()[-24:]
        post = ctx[tok_at + len(num["token"]):] if tok_at >= 0 else ""
        is_spec = bool(re.search(r"[≥≤<>=]\s*\+?$|\b(bar|gate|threshold|requires?|must|pre-?registered)"
                                 r"\b[^.]{0,16}$", pre)) \
            or bool(re.match(r"\s*%?\s*(CI|confidence)", post)) \
            or bool(re.match(r"[^.\d]{0,12}\b(bar|threshold|gate)\b", post))
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
        bound = bool(_TRIGGERS.search(bctx))
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
        if is_spec or is_hist:
            status, ref = "ABSTAIN", "spec-or-historical"
        elif hits:
            status = "VERIFIED"
            ref = f"{hits[0][0]}:{hits[0][1]}"
        elif bound:
            # NOTE (v0.3): a bulk-row match deliberately does NOT soften this to ABSTAIN — letting
            # claims ground in per-item arrays let 13/20 seeded mutants hide in row noise when it
            # was tried. The cure for a legitimate grid-cell cite is persisting it as a summary
            # field in an addendum receipt (the repair loop), not weakening the oath.
            status = "UNGROUNDED"
            ref = None
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
