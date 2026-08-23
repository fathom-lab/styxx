"""OATH v0.9 pre-fix census — the size and the SHAPE of the `is_spec` recall gap.

Run at the SHIPPED verifier, before any edit to ``styxx/certify.py``. Deterministic and local.

`is_spec` (the v0.1 SPEC-CONSTANT rule) abstains a number that is a pre-registered bar: its
receipt is the prereg document, not a result JSON. It recognises a bar only by an operator
character or bar vocabulary inside the 18-character window immediately BEFORE the token, so it
fires on ``bar >= 0.75`` and misses every bar whose operator or noun sits elsewhere.

Four measurements, because the class splits in two and the two halves behave OPPOSITELY:

  1. THE JSON-IDIOM ROSTER (repo-wide). A bar written as ``{"op": "<=", "value": 0.00648}`` puts
     its operator in a separate field. Counts the class, how many the shipped `is_spec` rescues,
     and -- the number that decides everything -- how many sit on lines the shipped OBLIGATION
     predicate binds.

  2. THE BAR-NOUN ROSTER (certified corpus). Prose that names the bar AFTER the number
     ("clears the 0.10 floor"). Same three counts.

  3. THE MUTATION LEDGER of the bar-noun roster at the shipped verifier. One significant digit
     perturbed per token, seeded. Splits the roster into mutants the shipped verifier CATCHES
     (UNGROUNDED) and mutants that come back VERIFIED against an unrelated leaf (FALSE
     ATTESTATION). An `is_spec` extension converts BOTH to ABSTAIN, so this is the price list:
     abstention buys away the false attestations and destroys the catches in the same stroke.

  4. THE RECEIPT-SIDE ALTERNATIVE, sized and reported: claims whose grounding leaf sits under a
     `frozen_gates`-like container. Reported so the design rejected in the prereg is rejected
     against a number.

  python papers/closed-model-frontier/oath_v09_isspec_census.py
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx.certify import (certify_doc, extract_numbers,                   # noqa: E402
                           _TRIGGERS, _TRIGGERS_CORR)
from styxx.corpus_audit import _resolve_receipts                           # noqa: E402

OUT = HERE / "oath_v09_isspec_census.json"
SEED = 9
MUT_SEEDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
ADJUDICATION_SAMPLE = 25

# ---- the two candidate predicates, defined HERE so the census runs at the UNMODIFIED verifier.
# Byte-identical copies land in styxx/certify.py only after this census is committed.
_JSON_OP_FIELD = re.compile(r'"(?:op|operator|cmp|comparison|direction|sense)"\s*:\s*"\s*'
                            r'(?:<=|>=|!=|==|<|>|=)\s*"')
_JSON_BAR_KEY = re.compile(r'"(?:value|bar|threshold|floor|ceiling|cutoff|min|max|alpha|target'
                           r'|bound)"\s*:\s*$')
_BAR_NOUN_POST = re.compile(r"^[ \t\-]{0,2}(?:floors?|ceilings?|cutoffs?|caps?|bounds?)\b", re.I)

# the SHIPPED is_spec, reproduced verbatim from certify.py (V05_APPROX_NOTATION is False on main)
_SHIPPED_SPEC_PRE = re.compile(r"[≥≤<>=]\s*\+?$|\b(bar|gate|threshold|requires?|must|pre-?registered)"
                               r"\b[^.]{0,16}$")
_SHIPPED_SPEC_POST_CI = re.compile(r"\s*%?\s*(CI|confidence)")
_SHIPPED_SPEC_POST_BAR = re.compile(r"[^.\d]{0,12}\b(bar|threshold|gate)\b")

_SPEC_PATH = re.compile(r"(frozen_gates|kill_gates|kill_gate|gates|bars|prereg|thresholds?)\b",
                        re.I)


def shipped_is_spec(pre: str, post: str) -> bool:
    return bool(_SHIPPED_SPEC_PRE.search(pre)) or bool(_SHIPPED_SPEC_POST_CI.match(post)) \
        or bool(_SHIPPED_SPEC_POST_BAR.match(post))


def is_bound_shipped(bctx: str, value: float, decimals: int) -> bool:
    """The shipped obligation predicate (trigger register / fractional-correlation / precision).

    This is the number that decides the cycle: only an obligated token can be CAUGHT when it is
    mutated, so only an obligated token has a catch for an abstention rule to destroy."""
    if _TRIGGERS.search(bctx):
        return True
    if decimals > 0 and -1.0 <= value <= 1.0 and _TRIGGERS_CORR.search(bctx):
        return True
    return decimals >= 7


def windows(line: str, token: str):
    """The pre/post windows `certify_doc` builds, for one token on one line."""
    ctx = line.strip().replace("−", "-")
    at = ctx.find(token)
    if at < 0:
        return ctx, "", ""
    return ctx, ctx[max(0, at - 18):at], ctx[at + len(token):]


def resolvable_docs():
    """Documents under papers/** carrying a certificate whose every recorded receipt resolves."""
    out = []
    for cp in sorted(ROOT.glob("papers/**/*.certificate.json")):
        if "anc" in cp.parts:
            continue
        doc = cp.with_name(cp.name.replace(".certificate.json", ".md"))
        if not doc.exists():
            continue
        try:
            rec = json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            continue
        receipts, missing, _ = _resolve_receipts(cp, rec)
        if receipts and not missing:
            out.append((doc, receipts))
    return out


def mutate_sig(tok: str, rng: random.Random) -> str:
    """Perturb one significant digit — fractional positions 1-6, else any integer digit."""
    if "." in tok:
        frac_at = tok.index(".") + 1
        frac = tok[frac_at:]
        sig = [i for i in range(min(6, len(frac)))
               if not (frac[i] == "0" and set(frac[:i]) <= {"0"})]
        pos = frac_at + rng.choice(sig or [0])
    else:
        pos = rng.choice([i for i, ch in enumerate(tok) if ch.isdigit()])
    old = int(tok[pos])
    return tok[:pos] + str(rng.choice([d for d in range(10) if d != old])) + tok[pos + 1:]


def substitute(line: str, tok: str, mut: str):
    """Land *mut* in place of *tok*, honouring the typographic minus.

    Extraction normalizes U+2212 to ASCII '-', so a negative token is REPORTED in ASCII while the
    document holds U+2212 and a bare ``line.replace`` silently no-ops — scoring a harness miss as
    a verifier miss. Inherited verbatim from `run_oath_v07_battery.py`, which owns the defect."""
    if tok in line:
        return line.replace(tok, mut, 1), True
    if tok.startswith("-"):
        alt, alt_mut = tok.replace("-", "−", 1), mut.replace("-", "−", 1)
        if alt in line:
            return line.replace(alt, alt_mut, 1), True
    return line, False


def main() -> int:
    docs = resolvable_docs()
    print(f"documents with fully-resolvable receipts: {len(docs)}")

    # ---------------------------------------------------------------- 1. JSON-idiom, repo-wide
    json_roster, md_scanned = [], 0
    for md in sorted(ROOT.glob("papers/**/*.md")):
        if "anc" in md.parts:
            continue
        md_scanned += 1
        text = md.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        op_lines = {i for i, ln in enumerate(lines, 1) if _JSON_OP_FIELD.search(ln)}
        if not op_lines:
            continue
        for e in extract_numbers(text):
            if e["line"] not in op_lines:
                continue
            ctx, pre, post = windows(lines[e["line"] - 1], e["token"])
            if not _JSON_BAR_KEY.search(pre):
                continue
            bctx = e.get("binding_context", e["context"])
            json_roster.append({
                "doc": md.name, "rel": md.relative_to(ROOT).as_posix(), "line": e["line"],
                "token": e["token"], "decimals": e["decimals"],
                "rescued_by_shipped_is_spec": shipped_is_spec(pre, post),
                "bound_shipped": is_bound_shipped(bctx, e["value"], e["decimals"]),
                "context": ctx[:160]})
    # The ADJUDICATION FRAME: every token on a line carrying a comparison-operator field, whether
    # or not it sits in value position. Defined by the operator field ALONE, so the clause's own
    # value-position requirement does not define its own test set. Disclosed limitation: the frame
    # is nearly coextensive with the clause (146 vs 145), so it bounds the clause's precision
    # tightly but supplies little independent evidence about its recall.
    frame, md2 = [], 0
    for md in sorted(ROOT.glob("papers/**/*.md")):
        if "anc" in md.parts:
            continue
        md2 += 1
        lines = md.read_text(encoding="utf-8", errors="replace").splitlines()
        op_lines = {i for i, ln in enumerate(lines, 1) if _JSON_OP_FIELD.search(ln)}
        if not op_lines:
            continue
        for e in extract_numbers(md.read_text(encoding="utf-8", errors="replace")):
            if e["line"] not in op_lines:
                continue
            ctx, pre, post = windows(lines[e["line"] - 1], e["token"])
            frame.append({"doc": md.name, "rel": md.relative_to(ROOT).as_posix(),
                          "line": e["line"], "token": e["token"],
                          "in_value_position": bool(_JSON_BAR_KEY.search(pre)),
                          "rescued_by_shipped_is_spec": shipped_is_spec(pre, post),
                          "context": ctx[:160]})
    print(f"    adjudication frame (operator-field lines): {len(frame)} tokens, "
          f"{sum(f['in_value_position'] for f in frame)} in value position, "
          f"{sum(f['rescued_by_shipped_is_spec'] for f in frame)} rescued today")

    certed = {p.name.replace(".certificate.json", ".md")
              for p in ROOT.glob("papers/**/*.certificate.json") if "anc" not in p.parts}
    resolvable_names = {d.name for d, _ in docs}
    json_docs = sorted({r["doc"] for r in json_roster})
    print(f"\n[1] JSON-idiom bar tokens repo-wide ({md_scanned} markdown documents): "
          f"{len(json_roster)} in {len(json_docs)} documents")
    print(f"    rescued by the shipped is_spec       : "
          f"{sum(r['rescued_by_shipped_is_spec'] for r in json_roster)}")
    print(f"    on lines the shipped predicate BINDS : "
          f"{sum(r['bound_shipped'] for r in json_roster)}")
    print(f"    in a document carrying a certificate : "
          f"{sum(1 for r in json_roster if r['doc'] in certed)}")
    print(f"    in a FULLY-RESOLVABLE certified doc  : "
          f"{sum(1 for r in json_roster if r['doc'] in resolvable_names)}")

    # ---- the live exhibit: committed certificates that already attest a JSON-idiom bar.
    live = []
    for cp in sorted(ROOT.glob("papers/**/*.certificate.json")):
        if "anc" in cp.parts:
            continue
        name = cp.name.replace(".certificate.json", ".md")
        keys = {(r["line"], r["token"]) for r in json_roster if r["doc"] == name}
        if not keys:
            continue
        try:
            rec = json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            continue
        for e in rec.get("ledger", []):
            if (e["line"], e["token"]) in keys:
                live.append({"certificate": cp.name, "line": e["line"], "token": e["token"],
                             "status": e["status"], "receipt_ref": e.get("receipt_ref"),
                             "context": e.get("context", "")[:150]})
    print(f"    LIVE in committed certificates       : {len(live)}")
    for r in live:
        print(f"      {r['certificate'][:52]:52s} L{r['line']:<5d} {r['token']:<8s} "
              f"{r['status']:9s} <- {r['receipt_ref']}")

    # ---------------------------------------------------------------- 2. bar-noun, certified corpus
    ledger_rows, bar_noun = [], []
    for doc, receipts in docs:
        lines = doc.read_text(encoding="utf-8", errors="replace").splitlines()
        try:
            cert = certify_doc(doc, receipts)
        except Exception as exc:                                  # pragma: no cover - defensive
            print(f"    SKIP {doc.name}: {exc}")
            continue
        for e in cert["ledger"]:
            ctx, pre, post = windows(lines[e["line"] - 1], e["token"])
            row = {"doc": doc.name, "line": e["line"], "token": e["token"],
                   "decimals": e["decimals"], "status": e["status"],
                   "receipt_ref": e["receipt_ref"], "context": ctx[:160]}
            ledger_rows.append(row)
            if _BAR_NOUN_POST.match(post) and not shipped_is_spec(pre, post):
                bctx = e.get("binding_context", e["context"])
                bar_noun.append({**row, "bound_shipped": is_bound_shipped(
                    bctx, e["value"], e["decimals"])})
    corpus_counts = collections.Counter(r["status"] for r in ledger_rows)
    print(f"\n[2] certified corpus: {len(ledger_rows)} tokens  {dict(corpus_counts)}")
    print(f"    bar-noun tokens the shipped is_spec misses: {len(bar_noun)} "
          f"{dict(collections.Counter(r['status'] for r in bar_noun))}")
    print(f"    of those, on lines the shipped predicate BINDS: "
          f"{sum(r['bound_shipped'] for r in bar_noun)}")

    # ---------------------------------------------------------------- 3. the mutation ledger
    # Run over MUT_SEEDS, not one seed. A single seed puts the catch/false-attestation contrast
    # on either side of zero (seed 7 reads 19 vs 17; seed 9 reads 16 vs 20), so a one-seed net is
    # noise and reporting it as the result would be cherry-picking. What does NOT move with the
    # seed is the quantity that matters: an abstention rule takes the CATCH column to zero at
    # every seed, because the predicate reads context and a mutant's context is unchanged.
    doc_by_name = {d.name: (d, rc) for d, rc in docs}
    per_seed, mut_rows, unlanded = [], [], 0
    for seed in MUT_SEEDS:
        rng = random.Random(seed)
        rows = []
        for r in bar_noun:
            doc, receipts = doc_by_name[r["doc"]]
            lines = doc.read_text(encoding="utf-8", errors="replace").splitlines()
            mut = mutate_sig(r["token"], rng)
            ml = list(lines)
            ml[r["line"] - 1], landed = substitute(ml[r["line"] - 1], r["token"], mut)
            unlanded += not landed
            with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False,
                                             encoding="utf-8") as tf:
                tf.write("\n".join(ml))
                tmp = Path(tf.name)
            try:
                cert = certify_doc(tmp, receipts)
            finally:
                tmp.unlink(missing_ok=True)
            e = next((x for x in cert["ledger"]
                      if x["line"] == r["line"] and x["token"] == mut), None)
            rows.append({"seed": seed, "doc": r["doc"], "line": r["line"], "token": r["token"],
                         "mutant": mut, "landed": landed, "clean_status": r["status"],
                         "bound_shipped": r["bound_shipped"],
                         "mutant_status": e["status"] if e else "NOT_EXTRACTED",
                         "mutant_ref": (e or {}).get("receipt_ref")})
        c = collections.Counter(x["mutant_status"] for x in rows)
        per_seed.append({"seed": seed, "caught_ungrounded": c["UNGROUNDED"],
                         "false_attested_verified": c["VERIFIED"], "abstained": c["ABSTAIN"],
                         "other": sum(v for k, v in c.items()
                                      if k not in ("UNGROUNDED", "VERIFIED", "ABSTAIN"))})
        mut_rows.extend(rows)
    n_seeds = len(MUT_SEEDS)
    caught = sum(s["caught_ungrounded"] for s in per_seed) / n_seeds
    false_attest = sum(s["false_attested_verified"] for s in per_seed) / n_seeds
    print(f"\n[3] one-digit mutation of every bar-noun token, SHIPPED verifier, "
          f"{n_seeds} seeds {MUT_SEEDS} ({unlanded} mutants did not land):")
    for s in per_seed:
        print(f"    seed {s['seed']:<3d} caught {s['caught_ungrounded']:3d}   "
              f"false-attested {s['false_attested_verified']:3d}   "
              f"abstained {s['abstained']:3d}")
    print(f"    mean CATCHES an is_spec extension would destroy      : {caught:.1f} "
          f"(range {min(s['caught_ungrounded'] for s in per_seed)}-"
          f"{max(s['caught_ungrounded'] for s in per_seed)})")
    print(f"    mean FALSE ATTESTATIONS an is_spec extension removes : {false_attest:.1f} "
          f"(range {min(s['false_attested_verified'] for s in per_seed)}-"
          f"{max(s['false_attested_verified'] for s in per_seed)})")
    print(f"    tokens on UNBOUND lines (no catch is possible there) : "
          f"{sum(1 for r in bar_noun if not r['bound_shipped'])} of {len(bar_noun)}")

    # ---------------------------------------------------------------- 4. receipt-side alternative
    spec_ref = [r for r in ledger_rows if r["status"] == "VERIFIED"
                and isinstance(r["receipt_ref"], str) and _SPEC_PATH.search(r["receipt_ref"])]
    bar_keys = {(r["doc"], r["line"], r["token"]) for r in bar_noun}
    print(f"\n[4] receipt-side alternative (grounding leaf under a frozen_gates-like container):")
    print(f"    VERIFIED tokens so grounded: {len(spec_ref)}")
    print(f"    already reached by the doc-side bar-noun predicate: "
          f"{sum(1 for r in spec_ref if (r['doc'], r['line'], r['token']) in bar_keys)}")

    # ---------------------------------------------------------------- frozen adjudication sample
    rs = random.Random(SEED)
    sample = rs.sample(frame, min(ADJUDICATION_SAMPLE, len(frame)))
    sample = sorted(sample, key=lambda r: (r["doc"], r["line"], r["token"]))

    report = {
        "prereg": "PREREG_oath_v09_is_spec_json_idiom_2026_08_23.md",
        "generated_at_verifier_sha256":
            hashlib.sha256((ROOT / "styxx" / "certify.py").read_bytes()).hexdigest(),
        "seed": SEED,
        "resolvable_documents": len(docs),
        "markdown_documents_scanned": md_scanned,
        "json_idiom": {
            "n": len(json_roster),
            "documents": len(json_docs),
            "rescued_by_shipped_is_spec": sum(r["rescued_by_shipped_is_spec"]
                                              for r in json_roster),
            "on_bound_lines": sum(r["bound_shipped"] for r in json_roster),
            "in_certified_document": sum(1 for r in json_roster if r["doc"] in certed),
            "in_fully_resolvable_document": sum(1 for r in json_roster
                                                if r["doc"] in resolvable_names),
            "decimal_widths": dict(sorted(collections.Counter(
                r["decimals"] for r in json_roster).items())),
            "live_in_committed_certificates": live,
            "roster": json_roster,
        },
        "bar_noun": {
            "n": len(bar_noun),
            "status_counts": dict(collections.Counter(r["status"] for r in bar_noun)),
            "on_bound_lines": sum(r["bound_shipped"] for r in bar_noun),
            "roster": bar_noun,
        },
        "mutation_ledger": {
            "seeds": list(MUT_SEEDS),
            "tokens_per_seed": len(bar_noun),
            "did_not_land": unlanded,
            "per_seed": per_seed,
            "mean_catches_destroyed_by_abstention": round(caught, 2),
            "mean_false_attestations_removed_by_abstention": round(false_attest, 2),
            "catches_range": [min(s["caught_ungrounded"] for s in per_seed),
                              max(s["caught_ungrounded"] for s in per_seed)],
            "false_attestation_range": [min(s["false_attested_verified"] for s in per_seed),
                                        max(s["false_attested_verified"] for s in per_seed)],
            "rows": mut_rows,
        },
        "receipt_side_alternative": {
            "verified_grounding_in_spec_like_path": len(spec_ref),
            "overlap_with_bar_noun": sum(1 for r in spec_ref
                                         if (r["doc"], r["line"], r["token"]) in bar_keys),
            "rows": spec_ref,
        },
        "corpus_status_counts": dict(corpus_counts),
        "adjudication_frame": {
            "definition": "every extract_numbers token on a papers/**/*.md line carrying a "
                          "comparison-operator field, value position or not",
            "n": len(frame),
            "in_value_position": sum(f["in_value_position"] for f in frame),
            "rescued_by_shipped_is_spec": sum(f["rescued_by_shipped_is_spec"] for f in frame),
            "roster": frame,
        },
        "frozen_adjudication_sample": sample,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n-> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
