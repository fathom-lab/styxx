"""Sort the mutation study's misses by CAUSE, because "58% detection" without this is misleading.

A bare detection rate invites one reading: the undetected fraction is a matter of running longer.
For this harness that is false, and the difference matters more than the rate.

The causes. Four were defined before the misses were assigned to them; EQUIVALENT was
added during the assignment, when one miss turned out to be provably unkillable rather than merely
unreached, and it is marked as the late addition it is:

  APERTURE_PAYLOAD  the mutation needs an input the generator's fixed literal lists cannot produce
                    — a receipt payload, a receipt-id form, or a manifest constant. `_manifest()`
                    draws payloads from ten byte-string literals and stamps every timestamp with
                    the same constant, so anything outside those ten is unreachable at any case
                    count. FIXABLE by strengthening the generator.

  APERTURE_DOCUMENT the mutation needs a document shape the grammar does not compose — a span of
                    exactly the cap length, an astral character inside a tag opener, a CR inside a
                    span. FIXABLE by strengthening the generator.

  OUT_OF_SURFACE    the mutation is in code the differential never runs, or whose effect never
                    reaches the compared verdict core. The JavaScript side has no repository at all
                    (SPEC B1 answers `no_repository`), so the whole tree-handle layer is compared
                    against nothing; the sidecar and receipt layers sit outside `verify`'s core
                    entirely. NOT fixable by fuzzing — it needs a different comparison.

  EQUIVALENT        an equivalent mutant: a syntactic change that provably alters no behaviour on
                    any input, so no test could ever kill it. Classic in mutation testing and the
                    reason a raw mutation score is never the whole story. It is NOT excluded from
                    the denominator here — the spec was frozen without this category and removing
                    it afterwards would be rescoring the run — so it is named instead, with the
                    proof, and the rate is left as specified.

  UNREACHABLE       no input could catch it. `WITHHELD` is declared in the verdict vocabulary and
                    emitted by no code path in either implementation, so a mutation on it is a
                    mutation on dead code; the sha256 length field's high word only matters for a
                    message of 2^32 bits. NOT a hole in the harness at all.

The first two are the harness's aperture and shrink when the generator grows. The third is its
SCOPE, and it is the finding that matters: "the two implementations agree" was always a statement
about a subset of each implementation, and until now nobody had said which subset.

Assignments are authored, not derived — they are a reading of each mutation's own `why`, which its
proposer wrote before the run and without seeing any coverage. They are published per miss so a
reader can dispute any one of them rather than take the summary.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

CAUSE = {
    # --- the payload / receipt-id / manifest-constant aperture -------------------------------
    "half-even becomes half-up in the JS BigInt rounder": (
        "APERTURE_PAYLOAD", "needs a value sitting exactly on a rounding tie at the printed "
        "precision; none of the ten payload literals produces one"),
    "Python quantize rounds half-up instead of half-even": (
        "APERTURE_PAYLOAD", "the mirror of the above, on the other side, and blind for the same "
        "reason"),
    "the scientific-notation threshold in Dec.toString moves by one decade": (
        "APERTURE_PAYLOAD", "needs a leaf with adjusted exponent -5; the payloads carry 1e5 and "
        "0.4211, nothing that small"),
    "the adjusted-exponent magnitude gate moves by one": (
        "APERTURE_PAYLOAD", "differs only at an adjusted exponent of exactly 320"),
    "prereg: digests are no longer case-folded": (
        "APERTURE_PAYLOAD", "every prereg: receipt the grammar emits is already lowercase"),
    "array index bound in pointer walking is off by one": (
        "APERTURE_PAYLOAD", "needs a pointer at an array's last index; the receipt-id list has no "
        "such pointer"),
    "the JSON reader stops refusing a BOM": (
        "APERTURE_PAYLOAD", "no payload literal begins with EF BB BF — the miss that started this "
        "study"),
    "strict JSON reads NaN/Infinity as float rather than Decimal": (
        "APERTURE_PAYLOAD", "no payload literal contains either token"),
    "JCS emits \n instead of the short \\n escape": (
        "APERTURE_PAYLOAD", "needs a newline inside a JCS-serialised manifest string; the manifest "
        "strings are drawn from three constants, none with a newline"),
    "the JS manifest stops case-folding authored_sha256": (
        "APERTURE_PAYLOAD", "authored_sha256 is always emitted lowercase by the generator"),
    "manifest core() emits authored_sha256 unsorted": (
        "APERTURE_PAYLOAD", "the list is always empty or a single element, so order is unobservable"),
    "the lone-surrogate refusal on a pointer leaf is neutered": (
        "APERTURE_PAYLOAD", "needs a surrogate escape in a payload; no literal contains a \\u "
        "escape at all"),
    "JSON Pointer unescaping applies ~0 before ~1 (RFC 6901 order reversed)": (
        "APERTURE_PAYLOAD", "needs a key containing ~ or /; no payload literal has one"),
    "occurrences counts overlapping matches": (
        "APERTURE_PAYLOAD", "needs a needle that overlaps itself in the haystack, e.g. aa in aaa"),
    "line slice starting past line 1 swallows the previous line's LF": (
        "APERTURE_PAYLOAD", "the receipt-id list carries #L1, #L1-L3 and #L9 — no slice starting "
        "at line 2"),

    # --- the document-shape aperture ----------------------------------------------------------
    "the span code-point cap becomes exclusive (300 now fails)": (
        "APERTURE_DOCUMENT", "only a span of exactly 300 code points changes verdict; the grammar "
        "builds over-cap spans by repeating inner text 20-60 times and lands nowhere near it"),
    "the opener's end offset is measured in UTF-16 units, not bytes": (
        "APERTURE_DOCUMENT", "needs an astral character inside a tag opener"),
    "CR stops counting as span whitespace": (
        "APERTURE_DOCUMENT", "needs a bare CR inside a span's inner text"),
    "the UTF-8 validator stops rejecting encoded surrogates": (
        "APERTURE_DOCUMENT", "needs CESU-8 style bytes; the invalid-UTF-8 documents the grammar "
        "makes are not of that shape"),
    "the three-byte overlong bound drops from U+0800 to U+0080": (
        "APERTURE_DOCUMENT", "needs a specific overlong encoding the grammar does not emit"),

    # --- outside the compared surface ---------------------------------------------------------
    "the committed-provenance rung bucket is renamed": (
        "OUT_OF_SURFACE", "provenance requires a repository tree; the JavaScript side has none, so "
        "nothing cross-checks it"),
    "SnapshotTree treats a symlink as a blob": (
        "OUT_OF_SURFACE", "the tree-handle layer is never exercised by the differential"),
    "verify falls back to the tree handle's own commit": (
        "OUT_OF_SURFACE", "same layer, same reason"),
    "render sorts its events, losing closer-before-opener adjacency": (
        "OUT_OF_SURFACE", "the sidecar layer is not on the path the differential compares"),
    "load_sidecar admits a zero-length span": (
        "OUT_OF_SURFACE", "same layer"),
    "the receipt digest covers coverage again (R9 undone)": (
        "OUT_OF_SURFACE", "the receipt layer sits outside the verdict core the digest covers"),
    "canonical end offset read at the closer's far edge": (
        "EQUIVALENT", "PROVED equivalent, not merely unobserved. The closer's whole range is cut "
        "from the canonical text: the builder does boundaries.set(a, a - removed), then "
        "removed += b - a, then boundaries.set(b, b - removed). So boundaries.get(closer_at) and "
        "boundaries.get(closer_end) both equal closer_at - removed_before, for every document. No "
        "input distinguishes this mutant, so its miss is not a hole in the harness"),

    # --- unreachable by any input -------------------------------------------------------------
    "sworn_total stops counting the WITHHELD verdict": (
        "UNREACHABLE", "WITHHELD is declared in the vocabulary and emitted by no code path in "
        "either implementation — this is a mutation on dead code"),
    "sha256 drops the high word of the 64-bit length field": (
        "UNREACHABLE", "the high word only matters for a message of 2^32 bits, about 512 MB"),
}

FIXABLE = {"APERTURE_PAYLOAD", "APERTURE_DOCUMENT"}


def main(argv=None):
    receipt = Path(argv[0]) if argv else ROOT / "conformance/sworn/mutation_coverage_2.json"
    R = json.loads(receipt.read_text(encoding="utf-8"))
    misses = [m for m in R["mutations"] if m["verdict"] == "missed" and not m.get("control")]

    unassigned = [m["name"] for m in misses if m["name"] not in CAUSE]
    extra = [k for k in CAUSE if k not in {m["name"] for m in misses}]

    rows, counts = [], Counter()
    for m in misses:
        cause, why = CAUSE.get(m["name"], ("UNASSIGNED", ""))
        counts[cause] += 1
        rows.append({"name": m["name"], "side": m["side"], "region": m.get("region"),
                     "cause": cause, "because": why})

    fixable = sum(v for k, v in counts.items() if k in FIXABLE)

    # How close was each CATCH to being a miss? A mutation caught by one case in five thousand is
    # caught, but only just: at a smaller guard it would have been invisible. The guard's size is a
    # choice and this is the evidence for it, so it is recorded rather than left to be re-derived.
    caught = [m for m in R["mutations"] if m["verdict"] == "caught" and not m.get("control")]
    marg = Counter()
    for m in caught:
        d = m.get("disagreements", 0)
        marg["exactly_1" if d == 1 else
             "from_2_to_5" if d <= 5 else
             "from_6_to_25" if d <= 25 else
             "from_26_to_250" if d <= 250 else "over_250"] += 1
    thin = sorted(({"name": m["name"], "side": m["side"], "region": m.get("region"),
                    "disagreements": m["disagreements"]}
                   for m in caught if m.get("disagreements", 0) <= 5),
                  key=lambda r: r["disagreements"])
    out = {
        "schema": "styxx.sworn.mutation-miss-taxonomy/v1",
        "receipt": str(receipt.relative_to(ROOT)).replace("\\", "/"),
        "definitions": {
            "APERTURE_PAYLOAD": "needs an input the generator's fixed literal lists cannot produce",
            "APERTURE_DOCUMENT": "needs a document shape the grammar does not compose",
            "OUT_OF_SURFACE": "in code the differential never runs, or whose effect never reaches "
                              "the compared verdict core",
            "UNREACHABLE": "no input could catch it: dead code, or a bound no practical input "
                           "reaches",
            "EQUIVALENT": "an equivalent mutant - proved to change no behaviour on any input, so "
                          "nothing could ever kill it; it stays in the denominator because the "
                          "spec was frozen without this category, and is named here instead",
        },
        "authored": ("assignments are a reading of each mutation's own `why`, written by its "
                     "proposer before the run and without sight of any coverage; they are "
                     "published per miss so a reader can dispute any one of them"),
        "counts": dict(counts),
        "misses_total": len(misses),
        "fixable_by_a_stronger_generator": fixable,
        "not_fixable_by_fuzzing": len(misses) - fixable,
        "misses": rows,
        "marginality_of_catches": {
            "note": ("disagreements produced by each caught mutation, out of %d cases. A catch at "
                     "the low end is a catch the guard nearly missed." % R["cases"]),
            "buckets": dict(marg),
            "caught_by_five_or_fewer": thin,
        },
        "unassigned": unassigned,
        "assignments_with_no_matching_miss": extra,
    }
    dest = ROOT / "papers" / "sworn" / "mutation_miss_taxonomy.json"
    dest.write_bytes((json.dumps(out, indent=1, sort_keys=True, ensure_ascii=False)
                      + "\n").encode("utf-8"))
    print("misses %d -> %s" % (len(misses), json.dumps(dict(counts))))
    print("fixable by a stronger generator: %d" % fixable)
    print("not fixable by fuzzing:          %d" % (len(misses) - fixable))
    if unassigned:
        print("UNASSIGNED (%d):" % len(unassigned))
        for n in unassigned:
            print("   -", n)
    if extra:
        print("assignments with no matching miss (%d):" % len(extra))
        for n in extra:
            print("   -", n)
    print("wrote", dest.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
