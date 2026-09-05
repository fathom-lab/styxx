# -*- coding: utf-8 -*-
"""Generate the sworn conformance set from the two sworn test files.

Built to ``papers/sworn/SPEC_sworn_conformance_vectors_v01_2026_09_05.md``. Runs the sources under
``conformance/sworn/recorder.py``, folds every recorded call into a content-addressed vector,
replays every vector through ``styxx.sworn`` (C6), refuses a moved core or a nondeterministic id
(C6), and writes the set under one digest (C7, C9). Label: the precondition for any second
verifier; no claim.

  python conformance/sworn/gen_vectors.py            regenerate in place; refuses a moved core
  python conformance/sworn/gen_vectors.py --check    regenerate in memory; exit 1 if set_sha256 differs
  python conformance/sworn/gen_vectors.py --replay   replay the committed set; exit 1 on any mismatch

Exit 0 is a set written or confirmed; exit 1 is a refusal with the reason printed. Nothing is
written on a refusal.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from styxx import sworn  # noqa: E402
from styxx._version import __version__  # noqa: E402
from conformance.sworn import CLOCK  # noqa: E402
from conformance.sworn.replay import replay_vector  # noqa: E402

SET_DIR = ROOT / "conformance" / "sworn"
SPEC_PATH = "papers/sworn/SPEC_sworn_conformance_vectors_v01_2026_09_05.md"
LABEL = "the precondition for any second verifier; no claim"
SCHEMA = "styxx.sworn.conformance/v1"
FAMILY_SCHEMA = "styxx.sworn.conformance.family/v1"
OBSERVER_SCHEMA = "styxx.sworn.conformance.observer/v1"
SOURCES = ["tests/test_sworn.py", "tests/test_sworn_attacks.py"]
SOURCES_NOTE = ("widened from the plan row's tests/test_sworn.py: R1's positive case, R5 at L2, R6's L2, "
                "rung_unknown and manifest/0.1 cases and R9's v1 re-derivation live only in "
                "tests/test_sworn_attacks.py::TestRules (SPEC, 'Why this exists')")
CORE_KEYS = ["schema", "format", "document", "commit", "manifest_digest", "spans", "counts", "sworn_total",
             "unresolved", "document_verdict", "document_malformed", "rungs", "certifies"]
CORE_RULE = "core_sha256 = sha256(utf8(jcs(core))); core = verify() output minus verifier minus coverage"
ID_RULE = "id = sha256(utf8(jcs({\"mode\": mode, \"inputs\": inputs})))"
SET_RULE = "set_sha256 = sha256(utf8(jcs(index minus set_sha256 minus provenance)))"
BLOB_RULE = "blobs.json maps sha256 -> base64; every blob hashes to its key; a JSON blob is UTF-8 text, indent 1, LF"

INPUT_KEYS = {
    "inline": ("name", "commit", "document", "manifest", "tree"),
    "sidecar": ("name", "commit", "sidecar", "manifest", "tree"),
    "canon": ("name", "commit", "document", "manifest"),
    "load": ("sidecar",),
    "manifest": ("manifest",),
    "receipt_check": ("receipt", "document", "sidecar", "manifest", "tree"),
}

FUZZ_TEST = "test_round_trip_is_asserted_over_a_seeded_fuzz_corpus"
FAMILY_BY_CLASS = {
    "TestLexer": "lexer", "TestCanonical": "canonical", "TestReceipts": "receipts", "TestGitTree": "tree",
    "TestNumeric": "numeric", "TestQuoteHashAbsent": "quote_hash_absent", "TestDocument": "document",
    "TestCoverage": "coverage", "TestVerdictReceipt": "receipt_v1", "TestInvariants": "invariants",
    "TestCLI": "cli", "TestDoctrine": "doctrine", "TestWorkedExamplesOnTheStruct1Receipt": "worked_examples",
    "TestSidecarHardening": "sidecar_hardening", "TestReceiptHardening": "receipt_hardening",
    "TestGamingLens": "gaming", "TestGamingLensFromTheAttackPass": "gaming",
    "TestRepaired": "attacks", "TestNotRepairedAndSaidSo": "attacks",
    "TestTheDenominatorWasTheWrongIdiom": "coverage", "TestRules": "rules", "TestSnapshotTree": "snapshot",
}

# The v0.2 rules (R1-R9) and the battery rows (A1-A12) each source test pins. A vector's rules is
# the union over its sources; a test absent here pins no rule by name.
RULES_BY_TEST = {
    "test_r1_a_manifest_receipt_takes_a_pointer_and_a_line_anchor": ["R1"],
    "test_a_receipt_outside_the_three_forms_is_malformed_not_narrative": ["R1"],
    "test_hash_over_a_partial_receipt_is_malformed": ["R1"],
    "test_absent_over_a_pointer_or_anchor_is_partial_and_malformed": ["R1"],
    "test_a_tag_inside_an_html_comment_is_a_hidden_commitment_v02_r2": ["R2"],
    "test_an_unclosed_tag_inside_a_comment_is_unclosed_and_refuses_the_sidecar": ["R2"],
    "test_a4_a_tag_hidden_in_an_html_comment_is_malformed_not_held": ["R2", "A4"],
    "test_a_short_quote_needle_over_a_whole_receipt_is_malformed_v02_r3": ["R3"],
    "test_a10_a_one_byte_needle_over_a_whole_receipt_is_malformed": ["R3", "A10"],
    "test_a10_a_short_needle_the_author_narrowed_is_exempt": ["R3", "A10"],
    "test_a_short_trivial_quote_is_malformed_not_held": ["R3"],
    "test_the_cap_is_three_hundred_code_points_never_bytes_v02_r4": ["R4"],
    "test_the_cap_no_longer_penalises_three_byte_scripts": ["R4"],
    "test_a_newline_inside_a_span_is_allowed_and_counts_toward_the_cap": ["R4"],
    "test_r5_attestation_bytes_are_a_receipt_and_no_signature_is_checked": ["R5"],
    "test_every_external_source_kind_resolves": ["R5"],
    "test_every_author_side_source_kind_is_malformed": ["R5"],
    "test_a_kind_of_source_outside_the_closed_vocabulary_is_malformed": ["R5"],
    "test_r6_rungs_l1_and_l2_resolve_l3_and_nonsense_are_unresolved_never_accusations": ["R6"],
    "test_r6_a_manifest_0_1_still_loads_and_never_reaches_l2": ["R6"],
    "test_a5_a_committed_receipt_holds_and_prints_that_its_authorship_was_not_checked": ["R7", "A5"],
    "test_a6_a_prereg_digest_proves_content_never_precedence": ["R7", "A6"],
    "test_a7_a_manifest_the_agent_minted_holds_and_the_rung_it_declares_is_printed": ["R6", "R7", "A7"],
    "test_coverage_is_always_advisory_and_the_estimate_is_gone_v02_r8": ["R8"],
    "test_the_floor_counts_every_narrative_sentence_and_the_diff_count_is_labelled": ["R8"],
    "test_fenced_code_is_not_counted_in_the_denominator": ["R8"],
    "test_coverage_of_an_unsworn_document_is_a_zero_floor_not_undefined": ["R8"],
    "test_a_result_shaped_document_no_longer_prints_a_near_one_coverage": ["R8"],
    "test_swearing_only_trivia_prints_its_coverage_beside_the_verdict": ["R8"],
    "test_a2_trivial_swearing_holds_and_the_floor_makes_the_padding_visible": ["R8", "A2"],
    "test_a3_the_stative_dodge_no_longer_shrinks_the_floor": ["R8", "A3"],
    "test_r9_the_v1_receipt_re_derives_without_its_coverage_block": ["R9"],
    "test_r9_a_committed_v0_receipt_still_checks_on_its_core": ["R9"],
    "test_the_receipt_is_content_addressed_and_re_derivable": ["R9"],
    "test_a_tampered_verdict_or_document_fails_re_derivation": ["R9"],
    "test_a_receipt_from_another_verifier_build_is_reported_not_hidden": ["R9"],
    "test_a_tampered_receipt_fails_and_never_crashes_or_refuses": ["R9"],
    "test_canon_render_verify_check_round_trip": ["R9"],
    "test_a1_the_rider_clause_holds_on_the_number_and_says_nothing_about_the_qualifier": ["A1"],
    "test_receipt_shopping_moves_oath_but_cannot_move_sworn": ["A8"],
    "test_a9_percent_and_fraction_do_not_coincide_survives_from_v01": ["A9"],
    "test_a11_post_hoc_tagging_is_undetectable_and_the_receipt_says_so": ["A11"],
    "test_a12_a_coincident_value_at_the_wrong_leaf_holds": ["A12"],
}

# One sentence per rule saying what a positive and a negative vector look like; the predicates
# are in tests/test_sworn_conformance.py, one per rule, and both shapes must be present there.
RULE_CONTRACT = {
    "R1": {"positive": "a core vector with a span whose receipt is rN#... and whose verdict is HELD",
           "negative": "a span whose receipt is rN#... and whose verdict is MALFORMED with reason receipt_form, absent_over_partial or hash_over_partial"},
    "R2": {"positive": "a vector tagged R2 with a span that is not hidden_commitment (the visible tag keeps its verdict)",
           "negative": "a span MALFORMED hidden_commitment"},
    "R3": {"positive": "a quote span against a fragment receipt (rN#... or path:...#...) that is HELD",
           "negative": "a span MALFORMED short_needle"},
    "R4": {"positive": "a vector tagged R4 with a span that is not length_cap (300 code points pass, whatever their byte length)",
           "negative": "a span MALFORMED length_cap"},
    "R5": {"positive": "a HELD span whose provenance kind_of_source is attestation",
           "negative": "a span MALFORMED kind_of_source_unknown or receipt_author_minted"},
    "R6": {"positive": "a core whose rungs count L1 or L2",
           "negative": "a span UNRESOLVED rung_unknown"},
    "R7": {"positive": "a span whose provenance form is rn, path or prereg",
           "negative": "a core whose rungs count unresolved"},
    "R8": {"positive": "a floor whose sentence_share is a number",
           "negative": "a floor whose sentence_share is null (nothing sworn over no sentences)"},
    "R9": {"positive": "a check vector with status VERIFIED",
           "negative": "a check vector with status FAILED"},
    "A1": {"shows": "a HELD numeric span (the rider clause holds on the number)"},
    "A2": {"shows": "SWORN-HELD with narrative sentences counted and a sentence_share under 1"},
    "A3": {"shows": "SWORN-HELD with narrative sentences counted and a sentence_share under 1"},
    "A4": {"shows": "MALFORMED hidden_commitment and document SWORN-FAILED"},
    "A5": {"shows": "a HELD span whose provenance form is path"},
    "A6": {"shows": "a HELD span whose provenance form is prereg"},
    "A7": {"shows": "a HELD span whose provenance rung is undeclared"},
    "A8": {"shows": "SWORN-FAILED with a path receipt FAILED value_mismatch (a larger pool offers nothing)"},
    "A9": {"shows": "FAILED value_mismatch"},
    "A10": {"shows": "MALFORMED short_needle"},
    "A11": {"shows": "a HELD span (the certifies sentence is inside the core)"},
    "A12": {"shows": "a HELD span (the author named the leaf)"},
}

# One row per SystemExit site in styxx/sworn.py: a code a second verifier is held to, and the
# substring this verifier's message must contain. A message no row matches refuses generation.
REFUSALS = [
    ("document_malformed", "no canonical text exists"),
    ("commit_form", "commit must be a full lowercase hex object id or None"),
    ("lexical_malformed", "cannot carry a lexically MALFORMED declaration"),
    ("round_trip", "canonical round trip does not reproduce the document bytes"),
    ("sidecar_spec_unknown", "unknown sidecar spec"),
    ("sidecar_keys", "sidecar keys"),
    ("sidecar_commit_form", "sidecar commit must be a full lowercase hex object id or null"),
    ("sidecar_text_not_string", "sidecar text must be a string"),
    ("sidecar_text_not_utf8", "sidecar text is not encodable as UTF-8"),
    ("sidecar_document_shape", "sidecar document must carry a string name and a string sha256"),
    ("sidecar_text_digest", "sidecar text does not hash to document.sha256"),
    ("sidecar_spans_not_list", "sidecar spans must be a list"),
    ("sidecar_span_shape", "is not an object with exactly start/end/receipt/kind"),
    ("sidecar_span_range", "are not a non-empty range in the text"),
    ("sidecar_spans_overlap", "spans are not ordered and non-overlapping"),
    ("sidecar_span_boundary", "is not on a UTF-8 character boundary"),
    ("sidecar_span_attribute", "cannot be carried by the inline tag grammar"),
    ("sidecar_manifest_shape", "sidecar manifest is not a"),
    ("manifest_spec", "unknown manifest spec"),
    ("manifest_receipts_shape", "manifest receipts must be an object keyed by receipt id"),
    ("manifest_authored_shape", "manifest authored_sha256 must be a list of hex strings"),
    ("sidecar_commit_disagrees", "and --commit says"),
    ("manifest_disagrees", "the supplied manifest disagrees with the embedded one"),
    ("nothing_to_verify", "nothing to verify"),
    ("sidecar_no_canonical_form", "the sidecar text has no canonical form"),
    ("sidecar_declaration_unrepresentable", "carries a declaration the sidecar form cannot represent"),
    ("sidecar_spans_not_reproduced", "does not reproduce its spans"),
]

REQUIRES_LEGEND = {
    "manifest": "a sworn/manifest/0.x object was given; a second verifier for rN and embedded blobs must pass these",
    "tree": "a tree snapshot was given (SnapshotTree semantics, C10); a second verifier may skip these and must report how many it skipped, per family",
    "git": "only a live git object store reproduces this; no vector in v0.1 carries it",
    "observer": "styxx.claimdetect; never required and never pinned (C4)",
}


class Refused(Exception):
    pass


# --------------------------------------------------------------------------- bytes and hashes

def _sha256(b: bytes) -> str:
    return sworn._sha256(b)


def _json_bytes(obj: Any) -> bytes:
    """Exactly what styxx.sworn._write_json_lf writes: indent 1, UTF-8, LF, trailing LF."""
    return (json.dumps(obj, indent=1, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _normalised_sha256(path: Path) -> str:
    return _sha256(path.read_bytes().replace(b"\r\n", b"\n"))


def _put(blobs: Dict[str, bytes], data: bytes) -> str:
    h = _sha256(data)
    blobs[h] = data
    return h


def _rule_key(r: str) -> Tuple[str, int]:
    return (r[0], int(r[1:]))


# --------------------------------------------------------------------------- the recorder run

def run_recorder(sources: List[str]) -> Tuple[List[dict], str]:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "records.jsonl"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["SWORN_RECORDER_OUT"] = str(out)
        env["PYTHONIOENCODING"] = "utf-8"
        cmd = [sys.executable, "-m", "pytest", *sources, "-q", "--no-header", "-p", "no:cacheprovider",
               "-p", "conformance.sworn.recorder"]
        r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        # the counts, not the wall time: provenance must not change between two stable runs
        summary = re.sub(r"\s+in\s+[0-9.]+s.*$", "", (r.stdout.strip().splitlines() or ["(no output)"])[-1])
        if r.returncode != 0:
            sys.stdout.write(r.stdout[-4000:] + "\n" + r.stderr[-2000:] + "\n")
            raise Refused("the sources did not pass under the recorder: %s" % summary)
        if not out.exists():
            raise Refused("the recorder wrote nothing")
        records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    return records, summary


# --------------------------------------------------------------------------- folding records

def _refusal_code(message: str) -> Tuple[str, str]:
    hits = [(code, match) for code, match in REFUSALS if match in message]
    if len(hits) != 1:
        raise Refused("refusal message matches %d table rows, not one: %r" % (len(hits), message[:200]))
    return hits[0]


def _tree_ref(t: dict, blobs: Dict[str, bytes]) -> dict:
    entries = {}
    for path in sorted(t["entries"]):
        e = t["entries"][path]
        sha = None
        if e.get("bytes") is not None:
            sha = _put(blobs, base64.b64decode(e["bytes"]))
        entries[path] = {"mode": e["mode"], "size": e.get("size"), "sha256": sha}
    return {"snapshot_commit": t["snapshot_commit"], "handle_commit": t["handle_commit"], "entries": entries}


def _object_blob(obj: Any, blobs: Dict[str, bytes], label: str) -> str:
    try:
        return _put(blobs, _json_bytes(obj))
    except (TypeError, ValueError) as e:
        raise Refused("%s is not representable as a JSON blob: %s"
                      % (label, _why_unrepresentable(e)))


def make_vector(rec: dict, blobs: Dict[str, bytes]) -> Tuple[dict, Optional[dict]]:
    mode = rec["mode"]
    inp = rec["inputs"]
    out = rec["outcome"]
    inputs: Dict[str, Any] = {}
    for key in INPUT_KEYS[mode]:
        v = inp.get(key)
        if v is None:
            inputs[key] = None
        elif key == "document":
            inputs[key] = _put(blobs, base64.b64decode(v))
        elif key in ("sidecar", "receipt", "manifest"):
            inputs[key] = _object_blob(v, blobs, key)
        elif key == "tree":
            inputs[key] = _tree_ref(v, blobs)
        else:
            inputs[key] = v
    vid = _sha256(sworn._jcs({"mode": mode, "inputs": inputs}).encode("utf-8"))
    observer = None
    kind = out["outcome"]
    if kind == "core":
        core = out["core"]
        text = out["core_jcs"].encode("utf-8")
        core_sha = _put(blobs, text)
        expect = {
            "outcome": "core", "core_sha256": core_sha,
            "document_verdict": core["document_verdict"], "counts": core["counts"],
            "sworn_total": core["sworn_total"], "unresolved": core["unresolved"], "rungs": core["rungs"],
            "spans": [{"at": s["at"], "receipt": s["receipt"], "kind": s["kind"], "verdict": s["verdict"],
                       "reason": s["reason"]} for s in core["spans"]],
            "floor": out["floor"],
        }
        observer = out["observer"]
    elif kind == "refused":
        code, match = _refusal_code(out["message"])
        expect = {"outcome": "refused", "refusal": {"where": out["where"], "code": code, "match": match}}
    elif kind == "sidecar":
        expect = {"outcome": "sidecar",
                  "sidecar_sha256": _sha256(out["sidecar_jcs"].encode("utf-8")),
                  "sidecar": _object_blob(out["sidecar"], blobs, "sidecar")}
    elif kind == "accepted":
        expect = {"outcome": "accepted"}
    elif kind == "manifest":
        expect = {"outcome": "manifest", "manifest": out["manifest"]}
    elif kind == "check":
        expect = {"outcome": "check", "check": out["check"]}
        observer = out["observer"]
    else:
        raise Refused("unknown outcome %r" % kind)
    requires = []
    if inputs.get("manifest") is not None:
        requires.append("manifest")
    if inputs.get("tree") is not None:
        requires.append("tree")
    vec = {"id": vid, "family": None, "sources": [rec["source"]], "rules": [], "mode": mode,
           "requires": requires, "inputs": inputs, "expect": expect}
    return vec, observer


def _test_name(source: str) -> str:
    return source.split("::")[-1].split("[")[0]


def family_of(sources: List[str]) -> str:
    src = min(sources)
    parts = src.split("::")
    test = _test_name(src)
    if test == FUZZ_TEST:
        return "fuzz"
    cls = parts[1] if len(parts) >= 3 else None
    return FAMILY_BY_CLASS.get(cls, "other")


def fold(records: List[dict]):
    blobs: Dict[str, bytes] = {}
    vectors: Dict[str, dict] = {}
    observer: Dict[str, dict] = {}
    skipped: List[dict] = []
    for rec in records:
        if "skip" in rec:
            skipped.append({"source": rec["source"], "where": rec["where"], "why": rec["skip"]})
            continue
        try:
            vec, obs = make_vector(rec, blobs)
        except Refused as e:
            if str(e).startswith("refusal message"):
                raise
            skipped.append({"source": rec["source"], "where": rec["where"], "why": str(e)})
            continue
        vid = vec["id"]
        if vid in vectors:
            old = vectors[vid]
            if old["expect"] != vec["expect"]:
                raise Refused("nondeterminism: id %s observed with two outcomes\n  %s\n  %s"
                              % (vid, sorted(set(old["sources"])), vec["sources"]))
            if rec["source"] not in old["sources"]:
                old["sources"].append(rec["source"])
        else:
            vectors[vid] = vec
            if obs is not None:
                observer[vid] = obs
    for vec in vectors.values():
        vec["sources"] = sorted(set(vec["sources"]))
        rules = set()
        for s in vec["sources"]:
            rules.update(RULES_BY_TEST.get(_test_name(s), []))
        vec["rules"] = sorted(rules, key=_rule_key)
        vec["family"] = family_of(vec["sources"])
    seen = set()
    uniq = []
    for s in skipped:
        key = (s["source"], s["where"], s["why"])
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    uniq.sort(key=lambda s: (s["source"], s["where"], s["why"]))
    return vectors, blobs, observer, uniq


# --------------------------------------------------------------------------- the set

def _reasons_seen(vectors: Dict[str, dict], blobs: Dict[str, bytes]):
    reasons, verdicts = set(), set()
    for v in vectors.values():
        if v["expect"]["outcome"] != "core":
            continue
        for s in v["expect"]["spans"]:
            verdicts.add(s["verdict"])
            if s["reason"] is not None:
                reasons.add(s["reason"])
        core = json.loads(blobs[v["expect"]["core_sha256"]].decode("utf-8"))
        dm = core.get("document_malformed")
        if isinstance(dm, dict) and dm.get("reason"):
            reasons.add(dm["reason"])
    return reasons, verdicts


def build_set(vectors: Dict[str, dict], blobs: Dict[str, bytes], observer: Dict[str, dict],
              skipped: List[dict], summary: str):
    families: Dict[str, List[dict]] = {}
    for v in vectors.values():
        families.setdefault(v["family"], []).append(v)
    family_files: Dict[str, bytes] = {}
    family_meta: Dict[str, dict] = {}
    for name in sorted(families):
        vs = sorted(families[name], key=lambda v: v["id"])
        data = _json_bytes({"schema": FAMILY_SCHEMA, "family": name, "count": len(vs), "vectors": vs})
        family_files[name] = data
        family_meta[name] = {"file": "vectors/%s.json" % name, "sha256": _sha256(data), "count": len(vs)}
    blob_store = {h: base64.b64encode(blobs[h]).decode("ascii") for h in sorted(blobs)}
    blobs_bytes = _json_bytes(blob_store)
    reasons, verdicts = _reasons_seen(vectors, blobs)
    where_by_code: Dict[str, set] = {}
    for v in vectors.values():
        if v["expect"]["outcome"] == "refused":
            r = v["expect"]["refusal"]
            where_by_code.setdefault(r["code"], set()).add(r["where"])
    refusal_codes = {code: {"match": match, "where": sorted(where_by_code.get(code, ()))} for code, match in REFUSALS}
    modes: Dict[str, int] = {}
    outcomes: Dict[str, int] = {}
    requires: Dict[str, int] = {}
    for v in vectors.values():
        modes[v["mode"]] = modes.get(v["mode"], 0) + 1
        outcomes[v["expect"]["outcome"]] = outcomes.get(v["expect"]["outcome"], 0) + 1
        key = "+".join(v["requires"]) or "none"
        requires[key] = requires.get(key, 0) + 1
    index: Dict[str, Any] = {
        "schema": SCHEMA,
        "format": sworn.SPEC,
        "spec": SPEC_PATH,
        "label": LABEL,
        "clock": CLOCK,
        "manifest_specs": list(sworn.MANIFEST_SPECS),
        "receipt_schemas": list(sworn.RECEIPT_SCHEMAS),
        "core_definition": CORE_KEYS,
        "core_rule": CORE_RULE,
        "id_rule": ID_RULE,
        "set_rule": SET_RULE,
        "blob_rule": BLOB_RULE,
        "sources": list(SOURCES),
        "sources_note": SOURCES_NOTE,
        "vector_count": len(vectors),
        "family_count": len(family_meta),
        "families": family_meta,
        "blobs": {"file": "blobs.json", "sha256": _sha256(blobs_bytes), "count": len(blob_store)},
        "observer": {"file": "observer.json", "note": "outside set_sha256: the observer's and the build's numbers (C4)"},
        "modes": {k: modes[k] for k in sorted(modes)},
        "outcomes": {k: outcomes[k] for k in sorted(outcomes)},
        "requires": {k: requires[k] for k in sorted(requires)},
        "requires_legend": REQUIRES_LEGEND,
        "refusal_codes": refusal_codes,
        "rule_contract": RULE_CONTRACT,
        "unvectored": {
            "reasons": sorted(set(sworn.REASONS) - reasons),
            "verdicts": sorted(set(sworn.VERDICTS) - verdicts),
            "refusal_codes": sorted(code for code, match in REFUSALS if code not in where_by_code),
            "skipped_count": len(skipped),
            "skipped": skipped,
        },
    }
    digest_body = {k: v for k, v in index.items() if k not in ("set_sha256", "provenance")}
    index["provenance"] = {
        "note": "outside set_sha256: which bytes generated the set, not what the set pins (C7)",
        "sworn_sha256": _normalised_sha256(ROOT / "styxx" / "sworn.py"),
        "sources_sha256": {s: _normalised_sha256(ROOT / s) for s in SOURCES},
        "python": platform.python_version(),
        "platform": sys.platform,
        "styxx_version": __version__,
        "pytest_summary": summary,
    }
    index["set_sha256"] = _sha256(sworn._jcs(digest_body).encode("utf-8"))
    obs_doc = {"schema": OBSERVER_SCHEMA,
               "note": "per vector id: diff_claim_*, unsworn_claims count, claimdetect_version, and for checks "
                       "coverage_reproduces and same_verifier_build. Outside set_sha256 (C4).",
               "vectors": {k: observer[k] for k in sorted(observer)}}
    return index, family_files, blobs_bytes, obs_doc


def write_set(out_dir: Path, index: dict, family_files: Dict[str, bytes], blobs_bytes: bytes, obs_doc: dict) -> None:
    (out_dir / "vectors").mkdir(parents=True, exist_ok=True)
    for stale in (out_dir / "vectors").glob("*.json"):
        if stale.stem not in family_files:
            stale.unlink()
    for name, data in family_files.items():
        with open(out_dir / "vectors" / ("%s.json" % name), "wb") as f:
            f.write(data)
    with open(out_dir / "blobs.json", "wb") as f:
        f.write(blobs_bytes)
    with open(out_dir / "observer.json", "wb") as f:
        f.write(_json_bytes(obs_doc))
    with open(out_dir / "index.json", "wb") as f:
        f.write(_json_bytes(index))


def load_committed(set_dir: Path):
    index_path = set_dir / "index.json"
    if not index_path.exists():
        return None, {}, {}
    index = json.loads(index_path.read_bytes().decode("utf-8"))
    vectors: Dict[str, dict] = {}
    for name, meta in index["families"].items():
        fam = json.loads((set_dir / meta["file"]).read_bytes().decode("utf-8"))
        for v in fam["vectors"]:
            vectors[v["id"]] = v
    blobs = json.loads((set_dir / "blobs.json").read_bytes().decode("utf-8"))
    return index, vectors, blobs


def _pin(v: dict) -> tuple:
    e = v["expect"]
    return (e["outcome"], e.get("core_sha256"), (e.get("refusal") or {}).get("code"), e.get("sidecar_sha256"),
            (e.get("manifest") or {}).get("digest"), json.dumps(e.get("check"), sort_keys=True))


def refuse_if_moved(old: Dict[str, dict], new: Dict[str, dict]) -> None:
    moved = [vid for vid in old if vid in new and _pin(old[vid]) != _pin(new[vid])]
    dropped = [vid for vid in old if vid not in new]
    added = [vid for vid in new if vid not in old]
    for vid in sorted(dropped):
        print("dropped: %s  %s" % (vid, " ".join(old[vid]["sources"])))
    for vid in sorted(added):
        print("added:   %s  %s" % (vid, " ".join(new[vid]["sources"])))
    if moved:
        for vid in sorted(moved):
            print("MOVED:   %s  %s" % (vid, " ".join(old[vid]["sources"])))
            print("         was %s" % (_pin(old[vid]),))
            print("         now %s" % (_pin(new[vid]),))
        raise Refused("%d vector(s) moved: a moved core is a finding about the verifier, never a reason to "
                      "rewrite the set; nothing written" % len(moved))


def replay_all(vectors: Dict[str, dict], blob_store: Dict[str, str]) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {}
    failures = []
    for vid in sorted(vectors):
        v = vectors[vid]
        status, detail = replay_vector(v, blob_store)
        fam = counts.setdefault(v["family"], {"pass": 0, "fail": 0, "skip": 0})
        fam[status] += 1
        if status == "fail":
            failures.append((vid, v["sources"], detail))
    for vid, sources, detail in failures[:20]:
        print("REPLAY FAILED: %s  %s\n    %s" % (vid, " ".join(sources), detail))
    if failures:
        raise Refused("%d vector(s) do not replay through styxx.sworn" % len(failures))
    return counts


def generate(sources: List[str]):
    records, summary = run_recorder(sources)
    vectors, blobs, observer, skipped = fold(records)
    if not vectors:
        raise Refused("no vectors recorded")
    index, family_files, blobs_bytes, obs_doc = build_set(vectors, blobs, observer, skipped, summary)
    blob_store = {h: base64.b64encode(b).decode("ascii") for h, b in blobs.items()}
    counts = replay_all(vectors, blob_store)
    return index, family_files, blobs_bytes, obs_doc, vectors, counts


def _report(index: dict, counts: Dict[str, Dict[str, int]]) -> None:
    for name, meta in index["families"].items():
        c = counts.get(name, {})
        print("  %-18s %5d vectors  replay pass=%d fail=%d skip=%d" % (name, meta["count"], c.get("pass", 0), c.get("fail", 0), c.get("skip", 0)))
    print("  vectors=%d blobs=%d skipped=%d unvectored reasons=%s verdicts=%s"
          % (index["vector_count"], index["blobs"]["count"], len(index["unvectored"]["skipped"]),
             index["unvectored"]["reasons"], index["unvectored"]["verdicts"]))
    print("  set_sha256 %s" % index["set_sha256"])



def _why_unrepresentable(e):
    """The lab's word for why, never the interpreter's. A CPython exception message is not
    stable across versions, and this string is pinned by the set's digest: py3.9-3.11
    refused a set py3.12 accepted, on nothing but this prose."""
    if isinstance(e, UnicodeEncodeError):
        return "carries text that is not encodable as UTF-8 (a lone surrogate)"
    if isinstance(e, ValueError):
        return "carries a value no canonical serialisation can hold (NaN or an infinity)"
    if isinstance(e, TypeError):
        return "carries a value JSON has no type for"
    return "is not JSON-representable"


def _drift_detail(committed: dict, regenerated: dict) -> str:
    """Which part of the set moved. A digest that only says "different" cannot be acted on."""
    lines = ["  what moved:"]
    cf = committed.get("families") or {}
    rf = regenerated.get("families") or {}
    for name in sorted(set(cf) | set(rf)):
        c = (cf.get(name) or {}).get("sha256")
        r = (rf.get(name) or {}).get("sha256")
        if c != r:
            lines.append("    family %-20s committed %s  regenerated %s  (count %s -> %s)"
                         % (name, str(c)[:16], str(r)[:16],
                            (cf.get(name) or {}).get("count"), (rf.get(name) or {}).get("count")))
    cb = (committed.get("blobs") or {}).get("sha256")
    rb = (regenerated.get("blobs") or {}).get("sha256")
    if cb != rb:
        lines.append("    blobs.json           committed %s  regenerated %s  (count %s -> %s)"
                     % (str(cb)[:16], str(rb)[:16], (committed.get("blobs") or {}).get("count"),
                        (regenerated.get("blobs") or {}).get("count")))
    skip = {"set_sha256", "provenance", "families", "blobs"}
    for key in sorted(set(committed) | set(regenerated)):
        if key in skip:
            continue
        if committed.get(key) != regenerated.get(key):
            lines.append("    index.%-15s committed %s" % (key, _short(committed.get(key))))
            lines.append("    %-21s regenerated %s" % ("", _short(regenerated.get(key))))
    if len(lines) == 1:
        lines.append("    nothing named above differs — the drift is in the index's own shape")
    return "\n".join(lines)


def _short(v) -> str:
    t = json.dumps(v, sort_keys=True, default=str)
    return t if len(t) <= 220 else t[:200] + "…(%d chars)" % len(t)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sources", nargs="+", default=SOURCES)
    ap.add_argument("--out", default=str(SET_DIR))
    ap.add_argument("--check", action="store_true", help="regenerate in memory; exit 1 if set_sha256 differs")
    ap.add_argument("--replay", action="store_true", help="replay the committed set; exit 1 on any mismatch")
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    try:
        if args.replay:
            index, vectors, blob_store = load_committed(out_dir)
            if index is None:
                raise Refused("no committed set at %s" % out_dir)
            counts = replay_all(vectors, blob_store)
            _report(index, counts)
            return 0
        index, family_files, blobs_bytes, obs_doc, vectors, counts = generate(args.sources)
        committed, old_vectors, _ = load_committed(out_dir)
        if args.check:
            _report(index, counts)
            if committed is None:
                raise Refused("no committed set to check against at %s" % out_dir)
            if committed["set_sha256"] != index["set_sha256"]:
                raise Refused("set_sha256 drifted: committed %s, regenerated %s\n%s"
                              % (committed["set_sha256"], index["set_sha256"],
                                 _drift_detail(committed, index)))
            print("CHECK OK: the set regenerates to its own digest")
            return 0
        if committed is not None:
            refuse_if_moved(old_vectors, vectors)
        write_set(out_dir, index, family_files, blobs_bytes, obs_doc)
        _report(index, counts)
        print("written: %s" % out_dir)
        return 0
    except Refused as e:
        print("REFUSED: %s" % e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
