# -*- coding: utf-8 -*-
"""RECEIPT v0 (DRAFT, unsigned): tests of the projector, the schema and the decisions ledger.

The two tests that decide whether the step failed (KERNEL_MAP step 1):

  (A) test_A_*: over every verdict receipt committed on the branch, the projected span verdicts
      and document verdict equal the committed ones except the changes receipt/v0/DECISIONS_v0.md
      lists by name, and that list is exact in both directions.
  (B) test_B_*: every in-scope conformance document (inline mode, nothing beyond a manifest),
      checked under two builds of the lab's checker in two child processes, gives one v0 digest
      and two different receipt digests of today's kind.

(B) needs a second build as a plain package tree. Its path is RECEIPT_V0_OTHER_BUILD, or the path
this box holds; where neither exists the test SKIPS and says so, because nothing on a hosted
runner can stand in for a second build.

The schema is checked twice: by a small validator written here (the keywords the schema uses and
no others) and, when the jsonschema package is importable, by that package as well.
"""
import ast
import base64
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
V0 = ROOT / "receipt" / "v0"
OTHER_BUILD = Path(os.environ.get("RECEIPT_V0_OTHER_BUILD",
                                  "C:/Users/heyzo/clawd/wt/_dd_8_0_0_lead/base_7_50_0"))


def _load_projector():
    spec = importlib.util.spec_from_file_location("receipt_to_v0", str(V0 / "to_v0.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load_projector()
SCHEMA = json.loads((V0 / "receipt-v0.schema.json").read_text(encoding="utf-8"))


def _sha(b):
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------------------------------------
# the small validator: exactly the keywords receipt-v0.schema.json uses
# ---------------------------------------------------------------------------------------------

_KEYWORDS = {"$schema", "$id", "$ref", "$defs", "title", "description", "type", "const", "enum",
             "pattern", "minimum", "maxLength", "maxItems", "required", "properties",
             "additionalProperties", "items", "anyOf", "allOf", "if", "then", "else"}


def _type_ok(v, t):
    if t == "null":
        return v is None
    if t == "boolean":
        return isinstance(v, bool)
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "string":
        return isinstance(v, str)
    if t == "array":
        return isinstance(v, list)
    if t == "object":
        return isinstance(v, dict)
    raise AssertionError("type %r is not used by the schema" % (t,))


def _eq(a, b):
    return type(a) is type(b) and a == b


def _errors(v, s, path="$"):
    assert set(s) <= _KEYWORDS, "schema keyword outside the small validator: %s" % (set(s) - _KEYWORDS)
    errs = []
    if "$ref" in s:
        assert s["$ref"].startswith("#/$defs/")
        errs += _errors(v, SCHEMA["$defs"][s["$ref"][8:]], path)
    if "type" in s:
        ts = s["type"] if isinstance(s["type"], list) else [s["type"]]
        if not any(_type_ok(v, t) for t in ts):
            errs.append("%s: type %s" % (path, ts))
    if "const" in s and not _eq(v, s["const"]):
        errs.append("%s: const %r" % (path, s["const"]))
    if "enum" in s and not any(_eq(v, e) for e in s["enum"]):
        errs.append("%s: enum, got %r" % (path, v))
    if "pattern" in s and isinstance(v, str) and not re.search(s["pattern"], v):
        errs.append("%s: pattern %s" % (path, s["pattern"]))
    if "minimum" in s and _type_ok(v, "integer") and v < s["minimum"]:
        errs.append("%s: minimum" % path)
    if "maxLength" in s and isinstance(v, str) and len(v) > s["maxLength"]:
        errs.append("%s: maxLength" % path)
    if "maxItems" in s and isinstance(v, list) and len(v) > s["maxItems"]:
        errs.append("%s: maxItems" % path)
    if isinstance(v, dict):
        for k in s.get("required", []):
            if k not in v:
                errs.append("%s: required %s" % (path, k))
        props = s.get("properties", {})
        for k, sub in props.items():
            if k in v:
                errs += _errors(v[k], sub, path + "." + k)
        if "additionalProperties" in s:
            ap = s["additionalProperties"]
            for k in v:
                if k not in props:
                    if ap is False:
                        errs.append("%s: additional property %s" % (path, k))
                    elif isinstance(ap, dict):
                        errs += _errors(v[k], ap, path + "." + k)
    if isinstance(v, list) and "items" in s:
        for i, x in enumerate(v):
            errs += _errors(x, s["items"], "%s[%d]" % (path, i))
    if "anyOf" in s and not any(not _errors(v, sub, path) for sub in s["anyOf"]):
        errs.append("%s: anyOf" % path)
    for sub in s.get("allOf", []):
        errs += _errors(v, sub, path)
    if "if" in s:
        branch = s.get("then") if not _errors(v, s["if"], path) else s.get("else")
        if branch is not None:
            errs += _errors(v, branch, path)
    return errs


def _lib_validator(def_name):
    try:
        import jsonschema
    except ImportError:
        return None
    return jsonschema.Draft202012Validator({"$schema": SCHEMA["$schema"],
                                            "$ref": "#/$defs/" + def_name,
                                            "$defs": SCHEMA["$defs"]})


_LIB = {}


def assert_valid(value, def_name):
    errs = _errors(value, {"$ref": "#/$defs/" + def_name})
    assert not errs, errs[:5]
    if def_name not in _LIB:
        _LIB[def_name] = _lib_validator(def_name)
    if _LIB[def_name] is not None:
        lib = [e.message for e in _LIB[def_name].iter_errors(value)]
        assert not lib, lib[:5]


def assert_invalid(value, def_name):
    assert _errors(value, {"$ref": "#/$defs/" + def_name})
    if def_name not in _LIB:
        _LIB[def_name] = _lib_validator(def_name)
    if _LIB[def_name] is not None:
        assert list(_LIB[def_name].iter_errors(value))


def check_core(core):
    """Shape, the digest, and the invariants a schema cannot say."""
    assert_valid(core, "verdictCore")
    rec = P.receipt_v0(core, verifier={"note": "beside the core"})
    assert_valid(rec, "receipt")
    assert rec["digest"] == _sha(P.jcs(core).encode("utf-8"))
    counts = {w: 0 for w in P.SPAN_VERDICTS}
    rungs = {}
    for r in core["spans"]:
        counts[r["verdict"]] += 1
        prov = r.get("provenance") or {}
        key = prov.get("rung") if prov.get("form") == "rn" else (
            "committed" if prov.get("form") in ("path", "prereg") else "unresolved")
        rungs[key] = rungs.get(key, 0) + 1
        if r["claim_sha256"] is not None and r["claim"] is not None:
            assert _sha(r["claim"].encode("utf-8")) == r["claim_sha256"]
    assert core["counts"] == counts and core["sworn_total"] == len(core["spans"])
    assert core["unresolved"] == counts["UNRESOLVED"] and core["rungs"] == rungs
    assert [r["at"] for r in core["spans"]] == sorted(r["at"] for r in core["spans"])
    # D1, restated independently of the projector
    if core["document_malformed"] is not None or counts["FAILED"] or counts["MALFORMED"]:
        want = "SWORN-FAILED"
    elif not core["spans"]:
        want = "UNSWORN"
    else:
        want = "SWORN-HELD" if counts["HELD"] else "SWORN-UNCHECKED"
    assert core["document_verdict"] == want
    if core["profile"] == P.PROFILE_RN:
        for r in core["spans"]:
            assert (r.get("provenance") or {}).get("form") in (None, "rn")
            assert r["reason"] is None or r["reason"] in P.REASONS_V0
            assert (r["start"] is None) == (r["claim_sha256"] is None)


# ---------------------------------------------------------------------------------------------
# the committed verdict receipts
# ---------------------------------------------------------------------------------------------

def _render(sidecar):
    """Sidecar to inline bytes (the sidecar form is outside v0; this is only how a committed
    document kept in that form is turned back into the bytes its receipt names)."""
    text = sidecar["text"].encode("utf-8")
    events = []
    for s in sidecar["spans"]:
        events.append((s["start"], ('<sworn r="%s" k="%s">' % (s["receipt"], s["kind"])).encode("utf-8")))
        events.append((s["end"], b"</sworn>"))
    for off, tag in reversed(events):
        text = text[:off] + tag + text[off:]
    return text


def _document_for(path, receipt):
    """(bytes, sidecar or None) when the document the receipt names sits beside it with the
    sha256 the receipt records; (None, None) otherwise."""
    beside = path.parent / receipt["document"]["name"]
    if not beside.is_file():
        return None, None
    raw = beside.read_bytes()
    side = None
    if beside.suffix == ".json":
        try:
            obj = json.loads(raw.decode("utf-8"))
        except ValueError:
            obj = None
        if isinstance(obj, dict) and obj.get("spec") == "sworn/0.1":
            side, raw = obj, _render(obj)
    if _sha(raw) != receipt["document"]["inline_sha256"]:
        return None, None
    return raw, side


def _committed():
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "--", "*.json"],
                         capture_output=True, check=True).stdout.decode("utf-8")
    found = []
    for rel in sorted(x for x in out.split("\0") if x):
        if rel.startswith("conformance/"):
            continue                       # vectors and blobs are inputs to a checker, not receipts
        p = ROOT / rel
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        if (isinstance(obj, dict) and isinstance(obj.get("spans"), list)
                and str(obj.get("schema", "")).startswith("styxx.sworn.verdict-receipt/")):
            found.append((rel, p, obj))
    return found


COMMITTED = _committed()


def _ledger(name):
    text = (V0 / "DECISIONS_v0.md").read_text(encoding="utf-8")
    m = re.search(r"<!-- %s:begin -->\n```\n(.*?)```\n<!-- %s:end -->" % (name, name), text, re.S)
    assert m, "DECISIONS_v0.md has no %s block" % name
    return [ln for ln in m.group(1).split("\n") if ln.strip()]


def _observed_changes():
    changes, unbound, by_schema = [], [], {}
    for rel, p, rec in COMMITTED:
        by_schema[rec["schema"]] = by_schema.get(rec["schema"], 0) + 1
        raw, _side = _document_for(p, rec)
        core = P.to_v0(rec, raw)
        check_core(core)
        if raw is None:
            unbound.append(rel)
        assert len(core["spans"]) == len(rec["spans"])
        for old, new in zip(rec["spans"], core["spans"]):
            assert (old["at"], old["receipt"], old["kind"]) == (new["at"], new["receipt"], new["kind"])
            if (old["verdict"], old["reason"]) != (new["verdict"], new["reason"]):
                changes.append("span %s@%d %s/%s -> %s/%s" % (rel, old["at"], old["verdict"], old["reason"],
                                                             new["verdict"], new["reason"]))
        if rec["document_verdict"] != core["document_verdict"]:
            changes.append("document %s %s -> %s" % (rel, rec["document_verdict"], core["document_verdict"]))
    return changes, unbound, by_schema


def test_A_committed_receipts_change_only_as_the_decisions_file_lists():
    assert COMMITTED, "no committed verdict receipt found: the test would pass over nothing"
    changes, unbound, by_schema = _observed_changes()
    listed = _ledger("verdict-changes")
    assert sorted(changes) == sorted(listed), {
        "happens_but_unlisted": sorted(set(changes) - set(listed)),
        "listed_but_does_not_happen": sorted(set(listed) - set(changes))}
    assert sorted(unbound) == sorted(_ledger("claims-unbound"))
    census = dict(ln.split(" = ") for ln in _ledger("census"))
    assert census == {
        "committed verdict receipts": str(len(COMMITTED)),
        "at /v1": str(by_schema.get("styxx.sworn.verdict-receipt/v1", 0)),
        "at /v2": str(by_schema.get("styxx.sworn.verdict-receipt/v2", 0)),
        "named *.sworn-receipt.json": str(sum(1 for rel, _, _ in COMMITTED
                                              if rel.endswith(".sworn-receipt.json"))),
        "span rows": str(sum(len(rec["spans"]) for _, _, rec in COMMITTED)),
        "span verdicts changed": str(sum(1 for c in changes if c.startswith("span "))),
        "document verdicts changed": str(sum(1 for c in changes if c.startswith("document "))),
        "claims bound (document on the branch)": str(len(COMMITTED) - len(unbound)),
        "claims unbound (document not on the branch)": str(len(unbound)),
    }


def test_A_the_committed_receipts_are_not_rewritten_by_projection():
    for rel, p, rec in COMMITTED[:10]:
        before = json.dumps(rec, sort_keys=True)
        P.to_v0(rec, _document_for(p, rec)[0])
        assert json.dumps(rec, sort_keys=True) == before
        assert json.loads(p.read_text(encoding="utf-8")) == rec


def test_committed_rn_only_documents_rederive_to_the_same_v0_digest():
    """Where a committed receipt binds only rN spans and its document and manifest are on the
    branch, the live checker's fresh core projects to the digest the committed receipt projects to."""
    sys.path.insert(0, str(ROOT))
    from styxx import sworn
    assert Path(sworn.__file__).resolve().parent.parent == ROOT
    done = 0
    for rel, p, rec in COMMITTED:
        raw, side = _document_for(p, rec)
        if raw is None or side is None or not rec["spans"]:
            continue
        if not all(re.fullmatch(r"r[1-9][0-9]*(#.*)?", s["receipt"] or "") for s in rec["spans"]):
            continue
        manifest = sworn.Manifest.from_dict(side["manifest"])
        fresh = sworn.verify(raw, name=rec["document"]["name"], manifest=manifest, tree=None,
                             commit=rec["commit"])
        a, b = P.to_v0(fresh, raw), P.to_v0(rec, raw)
        assert a["profile"] == P.PROFILE_RN
        assert P.digest_v0(a) == P.digest_v0(b), rel
        done += 1
    assert done >= 3, "expected the three CANNED harness receipts at least; re-derived %d" % done


# ---------------------------------------------------------------------------------------------
# (B) two builds, one v0 digest
# ---------------------------------------------------------------------------------------------

_CHILD = r'''
import base64, hashlib, json, sys
import styxx
from styxx import sworn
job = json.load(open(sys.argv[1], encoding="utf-8"))
out = {"styxx_file": styxx.__file__, "styxx_version": styxx.__version__,
       "sworn_sha256": hashlib.sha256(open(sworn.__file__, "rb").read()).hexdigest(), "results": []}
for it in job["items"]:
    raw = base64.b64decode(it["document_b64"])
    man = None
    if it["manifest_b64"] is not None:
        man = sworn.Manifest.from_dict(json.loads(base64.b64decode(it["manifest_b64"]).decode("utf-8")))
    core = sworn.verify(raw, name=it["name"], manifest=man, tree=None, commit=it["commit"])
    rec = sworn.issue_receipt(core, timestamp="2026-01-01T00:00:00Z")
    out["results"].append({"id": it["id"], "core": core, "digest_today": rec["digest"]})
json.dump(out, open(sys.argv[2], "w", encoding="utf-8"))
'''


def _in_scope_items():
    """Every conformance vector a manifest-only inline checker must answer: mode inline, requires
    within {manifest}, outcome core. Both committed sets."""
    items = []
    for set_dir in (ROOT / "conformance" / "sworn", ROOT / "conformance" / "sworn" / "receipt_v2"):
        blobs = json.loads((set_dir / "blobs.json").read_text(encoding="utf-8"))
        for fam in sorted((set_dir / "vectors").glob("*.json")):
            for v in json.loads(fam.read_text(encoding="utf-8"))["vectors"]:
                if (v["mode"] == "inline" and set(v["requires"]) <= {"manifest"}
                        and v["expect"]["outcome"] == "core"):
                    i = v["inputs"]
                    items.append({"id": v["id"], "set": set_dir.name, "family": v["family"],
                                  "name": i["name"], "commit": i["commit"],
                                  "document_b64": blobs[i["document"]],
                                  "manifest_b64": blobs[i["manifest"]] if i["manifest"] else None,
                                  "expect_document_verdict": v["expect"]["document_verdict"]})
    return items


def _run_build(build, job_path, out_path, child_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(build)
    env["PYTHONIOENCODING"] = "utf-8"
    subprocess.run([sys.executable, str(child_path), str(job_path), str(out_path)],
                   env=env, check=True, capture_output=True, cwd=str(out_path.parent))
    got = json.loads(out_path.read_text(encoding="utf-8"))
    assert Path(got["styxx_file"]).resolve().parent.parent == Path(build).resolve(), got["styxx_file"]
    return got


@pytest.fixture(scope="module")
def two_builds(tmp_path_factory):
    if not (OTHER_BUILD / "styxx" / "sworn.py").is_file():
        pytest.skip("test (B) needs a second build of the checker as a package tree at %s "
                    "(set RECEIPT_V0_OTHER_BUILD); NOT RUN here" % OTHER_BUILD)
    tmp = tmp_path_factory.mktemp("receipt_v0_builds")
    items = _in_scope_items()
    (tmp / "job.json").write_text(json.dumps({"items": items}), encoding="utf-8")
    (tmp / "child.py").write_text(_CHILD, encoding="utf-8")
    here = _run_build(ROOT, tmp / "job.json", tmp / "here.json", tmp / "child.py")
    other = _run_build(OTHER_BUILD, tmp / "job.json", tmp / "other.json", tmp / "child.py")
    return items, here, other


def test_B_two_builds_give_one_v0_digest_and_two_digests_of_todays_kind(two_builds):
    items, here, other = two_builds
    assert here["sworn_sha256"] != other["sworn_sha256"], "the two trees hold the same checker"
    assert len(items) >= 40 and len(here["results"]) == len(other["results"]) == len(items)
    families = {}
    for it, a, b in zip(items, here["results"], other["results"]):
        assert it["id"] == a["id"] == b["id"]
        doc = base64.b64decode(it["document_b64"])
        va, vb = P.to_v0(a["core"], doc), P.to_v0(b["core"], doc)
        check_core(va)
        assert va["profile"] == P.PROFILE_RN
        assert P.digest_v0(va) == P.digest_v0(vb), "v0 digest moved with the build: %s" % it["id"]
        assert a["digest_today"] != b["digest_today"], "today's digest did not move: %s" % it["id"]
        assert a["core"]["document_verdict"] == it["expect_document_verdict"]
        families[(it["set"], it["family"])] = families.get((it["set"], it["family"]), 0) + 1
    assert len(families) >= 10, families
    print("\n(B) %d documents, %d families, builds %s and %s: one v0 digest each, today's digests differ on all"
          % (len(items), len(families), here["styxx_version"], other["styxx_version"]))
    for k in sorted(families):
        print("    %-12s %-20s %d" % (k[0], k[1], families[k]))


def test_B_the_decisions_move_only_what_the_decisions_file_says_in_the_conformance_documents(two_builds):
    """Over the same documents: D1 and D2 are the only reasons a verdict differs from today's."""
    items, here, _ = two_builds
    d1 = d2_rows = d2_docs = 0
    for it, a in zip(items, here["results"]):
        old = a["core"]
        new = P.to_v0(old, base64.b64decode(it["document_b64"]))
        flipped = 0
        for s, t in zip(old["spans"], new["spans"]):
            if (s["verdict"], s["reason"]) != (t["verdict"], t["reason"]):
                assert (s["verdict"], t["verdict"], t["reason"]) == ("HELD", "FAILED", "rounds_to_zero")
                flipped += 1
        d2_rows += flipped
        if old["document_verdict"] != new["document_verdict"]:
            if flipped:
                assert (old["document_verdict"], new["document_verdict"]) == ("SWORN-HELD", "SWORN-FAILED")
                d2_docs += 1
            else:
                assert (old["document_verdict"], new["document_verdict"]) == ("SWORN-HELD", "SWORN-UNCHECKED")
                assert old["counts"]["HELD"] == 0 and old["counts"]["UNRESOLVED"] == old["sworn_total"]
                d1 += 1
    census = dict(ln.split(" = ") for ln in _ledger("conformance-census"))
    assert census == {"in-scope conformance documents": str(len(items)),
                      "document verdicts changed by D1": str(d1),
                      "span verdicts changed by D2": str(d2_rows),
                      "document verdicts changed by D2": str(d2_docs)}


# ---------------------------------------------------------------------------------------------
# D1, D2, D3 on small cases, through the live checker
# ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sworn():
    sys.path.insert(0, str(ROOT))
    from styxx import sworn as mod
    assert Path(mod.__file__).resolve().parent.parent == ROOT
    return mod


def _check(sworn, doc, receipts, rung="L2"):
    man = sworn.Manifest("harness-x", "turn-1", "2026-01-01T00:00:00Z", rung=rung)
    for rid, (body, kw) in receipts.items():
        man.add(rid, body, "tool_stdout", captured_at="2026-01-01T00:00:00Z", **kw)
    raw = doc.encode("utf-8")
    old = sworn.verify(raw, name="d.md", manifest=man, tree=None)
    new = P.to_v0(old, raw)
    check_core(new)
    if rung in ("L1", "L2", None):
        assert_valid(man.core(), "manifestCore")
        assert_valid(man.to_dict(), "manifest")
    else:
        assert_invalid(man.core(), "manifestCore")
    assert P.digest_v0(man.core()) == man.digest()        # the manifest digest under v0's form
    return old, new


def test_D1_nothing_checked_is_not_held(sworn):
    doc = 'A <sworn r="r1" k="numeric">rate of 0.42</sworn> and <sworn r="r9" k="numeric">n of 7</sworn>.'
    old, new = _check(sworn, doc.replace("r1", "r8"), {"r1": (b"0.4211", {})})
    assert old["document_verdict"] == "SWORN-HELD" and old["counts"]["HELD"] == 0
    assert new["document_verdict"] == "SWORN-UNCHECKED" and new["unresolved"] == 2
    old, new = _check(sworn, doc, {"r1": (b"0.4211", {})})
    assert (old["document_verdict"], new["document_verdict"]) == ("SWORN-HELD", "SWORN-HELD")
    assert new["counts"] == {"HELD": 1, "FAILED": 0, "UNRESOLVED": 1, "MALFORMED": 0}
    assert new["unresolved"] == 1                         # beside the verdict, inside the digest
    old, new = _check(sworn, doc, {"r1": (b"0.99", {})})
    assert new["document_verdict"] == "SWORN-FAILED"
    old, new = _check(sworn, "nothing sworn here", {})
    assert new["document_verdict"] == "UNSWORN" and new["sworn_total"] == 0
    old, new = _check(sworn, "```\nopen fence", {})
    assert new["document_verdict"] == "SWORN-FAILED" and new["document_malformed"]["reason"] == "unbalanced_fences"
    # one manifest string no longer turns a caught mismatch into a held document
    old, new = _check(sworn, doc, {"r1": (b"0.99", {})}, rung="L9")
    assert old["document_verdict"] == "SWORN-HELD" and new["document_verdict"] == "SWORN-UNCHECKED"


@pytest.mark.parametrize("printed,receipt,today,v0,reason", [
    ("0.", b"0.4211", "HELD", "FAILED", "rounds_to_zero"),
    ("0.00", b"0.004", "HELD", "FAILED", "rounds_to_zero"),
    ("0", b"-0.3", "HELD", "FAILED", "rounds_to_zero"),
    ("0.0", b"4e-9", "HELD", "FAILED", "rounds_to_zero"),
    ("0", b"0", "HELD", "HELD", None),
    ("0.00", b"0.000", "HELD", "HELD", None),
    ("0.0", b"0E-7", "HELD", "HELD", None),
    ("0.42", b"0.4211", "HELD", "HELD", None),
    ("1", b"0.6", "HELD", "HELD", None),
    ("0.01", b"0.005001", "HELD", "HELD", None),
    ("1", b"0.4211", "FAILED", "FAILED", "value_mismatch"),
    ("0.5", b"0.4211", "FAILED", "FAILED", "value_mismatch"),
])
def test_D2_a_nonzero_receipt_that_rounds_to_zero_does_not_hold(sworn, printed, receipt, today, v0, reason):
    old, new = _check(sworn, '<sworn r="r1" k="numeric">the share is %s</sworn>' % printed,
                      {"r1": (receipt, {})})
    assert old["spans"][0]["verdict"] == today
    assert (new["spans"][0]["verdict"], new["spans"][0]["reason"]) == (v0, reason)
    assert new["spans"][0]["detail"] == old["spans"][0]["detail"]
    assert new["document_verdict"] == ("SWORN-HELD" if v0 == "HELD" else "SWORN-FAILED")


def test_D3_the_claim_is_bound_in_the_row(sworn):
    inner = "le taux est de 0,42 \u2014 r\u00e9sum\u00e9 \U0001F9EA `needle that is long enough`"
    doc = 'x <sworn r="r1" k="quote">%s</sworn> y <!-- <sworn r="r1" k="quote">`hidden needle long`</sworn> -->' % inner
    old, new = _check(sworn, doc, {"r1": (b"... needle that is long enough ...", {})})
    row = new["spans"][0]
    assert row["verdict"] == "HELD" and row["claim"] == inner
    assert row["claim_sha256"] == _sha(inner.encode("utf-8"))
    hidden = new["spans"][1]
    assert hidden["reason"] == "hidden_commitment" and hidden["claim"] == "`hidden needle long`"
    # the cap: 300 code points are carried, 301 are not; the hash is carried either way
    for n, carried in ((300, True), (301, False)):
        body = "\u00e9" * (n - 2) + " 7"
        old, new = _check(sworn, '<sworn r="r1" k="numeric">%s</sworn>' % body, {"r1": (b"7", {})})
        row = new["spans"][0]
        assert row["claim_sha256"] == _sha(body.encode("utf-8"))
        assert (row["claim"] == body) is carried and (row["claim"] is None) is (not carried)
        assert (row["reason"] == "length_cap") is (not carried)
    # two documents that differ only inside a span no longer share a row
    a = _check(sworn, '<sworn r="r1" k="numeric">up 7</sworn>', {"r1": (b"7", {})})[1]
    b = _check(sworn, '<sworn r="r1" k="numeric">no 7</sworn>', {"r1": (b"7", {})})[1]
    assert a["spans"][0]["detail"] == b["spans"][0]["detail"]
    assert a["spans"][0]["claim_sha256"] != b["spans"][0]["claim_sha256"]


def test_D3_without_the_document_the_projection_is_legacy_and_says_so(sworn):
    doc = '<sworn r="r1" k="numeric">n is 7</sworn>'
    old, _ = _check(sworn, doc, {"r1": (b"7", {})})
    bare = P.to_v0(old)
    check_core(bare)
    assert bare["profile"] == P.PROFILE_LEGACY and bare["spans"][0]["claim_sha256"] is None
    with pytest.raises(P.Refusal) as e:
        P.to_v0(old, doc.encode("utf-8") + b" ")
    assert e.value.code == "document_mismatch"
    # a document with no canonical text binds no claim and stays in the rn profile
    doc = '<sworn r="r1" k="numeric">n is 7</sworn> </sworn>'
    old, new = _check(sworn, doc, {"r1": (b"7", {})})
    assert [r["start"] for r in new["spans"]] == [None, None]
    assert [r["claim_sha256"] for r in new["spans"]] == [None, None]
    assert new["profile"] == P.PROFILE_RN and new["document"]["canonical_sha256"] is None


def test_detail_carries_json_type_names_not_python_ones(sworn):
    for body, today, v0 in ((b"[1]", "list", "array"), (b'{"a":1}', "_Obj", "object"), (b"null", "NoneType", "null")):
        old, new = _check(sworn, '<sworn r="r1" k="numeric">n is 7</sworn>', {"r1": (body, {})})
        assert old["spans"][0]["detail"] == {"leaf_type": today}
        assert new["spans"][0]["detail"] == {"leaf_type": v0}


def test_the_core_leaves_out_the_build_the_sentence_and_the_observer(sworn):
    old, new = _check(sworn, '<sworn r="r1" k="hash">%s</sworn>' % _sha(b"x"), {"r1": (b"x", {})})
    assert {"verifier", "certifies", "coverage"} <= set(old)
    assert sorted(new) == sorted(SCHEMA["$defs"]["verdictCore"]["required"]) and len(new) == 13
    text = P.jcs(new)
    assert old["verifier"]["sworn_sha256"] not in text and old["verifier"]["styxx_version"] not in text
    assert old["certifies"][:40] not in text and "sentence_share" not in text
    rec = P.receipt_v0(new, verifier=old["verifier"])
    assert rec["digest"] == P.digest_v0(new) and rec["attestations"] == []
    assert_invalid(dict(rec, attestations=[{"by": "anyone"}]), "receipt")


# ---------------------------------------------------------------------------------------------
# the canonical form
# ---------------------------------------------------------------------------------------------

def test_jcs_known_answers():
    assert P.jcs({"b": [1, True, None, "x"], "a": {"z": -3, "y": ""}}) == '{"a":{"y":"","z":-3},"b":[1,true,null,"x"]}'
    assert P.jcs('"\\\b\f\n\r\t\x00\x1f\x7f\u00e9\u2028\U0001F600') == '"\\"\\\\\\b\\f\\n\\r\\t\\u0000\\u001f\x7f\u00e9\u2028\U0001F600"'
    assert P.jcs(2 ** 53) == "9007199254740992" and P.jcs(-(2 ** 53)) == "-9007199254740992"
    assert P.jcs({"B": 1, "a": 2, "_": 3, "A": 4}) == '{"A":4,"B":1,"_":3,"a":2}'
    assert P.digest_v0({}) == _sha(b"{}")


@pytest.mark.parametrize("value,code", [
    (1.5, "float"), (1.0, "float"), ({"a": [float("nan")]}, "float"),
    (2 ** 53 + 1, "int_range"), (-(2 ** 53) - 1, "int_range"),
    ({"\u00e9": 1}, "non_ascii_key"), ({"\x7f": 1}, "non_ascii_key"), ({1: 1}, "type"),
    ("\ud800", "lone_surrogate"), (b"bytes", "type"), ({1, 2}, "type"),
])
def test_jcs_refuses(value, code):
    with pytest.raises(P.Refusal) as e:
        P.jcs(value)
    assert e.value.code == code and code in P.REFUSALS


def test_jcs_depth():
    def nest(n):
        v = "x"
        for _ in range(n):
            v = [v]
        return v
    assert P.jcs(nest(32)).startswith("[" * 32)
    with pytest.raises(P.Refusal) as e:
        P.jcs(nest(33))
    assert e.value.code == "depth"


def test_jcs_agrees_with_the_labs_own_on_every_projected_committed_core(sworn):
    from styxx.attestation import jcs as lab_jcs
    for rel, p, rec in COMMITTED:
        core = P.to_v0(rec, _document_for(p, rec)[0])
        assert P.jcs(core) == lab_jcs(core), rel


# ---------------------------------------------------------------------------------------------
# the projector's refusals, and its purity
# ---------------------------------------------------------------------------------------------

def _good(sworn):
    return _check(sworn, '<sworn r="r1" k="numeric">n is 7</sworn>', {"r1": (b"7", {})})[0]


@pytest.mark.parametrize("edit,code", [
    (lambda c: "not a core", "not_a_core"),
    (lambda c: dict(c, schema="styxx.sworn.verdict-receipt/v0"), "schema_out_of_scope"),
    (lambda c: dict(c, schema="receipt/v0-draft"), "schema_out_of_scope"),
    (lambda c: dict(c, format="sworn/0.2"), "format_unknown"),
    (lambda c: dict(c, counts=dict(c["counts"], HELD=2)), "counts_mismatch"),
    (lambda c: dict(c, counts=dict(c["counts"], HELD=0, WITHHELD=1),
                    spans=[dict(c["spans"][0], verdict="WITHHELD")]), "verdict_out_of_scope"),
    (lambda c: dict(c, counts=dict(c["counts"], HELD=0, FAILED=1),
                    spans=[dict(c["spans"][0], verdict="FAILED", reason="made_up")]), "reason_unknown"),
    (lambda c: dict(c, spans=[dict(c["spans"][0], reason="value_mismatch")]), "row_shape"),
    (lambda c: dict(c, document_malformed={"reason": "other"}), "not_a_core"),
    (lambda c: dict(c, rungs={"L2": 1.0}), "float"),
])
def test_the_projector_refuses_rather_than_guesses(sworn, edit, code):
    with pytest.raises(P.Refusal) as e:
        P.to_v0(edit(_good(sworn)))
    assert e.value.code == code and code in P.REFUSALS


def test_a_claim_that_is_not_where_the_row_says_is_refused(sworn):
    old = _good(sworn)
    moved = dict(old, spans=[dict(old["spans"][0], at=1)])
    with pytest.raises(P.Refusal) as e:
        P.to_v0(moved, b'<sworn r="r1" k="numeric">n is 7</sworn>')
    assert e.value.code == "claim_unlocatable"


def test_the_closed_lists_in_the_text_the_schema_and_the_code_are_one_list():
    text = (V0 / "RECEIPT_v0.md").read_text(encoding="utf-8")
    assert sorted(SCHEMA["$defs"]["reason"]["enum"]) == sorted(P.REASONS_V0)
    assert sorted(SCHEMA["$defs"]["reasonLegacyOnly"]["enum"]) == sorted(P.REASONS_LEGACY_ONLY)
    assert SCHEMA["$defs"]["spanVerdict"]["enum"] == list(P.SPAN_VERDICTS)
    assert SCHEMA["$defs"]["documentVerdict"]["enum"] == list(P.DOCUMENT_VERDICTS)
    for word in (P.REASONS_V0 + P.REASONS_LEGACY_ONLY + P.SPAN_VERDICTS + P.DOCUMENT_VERDICTS
                 + P.DOCUMENT_REASONS + P.REFUSALS[:6] + (P.SCHEMA_V0, P.PROFILE_RN, P.PROFILE_LEGACY)):
        assert "`%s`" % word in text, "RECEIPT_v0.md does not name %s" % word
    assert text.split("\n")[2].startswith("Status: DRAFT, unsigned, nothing frozen")


def test_to_v0_is_pure_by_its_source():
    """Standard library only, and only hashlib of it; no call that reads a clock, opens a file,
    starts a process or reaches a network; no module-level state that a call changes."""
    src = (V0 / "to_v0.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
            assert node.level == 0
    assert imported == {"hashlib"}, imported
    assert imported <= set(sys.stdlib_module_names)
    banned_calls = {"open", "exec", "eval", "compile", "__import__", "input", "print", "globals",
                    "setattr", "delattr", "breakpoint"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in banned_calls, node.func.id
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise AssertionError("module state changed from inside a function")
    code_only = "\n".join(ast.unparse(n) for n in tree.body
                          if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)))
    for word in ("subprocess", "socket", "urllib", "http", "datetime", "time.", "os.", "pathlib",
                 "random", "write(", "environ"):
        assert word not in code_only, word
    assert "styxx" not in imported and not re.search(r"^\s*(from|import)\s+styxx", src, re.M)
