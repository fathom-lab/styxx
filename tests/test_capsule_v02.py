"""OATH Capsule v0.2: the agent-handoff battery.

Mint refusals R1-R7, the tamper battery (binding stage, the K2 re-sealed forgery,
instrument skew, ambiguous payloads), the advisory discipline around unparsed_claims,
CRLF byte-faithfulness, JCS parity between the Python canonicalizer and the template's
inline JS, and the v0.1 regression through the spec dispatcher.

Spec: papers/closed-model-frontier/SPEC_oath_capsule_v02_2026_08_31.md
"""
from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys

import pytest

from styxx.attestation import jcs
from styxx.capsule import (
    _BEGIN,
    _END,
    _TEMPLATE_V02,
    _gate_binding_hash,
    _v02_folds,
    create_capsule_diffgate,
    verify_capsule,
)
from styxx.diffgate import gate_diff_text

SUMMARY_PASS = (
    "Added styxx/newmod.py with the frobnicate helper — touché, très vite. "
    "Deleted old_helper.py. All tests pass. "
    "This change also refactors the widget pipeline for clarity.\n")

DIFF_PASS = (
    "diff --git a/styxx/newmod.py b/styxx/newmod.py\n"
    "new file mode 100644\n"
    "--- /dev/null\n"
    "+++ b/styxx/newmod.py\n"
    "@@ -0,0 +1,2 @@\n"
    "+def frobnicate(x):\n"
    "+    return x + 1\n"
    "diff --git a/old_helper.py b/old_helper.py\n"
    "deleted file mode 100644\n"
    "--- a/old_helper.py\n"
    "+++ /dev/null\n"
    "@@ -1,2 +0,0 @@\n"
    "-def old(x):\n"
    "-    return x\n")

# A summary that still earns a CONTRADICTED after EXTERNAL-1 disabled the
# path-claim accusation (RESULT_external1_the_gate_fails_in_the_wild_2026_08_31):
# the file-count claim is checked by counting the diff, not by matching prose
# paths, so it is untouched by that repair. The old fixture claimed a
# nonexistent file and now correctly returns UNCHECKABLE rather than accusing.
SUMMARY_LIE = (
    "This change is small: 7 files changed in total across the codebase.\n")


def _mint(tmp_path, summary=SUMMARY_PASS, diff=DIFF_PASS, gate=None, name="s"):
    sp = tmp_path / f"{name}.md"
    sp.write_text(summary, encoding="utf-8")
    dp = tmp_path / f"{name}.diff"
    dp.write_text(diff, encoding="utf-8")
    gp = None
    if gate is not None:
        gp = tmp_path / f"{name}.gate.json"
        gp.write_text(json.dumps(gate), encoding="utf-8")
    out = tmp_path / f"{name}.capsule.html"
    return create_capsule_diffgate(sp, dp, out, gp)


def _payload(path):
    html = path.read_text(encoding="utf-8")
    i = html.index(_BEGIN) + len(_BEGIN)
    j = html.index(_END, i)
    return html, i, j, json.loads(html[i:j])


def _rewrite(path, html, i, j, payload):
    body = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    path.write_text(html[:i] + body + html[j:], encoding="utf-8")


def _live_gate(summary=SUMMARY_PASS, diff=DIFF_PASS):
    return gate_diff_text(summary, diff, run=None, strict=False).to_dict()


# ------------------------------------------------------------------- minting

def test_happy_pass_mints_and_verifies(tmp_path):
    out = _mint(tmp_path)
    _, _, _, p = _payload(out)
    assert p["spec"] == "styxx-oath/capsule/v0.2"
    assert set(p) == {"spec", "created", "summary", "diff", "gate", "binding", "verifier"}
    assert p["gate"] == _live_gate()
    assert p["binding"]["gate"]["alg"] == "sha256-jcs"
    r = verify_capsule(out)
    assert r["ok"], r["problems"]
    assert r["verdict"] == "PASS"
    assert r["gate_reproduced"] is True


def test_fail_gate_mints_normally(tmp_path):
    """Refusal is never about the verdict's color."""
    out = _mint(tmp_path, summary=SUMMARY_LIE)
    _, _, _, p = _payload(out)
    assert p["gate"]["verdict"] == "FAIL"
    assert any(c["verdict"] == "CONTRADICTED" for c in p["gate"]["claims"])
    r = verify_capsule(out)
    assert r["ok"], r["problems"]
    assert r["verdict"] == "FAIL"


def test_zero_claim_summary_mints(tmp_path):
    out = _mint(tmp_path, summary="General cleanup work across the module.\n")
    _, _, _, p = _payload(out)
    assert p["gate"]["claims"] == []
    assert verify_capsule(out)["ok"]


def test_uncheckable_tests_pass_mints(tmp_path):
    out = _mint(tmp_path)
    _, _, _, p = _payload(out)
    tp = [c for c in p["gate"]["claims"] if c["kind"] == "tests_pass"]
    assert tp and all(c["verdict"] == "UNCHECKABLE" for c in tp)


def test_r1_missing_inputs_refused(tmp_path):
    sp = tmp_path / "s.md"
    sp.write_text(SUMMARY_PASS, encoding="utf-8")
    with pytest.raises(SystemExit, match="REFUSED: cannot read diff"):
        create_capsule_diffgate(sp, tmp_path / "absent.diff", tmp_path / "o.html")


def test_r3_non_utf8_diff_refused(tmp_path):
    sp = tmp_path / "s.md"
    sp.write_text(SUMMARY_PASS, encoding="utf-8")
    dp = tmp_path / "s.diff"
    dp.write_bytes(b"\xff\xfe\x00garbage")
    with pytest.raises(SystemExit, match="REFUSED: cannot read diff as UTF-8"):
        create_capsule_diffgate(sp, dp, tmp_path / "o.html")


def test_r2_unknown_diffgate_version_refused(tmp_path):
    g = _live_gate()
    g["diffgate"] = "v1"
    with pytest.raises(SystemExit, match="unknown diffgate version"):
        _mint(tmp_path, gate=g)


def test_r4_environment_leg_refused(tmp_path):
    g = _live_gate()
    for c in g["claims"]:
        if c["kind"] == "tests_pass":
            c["verdict"] = "VERIFIED"
            c["why"] = "run passed"
    with pytest.raises(SystemExit, match="environment legs cannot be capsuled"):
        _mint(tmp_path, gate=g)


def test_r5_stale_gate_refused(tmp_path):
    stale = _live_gate(summary="Added styxx/other.py entirely.\n")
    with pytest.raises(SystemExit, match="does not reproduce from these bytes"):
        _mint(tmp_path, gate=stale)


def test_r5_strict_verdict_refused_with_policy_message(tmp_path):
    g = _live_gate()
    assert any(c["verdict"] == "UNCHECKABLE" for c in g["claims"])
    assert g["verdict"] == "PASS"
    g["verdict"] = "FAIL"  # a strict=True record over the same bytes
    with pytest.raises(SystemExit, match="non-strict by policy"):
        _mint(tmp_path, gate=g)


def test_r5_repo_refs_in_base_head_accepted(tmp_path):
    g = _live_gate()
    g["base"], g["head"] = "main", "feat/x"  # a repo-produced gate, otherwise identical
    out = _mint(tmp_path, gate=g)
    _, _, _, p = _payload(out)
    assert p["gate"]["base"] == p["gate"]["head"] == "(diff-text)"  # discarded, not sealed


def test_r6_unmeasured_refused(tmp_path):
    with pytest.raises(SystemExit, match="cannot carry proof of a non-measurement"):
        _mint(tmp_path, diff="")


def test_crlf_twins_bind_identically(tmp_path):
    a = _mint(tmp_path, name="lf")
    sp = tmp_path / "crlf.md"
    sp.write_bytes(SUMMARY_PASS.replace("\n", "\r\n").encode("utf-8"))
    dp = tmp_path / "crlf.diff"
    dp.write_bytes(DIFF_PASS.replace("\n", "\r\n").encode("utf-8"))
    b = create_capsule_diffgate(sp, dp, tmp_path / "crlf.capsule.html")
    pa = _payload(a)[3]
    pb = _payload(b)[3]
    assert pa["binding"] == pb["binding"]
    assert verify_capsule(b)["ok"]


def test_marker_text_in_summary_mints_uniquely(tmp_path):
    hostile = (SUMMARY_PASS +
               'See <script type="application/json" id="oath-capsule"> and </script> '
               "for details.\n")
    out = _mint(tmp_path, summary=hostile)
    html = out.read_text(encoding="utf-8")
    assert html.count(_BEGIN) == 1
    assert verify_capsule(out)["ok"]


def test_r7_corrupted_render_refused(tmp_path, monkeypatch):
    import styxx.capsule as cap
    real = cap._render_html_v02
    monkeypatch.setattr(cap, "_render_html_v02",
                        lambda p: real(p) + _BEGIN + "{}" + _END)
    with pytest.raises(SystemExit, match="marker is not unique"):
        _mint(tmp_path)


# ------------------------------------------------------------------ tampering

def test_tampered_summary_fails_at_binding(tmp_path):
    out = _mint(tmp_path)
    html, i, j, p = _payload(out)
    raw = bytearray(base64.b64decode(p["summary"]["b64"]))
    raw[0] ^= 0x01
    p["summary"]["b64"] = base64.b64encode(bytes(raw)).decode()
    _rewrite(out, html, i, j, p)
    r = verify_capsule(out)
    assert not r["ok"] and r["stage"] == "binding"
    assert any("summary bytes" in x for x in r["problems"])


def test_tampered_diff_fails_at_binding(tmp_path):
    out = _mint(tmp_path)
    html, i, j, p = _payload(out)
    raw = bytearray(base64.b64decode(p["diff"]["b64"]))
    raw[-1] ^= 0x01
    p["diff"]["b64"] = base64.b64encode(bytes(raw)).decode()
    _rewrite(out, html, i, j, p)
    r = verify_capsule(out)
    assert not r["ok"] and r["stage"] == "binding"


def test_verdict_flip_only_fails_at_binding(tmp_path):
    out = _mint(tmp_path)
    html, i, j, p = _payload(out)
    p["gate"]["verdict"] = "FAIL"
    _rewrite(out, html, i, j, p)
    r = verify_capsule(out)
    assert not r["ok"] and r["stage"] == "binding"
    assert any("sha256-jcs" in x for x in r["problems"])


def test_k2_reseal_caught_by_reexecution(tmp_path):
    """Flip a claim verdict AND recompute binding.gate — hashes pass, the
    instrument catches it. The forger who re-seals still cannot forge."""
    out = _mint(tmp_path)
    html, i, j, p = _payload(out)
    victim = next(c for c in p["gate"]["claims"] if c["verdict"] == "VERIFIED")
    victim["verdict"] = "CONTRADICTED"
    p["gate"]["verdict"] = "FAIL"  # keep the fold coherent — a careful forger
    p["binding"]["gate"]["value"] = _gate_binding_hash(p["gate"])
    _rewrite(out, html, i, j, p)
    r = verify_capsule(out)
    assert not r["ok"] and r["stage"] == "reproduced"
    assert any("gate.claims" in x or "gate.verdict" in x for x in r["problems"])


def test_duplicate_payload_refused(tmp_path):
    out = _mint(tmp_path)
    html = out.read_text(encoding="utf-8")
    out.write_text(html + _BEGIN + "{}" + _END, encoding="utf-8")
    r = verify_capsule(out)
    assert not r["ok"]
    assert any("ambiguous" in x for x in r["problems"])


def test_alien_spec_refused(tmp_path):
    out = _mint(tmp_path)
    html, i, j, p = _payload(out)
    p["spec"] = "styxx-oath/capsule/v9.9"
    _rewrite(out, html, i, j, p)
    r = verify_capsule(out)
    assert not r["ok"] and r["stage"] == "spec"


def test_capsule_file_crlf_rewrite_still_verifies(tmp_path):
    out = _mint(tmp_path)
    out.write_bytes(out.read_text(encoding="utf-8").replace("\n", "\r\n")
                    .encode("utf-8"))
    assert verify_capsule(out)["ok"]


def test_instrument_skew_classified(tmp_path):
    out = _mint(tmp_path)
    html, i, j, p = _payload(out)
    p["gate"]["diffgate"] = "v-fake"
    p["binding"]["gate"]["value"] = _gate_binding_hash(p["gate"])
    _rewrite(out, html, i, j, p)
    r = verify_capsule(out)
    assert not r["ok"]
    assert any("INSTRUMENT SKEW" in x for x in r["problems"])


# ------------------------------------------------------------------- advisory

def test_unparsed_divergence_is_advisory_not_tamper(tmp_path):
    out = _mint(tmp_path)
    html, i, j, p = _payload(out)
    p["gate"]["unparsed_claims"] = ["a sentence claimdetect elsewhere flagged"]
    p["binding"]["gate"]["value"] = _gate_binding_hash(p["gate"])
    _rewrite(out, html, i, j, p)
    r = verify_capsule(out)
    assert r["ok"], r["problems"]
    assert any("unparsed_claims" in a for a in r["advisory"])


def test_claimdetect_unavailable_is_skipped_advisory(tmp_path, monkeypatch):
    out = _mint(tmp_path)
    monkeypatch.setitem(sys.modules, "styxx.claimdetect", None)
    r = verify_capsule(out)
    assert r["ok"], r["problems"]
    assert any("SKIPPED" in a for a in r["advisory"])


# ---------------------------------------------------------------- invariants

def test_v02_folds_on_live_gates():
    assert _v02_folds(_live_gate()) == []
    assert _v02_folds(_live_gate(summary=SUMMARY_LIE)) == []
    broken = _live_gate()
    broken["verdict"] = "FAIL"
    assert any("verdict fold" in x for x in _v02_folds(broken))
    broken2 = _live_gate()
    broken2["uncovered_sentences"] += 1
    assert any("uncovered count" in x for x in _v02_folds(broken2))
    broken3 = _live_gate()
    broken3["base"] = "main"
    assert any("base/head" in x for x in _v02_folds(broken3))


def test_v01_regression_through_dispatcher(tmp_path):
    """The dispatcher must route v0.1 capsules through the untouched branch."""
    from styxx.capsule import create_capsule
    from styxx.certify import certify_doc
    doc = tmp_path / "d.md"
    doc.write_text("The run scored 0.75 accuracy over 40 items.\n", encoding="utf-8")
    rec = tmp_path / "r.json"
    rec.write_text(json.dumps({"eval": {"accuracy": 0.75, "items": 40}}),
                   encoding="utf-8")
    cert = certify_doc(doc, [rec])
    cp = tmp_path / "d.certificate.json"
    cp.write_text(json.dumps(cert), encoding="utf-8")
    out = tmp_path / "d.capsule.html"
    create_capsule(doc, [rec], cp, out)
    r = verify_capsule(out)
    assert r["ok"], r["problems"]
    assert r["spec"] == "styxx-oath/capsule/v0.1"


def test_jcs_parity_python_vs_template_js(tmp_path):
    """The template's inline JS canonicalizer must produce byte-identical JCS
    to styxx.attestation.jcs for a real (unicode-bearing) gate record."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node unavailable")
    start = _TEMPLATE_V02.index("const jcs = ")
    end = _TEMPLATE_V02.index("+ '}';", start) + len("+ '}';")
    js_fn = _TEMPLATE_V02[start:end]
    gate = _live_gate()
    fx = tmp_path / "gate.json"
    fx.write_text(json.dumps(gate, ensure_ascii=False), encoding="utf-8")
    script = tmp_path / "jcs_check.js"
    script.write_text(
        js_fn + "\n"
        "const fs = require('fs');\n"
        "const g = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));\n"
        "fs.writeFileSync(process.argv[3], jcs(g), 'utf8');\n",
        encoding="utf-8")
    outp = tmp_path / "js.out"
    subprocess.run([node, str(script), str(fx), str(outp)], check=True)
    assert outp.read_text(encoding="utf-8") == jcs(gate)


def test_template_renders_disclosures():
    """The layer-1 template carries the honest-boundary machinery it promises."""
    for needle in ("could not be located for painting",
                   "timestamp unsealed",
                   "display only, not parsed",
                   "environment leg — refused at mint by construction",
                   "listed, never judged",
                   "UNVERIFIED RENDERING"):
        assert needle in _TEMPLATE_V02, needle
