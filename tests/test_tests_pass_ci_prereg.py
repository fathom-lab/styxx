# -*- coding: utf-8 -*-
"""PREREG_tests_pass_ci_2026_11_02 (papers/tests-pass-ci), on CPU, offline, with no data of the window.

What is pinned here: the gates block hashes to the digest the PREREG's section 19 prints (f2c378b3..., 20,839 bytes) and
parses through the lab's own `_select_gates_block` with one fence and no duplicate key; every one of the 16,384 boolean
assignments of the fourteen gates reaches a row, every one of the 45 rows is reached, by the stated counts, and every
gate is named by some row; `Experiment` (power basis and non-vacuity refusals on) scores a receipt of Part II P14's
shape to each of the 45 verdicts and reads no gate on a smoke, and the per-agent block sends a cleared pooled line to
rule R's rows; the Wilson and Clopper-Pearson numbers the bars and section 9 quote re-derive from `styxx.mind.wilson`
and the exact binomial; the lock table's digests of the two prompts, rule R's summaries, the vendored testreceipt
files and the forecast are the bytes in the folder; the forecast's rows are the block's halves and its probabilities
sum to one; the collector's seal guard refuses this file's own draft text and opens on a signed, filled copy of it on
origin/main with the running code; the folder is byte-pinned (-text); and none of the documents prints the words the
lab never prints. Nothing here reads a pull request, a CI record or a log; nothing needs a network.
The `Experiment` tests skip when the PREREG is not committed in the checkout, because the protocol refuses a draft on
disk.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import itertools
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests import _tests_pass_ci_fakes  # noqa: F401  (puts the collector on sys.path)

ROOT = Path(__file__).resolve().parent.parent
FOLDER = ROOT / "papers" / "tests-pass-ci"
PREREG = FOLDER / "PREREG_tests_pass_ci_2026_11_02.md"
FORECAST = FOLDER / "FORECAST_tests_pass_ci_2026_11_02.md"
README = FOLDER / "README.md"
COLLECT = FOLDER / "collect"
TEST_FILE = Path(__file__).resolve()

GATES_SHA256 = "f2c378b369b6b0b3917fc2f001f02b8364dbd8f5b8f77582c33fc346401d96d8"
GATES_BYTES = 20839
GATES = ["G_A1_V", "G_A1_LOW", "G_A1_HIGH", "G_A1_C", "G_A1_ADD", "G_N", "G_KEYS", "G_P2_H", "G_P2_N", "G_W",
         "G_AG", "G_PC", "G_RN", "G_RULE"]
A2_HALVES = ["COUNTS_ONLY__a2_under_400", "INVALID__a2_keys_not_sealed_in_order",
             "INVALID__a2_panel_void_outside_seat", "INVALID__a2_one_model_family_counts_only",
             "KILL__no_tests_pass_accuser", "PART_NOT_RARE__new_prereg_may_propose_part_accuser",
             "NO_ACCUSER__rule_underpowered", "NO_ACCUSER__rule_below_floor",
             "FLOOR_MET__new_prereg_may_propose_accuser"]
A1_HALVES = ["A1_VOID", "A1_LOWER", "A1_HIGHER", "A1_STANDS", "A1_UNSETTLED"]
VERDICTS = [a2 + "__" + a1 for a2 in A2_HALVES for a1 in A1_HALVES]
ROW_COUNTS = [4096, 2048, 1024, 256, 768, 2048, 1024, 512, 128, 384, 1024, 512, 256, 64, 192, 512, 256, 128, 32, 96,
              64, 32, 16, 4, 12, 64, 32, 16, 4, 12, 192, 96, 48, 12, 36, 96, 48, 24, 6, 18, 96, 48, 24, 6, 18]


def _text() -> str:
    return PREREG.read_text(encoding="utf-8")


def _block():
    from styxx.protocol import _select_gates_block
    return _select_gates_block(_text())


def _spec():
    def no_dup(pairs):
        d = {}
        for k, v in pairs:
            assert k not in d, f"duplicate key {k}"
            d[k] = v
        return d
    return json.loads(_block(), object_pairs_hook=no_dup)


def _walk(rows, fired):
    for i, row in enumerate(rows):
        if all(fired.get(k) == v for k, v in row["when"].items()):
            return i
    return None


def _experiment():
    from styxx.protocol import Experiment, PrologueError
    try:
        return Experiment(PREREG, repo_root=ROOT, require_power_basis=True, require_nonvacuous_gates=True)
    except PrologueError:
        pytest.skip("the PREREG is not committed in this checkout; the protocol refuses a draft on disk")


def _cp_lower(k, n, alpha=0.05):
    """Part II P15: the one-sided 95% Clopper-Pearson lower bound."""
    if n <= 0 or k <= 0:
        return 0.0
    if k == n:
        return alpha ** (1.0 / n)

    def tail(p):
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))

    lo, hi = 0.0, 1.0
    while hi - lo > 1e-12:
        mid = (lo + hi) / 2
        if tail(mid) < 0.05:
            lo = mid
        else:
            hi = mid
    return lo


# -- the gates block --------------------------------------------------------------------------------------

def test_gates_block_digest_and_size_are_the_ones_section_19_prints():
    raw = _block().encode("utf-8")
    assert hashlib.sha256(raw).hexdigest() == GATES_SHA256
    assert len(raw) == GATES_BYTES
    text = _text()
    assert text.count("```gates") == 1
    assert GATES_SHA256 in text  # section 19 prints it


def test_gates_block_parses_with_one_fence_and_no_duplicate_keys():
    spec = _spec()
    assert list(spec["gates"]) == GATES
    assert spec["smoke_verdict"] == "INVALID__smoke_plumbing_only"
    assert [r["verdict"] for r in spec["outcomes"]] == VERDICTS
    for g in spec["gates"].values():
        assert isinstance(g["power_basis"], str) and len(g["power_basis"].strip()) > 40
        assert isinstance(g["metric_means"], str) and g["metric_means"].strip()
        assert g["metric"].startswith("result.")
        assert g["op"] in (">=", "<=", ">", "<", "==")
    assert all("withdraws" not in r for r in spec["outcomes"])


def test_every_assignment_reaches_a_row_and_every_row_is_reached_by_the_stated_counts():
    rows = _spec()["outcomes"]
    reached = [0] * len(rows)
    for bits in itertools.product((False, True), repeat=len(GATES)):
        i = _walk(rows, dict(zip(GATES, bits)))
        assert i is not None, dict(zip(GATES, bits))
        reached[i] += 1
    assert reached == ROW_COUNTS
    assert sum(reached) == 16384
    assert {k for row in rows for k in row["when"]} == set(GATES), "a gate no row names is decoration"
    T = dict(zip(GATES, [True] * len(GATES)))
    v = lambda **kw: VERDICTS[_walk(rows, dict(T, **kw))]  # noqa: E731
    assert v(G_A1_LOW=False, G_A1_HIGH=False) == "KILL__no_tests_pass_accuser__A1_STANDS"
    # the instrument's failures come before any rate
    assert v(G_N=False, G_A1_V=False).startswith("COUNTS_ONLY__a2_under_400__A1_VOID")
    assert v(G_KEYS=False, G_A1_LOW=False, G_A1_HIGH=False).startswith("INVALID__a2_keys_not_sealed_in_order")
    assert v(G_P2_H=False, G_A1_LOW=False, G_A1_HIGH=False).startswith("INVALID__a2_panel_void_outside_seat")
    # a human and a Claude seat alone are not the decided panel (the integrity review)
    assert v(G_P2_N=False, G_A1_LOW=False, G_A1_HIGH=False).startswith("INVALID__a2_one_model_family_counts_only")
    # the part line decides only when the whole-suite line is cleared, pooled and per agent
    assert v(G_PC=False, G_A1_LOW=False, G_A1_HIGH=False).startswith("PART_NOT_RARE__")
    assert v(G_W=False, G_PC=False, G_RN=False, G_A1_LOW=False, G_A1_HIGH=False).startswith("NO_ACCUSER__rule_underpowered")
    # the per-agent block: a pooled "rare" over an agent shown at or above 1 in 100 falls to rule R's rows
    assert v(G_AG=False, G_RN=False, G_A1_LOW=False, G_A1_HIGH=False).startswith("NO_ACCUSER__rule_underpowered")
    assert v(G_AG=False, G_RULE=False, G_A1_LOW=False, G_A1_HIGH=False).startswith("NO_ACCUSER__rule_below_floor")
    assert v(G_W=False, G_A1_LOW=False, G_A1_HIGH=False).startswith("FLOOR_MET__")
    # A1: worst case decides LOWER, strict decides HIGHER, item agreement decides STANDS, the rest is UNSETTLED
    assert v(G_A1_LOW=True).endswith("__A1_LOWER")
    assert v(G_A1_LOW=False, G_A1_HIGH=True).endswith("__A1_HIGHER")
    assert v(G_A1_LOW=False, G_A1_HIGH=False, G_A1_C=False).endswith("__A1_UNSETTLED")
    assert v(G_A1_LOW=False, G_A1_HIGH=False, G_A1_ADD=False).endswith("__A1_UNSETTLED")
    assert v(G_A1_V=False, G_A1_LOW=True).endswith("__A1_VOID")


A1 = {"A1_VOID": {"panel_valid": 0, "confirmed": -1, "added": -1, "k2025_strict": -1, "k2025_worst": -1},
      "A1_LOWER": {"panel_valid": 1, "confirmed": 22, "added": 3, "k2025_strict": 25, "k2025_worst": 30},
      "A1_HIGHER": {"panel_valid": 1, "confirmed": 41, "added": 16, "k2025_strict": 57, "k2025_worst": 60},
      "A1_STANDS": {"panel_valid": 1, "confirmed": 36, "added": 3, "k2025_strict": 39, "k2025_worst": 43},
      "A1_UNSETTLED": {"panel_valid": 1, "confirmed": 30, "added": 2, "k2025_strict": 32, "k2025_worst": 38}}


def _a2(n, keys, h, nq, kw, n_eff, ag, kpc, firings, correct):
    from styxx.mind import wilson
    return {"n_claimers_ci": n, "keys_in_order": keys, "human_qualified": h, "nonclaude_qualified": nq,
            "whole_wilson_upper_worst": wilson(kw, n_eff)[1], "agent_whole_wilson_lower_max": ag,
            "part_contradicted_wilson_upper": wilson(kpc, n_eff)[1],
            "rule_firings": firings, "rule_cp_lower": _cp_lower(correct, firings)}


def _a2_values():
    sent = {"n_claimers_ci": 350, "keys_in_order": -1, "human_qualified": -1, "nonclaude_qualified": -1,
            "whole_wilson_upper_worst": 1.0, "agent_whole_wilson_lower_max": 1.0,
            "part_contradicted_wilson_upper": 1.0, "rule_firings": -1, "rule_cp_lower": 0.0}
    return {"COUNTS_ONLY__a2_under_400": sent,
            "INVALID__a2_keys_not_sealed_in_order": _a2(1050, 0, 1, 1, 2, 870, 0.003, 3, 4, 2),
            "INVALID__a2_panel_void_outside_seat": _a2(1050, 1, 0, 1, 2, 870, 0.003, 3, 4, 2),
            "INVALID__a2_one_model_family_counts_only": _a2(1050, 1, 1, 0, 2, 870, 0.003, 3, 4, 2),
            "KILL__no_tests_pass_accuser": _a2(1050, 1, 1, 1, 2, 870, 0.003, 3, 3, 2),
            "PART_NOT_RARE__new_prereg_may_propose_part_accuser": _a2(1050, 1, 1, 1, 2, 870, 0.003, 22, 3, 2),
            "NO_ACCUSER__rule_underpowered": _a2(1050, 1, 1, 1, 8, 870, 0.004, 4, 9, 5),
            "NO_ACCUSER__rule_below_floor": _a2(1300, 1, 1, 1, 80, 1080, 0.06, 6, 64, 60),
            "FLOOR_MET__new_prereg_may_propose_accuser": _a2(1300, 1, 1, 1, 90, 1080, 0.07, 6, 62, 62)}


def test_experiment_scores_the_receipt_shape_to_every_verdict():
    exp = _experiment()
    assert exp.gates_sha256 == GATES_SHA256
    assert exp.vacuous_gates == [] and exp.undeclared_power_gates == []
    a2v = _a2_values()
    for want in VERDICTS:
        a2_name, a1_name = want.rsplit("__A1_", 1)
        result = {"result": {"a1": dict(A1["A1_" + a1_name]), "a2": dict(a2v[a2_name])}}
        assert exp.score(result).verdict == want
    blocked = {"result": {"a1": dict(A1["A1_STANDS"]),
                          "a2": dict(a2v["KILL__no_tests_pass_accuser"], agent_whole_wilson_lower_max=0.0123)}}
    assert exp.score(blocked).verdict == "NO_ACCUSER__rule_underpowered__A1_STANDS"
    sm = exp.score({"result": {"a1": dict(A1["A1_STANDS"]), "a2": dict(a2v["KILL__no_tests_pass_accuser"])}},
                   smoke=True)
    assert sm.verdict == "INVALID__smoke_plumbing_only" and len(sm.gates) == 0


def test_experiment_refuses_a_missing_metric():
    exp = _experiment()
    from styxx.protocol import GateSpecError
    r = {"result": {"a1": dict(A1["A1_STANDS"]), "a2": dict(_a2_values()["KILL__no_tests_pass_accuser"])}}
    del r["result"]["a2"]["whole_wilson_upper_worst"]
    with pytest.raises(GateSpecError):
        exp.score(r)


# -- the arithmetic the bars and section 9 quote ------------------------------------------------------------

def test_wilson_and_clopper_pearson_numbers_behind_the_bars():
    from styxx.mind import wilson

    def nmin(k):
        n = k + 1
        while wilson(k, n)[1] >= 0.01:
            n += 1
        return n

    assert [nmin(k) for k in range(7)] == [381, 563, 726, 878, 1025, 1166, 1305]
    assert wilson(3, 1000)[1] < 0.01 <= wilson(4, 1000)[1]                     # section 3.4: 3 at 1,000, not 4
    assert round(wilson(4, 1000)[1], 6) == 0.010240
    assert wilson(2, 870)[1] < 0.01 <= wilson(3, 870)[1]                       # n_eff 870 clears with 2
    assert wilson(16, 870)[1] < 0.03 <= wilson(17, 870)[1]                     # G_PC at n_eff 870
    assert wilson(19, 1000)[1] < 0.03 <= wilson(20, 1000)[1]                   # and at 1,000
    assert round(wilson(20, 1000)[1], 4) == 0.0307
    assert round(wilson(2, 644)[1], 6) == 0.011252                             # testreceipt's 2026 upper bound
    lo, hi = wilson(42, 481)
    assert round(lo, 6) == 0.065248 and round(hi, 6) == 0.115928              # its 2025 headline interval
    assert 31 / 481 < lo <= 32 / 481 and 55 / 481 <= hi < 56 / 481             # G_A1_LOW 31, G_A1_HIGH 56
    assert round(wilson(90, 100)[0], 4) == 0.8256                              # AU4 at 90 of 100
    assert 34 / 42 > 0.8                                                       # G_A1_C
    assert round(_cp_lower(59, 59), 4) == 0.9505 and round(_cp_lower(58, 58), 4) == 0.9497   # G_RN
    assert round(59 / 1764, 3) == 0.033                                        # section 9.3: rule R's floor


def test_power_numbers_section_9_3_prints():
    from styxx.mind import wilson

    def kmax(n):
        k = -1
        while wilson(k + 1, n)[1] < 0.01:
            k += 1
        return k

    def cdf(k, n, p):
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))

    n = 870
    got = [round(cdf(kmax(n), n, p), 3) for p in (0.001, 0.002, 0.003, 0.005, 0.0075, 0.01)]
    assert got == [0.942, 0.747, 0.516, 0.19, 0.042, 0.008]
    assert round(cdf(kmax(1050), 1050, 0.005), 2) == 0.40 and round(cdf(kmax(1292), 1292, 0.005), 2) == 0.37


# -- the lock table ------------------------------------------------------------------------------------------

def _fenced_after(text, head):
    i = text.index(head)
    a = text.index("```\n", i) + 4
    b = text.index("\n```", a)
    return text[a:b]


def _lock_row(text, starts):
    rows = [ln for ln in text.splitlines() if ln.startswith("| " + starts)]
    assert len(rows) == 1, (starts, rows)
    return rows[0]


def test_prompt_and_summary_digests_are_the_bytes_of_part_ii():
    text = _text()
    for head, row, size in (("### P12.1 Step 1", "the step-1 prompt", 1439), ("### P12.2 Step 2", "the step-2 prompt", 1734),
                            ("### P17.2 Test-runner failure summaries", "rule R's failure summaries", 280)):
        raw = _fenced_after(text, head).encode("utf-8")
        assert len(raw) == size, head
        assert hashlib.sha256(raw).hexdigest() in _lock_row(text, row), head
    import tpc_rule
    assert "\n".join(tpc_rule.SUMMARY_LINES) == _fenced_after(text, "### P17.2 Test-runner failure summaries")


def test_vendored_testreceipt_pins_are_the_lock_table_rows():
    import tpc_screen
    text = _text()
    for name, row in (("claims.py", "testreceipt's claim reader"), ("ci.py", "testreceipt's CI reader"),
                      ("reconcile.py", "testreceipt's reconcile"), ("LICENSE", "testreceipt's licence")):
        assert tpc_screen.VENDOR_PINS[name] in _lock_row(text, row)
    assert tpc_screen.vendor_digests() == tpc_screen.VENDOR_PINS


def test_the_forecast_is_the_sealed_bytes_and_covers_the_rows():
    raw = FORECAST.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    text = _text()
    assert digest in _lock_row(text, "the forecast")
    assert text.count(digest) >= 3  # the header, section 13 and the lock table
    assert f"{len(raw):,} bytes" in text
    assert b"\r" not in raw and not raw.startswith(b"\xef\xbb\xbf")
    ft = raw.decode("utf-8")
    sec1 = ft.split("## 1.")[1].split("## 2.")[0]
    sec2 = ft.split("## 2.")[1].split("## 3.")[0]
    rows1 = re.findall(r"^\| `([A-Za-z0-9_]+)` \| ([0-9.]+) \|$", sec1, re.M)
    rows2 = re.findall(r"^\| `([A-Za-z0-9_]+)` \| ([0-9.]+) \|$", sec2, re.M)
    assert [r[0] for r in rows1] == A2_HALVES
    assert [r[0] for r in rows2] == A1_HALVES
    assert abs(sum(float(r[1]) for r in rows1) - 1) < 1e-9 and abs(sum(float(r[1]) for r in rows2) - 1) < 1e-9
    assert "0.8889" in ft and "0.8000" in ft and "60%" in ft  # the reference scores and the miss rule


def test_the_lock_rows_left_for_the_signing_commit_are_the_collector_and_p2s_decoys():
    text = _text()
    to_fill = [ln for ln in text.splitlines() if ln.startswith("| ") and "TO-FILL at the signing commit" in ln]
    assert len(to_fill) == 2
    assert to_fill[0].startswith("| the collector") and to_fill[1].startswith("| decoy set digest of packet P2")
    import tpc_seal  # the guard requires exactly these rows to be filled at the signing commit
    assert all(sum(r.startswith(p) for r in to_fill) == 1 for p in tpc_seal.SIGNING_ROWS)
    assert all(r.startswith(tpc_seal.SIGNING_ROWS) for r in to_fill)


# -- the seal guard against this file's own text ------------------------------------------------------------

def _guard(tmp_path, blob, code):
    import tpc_seal
    from tpc_const import UTC
    sealed = tmp_path / "SEALED"
    sealed.write_bytes(b"prereg_commit: " + b"a" * 40 + b"\nfreeze_anchor: 2026-11-06T12:00:00Z\n")
    return tpc_seal.SealGuard(sealed_file=sealed, repo=tmp_path, now=lambda: dt.datetime(2026, 11, 8, tzinfo=UTC),
                              blob_reader=lambda r, c, p: blob, fetch=lambda r: "ok", on_main=lambda r, c: True,
                              running_code=lambda: code, signing_commit=lambda r, p: "a" * 40,
                              main_blob=lambda r, p: blob)


def test_the_seal_guard_refuses_this_draft_and_opens_on_a_signed_filled_copy(tmp_path):
    """The guard reads the PREREG's own markers: this draft is unsigned and has rows TO-FILL at the signing commit; a
    copy signed in the house form, with those rows filled and the collector row holding the running code's digest,
    opens it. Nothing else of the file is read by the guard."""
    import tpc_seal
    raw = PREREG.read_bytes()
    code = tpc_seal.code_digests(COLLECT)["combined"]
    shut = _guard(tmp_path, raw, code)
    assert not shut.open and "no positive" in shut.state["why"]
    text = raw.decode("utf-8")
    assert text.count("**Signature.** Unsigned.") == 1
    signed = text.replace("**Signature.** Unsigned.", "**Signature.** Signed by the operator: \"go\" (synthetic).")
    lines = []
    for ln in signed.split("\n"):
        if ln.startswith("| the collector") and "TO-FILL at the signing commit" in ln:
            ln = ln.replace("TO-FILL at the signing commit", "`" + code + "`")
        elif ln.startswith("| decoy set digest of packet P2"):
            ln = ln.replace("TO-FILL at the signing commit", "`" + "d" * 64 + "`")
        lines.append(ln)
    signed = "\n".join(lines)
    half = signed.replace("`" + "d" * 64 + "`", "TO-FILL at the signing commit")
    assert not _guard(tmp_path, half.encode("utf-8"), code).open
    assert not _guard(tmp_path, signed.encode("utf-8"), "e" * 64).open
    g = _guard(tmp_path, signed.encode("utf-8"), code)
    assert g.open, g.state
    assert tpc_seal.collector_row_digests(signed) == [code]


# -- the folder's bytes and words ----------------------------------------------------------------------------

def _folder_files():
    return sorted(p for p in FOLDER.rglob("*") if p.is_file() and "__pycache__" not in p.parts)


def test_the_folder_is_byte_pinned_lf_and_without_a_bom():
    attrs = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "papers/tests-pass-ci/** -text" in attrs.splitlines()
    for p in _folder_files():
        raw = p.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf"), p
        assert b"\r\n" not in raw, f"{p} holds CRLF: the lock table's digests would differ by checkout"


def test_the_lab_never_prints_these_words():
    src = ROOT / "scripts" / "zenodo_publish_v7_48_0.py"
    if not src.exists():
        pytest.skip("the lab's BANNED tuple is not in this checkout")
    m = re.search(r"^BANNED = \((.*?)\)$", src.read_text(encoding="utf-8"), re.S | re.M)
    assert m, "BANNED tuple not found"
    words = re.findall(r'"([^"]+)"', m.group(1))
    assert len(words) >= 8
    banned = re.compile(r"\b(" + "|".join(re.escape(w) + "s?" for w in words) + r")\b", re.I)
    # GitHub's GraphQL paging argument is a banned word used as a name, not prose; the word is stored reversed here
    graphql_page_arg = re.compile(r"\b" + "tsrif"[::-1] + r": (?:%d|\d+)")
    # the vendored testreceipt files are a stranger's bytes, pinned unchanged (section 14): not the lab's words
    files = [p for p in _folder_files() if p.suffix in (".md", ".py") and "vendor" not in p.parts]
    files += [TEST_FILE, ROOT / "tests" / "_tests_pass_ci_fakes.py"]
    files += sorted((ROOT / "tests").glob("test_tests_pass_ci_collector_*.py"))
    for p in files:
        body = graphql_page_arg.sub("", p.read_bytes().decode("utf-8"))
        hits = [h.group(0) for h in banned.finditer(body)]
        assert not hits, f"{p.name}: {hits}"


def test_the_surface_lint_finds_nothing_in_the_folder_documents():
    lint = ROOT / "scripts" / "surface_lint.py"
    if not lint.exists():
        pytest.skip("scripts/surface_lint.py is not in this checkout")
    docs = [PREREG, FORECAST, README, COLLECT / "README.md"]
    r = subprocess.run([sys.executable, str(lint), "--files"] + [str(p) for p in docs] + ["--json"],
                       capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    report = json.loads(r.stdout)
    assert report["files_read"] == len(docs)
    assert report["hits"] == [], report["hits"]
