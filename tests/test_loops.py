# -*- coding: utf-8 -*-
"""styxx.loops — characterized against the loop it was built from.

Ground truth: the `outcome` contamination found on 2026-08-19. `write_audit`
derived `outcome` (ground truth) from `gate` (the classifier's own verdict), and
six consumers read it believing it. The corpus below is the real shape of both
halves, inlined so the characterization does not depend on clone depth.
"""
import pytest

from styxx.loops import LIMITS, scan_path, scan_source

# ── the derivation: a field computed from the system's own verdict ──────────

WRITE_AUDIT = '''
def write_audit(entry):
    if entry.get("outcome") is None and config.auto_feedback_enabled():
        gate = entry.get("gate")
        if gate == "pass":
            entry["outcome"] = "correct"
        elif gate in ("warn", "fail"):
            entry["outcome"] = "incorrect"
'''

# ── consumers that trusted it, by their real names ─────────────────────────

CALIBRATE = '''
def _load_labeled_entries():
    entries = load_audit(last_n=5000)
    return [e for e in entries if e.get("outcome") in ("correct", "incorrect")]
'''

LEARNED_CLASSIFIER = '''
def _load_training_data():
    for e in load_audit(last_n=5000):
        if e.get("outcome") != "correct":
            continue
        texts.append(e.get("prompt"))
'''

# ── the same consumer AFTER the fix: it consults provenance ────────────────

CALIBRATE_GUARDED = '''
def _load_labeled_entries():
    entries = load_audit(last_n=5000)
    labeled = [e for e in entries if e.get("outcome") in ("correct", "incorrect")]
    usable = [e for e in labeled if e.get("outcome_source") != "auto"]
    return usable
'''


def _report(*sources):
    """Assemble a LoopReport from several inlined modules."""
    from styxx.loops import LoopReport
    rep = LoopReport()
    for i, src in enumerate(sources):
        v = scan_source(src, f"mod{i}.py")
        rep.derivations.extend(v.derivations)
        rep.trust_sites.extend(v.trust)
        rep.files_scanned += 1
    return rep


def test_finds_the_outcome_loop_it_was_built_from():
    rep = _report(WRITE_AUDIT, CALIBRATE, LEARNED_CLASSIFIER)
    loops = rep.loops()
    assert "outcome" in loops, "the field the whole module exists for"

    info = loops["outcome"]
    assert any(d.source_field == "gate" for d in info["derivations"]), \
        "ground truth derived from the classifier's own verdict"
    fns = {s.function for s in info["trust_sites"]}
    assert {"_load_labeled_entries", "_load_training_data"} <= fns
    assert info["n_high_stakes"] >= 2, "calibrating and training are high-stakes"


def test_a_provenance_guard_clears_the_consumer():
    """A codebase that HAS fixed the loop must stop being flagged, or the screen
    gets ignored. The guard is credited function-wide, which over-credits —
    the direction that loses findings rather than inventing them."""
    unguarded = _report(WRITE_AUDIT, CALIBRATE).loops()["outcome"]
    guarded = _report(WRITE_AUDIT, CALIBRATE_GUARDED).loops()["outcome"]

    # the consumer clears once it consults provenance
    assert guarded["n_guarded"] >= 1
    assert guarded["n_high_stakes"] < unguarded["n_high_stakes"]

    def _fn_flags(info, name):
        return next((s.guarded, s.high_stakes) for s in info["trust_sites"]
                    if s.function == name)
    assert _fn_flags(unguarded, "_load_labeled_entries") == (False, True)
    assert _fn_flags(guarded, "_load_labeled_entries") == (True, True)

    # KNOWN false positive, pinned rather than asserted away: write_audit reads
    # `outcome` only to check whether one is already set before stamping. That
    # is a presence check, not trust in the value -- but the rule counts any
    # comparison on the field, and "audit" is in the high-stakes vocabulary. It
    # stays flagged in both reports.
    assert _fn_flags(guarded, "write_audit") == (False, True)


def test_a_field_nobody_trusts_is_not_a_loop():
    """Derivation alone is not a defect — a cached total or a display string is
    derived and harmless. It takes a consumer to close the loop."""
    src = '''
def write(entry):
    if entry.get("gate") == "pass":
        entry["banner_text"] = "all clear"
'''
    assert "banner_text" not in _report(src).loops()


def test_derivation_requires_a_sibling_field_not_just_any_condition():
    """`rec[F] = ...` under a condition on something OTHER than the record is
    an ordinary write, not self-derivation."""
    src = '''
def write(entry, user_said_ok):
    if user_said_ok:
        entry["outcome"] = "correct"

def read(e):
    if e.get("outcome") == "correct":
        return True
'''
    assert "outcome" not in _report(src).loops()


def test_report_states_its_limits_and_never_calls_nothing_clean(tmp_path):
    rep = scan_path(__file__)
    assert rep.measured is True
    assert LIMITS in rep.render()
    assert rep.as_dict()["limits"] == LIMITS

    empty = scan_path(tmp_path)          # a directory with no python in it
    assert empty.measured is False
    assert "SCANNED NOTHING" in empty.render()
    assert "SCANNED NOTHING" in repr(empty)


def test_cli_exits_zero_even_with_findings(tmp_path, capsys):
    """A screen that can fail a build is a screen someone silences."""
    from styxx.loops import main
    (tmp_path / "m.py").write_text(WRITE_AUDIT + CALIBRATE, encoding="utf-8")
    assert main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "outcome" in out and "derived" in out


def test_json_round_trips():
    import json
    rep = _report(WRITE_AUDIT, CALIBRATE)
    payload = rep.as_dict()
    assert json.dumps(payload)
    assert payload["loops"]["outcome"]["derivations"][0]["derived_from"] == "gate"
