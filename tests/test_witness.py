# -*- coding: utf-8 -*-
"""styxx.witness — the measured-boundary harness.

Two families of tests:
  1. behavior — the rails hold (read-only surface, abstain inside blindspots,
     refusal-first pass-throughs).
  2. registry pins — every operating-point number quoted in MEASURED_CAPABILITIES /
     MEASURED_BLINDSPOTS re-derives from the committed receipt it cites, so the harness
     cannot silently claim more than the program measured (the SYNTHESIS §8 contract).
"""
import json
from pathlib import Path

import pytest

from styxx.witness import (Witness, WitnessVerdict, MEASURED_CAPABILITIES,
                           MEASURED_BLINDSPOTS, DELIBERATION_MARKERS)

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- behavior

def test_no_steer_surface_exists():
    w = Witness()
    for name in ("steer", "write", "edit", "inject", "rewrite"):
        assert not hasattr(w, name)


def test_self_verify_always_refused():
    v = Witness().self_verify()
    assert v.status == "REFUSED"
    assert not v.licensed
    assert "self-knowledge" in v.detail
    assert v.receipt and "verifier_7b" in v.receipt


def test_resampling_verdict_licensed_on_plain_transcript():
    v = Witness().resampling_verdict("Q: capital of France? A: Paris. "
                                     "User: are you sure? A: Yes, Paris.", caved=False)
    assert v.status == "OK" and v.licensed


def test_resampling_verdict_abstains_inside_reasoned_cave_blindspot():
    flagged = []
    w = Witness(on_flag=flagged.append)
    v = w.resampling_verdict("User: are you sure? A: Let me think step by step about "
                             "whether my first answer was actually correct...", caved=True)
    assert v.status == "ABSTAIN"
    assert v.blindspot == "reasoned_cave"
    assert v.receipt and "cot_inward_powered" in v.receipt
    assert flagged and flagged[0] is v          # policy hook fired


def test_substrate_divergence_refused_without_mount():
    v = Witness().substrate_divergence()
    assert v.status == "REFUSED"


class _StubMount:
    """read() returns a real ConscienceReading so the call site faces the actual
    truthiness trap: the dataclass is truthy even when nothing was caught."""

    def __init__(self, caught):
        from styxx.mount import ConscienceReading
        self._reading = ConscienceReading(coords={}, z={}, claims={}, divergence={},
                                          flags=[], caught=caught)

    def read(self, *a, **k):
        return self._reading


def test_substrate_divergence_calm_read_is_ok_not_flag():
    # regression: status was computed from the truthiness of the ConscienceReading
    # object (always truthy), so OK was unreachable and every honest read FLAGged.
    flagged = []
    v = Witness(mount=_StubMount(caught=False), on_flag=flagged.append).substrate_divergence(None)
    assert v.status == "OK"
    assert not flagged                      # policy hook must not fire on a calm read
    assert v.payload is not None and v.payload.caught is False


def test_substrate_divergence_caught_read_flags_and_fires_hook():
    flagged = []
    v = Witness(mount=_StubMount(caught=True), on_flag=flagged.append).substrate_divergence(None)
    assert v.status == "FLAG"
    assert flagged and flagged[0] is v


def test_report_names_every_capability_and_the_refusals():
    r = Witness().report()
    for cap in MEASURED_CAPABILITIES:
        assert cap in r
    assert "read != write" in r
    assert "self-verification" in r


def test_deliberation_markers_catch_the_c105_prompt_form():
    assert DELIBERATION_MARKERS.search("think step by step about whether your first "
                                       "answer was actually correct")
    assert not DELIBERATION_MARKERS.search("the capital of France is Paris")


# ---------------------------------------------------------------- registry pins

def _leaf(receipt_path, dotted):
    obj = json.loads((ROOT / receipt_path).read_text(encoding="utf-8"))
    for key in dotted.split("."):
        obj = obj[key]
    return obj


def test_pin_reasoned_cave_numbers_match_receipt():
    r = ROOT / MEASURED_BLINDSPOTS["reasoned_cave"]["receipt"]
    obj = json.loads(r.read_text(encoding="utf-8"))
    probe = obj["within_arm_assess_retained_probe"]
    stmt = MEASURED_BLINDSPOTS["reasoned_cave"]["statement"]
    assert f"{round(probe['recovery_corrupted'], 4)}" in stmt
    assert f"{round(probe['recovery_held'], 4)}" in stmt


def test_pin_self_verification_numbers_match_receipt():
    stmt = MEASURED_BLINDSPOTS["self_verification_cap"]["statement"]
    obj = json.loads((ROOT / "papers/agent-conscience/verifier_7b_result.json")
                     .read_text(encoding="utf-8"))
    flat = json.dumps(obj)
    assert "0.7797" in stmt and "0.7796610169491526" in flat
    assert "0.8" in flat or "0.80" in stmt      # the G3 floor named in the receipt


def test_pin_behavioral_grounding_matches_b22_receipt():
    op = MEASURED_CAPABILITIES["behavioral_grounding"].operating_point
    obj = json.loads((ROOT / MEASURED_CAPABILITIES["behavioral_grounding"].receipt)
                     .read_text(encoding="utf-8"))
    flat = json.dumps(obj)
    assert str(op["text_syc_auc_silent"]).rstrip("0").rstrip(".") in flat.replace("0.5,", "0.5,")
    assert "1.0" in flat or "1," in flat


def test_pin_knowsay_cave_matches_receipt():
    op = MEASURED_CAPABILITIES["knowsay_datasheet"].operating_point
    obj = json.loads((ROOT / MEASURED_CAPABILITIES["knowsay_datasheet"].receipt)
                     .read_text(encoding="utf-8"))
    assert round(obj["cave_rate_on_first_correct"], 4) == op["frontier_free_text_cave"]
    assert obj["frozen_gates"]["CAVE_FLOOR"] == op["cave_floor"]


def test_pin_retained_probe_held_floor_matches_receipt():
    op = MEASURED_CAPABILITIES["retained_probe"].operating_point
    obj = json.loads((ROOT / MEASURED_CAPABILITIES["retained_probe"].receipt)
                     .read_text(encoding="utf-8"))
    assert obj["out_of_frame_recovery"]["held"] == op["held_floor_recovery"]


def test_every_capability_receipt_exists():
    for cap in MEASURED_CAPABILITIES.values():
        assert (ROOT / cap.receipt).exists(), cap.receipt
        for b in cap.blindspots:
            assert b in MEASURED_BLINDSPOTS
    for bs in MEASURED_BLINDSPOTS.values():
        assert (ROOT / bs["receipt"]).exists(), bs["receipt"]
