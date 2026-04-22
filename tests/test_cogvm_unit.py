# -*- coding: utf-8 -*-
"""Unit tests for styxx.cogvm — parser, opcode types, structural
validation. Does NOT require a GPU or any model download. Covers
behavior that must hold independent of the end-to-end steering
demo in benchmarks/cogvm_demo/."""

from __future__ import annotations

import pytest

from styxx.cogvm import (
    Program, WRITE, GENERATE, WATCH, HALT, RETRY, SWITCH,
    ProgramResult, _compile_predicate,
)


# ══════════════════════════════════════════════════════════════════
# Predicate parser — string form
# ══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize(
    "pred,state,expected", [
        ("confab > 0.7",     {"confab": 0.71},    True),
        ("confab > 0.7",     {"confab": 0.70},    False),
        ("confab >= 0.7",    {"confab": 0.70},    True),
        ("confab < 0.5",     {"confab": 0.49},    True),
        ("confab < 0.5",     {"confab": 0.50},    False),
        ("confab <= 0.5",    {"confab": 0.50},    True),
        ("refuse == 0.5",    {"refuse": 0.5},     True),
        ("refuse != 0.5",    {"refuse": 0.4},     True),
        # Negative floats
        ("x > -1.0",         {"x": -0.5},         True),
        ("x > -1.0",         {"x": -2.0},         False),
    ],
)
def test_predicate_parses_and_evaluates(pred, state, expected):
    fn = _compile_predicate(pred)
    assert fn(state) is expected


def test_predicate_missing_key_returns_false():
    fn = _compile_predicate("confab > 0.7")
    # Key not in state -> predicate is defined as False, not raise.
    assert fn({"something_else": 0.9}) is False


def test_predicate_garbage_raises():
    with pytest.raises(ValueError):
        _compile_predicate("not a predicate")
    with pytest.raises(ValueError):
        _compile_predicate("confab ?? 0.5")


def test_predicate_callable_passes_through():
    # Caller may hand a lambda instead of a string; that's allowed.
    fn = _compile_predicate(lambda state: state.get("foo", 0) > 100)
    assert fn({"foo": 101}) is True
    assert fn({"foo": 99}) is False


def test_predicate_records_structure_for_debug():
    """The compiled closure gets __styxx_predicate__ metadata so the
    runtime can display which task+op+number fired."""
    fn = _compile_predicate("sycophant_pressure >= 0.3")
    info = getattr(fn, "__styxx_predicate__", None)
    assert info is not None
    assert info == ("sycophant_pressure", ">=", 0.3)


# ══════════════════════════════════════════════════════════════════
# Opcode / program structural checks
# ══════════════════════════════════════════════════════════════════

def test_program_dataclass_shapes():
    prog = Program(ops=[
        WRITE({"comply_refuse": -2.0}),
        GENERATE(
            max_new_tokens=30,
            watches=[WATCH("confab > 0.7", HALT())]
        ),
    ])
    assert isinstance(prog.ops[0], WRITE)
    assert isinstance(prog.ops[1], GENERATE)
    assert isinstance(prog.ops[1].watches[0], WATCH)
    assert isinstance(prog.ops[1].watches[0].action, HALT)
    # Default max_retries
    assert prog.max_retries == 3


def test_generate_watch_actions_are_distinct_types():
    """HALT, RETRY, SWITCH are all distinct action types the runtime
    must dispatch on. They are NOT interchangeable."""
    assert HALT is not RETRY
    assert HALT is not SWITCH
    assert RETRY is not SWITCH
    # Retry has a profile field; others differ in shape
    r = RETRY(profile={"confab": -2.0})
    assert r.profile == {"confab": -2.0}
    s = SWITCH(profile={"confab": 0.0})
    assert s.profile == {"confab": 0.0}


def test_program_result_serializable():
    r = ProgramResult(
        output_text="I don't have information about that.",
        output_tokens=9,
        halt_reason="HALT: confab_prompt > 0.7 (=0.78) at token 12",
        retries_used=0,
        trace=["WRITE {'comply_refuse': 0.0}",
               "GENERATE max_new=90, 1 watches",
               "WATCH fired: ... -> HALT"],
        final_profile={"comply_refuse": 0.0},
        probe_readings_last={"comply_refuse": 0.33, "confab_prompt": 0.78},
    )
    d = r.as_dict()
    assert d["output_tokens"] == 9
    assert d["retries_used"] == 0
    assert d["probe_readings_last"]["confab_prompt"] == 0.78
    assert isinstance(d["trace"], list)


# ══════════════════════════════════════════════════════════════════
# Import-time surface — public API is stable
# ══════════════════════════════════════════════════════════════════

def test_public_api_exported():
    from styxx import cogvm
    for name in ("Program", "ProgramResult",
                 "WRITE", "GENERATE", "WATCH",
                 "HALT", "RETRY", "SWITCH"):
        assert name in cogvm.__all__, f"missing from __all__: {name}"
        assert hasattr(cogvm, name), f"missing attribute: {name}"
