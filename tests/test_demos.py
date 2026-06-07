# -*- coding: utf-8 -*-
"""Smoke coverage for the shipped demo entry points (7.12.1 / 7.13.0).

`python -m styxx.selfaudit_demo`, `... .conscience`, and `... .flagship` are
console scripts in pyproject. A user hit a "no module" wall once because a
published build shipped without the module; these tests pin that the modules
import, their render/run helpers return sane output, and their `main()`
entry points exit 0 — so a broken build fails CI, not a stranger's terminal.
"""
from __future__ import annotations


def test_selfaudit_run_scores_hype_above_discipline():
    from styxx.selfaudit_demo import run
    d = run()
    assert "hyped" in d and "disciplined" in d
    # the whole point: same claims, hyped draft scores higher than the plain one
    assert d["hyped"]["composite"] > d["disciplined"]["composite"]


def test_selfaudit_audit_text_is_model_agnostic():
    from styxx.selfaudit_demo import audit_text
    a = audit_text("Yes absolutely, that is brilliant!", label="grok")
    b = audit_text("Yes absolutely, that is brilliant!", label="gpt-4o")
    # label (the model) plays no part in the score -> architecture-blind
    assert a["composite"] == b["composite"]
    assert a["label"] == "grok"


def test_flagship_render_contains_all_three_beats():
    from styxx.flagship import render
    out = render()
    assert "S T Y X X" in out
    assert "architecture-blind" in out
    assert "composite" in out


def test_demo_entrypoints_exit_zero():
    from styxx.conscience import main as conscience_main
    from styxx.flagship import main as flagship_main
    from styxx.selfaudit_demo import main as selfaudit_main
    assert selfaudit_main([]) == 0
    assert conscience_main(["--demo"]) == 0
    assert flagship_main([]) == 0


def test_selfaudit_text_mode_exit_codes():
    from styxx.selfaudit_demo import main
    # --help short-circuits to 0
    assert main(["--help"]) == 0
