"""A gate no outcome depends on is decoration wearing a bar's clothes.

Adopted from `honest-signal` (github.com/alexcard3/honest-signal), whose preregistration firewall
refuses a merge when the kill criterion is vacuous. The frozen prior-art survey
(`papers/closed-model-frontier/RESULT_oath_prior_art_survey_2026_08_26.md`) found that tool
occupying the enforcement mechanism this lab believed was its own, and this check is the half we
did not have — despite "a leg that cannot fail must not gate" appearing in our own
preregistrations, and despite the v0.11 drafting record naming a BLOCKER where "the warrant gate
as first drafted could not fail".

Corpus state when the check was added: **1 of 40** gated preregs trips it, and that one is
DISCLOSED rather than hidden — its `power_basis` reads "RECORDED, NOT GATED -- value 0 makes this
unfailable by construction and that is deliberate." So the finding is not a lie in the corpus; it
is a gap in the schema, which offers no way to say *compute and report this, but it is not a bar*.
An honest author put it in `gates` with an unfailable bar and explained why in prose.
"""
from __future__ import annotations

import json

import pytest

from styxx.protocol import Experiment, GateSpecError

pytest.importorskip("subprocess")

TOTAL_TABLE = [
    {"when": {"G0": False}, "verdict": "INVALID__plumbing"},
    {"when": {"G0": True, "G1": True}, "verdict": "PASS"},
    {"when": {"G0": True, "G1": False}, "verdict": "CLOSED_NEGATIVE"},
]


def _prereg(tmp_path, gates, outcomes, repo=True):
    """A committed prereg — `Experiment` refuses to score an uncommitted one, by design."""
    import subprocess
    d = tmp_path / "r"
    d.mkdir(exist_ok=True)
    if repo:
        subprocess.run(["git", "init", "-q"], cwd=d, check=True)
        subprocess.run(["git", "config", "user.email", "t@t"], cwd=d, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=d, check=True)
    body = {"gates": gates, "outcomes": outcomes, "smoke_verdict": "INVALID__smoke"}
    p = d / "PREREG_x.md"
    p.write_text("# t\n\n```gates\n" + json.dumps(body) + "\n```\n", encoding="utf-8")
    if repo:
        subprocess.run(["git", "add", "-A"], cwd=d, check=True)
        subprocess.run(["git", "commit", "-qm", "freeze"], cwd=d, check=True)
    return p


def test_a_gate_no_outcome_mentions_is_vacuous(tmp_path):
    gates = {"G0": {"metric": "a", "op": ">=", "value": 1},
             "G1": {"metric": "b", "op": ">=", "value": 1},
             "G2": {"metric": "c", "op": ">=", "value": 0}}   # named by no outcome row
    e = Experiment(_prereg(tmp_path, gates, TOTAL_TABLE))
    assert e.vacuous_gates == ["G2"]


def test_gates_every_outcome_depends_on_are_not_vacuous(tmp_path):
    gates = {"G0": {"metric": "a", "op": ">=", "value": 1},
             "G1": {"metric": "b", "op": ">=", "value": 1}}
    e = Experiment(_prereg(tmp_path, gates, TOTAL_TABLE))
    assert e.vacuous_gates == []


def test_it_reports_by_default_and_refuses_only_when_asked(tmp_path):
    """Same contract as `require_power_basis`: a new refusal must not retroactively invalidate
    40 frozen preregistrations, whose bars never move."""
    gates = {"G0": {"metric": "a", "op": ">=", "value": 1},
             "G1": {"metric": "b", "op": ">=", "value": 1},
             "G2": {"metric": "c", "op": ">=", "value": 0}}
    p = _prereg(tmp_path, gates, TOTAL_TABLE)
    assert Experiment(p).vacuous_gates == ["G2"]          # reported, not raised
    with pytest.raises(GateSpecError) as ex:
        Experiment(p, require_nonvacuous_gates=True)
    assert "G2" in str(ex.value)
    assert "must not gate" in str(ex.value)


def test_single_polarity_mention_is_deliberately_not_flagged(tmp_path):
    """A gate appearing once as true, with a wildcard row catching false, DOES decide the
    verdict. Flagging it would be a false positive, and the totality check already refuses the
    case where nothing catches the other branch."""
    gates = {"G0": {"metric": "a", "op": ">=", "value": 1}}
    outcomes = [{"when": {"G0": True}, "verdict": "PASS"},
                {"when": {}, "verdict": "FAIL"}]
    e = Experiment(_prereg(tmp_path, gates, outcomes))
    assert e.vacuous_gates == []


def test_the_verdict_carries_it(tmp_path):
    """A reader of the verdict must see the vacuity, not have to re-parse the prereg."""
    gates = {"G0": {"metric": "a", "op": ">=", "value": 1},
             "G1": {"metric": "b", "op": ">=", "value": 1},
             "G2": {"metric": "c", "op": ">=", "value": 0}}
    e = Experiment(_prereg(tmp_path, gates, TOTAL_TABLE))
    v = e.score({"a": 5, "b": 5, "c": 5})
    assert v.vacuous_gates == ["G2"]


def test_smoke_verdict_carries_it_too(tmp_path):
    gates = {"G0": {"metric": "a", "op": ">=", "value": 1},
             "G1": {"metric": "b", "op": ">=", "value": 1},
             "G2": {"metric": "c", "op": ">=", "value": 0}}
    e = Experiment(_prereg(tmp_path, gates, TOTAL_TABLE))
    v = e.score({"a": 5, "b": 5, "c": 5}, smoke=True)
    assert v.vacuous_gates == ["G2"]
