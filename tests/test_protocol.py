# -*- coding: utf-8 -*-
"""styxx.protocol — the research loop as enforceable machinery.

The tests exercise the refusals, because the refusals ARE the product: an uncommitted
prereg cannot score, a missing gates block cannot score, a bar cannot be supplied at
scoring time, a smoke cannot look good, and a non-total outcome table surfaces as a
design bug instead of a guess.
"""
import json
import subprocess

import pytest

from styxx.protocol import Experiment, PrologueError, GateSpecError

GATES_BLOCK = """
# PREREG — synthetic

## Gates

```gates
{"gates": {"G0": {"metric": "llama_top1", "op": ">=", "value": 0.29},
           "G1": {"metric": "cells.gemma_top1", "op": ">=", "value": 0.143}},
 "outcomes": [{"when": {"G0": false}, "verdict": "INVALID__pipeline_broken"},
              {"when": {"G0": true, "G1": true}, "verdict": "DOOR_OPENS"},
              {"when": {"G0": true, "G1": false}, "verdict": "CLOSED_NEGATIVE"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```
"""


def _git_repo_with_prereg(tmp_path, text=GATES_BLOCK, commit=True):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    p = tmp_path / "PREREG_synthetic.md"
    p.write_text(text, encoding="utf-8")
    if commit:
        subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-qm", "freeze"], cwd=tmp_path, check=True)
    return p


def test_uncommitted_prereg_refuses_to_score(tmp_path):
    p = _git_repo_with_prereg(tmp_path, commit=False)
    with pytest.raises(PrologueError, match="draft"):
        Experiment(p)


def test_missing_gates_block_refuses(tmp_path):
    p = _git_repo_with_prereg(tmp_path, text="# PREREG with no gates\n")
    with pytest.raises(GateSpecError, match="gates block"):
        Experiment(p)


def test_verdict_walks_the_frozen_table(tmp_path):
    e = Experiment(_git_repo_with_prereg(tmp_path))
    v = e.score({"llama_top1": 0.80, "cells": {"gemma_top1": 0.7857}})
    assert v.verdict == "DOOR_OPENS" and v.gates == {"G0": True, "G1": True}
    v = e.score({"llama_top1": 0.20, "cells": {"gemma_top1": 0.7857}})
    assert v.verdict == "INVALID__pipeline_broken"      # a glowing cell cannot rescue G0
    v = e.score({"llama_top1": 0.80, "cells": {"gemma_top1": 0.01}})
    assert v.verdict == "CLOSED_NEGATIVE"


def test_no_api_exists_to_pass_a_bar_at_scoring_time(tmp_path):
    e = Experiment(_git_repo_with_prereg(tmp_path))
    with pytest.raises(TypeError):
        e.score({"llama_top1": 0.28, "cells": {"gemma_top1": 0.9}}, g0_floor=0.20)


def test_smoke_is_invalid_by_type_even_with_perfect_numbers(tmp_path):
    e = Experiment(_git_repo_with_prereg(tmp_path))
    v = e.score({"llama_top1": 1.0, "cells": {"gemma_top1": 1.0}}, smoke=True)
    assert v.verdict == "INVALID__smoke_plumbing_only" and v.smoke


def test_partial_outcome_table_surfaces_as_design_bug(tmp_path):
    spec = {"gates": {"G0": {"metric": "llama_top1", "op": ">=", "value": 0.29}},
            "outcomes": [{"when": {"G0": True}, "verdict": "OK"}],   # no G0-false row
            "smoke_verdict": "INVALID__smoke_plumbing_only"}
    text = "# PREREG partial\n\n```gates\n" + json.dumps(spec) + "\n```\n"
    p = _git_repo_with_prereg(tmp_path, text=text)
    e = Experiment(p)
    with pytest.raises(GateSpecError, match="not total"):
        e.score({"llama_top1": 0.10})


def test_gates_hash_travels_with_the_verdict(tmp_path):
    e = Experiment(_git_repo_with_prereg(tmp_path))
    v = e.score({"llama_top1": 0.80, "cells": {"gemma_top1": 0.7857}})
    assert len(v.gates_sha256) == 64 and len(v.prereg_commit) == 40


def test_missing_metric_refuses_rather_than_guesses(tmp_path):
    e = Experiment(_git_repo_with_prereg(tmp_path))
    with pytest.raises(GateSpecError, match="not present"):
        e.score({"cells": {"gemma_top1": 0.7857}})
