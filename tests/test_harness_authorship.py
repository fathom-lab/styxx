# -*- coding: utf-8 -*-
"""SWALLOW-8's instrument: every workflow-touching commit put in one of three classes -- agent,
automation, human -- by a stated rule on its author, subject and body, and joined to the
SWALLOW-7 gate's records without writing a name. The tests hold the rule on shaped records, the
join on the scripted history, and determinism."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("yaml")
from benchmarks.harness_mutation import authorship as A  # noqa: E402
from benchmarks.harness_mutation import differential as D  # noqa: E402
from tests.test_harness_history import _repo  # noqa: E402

# frozen at the sha256 the SWALLOW-8 receipt names (papers/harness/swallow8_receipt.json.gz); a change needs a new receipt, not a new pin
INSTRUMENT_SHA256 = "c3fb6e422a5d068df9f57ac3d0aac641b4c140337f09a28a90e8688a677d06ab"


def test_the_instrument_is_the_one_the_receipt_names():
    import hashlib
    assert hashlib.sha256((ROOT / "benchmarks" / "harness_mutation" / "authorship.py").read_bytes()).hexdigest() == INSTRUMENT_SHA256


def test_the_rule_on_shaped_records():
    c = A.classify
    # an agent's own name, address, trailer, banner or branch
    assert c("Copilot", "1+Copilot@users.noreply.github.com", "[CI] Agentic workflows: Update gh-aw generated assets", "") == ("agent", "author-name")
    assert c("devin-ai-integration[bot]", "1+devin-ai-integration[bot]@users.noreply.github.com", "Add tests", "") == ("agent", "author-name")
    assert c("Jane", "j@example.org", "Improve docs", "Co-authored-by: Copilot <1+Copilot@users.noreply.github.com>") == ("agent", "co-authored-by")
    assert c("Jane", "j@example.org", "Add feature", "Some text\n\nCo-Authored-By: Claude Opus 4 <noreply@anthropic.com>") == ("agent", "co-authored-by")
    assert c("Jane", "j@example.org", "Add feature", "🤖 Generated with [Claude Code](https://claude.com/claude-code)") == ("agent", "generated-with")
    assert c("georgi", "g@example.org", "Merge pull request #2459 from org/claude/add-nodejs-code-bots-9e046", "") == ("agent", "branch-prefix")
    assert c("Org", "o@example.org", "Merge pull request #3212 from Org/codex/modify-browserslist-database-workflow", "body") == ("agent", "branch-prefix")
    # a person's first name is not a signal
    assert c("Claude Dupont", "cd@example.org", "fix ci", "") == ("human", None)
    assert c("Devin Smith", "d@example.org", "ci: tweak", "") == ("human", None)
    # bots that are not coding agents
    assert c("dependabot[bot]", "1+dependabot[bot]@users.noreply.github.com", "Bump actions/checkout from 3 to 4", "") == ("automation", "bot-author")
    assert c("github-actions[bot]", "1+github-actions[bot]@users.noreply.github.com", "chore: release", "") == ("automation", "bot-author")
    assert c("renovate[bot]", "x@renovateapp.com", "Update dependency", "") == ("automation", "bot-author")
    # an agent signal wins over a bot author
    assert c("github-actions[bot]", "1+github-actions[bot]@users.noreply.github.com", "[CI] Agentic workflows: Update gh-aw generated assets", "") == ("agent", "generated-with")
    # everything else is human, including an agent-assisted commit that left no signature
    assert c("Ryan", "r@example.org", "feat(infra): Add backend selective testing workflow (#105500)", "Adds a shadow job") == ("human", None)
    assert c("", "", "", "") == ("human", None)


def test_the_join_on_the_scripted_history(tmp_path):
    tree = _repo(tmp_path)
    tip = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    rec7 = D.repo_differential(tree, tip, "fixture", sample_every=0)
    classes = A.commit_classes(tree, tip)
    assert len(classes) == 9 and all(v["class"] == "human" and v["signal"] is None for v in classes.values())
    joined = A.join_repo(rec7, classes)
    assert joined["unclassified"] == 0 and len(joined["commits"]) == 9
    assert [c["class"] for c in joined["commits"]] == ["human"] * 9
    assert [c["fires"] for c in joined["commits"]] == [True, True, False, False, False, True, False, False, False]
    assert set(joined["commits"][1]) >= {"sha", "time", "root", "fires", "new_hidden", "removed_hidden", "class", "signal", "checks"}
    assert joined["commits"][1]["checks"] == [{"kind": "acquired", "verdict": "SWALLOWED", "continue_on_error": True, "repair": "no-continue-on-error", "lines": 1}]
    assert joined["commits"][2].get("checks", []) == [] and "checks" not in joined["commits"][3]   # a removal has a detail but no new check; a quiet commit has none
    for c in joined["commits"]:                       # no name, no address, ever
        assert not any(k in c for k in ("author", "author_name", "author_email", "email", "name"))
    s = A.summary({"repos": [joined]})
    assert s["commits"] == 8 and s["unclassified"] == 0
    assert s["by_class"]["human"]["commits"] == 8 and s["by_class"]["human"]["firing_commits"] == 2 and s["by_class"]["human"]["new_hidden_checks"] == 2
    assert s["by_class"]["agent"]["commits"] == 0 and s["by_class"]["automation"]["commits"] == 0
    assert s["by_year"] == {"2023": {"agent": 0, "automation": 0, "human": 3, "firing": 1}, "2024": {"agent": 0, "automation": 0, "human": 5, "firing": 1}}   # the root (2023) is excluded


def test_the_instrument_is_deterministic(tmp_path):
    tree = _repo(tmp_path)
    tip = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert A.commit_classes(tree, tip) == A.commit_classes(tree, tip)
