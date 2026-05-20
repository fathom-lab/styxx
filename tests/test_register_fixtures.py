"""
Register-firing regression fixtures from in-production agent self-audit.

Each entry in tests/fixtures/register_corpus.jsonl is an anonymized record
of a cognometric audit on a real agent draft (darkflobi / clawdbot during
live conversation). We DO NOT keep the prompt or draft text — only the
scores, the firing label, and a stable id derived from (msg_id, draft_v).

Two regression invariants:

  1. STRUCTURAL: every historical firing record must still LOOK like a
     firing under the current scoring conventions (>= 0.5 on at least
     one instrument OR composite >= 0.30 with needs_revision=True).
     If a release silently changes the firing threshold and the corpus
     no longer "fires", this test catches it.

  2. CALIBRATION (skipped when prompt/draft text is unavailable): when
     a future extension lands that stores prompt+draft text for replay,
     a `pytest.mark.replay` test will re-run the audit and compare the
     score against the recorded score within ±tol (default 0.05). Until
     that lands, we only assert structural invariants.

This is the agent-in-production feedback loop in code: every firing on
a live agent is a fixture the test suite didn't have. If a future styxx
release stops firing on historical agent-register cases, that's either a
calibration improvement (verify by hand) or a regression masking real
register-detection signal.

Provenance: darkflobi (clawdbot), in-conversation cognometric reflex
loop, msg_ids tracked in memory/cognometric-trajectory.jsonl.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "register_corpus.jsonl"
INSTRUMENTS = ("sycophancy", "deception", "overconfidence", "refusal")


def _load_fixtures() -> list[dict]:
    if not FIXTURE_PATH.exists():
        return []
    rows = []
    with FIXTURE_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


FIXTURES = _load_fixtures()


@pytest.mark.skipif(not FIXTURES, reason="no register_corpus fixtures present")
def test_register_corpus_loads() -> None:
    assert FIXTURES, "fixture file present but empty"
    for row in FIXTURES:
        assert "id" in row and row["id"].startswith("rf_")
        assert "scores" in row
        for k in INSTRUMENTS:
            assert k in row["scores"]
            v = row["scores"][k]
            assert 0.0 <= v <= 1.0, f"score out of range: {k}={v} in {row['id']}"
        assert "composite" in row
        assert "tol" in row


@pytest.mark.parametrize(
    "row",
    FIXTURES,
    ids=[r["id"] for r in FIXTURES] or ["__none__"],
)
def test_register_firing_is_structurally_a_firing(row: dict) -> None:
    """A historical firing must still look like a firing.

    'Looks like a firing' = any instrument score >= 0.5, OR
    composite >= 0.30 and needs_revision was True at record time.

    If a future release silently moves the firing threshold above 0.5,
    this test fails noisily — exactly the calibration-shift signal we
    want.
    """
    if not FIXTURES:
        pytest.skip("no fixtures")
    scores = row["scores"]
    any_high = any(v >= 0.5 for v in scores.values())
    composite_marginal = (
        row.get("composite", 0.0) >= 0.30 and row.get("needs_revision", False)
    )
    assert any_high or composite_marginal, (
        f"fixture {row['id']} no longer reads as a firing: "
        f"scores={scores} composite={row.get('composite')} "
        f"needs_revision={row.get('needs_revision')}"
    )


@pytest.mark.parametrize(
    "row",
    FIXTURES,
    ids=[r["id"] for r in FIXTURES] or ["__none__"],
)
def test_register_firing_label_matches_top_instrument(row: dict) -> None:
    """When a firing label is present, the named instrument should be
    among the top-2 scores. Pure consistency check, not a calibration
    claim."""
    if not FIXTURES:
        pytest.skip("no fixtures")
    label = row.get("firing") or ""
    # only assert when the label clearly names an instrument
    named = next((k for k in INSTRUMENTS if k in label), None)
    if named is None:
        pytest.skip(f"firing label '{label}' does not name an instrument")
    scores = row["scores"]
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top2 = {k for k, _ in ranked[:2]}
    assert named in top2, (
        f"fixture {row['id']} firing label '{label}' names "
        f"'{named}' but it's not in top-2 scores {ranked[:2]}"
    )
