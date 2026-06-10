"""Offline tests for styxx.token — the read-only $STYXX tier lookup. No network."""
from styxx.token import tier_for_balance, get_tier, TIERS, STYXX_MINT, Tier


def test_tier_thresholds():
    assert tier_for_balance(0).name == "Public"
    assert tier_for_balance(9_999).name == "Public"
    assert tier_for_balance(10_000).name == "Supporter"
    assert tier_for_balance(99_999).name == "Supporter"
    assert tier_for_balance(100_000).name == "Validator"
    assert tier_for_balance(120_000).name == "Validator"
    assert tier_for_balance(1_000_000).name == "Governor"
    assert tier_for_balance(10_000_000).name == "Core"
    assert tier_for_balance(50_000_000).name == "Core"


def test_tiers_monotonic_and_typed():
    levels = [t.level for t in TIERS]
    mins = [t.min_styxx for t in TIERS]
    assert levels == sorted(levels) == [0, 1, 2, 3, 4]
    assert mins == sorted(mins)
    assert all(isinstance(t, Tier) for t in TIERS)


def test_mint_constant():
    assert STYXX_MINT.endswith("pump") and len(STYXX_MINT) > 30


def test_get_tier_is_callable():
    # smoke: the live lookup exists and is wired to the pure mapping (no network call here)
    assert callable(get_tier)
