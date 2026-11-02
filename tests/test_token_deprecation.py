"""styxx.token is deprecated, not removed: it warns, names the retired mint and the doc that
retires it, and keeps every public name. Offline; no network call is made."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ("STYXX_MINT", "DEFAULT_RPC", "Tier", "TIERS", "tier_for_balance", "get_balance", "get_tier")


def _fresh_import():
    sys.modules.pop("styxx.token", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        __import__("styxx.token")          # a builtin, so the importer's frame is this one
    mod = sys.modules["styxx.token"]
    return mod, [w for w in caught if issubclass(w.category, DeprecationWarning)]


def test_importing_the_module_warns_once_naming_the_retired_mint_and_the_doc():
    mod, dep = _fresh_import()
    assert len(dep) == 1, [str(w.message) for w in dep]
    msg = str(dep[0].message)
    assert mod.STYXX_MINT in msg and "retired" in msg and "docs/token/README.md" in msg
    assert dep[0].filename == __file__, "the warning should point at the importer, not at importlib"


def test_the_mint_it_names_is_the_one_the_token_doc_lists_as_retired():
    mod, _ = _fresh_import()
    doc = (ROOT / "docs" / "token" / "README.md").read_text(encoding="utf-8")
    rows = [ln for ln in doc.splitlines() if mod.STYXX_MINT in ln]
    assert len(rows) == 1 and "retired" in rows[0], rows


def test_no_public_name_is_removed():
    mod, _ = _fresh_import()
    assert all(hasattr(mod, name) for name in PUBLIC)
    assert mod.tier_for_balance(120_000).name == "Validator"


def test_a_balance_lookup_warns_before_it_touches_the_network(monkeypatch):
    mod, _ = _fresh_import()

    def offline(*args, **kwargs):
        raise OSError("offline: no network in this test")

    monkeypatch.setattr(mod.urllib.request, "urlopen", offline)
    with pytest.warns(DeprecationWarning, match="retired"):
        with pytest.raises(OSError, match="offline"):
            mod.get_balance("wallet-address")


def test_the_cli_prints_the_deprecation_to_stderr(monkeypatch, capsys):
    mod, _ = _fresh_import()

    def offline(wallet):
        raise OSError("offline")

    monkeypatch.setattr(mod, "get_tier", offline)
    assert mod._main(["wallet-address"]) == 0
    captured = capsys.readouterr()
    assert "deprecated" in captured.err and mod.STYXX_MINT in captured.err
    assert "tier 0 - Public" in captured.out
