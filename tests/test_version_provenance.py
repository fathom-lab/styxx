"""Guards for the version source-of-truth and its receipt provenance.

styxx is a cognitive-*observability* product: the version it stamps into every
attestation / vitals receipt is a provenance claim about the code that produced
the receipt. Before 7.17.3 that version came only from installed-distribution
metadata, which can be stale relative to the running code — so the product could
issue receipts with a wrong version about itself. 7.17.3 makes ``styxx/_version.py``
the single source of truth (read by both the runtime and the build), with the
installed metadata downgraded to a cross-check surfaced as ``__version_mismatch__``.

These tests fail if any link in that chain breaks:
  source literal -> runtime __version__ -> build/metadata version -> receipt stamp.
"""

import ast
import re
from pathlib import Path

import pytest

import styxx
import styxx._version


SEMVER = re.compile(r"^\d+\.\d+\.\d+([.\-+].*)?$")


def test_runtime_version_is_the_source_literal():
    # The attribute callers read must BE the source-of-truth literal, not a
    # metadata lookup that can drift from the running code.
    assert styxx.__version__ == styxx._version.__version__
    assert SEMVER.match(styxx.__version__), styxx.__version__


def test_version_module_is_import_free():
    # setuptools reads styxx._version.__version__ via `attr` at build time, which
    # AST-parses the file WITHOUT importing the package — but only if the module
    # is a bare literal with no imports. Any import here would force setuptools to
    # fall back to importing styxx during the build (slow, and a circular-import
    # risk). Lock the file down to assignments only.
    src = Path(styxx._version.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    bad = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not bad, f"styxx/_version.py must not import anything; found {len(bad)} import(s)"
    # And it must actually define __version__ as a string literal.
    assigns = {
        t.id: n.value
        for n in tree.body
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
    }
    assert "__version__" in assigns
    assert isinstance(assigns["__version__"], ast.Constant)
    assert isinstance(assigns["__version__"].value, str)


def _load_pyproject():
    pp = Path(styxx.__file__).resolve().parent.parent / "pyproject.toml"
    if not pp.exists():
        pytest.skip("pyproject.toml absent (installed wheel, not a source checkout)")
    text = pp.read_text(encoding="utf-8")
    try:
        import tomllib  # py3.11+
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore
        except ModuleNotFoundError:
            return None, text
    return tomllib.loads(text), text


def test_pyproject_reads_the_same_source_of_truth():
    # The build MUST derive its version from styxx/_version.py too, or a release
    # could ship a wheel whose metadata version disagrees with the running code.
    data, text = _load_pyproject()
    if data is not None:
        assert "version" in data["project"].get("dynamic", []), \
            "pyproject [project].dynamic must contain 'version'"
        assert "version" not in data["project"], \
            "pyproject must NOT also pin a static version (would shadow the dynamic source)"
        attr = data["tool"]["setuptools"]["dynamic"]["version"]["attr"]
        assert attr == "styxx._version.__version__", attr
    else:  # no TOML parser on this interpreter — fall back to text assertions
        assert 'dynamic = ["version"]' in text
        assert 'attr = "styxx._version.__version__"' in text


def test_receipt_stamp_traces_to_source_of_truth():
    # The version baked into attestation / vitals receipts must be the source
    # literal — this is the actual provenance guarantee the product makes.
    import styxx.attestation as attestation
    assert attestation._STYXX_VERSION == styxx._version.__version__


def test_mismatch_flag_is_none_in_a_consistent_env():
    # In a correctly-installed env (CI does `pip install -e .`), source and
    # installed metadata agree, so the stale-install flag is None.
    mm = styxx.__version_mismatch__
    assert mm is None, f"unexpected version mismatch: {mm}"


def test_mismatch_flag_shape_is_a_pair_when_set():
    # Documented contract: None, or a (source, installed) 2-tuple of strings.
    mm = styxx.__version_mismatch__
    assert mm is None or (isinstance(mm, tuple) and len(mm) == 2
                          and all(isinstance(x, str) for x in mm))


def test_doctor_flags_a_stale_install(monkeypatch):
    # When the cross-check trips, `run_doctor`'s version line must warn (not show
    # a clean OK) so a desynced environment is visible at a glance.
    import styxx.doctor as doctor
    monkeypatch.setattr(doctor, "__version_mismatch__", ("7.17.3", "7.17.2"))
    r = doctor._check_styxx_version()
    assert r.status == "warn"
    assert "7.17.2" in r.label and "7.17.3" in r.label
    assert "stale" in r.detail.lower()


def test_doctor_version_ok_when_consistent(monkeypatch):
    import styxx.doctor as doctor
    monkeypatch.setattr(doctor, "__version_mismatch__", None)
    r = doctor._check_styxx_version()
    assert r.status == "ok"
    assert styxx.__version__ in r.label
