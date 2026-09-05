# -*- coding: utf-8 -*-
"""The purity boundary between the verifiers and the adapters, in both directions.

styxx/sworn.py and styxx/evidence.py are pure functions of bytes; styxx/harness/ is where the
ambient world (a path, an environment variable, stdin, the clock) is allowed in. The adapters
import from the verifiers; the verifiers never import from the adapters, by AST, by string
constant, and by sys.modules in a fresh interpreter. Also here: the packaging pin and the word
law over the new files.

LOAD-BEARING: test_no_verifier_imports_the_harness_package,
test_a_fresh_interpreter_that_imports_the_verifiers_never_loads_the_harness.
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

from styxx.evidence import _FORBIDDEN_MODULES

ROOT = Path(__file__).resolve().parent.parent
VERIFIERS = [ROOT / "styxx" / "sworn.py", ROOT / "styxx" / "evidence.py", ROOT / "styxx" / "__init__.py"]
ADAPTERS = sorted((ROOT / "styxx" / "harness").glob("*.py"))
HOOKS = ROOT / "integrations" / "claude-code" / "sworn-hooks"
NEW_FILES = ADAPTERS + sorted(HOOKS.glob("*.py")) + [HOOKS / "README.md"]
FORBIDDEN_WORDS = re.compile(r"\b(first|nobody|novel|wire|protocol|self-verifying|tamper-proof|immutable)\b", re.I)


def imports_of(tree):
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend((a.name, 0) for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            out.append((node.module or "", node.level))
            if node.level >= 1 and node.module is None:
                out.extend((a.name, node.level) for a in node.names)
    return out


@pytest.mark.parametrize("path", VERIFIERS, ids=lambda p: p.name)
def test_no_verifier_imports_the_harness_package(path):
    """LOAD-BEARING."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for name, level in imports_of(tree):
        assert not (name == "styxx.harness" or name.startswith("styxx.harness.")), (path.name, name)
        if level >= 1:
            assert not (name == "harness" or name.startswith("harness.")), (path.name, name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert node.value != "styxx.harness" and not node.value.startswith("styxx.harness."), \
                "%s smuggles the package name as a string" % path.name


def test_a_fresh_interpreter_that_imports_the_verifiers_never_loads_the_harness():
    """LOAD-BEARING."""
    code = "import styxx.sworn, styxx.evidence, sys; print('styxx.harness' in sys.modules)"
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True,
                       encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False"


@pytest.mark.parametrize("path", ADAPTERS, ids=lambda p: p.name)
def test_an_adapter_imports_from_styxx_only_the_verifiers_the_reader_and_the_canonicaliser(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    allowed = ("styxx.sworn", "styxx.evidence", "styxx.attestation", "styxx.harness")
    for name, level in imports_of(tree):
        if name.startswith("styxx"):
            assert name in allowed or any(name.startswith(a + ".") for a in allowed), (path.name, name)


def test_the_post_tool_entry_script_imports_nothing_from_styxx():
    tree = ast.parse((HOOKS / "post-tool.py").read_text(encoding="utf-8"))
    for name, _ in imports_of(tree):
        assert not name.startswith("styxx"), name


@pytest.mark.parametrize("name", ["junit.py", "github.py"])
def test_the_pure_adapters_reference_no_ambient_module_below_main(name):
    tree = ast.parse((ROOT / "styxx" / "harness" / name).read_text(encoding="utf-8"))
    banned = set(_FORBIDDEN_MODULES)
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert not ({a.name.split(".")[0] for a in node.names} & banned), name
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in banned, name
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef) or fn.name == "main":
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Import):
                assert not ({a.name.split(".")[0] for a in node.names} & banned), (name, fn.name)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                assert node.value.id not in banned, (name, fn.name, node.value.id)
                assert "%s.%s" % (node.value.id, node.attr) not in ("Path.cwd", "Path.home"), (name, fn.name)


def test_the_claude_code_adapter_imports_styxx_only_inside_finalise():
    tree = ast.parse((ROOT / "styxx" / "harness" / "claude_code.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            assert not any(n.startswith("styxx") for n in names), names
    where = [fn.name for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef)
             and any(isinstance(n, ast.ImportFrom) and (n.module or "").startswith("styxx") for n in ast.walk(fn))]
    assert where == ["finalise"]


def test_pyproject_ships_the_package():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    packages = text.split("[tool.setuptools]")[1].split("[tool.setuptools.package-data]")[0]
    assert '"styxx.harness"' in packages


def test_the_package_is_importable_and_names_its_three_adapters():
    import styxx.harness as H
    assert set(H.__all__) >= {"junit", "github", "claude_code"}
    assert "adapters, never a recorder" in H.LABEL
    assert "weak" in H.L1_WEAK


@pytest.mark.parametrize("path", NEW_FILES, ids=lambda p: p.name)
def test_the_word_law_holds_over_every_new_file(path):
    text = path.read_text(encoding="utf-8")
    hits = sorted({m.group(0).lower() for m in FORBIDDEN_WORDS.finditer(text)})
    assert not hits, "%s writes %s" % (path.name, hits)


def test_every_new_python_file_compiles_on_the_floor_syntax():
    """No match/case and no runtime `X | Y` unions: the floor is 3.9 and CI runs it."""
    for p in ADAPTERS + sorted(HOOKS.glob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        assert not any(type(n).__name__ == "Match" for n in ast.walk(tree)), p.name
