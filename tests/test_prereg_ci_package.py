# -*- coding: utf-8 -*-
"""prereg-ci B0: the package ships, its public surface is what the design names, and it holds the
house rules (Python floor 3.9, a pure core, private reuse inside core.py only, the word law).

DESIGN 3.1: "The public surface is one small API (prereg_ci.core.rescore, .compare, .licences)";
"the private uses stay inside core.py, pinned by the conformance vectors". PLAN 1.1: styxx/__init__.py
does not re-export the package, so the public surface of styxx itself does not move.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "styxx" / "prereg_ci"
MODULES = sorted(PKG.glob("*.py"))
DESIGN_NAMES = {"rescore", "compare", "licences"}
NEW_FILES = (MODULES + sorted((ROOT / "conformance" / "prereg_ci").glob("*.py"))
             + sorted((ROOT / "conformance" / "prereg_ci").glob("*.md"))
             + sorted((ROOT / "tests").glob("test_prereg_ci_*.py")))
# the house words, held reversed so this file does not print them (as
# tests/test_unpriced_priority_sentences_withdrawn.py holds the ordinal)
_HOUSE = [w[::-1] for w in ("tsrif", "levon", "yranoitulover", "gnikaerbdnuorg", "hguorhtkaerb",
                            "foorp-repmat", "gniyfirev-fles", "rotceted noitanicullah")]
HOUSE_RX = re.compile("|".join(re.escape(w).replace(r"\-", r"[\s\-]?").replace(r"\ ", r"[\s\-]+")
                               for w in _HOUSE), re.IGNORECASE)
# priority sentences, reversed for the same reason
PRIORITY_RX = re.compile("|".join(re.escape(w[::-1]).replace(r"\ ", r"\s+") + r"\b"
                                  for w in ("sah eno on", "ylno eht")), re.IGNORECASE)


def test_pyproject_ships_the_package():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    packages = text.split("[tool.setuptools]")[1].split("[tool.setuptools.package-data]")[0]
    assert '"styxx.prereg_ci"' in packages


def test_the_package_exports_only_names_the_design_gives():
    import styxx.prereg_ci as P
    assert set(P.__all__) <= DESIGN_NAMES and {"rescore", "compare"} <= set(P.__all__)
    for name in P.__all__:
        assert callable(getattr(P, name)), name
    assert P.VERSION == "v0"


def test_core_defines_no_public_name_beyond_the_design():
    tree = ast.parse((PKG / "core.py").read_text(encoding="utf-8"))
    public = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            public.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            public.update(t.id for t in targets if isinstance(t, ast.Name))
    public = {n for n in public if not n.startswith("_")}
    assert public <= DESIGN_NAMES, sorted(public - DESIGN_NAMES)


def test_every_module_parses_on_the_floor_and_carries_the_future_import():
    assert {p.name for p in MODULES} >= {"__init__.py", "core.py"}
    for p in MODULES:
        src = p.read_text(encoding="utf-8")
        tree = ast.parse(src, feature_version=(3, 9))
        assert not any(type(n).__name__ == "Match" for n in ast.walk(tree)), p.name
        assert any(isinstance(n, ast.ImportFrom) and n.module == "__future__"
                   and any(a.name == "annotations" for a in n.names) for n in tree.body), p.name


def test_core_is_pure():
    """No process, socket, URL, file or clock in core.py: a FAIL is a function of the bytes it gets."""
    tree = ast.parse((PKG / "core.py").read_text(encoding="utf-8"))
    banned = {"subprocess", "socket", "urllib", "http", "os", "pathlib", "time", "datetime", "shutil",
              "tempfile", "requests"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not {a.name.split(".")[0] for a in node.names} & banned, ast.dump(node)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in banned, node.module
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in ("open", "exec", "eval", "__import__"), node.func.id


def test_private_reuse_stays_inside_core():
    for p in MODULES:
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "") in (
                    "styxx.protocol", "styxx.research_capsule", "styxx.charon", "styxx.sworn"):
                private = [a.name for a in node.names if a.name.startswith("_")]
                assert p.name == "core.py" or not private, (p.name, private)
            if isinstance(node, ast.Attribute) and node.attr.startswith("_") and not node.attr.startswith("__"):
                if isinstance(node.value, ast.Name) and node.value.id in ("_protocol", "_capsule"):
                    assert p.name == "core.py", (p.name, node.attr)


def test_styxx_does_not_re_export_the_package():
    tree = ast.parse((ROOT / "styxx" / "__init__.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert "prereg_ci" not in (node.module or ""), node.module
            assert not any(a.name == "prereg_ci" for a in node.names)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert "prereg_ci" not in node.value


def test_the_word_law_over_every_new_file():
    for p in NEW_FILES:
        text = p.read_text(encoding="utf-8")
        assert not HOUSE_RX.search(text), (p.name, HOUSE_RX.search(text).group(0))
        assert not PRIORITY_RX.search(text), (p.name, PRIORITY_RX.search(text).group(0))


def test_the_new_files_are_utf8_without_a_bom():
    for p in NEW_FILES:
        assert not p.read_bytes().startswith(b"\xef\xbb\xbf"), p.name
