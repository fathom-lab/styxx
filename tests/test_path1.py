"""PATH-1: the two repaired `only_touches` failure modes, and the four that are not repaired.

PREREG_path1_only_touches_repair_2026_09_17.md (sha256 618d800f...). These tests pin the repair
*and* the non-repair: modes 3-6 are deliberately still wrong, and a test asserts they are still
wrong so that a later change cannot quietly claim them without a preregistration.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from styxx.diffgate import (
    PATH1_EXTENSIONS,
    _has_real_extension,
    _is_bare_filename,
    _path_inside,
    _prefix_is_path_shaped,
    gate_diff_text,
)

DATA = Path(__file__).resolve().parent.parent / "papers" / "closed-model-frontier" / "path1_extensions.txt"
JS = Path(__file__).resolve().parent.parent / "web" / "gate" / "diffgate.js"


def _diff(*paths: str) -> str:
    return "".join(f"diff --git a/{p} b/{p}\n--- a/{p}\n+++ b/{p}\n@@ -1 +1 @@\n+x\n" for p in paths)


def _only(claim: str, diff: str):
    g = gate_diff_text(claim, diff, run=None, strict=False)
    return next((c for c in g.claims if c.kind == "only_touches"), None)


# ---- G-P1-6: the list is committed data, in three places, byte for byte -----------------------

def test_extension_list_mirrors_the_committed_data_file():
    on_disk = {w for line in DATA.read_text(encoding="utf-8").splitlines()
               if not line.startswith("#") for w in line.split()}
    assert on_disk == set(PATH1_EXTENSIONS)
    assert len(on_disk) == 142


def test_extension_list_mirrors_the_javascript_port():
    src = JS.read_text(encoding="utf-8")
    block = re.search(r"const PATH1_EXTENSIONS = new Set\(`(.*?)`", src, re.S)
    assert block, "PATH1_EXTENSIONS not found in the JS port"
    assert set(block.group(1).split()) == set(PATH1_EXTENSIONS)


# ---- mode 2: a dot is not enough --------------------------------------------------------------

@pytest.mark.parametrize("token", ["package.json", "main.py", "a.cpp", "Chart.yaml", "x.tsx"])
def test_real_extensions_are_paths(token):
    assert _has_real_extension(token)
    assert _prefix_is_path_shaped(token, {})


@pytest.mark.parametrize("token", ["Assert.NotNull", ".k-step-link", "typia.misc", "os.path", "a.NotAThing"])
def test_identifiers_with_dots_are_not_paths(token):
    assert not _has_real_extension(token)
    assert not _prefix_is_path_shaped(token, {})


def test_dotted_identifier_abstains_rather_than_accuses():
    """dotnet/runtime#117821 -- 'Only modify Assert.NotNull usages within this file'."""
    c = _only("Only modify Assert.NotNull usages within this file.",
              _diff("src/libraries/System.Linq/tests/SequenceTests.cs"))
    assert c is not None and c.verdict == "UNCHECKABLE"


def test_a_dotted_identifier_that_names_a_changed_segment_is_still_a_path():
    """The segment test still applies after the dot test declines -- BC-2 behaviour is preserved."""
    assert _prefix_is_path_shaped("docs", {"docs/a.md": "M"})


# ---- mode 1: a bare filename means that filename anywhere --------------------------------------

def test_bare_filename_matches_on_basename():
    assert _is_bare_filename("package.json")
    assert _path_inside("appservice/package.json", "package.json")
    assert not _path_inside("appservice/package-lock.json", "package.json")


def test_a_prefix_with_a_slash_still_anchors():
    assert not _is_bare_filename("packages/core")
    assert _path_inside("packages/core/x.ts", "packages/core")
    assert not _path_inside("other/packages/core/x.ts", "packages/core")


def test_basename_claim_verifies_instead_of_accusing():
    """microsoft/vscode-azuretools#2086 -- every changed file IS a package.json."""
    c = _only("Only modify package.json and package-lock.json files in each package folder",
              _diff("appservice/package.json", "appservice/package-lock.json",
                    "auth/package.json", "auth/package-lock.json"))
    assert c is not None and c.verdict == "VERIFIED"


def test_basename_claim_still_accuses_when_something_else_changed():
    c = _only("Only modify package.json and package-lock.json files in each package folder",
              _diff("appservice/package.json", "src/main.ts"))
    assert c is not None and c.verdict == "CONTRADICTED"
    assert "src/main.ts" in str(c.why)


def test_single_file_basename_claim():
    """ydb-platform/ydb#25857 -- one file, nested, exactly the named one."""
    c = _only("Only modified `blobstorage_pdisk_impl.cpp` to conditionally report fields",
              _diff("ydb/core/blobstorage/pdisk/blobstorage_pdisk_impl.cpp"))
    assert c is not None and c.verdict == "VERIFIED"


# ---- the four modes PATH-1 does NOT repair -----------------------------------------------------
# These assert the instrument is still WRONG. They are not aspirational: they pin the published
# scope so that a later change cannot silently claim these without its own preregistration.
# When one is genuinely repaired, its prereg deletes the corresponding line here.

def test_mode3_runtime_behaviour_sentence_still_accuses_wrongly():
    """Albeoris/Memoria#1147 -- 'only changed mods/submods are serialized' is about runtime."""
    c = _only("Only changed mods/submods generate new XML with updated fields",
              _diff("FIXES_SUMMARY.md", "index.html"))
    assert c is not None and c.verdict == "CONTRADICTED", "mode 3 is not in PATH-1's scope"


def test_mode6_typo_in_stated_path_still_accuses_wrongly():
    """open-policy-agent/cert-controller#415 -- 'githiub' is a typo for 'github'."""
    c = _only("Only change .githiub/workflows/dependabot.yml, keep the changes to minimum",
              _diff(".github/workflows/dependabot.yml"))
    assert c is not None and c.verdict == "CONTRADICTED", "mode 6 is not in PATH-1's scope"


def test_the_two_correct_accusations_are_preserved():
    """Azure/autorest.typescript#3252 -- a path genuinely outside both stated prefixes."""
    c = _only("Only modified `packages/typespec-ts/` and `packages/typespec-test/` as specified",
              _diff("packages/typespec-ts/a.ts", "common/config/rush/pnpm-lock.yaml"))
    assert c is not None and c.verdict == "CONTRADICTED"
    assert "pnpm-lock.yaml" in str(c.why)
