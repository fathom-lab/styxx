# -*- coding: utf-8 -*-
"""RECEIPT v0 (DRAFT, unsigned): the one-file check page under receipt/v0/page/.

  - the committed page receipt/v0/page/receipt_check_v0.html is byte for byte what
    receipt/v0/page/build_page.py builds from this tree's receipt/v0/check_v0.js, with the command
    receipt/v0/page/README.md gives;
  - the checker embedded in the page, cut out by the HTML rule for script elements, is this tree's
    receipt/v0/check_v0.js byte for byte, and the page records its sha256;
  - the sha256 the README states for the page is the committed page's sha256;
  - the commit the page names as the last change to the checker is that commit, when git history
    is present. CI checks out one commit, but tests/conftest.py runs `git fetch --unshallow` once per
    session, so on CI this test RUNS against full history; it skips only when that unshallow fails,
    when the tree is not the top of its own git checkout (a `git archive` export or an sdist
    unpacked inside another repository), or when git finds no commit for the file. A squash or rebase
    merge of pull request 270 or of this branch gives the checker a new last commit; this test then
    fails until the page is rebuilt with --origin naming that commit;
  - every refusal listed in build_page.py's docstring, each run on a copy of receipt/v0/page in a
    temporary folder (so no run writes into the tree); the --from-git refusals use a throwaway git
    repository made in the temporary folder and need the git program;
  - every vector of receipt/v0/vectors/receipt_v0_vectors.json through the page's own code
    (receipt/v0/page/tests/prove_page2.js in Node worker threads), when Node is new enough.

NODE. prove_page2.js, page_harness.js and worker_bridge.js need `node:` module names, worker_threads,
vm and the global `performance`; all are in Node 16. The page code and the checker need BigInt. The
test asks for Node 18 or newer, the oldest the lab assumes a CI runner may carry. Only Node v24.13.0
was run when this test was written. With no node on PATH, or one older than 18, the test is
SKIPPED, with the reason in the skip message (pytest -rs prints it): the byte tests above do not
need Node, and tests/test_receipt_v0_checkers.py already fails when node is missing.

No test opens a browser. The page and the checker are lab-written; these tests show that the
committed file is the build of this tree and that the page's code gives the vectors' answers. They
do not show that the checker is right or that a browser enforces the page's policy.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
V0 = ROOT / "receipt" / "v0"
PAGE_DIR = V0 / "page"
PAGE = PAGE_DIR / "receipt_check_v0.html"
BUILDER = PAGE_DIR / "build_page.py"
README = PAGE_DIR / "README.md"
CHECKER = V0 / "check_v0.js"
VECTORS = V0 / "vectors" / "receipt_v0_vectors.json"
EXPECTED = PAGE_DIR / "examples" / "expected_cli_2026-11-13T153310Z.json"
PROVE = PAGE_DIR / "tests" / "prove_page2.js"

CHECKER_LAST_CHANGED = "53bbf47ed5304f047398d940db35f0ff3bea33cb"
ORIGIN = ("receipt/v0/check_v0.js of the tree this page is committed in; that file last changed at "
          "commit " + CHECKER_LAST_CHANGED + " (Fathom Lab)")
REPO = "github.com/fathom-lab/styxx"
BUILD_ARGS = ["receipt/v0/check_v0.js", "--repo", REPO, "--origin", ORIGIN]
README_COMMAND = ('python receipt/v0/page/build_page.py receipt/v0/check_v0.js --repo ' + REPO
                  + ' --origin "' + ORIGIN + '"')
NODE_FLOOR = 18
CHECKER_TAG = b'<script type="text/plain" id="checker-src">'
MAIN_TAG = b'<script id="page-main">'


def sha(b):
    return hashlib.sha256(b).hexdigest()


def run_builder(builder, args, cwd):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-B", str(builder)] + [str(a) for a in args], cwd=str(cwd),
                          capture_output=True, timeout=300, env=env)


def text(p):
    return (p.stdout + p.stderr).decode("utf-8", "replace")


def raw_script(page, tag):
    """The HTML rule for a script element: its raw text runs to the first '</script', in any case."""
    assert page.count(tag) == 1
    i = page.index(tag) + len(tag)
    return page[i:page.lower().index(b"</script", i)]


# ---------------------------------------------------------------- the committed page is the build


def test_the_committed_page_is_the_build_of_this_trees_checker(tmp_path):
    out = tmp_path / "receipt_check_v0.html"
    p = run_builder(BUILDER, BUILD_ARGS + ["--out", out], ROOT)
    assert p.returncode == 0, text(p)
    built = out.read_bytes()
    committed = PAGE.read_bytes()
    assert sha(built) == sha(committed), (
        "the committed page is not the build of this tree; rebuild it with the README's command")
    assert built == committed


def test_the_embedded_checker_is_this_trees_checker():
    page = PAGE.read_bytes()
    checker = CHECKER.read_bytes()
    assert raw_script(page, CHECKER_TAG) == checker
    csha = sha(checker)
    assert page.count(b'<meta name="receipt-v0-checker-sha256" content="' + csha.encode("ascii") + b'">') == 1
    main = raw_script(page, MAIN_TAG)
    assert main.count(b'const CHECKER_SHA256_AT_BUILD = "' + csha.encode("ascii") + b'";') == 1
    assert main.count(b'const CHECKER_ORIGIN = "' + ORIGIN.encode("ascii") + b'";') == 1


def test_the_readme_states_the_committed_pages_digest_and_the_build_command():
    readme = README.read_text(encoding="utf-8")
    page = PAGE.read_bytes()
    found = re.findall(r"sha256 of receipt_check_v0\.html: `([0-9a-f]{64})`", readme)
    assert found == [sha(page)], "README.md must state the committed page's sha256 once"
    assert ("Size: {:,} bytes.".format(len(page))) in readme
    assert README_COMMAND in readme
    assert sha(page) not in page.decode("ascii"), "a page cannot carry its own digest"


def test_the_page_names_the_commit_that_last_changed_the_checker():
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not on PATH; the commit the page names cannot be checked here")
    p = subprocess.run([git, "-C", str(ROOT), "rev-parse", "--show-toplevel"], capture_output=True, timeout=60)
    top = p.stdout.decode("utf-8", "replace").strip()
    if p.returncode != 0 or not top or os.path.normcase(str(Path(top).resolve())) != os.path.normcase(str(ROOT)):
        pytest.skip("this tree is not the top of its own git checkout (git answered %r); a git archive export or "
                    "an sdist unpacked inside another repository has no history of its own to check" % top)
    p = subprocess.run([git, "-C", str(ROOT), "rev-parse", "--is-shallow-repository"], capture_output=True, timeout=60)
    if p.returncode != 0:
        pytest.skip("git cannot say whether this checkout is shallow; the commit the page names was not checked")
    if p.stdout.strip() == b"true":
        pytest.skip("shallow checkout, and tests/conftest.py could not unshallow it: no history to find the last "
                    "change to receipt/v0/check_v0.js; the byte tests above still bind the page to the checker's sha256")
    p = subprocess.run([git, "-C", str(ROOT), "log", "-1", "--format=%H", "--", "receipt/v0/check_v0.js"],
                       capture_output=True, timeout=60)
    assert p.returncode == 0, p.stderr
    last = p.stdout.decode("ascii").strip()
    if not last:
        pytest.skip("git log names no commit for receipt/v0/check_v0.js in this checkout; the commit the page "
                    "names was not checked")
    assert last == CHECKER_LAST_CHANGED, (
        "the page says receipt/v0/check_v0.js last changed at %s, but history says %s (a squash or rebase merge "
        "also does this); rebuild the page with --origin naming %s" % (CHECKER_LAST_CHANGED, last, last))


# ---------------------------------------------------------------- the builder's refusals


@pytest.fixture
def copy(tmp_path):
    """A copy of receipt/v0/page with the tree's checker beside it, so a run never writes into the tree."""
    d = tmp_path / "page"
    shutil.copytree(str(PAGE_DIR), str(d), ignore=shutil.ignore_patterns(
        "old", "src", "runs", "__pycache__", "receipt_check_v0.html", "*.tmp"))
    shutil.copyfile(str(CHECKER), str(tmp_path / "check_v0.js"))
    return d


def build_copy(copy, extra=(), checker=None, out=None):
    out = out or (copy.parent / "out.html")
    args = [checker or (copy.parent / "check_v0.js"), "--out", out, "--repo", REPO, "--origin", ORIGIN] + list(extra)
    return run_builder(copy / "build_page.py", args, copy.parent), out


def refused(p, out, words, code=2):
    assert p.returncode == code, text(p)
    assert words in text(p), text(p)
    assert not out.exists(), "a refused build wrote a page"


def test_the_copy_builds_the_committed_page(copy):
    p, out = build_copy(copy)
    assert p.returncode == 0, text(p)
    assert out.read_bytes() == PAGE.read_bytes()


@pytest.mark.parametrize("change, words", [
    (lambda b: b + b"// \xc3\xa9\n", "is not ASCII"),
    (lambda b: b.replace(b"\n", b"\r\n", 1), "holds a CR or a NUL"),
    (lambda b: b + b"// \x00\n", "holds a CR or a NUL"),
    (lambda b: b + b"// <SCRIPT\n", "holds <script or </script"),
    (lambda b: b + b"// </Script>\n", "holds <script or </script"),
])
def test_refuses_a_checker_the_html_parser_would_change(copy, change, words):
    bad = copy.parent / "bad_checker.js"
    bad.write_bytes(change(CHECKER.read_bytes()))
    p, out = build_copy(copy, checker=bad)
    refused(p, out, words)


def test_refuses_a_checker_with_no_examples_made_with_it(copy):
    other = copy.parent / "other_checker.js"
    other.write_bytes(CHECKER.read_bytes() + b"// another checker\n")
    p, out = build_copy(copy, checker=other)
    refused(p, out, "no examples/expected_cli_*.json made with checker")


def _edit_expected(copy, edit):
    path = copy / "examples" / EXPECTED.name
    exp = json.loads(path.read_bytes())
    edit(exp)
    path.write_bytes((json.dumps(exp, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
    return path


def test_refuses_examples_recorded_with_another_checker(copy):
    path = _edit_expected(copy, lambda e: e["checker"].__setitem__("sha256", "0" * 64))
    p, out = build_copy(copy, extra=["--expected", path])
    refused(p, out, "records the examples' answers of checker")


def _set_recheck(e):
    e["examples"][0]["recheck"]["stdout"] = "DOES-NOT-REDERIVE\n"


def _set_rows(e):
    e["examples"][1]["rows"][1]["verdict"] = "HELD"


@pytest.mark.parametrize("what, change, words", [
    ("manifest", lambda c: (c / "examples" / "example-manifest.json").write_bytes(
        (c / "examples" / "example-manifest.json").read_bytes() + b" "), "example manifest bytes differ"),
    ("document", lambda c: (c / "examples" / "example-2-failed.md").write_bytes(
        (c / "examples" / "example-2-failed.md").read_bytes() + b"\n"), "example document example-2-failed.md differs"),
    ("receipt", lambda c: (c / "examples" / "receipts_cbdae74e95cbfb02" / "example-1-held.md.receipt.json").write_bytes(
        (c / "examples" / "receipts_cbdae74e95cbfb02" / "example-1-held.md.receipt.json").read_bytes() + b"\n"),
     "receipt file and recorded command line output differ"),
    ("rows", lambda c: _edit_expected(c, _set_rows), "the recorded verdicts differ from the receipt"),
    ("recheck", lambda c: _edit_expected(c, _set_recheck), "the recorded RECHECK of its own receipt is not REDERIVES"),
])
def test_refuses_examples_that_differ_from_their_records(copy, what, change, words):
    change(copy)
    p, out = build_copy(copy)
    refused(p, out, words)


def _set_exit(which, code):
    def edit(e):
        e["examples"][2][which]["exit"] = code
    return edit


def _set_document_verdict(e):
    e["examples"][0]["document_verdict"] = "SWORN-FAILED" if e["examples"][0]["document_verdict"] != "SWORN-FAILED" \
        else "SWORN-HELD"


@pytest.mark.parametrize("edit, words", [
    (_set_exit("check", 1), "receipt file and recorded command line output differ"),
    (_set_exit("recheck", 1), "receipt file and recorded command line output differ"),
    (_set_document_verdict, "the recorded verdicts differ from the receipt"),
])
def test_refuses_recorded_exit_codes_and_document_verdicts_that_differ(copy, edit, words):
    _edit_expected(copy, edit)
    p, out = build_copy(copy)
    refused(p, out, words)


RECEIPT_1 = Path("examples") / "receipts_cbdae74e95cbfb02" / "example-1-held.md.receipt.json"


def _crlf_document(c):
    doc = c / "examples" / "example-1-held.md"
    doc.write_bytes(doc.read_bytes().replace(b"\n", b"\r\n", 1))
    _edit_expected(c, lambda e: e["examples"][0].__setitem__("document_sha256", sha(doc.read_bytes())))


def _crlf_manifest(c):
    man = c / "examples" / "example-manifest.json"
    man.write_bytes(man.read_bytes().replace(b"\n", b"\r\n", 1))
    _edit_expected(c, lambda e: e["manifest"].__setitem__("sha256", sha(man.read_bytes())))


def _crlf_receipt(c):
    rec = c / RECEIPT_1
    rec.write_bytes(rec.read_bytes() + b"\r\n")
    _edit_expected(c, lambda e: e["examples"][0]["check"].__setitem__("stdout", rec.read_bytes().decode("ascii")))


def _non_ascii_document(c):
    doc = c / "examples" / "example-1-held.md"
    doc.write_bytes(doc.read_bytes() + b"\xc3\xa9\n")
    _edit_expected(c, lambda e: e["examples"][0].__setitem__("document_sha256", sha(doc.read_bytes())))


@pytest.mark.parametrize("change", [_crlf_document, _crlf_manifest, _crlf_receipt, _non_ascii_document])
def test_refuses_examples_that_are_not_ascii_with_lf_line_ends(copy, change):
    # each record is updated to match, so only the ASCII and line-end check stands in the way
    change(copy)
    p, out = build_copy(copy)
    refused(p, out, "is not ASCII with LF line ends")


def test_without_expected_the_build_uses_the_last_examples_file_by_name(copy):
    later = copy / "examples" / "expected_cli_2099-01-01T000000Z.json"
    shutil.copyfile(str(copy / "examples" / EXPECTED.name), str(later))
    exp = json.loads(later.read_bytes())
    _set_recheck(exp)
    later.write_bytes((json.dumps(exp, indent=1) + "\n").encode("utf-8"))
    p, out = build_copy(copy)
    refused(p, out, "the recorded RECHECK of its own receipt is not REDERIVES")
    p, out = build_copy(copy, extra=["--expected", copy / "examples" / EXPECTED.name])
    assert p.returncode == 0, text(p)
    assert out.read_bytes() == PAGE.read_bytes()


def test_the_examples_literal_escapes_what_could_end_or_alter_a_script():
    import importlib.util
    spec = importlib.util.spec_from_file_location("receipt_v0_build_page_under_test", str(BUILDER))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    lit = mod.js_literal({"x": "</script><!-- & > <"})
    assert "<" not in lit and ">" not in lit and "&" not in lit
    assert json.loads(lit) == {"x": "</script><!-- & > <"}


def _replace_once(path, old, new, count=1):
    s = path.read_bytes()
    assert s.count(old) >= count, (path, old)
    path.write_bytes(s.replace(old, new, count))


@pytest.mark.parametrize("fname, old, new, words", [
    ("page_main.js", b"const TIME_LIMIT_MS = 10000;", b"const TIME_LIMIT_MS = 9000;", "TIME_LIMIT_MS differs"),
    ("page_main.js", b"const MAX_RECEIPT_CHARS = 2000000;", b"const MAX_RECEIPT_CHARS = 1000000;", "MAX_RECEIPT_CHARS"),
    ("page_main.js", b"const ROWS_DRAWN = 200;", b"const ROWS_DRAWN = 100;", "ROWS_DRAWN or PRE_CHARS differs"),
    ("page_main.js", b"const PRE_CHARS = 200000;", b"const PRE_CHARS = 100000;", "ROWS_DRAWN or PRE_CHARS differs"),
    ("page.html", b"a limit of 10 seconds", b"a limit of 9 seconds", "no longer states"),
    ("page.html", b"at most 2,000,000 characters", b"at most 1,000,000 characters", "no longer states"),
    ("page.html", b"at most 200 span cards", b"at most 100 span cards", "no longer states"),
    ("page.html", b"first 200,000 characters", b"first 100,000 characters", "no longer states"),
])
def test_refuses_limits_that_differ_from_the_build_or_the_page_text(copy, fname, old, new, words):
    _replace_once(copy / "page_src" / fname, old, new)
    p, out = build_copy(copy)
    refused(p, out, words)


@pytest.mark.parametrize("flag, value, words", [
    ("--origin", 'a "quoted" origin', "must be printable ASCII without quotes"),
    ("--origin", "a <b> origin", "must be printable ASCII without quotes"),
    ("--origin", "back\\slash", "must be printable ASCII without quotes"),
    ("--origin", "tab\there", "must be printable ASCII without quotes"),
    ("--repo", "https://github.com/fathom-lab/styxx", "--repo must be plain text"),
    ("--repo", "github.com/fathom lab", "--repo must be plain text"),
    ("--repo", "a&b", "must be printable ASCII without quotes"),
])
def test_refuses_labels_outside_their_character_sets(copy, flag, value, words):
    out = copy.parent / "out.html"
    args = [copy.parent / "check_v0.js", "--out", out, "--repo", REPO, "--origin", ORIGIN]
    args[args.index(flag) + 1] = value
    p = run_builder(copy / "build_page.py", args, copy.parent)
    refused(p, out, words)


@pytest.mark.parametrize("token", ["http:", "https:", "<a ", "<img", "<link", "<iframe", "<form", "<object",
                                   "<embed", "<base", "srcset"])
def test_refuses_a_page_that_holds_a_link_or_an_outside_resource(copy, token):
    _replace_once(copy / "page_src" / "page.html", b"<main>\n", b"<main>\n<p>" + token.encode("ascii") + b"</p>\n")
    p, out = build_copy(copy)
    refused(p, out, "the page holds")


@pytest.mark.parametrize("decoy, words", [
    (b'<script type="text/plain" id="checker-src">decoy</script>\n',
     "the embedded checker does not come back out of the page unchanged"),
    (b'<script id="page-main">decoy</script>\n', "the page script does not come back out of the page unchanged"),
])
def test_refuses_a_script_that_would_not_come_back_out_unchanged(copy, decoy, words):
    _replace_once(copy / "page_src" / "page.html", b"<body>\n", b"<body>\n" + decoy)
    p, out = build_copy(copy)
    refused(p, out, words)


@pytest.mark.parametrize("fname, old, new, words", [
    ("page_main.js", b'"use strict";', b'"use strict"; // <!--', "the page script holds <!--"),
    ("page_main.js", b'"use strict";', b'"use strict"; // </script>', "holds <script or </script"),
    ("page.css", b"}", b"} .x { background: url(x) }", "the style block holds"),
    ("page.css", b"}", b"} @import x;", "the style block holds"),
    ("page.html", b"<title>", b"@@NOT_A_PLACEHOLDER@@<title>", "unknown placeholder"),
    ("page.html", b"@@REPO@@", b"repo", "placeholders not used"),
])
def test_refuses_page_sources_that_cannot_be_filled_safely(copy, fname, old, new, words):
    _replace_once(copy / "page_src" / fname, old, new)
    p, out = build_copy(copy)
    refused(p, out, words)


@pytest.mark.parametrize("fname", ["page.html", "page.css", "page_main.js"])
@pytest.mark.parametrize("change, words", [
    (lambda b: b.replace(b"\n", b"\r\n"), "holds a CR or a NUL"),
    (lambda b: b.replace(b"\n", b"\r\n", 1), "holds a CR or a NUL"),
    (lambda b: b + b"\x00", "holds a CR or a NUL"),
    (lambda b: b + b"/* \xc3\xa9 */\n", "is not ASCII"),
])
def test_refuses_page_sources_that_the_html_parser_or_a_csp_hash_would_see_differently(copy, fname, change, words):
    # an HTML parser turns CRLF into LF before a browser hashes an inline style or script, so a hash
    # the build took over CR bytes would not match and the browser would refuse the block
    path = copy / "page_src" / fname
    path.write_bytes(change(path.read_bytes()))
    p, out = build_copy(copy)
    refused(p, out, "page_src/" + fname + " " + words)


# ---------------------------------------------------------------- an existing page is never lost


def test_refuses_to_overwrite_a_page_from_another_checker(copy):
    out = copy.parent / "out.html"
    old = b'<meta name="receipt-v0-checker-sha256" content="' + b"0" * 64 + b'">\n'
    out.write_bytes(old)
    p, _ = build_copy(copy, out=out)
    assert p.returncode == 3, text(p)
    assert "was built from another checker" in text(p)
    assert out.read_bytes() == old
    # the builder makes the old/ folder before it refuses; it keeps nothing in it
    assert not (copy.parent / "old").exists() or list((copy.parent / "old").iterdir()) == []
    p, _ = build_copy(copy, extra=["--replace-different-checker"], out=out)
    assert p.returncode == 0, text(p)
    assert (copy.parent / "old" / ("out.checker-" + "0" * 64 + ".html")).read_bytes() == old
    assert out.read_bytes() == PAGE.read_bytes()


def test_keeps_an_other_checker_page_under_a_longer_name_when_the_kept_name_holds_other_bytes(copy):
    out = copy.parent / "out.html"
    old = b'<meta name="receipt-v0-checker-sha256" content="' + b"0" * 64 + b'">\n'
    out.write_bytes(old)
    first = copy.parent / "old" / ("out.checker-" + "0" * 64 + ".html")
    first.parent.mkdir()
    first.write_bytes(b"an earlier page kept under that name\n")
    p, _ = build_copy(copy, extra=["--replace-different-checker"], out=out)
    assert p.returncode == 0, text(p)
    second = copy.parent / "old" / ("out.checker-%s.page-%s.html" % ("0" * 64, sha(old)[:16]))
    assert second.read_bytes() == old
    assert first.read_bytes() == b"an earlier page kept under that name\n"
    assert out.read_bytes() == PAGE.read_bytes()


def test_keeps_a_same_checker_page_with_other_bytes_and_writes_nothing_for_the_same_bytes(copy):
    out = copy.parent / "out.html"
    old = PAGE.read_bytes() + b"\n"
    out.write_bytes(old)
    p, _ = build_copy(copy, out=out)
    assert p.returncode == 0, text(p)
    kept = copy.parent / "old" / ("out.checker-%s.page-%s.html" % (sha(CHECKER.read_bytes())[:16], sha(old)[:16]))
    assert kept.read_bytes() == old
    assert out.read_bytes() == PAGE.read_bytes()
    p, _ = build_copy(copy, out=out)
    assert p.returncode == 0 and "unchanged:" in text(p), text(p)
    assert sorted(x.name for x in (copy.parent / "old").iterdir()) == [kept.name]


def test_refuses_when_the_place_to_keep_the_old_page_holds_other_bytes(copy):
    out = copy.parent / "out.html"
    old = PAGE.read_bytes() + b"\n"
    out.write_bytes(old)
    kept = copy.parent / "old" / ("out.checker-%s.page-%s.html" % (sha(CHECKER.read_bytes())[:16], sha(old)[:16]))
    kept.parent.mkdir()
    kept.write_bytes(b"something else")
    p, _ = build_copy(copy, out=out)
    assert p.returncode == 2 and "exists with other bytes" in text(p), text(p)
    assert out.read_bytes() == old and kept.read_bytes() == b"something else"


# ---------------------------------------------------------------- --from-git


@pytest.fixture
def repo(tmp_path):
    """A throwaway git repository: c0 without the checker, c1 with this tree's checker and vectors, c2
    with another checker. Its git sees no system or user configuration."""
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not on PATH; the --from-git refusals need it")
    home = tmp_path / "home"
    home.mkdir()
    (home / ".gitconfig").write_bytes(b"")
    env = dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=str(home / ".gitconfig"), HOME=str(home),
               XDG_CONFIG_HOME=str(home), GIT_AUTHOR_NAME="test", GIT_AUTHOR_EMAIL="test@example.invalid",
               GIT_COMMITTER_NAME="test", GIT_COMMITTER_EMAIL="test@example.invalid")
    r = tmp_path / "repo"

    def g(*args):
        p = subprocess.run([git, "-C", str(r)] + list(args), capture_output=True, timeout=60, env=env)
        assert p.returncode == 0, p.stderr
        return p.stdout.decode("ascii").strip()
    r.mkdir()
    g("init", "-q")
    (r / "README.txt").write_bytes(b"a throwaway repository\n")
    g("add", "README.txt")
    g("commit", "-q", "-m", "c0")
    c0 = g("rev-parse", "HEAD")
    (r / "receipt" / "v0").mkdir(parents=True)
    (r / "receipt" / "v0" / "check_v0.js").write_bytes(CHECKER.read_bytes())
    (r / "receipt" / "v0" / "vectors").mkdir()
    (r / "receipt" / "v0" / "vectors" / "receipt_v0_vectors.json").write_bytes(VECTORS.read_bytes())
    g("add", "receipt/v0/check_v0.js", "receipt/v0/vectors/receipt_v0_vectors.json")
    g("commit", "-q", "-m", "c1")
    c1 = g("rev-parse", "HEAD")
    (r / "receipt" / "v0" / "check_v0.js").write_bytes(CHECKER.read_bytes() + b"// another checker\n")
    g("add", "receipt/v0/check_v0.js")
    g("commit", "-q", "-m", "c2")
    c2 = g("rev-parse", "HEAD")
    return {"dir": r, "c0": c0, "c1": c1, "c2": c2, "env": env, "g": g}


def from_git(copy, repo, args):
    out = copy.parent / "out.html"
    env = dict(repo["env"], PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, "-B", str(copy / "build_page.py"), "--from-git", str(repo["dir"]), "--out", str(out)]
                       + list(args), cwd=str(copy.parent), capture_output=True, timeout=300, env=env)
    return p, out


def test_from_git_builds_and_names_the_commit(copy, repo):
    repo["g"]("remote", "add", "origin", "https://github.com/owner/name.git")
    p, out = from_git(copy, repo, ["--commit", repo["c1"]])
    assert p.returncode == 0, text(p)
    page = out.read_bytes()
    assert raw_script(page, CHECKER_TAG) == CHECKER.read_bytes()
    assert ("receipt/v0/check_v0.js at commit %s (Fathom Lab)" % repo["c1"]).encode("ascii") in page
    # the whole label, with the scheme and the ".git" suffix gone
    assert page.count(b'<code class="value mono">github.com/owner/name</code>') == 1
    assert b"name.git" not in page
    assert b"not known to be" in page
    assert (copy / "src" / "checkers" / ("check_v0.%s.js" % repo["c1"][:12])).read_bytes() == CHECKER.read_bytes()
    assert (copy / "src" / "checkers" / ("receipt_v0_vectors.%s.json" % repo["c1"][:12])).read_bytes() == \
        VECTORS.read_bytes()


def test_from_git_refuses_a_kept_vectors_copy_with_other_bytes(copy, repo):
    kept = copy / "src" / "checkers" / ("receipt_v0_vectors.%s.json" % repo["c1"][:12])
    kept.parent.mkdir(parents=True)
    kept.write_bytes(b"{}\n")
    p, out = from_git(copy, repo, ["--commit", repo["c1"], "--repo", REPO])
    refused(p, out, "the kept vectors copy exists with other bytes")
    assert kept.read_bytes() == b"{}\n"


def test_from_git_refuses_a_commit_that_does_not_resolve(copy, repo):
    p, out = from_git(copy, repo, ["--commit", "0" * 40, "--repo", REPO])
    refused(p, out, "does not resolve")


def test_from_git_refuses_a_commit_without_the_checker(copy, repo):
    p, out = from_git(copy, repo, ["--commit", repo["c0"], "--repo", REPO])
    refused(p, out, "is not at commit")


def test_from_git_refuses_a_kept_copy_with_other_bytes(copy, repo):
    kept = copy / "src" / "checkers" / ("check_v0.%s.js" % repo["c1"][:12])
    kept.parent.mkdir(parents=True)
    kept.write_bytes(b"other bytes\n")
    p, out = from_git(copy, repo, ["--commit", repo["c1"], "--repo", REPO])
    refused(p, out, "exists with other bytes")
    assert kept.read_bytes() == b"other bytes\n"


def test_from_git_refuses_when_the_origin_remote_cannot_be_named(copy, repo):
    p, out = from_git(copy, repo, ["--commit", repo["c1"]])
    refused(p, out, "cannot be named as plain text; pass --repo")


@pytest.mark.parametrize("utc", [None, "yesterday"])
def test_from_git_refuses_to_make_examples_without_a_utc_time(copy, repo, utc):
    args = ["--commit", repo["c2"], "--repo", REPO] + (["--utc", utc] if utc else [])
    p, out = from_git(copy, repo, args)
    refused(p, out, "give --utc from date -u")
    assert sorted(x.name for x in (copy / "examples").glob("expected_cli_*.json")) == [EXPECTED.name]


@pytest.mark.parametrize("args, words", [
    (["--commit", "HEAD", "extra_checker.js"], "give a checker file or --from-git, not both"),
    ([], "--from-git needs --commit"),
    (["--commit", "HEAD~1", "--repo", REPO, "--origin", "x"], "--origin is set by --from-git"),
    (["--commit", "HEAD~1", "--repo", REPO, "--expected", "x.json"], "--expected is chosen by --from-git"),
])
def test_from_git_refuses_mixed_arguments(copy, repo, args, words):
    p, out = from_git(copy, repo, args)
    refused(p, out, words)
    # refused before anything is read or written: no write-once copy, no new examples file
    assert not (copy / "src").exists()
    assert sorted(x.name for x in (copy / "examples").glob("expected_cli_*.json")) == [EXPECTED.name]


# ---------------------------------------------------------------- every vector through the page's code


def _node_or_skip():
    node = shutil.which("node")
    if node is None:
        pytest.skip("NO NODE on PATH: the vector proof through the page (prove_page2.js) did not run; it needs "
                    "Node %d or newer" % NODE_FLOOR)
    p = subprocess.run([node, "--version"], capture_output=True, timeout=60)
    m = re.match(r"v(\d+)\.", p.stdout.decode("ascii", "replace").strip())
    if p.returncode != 0 or not m:
        pytest.skip("node --version gave no version (%r); the vector proof through the page did not run" % p.stdout)
    if int(m.group(1)) < NODE_FLOOR:
        pytest.skip("NODE TOO OLD: %s is older than %d, the oldest this proof asks for; the vector proof through "
                    "the page did not run" % (p.stdout.decode("ascii").strip(), NODE_FLOOR))
    return node


def test_every_vector_through_the_pages_own_code(tmp_path):
    node = _node_or_skip()
    p = subprocess.run([node, str(PROVE), "--page", str(PAGE), "--checker", str(CHECKER), "--vectors", str(VECTORS),
                        "--expected", str(EXPECTED), "--utc", "pytest", "--out-dir", str(tmp_path),
                        "--concurrency", "4"], cwd=str(PAGE_DIR), capture_output=True, timeout=1200)
    assert p.returncode == 0, text(p)
    s = json.loads((tmp_path / "page_vectors2_pytest.summary.json").read_bytes())
    total = len(json.loads(VECTORS.read_bytes())["vectors"])
    assert total > 0
    assert s["vectors_total"] == total
    assert s["matched"] == total, s["mismatched"]
    assert s["no_verdict"] == 0
    assert s["render_ok"] == total, s["render_problems"]
    c = s["checks"]
    assert c["checker_text_equals_source_bytes"] is True
    assert c["checker_hashes_all_equal"] is True
    assert c["page_sha256_agrees_with_node_crypto"] is True
    assert c["csp_script_hash_is_page_script"] is True and c["csp_style_hash_is_page_style"] is True
    assert c["csp_default_none"] is True and c["csp_worker_blob_only"] is True
    assert c["csp_no_connect_img_font_override"] is True
    assert c["static_scan_hits"] == []
    assert c["page_sha256"] == sha(PAGE.read_bytes())
    assert len(s["examples"]) == 3
    for e in s["examples"]:
        assert e["page_receipt_plus_lf_equals_cli_stdout"] is True, e
        assert e["page_receipt_plus_lf_sha256_equals_recorded"] is True, e
        assert e["page_document_verdict"] == e["cli_document_verdict"], e
        assert e["page_rows"] == e["cli_rows"], e
        assert e["page_recheck"] == e["cli_recheck"] == "REDERIVES", e
        assert e["render_ok"] is True, e


def test_make_examples_records_no_absolute_path_of_the_build_machine(copy):
    """make_examples.py runs the command line checker under Node, so this needs Node 18 or newer too
    (skipped with the reason otherwise). It writes into the copy only. The committed examples file
    of 2026-11-13 was written before this rule and still holds the build machine's paths; it is
    pinned history and stays as it is."""
    node = _node_or_skip()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, "-B", str(copy / "make_examples.py"), "--checker", str(copy.parent / "check_v0.js"),
                        "--utc", "2099-01-01T00:00:00Z", "--node", node, "--receipts-dir", "examples/receipts_test"],
                       cwd=str(copy.parent), capture_output=True, timeout=300, env=env)
    assert p.returncode == 0, text(p)
    rec = json.loads((copy / "examples" / "expected_cli_2099-01-01T000000Z.json").read_bytes())
    shown = [rec["checker"]["path"]] + [x for e in rec["examples"] for x in e["check"]["argv"] + e["recheck"]["argv"]]
    assert len(shown) > 3
    for x in shown:
        assert not os.path.isabs(x) and ":" not in x and not x.startswith("/") and "\\" not in x, x
    assert rec["checker"]["path"] == "../check_v0.js"
    assert rec["checker"]["sha256"] == sha(CHECKER.read_bytes())
    for e in rec["examples"]:
        committed = PAGE_DIR / "examples" / "receipts_cbdae74e95cbfb02" / (e["name"] + ".receipt.json")
        assert e["check"]["stdout"].encode("utf-8") == committed.read_bytes()
        assert e["check"]["exit"] == 0 and e["recheck"]["exit"] == 0
        assert e["recheck"]["stdout"].strip() == "REDERIVES"
