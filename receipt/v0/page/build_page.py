"""Builds the one-file RECEIPT v0 check page.

From a checker file:
    python build_page.py CHECKER.js [--out PATH] [--origin LABEL] [--repo LABEL]
                         [--commit-published yes|no|unknown] [--expected FILE]
                         [--replace-different-checker]

From a commit of a git checkout, in one command (reads only; never checks out or writes there):
    python build_page.py --from-git REPO_DIR --commit COMMIT --utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
                         [--replace-different-checker] [--out PATH]
  It takes receipt/v0/check_v0.js at COMMIT (git show), keeps it as src/checkers/check_v0.<commit12>.js,
  keeps that commit's vectors as src/checkers/receipt_v0_vectors.<commit12>.json, names the repository
  from its `origin` remote as plain text without a scheme, records whether a remote branch the checkout
  knows holds the commit, writes the examples' command line answers for that checker with
  make_examples.py (a new examples/expected_cli_<utc>.json and receipts under
  examples/receipts_<checker sha16>/) unless one for that checker exists, and builds.

Writes dist/receipt_check_v0.html (or --out). The checker's bytes go into the page unchanged, inside
<script type="text/plain" id="checker-src">, and their sha256 is recorded in the page.

The build REFUSES (exit 2, or 3 for an existing page from another checker), and writes no page, when
(with --from-git, the write-once copies under src/checkers/ and a new examples file may already be written):
  - the checker holds anything but ASCII, a CR, a NUL, "<script" or "</script" in any case;
  - the examples file was made with another checker (its recorded checker sha256 differs), its example
    bytes differ from its records, an example's receipt differs from the recorded stdout, an example's
    recorded span verdicts differ from the receipt, or a recorded RECHECK is not REDERIVES;
  - the page script's limits differ from the build's (10 s, 2,000,000 receipt characters, 200 rows,
    200,000 drawn characters) or from the numbers the page text states;
  - --origin, --repo hold anything outside their plain character sets (no scheme, no quotes);
  - the page, once filled, holds http:, https:, <a , <img, <link, <iframe, <form, <object, <embed,
    <base or srcset, or a script that would not come back out of the HTML parser unchanged;
  - with --from-git: the commit does not resolve, the file is not at that commit, a kept copy under
    src/checkers/ exists with other bytes, the origin remote cannot be named without --repo, or no
    --utc is given when the examples must be made;
  - the output exists and was built from another checker, unless --replace-different-checker.

An existing page at the output path is never lost:
  - built from a different checker: kept first as old/receipt_check_v0.checker-<its checker sha256>.html;
  - built from the same checker with other bytes: kept as
    old/receipt_check_v0.checker-<sha16>.page-<page sha16>.html, then the new one is written;
  - identical bytes: nothing is written.

Standard library only. Deterministic: the same inputs give the same page bytes (no clock is read).
"""
import argparse
import base64
import glob
import hashlib
import html
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "page_src")
DEFAULT_OUT = os.path.join(HERE, "dist", "receipt_check_v0.html")
META_RE = re.compile(rb'<meta name="receipt-v0-checker-sha256" content="([0-9a-f]{64})">')
PLACEHOLDER = re.compile(r"@@([A-Z0-9_]+)@@")
TIME_LIMIT_MS = 10000          # must equal TIME_LIMIT_MS in page_main.js; the page text says 10 seconds
MAX_RECEIPT_CHARS = 2000000    # page_main.js holds it twice (page core and workerMain); the page text says 2,000,000
ROWS_DRAWN = 200
PRE_CHARS = 200000
CHECKER_PATH_IN_REPO = "receipt/v0/check_v0.js"
VECTORS_PATH_IN_REPO = "receipt/v0/vectors/receipt_v0_vectors.json"
PUBLISHED_WORDS = {
    "yes": "yes: when this page was built, the build's copy of the repository knew a remote branch holding that commit.",
    "no": "not known to be: when this page was built, that commit was on no remote branch the build's copy of the "
          "repository knew of, so it may not be fetchable from the repository yet.",
    "unknown": "not recorded when this page was built.",
}


def die(msg, code=2):
    sys.stderr.write("build_page: " + msg + "\n")
    sys.exit(code)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def fill(template, values, what):
    """One pass over the template: inserted values are never scanned for placeholders."""
    seen = set()

    def sub(m):
        k = m.group(1)
        if k not in values:
            die("%s: unknown placeholder @@%s@@" % (what, k))
        seen.add(k)
        return values[k]
    out = PLACEHOLDER.sub(sub, template)
    missing = set(values) - seen
    if missing:
        die("%s: placeholders not used: %s" % (what, sorted(missing)))
    return out


def raw_text_ok(b, what, allow_comment_open):
    """Text that goes inside a <script> element must come back out of the HTML parser unchanged."""
    if any(x >= 0x80 for x in b):
        die("%s is not ASCII; refusing (the page embeds only ASCII raw text)" % what)
    if b"\r" in b or b"\x00" in b:
        die("%s holds a CR or a NUL; the HTML parser would change it" % what)
    low = b.lower()
    if b"</script" in low or b"<script" in low:
        die("%s holds <script or </script; it cannot sit inside a script element unchanged" % what)
    if not allow_comment_open and b"<!--" in b:
        die("%s holds <!--" % what)


def load_examples(expected_path, checker_sha):
    with open(expected_path, "rb") as f:
        exp_bytes = f.read()
    exp = json.loads(exp_bytes)
    rel = os.path.relpath(expected_path, HERE).replace(os.sep, "/")
    if exp.get("checker", {}).get("sha256") != checker_sha:
        die("%s records the examples' answers of checker %s, not of this checker %s; run make_examples.py with this checker "
            "(or build with --from-git, which does)" % (rel, exp.get("checker", {}).get("sha256"), checker_sha))
    titles = {
        "example-1-held.md": "example 1: every span held",
        "example-2-failed.md": "example 2: one span failed",
        "example-3-unresolved.md": "example 3: the receipt cannot be resolved",
    }
    out = []
    with open(os.path.join(HERE, exp["manifest"]["file"]), "rb") as f:
        man = f.read()
    if sha(man) != exp["manifest"]["sha256"]:
        die("example manifest bytes differ from the expected file's record")
    for i, e in enumerate(exp["examples"]):
        with open(os.path.join(HERE, e["document_file"]), "rb") as f:
            doc = f.read()
        with open(os.path.join(HERE, e["receipt_file"]), "rb") as f:
            rec = f.read()
        if sha(doc) != e["document_sha256"]:
            die("example document %s differs from the expected file's record" % e["name"])
        stdout = e["check"]["stdout"].encode("utf-8")
        if rec != stdout or e["check"]["exit"] != 0 or e["recheck"]["exit"] != 0:
            die("example %s: receipt file and recorded command line output differ" % e["name"])
        if e["recheck"]["stdout"].strip() != "REDERIVES":
            die("example %s: the recorded RECHECK of its own receipt is not REDERIVES" % e["name"])
        core = json.loads(stdout)["core"]
        rows = [[r["verdict"], r["reason"]] for r in core["spans"]]
        if rows != [[r["verdict"], r["reason"]] for r in e["rows"]] or core["document_verdict"] != e["document_verdict"]:
            die("example %s: the recorded verdicts differ from the receipt" % e["name"])
        for b, w in ((doc, "document"), (man, "manifest"), (rec, "receipt")):
            if any(x >= 0x80 for x in b) or b"\r" in b:
                die("example %s %s is not ASCII with LF line ends; the page loads examples as text" % (e["name"], w))
        out.append({
            "id": "ex%d" % (i + 1),
            "title": titles.get(e["name"], e["name"]),
            "name": e["name"],
            "document": doc.decode("ascii"),
            "manifest": man.decode("ascii"),
            "receipt": rec.decode("ascii"),
            "expected": {
                "file": rel,
                "document_verdict": e["document_verdict"],
                "digest": e["digest"],
                "check_stdout_sha256": e["check_stdout_sha256"],
                "recheck": e["recheck"]["stdout"].strip(),
                "rows": rows,
            },
        })
    return out, rel, sha(exp_bytes)


def js_literal(v):
    """JSON as a JavaScript literal that cannot end or alter a script element."""
    s = json.dumps(v, ensure_ascii=True, indent=1, sort_keys=True)
    return s.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def csp_hash(text):
    return "'sha256-" + base64.b64encode(hashlib.sha256(text.encode("utf-8")).digest()).decode("ascii") + "'"


def git(repo, *args):
    p = subprocess.run(["git", "-C", repo] + list(args), capture_output=True, timeout=60)
    return p.returncode, p.stdout, p.stderr.decode("utf-8", "replace").strip()


def write_once(path, data, what):
    if os.path.exists(path):
        with open(path, "rb") as f:
            if f.read() != data:
                die("%s exists with other bytes: %s; refusing to replace it" % (what, path))
        return "unchanged"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    with open(path, "rb") as f:
        if f.read() != data:
            die("write did not land: " + path)
    return "written"


def repo_label_from_url(url):
    u = url.strip()
    m = re.match(r"^(?:[a-z+]+://)?(?:[^@/]+@)?([A-Za-z0-9.-]+)[:/]+(.+?)(?:\.git)?/?$", u)
    if not m:
        return None
    label = m.group(1) + "/" + m.group(2)
    return label if re.fullmatch(r"[A-Za-z0-9._/-]+", label) else None


def from_git(a):
    """Resolves --from-git into (checker path, origin label, repo label, published word, expected path)."""
    repo = a.from_git
    if not a.commit:
        die("--from-git needs --commit")
    rc, out, err = git(repo, "rev-parse", "--verify", a.commit + "^{commit}")
    if rc != 0:
        die("the commit %r does not resolve in %s: %s" % (a.commit, repo, err))
    full = out.decode("ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", full):
        die("unexpected commit id %r" % full)
    rc, checker, err = git(repo, "show", full + ":" + CHECKER_PATH_IN_REPO)
    if rc != 0:
        die("%s is not at commit %s: %s" % (CHECKER_PATH_IN_REPO, full, err))
    raw_text_ok(checker, "the checker at %s" % full[:12], allow_comment_open=True)
    cdir = os.path.join(HERE, "src", "checkers")
    cpath = os.path.join(cdir, "check_v0.%s.js" % full[:12])
    print("checker copy:", write_once(cpath, checker, "the kept checker copy"), cpath)
    rc, vec, err = git(repo, "show", full + ":" + VECTORS_PATH_IN_REPO)
    if rc == 0:
        vpath = os.path.join(cdir, "receipt_v0_vectors.%s.json" % full[:12])
        print("vectors copy:", write_once(vpath, vec, "the kept vectors copy"), vpath)
    else:
        print("vectors: none at that commit (%s)" % err)
    repo_label = a.repo
    if repo_label is None:
        rc, url, err = git(repo, "remote", "get-url", "origin")
        repo_label = repo_label_from_url(url.decode("utf-8", "replace")) if rc == 0 else None
        if repo_label is None:
            die("the origin remote of %s cannot be named as plain text; pass --repo" % repo)
    rc, br, err = git(repo, "branch", "-r", "--contains", full)
    published = a.commit_published or ("yes" if rc == 0 and br.strip() else ("no" if rc == 0 else "unknown"))
    origin = "%s at commit %s (Fathom Lab)" % (CHECKER_PATH_IN_REPO, full)
    csha = sha(checker)
    expected = latest_expected_for(csha)
    if expected is None:
        if not a.utc or not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", a.utc):
            die("no examples file for checker %s yet; give --utc from date -u so make_examples.py can write one" % csha)
        p = subprocess.run([sys.executable, os.path.join(HERE, "make_examples.py"), "--checker", cpath, "--utc", a.utc,
                            "--receipts-dir", "examples/receipts_%s" % csha[:16]], capture_output=True, timeout=300)
        sys.stdout.write(p.stdout.decode("utf-8", "replace"))
        if p.returncode != 0:
            die("make_examples.py refused or failed: " + p.stderr.decode("utf-8", "replace").strip())
        expected = latest_expected_for(csha)
        if expected is None:
            die("make_examples.py wrote no examples file for this checker")
    return cpath, origin, repo_label, published, expected


def latest_expected_for(checker_sha):
    found = []
    for p in sorted(glob.glob(os.path.join(HERE, "examples", "expected_cli_*.json"))):
        try:
            with open(p, "rb") as f:
                if json.loads(f.read()).get("checker", {}).get("sha256") == checker_sha:
                    found.append(p)
        except (OSError, ValueError):
            continue
    return found[-1] if found else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checker", nargs="?")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--origin", default=None)
    ap.add_argument("--repo", default=None, help="the repository as plain text without a scheme, e.g. github.com/OWNER/NAME")
    ap.add_argument("--commit-published", choices=sorted(PUBLISHED_WORDS), default=None)
    ap.add_argument("--expected", default=None, help="examples/expected_cli_*.json (default: the last by name made with this checker)")
    ap.add_argument("--replace-different-checker", action="store_true")
    ap.add_argument("--from-git", default=None, help="a git checkout to read the checker from (read only)")
    ap.add_argument("--commit", default=None)
    ap.add_argument("--utc", default=None, help="the time from date -u, for a new examples file")
    a = ap.parse_args()

    if a.from_git:
        if a.checker:
            die("give a checker file or --from-git, not both")
        checker_path, origin, repo_label, published, expected = from_git(a)
        if a.origin is not None:
            die("--origin is set by --from-git")
        if a.expected is not None:
            die("--expected is chosen by --from-git")
    else:
        if not a.checker:
            die("give a checker file, or --from-git REPO --commit COMMIT")
        checker_path = a.checker
        origin = a.origin if a.origin is not None else "not recorded at build"
        repo_label = a.repo if a.repo is not None else "not recorded at build"
        published = a.commit_published or "unknown"
        expected = a.expected

    with open(checker_path, "rb") as f:
        checker = f.read()
    raw_text_ok(checker, "the checker", allow_comment_open=True)
    checker_sha = sha(checker)

    if expected is None:
        expected = latest_expected_for(checker_sha)
        if expected is None:
            die("no examples/expected_cli_*.json made with checker %s; run make_examples.py with it first" % checker_sha)
    examples, expected_rel, expected_sha = load_examples(expected, checker_sha)

    with open(os.path.join(SRC, "page_main.js"), "r", encoding="ascii", newline="") as f:
        main_tpl = f.read()
    with open(os.path.join(SRC, "page.css"), "r", encoding="ascii", newline="") as f:
        css = f.read()
    with open(os.path.join(SRC, "page.html"), "r", encoding="ascii", newline="") as f:
        page_tpl = f.read()
    if "const TIME_LIMIT_MS = %d;" % TIME_LIMIT_MS not in main_tpl:
        die("page_main.js TIME_LIMIT_MS differs from the build's (the page text says 10 seconds)")
    if main_tpl.count("const MAX_RECEIPT_CHARS = %d;" % MAX_RECEIPT_CHARS) != 2 or main_tpl.count("const MAX_RECEIPT_CHARS") != 2:
        die("page_main.js must hold MAX_RECEIPT_CHARS = %d twice (page core and workerMain)" % MAX_RECEIPT_CHARS)
    if "const ROWS_DRAWN = %d;" % ROWS_DRAWN not in main_tpl or "const PRE_CHARS = %d;" % PRE_CHARS not in main_tpl:
        die("page_main.js ROWS_DRAWN or PRE_CHARS differs from the build's")
    for words in ("at most 2,000,000 characters", "at most 200 span cards", "first 200,000 characters", "a limit of 10 seconds"):
        if words not in page_tpl:
            die("page.html no longer states %r; the limits and the text must agree" % words)
    for label, v in (("--origin", origin), ("--repo", repo_label)):
        if not re.fullmatch(r"[\x20-\x7e]*", v) or '"' in v or "\\" in v or "<" in v or "&" in v:
            die("%s must be printable ASCII without quotes, backslashes, < or &" % label)
    if repo_label != "not recorded at build" and not re.fullmatch(r"[A-Za-z0-9._/-]+", repo_label):
        die("--repo must be plain text such as github.com/OWNER/NAME (letters, digits, . _ / -), no scheme")

    main_js = fill(main_tpl, {
        "CHECKER_SHA256": checker_sha,
        "CHECKER_ORIGIN": origin,
        "EXAMPLES_JSON": js_literal(examples),
    }, "page_main.js")
    raw_text_ok(main_js.encode("ascii"), "the page script", allow_comment_open=False)
    if "</style" in css.lower() or "url(" in css.lower() or "@import" in css.lower():
        die("the style block holds </style, url( or @import")

    csp = "; ".join([
        "default-src 'none'",
        "script-src " + csp_hash(main_js),
        "style-src " + csp_hash(css),
        "worker-src blob:",
        "base-uri 'none'",
        "form-action 'none'",
        "require-trusted-types-for 'script'",
        "trusted-types receipt-v0-worker",
    ])
    page = fill(page_tpl, {
        "CSP_ATTR": html.escape(csp, quote=True),
        "CSP_TEXT": html.escape(csp.replace("; ", ";\n"), quote=False),
        "CHECKER_SHA256": checker_sha,
        "REPO": html.escape(repo_label, quote=True),
        "COMMIT_PUBLISHED": html.escape(PUBLISHED_WORDS[published], quote=True),
        "STYLE": css,
        "CHECKER": checker.decode("ascii"),
        "MAIN": main_js,
    }, "page.html")
    out_bytes = page.encode("utf-8")

    # what the HTML parser will hand back: the raw text up to the first </script after the tag
    def extract(b, tag):
        i = b.index(tag) + len(tag)
        j = b.lower().index(b"</script", i)
        return b[i:j]
    if extract(out_bytes, b'<script type="text/plain" id="checker-src">') != checker:
        die("the embedded checker does not come back out of the page unchanged")
    if extract(out_bytes, b'<script id="page-main">') != main_js.encode("ascii"):
        die("the page script does not come back out of the page unchanged")
    for bad in (b"http:", b"https:", b"<a ", b"<img", b"<link", b"<iframe", b"<form", b"<object", b"<embed", b"<base", b"srcset"):
        if bad in out_bytes.lower():
            die("the page holds %r" % bad)

    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    kept = None
    if os.path.exists(out):
        with open(out, "rb") as f:
            old = f.read()
        if old == out_bytes:
            print("unchanged:", out)
            report(out, out_bytes, checker_sha, csp, expected_rel, expected_sha, None, origin, repo_label, published)
            return
        m = META_RE.search(old)
        old_checker = m.group(1).decode("ascii") if m else None
        olddir = os.path.join(os.path.dirname(out), "old")
        os.makedirs(olddir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(out))[0]
        if old_checker != checker_sha:
            if not a.replace_different_checker:
                die("%s was built from another checker (%s); refusing to overwrite it. "
                    "Pass --replace-different-checker to keep it under old/ and write the new page."
                    % (out, old_checker or "none recorded"), code=3)
            kept = os.path.join(olddir, "%s.checker-%s.html" % (stem, old_checker or "unrecorded-" + sha(old)[:16]))
            if os.path.exists(kept):
                with open(kept, "rb") as f:
                    if f.read() != old:
                        kept = os.path.join(olddir, "%s.checker-%s.page-%s.html" % (stem, old_checker or "unrecorded", sha(old)[:16]))
        else:
            kept = os.path.join(olddir, "%s.checker-%s.page-%s.html" % (stem, checker_sha[:16], sha(old)[:16]))
        if os.path.exists(kept):
            with open(kept, "rb") as f:
                if f.read() != old:
                    die("refusing: %s exists with other bytes" % kept)
        else:
            with open(kept, "wb") as f:
                f.write(old)
            with open(kept, "rb") as f:
                if f.read() != old:
                    die("the old page was not kept intact at %s" % kept)
    tmp = out + ".tmp"
    with open(tmp, "wb") as f:
        f.write(out_bytes)
    os.replace(tmp, out)
    with open(out, "rb") as f:
        if f.read() != out_bytes:
            die("write did not land: " + out)
    print("written:", out)
    report(out, out_bytes, checker_sha, csp, expected_rel, expected_sha, kept, origin, repo_label, published)


def report(out, b, checker_sha, csp, expected_rel, expected_sha, kept, origin, repo_label, published):
    print("page bytes:", len(b))
    print("page sha256:", sha(b))
    print("checker sha256:", checker_sha)
    print("checker origin:", origin)
    print("repository:", repo_label, "| commit published:", published)
    print("examples from:", expected_rel, "sha256", expected_sha)
    print("csp:", csp)
    if kept:
        print("previous page kept at:", kept)


if __name__ == "__main__":
    main()
