"""EXTERNAL-5: the 96 surviving accusations, re-read against the live pull request.

Prereg: PREREG_external5_survivors_at_source_2026_09_16.md.

    python external5_source.py fetch     # live diffs -> external5_items.jsonl (gitignored; URLs live here)
    python external5_source.py score     # items + external5_crosscheck.json -> external5_summary.json (committed)

Population: every claim with verdict CONTRADICTED in `external3_ledger.jsonl` (the BC-2 ledger).
Source: the pull request's unified diff as GitHub serves it today
(`patch-diff.githubusercontent.com/raw/OWNER/REPO/pull/N.diff`) — the same bytes the CLI's
`--pr URL` door reads. Deviation from the prereg, recorded in the RESULT: the prereg named the
PR's files page for paths and the REST `patch` fields for added lines; the `.diff` endpoint gives
both in one read, through the instrument's own `parse_unified_diff`, and needs no HTML parsing. The
REST `changed_files` cross-check on 20 seeded PRs is kept (fetched separately, browser-side, into
`external5_crosscheck.json` as {html_url: changed_files}).

The re-reading is the instrument itself: `gate_diff_text(claim sentence, live diff)` under the
BC-2 + COMPAT-1 checkout, taking the claim of the same kind and the same captured detail. Its
CONTRADICTED is UPHELD at the source, its VERIFIED is OVERTURNED, anything else (parse failure, no
Python in the live diff, the sentence no longer read) is UNREACHABLE and counted for neither side.
No diff is stored; each item keeps the diff's sha256, its file count and the facts the reading used.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from styxx import diffgate as dg  # noqa: E402

LEDGER = HERE / "external3_ledger.jsonl"
ITEMS = HERE / "external5_items.jsonl"
CROSS = HERE / "external5_crosscheck.json"
SUMMARY = HERE / "external5_summary.json"
STAT_LINE = re.compile(r"insertions?\(\+\)|deletions?\(-\)")   # the BC-2 census rule (external3_gates.py), verbatim
INSTRUMENT_SHA = hashlib.sha256(Path(dg.__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def survivors() -> list:
    out = []
    with LEDGER.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            for i, c in enumerate(r["claims"]):
                if c["verdict"] == "CONTRADICTED":
                    out.append({"pr_id": r["pr_id"], "agent": r.get("agent"), "url": r["html_url"],
                                "claim_index": i, "kind": c["kind"], "text": c["text"],
                                "detail": c["detail"], "corpus_why": c["why"]})
    return out


# Repositories renamed since the corpus was cut. GitHub answers the old name with a redirect through
# github.com, which the environment this ran in cannot fetch; the new names were read off the
# browser's address bar on 2026-09-16 and are pinned here so the run is reproducible. Every other
# PR is fetched under its corpus name.
RENAMED = {
    "BeehiveInnovations/zen-mcp-server": "BeehiveInnovations/pal-mcp-server",
    "ceedaragents/cyrus": "cyrusagents/cyrus",
    "dotnet/aspire": "microsoft/aspire",
    "githubnext/gh-aw": "github/gh-aw",
    "majorsilence/My-FyiReporting": "majorsilence/Reporting",
    "stargately/beancount-mobile": "bex-co/beancount-io",
}


# Two pull requests whose `.diff` endpoint answers 404 while the REST API serves them. Their file
# lists (filename, status, previous name) were read from `pulls/N/files` in a browser on 2026-09-16
# and are pinned here verbatim; the instrument reads them as a headers-only unified diff. Both
# carry files_changed_count claims, which need nothing but the paths.
REST_FILES = {
    "https://github.com/inkeep/agents/pull/655": [
        ["README.md", "modified", None], ["agents-docs/_snippets/copy-trace.mdx", "added", None],
        ["agents-docs/content/docs/community/inkeep-community.mdx", "modified", None],
        ["agents-docs/content/docs/concepts.mdx", "modified", None],
        ["agents-docs/content/docs/get-started/traces.mdx", "modified", None],
        ["agents-docs/content/docs/overview.mdx", "modified", None],
        ["agents-docs/content/docs/self-hosting/vercel.mdx", "modified", None],
        ["agents-docs/content/docs/talk-to-your-agents/vercel-ai-sdk.mdx", "modified", None],
        ["agents-docs/content/docs/troubleshooting.mdx", "modified", None],
        ["agents-docs/content/docs/typescript-sdk/project-structure.mdx", "modified", None],
        ["agents-docs/content/docs/typescript-sdk/structured-outputs/data-components.mdx", "modified", None],
        ["agents-docs/content/docs/visual-builder/structured-outputs/data-components.mdx", "modified", None],
        ["agents-docs/source.config.ts", "modified", None],
        ["agents-docs/src/components/navbar/index.tsx", "modified", None],
    ],
    "https://github.com/inkeep/agents/pull/663": [
        ["README.md", "modified", None], ["agents-docs/_snippets/copy-trace.mdx", "modified", None],
        ["agents-docs/_snippets/ui/CustomizationTip.mdx", "added", None],
        ["agents-docs/content/docs/get-started/push-pull.mdx", "modified", None],
        ["agents-docs/content/docs/get-started/quick-start.mdx", "modified", None],
        ["agents-docs/content/docs/overview.mdx", "modified", None],
        ["agents-docs/content/docs/talk-to-your-agents/a2a.mdx", "modified", None],
        ["agents-docs/content/docs/talk-to-your-agents/chat-api.mdx", "renamed", "agents-docs/content/docs/talk-to-your-agents/api.mdx"],
        ["agents-docs/content/docs/talk-to-your-agents/react/chat-button.mdx", "modified", None],
        ["agents-docs/content/docs/talk-to-your-agents/react/custom-trigger.mdx", "modified", None],
        ["agents-docs/content/docs/talk-to-your-agents/react/embedded-chat.mdx", "modified", None],
        ["agents-docs/content/docs/talk-to-your-agents/react/side-bar-chat.mdx", "modified", None],
        ["agents-docs/content/docs/talk-to-your-agents/vercel-ai-sdk/ai-elements.mdx", "added", None],
        ["agents-docs/content/docs/talk-to-your-agents/vercel-ai-sdk/use-chat.mdx", "renamed", "agents-docs/content/docs/talk-to-your-agents/vercel-ai-sdk.mdx"],
        ["agents-docs/content/docs/typescript-sdk/external-agents.mdx", "modified", None],
        ["agents-docs/content/docs/visual-builder/headers.mdx", "modified", None],
        ["agents-docs/content/docs/visual-builder/sub-agents.mdx", "modified", None],
        ["agents-docs/navigation.ts", "modified", None], ["agents-docs/redirects.json", "modified", None],
    ],
}


def rest_headers_diff(files: list) -> str:
    out = []
    for name, status, prev in files:
        if status == "added":
            out += [f"--- /dev/null", f"+++ b/{name}"]
        elif status == "removed":
            out += [f"--- a/{name}", f"+++ /dev/null"]
        else:
            out += [f"--- a/{prev or name}", f"+++ b/{name}"]
        out.append("@@ -0,0 +0,0 @@")
    return "\n".join(out) + "\n"


# One pull request whose page and REST record answer 404 while the `.diff` endpoint still serves a
# diff (found by the seeded cross-check). A diff nobody can see the PR of is not a source; its
# items are UNREACHABLE.
ORPHANED = {"https://github.com/sightread/sightread/pull/201"}

DIFF_GIT = re.compile(r"^diff --git a/(.*?) b/(.*)$", re.M)


def complete_paths(diff_text: str) -> tuple[str, int, int]:
    """The .diff marks a binary change as `Binary files … differ` with no `---`/`+++` headers, and
    `parse_unified_diff` registers nothing for it — an instrument defect found by this run's
    cross-check (a truthful "9 files changed" over eight PNGs and one .scss read as 1). The prereg
    defines the live count as the files page's list, which includes binaries, so the file list is
    completed here from the `diff --git` headers: each unregistered path gets an empty synthetic
    header so the instrument sees it. Returns (augmented diff, files by the instrument's parse,
    files by the headers)."""
    status, _ = dg.parse_unified_diff(diff_text)
    seen = set(status)
    extra = []
    for m in DIFF_GIT.finditer(diff_text):
        new = dg._norm(m.group(2))
        old = dg._norm(m.group(1))
        if new not in seen and old not in seen:
            seen.add(new)
            extra += [f"--- a/{m.group(1)}", f"+++ b/{m.group(2)}", "@@ -0,0 +0,0 @@"]
    n_all = len(seen)
    aug = diff_text if not extra else diff_text.rstrip("\n") + "\n" + "\n".join(extra) + "\n"
    return aug, len(status), n_all


def diff_url(html_url: str) -> str:
    m = re.match(r"https://github\.com/([^/]+)/([^/]+)/pull/(\d+)", html_url)
    if not m:
        raise ValueError(html_url)
    slug = RENAMED.get(f"{m.group(1)}/{m.group(2)}", f"{m.group(1)}/{m.group(2)}")
    return f"https://patch-diff.githubusercontent.com/raw/{slug}/pull/{m.group(3)}.diff"


def fetch(url: str) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": "styxx-external5/1.0 (+https://github.com/fathom-lab/styxx)"})
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, b""
        except Exception:  # noqa: BLE001 — one retry, then unreachable
            if attempt:
                return 0, b""
            time.sleep(3)
    return 0, b""


def read(item: dict, diff_text: str) -> dict:
    """The instrument's own reading of the claim sentence against the live diff."""
    g = dg.gate_diff_text(item["text"], diff_text, run=None, strict=False)
    same = [c for c in g.claims if c.kind == item["kind"]]
    want = item["detail"]
    # the same captured detail (n / prefix), so a sentence carrying two counts is matched on the right one
    pick = [c for c in same if all(c.detail.get(k) == v for k, v in want.items() if k in ("n", "prefix", "prefix2"))]
    c = (pick or same or [None])[0]
    if c is None:
        return {"verdict": None, "why": "the sentence is no longer read as this kind", "outcome": "UNREACHABLE"}
    outcome = {"CONTRADICTED": "UPHELD", "VERIFIED": "OVERTURNED"}.get(c.verdict, "UNREACHABLE")
    return {"verdict": c.verdict, "why": c.why, "outcome": outcome, "measured": g.measured}


def cmd_fetch() -> int:
    items = survivors()
    by_url: dict = {}
    print(f"{len(items)} surviving accusations on {len({i['url'] for i in items})} pull requests")
    out = []
    for n, it in enumerate(items, 1):
        url = it["url"]
        if url not in by_url:
            status, body = fetch(diff_url(url))
            endpoint = diff_url(url)
            if status != 200 and url in REST_FILES:
                body = rest_headers_diff(REST_FILES[url]).encode("utf-8"); status = 200
                endpoint = "REST pulls/N/files (headers only), read in a browser on 2026-09-16; the .diff endpoint answered 404"
            text = body.decode("utf-8", errors="replace") if body else ""
            aug, n_parse, n_all = complete_paths(text) if text else (text, 0, 0)
            by_url[url] = {"http": status, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest() if body else None,
                           "live_files": n_all, "files_by_parse": n_parse, "binary_unregistered": n_all - n_parse,
                           "text": aug, "endpoint": endpoint, "orphaned": url in ORPHANED}
            time.sleep(0.5)
        src = by_url[url]
        rec = {k: v for k, v in it.items()}
        rec["source"] = {k: v for k, v in src.items() if k != "text"}
        rec["shape"] = "git stat line" if (it["kind"] == "files_changed_count" and STAT_LINE.search(it["text"])) else "other"
        if src["orphaned"]:
            rec["reading"] = {"verdict": None, "why": "PR page and REST record answer 404; the diff endpoint still serves a diff, not trusted", "outcome": "UNREACHABLE"}
        elif src["http"] != 200 or not src["text"] or src["live_files"] == 0:
            rec["reading"] = {"verdict": None, "why": f"no diff served (http {src['http']}, {src['bytes']} bytes)", "outcome": "UNREACHABLE"}
        else:
            rec["reading"] = read(it, src["text"])
        out.append(rec)
        print(f"  {n:3d}/{len(items)} {it['kind']:20s} {rec['reading']['outcome']:11s} {rec['reading']['why'][:70]}")
    with ITEMS.open("w", encoding="utf-8") as fh:
        for rec in out:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"-> {ITEMS.name} ({len(out)} items; instrument {INSTRUMENT_SHA[:16]}…)")
    return 0


def cmd_score() -> int:
    items = [json.loads(l) for l in ITEMS.open(encoding="utf-8")]
    cross = json.loads(CROSS.read_text(encoding="utf-8")) if CROSS.exists() else {}
    by_kind: dict = {}
    for it in items:
        k = it["kind"]; o = it["reading"]["outcome"]
        by_kind.setdefault(k, Counter())[o] += 1
        if k == "files_changed_count":
            by_kind.setdefault(f"files_changed_count / {it['shape']}", Counter())[o] += 1
    total = Counter(it["reading"]["outcome"] for it in items)
    reached = total["UPHELD"] + total["OVERTURNED"]

    def rate(c: Counter):
        d = c["UPHELD"] + c["OVERTURNED"]
        return None if d == 0 else round(c["UPHELD"] / d, 4)

    # REST cross-check: the live file count against changed_files for the seeded 20, twice — as the
    # instrument's parse read it (the first run; three disagreements, the prereg's consequence
    # applies) and with the file list completed from the diff --git headers.
    cc = json.loads(CROSS.read_text(encoding="utf-8"))["changed_files"] if CROSS.exists() else {}
    cross = cc
    seen_all, seen_parse, orph = {}, {}, {}
    for it in items:
        seen_all.setdefault(it["url"], it["source"]["live_files"])
        seen_parse.setdefault(it["url"], it["source"].get("files_by_parse"))
        orph.setdefault(it["url"], it["source"].get("orphaned"))
    agree = disagree = 0; missing = []; first_agree = first_disagree = 0; detail = []
    for url, n_rest in cross.items():
        if url not in seen_all:
            missing.append(url); continue
        ok_first = seen_parse[url] == n_rest
        ok_now = seen_all[url] == n_rest
        first_agree += ok_first; first_disagree += (not ok_first)
        agree += ok_now; disagree += (not ok_now)
        if not ok_first or not ok_now:
            detail.append({"rest": n_rest, "by_parse": seen_parse[url], "by_headers": seen_all[url],
                           "orphaned": bool(orph[url])})
    binaries = sum(1 for u in seen_all if next(it for it in items if it["url"] == u)["source"].get("binary_unregistered"))
    n_bin_items = sum(1 for it in items if it["source"].get("binary_unregistered"))
    partial = reached < 0.9 * len(items)
    summary = {
        "prereg": "PREREG_external5_survivors_at_source_2026_09_16.md",
        "instrument_sha256": INSTRUMENT_SHA,
        "source": "patch-diff.githubusercontent.com/raw/OWNER/REPO/pull/N.diff, fetched 2026-09-16 (deviation from the prereg's files-page reading, recorded in the RESULT)",
        "population": len(items),
        "pull_requests": len({it["url"] for it in items}),
        "outcomes": dict(total),
        "reach": {"reached": reached, "rate": round(reached / len(items), 4), "G-E5-1_pass": not partial},
        "by_kind": {k: {"counts": dict(c), "upheld_rate": rate(c)} for k, c in by_kind.items()},
        "rest_crosscheck": {"n": len(cross),
                            "first_reading_by_instrument_parse": {"agree": first_agree, "disagree": first_disagree,
                                                                  "prereg_consequence": "files_changed_count voided as a preregistered figure"},
                            "completed_from_diff_git_headers": {"agree": agree, "disagree": disagree,
                                                                "remaining_disagreement": "the orphaned PR (page and REST 404, diff still served), excluded as UNREACHABLE"},
                            "disagreements": detail, "not_in_population": missing},
        "binary_files_unregistered_by_parse_unified_diff": {"pull_requests": binaries, "items_affected": n_bin_items,
                                                            "note": "instrument defect: every raw-diff door misses `Binary files … differ` entries; filed as an issue, repair prereg owed"},
        "floor_question": {k: (None if rate(c) is None else rate(c) >= 0.95) for k, c in by_kind.items()},
        "note": "UPHELD means the description does not match the pull request as it stands today; OVERTURNED means the corpus reconstruction disagreed with the live diff. No PR named here; URLs stay in the gitignored items file.",
    }
    SUMMARY.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(cmd_fetch() if (sys.argv[1:] or ["fetch"])[0] == "fetch" else cmd_score())
