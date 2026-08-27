"""Collect and FREEZE the external corpus — the population this lab did not write.

Protocol: `PROTOCOL_oath_external_corpus_2026_08_27.md`, frozen before the first request.
This script implements it and nothing else. It freezes no clause and licenses no fix.

## What this does that the pilot did not

`oath_ext_recon.py` (14 repositories, 2026-08-26) cloned each repository, certified it, and threw
the content away. Its numbers can only be reproduced by re-fetching from GitHub, and it had four
silent-drop paths that removed repositories from the denominator without counting them: a 300s
clone timeout, a 2-page search cap that stopped on any error, receipts that failed to parse, and
READMEs over the size cap. A population defined by "what survived the fetch" is the same defect
this lane has now catalogued nine times.

This collector:

* **fetches over the API instead of cloning** — no timeout that silently drops large repositories,
  and exact provenance (`repo`, default branch, commit sha, path, byte length, `sha256`) for every
  file read;
* **gives every selected repository a terminal status.** There is no path out of `collect_repo`
  that does not return a record. `NO_DOC`, `NO_RECEIPT`, `DOC_TOO_LARGE`, `TREE_TRUNCATED`,
  `FETCH_FAILED`, `RECEIPTS_UNPARSEABLE` are outcomes, not absences;
* **takes receipts in TREE order**, not in the order the name list happens to be written. The
  pilot iterated `RECEIPT_NAMES` and capped at 12, so `all_results.json` files were always taken
  first and later names could never be reached in a repository with many receipts;
* **persists the measurement surface** — per-token ledger with a capped context excerpt, plus
  per-file hashes — so the corpus is replicable offline and verifiable against a re-fetch. Full
  third-party document and receipt bodies are NOT vendored; see the protocol's persistence section.

## Safety

External repositories are READ over HTTPS. Nothing from them is imported, executed, or evaluated.
This harness fetches bytes, hashes them, parses JSON, and runs the shipped verifier over them.

Every fetched blob is cached on disk under `$OATH_EXT_CACHE` (default: a temp directory), keyed by
`repo@sha/path`, so a re-run at the same shas costs no network and reproduces byte-identically.

  python papers/closed-model-frontier/oath_external_corpus.py [--max-repos N] [--workdir DIR]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx.certify import _TRIGGERS, certify_doc          # noqa: E402

MANIFEST = HERE / "oath_external_corpus.json"
LEDGER = HERE / "oath_external_corpus_ledger.jsonl"

# --- frozen selection parameters (see the protocol; do not tune these to taste) -----------------
SEARCH_QUERIES = (
    "filename:all_results.json",          # 1  HuggingFace Trainer   (pilot)
    "filename:eval_results.json",         # 2  HuggingFace Trainer   (pilot)
    "filename:metrics.json",              # 3  MLflow / DVC / generic
    "filename:results.json",              # 4  generic / hand-rolled
    "filename:scores.json",               # 5  generic / leaderboard
    "filename:benchmark_results.json",    # 6  benchmarking harnesses
    "filename:evaluation_results.json",   # 7  eval harnesses
)
MAX_REPOS_PER_QUERY = 20
MAX_REPOS_TOTAL = 140
# A total cap below per-query-cap x query-count deletes trailing queries DETERMINISTICALLY.
# At 20x7=140 against a 120 total, queries 1-6 filled their caps and query 7
# (`evaluation_results.json`, the non-HuggingFace eval-harness arm the protocol exists to
# reach) was never issued a single request. Found by red team before collection ran.
assert MAX_REPOS_PER_QUERY * len(SEARCH_QUERIES) <= MAX_REPOS_TOTAL, (
    f"total cap {MAX_REPOS_TOTAL} < {MAX_REPOS_PER_QUERY} x {len(SEARCH_QUERIES)}: "
    f"trailing queries would be silently deleted")
RECEIPT_NAMES = frozenset({
    "all_results.json", "eval_results.json", "test_results.json", "train_results.json",
    "metrics.json", "results.json", "scores.json", "benchmark_results.json",
    "evaluation_results.json",
})
MAX_RECEIPTS_PER_DOC = 12
MAX_RECEIPT_BYTES = 2_000_000
MAX_DOC_BYTES = 400_000
DOC_BASENAME = "readme.md"        # matched case-INSENSITIVELY, at the repository root only
_README_LIKE = "readme"           # recorded for NO_DOC repos so NO_PAIR stays interpretable
CONTEXT_CHARS = 200                      # protocol: excerpts capped, bodies not vendored

# The pilot's 14, recorded so they can be reported as a REPLICATION arm rather than as reach.
PILOT_REPOS = frozenset(
    e["repo"] for e in json.loads((HERE / "oath_ext_recon.json").read_text(encoding="utf-8"))
    ["per_repo"]
)

CACHE = Path(os.environ.get("OATH_EXT_CACHE",
                            Path(os.environ.get("TEMP", "/tmp")) / "oath_ext_corpus_cache"))
RAW = "https://raw.githubusercontent.com/{repo}/{sha}/{path}"
_SEARCH_SPACING_S = 6.5                  # code search is 10 req/min authenticated


def _token() -> str:
    return (ROOT.parent / "secrets" / "fathomlab-github.txt").read_text(encoding="utf-8").strip()


def gh_json(args: list[str], timeout: int = 90):
    """`gh api ...` -> parsed JSON, or None. The token is passed by env, never on argv."""
    env = dict(os.environ)
    env["GH_TOKEN"] = _token()
    try:
        r = subprocess.run(["gh", "api", *args], capture_output=True, text=True,
                           timeout=timeout, encoding="utf-8", errors="replace", env=env)
    except Exception:
        return None
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout)
    except Exception:
        return None


def search_repos(max_total: int) -> tuple[list[dict], list[dict]]:
    """Distinct repositories in the API's own order. Returns (selected, query_accounting).

    Code search returns FILES, not repositories, so reaching N distinct repositories requires
    paging. How far the paging got is recorded per query rather than left implicit.
    """
    seen, out, accounting = set(), [], []
    for qi, q in enumerate(SEARCH_QUERIES, start=1):
        taken, pages, total_hits, stopped = 0, 0, None, "cap_reached"
        for page in range(1, 11):
            if taken >= MAX_REPOS_PER_QUERY or len(out) >= max_total:
                break
            payload = gh_json(["-X", "GET", "search/code", "-f", f"q={q}",
                               "-f", "per_page=100", "-f", f"page={page}"])
            time.sleep(_SEARCH_SPACING_S)
            if payload is None:
                stopped = f"api_error_page_{page}"
                break
            pages += 1
            total_hits = payload.get("total_count", total_hits)
            items = payload.get("items", [])
            if not items:
                stopped = "exhausted"
                break
            for it in items:
                full = (it.get("repository") or {}).get("full_name")
                if not full or full in seen:
                    continue
                seen.add(full)
                out.append({"repo": full, "query": q, "query_index": qi,
                            "rank_within_query": taken})
                taken += 1
                if taken >= MAX_REPOS_PER_QUERY or len(out) >= max_total:
                    break
        accounting.append({"query": q, "query_index": qi, "repos_taken": taken,
                           "pages_read": pages, "api_total_hits": total_hits,
                           "stopped_because": stopped})
        if len(out) >= max_total:
            for rest in SEARCH_QUERIES[qi:]:
                accounting.append({"query": rest, "query_index": None, "repos_taken": 0,
                                   "pages_read": 0, "api_total_hits": None,
                                   "stopped_because": "total_cap_reached_before_this_query"})
            break
    return out, accounting


def head_of(repo: str):
    meta = gh_json([f"repos/{repo}"])
    if not meta or meta.get("private"):
        return None, None, None
    branch = meta.get("default_branch")
    if not branch:
        return None, None, None
    ref = gh_json([f"repos/{repo}/commits/{branch}"])
    if not ref or not ref.get("sha"):
        return None, branch, None
    return meta.get("license", {}).get("spdx_id") if meta.get("license") else None, branch, ref["sha"]


def tree_paths(repo: str, sha: str):
    t = gh_json([f"repos/{repo}/git/trees/{sha}?recursive=1"], timeout=120)
    if t is None:
        return None, False
    entries = [e for e in t.get("tree", []) if e.get("type") == "blob"]
    return entries, bool(t.get("truncated"))


def fetch(repo: str, sha: str, path: str, cap: int):
    """Bytes at a pinned sha, cached on disk. Returns (bytes, sha256) or (None, reason)."""
    key = hashlib.sha256(f"{repo}@{sha}/{path}".encode()).hexdigest()[:32]
    blob = CACHE / key
    if blob.exists():
        raw = blob.read_bytes()
        return raw, hashlib.sha256(raw).hexdigest()
    url = RAW.format(repo=repo, sha=sha, path=urllib.request.quote(path))
    req = urllib.request.Request(url, headers={"User-Agent": "fathom-lab-oath-corpus"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read(cap + 1)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return None, "fetch_failed"
    if len(raw) > cap:
        return None, "over_cap"
    CACHE.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(raw)
    return raw, hashlib.sha256(raw).hexdigest()


_DIGIT = re.compile(r"\d")


def _obligating_words(line: str) -> list[str]:
    return sorted({m.group(0).lower() for m in _TRIGGERS.finditer(line)})


def receipt_candidates(entries: list[dict]) -> list[dict]:
    """Receipt-named blobs in TREE order, deliberately.

    The pilot iterated its RECEIPT_NAMES tuple and capped at 12, so in any repository with more
    than twelve receipts the ones named `all_results.json` were always taken and the later names
    could never be reached — the cap silently selected by filename rather than by position. Tree
    order is not neutral either (it is roughly alphabetical by path), but it does not correlate
    with the name list, and it is the order the host reports rather than one this lab chose.
    """
    return [e for e in entries if e["path"].rsplit("/", 1)[-1] in RECEIPT_NAMES]


def collect_repo(entry: dict, workdir: Path) -> dict:
    """Every path returns a record. There is no silent drop."""
    repo = entry["repo"]
    rec = {**entry, "is_pilot_repo": repo in PILOT_REPOS, "status": None,
           "sha": None, "branch": None, "license": None, "files": [], "counts": {},
           "tokens": []}
    lic, branch, sha = head_of(repo)
    rec["license"], rec["branch"], rec["sha"] = lic, branch, sha
    if not sha:
        rec["status"] = "HEAD_UNAVAILABLE"
        return rec

    entries, truncated = tree_paths(repo, sha)
    rec["tree_truncated"] = truncated
    if entries is None:
        rec["status"] = "TREE_UNAVAILABLE"
        return rec
    if truncated:
        # Recorded, not skipped: a truncated tree means the receipt list may be incomplete, and a
        # reader must be able to see that rather than infer a clean read.
        rec["status_note"] = "tree listing truncated by the API; receipt set may be partial"

    def _root_basename(path):
        return path.rsplit("/", 1)[-1].lower() if "/" not in path else None

    doc_entry = next((e for e in entries
                      if _root_basename(e["path"]) == DOC_BASENAME), None)
    if doc_entry is None:
        # `path in DOC_NAMES` was an exact match over four spellings, so `ReadMe.md`,
        # `README.rst` and `docs/README.md` all became NO_DOC -- and the outcome table
        # then read NO_PAIR as "nobody publishes a claim document". That inference does
        # not follow from a string match, so the residual is now measured, not assumed.
        rec["readme_like_paths_seen"] = sorted(
            e["path"] for e in entries
            if e["path"].rsplit("/", 1)[-1].lower().startswith(_README_LIKE))[:12]
        rec["status"] = "NO_DOC"
        return rec

    doc_bytes, doc_sha = fetch(repo, sha, doc_entry["path"], MAX_DOC_BYTES)
    if doc_bytes is None:
        rec["status"] = "DOC_TOO_LARGE" if doc_sha == "over_cap" else "FETCH_FAILED"
        return rec

    cand = receipt_candidates(entries)
    rec["receipts_seen_in_tree"] = len(cand)
    # Three distinct facts, previously collapsed into one counter. `receipts_unparseable` was
    # incremented for a network failure and for an over-cap file as well as for malformed JSON,
    # so quoting it as a fact about other people's JSON would have been wrong.
    receipts, bad = [], {"unparseable": 0, "fetch_failed": 0, "over_cap": 0}
    for e in cand:
        if len(receipts) >= MAX_RECEIPTS_PER_DOC:
            break
        raw, rsha = fetch(repo, sha, e["path"], MAX_RECEIPT_BYTES)
        if raw is None:
            bad["over_cap" if rsha == "over_cap" else "fetch_failed"] += 1
            continue
        try:
            json.loads(raw.decode("utf-8", errors="replace"))
        except Exception:
            bad["unparseable"] += 1
            continue
        receipts.append({"path": e["path"], "bytes": len(raw), "sha256": rsha, "raw": raw})
    rec["receipts_rejected"] = bad
    rec["receipts_unparseable"] = bad["unparseable"]
    if not receipts:
        if not cand:
            rec["status"] = "NO_RECEIPT"
        elif bad["unparseable"] >= bad["fetch_failed"] + bad["over_cap"]:
            rec["status"] = "RECEIPTS_UNPARSEABLE"
        else:
            rec["status"] = "RECEIPTS_UNFETCHABLE"
        return rec

    stage = workdir / repo.replace("/", "__")
    stage.mkdir(parents=True, exist_ok=True)
    doc_path = stage / "DOC.md"
    doc_path.write_bytes(doc_bytes)
    rpaths = []
    for i, r in enumerate(receipts):
        p = stage / f"r{i}_{r['path'].rsplit('/', 1)[-1]}"
        p.write_bytes(r["raw"])
        rpaths.append(p)

    try:
        cert = certify_doc(doc_path, rpaths)
    except Exception as exc:
        rec["status"] = "CERTIFY_ERROR"
        rec["error"] = f"{type(exc).__name__}: {exc}"[:200]
        return rec

    lines = doc_bytes.decode("utf-8", errors="replace").splitlines()
    for e in cert["ledger"]:
        line = lines[e["line"] - 1] if 0 < e["line"] <= len(lines) else ""
        rec["tokens"].append({
            "line": e["line"], "col": e.get("col"), "token": e["token"], "value": e["value"],
            "status": e["status"], "receipt_ref": e["receipt_ref"],
            "obligating_words": _obligating_words(line),
            "context": line.strip()[:CONTEXT_CHARS],
        })
    rec["counts"] = cert["counts"]
    rec["verdict"] = cert["verdict"]
    # Rows 4/5 compare arms. An arm whose READMEs carry no line with BOTH a numeral and a trigger
    # word has no obligation surface at all, and "it abstained on everything" there is a fact
    # about the filename query, not about external prose. Counted so the rows can be read against
    # it rather than around it.
    rec["obligation_surface_lines"] = sum(
        1 for ln in lines if _DIGIT.search(ln) and _TRIGGERS.search(ln))
    rec["doc_lines"] = len(lines)
    rec["files"] = ([{"role": "document", "path": doc_entry["path"],
                      "bytes": len(doc_bytes), "sha256": doc_sha}]
                    + [{"role": "receipt", "path": r["path"], "bytes": r["bytes"],
                        "sha256": r["sha256"]} for r in receipts])
    rec["status"] = "CERTIFIED"
    return rec


def internal_control() -> dict:
    """The null arm: this laboratory's own corpus, RE-CERTIFIED under the pinned verifier.

    An earlier version summed the `counts` stored in committed certificates. That made the
    protocol's promise -- "the same verifier, at the same `verifier_sha256`" -- FALSE: the 186
    certificates under `papers/` carry TEN distinct `verifier_sha256` values and only four were
    produced by the current `styxx/certify.py`. Summing them and stamping today's verifier sha on
    the manifest would have published a mixture across ten instrument versions as a matched
    control, in a document whose whole thesis is that a number without a matched control is a
    number about the frame.

    So every document is re-certified live here, and what drifted is recorded rather than
    smoothed.
    """
    from styxx.corpus_audit import audit_document, discover_certificates
    counts = {"VERIFIED": 0, "ABSTAIN": 0, "UNGROUNDED": 0}
    docs, changed, failed, roster = 0, [], 0, []
    for cp in discover_certificates(ROOT / "papers"):
        try:
            r = audit_document(cp, search_root=ROOT / "papers")
            live = r["counts"]
        except Exception:
            failed += 1
            continue
        for k in counts:
            counts[k] += live.get(k, 0)
        docs += 1
        if r.get("verdict_changed"):
            changed.append(cp.name)
        try:
            stored = json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            stored = {}
        roster.append({"certificate": cp.name,
                       "document_sha256": stored.get("document_sha256"),
                       "recorded_verifier_sha256": stored.get("verifier_sha256")})
    total = sum(counts.values()) or 1
    return {
        "arm": "this laboratory's own corpus, re-certified live under the pinned verifier",
        "documents": docs, "recertification_failures": failed, "tokens": total,
        "status_counts": counts,
        "abstain_share": round(counts["ABSTAIN"] / total, 4),
        "accusation_share": round(counts["UNGROUNDED"] / total, 4),
        "verdicts_changed_vs_recorded": sorted(changed),
        "distinct_recorded_verifier_shas": len(
            {r["recorded_verifier_sha256"] for r in roster}),
        "certificates": roster,
        "disclosure_accusation_column": (
            "The accusation share here is a rate over ELEVEN events in THREE documents, all "
            "dated 2026-08-26, and it is not independent of the treatment arm: two of the three "
            "are documents ABOUT the external corpus, whose accused tokens are external numbers "
            "QUOTED inside internal prose. It carries no interval and licenses no ratio."),
        "disclosure_frame_grows": (
            "The control corpus is whatever is committed under papers/ at run time -- 178 when "
            "the pilot was written, 186 now -- and this collection's own RESULT will enter it. "
            "The certificate roster with per-document sha256 is recorded above so the arm is "
            "reproducible even though it is not stable."),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-repos", type=int, default=MAX_REPOS_TOTAL)
    ap.add_argument("--workdir", default=None)
    a = ap.parse_args()

    started = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    t0 = time.time()
    selected, accounting = search_repos(a.max_repos)
    print(f"selected {len(selected)} repositories across {len(SEARCH_QUERIES)} queries")

    workdir = Path(a.workdir) if a.workdir else CACHE / "stage"
    workdir.mkdir(parents=True, exist_ok=True)

    records = []
    for i, entry in enumerate(selected, start=1):
        rec = collect_repo(entry, workdir)
        records.append(rec)
        print(f"  [{i:>3}/{len(selected)}] {rec['status']:<22} {entry['repo'][:52]}")

    certified = [r for r in records if r["status"] == "CERTIFIED"]
    tokens = [t for r in certified for t in r["tokens"]]
    counts = {s: sum(1 for t in tokens if t["status"] == s)
              for s in ("VERIFIED", "ABSTAIN", "UNGROUNDED")}
    total = sum(counts.values()) or 1
    status_tally = {}
    for r in records:
        status_tally[r["status"]] = status_tally.get(r["status"], 0) + 1

    ctrl = internal_control()
    ext_share = {"abstain_share": round(counts["ABSTAIN"] / total, 4),
                 "accusation_share": round(counts["UNGROUNDED"] / total, 4)}

    # Per-arm, so rows 4/5 are read against their own n instead of around it. "Behave like" is not
    # a predicate; these are DESCRIPTIVE and gate nothing.
    per_arm = []
    for q in SEARCH_QUERIES:
        arm = [r for r in certified if r["query"] == q]
        at = [t for r in arm for t in r["tokens"]]
        n = len(at) or 0
        cc = {st: sum(1 for t in at if t["status"] == st)
              for st in ("VERIFIED", "ABSTAIN", "UNGROUNDED")}
        per_arm.append({
            "query": q,
            "is_pilot_query": q in SEARCH_QUERIES[:2],
            "repos_certified": len(arm),
            "repos_selected": sum(1 for r in records if r["query"] == q),
            "tokens": n,
            "status_counts": cc,
            "abstain_share": round(cc["ABSTAIN"] / n, 4) if n else None,
            "accusation_share": round(cc["UNGROUNDED"] / n, 4) if n else None,
            "obligation_surface_lines": sum(r.get("obligation_surface_lines", 0) for r in arm),
            "comparable": n >= 200,
        })
    no_doc = [r for r in records if r["status"] == "NO_DOC"]

    # styxx.discriminates is deliberately NOT called here, and the refusal is the point.
    #
    # An earlier draft of this script ran discrimination_report() with the external corpus as the
    # single "candidate" and the internal corpus as the "control". That is a category error:
    # discriminates compares candidate RULES scored on a shared frame, and asks whether any rule
    # beats doing nothing. There is one rule here — the shipped verifier — and two POPULATIONS.
    # Passing a population where the tool expects a rule would have produced a verdict-shaped
    # object that means nothing, which is a fair description of the defect this whole lane exists
    # to catch. Reaching for the newest instrument because it is available is how a marker becomes
    # a class.
    #
    # What the two arms need is a comparison, reported as one, with the confound named. The
    # discrimination check belongs to whatever future cycle proposes candidate CLAUSES over this
    # corpus, and the protocol says so.
    arms = {
        "external": ext_share,
        "internal_control": {"abstain_share": ctrl["abstain_share"],
                             "accusation_share": ctrl["accusation_share"]},
        "abstain_ratio_external_over_internal":
            round(ext_share["abstain_share"] / ctrl["abstain_share"], 2)
            if ctrl["abstain_share"] else None,
        "confound_that_is_not_ruled_out": (
            "The internal arm's documents cite receipts their own authors chose FOR them; the "
            "external arm's are certified against whatever receipt-named files happen to sit in "
            "the tree. That difference IS the contract, so the contrast measures contract-keeping "
            "and instrument behaviour together and cannot separate them. No causal claim is made "
            "from this pair."),
    }

    with LEDGER.open("w", encoding="utf-8", newline="\n") as fh:
        for r in certified:
            for t in r["tokens"]:
                fh.write(json.dumps({"repo": r["repo"], "sha": r["sha"], **t},
                                    ensure_ascii=False) + "\n")

    manifest = {
        "corpus": "OATH external corpus — the population this lab did not write",
        "protocol": "papers/closed-model-frontier/PROTOCOL_oath_external_corpus_2026_08_27.md",
        "status": "COLLECTED. Freezes no clause and licenses no fix.",
        "collected_at_utc": started,
        "elapsed_s": round(time.time() - t0, 1),
        "verifier_sha256": hashlib.sha256(
            (ROOT / "styxx" / "certify.py").read_bytes()).hexdigest(),
        "selection": {
            "queries": list(SEARCH_QUERIES),
            "rule": ("distinct repositories in the GitHub code-search API's own returned order; "
                     "no inspection, no substitution, no skipping"),
            "max_repos_per_query": MAX_REPOS_PER_QUERY,
            "max_repos_total_frozen": MAX_REPOS_TOTAL,
            "max_repos_total_used": a.max_repos,
            "freeze_note": ("The collection run is issued with NO arguments. Any --max-repos "
                            "override is a different selection rule and voids this freeze; the "
                            "used value is recorded beside the frozen one so a reader can check. "
                            "The total cap is 20x7 exactly, because a total below "
                            "per-query-cap x query-count deletes trailing queries silently."),
            "per_query": accounting,
            "disclosure": ("GitHub's default ordering is 'best match', not random, and code "
                           "search does not index every public repository. This is a CONVENIENCE "
                           "SAMPLE of a mechanically-defined population and supports no inference "
                           "about base rates anywhere."),
        },
        "accounting": {
            "repos_selected": len(records), "by_status": status_tally,
            "pilot_repos_redrawn": sum(1 for r in records if r["is_pilot_repo"]),
            "pilot_repos_note": ("The frozen rule does not seek the pilot's 14 out. Any that "
                                 "reappear are flagged and excluded from every claim that this "
                                 "corpus is new; if this is zero, no replication is claimed."),
            "no_doc_repos": len(no_doc),
            "no_doc_with_readme_like_paths": sum(
                1 for r in no_doc if r.get("readme_like_paths_seen")),
            "no_doc_note": ("NO_DOC means no root file whose basename case-folds to readme.md. "
                            "Repositories carrying some OTHER readme-like path are counted "
                            "separately so NO_PAIR is not read as 'publishes no claim document'."),
        },
        "per_arm": per_arm,
        "external_arm": {"documents": len(certified), "tokens": total,
                         "status_counts": counts, **ext_share,
                         "verdicts": {v: sum(1 for r in certified if r.get("verdict") == v)
                                      for v in {r.get("verdict") for r in certified}}},
        "control_arm": ctrl,
        "arm_comparison": arms,
        "per_repo": [{k: v for k, v in r.items() if k != "tokens"} for r in records],
        "what_this_does_not_show": (
            "Base rates. A convenience sample from one host's best-match ordering over README "
            "files. It also does not show whether any accusation is a real catch — that is the "
            "adjudication arm, and until it runs no accusation here may be called false."),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8", newline="\n")

    print()
    print(f"external : {len(certified)} docs  {total} tokens  "
          f"abstain {ext_share['abstain_share']}  accusations {counts['UNGROUNDED']}")
    print(f"control  : {ctrl['documents']} docs  {ctrl['tokens']} tokens  "
          f"abstain {ctrl['abstain_share']}  accusations {ctrl['status_counts']['UNGROUNDED']}")
    print(f"-> {MANIFEST.name}  /  {LEDGER.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
