"""In-place edit of the styxx 7.50.0 Zenodo description (2026-10-23): the ceiling caveat.

The record (10.5281/zenodo.23594959, published 2026-10-23) quotes the April PyPI positioning that
7.50.0 replaced, "HaluEval-QA AUC 0.998", without the words this lab's house rule requires beside
that figure (tests/test_release_ceiling_caveat.py: "ceiling" within three lines). This edit adds
them, in the wording of the 7.48.0 precedent (zenodo_edit_v7_48_0_2026_09_29b.py, fc8843e0), and
changes nothing else. The operator approved it in chat on 2026-10-23 ("Fix both").

Run: py -3.12 zenodo_edit_software_v7_50_0_2026_10_23.py --token-file PATH [--apply]
Without --apply it is a dry run: it reads the deposition, runs the checks and writes the before and
edited metadata, and sends nothing. Before and after: DOI, concept, version and file checksums
unchanged, the record published. The token is sent only in an Authorization header and is never
printed or written.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from zenodo_deposit_software_v7_50_0 import load_token  # noqa: E402

API = "https://zenodo.org/api"
DEP = "23594959"
EXPECT = {"doi": "10.5281/zenodo.23594959", "conceptrecid": "19758618", "version": "7.50.0"}
UA = "fathom-lab-styxx-research/1 (+https://github.com/fathom-lab/styxx)"
TOKEN = ""

OLD = '("Cognitive observability", "HaluEval-QA AUC 0.998")'
NEW = ('("Cognitive observability", "HaluEval-QA AUC 0.998", a register-detection figure at a '
       'documented construct ceiling)')
CHARTER = ("groundbreaking", "revolutionary", "breakthrough", "novel", "tamper-proof",
           "self-verifying", "hallucination detector")


def call(method: str, path: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(API + path, data=data, method=method, headers={
        "Authorization": "Bearer " + TOKEN, "User-Agent": UA,
        "Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        msg = e.read()[:300].decode("utf-8", "replace")
        raise RuntimeError(f"{method} {path} -> HTTP {e.code}: {msg}") from None
    return json.loads(raw) if raw else {}


def check_record(d: dict) -> list[tuple[str, bool]]:
    m = d.get("metadata", {})
    files = sorted((f.get("filename") or f.get("key"), f.get("checksum")) for f in d.get("files", []))
    return [
        ("doi unchanged", (d.get("doi") or m.get("doi")) == EXPECT["doi"]),
        ("concept unchanged", str(d.get("conceptrecid")) == EXPECT["conceptrecid"]),
        ("version unchanged", m.get("version") == EXPECT["version"]),
        ("files present", len(files) == 3 and all(c for _, c in files)),
        ("published", bool(d.get("submitted")) and d.get("state") == "done"),
    ]


def charter_hits(text: str) -> list[str]:
    t = text.lower()
    return [b for b in CHARTER if any(f"{p}{b}" in t for p in (" ", ">", '"', "("))]


def build(meta: dict) -> dict:
    m = json.loads(json.dumps(meta))
    n = m["description"].count(OLD)
    if n != 1:
        raise SystemExit(f"REFUSED: expected one occurrence of the positioning sentence, found {n}")
    m["description"] = m["description"].replace(OLD, NEW)
    before, after = charter_hits(meta["description"]), charter_hits(m["description"])
    if len(after) > len(before):
        raise SystemExit(f"REFUSED: the edit adds charter words: {sorted(set(after) - set(before))}")
    return m


def main() -> int:
    global TOKEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-file", required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    TOKEN, source = load_token(a.token_file)
    print(f"token: from {source} (value not shown)")
    before = call("GET", f"/deposit/depositions/{DEP}")
    pre = check_record(before)
    files_before = sorted((f.get("filename"), f.get("checksum")) for f in before.get("files", []))
    for n, ok in pre:
        print(f"  [{'PASS' if ok else 'FAIL'}] before: {n}")
    if not all(ok for _, ok in pre):
        return 2
    new_meta = build(before["metadata"])
    (HERE / "metadata.before_edit_2026_10_23.json").write_text(
        json.dumps(before["metadata"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (HERE / "metadata.edit_2026_10_23.json").write_text(
        json.dumps(new_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("built: metadata.before_edit_2026_10_23.json, metadata.edit_2026_10_23.json")
    print(f"  the one change: {OLD!r} -> {NEW!r}")
    if not a.apply:
        print("dry run: nothing sent")
        return 0
    call("POST", f"/deposit/depositions/{DEP}/actions/edit")
    try:
        call("PUT", f"/deposit/depositions/{DEP}", {"metadata": new_meta})
        call("POST", f"/deposit/depositions/{DEP}/actions/publish")
    except Exception as e:  # leave the record as it was
        print(f"FAILED mid-edit: {e}; discarding the edit")
        call("POST", f"/deposit/depositions/{DEP}/actions/discard")
        return 3
    after = call("GET", f"/deposit/depositions/{DEP}")
    post = check_record(after)
    files_after = sorted((f.get("filename"), f.get("checksum")) for f in after.get("files", []))
    post.append(("file checksums unchanged", files_after == files_before))
    post.append(("edited text is live", after["metadata"].get("description") == new_meta["description"]))
    for n, ok in post:
        print(f"  [{'PASS' if ok else 'FAIL'}] after: {n}")
    (HERE / "metadata.after_edit_2026_10_23_readback.json").write_text(
        json.dumps(after["metadata"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    receipt = {"action": "software_record_metadata_edited", "deposit_id": DEP, "doi": EXPECT["doi"],
               "edit": "ceiling caveat beside the 0.998 figure the positioning sentence quotes",
               "approved": "operator, in chat, 2026-10-23 (\"Fix both\")",
               "precedent": "fc8843e0 (7.48.0, 2026-09-29b)",
               "checks_after": dict(post),
               "edited_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "fields_changed": ["description"]}
    (HERE / "zenodo-edit-receipt-software-v7.50.0-2026-10-23.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0 if all(ok for _, ok in post) else 4


if __name__ == "__main__":
    sys.exit(main())
