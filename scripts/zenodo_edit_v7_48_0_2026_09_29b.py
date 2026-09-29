"""Second in-place edit of the styxx 7.48.0 Zenodo description (2026-09-29): the ceiling caveat.

The first edit (zenodo_edit_v7_48_0_2026_09_29.py, 17:06:54Z) added a bullet quoting the
HaluEval-QA AUC 0.998 without the words this lab's house rule requires beside that figure
(tests/test_release_ceiling_caveat.py: "ceiling" within three lines). This edit adds them and
changes nothing else.

Run: py -3.12 zenodo_edit_v7_48_0_2026_09_29b.py --token-file PATH [--apply]
Same checks as the first edit, before and after: DOI, concept, version and file md5s unchanged.
The token is sent only in an Authorization header and never printed or written.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import zenodo_edit_v7_48_0_2026_09_29 as first  # noqa: E402  (call, check_record, BANNED, DEP)
from zenodo_deposit_software_v7_48_0 import load_token  # noqa: E402

OLD = ("That number belongs to <code>guardrail.check</code> scored against a grounding passage, and "
       "the README row that reports it also reports two failures")
NEW = ("That number belongs to <code>guardrail.check</code> scored against a grounding passage, a "
       "register-detection figure at a documented construct ceiling, and the README row that reports "
       "it also reports two failures")


def build(meta: dict) -> dict:
    m = json.loads(json.dumps(meta))
    n = m["description"].count(OLD)
    if n != 1:
        raise SystemExit(f"REFUSED: expected one occurrence of the bullet, found {n}")
    m["description"] = m["description"].replace(OLD, NEW)
    text = (m["description"] + " " + (m.get("notes") or "") + " " + m["title"]).lower()
    hits = [b for b in first.BANNED if any(f"{p}{b}" in text for p in (" ", ">", '"', "("))]
    if hits:
        raise SystemExit(f"REFUSED: charter words in the edited text: {hits}")
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-file", required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    first.TOKEN, source = load_token(a.token_file)
    print(f"token: from {source} (value not shown)")
    dep = first.DEP
    before = first.call("GET", f"/deposit/depositions/{dep}")
    pre = first.check_record(before)
    for n, ok in pre:
        print(f"  [{'PASS' if ok else 'FAIL'}] before: {n}")
    if not all(ok for _, ok in pre):
        return 2
    new_meta = build(before["metadata"])
    (HERE / "metadata.before_edit_b.json").write_text(
        json.dumps(before["metadata"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (HERE / "metadata.edit_2026_09_29b.json").write_text(
        json.dumps(new_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("built: metadata.before_edit_b.json, metadata.edit_2026_09_29b.json")
    if not a.apply:
        print("dry run: nothing sent")
        return 0
    first.call("POST", f"/deposit/depositions/{dep}/actions/edit")
    try:
        first.call("PUT", f"/deposit/depositions/{dep}", {"metadata": new_meta})
        first.call("POST", f"/deposit/depositions/{dep}/actions/publish")
    except Exception as e:  # leave the record as it was
        print(f"FAILED mid-edit: {e}; discarding the edit")
        first.call("POST", f"/deposit/depositions/{dep}/actions/discard")
        return 3
    after = first.call("GET", f"/deposit/depositions/{dep}")
    post = first.check_record(after)
    post.append(("edited text is live", after["metadata"].get("description") == new_meta["description"]))
    for n, ok in post:
        print(f"  [{'PASS' if ok else 'FAIL'}] after: {n}")
    (HERE / "metadata.after_edit_b_readback.json").write_text(
        json.dumps(after["metadata"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    receipt = {"action": "software_record_metadata_edited", "deposit_id": dep, "doi": first.EXPECT["doi"],
               "edit": "b (ceiling caveat beside the 0.998 figure)",
               "checks_after": dict(post),
               "edited_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "fields_changed": ["description"]}
    (HERE / "zenodo-edit-receipt-software-v7.48.0-2026-09-29b.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0 if all(ok for _, ok in post) else 4


if __name__ == "__main__":
    sys.exit(main())
