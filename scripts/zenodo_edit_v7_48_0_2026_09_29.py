"""Correct the published styxx 7.48.0 Zenodo record's description in place (2026-09-29).

Run: py -3.12 zenodo_edit_v7_48_0_2026_09_29.py --token-file PATH [--apply]
Without --apply it only builds the edited metadata and writes it beside this script. With
--apply it opens an edit of deposition 23042251, PUTs the corrected metadata, publishes, then
re-reads the record and checks that the DOI, the concept, the version and the three files did
not change. The operator asked on 2026-09-29 that everything published be made correct.
The token is sent only in an Authorization header and never printed or written.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from zenodo_deposit_software_v7_48_0 import load_token  # noqa: E402  ([ZENODO] zenodo_token only)

API = "https://zenodo.org/api"
DEP = "23042251"
UA = "fathom-lab-styxx-research/1 (+https://github.com/fathom-lab/styxx)"
EXPECT = {"doi": "10.5281/zenodo.23042251", "conceptrecid": "19758618", "version": "7.48.0",
          "files": {"styxx-v7.48.0-source-bundle.zip": "1145aff5e0fe5f532ee2f7e8ab01f9fe",
                    "styxx-7.48.0-py3-none-any.whl": "4304824ee5e3f0335586ba82a1ddd771",
                    "styxx-7.48.0.tar.gz": "0fc7b2fcf57762d633d6fffc1415939b"}}
BANNED = ("first", "novel", "revolutionary", "groundbreaking", "breakthrough", "tamper-proof",
          "self-verifying", "hallucination detector")
TOKEN = ""


def zdoi(n: str) -> str:
    return f'<a href="https://doi.org/10.5281/zenodo.{n}">10.5281/zenodo.{n}</a>'


# (old, new): each old string must occur exactly once in the published description
DESC_EDITS = [
    (f'{zdoi("19746215")}) and belongs',
     f'{zdoi("19746215")}; its deposited foundations outline predates the 2026-06-21 scope note, '
     'D5 in <code>zenodo/MANIFEST.json</code>) and belongs'),
    ('The previous version the lab holds a deposit receipt for is\nv6.2.0 (' + zdoi("19758619")
     + ',\n2026-04-25).</p>',
     'The previous version in this concept is\nv6.2.0 (' + zdoi("19758619") + ',\n2026-04-25). '
     'styxx 7.2.0, 7.7.7 and 7.7.9 were archived as versions of the research-series concept '
     + zdoi("19326174") + ' (Fathom v23, v24 and v25: ' + zdoi("20130041") + ', '
     + zdoi("20418532") + ' and ' + zdoi("20419662") + '), not in this concept.</p>'),
    ('1689 reproduce the verdict core digest, 0 disagree, 1929 skipped.',
     '1689 reproduce the verdict core digest, 0 disagree, 1931 skipped of the 3620 in this tree '
     '(replayed 2026-09-29; the committed replay report, run on 2026-09-05 against 3618 vectors, '
     'says 1929).'),
    ('every line is a verdict re-derived from bytes, 243 lines at ship; <code>verify</code> separates a',
     'every line is a verdict re-derived from bytes; it held 243 entries when charon v0.1 merged '
     '(2026-09-05) and holds 257 in this tree; <code>verify</code> separates a'),
    ('against a head pinned outside it (<code>--expect-head</code>); nothing in it is immutable, and\n'
     'SAME_LINE 243, TAMPER 0 at ship is a determinism check, not a stability result.</li>',
     'against a head pinned outside it (<code>--expect-head</code>); nothing in it is immutable. '
     "SAME_LINE 243, TAMPER 0 was charon v0.1's verify on the day its log was built, a determinism "
     'check, not a stability result. The verify result committed in this tree (2026-09-14, 254 '
     'entries, head <code>7817c731</code>) reads SAME_LINE 11, MOVED_VERIFIER 243, TAMPER 0 and was '
     'not re-run after the last three entries. A verify over all 257 entries, run on 2026-09-29 and '
     'committed in pull request #167 (<code>papers/charon/charon_verify_result_2026_09_29.json</code>), '
     'reads SAME_LINE 14, MOVED_VERIFIER 243, SKEW 0, DRIFT 0, TAMPER 0.</li>'),
    ('<li><strong>Erratum: the staged "owed" note.</strong>',
     '<li><strong>Erratum: the "Staged for 7.48.0" note.</strong>'),
    ('regenerated in place"; the NOTE records that conflict and leaves the RESULT unedited. A CI run on\n'
     'the regenerated set is owed.</li>',
     'regenerated in place"; the NOTE records that conflict and leaves the RESULT unedited. A CI run on\n'
     'the regenerated set is owed. (Corrected 2026-09-29: two <code>tests</code> runs on the regenerated '
     'set passed before the tag, 36167419000 on 99118487 and 36173712926 on 1218dbad. C7 skips when its '
     'subprocess dies, so a pass does not show that it ran; only pull request #163 records C7 passing, '
     'in the earlier of the two.)</li>'),
    ('its own pinned checkout, is unchanged.</li>',
     'its own pinned checkout, is unchanged. <code>sworn/examples/sworn.yml</code> line 27 says the '
     'same, and also that this repository dogfoods the action, which no workflow here does; pull '
     'request #167 corrects it.</li>'),
    ('version DOI; the software concept DOI is 10.5281/zenodo.19758618. <code>README.md</code> labels\n'
     '10.5281/zenodo.19326174, the concept record of the Fathom research-paper series, as the\n'
     'always-latest DOI in the software badge row and link table. Both are recorded as open defects (D1,\n'
     'D3, D4) in <code>zenodo/MANIFEST.json</code> and are left for the maintainer to decide.</li>',
     'version DOI; the software concept DOI is 10.5281/zenodo.19758618, and the default branch '
     'corrected the line after this deposit (a4732c52, pull request #166; D1 resolved). '
     '<code>README.md</code> labels\n10.5281/zenodo.19326174, the concept record of the Fathom '
     'research-paper series, as the\nalways-latest DOI in the software badge row and link table; that '
     'is recorded as open defects D3 and D4 in <code>zenodo/MANIFEST.json</code> and left for the '
     'maintainer to decide.</li>\n\n'
     '<li><code>CITATION.cff</code> still lists the keyword <code>hallucination-detection</code> (line '
     '18), which this release removed from <code>pyproject.toml</code> per the charter. Its preferred '
     'citation (line 46, 10.5281/zenodo.19777921), which GitHub\'s "Cite this repository" prompt '
     'renders, is the April version of the position paper, without the 2026-06-21 scope erratum (D2 in '
     '<code>zenodo/MANIFEST.json</code>).</li>\n\n'
     '<li>The tree and the wheel carry priority claims this lab has not earned, in the docstrings of '
     '<code>styxx/__init__.py</code>, <code>forecast.py</code>, <code>intercept.py</code> (its demo '
     'prints one), <code>critique.py</code> and <code>hallucination.py</code> and in '
     "<code>attack/universal_suffixes_v0.json</code>; wording the charter rules out for the lab's own "
     'instruments in <code>adapters/guardrails.py</code> and <code>admissibility.py</code>; and an '
     'unsurveyed comparison on README line 422. None was surveyed. Pull request #165 withdraws them for '
     'the next release.</li>\n\n'
     '<li>The PyPI summary for 7.48.0 ties the HaluEval-QA AUC of 0.998 to the '
     '<code>@styxx.profile</code> readout. That number belongs to <code>guardrail.check</code> scored '
     'against a grounding passage, and the README row that reports it also reports two failures, DROP '
     '0.424 and FinanceBench 0.492.</li>'),
]
CLOSING = ("<p><em>This description was corrected on 2026-09-29 after an audit of the release's public "
           "surfaces. The DOI and the files did not change. The metadata as originally published and as "
           "edited are committed in pull request #167 as "
           "<code>release/zenodo-metadata-software-v7.48.0-as-published.json</code> and "
           "<code>release/zenodo-metadata-software-v7.48.0-edit-2026-09-29.json</code>.</em></p>")
NOTES_OLD = "that line is a known defect."
NOTES_NEW = ("that line is a known defect, corrected on the default branch after this deposit "
             "(a4732c52, #166). The description was corrected on 2026-09-29; the DOI and files are "
             "unchanged.")


def call(method: str, path: str, body: dict | None = None) -> dict:
    headers = {"Authorization": f"Bearer {TOKEN}", "User-Agent": UA}
    if body is not None:
        headers["Content-Type"] = "application/json"
    r = requests.request(method, f"{API}{path}", timeout=120, headers=headers,
                         data=json.dumps(body) if body is not None else None)
    if r.status_code >= 400:
        text = r.text.replace(TOKEN, "<token>") if TOKEN else r.text
        raise RuntimeError(f"HTTP {r.status_code} on {method} {path}: {text[:800]}")
    return r.json() if r.text else {}


def build(meta: dict) -> dict:
    m = json.loads(json.dumps(meta))
    d = m["description"]
    for old, new in DESC_EDITS:
        n = d.count(old)
        if n != 1:
            raise SystemExit(f"REFUSED: expected one occurrence, found {n}: {old[:90]!r}")
        d = d.replace(old, new)
    anchor = "<p><strong>Files in this record</strong></p>"
    if d.count(anchor) != 1:
        raise SystemExit("REFUSED: files anchor")
    m["description"] = d.replace(anchor, CLOSING + "\n\n" + anchor)
    if (m.get("notes") or "").count(NOTES_OLD) != 1:
        raise SystemExit("REFUSED: notes anchor")
    m["notes"] = m["notes"].replace(NOTES_OLD, NOTES_NEW)
    for x in m.get("related_identifiers", []):
        if x.get("identifier") in ("10.5281/zenodo.19746215", "10.5281/zenodo.19326174"):
            x["resource_type"] = "publication-workingpaper"
    text = (m["description"] + " " + m["notes"] + " " + m["title"]).lower()
    hits = [b for b in BANNED if any(f"{p}{b}" in text for p in (" ", ">", '"', "("))]
    if hits:
        raise SystemExit(f"REFUSED: charter words in the edited text: {hits}")
    return m


def check_record(d: dict) -> list[tuple[str, bool]]:
    files = {f.get("filename"): f.get("checksum", "").replace("md5:", "") for f in d.get("files") or []}
    return [("doi unchanged", (d.get("doi") or d["metadata"].get("doi")) == EXPECT["doi"]),
            ("concept unchanged", str(d.get("conceptrecid")) == EXPECT["conceptrecid"]),
            ("version unchanged", d["metadata"].get("version") == EXPECT["version"]),
            ("files unchanged", files == EXPECT["files"]),
            ("published", d.get("submitted") is True and d.get("state") == "done")]


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
    for n, ok in pre:
        print(f"  [{'PASS' if ok else 'FAIL'}] before: {n}")
    if not all(ok for _, ok in pre):
        return 2
    new_meta = build(before["metadata"])
    (HERE / "metadata.as_published.json").write_text(
        json.dumps(before["metadata"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (HERE / "metadata.edit_2026_09_29.json").write_text(
        json.dumps(new_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("built: metadata.as_published.json, metadata.edit_2026_09_29.json")
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
    live = (after["metadata"].get("notes") == new_meta["notes"]
            and "SAME_LINE 14" in (after["metadata"].get("description") or ""))
    post.append(("edited text is live", live))
    for n, ok in post:
        print(f"  [{'PASS' if ok else 'FAIL'}] after: {n}")
    (HERE / "metadata.after_edit_readback.json").write_text(
        json.dumps(after["metadata"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    receipt = {"action": "software_record_metadata_edited", "deposit_id": DEP, "doi": EXPECT["doi"],
               "checks_after": dict(post),
               "edited_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "fields_changed": ["description", "notes", "related_identifiers[].resource_type"]}
    (HERE / "zenodo-edit-receipt-software-v7.48.0-2026-09-29.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0 if all(ok for _, ok in post) else 4


if __name__ == "__main__":
    sys.exit(main())
