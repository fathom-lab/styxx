# -*- coding: utf-8 -*-
"""The builder on pull requests opened before 2026-11-02: items rendered from the collector's development store,
a P2 dry run under the stand-in hash with the invented decoy fixture, packet Q's selection, and the human seat's
forms. Writes only under the output folder it is given; reads the development store and nothing of the window.

    py -3.12 checks/builder_dev_check.py <development store> <output folder>

Nothing written here is a packet any seat reads: the P2 packet carries the stand-in hash, and the decoys are the
invented fixture of checks/synthetic_decoys.py, never sealed into a lock row.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import synthetic_decoys  # noqa: E402
import tpc_decoys  # noqa: E402
import tpc_items  # noqa: E402
import tpc_packet  # noqa: E402
import tpc_seats  # noqa: E402
from tpc_const import WINDOW_START, parse_utc  # noqa: E402


def main() -> int:
    store, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rows = tpc_packet._read_rows(store)
    late = [r for r in rows if parse_utc(r["created_at"]) >= WINDOW_START]
    if late:
        print(f"refused: {len(late)} rows opened on or after 2026-11-02")
        return 2
    report: dict = {"store": str(store), "rows": len(rows)}
    hows, sizes, desc_sizes, url_left, repo_left = Counter(), [], [], 0, 0
    rendered = 0
    for r in rows:
        if not r.get("eligible") or not (r.get("ci") or {}).get("has_ci_test_result"):
            continue
        if r.get("screen") != "claimer" and not (r.get("screen") == "none" and r["ci"]["outcome"] == "tests failed"):
            continue
        it = tpc_items.build_item(tpc_items.body_from_row(store, r), tpc_items.ci_from_row(store, r), r.get("repo"))
        rendered += 1
        sizes.append(len(it["item_text"]))
        desc_sizes.append(len(it["step1_text"]))
        for x in it["build"]["logs"]:
            hows[x["how"]] += 1
        url_left += len(re.findall(r"https?://", it["item_text"]))
        repo_left += it["item_text"].lower().count((r.get("repo") or "\0").lower())
    report["items_rendered"] = rendered
    report["log_excerpts_by_how"] = dict(hows)
    report["item_chars"] = {"max": max(sizes or [0]), "median": sorted(sizes)[len(sizes) // 2] if sizes else 0}
    report["description_chars_max"] = max(desc_sizes or [0])
    report["urls_left_in_items"] = url_left
    report["repository_names_left_in_items"] = repo_left
    set_obj, probs = tpc_decoys.build_set(synthetic_decoys.spec("P2"))
    assert not probs, probs
    built, record = tpc_packet.build_p2(store, set_obj["decoys"], beacon=tpc_packet.stand_in_beacon(), dry_run=True)
    pdir = out / "dryrun_P2"
    digests = tpc_packet.write_packet(pdir, built, record, salt_dir=pdir, dry_run=True)
    report["p2_dry_run"] = {"items": len(built["public"]["items"]), "selection": record["selection"],
                            "drawn_without_record": record["drawn_without_record"], "key_digest": built["key_digest"],
                            "packet_sha256": digests["packet_P2.json"]}
    class NoSeat:
        name = "C"

        def describe(self):
            return {}

        def send(self, spool_path):
            raise AssertionError("a stand-in packet reached a seat")
    try:
        tpc_seats.run_seat(pdir / "packet_P2.json", NoSeat(), out / "dryrun_P2_C")
        report["stand_in_packet_refused_by_seats"] = False
    except tpc_seats.SeatError as e:
        report["stand_in_packet_refused_by_seats"] = str(e)
    claim_rows = [r for r in rows if tpc_packet._claim_ci(r) and tpc_packet._ci_shows_failure(r)]
    report["q_candidates"] = len(claim_rows)
    if len(claim_rows) >= 30:
        q_set, _ = tpc_decoys.build_set(synthetic_decoys.spec("Q"))
        q_built, q_rec = tpc_packet.build_q(store, q_set["decoys"])
        qdir = out / "dryrun_Q"
        tpc_packet.write_packet(qdir, q_built, q_rec, salt_dir=qdir, dry_run=True)
        report["q_dry_run"] = {"items": len(q_built["public"]["items"]), "key_digest": q_built["key_digest"]}
        # a dry run's packet, read by no seat: its key digest is in no publication record, so the check stands in
        hx = tpc_seats.export_h(qdir / "packet_Q.json", 1, out / "dryrun_Q_H", published=lambda packet, sha: True)
        report["q_human_form"] = {"sittings": hx["sittings"], "items": hx["items"]}
    data = (json.dumps(report, indent=1) + "\n").encode("utf-8")
    (out / "builder_dev_check.json").write_bytes(data)
    sys.stdout.buffer.write(data)
    print("sha256", hashlib.sha256(data).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
