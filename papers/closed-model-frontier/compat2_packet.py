"""COMPAT-2 blind adjudication packet — build now, score in the panel's own cycle.

Prereg: PREREG_compat2_surface_and_panel_2026_09_16.md, seed 20260917.  Three arms from the
COMPAT-2 ledger (`compat2_ledger.jsonl`, the HARNESS-1 fold under the COMPAT-2 reading):

  candidates  every PR whose compat reading is a candidate (a covered language, at least one
              removed public definition on the surface); a seeded sample of 120 if there are more
  decoy_a     30 PRs with a compatibility claim, a covered language in the diff, and no removed
              public name (the reading's miss rate)
  decoy_b     30 PRs whose removed public names are all scaffolding (whether the scaffolding rule
              hides breaks)

Shuffled into one packet.  An item carries the compatibility sentence, the PR title, and the
reconstructed diff of every file with removed lines — capped at 400 lines per file and 3,000 per
item, the cap stated on the item — and never the gate's reason, the removed names, the arm, or
the PR's URL.  The key (item -> arm, pr_id, claim index) is written SEALED; its salted SHA-256 is
printed at build time and committed before any answer exists.  `score` refuses to run unless the
sealed key still hashes to the committed digest.

The question every seat answers, verbatim from the prereg:

  Does this diff remove or change something a user of this project's public interface could
  already be depending on — a function, class, type, export, endpoint, option — so that
  "no breaking changes / backward compatible" is false for that user?
  Answers: break | compatible | cannot tell

  python compat2_packet.py build     # writes compat2_packet.json, compat2_key_SEALED.json, compat2_key_digest.txt
  python compat2_packet.py score     # reads compat2_answers.json ({item_id: {seat: answer}}), writes compat2_adjudication.json
"""
from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

LEDGER = HERE / "compat2_ledger.jsonl"
DB = HERE / "external6_shelf.sqlite"
PACKET = HERE / "compat2_packet.json"
KEY = HERE / "compat2_key_SEALED.json"
DIGEST = HERE / "compat2_key_digest.txt"
ANSWERS = HERE / "compat2_answers.json"
RESULT = HERE / "compat2_adjudication.json"

SEED = 20260917
SALT = "styxx-compat2-blind-2026-09-17"
N_CAND, N_A, N_B = 120, 30, 30
CAP_FILE, CAP_ITEM = 400, 3000
QUESTION = ("Does this diff remove or change something a user of this project's public interface could "
            "already be depending on — a function, class, type, export, endpoint, option — so that "
            '"no breaking changes / backward compatible" is false for that user? '
            "Answer exactly one of: break | compatible | cannot tell")
FLOOR_LICENSE, FLOOR_NARROW, DECOY_B_RETIRE = 0.95, 0.80, 0.20


def _diff_of_removed_files(pr_id: int, con) -> tuple[str, dict]:
    from external1_harness import reconstruct
    from external6_harness import fold_rows

    rows = con.execute("SELECT sha, filename, status, patch FROM f WHERE pr_id=?", (pr_id,)).fetchall()
    commits = con.execute("SELECT sha, message, rows FROM c WHERE pr_id=?", (pr_id,)).fetchall()
    files, _facts = fold_rows(commits, rows)
    diff, _implied = reconstruct(files)
    chunks = ["diff --git " + c for c in diff.split("diff --git ") if c.strip()]
    kept, total, cut_files, cut_item = [], 0, 0, False
    for ch in chunks:
        lines = ch.splitlines()
        if not any(l.startswith("-") and not l.startswith("---") for l in lines):
            continue
        if len(lines) > CAP_FILE:
            lines = lines[:CAP_FILE] + [f"… [{len(ch.splitlines()) - CAP_FILE} more lines of this file not shown]"]
            cut_files += 1
        if total + len(lines) > CAP_ITEM:
            cut_item = True
            break
        kept.append("\n".join(lines))
        total += len(lines)
    text = "\n".join(kept)
    if cut_item:
        text += f"\n… [further files with removed lines not shown: the item cap of {CAP_ITEM} lines was reached]"
    return text, {"files_shown": len(kept), "files_with_removed_lines": sum(
        1 for c in chunks if any(l.startswith("-") and not l.startswith("---") for l in c.splitlines())),
        "files_truncated": cut_files, "item_truncated": cut_item, "lines": total}


def build() -> int:
    rng = random.Random(SEED)
    cand, dec_a, dec_b = [], [], []
    with LEDGER.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            for i, c in enumerate(r["claims"]):
                if c["kind"] != "compat_claim":
                    continue
                d = c["detail"]
                item = {"pr_id": r["pr_id"], "title": None, "claim_index": i, "text": c["text"]}
                if d.get("compat2_candidate"):
                    cand.append(item)
                elif d.get("languages") and not d.get("removed"):
                    dec_a.append(item)
                elif d.get("languages") and d.get("removed") and not d.get("compat2_candidate"):
                    dec_b.append(item)
                break                                # one reading per PR: the diff is the same
    print(f"population: {len(cand)} candidates, {len(dec_a)} decoy_a-eligible, {len(dec_b)} decoy_b-eligible")
    sample_c = rng.sample(cand, min(N_CAND, len(cand)))
    sample_a = rng.sample(dec_a, min(N_A, len(dec_a)))
    sample_b = rng.sample(dec_b, min(N_B, len(dec_b)))

    con = sqlite3.connect(DB)
    items = []

    def add(item, arm):
        title = con.execute("SELECT title FROM pr WHERE id=?", (item["pr_id"],)).fetchone()[0] or ""
        diff, facts = _diff_of_removed_files(item["pr_id"], con)
        items.append({"_arm": arm, "_pr_id": item["pr_id"], "_claim_index": item["claim_index"],
                      "claim_text": item["text"], "pr_title": title, "diff": diff,
                      "diff_facts": facts, "question": QUESTION})

    for it in sample_c:
        add(it, "candidate")
    for it in sample_a:
        add(it, "decoy_a")
    for it in sample_b:
        add(it, "decoy_b")
    con.close()
    # The id is assigned AFTER the shuffle, never before it.  EXTERNAL-1's packet numbered its
    # items in arm order and shuffled the list afterwards, so the id itself carried the arm: in
    # the published packet the fifteen synthetic decoys are exactly the last fifteen ids, readable
    # without the sealed key.  Filed as its own issue; here the order is the only thing shuffled
    # and the id is drawn from the shuffled position, so no item's id says anything about its arm.
    rng.shuffle(items)
    key = {}
    for n, it in enumerate(items):
        iid = f"C2-{n:03d}"
        key[iid] = {"arm": it.pop("_arm"), "pr_id": it.pop("_pr_id"), "claim_index": it.pop("_claim_index")}
        it["id"] = iid
    arms_in_id_order = [key[f"C2-{n:03d}"]["arm"] for n in range(len(items))]
    runs = [a for a, _g in __import__("itertools").groupby(arms_in_id_order)]
    assert len(runs) > len(set(arms_in_id_order)) + 2, "ids still cluster by arm; the shuffle did not take"
    PACKET.write_text(json.dumps(
        {"prereg": "PREREG_compat2_surface_and_panel_2026_09_16.md", "seed": SEED, "n_items": len(items),
         "blinding": ("item ids are assigned after the shuffle, so an id carries nothing about its arm; "
                      "the packet contains no verdict, no reason, no removed names and no URL"),
         "caps": {"lines_per_file": CAP_FILE, "lines_per_item": CAP_ITEM},
         "instructions": ("For each item read the compatibility sentence, the PR title and the diff of the "
                          "files with removed lines, and answer the question with exactly one of: break, "
                          "compatible, cannot tell. You are not told what the instrument read, which "
                          "items it flagged, or how many it flagged. Answer every item."),
         "question": QUESTION, "items": items}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    body = json.dumps(key, sort_keys=True, ensure_ascii=False)
    KEY.write_text(body + "\n", encoding="utf-8")
    digest = hashlib.sha256((SALT + body).encode("utf-8")).hexdigest()
    arms = Counter(v["arm"] for v in key.values())
    DIGEST.write_text(f"sha256(salt+key) = {digest}\nsalt = {SALT}\nitems = {len(items)}\n"
                      f"arms = {json.dumps(dict(arms), sort_keys=True)}\n"
                      f"packet sha256 = {hashlib.sha256(PACKET.read_bytes()).hexdigest()}\n", encoding="utf-8")
    print(f"packet: {len(items)} items ({dict(arms)}) -> {PACKET.name} ({PACKET.stat().st_size:,} bytes)")
    print(f"SEALED KEY DIGEST (commit this before adjudicating):\n  {digest}")
    return 0


def score() -> int:
    key = json.loads(KEY.read_text(encoding="utf-8"))
    body = json.dumps(key, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256((SALT + body).encode("utf-8")).hexdigest()
    committed = DIGEST.read_text(encoding="utf-8").splitlines()[0].split("=")[-1].strip()
    if digest != committed:
        sys.exit(f"sealed key hashes to {digest[:16]}…, not the committed {committed[:16]}…; refusing to score")
    answers = json.loads(ANSWERS.read_text(encoding="utf-8"))
    seats = sorted({s for v in answers.values() for s in v})
    if len(seats) < 3:
        sys.exit(f"three seats are required; found {seats}")
    per_arm = {"candidate": Counter(), "decoy_a": Counter(), "decoy_b": Counter()}
    disagree = 0
    unanswered = 0
    for iid, meta in key.items():
        votes = answers.get(iid) or {}
        if len(votes) < len(seats):
            unanswered += 1
        norm = [str(votes.get(s, "cannot tell")).strip().lower() for s in seats]
        n_break = norm.count("break")
        n_compat = norm.count("compatible")
        if len(set(norm)) > 1:
            disagree += 1
        # majority decides; `cannot tell` and ties resolve AGAINST the instrument
        if n_break > len(seats) / 2:
            call = "break"
        elif n_compat > len(seats) / 2:
            call = "compatible"
        else:
            call = "against_instrument"
        per_arm[meta["arm"]][call] += 1
    c = per_arm["candidate"]
    n_c = sum(c.values())
    precision = c["break"] / n_c if n_c else None
    # on the candidate arm an unresolved item is a false accusation; on the decoy arms it is a break
    a, b = per_arm["decoy_a"], per_arm["decoy_b"]
    a_break = (a["break"] + a["against_instrument"]) / sum(a.values()) if sum(a.values()) else None
    b_break = (b["break"] + b["against_instrument"]) / sum(b.values()) if sum(b.values()) else None
    outcome = ("licensed" if precision is not None and precision >= FLOOR_LICENSE else
               "not_licensed_narrow_the_rule" if precision is not None and precision >= FLOOR_NARROW else
               "rule_is_wrong")
    payload = {"prereg": "PREREG_compat2_surface_and_panel_2026_09_16.md", "seats": seats,
               "items": len(key), "items_missing_a_vote": unanswered, "items_with_seat_disagreement": disagree,
               "per_arm": {k: dict(v) for k, v in per_arm.items()},
               "candidate_precision": precision, "decoy_a_break_rate": a_break, "decoy_b_break_rate": b_break,
               "floors": {"license": FLOOR_LICENSE, "narrow": FLOOR_NARROW, "decoy_b_retires_the_filter_at": DECOY_B_RETIRE},
               "outcome": outcome,
               "scaffolding_filter": ("retired as a filter" if b_break is not None and b_break >= DECOY_B_RETIRE else "stands")}
    RESULT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=1))
    return 0


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in ("build", "score"):
        sys.exit(__doc__)
    return build() if sys.argv[1] == "build" else score()


if __name__ == "__main__":
    sys.exit(main())
