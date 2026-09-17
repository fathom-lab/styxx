"""EXTERNAL-1 blind adjudication packet — build, then (separately) score.

Prereg: PREREG_external1_aidev_2026_08_31.md, seed 20260831.
100 accusations + 30 decoys (15 gate-VERIFIED, 15 synthetic contradictions made
by perturbing a verified claim's path). Shuffled. The adjudicator sees the claim
text and the PR's real changed-file facts — never the gate's verdict or reason.

The key is written to a SEALED file whose salted SHA-256 is printed at build
time and committed BEFORE any adjudication is recorded. Scoring refuses to run
unless the sealed key still hashes to the committed digest.

  python external1_packet.py build                  # packet + sealed key, ids after the shuffle
  python external1_packet.py build --as-published   # regenerates the PUBLISHED packet exactly
  python external1_packet.py score                  # after answers exist

Item ids (issue #125). The packet that was published and adjudicated was built by
a version of this file that numbered each item as it was added (100 sampled
accusations, then 15 gate-VERIFIED decoys, then 15 synthetic contradictions) and
shuffled the list afterwards. The shuffle moved the items and renumbered nothing,
so the id carries the arm: E1-000..E1-099 are accusations, E1-100..E1-114
verified decoys, E1-115..E1-129 synthetic contradictions. The last range can be
read off the published packet alone, because those fifteen items show the `zz_`
path perturbation. The protocol's blinding was weaker than it asserted, on the id.

`build` now assigns each id from the item's shuffled position, as
compat2_packet.py does, and refuses to write if the arms still cluster in id order.

Recipe for regenerating the published packet. external1_packet.json, the sealed
key and external1_key_digest.txt are EXTERNAL-1's receipts; the repair does not
rebuild them. Put the gitignored external1_shelf.sqlite in place, and as
external1_ledger.jsonl the ledger the packet was drawn from: the pre-correction
ledger whose counts are external1_summary_PREFIX.json (7,029 CONTRADICTED, 16,868
VERIFIED claims). The ledger regenerated for CORRECTION_external1_cause_2026_08_31.md
(665 / 17,887, external1_summary.json) yields a different sample under any version
of this file. Then run `build --as-published`. It draws the same sample and the
same shuffle (Random.shuffle's draws depend only on the list's length) and numbers
every item by its pre-shuffle, arm-order position, which reproduces the published
numbering, the key and the digest in external1_key_digest.txt. It reproduces the
leak with them, on purpose: the leak is part of the record. The published id order
follows from the two population counts alone, and
tests/test_external1_packet_ids.py pins it against the committed packet.

Neither mode overwrites an existing packet, key or digest whose contents would
change. A run under the repaired numbering is a new cycle with its own
preregistration and its own paths, never a rewrite of this one.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import random
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "external1_ledger.jsonl"
DB = HERE / "external1_shelf.sqlite"
PACKET = HERE / "external1_packet.json"
KEY = HERE / "external1_key_SEALED.json"
DIGEST = HERE / "external1_key_digest.txt"
ANSWERS = HERE / "external1_answers.json"
RESULT = HERE / "external1_adjudication.json"

SEED = 20260831
SALT = "styxx-external1-blind-2026-08-31"
N_ACC, N_VER, N_SYN = 100, 15, 15


def _facts(con, pr_id):
    rows = con.execute("SELECT filename, status FROM f WHERE pr_id=?", (pr_id,)).fetchall()
    seen, out = set(), []
    for fn, st in rows:
        if fn and fn not in seen:
            seen.add(fn)
            out.append({"path": fn, "status": (st or "").lower() or "modified"})
    return out


def arm_runs(arms_in_id_order) -> int:
    """How many maximal runs of one arm the id order contains (3 = every arm contiguous)."""
    return sum(1 for _arm, _g in itertools.groupby(arms_in_id_order))


def build(as_published: bool = False) -> int:
    rng = random.Random(SEED)
    acc, ver = [], []
    with LEDGER.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            for i, c in enumerate(r["claims"]):
                item = {"pr_id": r["pr_id"], "agent": r["agent"], "url": r["html_url"],
                        "claim_index": i, "kind": c["kind"], "text": c["text"],
                        "detail": c["detail"]}
                if c["verdict"] == "CONTRADICTED":
                    acc.append(item)
                elif c["verdict"] == "VERIFIED":
                    ver.append(item)
    print(f"population: {len(acc)} accusations, {len(ver)} verified")

    sample_acc = rng.sample(acc, N_ACC)
    sample_ver = rng.sample(ver, N_VER + N_SYN)
    decoy_ver = sample_ver[:N_VER]
    to_perturb = sample_ver[N_VER:]

    con = sqlite3.connect(DB)
    entries = []          # (arm-order position, packet fields without the id, key entry)

    def add(item, truth, note):
        facts = _facts(con, item["pr_id"])
        entries.append((len(entries),
                        {"agent": item["agent"], "url": item["url"],
                         "claim_kind": item["kind"], "claim_text": item["text"],
                         "claim_detail": item["detail"], "changed_files": facts},
                        {"truth": truth, "note": note, "pr_id": item["pr_id"],
                         "claim_index": item["claim_index"]}))

    for it in sample_acc:
        add(it, "gate_says_contradicted", "sampled accusation")
    for it in decoy_ver:
        add(it, "decoy_verified", "gate verified this")
    for it in to_perturb:
        p = dict(it)
        d = dict(p.get("detail") or {})
        path = d.get("path") or ""
        if path:
            parts = path.rsplit("/", 1)
            parts[-1] = "zz_" + parts[-1]
            newp = "/".join(parts)
            p["text"] = p["text"].replace(path, newp)
            d["path"] = newp
            p["detail"] = d
        add(p, "decoy_synthetic_contradiction",
            "verified claim with its path perturbed — must read as contradicted")

    con.close()
    # The shuffle is the published one in both modes: Random.shuffle's draws depend only on the
    # list's length. What differs is where the id comes from. The published packet numbered each
    # item in arm order before this shuffle, so the id carried the arm (#125); the repaired build
    # numbers it from its shuffled position, as compat2_packet.py does.
    rng.shuffle(entries)
    items, key = [], {}
    for pos, (arm_pos, fields, truth) in enumerate(entries):
        iid = f"E1-{(arm_pos if as_published else pos):03d}"
        items.append({"id": iid, **fields})
        key[iid] = truth
    arms_in_id_order = [key[iid]["truth"] for iid in sorted(key)]
    runs, n_arms = arm_runs(arms_in_id_order), len(set(arms_in_id_order))
    if as_published:
        print(f"AS PUBLISHED: ids numbered in arm order ({runs} runs over {n_arms} arms); "
              f"the id carries the arm (#125). This reproduces the record, leak included.")
    elif runs <= n_arms + 2:
        raise AssertionError(f"ids still cluster by arm ({runs} runs over {n_arms} arms); "
                             f"the shuffle did not take")

    packet_text = json.dumps(
        {"prereg": "PREREG_external1_aidev_2026_08_31.md", "seed": SEED,
         "n_items": len(items),
         "instructions": ("For each item answer SUPPORTED or CONTRADICTED: do the "
                          "PR's changed files support the claim, or contradict it? "
                          "You are not told the gate's verdict. Answer every item."),
         "items": items}, indent=1, ensure_ascii=False) + "\n"
    body = json.dumps(key, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256((SALT + body).encode("utf-8")).hexdigest()
    digest_text = f"sha256(salt+key) = {digest}\nsalt = {SALT}\nitems = {len(items)}\n"
    outputs = ((PACKET, packet_text), (KEY, body + "\n"), (DIGEST, digest_text))
    # An adjudicated packet is a receipt. Rewriting it under other ids would leave the answers
    # keyed to items they were not given, and `score` would still pass its digest check.
    clash = [p.name for p, text in outputs
             if p.exists() and p.read_text(encoding="utf-8") != text]
    if clash:
        print(f"REFUSED: {', '.join(clash)} already exist with different contents and are "
              f"EXTERNAL-1's receipts. To regenerate the published packet run "
              f"`build --as-published`; a build under the repaired numbering is a new cycle "
              f"with its own prereg and its own paths.")
        return 1
    for p, text in outputs:
        p.write_text(text, encoding="utf-8")
    print(f"packet: {len(items)} items -> {PACKET.name}")
    print(f"SEALED KEY DIGEST (commit this before adjudicating):\n  {digest}")
    return 0


def score() -> int:
    key = json.loads(KEY.read_text(encoding="utf-8"))
    body = json.dumps(key, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256((SALT + body).encode("utf-8")).hexdigest()
    committed = DIGEST.read_text(encoding="utf-8")
    if digest not in committed:
        print("REFUSED: sealed key does not match the committed digest.")
        return 1
    ans = json.loads(ANSWERS.read_text(encoding="utf-8"))
    dec_ok = dec_n = 0
    tp = fp = 0
    misses = []
    for iid, truth in key.items():
        a = ans.get(iid)
        if a is None:
            print(f"REFUSED: item {iid} unanswered")
            return 1
        t = truth["truth"]
        if t == "decoy_verified":
            dec_n += 1
            dec_ok += (a == "SUPPORTED")
            if a != "SUPPORTED":
                misses.append((iid, t, a))
        elif t == "decoy_synthetic_contradiction":
            dec_n += 1
            dec_ok += (a == "CONTRADICTED")
            if a != "CONTRADICTED":
                misses.append((iid, t, a))
        else:
            if a == "CONTRADICTED":
                tp += 1
            else:
                fp += 1
    reliable = dec_ok >= 27
    precision = tp / (tp + fp) if (tp + fp) else None
    out = {"decoys_correct": dec_ok, "decoys_total": dec_n,
           "adjudicator_reliable": reliable,
           "accusations_scored": tp + fp,
           "accusations_upheld": tp, "accusations_rejected": fp,
           "precision": None if not reliable else round(precision, 4),
           "gate_G_E1_pass": bool(reliable and precision is not None
                                  and precision >= 0.95),
           "decoy_misses": misses}
    RESULT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    if not reliable:
        print("\nADJUDICATION VOID — fewer than 27/30 decoys correct. "
              "No headline number may be published.")
    return 0


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    cmd = args.pop(0) if args else "build"
    if cmd == "build" and args in ([], ["--as-published"]):
        return build(as_published=bool(args))
    if cmd == "score" and not args:
        return score()
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
