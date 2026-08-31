"""EXTERNAL-1 blind adjudication packet — build, then (separately) score.

Prereg: PREREG_external1_aidev_2026_08_31.md, seed 20260831.
100 accusations + 30 decoys (15 gate-VERIFIED, 15 synthetic contradictions made
by perturbing a verified claim's path). Shuffled. The adjudicator sees the claim
text and the PR's real changed-file facts — never the gate's verdict or reason.

The key is written to a SEALED file whose salted SHA-256 is printed at build
time and committed BEFORE any adjudication is recorded. Scoring refuses to run
unless the sealed key still hashes to the committed digest.

  python external1_packet.py build            # writes packet + sealed key
  python external1_packet.py score            # after answers exist
"""
from __future__ import annotations

import hashlib
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


def build() -> int:
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
    items, key = [], {}

    def add(item, truth, note):
        iid = f"E1-{len(items):03d}"
        facts = _facts(con, item["pr_id"])
        items.append({"id": iid, "agent": item["agent"], "url": item["url"],
                      "claim_kind": item["kind"], "claim_text": item["text"],
                      "claim_detail": item["detail"], "changed_files": facts})
        key[iid] = {"truth": truth, "note": note, "pr_id": item["pr_id"],
                    "claim_index": item["claim_index"]}

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
    rng.shuffle(items)
    PACKET.write_text(json.dumps(
        {"prereg": "PREREG_external1_aidev_2026_08_31.md", "seed": SEED,
         "n_items": len(items),
         "instructions": ("For each item answer SUPPORTED or CONTRADICTED: do the "
                          "PR's changed files support the claim, or contradict it? "
                          "You are not told the gate's verdict. Answer every item."),
         "items": items}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    body = json.dumps(key, sort_keys=True, ensure_ascii=False)
    KEY.write_text(body + "\n", encoding="utf-8")
    digest = hashlib.sha256((SALT + body).encode("utf-8")).hexdigest()
    (HERE / "external1_key_digest.txt").write_text(
        f"sha256(salt+key) = {digest}\nsalt = {SALT}\nitems = {len(items)}\n",
        encoding="utf-8")
    print(f"packet: {len(items)} items -> {PACKET.name}")
    print(f"SEALED KEY DIGEST (commit this before adjudicating):\n  {digest}")
    return 0


def score() -> int:
    key = json.loads(KEY.read_text(encoding="utf-8"))
    body = json.dumps(key, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256((SALT + body).encode("utf-8")).hexdigest()
    committed = (HERE / "external1_key_digest.txt").read_text(encoding="utf-8")
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


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    sys.exit(build() if cmd == "build" else score())
