"""G-S3: build the fresh blind packet on HELD-OUT accusations after V13+V14.

Prereg: PREREG_v14_repair_2026_08_31.md. Same protocol as EXTERNAL-1 — 100
sampled accusations plus 30 sealed decoys, shuffled, adjudicator shown only the
claim and the PR's real changed files. New seed, new sample, held-out only.

Precision >= 0.95 here is the ONLY thing that re-enables the accusing verdict.

  python papers/closed-model-frontier/v14_packet.py build
  python papers/closed-model-frontier/v14_packet.py score
"""
from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import styxx.diffgate as DG                                    # noqa: E402
from external1_harness import reconstruct                      # noqa: E402

DB = HERE / "external1_shelf.sqlite"
PACKET = HERE / "v14_packet.json"
KEY = HERE / "v14_key_SEALED.json"
DIGEST = HERE / "v14_key_digest.txt"
ANSWERS = HERE / "v14_answers.json"
RESULT = HERE / "v14_adjudication.json"

SEED = 20260901
SALT = "styxx-v14-blind-2026-09-01"
N_ACC, N_VER, N_SYN = 100, 15, 15


def bucket(repo_url: str) -> int:
    n = (repo_url or "").strip().rstrip("/").lower()
    return int(hashlib.sha256(n.encode("utf-8")).hexdigest()[:8], 16) % 10


def build() -> int:
    DG.WITHHOLD_PATH_ACCUSATION = False        # measure the branch as it would accuse
    con = sqlite3.connect(DB)
    acc, ver = [], []
    facts = {}

    for pid, agent, title, body, url in con.execute(
            "SELECT id, agent, title, body, html_url FROM pr "
            "WHERE body IS NOT NULL AND body != ''"):
        if bucket("/".join((url or "").split("/")[:5])) < 3:
            continue                            # DEVELOPMENT — designed on, not scored
        rows = con.execute("SELECT filename, status, patch FROM f WHERE pr_id=?",
                           (pid,)).fetchall()
        if not rows:
            continue
        diff, implied = reconstruct(rows)
        parsed, _ = DG.parse_unified_diff(diff)
        if parsed != implied:
            continue
        g = DG.gate_diff_text(f"{title or ''}\n\n{body}", diff, run=None, strict=False)
        seen = {}
        for fn, st, _p in rows:
            if fn and fn not in seen:
                seen[fn] = (st or "").lower() or "modified"
        for i, c in enumerate(g.claims):
            if not c.kind.startswith("file_"):
                continue
            item = {"pr_id": pid, "agent": agent, "url": url, "claim_index": i,
                    "kind": c.kind, "text": c.text, "detail": c.detail}
            if c.verdict == "CONTRADICTED":
                acc.append(item)
                facts[pid] = seen
            elif c.verdict == "VERIFIED":
                ver.append(item)
                facts[pid] = seen
    con.close()
    print(f"HELD-OUT population after V13+V14: {len(acc)} accusations, "
          f"{len(ver)} verified")
    if len(acc) < N_ACC or len(ver) < N_VER + N_SYN:
        print("REFUSED: population too small for the frozen sample sizes")
        return 1

    rng = random.Random(SEED)
    sample_acc = rng.sample(acc, N_ACC)
    sample_ver = rng.sample(ver, N_VER + N_SYN)

    items, key = [], {}

    def add(it, truth, note):
        iid = f"V14-{len(items):03d}"
        fs = [{"path": p, "status": s} for p, s in facts[it["pr_id"]].items()]
        items.append({"id": iid, "agent": it["agent"], "url": it["url"],
                      "claim_kind": it["kind"], "claim_text": it["text"],
                      "claim_detail": it["detail"], "changed_files": fs})
        key[iid] = {"truth": truth, "note": note, "pr_id": it["pr_id"],
                    "claim_index": it["claim_index"]}

    for it in sample_acc:
        add(it, "gate_says_contradicted", "sampled accusation")
    for it in sample_ver[:N_VER]:
        add(it, "decoy_verified", "gate verified this")
    for it in sample_ver[N_VER:]:
        p2 = dict(it)
        d = dict(p2.get("detail") or {})
        path = d.get("path") or ""
        if path:
            parts = path.rsplit("/", 1)
            parts[-1] = "zz_" + parts[-1]
            newp = "/".join(parts)
            p2["text"] = p2["text"].replace(path, newp)
            d["path"] = newp
            p2["detail"] = d
        add(p2, "decoy_synthetic_contradiction",
            "verified claim with its path perturbed — must read as contradicted")

    rng.shuffle(items)
    PACKET.write_text(json.dumps(
        {"prereg": "PREREG_v14_repair_2026_08_31.md", "seed": SEED,
         "split": "HELD-OUT only", "n_items": len(items),
         "instructions": ("For each item answer SUPPORTED or CONTRADICTED: do the "
                          "PR's changed files support the claim, or contradict it? "
                          "You are not told the gate's verdict."),
         "items": items}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    body = json.dumps(key, sort_keys=True, ensure_ascii=False)
    KEY.write_text(body + "\n", encoding="utf-8")
    dig = hashlib.sha256((SALT + body).encode("utf-8")).hexdigest()
    DIGEST.write_text(f"sha256(salt+key) = {dig}\nsalt = {SALT}\n"
                      f"items = {len(items)}\nsplit = HELD-OUT\n", encoding="utf-8")
    print(f"packet: {len(items)} items -> {PACKET.name}")
    print(f"SEALED KEY DIGEST (commit before adjudicating):\n  {dig}")
    return 0


def score() -> int:
    key = json.loads(KEY.read_text(encoding="utf-8"))
    body = json.dumps(key, sort_keys=True, ensure_ascii=False)
    dig = hashlib.sha256((SALT + body).encode("utf-8")).hexdigest()
    if dig not in DIGEST.read_text(encoding="utf-8"):
        print("REFUSED: sealed key does not match the committed digest.")
        return 1
    ans = json.loads(ANSWERS.read_text(encoding="utf-8"))
    dec_ok = dec_n = tp = fp = 0
    misses = []
    for iid, t in key.items():
        a = ans.get(iid)
        if a is None:
            print(f"REFUSED: {iid} unanswered")
            return 1
        truth = t["truth"]
        if truth == "decoy_verified":
            dec_n += 1
            dec_ok += (a == "SUPPORTED")
            if a != "SUPPORTED":
                misses.append((iid, truth, a))
        elif truth == "decoy_synthetic_contradiction":
            dec_n += 1
            dec_ok += (a == "CONTRADICTED")
            if a != "CONTRADICTED":
                misses.append((iid, truth, a))
        else:
            tp += (a == "CONTRADICTED")
            fp += (a != "CONTRADICTED")
    reliable = dec_ok >= 27
    prec = tp / (tp + fp) if (tp + fp) else None
    out = {"split": "HELD-OUT", "decoys_correct": dec_ok, "decoys_total": dec_n,
           "adjudicator_reliable": reliable, "accusations_scored": tp + fp,
           "upheld": tp, "rejected": fp,
           "precision": None if not reliable else round(prec, 4),
           "G_S3_pass": bool(reliable and prec is not None and prec >= 0.95),
           "decoy_misses": misses}
    RESULT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    if not reliable:
        print("\nVOID — fewer than 27/30 decoys. No headline number.")
    return 0


if __name__ == "__main__":
    sys.exit(build() if (len(sys.argv) < 2 or sys.argv[1] == "build") else score())
