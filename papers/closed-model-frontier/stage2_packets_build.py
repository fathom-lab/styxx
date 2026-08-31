"""Build Stage 2 packets: PREREG_claim_detector + AMENDMENT_..._stage2 (census of 38).

Deterministic (seed 20260831). Flagged arm is a CENSUS — every available STRUCT-1 flag, no
selection freedom. Controls matched at the same n from the same available pool. The same 30
frozen decoys ride in every packet. Truths never enter the repository at build time: the key
is written to the sealed directory and only its salted SHA-256 is committed here.

  python papers/closed-model-frontier/stage2_packets_build.py
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx.claimdetect import detect, STRUCT1_VERSION                   # noqa: E402
from styxx.diffgate import _TEMPLATES, _PATH_KINDS, _names_without_claiming   # noqa: E402

PIN_BASE, PIN_HEAD = "origin/main", "a6994ac"
SEED = 20260831
N_PACKETS = 3
SEALED = Path(os.environ.get("STYXX_SEALED_DIR", r"C:\Users\heyzo\clawd\styxx-sealed"))
PACKETS_OUT = HERE / "stage2_packets.json"
KEYHASH_OUT = HERE / "stage2_key.sha256"


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=str(ROOT), capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout


def sentences(msg: str):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", msg) if s.strip()]


def template_flags(s: str) -> bool:
    for kind, rx in _TEMPLATES:
        for m in rx.finditer(s):
            if kind in _PATH_KINDS and _names_without_claiming(s, m):
                continue
            return True
    return False


def main() -> int:
    prior = json.loads((HERE / "agent_claim_packets.json").read_text(encoding="utf-8"))
    already = {s["text"] for p in prior["packets"] for s in p["sentences"]}

    shas = git("rev-list", "--reverse", f"{PIN_BASE}..{PIN_HEAD}").split()
    corpus = []
    for sha in shas:
        for s in sentences(git("log", "-1", "--format=%B", sha)):
            corpus.append({"sha": sha[:9], "text": s})

    avail = [c for c in corpus if c["text"] not in already and not template_flags(c["text"])]
    flagged = [c for c in avail if detect(c["text"]).is_claim]
    unflagged = [c for c in avail if not detect(c["text"]).is_claim]

    # dedupe by text, keeping first occurrence — a repeated sentence is one claim to judge
    seen, flagged_u = set(), []
    for c in flagged:
        if c["text"] not in seen:
            seen.add(c["text"])
            flagged_u.append(c)
    unflagged_u, seen2 = [], set()
    for c in unflagged:
        if c["text"] not in seen2 and c["text"] not in seen:
            seen2.add(c["text"])
            unflagged_u.append(c)

    n = len(flagged_u)                       # the census: however many exist
    controls = random.Random(SEED).sample(unflagged_u, n)
    print(f"flagged CENSUS n={n}   controls matched n={len(controls)}")
    assert n >= 30, f"below the amended floor before adjudication even ran: {n}"

    decoys = json.loads((SEALED / "agent_claim_decoys.json").read_text(encoding="utf-8"))
    assert len(decoys) == 30

    items = ([{"arm": "flagged", "sha": c["sha"], "text": c["text"]} for c in flagged_u]
             + [{"arm": "control", "sha": c["sha"], "text": c["text"]} for c in controls])
    random.Random(SEED).shuffle(items)
    parts = [items[i::N_PACKETS] for i in range(N_PACKETS)]      # even, deterministic

    packets, key = [], {}
    for k, part in enumerate(parts, 1):
        blob = part + [{"arm": "decoy", "decoy_id": d["id"], "truth": d["truth"],
                        "class": d["class"], "text": d["text"]} for d in decoys]
        random.Random(SEED + k).shuffle(blob)
        rows = []
        for i, it in enumerate(blob, 1):
            sid = f"s2p{k}-{i:03d}"
            rows.append({"id": sid, "text": it["text"]})
            key[sid] = {kk: vv for kk, vv in it.items() if kk != "text"}
        packets.append({"packet": k, "n": len(rows), "sentences": rows})

    payload = {
        "stage": 2,
        "prereg": "PREREG_claim_detector_2026_08_30.md",
        "amendment": "AMENDMENT_claim_detector_stage2_2026_08_31.md",
        "struct1_version": STRUCT1_VERSION,
        "pin": {"base": PIN_BASE, "head": PIN_HEAD, "commits": len(shas),
                "sentences": len(corpus)},
        "counts": {"available_after_exclusions": len(avail),
                   "flagged_census": n, "controls": len(controls),
                   "decoys_per_packet": len(decoys), "packets": N_PACKETS},
        "gates": {"G-S2P_bar": 0.2061, "G-S2LIFT": "flagged A-share > control A-share",
                  "floor_per_arm": 30},
        "packets": packets,
    }
    PACKETS_OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
                           encoding="utf-8")

    kb = json.dumps(key, sort_keys=True, ensure_ascii=False).encode("utf-8")
    (SEALED / "stage2_key.json").write_bytes(kb)
    salt = (SEALED / "agent_claim_key_salt.txt").read_text(encoding="utf-8").strip()
    digest = hashlib.sha256(kb + salt.encode("utf-8")).hexdigest()
    KEYHASH_OUT.write_text(digest + "\n", encoding="utf-8")

    print(f"packets: {[p['n'] for p in packets]} (each carries all {len(decoys)} decoys)")
    print(f"key sealed outside repo; salted sha256 -> {KEYHASH_OUT.name}: {digest[:16]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
