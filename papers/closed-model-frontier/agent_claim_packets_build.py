"""Build the blind ground-truth packets for PREREG_agent_claim_extractor_baseline_2026_08_30.

Construction, not measurement: everything here is deterministic (seed=20260830) and frozen by
the prereg before any seat runs. Key material never enters the repository at freeze — decoy
truths are read from, and the assembled answer key is written to, the sealed directory OUTSIDE
the repo; only the salted SHA-256 of the key is written in-repo. The plaintext key and salt are
committed after every seat output is recorded, and the fold verifies the hash.

  python papers/closed-model-frontier/agent_claim_packets_build.py
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

from styxx.diffgate import _TEMPLATES, _PATH_KINDS, _names_without_claiming   # noqa: E402

PIN_BASE = "origin/main"
PIN_HEAD = "a6994ac"
EXPECT_COMMITS = 57
EXPECT_SENTENCES = 2824
SEED = 20260830
N_SAMPLE = 294
N_PACKETS = 3
SEALED = Path(os.environ.get("STYXX_SEALED_DIR", r"C:\Users\heyzo\clawd\styxx-sealed"))

PACKETS_OUT = HERE / "agent_claim_packets.json"
KEYHASH_OUT = HERE / "agent_claim_key.sha256"


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout


def _sentences(msg: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", msg) if s.strip()]


def _flag_kinds(sent: str) -> list[str]:
    """The prereg's FLAGGED predicate: any template fires, after the referential guard."""
    kinds = []
    for kind, rx in _TEMPLATES:
        for m in rx.finditer(sent):
            if kind in _PATH_KINDS and _names_without_claiming(sent, m):
                continue
            kinds.append(kind)
            break
    return kinds


def main() -> int:
    shas = _git("rev-list", "--reverse", f"{PIN_BASE}..{PIN_HEAD}").split()
    assert len(shas) == EXPECT_COMMITS, f"pin drifted: {len(shas)} commits, prereg says {EXPECT_COMMITS}"

    corpus = []
    for sha in shas:
        for i, s in enumerate(_sentences(_git("log", "-1", "--format=%B", sha))):
            corpus.append({"sha": sha[:9], "idx": i, "text": s, "kinds": _flag_kinds(s)})
    assert len(corpus) == EXPECT_SENTENCES, \
        f"corpus drifted: {len(corpus)} sentences, prereg says {EXPECT_SENTENCES}"

    flagged = [c for c in corpus if c["kinds"]]
    unflagged = [c for c in corpus if not c["kinds"]]
    sample = random.Random(SEED).sample(unflagged, N_SAMPLE)

    # DEV / HELD-OUT split of commits, recorded publicly (the labels are what gets sealed).
    shuffled = shas[:]
    random.Random(SEED).shuffle(shuffled)
    n_dev = (2 * len(shuffled)) // 3
    dev = sorted(s[:9] for s in shuffled[:n_dev])
    heldout = sorted(s[:9] for s in shuffled[n_dev:])

    decoys = json.loads((SEALED / "agent_claim_decoys.json").read_text(encoding="utf-8"))
    assert len(decoys) == 30

    selected = flagged + sample
    random.Random(SEED).shuffle(selected)
    per = (len(selected) + N_PACKETS - 1) // N_PACKETS
    parts = [selected[i * per:(i + 1) * per] for i in range(N_PACKETS)]

    packets, key = [], {}
    for k, part in enumerate(parts, 1):
        items = ([{"source": "corpus", "sha": c["sha"], "idx": c["idx"],
                   "kinds": c["kinds"], "text": c["text"]} for c in part]
                 + [{"source": "decoy", "decoy_id": d["id"], "truth": d["truth"],
                     "class": d["class"], "text": d["text"]} for d in decoys])
        random.Random(SEED + k).shuffle(items)
        sents = []
        for i, it in enumerate(items, 1):
            sid = f"p{k}-{i:03d}"
            sents.append({"id": sid, "text": it["text"]})
            key[sid] = {kk: vv for kk, vv in it.items() if kk != "text"}
        packets.append({"packet": k, "n": len(sents), "sentences": sents})

    payload = {
        "prereg": "PREREG_agent_claim_extractor_baseline_2026_08_30.md",
        "pin": {"base": PIN_BASE, "head": PIN_HEAD, "commits": len(shas),
                "sentences": len(corpus)},
        "counts": {"flagged": len(flagged), "unflagged_remainder": len(unflagged),
                   "sampled": N_SAMPLE, "decoys_per_packet": len(decoys),
                   "packets": N_PACKETS},
        "split": {"dev_commits": dev, "heldout_commits": heldout},
        "packets": packets,
    }
    PACKETS_OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
                           encoding="utf-8")

    key_bytes = json.dumps(key, sort_keys=True, ensure_ascii=False).encode("utf-8")
    (SEALED / "agent_claim_key.json").write_bytes(key_bytes)
    salt = (SEALED / "agent_claim_key_salt.txt").read_text(encoding="utf-8").strip()
    digest = hashlib.sha256(key_bytes + salt.encode("utf-8")).hexdigest()
    KEYHASH_OUT.write_text(digest + "\n", encoding="utf-8")

    print(f"corpus {len(corpus)} sentences / {len(shas)} commits")
    print(f"flagged {len(flagged)}  (kinds: "
          f"{sorted(set(k for c in flagged for k in c['kinds']))})")
    print(f"sampled {N_SAMPLE} of {len(unflagged)} unflagged")
    print(f"packets: {[p['n'] for p in packets]}  (each carries all {len(decoys)} decoys)")
    print(f"split: {len(dev)} dev / {len(heldout)} held-out commits")
    print(f"key sealed outside repo; salted sha256 -> {KEYHASH_OUT.name}: {digest[:16]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
