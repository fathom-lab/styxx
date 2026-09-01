"""Build the corpus census receipt: every certificate in the repo, hashed.

One JSON leaf per certificate — name, verdict, counts, sha256 of the stored
certificate bytes — plus the totals. This is the receipt the corpus capsule
seals: change any certificate anywhere in the program's history and the
census hash no longer matches it.

  python papers/closed-model-frontier/build_corpus_census.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx.corpus_audit import discover_certificates          # noqa: E402

OUT = HERE / "corpus_census.json"


def main() -> int:
    certs = {}
    held = failed = 0
    for cp in sorted(discover_certificates(ROOT / "papers")):
        if cp.name.startswith("CORPUS_STATE_"):
            # The snapshot cannot contain its own oath — the CORPUS_STATE
            # certificate postdates the census it certifies, by construction.
            continue
        b = cp.read_bytes()
        c = json.loads(b.decode("utf-8"))
        v = c.get("verdict", "?")
        held += v == "OATH-HELD"
        failed += v == "OATH-FAILED"
        certs[cp.name] = {
            "verdict": v,
            "counts": c.get("counts"),
            "sha256": hashlib.sha256(b).hexdigest(),
        }
    payload = {
        "census": "styxx-corpus/v1",
        "totals": {"certificates": len(certs), "held": held, "failed": failed},
        "certificates": certs,
    }
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"census: {len(certs)} certificates | HELD {held}  FAILED {failed}")
    print(f"-> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
