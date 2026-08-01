"""OATH v0.6 battery — G1 (recall) + G2 (catch) per PREREG_oath_v06_shascrub_recall_2026_07_31.

Draws a seeded sample of 20 distinct >=7-fractional-digit decimal tokens from certified docs,
checks post-fix extraction (G1 >=18/20), then single-digit-mutates each in place and certifies
the mutated doc against its certificate's recorded receipt set (G2 >=16/20 UNGROUNDED).
Non-destructive: mutants are written to temp files only.
"""
from __future__ import annotations

import json
import random
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from styxx.certify import certify_doc, extract_numbers          # noqa: E402
from styxx.corpus_audit import mutate_token, _resolve_receipts  # noqa: E402

SEED = 1
N = 20
FULLPREC = re.compile(r"\d+\.\d{7,}")


def main() -> int:
    rng = random.Random(SEED)
    # candidate pool: (cert_path, doc_path, line_no, token) for every full-precision token
    pool = []
    for cp in sorted(ROOT.glob("papers/**/*.certificate.json")):
        doc = cp.with_name(cp.name.replace(".certificate.json", ".md"))
        if not doc.exists():
            continue
        for ln_no, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            for m in FULLPREC.finditer(line):
                pool.append((cp, doc, ln_no, m.group(0)))
    # sample 20 distinct tokens (distinct by (doc, token))
    seen, sample = set(), []
    for item in rng.sample(pool, len(pool)):
        key = (item[1].name, item[3])
        if key in seen:
            continue
        seen.add(key)
        sample.append(item)
        if len(sample) == N:
            break

    g1_hits, g2_caught, items = 0, 0, []
    for cp, doc, ln_no, tok in sample:
        text = doc.read_text(encoding="utf-8")
        lines = text.splitlines()
        line = lines[ln_no - 1]
        extracted = any(e["token"] == tok and e["line"] == 1
                        for e in extract_numbers(line))
        g1_hits += extracted
        # G2: mutate in place, certify against the cert's receipt set
        cert = json.loads(cp.read_text(encoding="utf-8"))
        receipts, missing, _ = _resolve_receipts(cp, cert)
        status = None
        if receipts and not missing:
            mut = mutate_token(tok, rng)
            ml = list(lines)
            ml[ln_no - 1] = line.replace(tok, mut, 1)
            with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False,
                                             encoding="utf-8") as tf:
                tf.write("\n".join(ml))
                tmp = Path(tf.name)
            try:
                mc = certify_doc(tmp, receipts)
            finally:
                tmp.unlink(missing_ok=True)
            status = next((e["status"] for e in mc["ledger"]
                           if e["line"] == ln_no and e["token"] == mut), "NOT_EXTRACTED")
        else:
            status = "RECEIPTS_UNRESOLVED"
        caught = status == "UNGROUNDED"
        g2_caught += caught
        items.append({"doc": doc.name, "line": ln_no, "token": tok,
                      "extracted": bool(extracted), "mutant_status": status})

    report = {
        "prereg": "PREREG_oath_v06_shascrub_recall_2026_07_31.md",
        "seed": SEED, "n": N, "pool_size": len(pool),
        "G1": {"extracted": g1_hits, "bar": 18, "pass": g1_hits >= 18},
        "G2": {"caught_ungrounded": g2_caught, "bar": 16, "pass": g2_caught >= 16},
        "items": items,
    }
    out = HERE / "oath_v06_battery_result.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"pool {len(pool)} full-precision tokens across certified docs")
    print(f"G1 recall: {g1_hits}/{N} extracted (bar 18) -> {'PASS' if g1_hits >= 18 else 'FAIL'}")
    print(f"G2 catch : {g2_caught}/{N} UNGROUNDED (bar 16) -> {'PASS' if g2_caught >= 16 else 'FAIL'}")
    for it in items:
        if not it["extracted"] or it["mutant_status"] != "UNGROUNDED":
            print(f"  [miss] {it['doc']} L{it['line']} {it['token']} -> "
                  f"extracted={it['extracted']} mutant={it['mutant_status']}")
    return 0 if (g1_hits >= 18 and g2_caught >= 16) else 1


if __name__ == "__main__":
    sys.exit(main())
