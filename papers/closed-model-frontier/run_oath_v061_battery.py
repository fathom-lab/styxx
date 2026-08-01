"""OATH v0.6.1 battery v2 — G1/G2/G2b per PREREG_oath_v061_shascrub_epsilon_2026_07_31.

Corrections over the failed v0.6 battery (all three measured mechanisms):
  (a) sampling conditioned on the doc's certificate receipts RESOLVING next to the doc;
  (b) sampling conditioned on the token's line being TRIGGER-BOUND (the verifier's own
      obligation precondition, `_TRIGGERS` or the v0.4 fractional-correlation register,
      evaluated with table-header binding context);
  (c) mutation restricted to a significant fractional digit among positions 1-6.
G2b reports the unbound-line share of the FULL unconditioned pool (the trigger-recall debt's
measured size on this class) beside the gate. Non-destructive: mutants in temp files only.
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
from styxx.certify import (certify_doc, extract_numbers,          # noqa: E402
                           _TRIGGERS, _TRIGGERS_CORR)
from styxx.corpus_audit import _resolve_receipts                  # noqa: E402

SEED = 1
N = 20
FULLPREC = re.compile(r"\d+\.\d{7,}")


def bound_context(doc_text: str, ln_no: int) -> str:
    """The binding context the verifier itself would use (line + table header if any)."""
    for e in extract_numbers(doc_text):
        if e["line"] == ln_no:
            return e.get("binding_context", e["context"])
    lines = doc_text.splitlines()
    return lines[ln_no - 1].strip() if ln_no <= len(lines) else ""


def is_bound(bctx: str, value: float, decimals: int) -> bool:
    if _TRIGGERS.search(bctx):
        return True
    return bool(decimals > 0 and -1.0 <= value <= 1.0 and _TRIGGERS_CORR.search(bctx))


def mutate_sig16(tok: str, rng: random.Random) -> str:
    """Perturb one significant fractional digit among positions 1-6."""
    frac_at = tok.index(".") + 1
    frac = tok[frac_at:]
    sig = [i for i in range(min(6, len(frac))) if not (frac[i] == "0" and set(frac[:i]) <= {"0"})]
    pos = frac_at + rng.choice(sig or [0])
    old = int(tok[pos])
    new = rng.choice([d for d in range(10) if d != old])
    return tok[:pos] + str(new) + tok[pos + 1:]


def main() -> int:
    rng = random.Random(SEED)
    pool, eligible = [], []
    for cp in sorted(ROOT.glob("papers/**/*.certificate.json")):
        doc = cp.with_name(cp.name.replace(".certificate.json", ".md"))
        if not doc.exists():
            continue
        text = doc.read_text(encoding="utf-8")
        cert = json.loads(cp.read_text(encoding="utf-8"))
        receipts, missing, _ = _resolve_receipts(cp, cert)
        resolvable = bool(receipts) and not missing
        for ln_no, line in enumerate(text.splitlines(), 1):
            for m in FULLPREC.finditer(line):
                tok = m.group(0)
                bctx = bound_context(text, ln_no)
                bound = is_bound(bctx, float(tok), len(tok.split(".")[1]))
                pool.append((doc.name, tok, bound))
                if resolvable and bound:
                    eligible.append((cp, doc, receipts, ln_no, tok))
    unbound_share = round(sum(1 for *_, b in pool if not b) / max(len(pool), 1), 4)

    seen, sample = set(), []
    for item in rng.sample(eligible, len(eligible)):
        key = (item[1].name, item[4])
        if key in seen:
            continue
        seen.add(key)
        sample.append(item)
        if len(sample) == N:
            break

    g1_hits, g2_caught, items = 0, 0, []
    for cp, doc, receipts, ln_no, tok in sample:
        text = doc.read_text(encoding="utf-8")
        lines = text.splitlines()
        line = lines[ln_no - 1]
        extracted = any(e["token"] == tok for e in extract_numbers(line))
        g1_hits += extracted
        mut = mutate_sig16(tok, rng)
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
        g2_caught += status == "UNGROUNDED"
        items.append({"doc": doc.name, "line": ln_no, "token": tok, "mutant": mut,
                      "extracted": bool(extracted), "mutant_status": status})

    report = {
        "prereg": "PREREG_oath_v061_shascrub_epsilon_2026_07_31.md",
        "seed": SEED, "n": N,
        "pool_size_unconditioned": len(pool),
        "pool_size_eligible": len(eligible),
        "G1": {"extracted": g1_hits, "bar": 18, "pass": g1_hits >= 18},
        "G2": {"caught_ungrounded": g2_caught, "bar": 16, "pass": g2_caught >= 16},
        "G2b_unbound_share_of_full_pool": unbound_share,
        "items": items,
    }
    out = HERE / "oath_v061_battery_result.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"pool: {len(pool)} full-precision tokens | eligible (resolvable+bound): {len(eligible)}")
    print(f"G1 recall: {g1_hits}/{N} (bar 18) -> {'PASS' if g1_hits >= 18 else 'FAIL'}")
    print(f"G2 catch : {g2_caught}/{N} (bar 16) -> {'PASS' if g2_caught >= 16 else 'FAIL'}")
    print(f"G2b unbound-line share of full pool: {unbound_share}")
    for it in items:
        if not it["extracted"] or it["mutant_status"] != "UNGROUNDED":
            print(f"  [miss] {it['doc']} L{it['line']} {it['token']} -> mutant={it['mutant_status']}")
    return 0 if (g1_hits >= 18 and g2_caught >= 16) else 1


if __name__ == "__main__":
    sys.exit(main())
