"""OATH v0 corpus attestation — put EVERY finding document in the repo under oath.

Dogfood of styxx.certify at scale: for each git-tracked FINDING document, resolve the receipt JSONs it
explicitly cites, ground every numeric claim against them, and emit one corpus-level attestation receipt.
Deterministic, local, $0 — anyone can re-run it.

Receipt binding (v0 corpus rule, stated not hidden):
  - A doc grounds ONLY against .json files it cites by name (no receipt discovery — citing is part of
    the oath; an uncited receipt grounds nothing).
  - Mentions are resolved same-dir first, then repo-relative, then by basename within the doc's paper
    dir. Unresolved citations are reported — a dead receipt pointer is itself a provenance gap.
  - `*_SMOKE_INVALID*` receipts are never read (smoke runs are not results — frozen discipline).
  - Docs with zero resolvable receipts are classed UNBOUND and excluded from OATH-FAILED stats: that is
    a binding limitation of the corpus runner, not a verdict on the doc.
  - `_oath_mutants/` fixtures are excluded (they are the validator's corrupted-doc battery).

Output:
  papers/closed-model-frontier/oath_corpus_attestation.json   (machine receipt)

Usage:
  python papers/closed-model-frontier/run_oath_corpus.py
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
from styxx.certify import certify_doc  # noqa: E402

_JSON_MENTION = re.compile(r"[\w./\\-]+\.json\b")


def finding_docs() -> list[Path]:
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "papers/*FINDING*"],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    return [REPO / p for p in out
            if p.endswith(".md") and "_oath_mutants/" not in p]


def resolve_receipts(doc: Path, text: str) -> tuple[list[Path], list[str]]:
    """Receipt .json files the doc cites, resolved to real paths; plus unresolved citations."""
    resolved: dict[Path, None] = {}   # ordered set
    unresolved: list[str] = []
    for raw in dict.fromkeys(_JSON_MENTION.findall(text)):  # unique, doc order
        name = raw.replace("\\", "/").lstrip("./")
        if "SMOKE_INVALID" in name or name.endswith(".certificate.json"):
            continue   # never ground in smoke runs; certs attest docs, they are not receipts
        cands = [doc.parent / Path(name).name, REPO / name, doc.parent / name]
        hit = next((c for c in cands if c.is_file()), None)
        if hit is None:
            hits = sorted(doc.parent.rglob(Path(name).name))
            hit = hits[0] if hits else None
        if hit is not None:
            resolved[hit.resolve()] = None
        else:
            unresolved.append(raw)
    return list(resolved), unresolved


def main() -> int:
    docs = finding_docs()
    print(f"corpus: {len(docs)} finding docs under oath\n", flush=True)
    t0 = time.time()
    entries, failed, unbound, broken = [], [], [], []
    for doc in docs:
        rel = doc.relative_to(REPO).as_posix()
        text = doc.read_text(encoding="utf-8", errors="replace")
        receipts, unresolved = resolve_receipts(doc, text)
        entry = {"doc": rel, "n_receipts": len(receipts),
                 "receipts": [r.relative_to(REPO).as_posix() if r.is_relative_to(REPO) else str(r)
                              for r in receipts],
                 "unresolved_citations": unresolved}
        if not receipts:
            entry["class"] = "UNBOUND"
            unbound.append(entry)
            entries.append(entry)
            continue
        try:
            cert = certify_doc(doc, receipts)
        except Exception as e:   # malformed receipt etc. — report, never crash the corpus pass
            entry.update({"class": "BINDING-ERROR", "error": f"{type(e).__name__}: {e}"})
            broken.append(entry)
            entries.append(entry)
            continue
        entry.update({"class": "CERTIFIED", "verdict": cert["verdict"], "counts": cert["counts"],
                      "ungrounded": [{"line": u["line"], "token": u["token"], "context": u["context"]}
                                     for u in cert["ungrounded"]]})
        if cert["verdict"] != "OATH-HELD":
            failed.append(entry)
        entries.append(entry)
        mark = "OK " if cert["verdict"] == "OATH-HELD" else "FAIL"
        c = cert["counts"]
        print(f"  [{mark}] {rel}  v={c['VERIFIED']} a={c['ABSTAIN']} u={c['UNGROUNDED']}", flush=True)

    certified = [e for e in entries if e["class"] == "CERTIFIED"]
    held = [e for e in certified if e["verdict"] == "OATH-HELD"]
    totals = {k: sum(e["counts"][k] for e in certified) for k in ("VERIFIED", "ABSTAIN", "UNGROUNDED")}
    receipt = {
        "attestation": "styxx OATH v0 corpus attestation (every finding doc under oath)",
        "date": "2026-06-10",
        "verifier": "styxx.certify",
        "verifier_sha256": hashlib.sha256((REPO / "styxx" / "certify.py").read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "n_docs": len(docs),
        "n_certified": len(certified), "n_oath_held": len(held),
        "n_oath_failed": len(failed), "n_unbound": len(unbound), "n_binding_error": len(broken),
        "claim_totals": totals,
        "oath_failed_docs": failed,
        "unbound_docs": [e["doc"] for e in unbound],
        "binding_errors": broken,
        "entries": entries,
    }
    out = HERE / "oath_corpus_attestation.json"
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"\ncertified={len(certified)} (held={len(held)} failed={len(failed)}) "
          f"unbound={len(unbound)} binding-error={len(broken)}  "
          f"claims: V={totals['VERIFIED']} A={totals['ABSTAIN']} U={totals['UNGROUNDED']}  "
          f"elapsed={time.time()-t0:.1f}s\n-> {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
