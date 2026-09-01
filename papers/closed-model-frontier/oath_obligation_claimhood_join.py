"""Does obligation predict claimhood? The join of two measurements already paid for.

`RESULT_unobligated_oath_2026_08_28.md` counted the split: 0.5811 of this corpus's verifications
were volunteered, not obligated. What it could not say is whether that matters for QUALITY — a
volunteered oath may be exactly as likely to sit on a real claim as a required one.

The 2026-08-27 blind panels already adjudicated claimhood for 225 VERIFIED tokens (150 internal,
75 external), three seats each, majority verdicts, decoy-blinded. The epistemics annotation now
makes each of those tokens taggable as obligated or volunteered. This joins the two — no new
adjudication, no new judgement, a pure recombination of committed evidence.

The question, exactly: **among panel-adjudicated VERIFIED tokens, is the claim-share of
volunteered oaths lower than that of obligated oaths?**

* If yes — the unobligated 58% is a quality problem, and the certificate's verified count is
  diluted by oaths that are disproportionately about non-claims.
* If no — obligation adds nothing to verification quality, and the predicate's only measured
  effect is choosing what to ACCUSE.

Either answer is a finding. Both arms are reported with their n; no significance is asserted at
these sizes.

Method per token: re-certify the document live at the pinned verifier, locate the panel token by
(line, token) in the ledger, read its epistemics. Internal documents resolve from papers/;
external documents resolve from the frozen fetch cache at the pinned sha, and any token whose
document cannot be re-certified is reported as unresolved rather than silently dropped.

  python papers/closed-model-frontier/oath_obligation_claimhood_join.py
"""
from __future__ import annotations

import collections
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from styxx.certify import certify_doc                                   # noqa: E402
from styxx.corpus_audit import _resolve_receipts                        # noqa: E402

OUT = HERE / "oath_obligation_claimhood_join.json"
CACHE = Path(os.environ.get("OATH_EXT_CACHE",
                            Path(os.environ.get("TEMP", "/tmp")) / "oath_ext_corpus_cache"))


def internal_tokens():
    """(doc_path, receipts, line, token, panel_verdict) for the internal VERIFIED arm."""
    res = json.loads((HERE / "oath_internal_result.json").read_text(encoding="utf-8"))
    for r in res["per_arm_detail"]["VERIFIED"]:
        # r["repo"] is the document filename; find it under papers/ and its cited receipts via
        # its committed certificate, same as the internal ledger builder did.
        yield r["repo"], r["line"], r["token"], r["verdict"]


def external_tokens():
    res = json.loads((HERE / "oath_adjudication_result.json").read_text(encoding="utf-8"))
    for r in res["per_arm_detail"]["VERIFIED"]:
        yield r["repo"], r["line"], r["token"], r["verdict"]


def certify_internal(doc_name: str):
    hits = [p for p in (ROOT / "papers").rglob(doc_name) if ".claude" not in p.parts]
    if len(hits) != 1:
        return None
    doc = hits[0]
    cert_path = doc.parent / (doc.stem + ".certificate.json")
    if not cert_path.exists():
        return None
    stored = json.loads(cert_path.read_text(encoding="utf-8"))
    paths, _m, _d = _resolve_receipts(cert_path, stored, ROOT / "papers")
    if not paths:
        return None
    return certify_doc(doc, paths)


_EXT_MANIFEST = None


def certify_external(repo: str):
    """Rebuild the external document + receipts from the frozen fetch cache, then certify."""
    global _EXT_MANIFEST
    if _EXT_MANIFEST is None:
        _EXT_MANIFEST = json.loads(
            (HERE / "oath_external_corpus.json").read_text(encoding="utf-8"))
    rec = next((r for r in _EXT_MANIFEST["per_repo"] if r["repo"] == repo), None)
    if not rec or rec.get("status") != "CERTIFIED":
        return None
    import hashlib
    stage = Path(tempfile.mkdtemp())
    doc_path = None
    rpaths = []
    for f in rec["files"]:
        key = hashlib.sha256(f"{repo}@{rec['sha']}/{f['path']}".encode()).hexdigest()[:32]
        blob = CACHE / key
        if not blob.exists():
            return None                      # cache evaporated; reported as unresolved
        raw = blob.read_bytes()
        if hashlib.sha256(raw).hexdigest() != f["sha256"]:
            return None                      # cache corrupt; refuse rather than trust
        if f["role"] == "document":
            doc_path = stage / "DOC.md"
            doc_path.write_bytes(raw)
        else:
            p = stage / f"r{len(rpaths)}_{Path(f['path']).name}"
            p.write_bytes(raw)
            rpaths.append(p)
    if not doc_path or not rpaths:
        return None
    return certify_doc(doc_path, rpaths)


def find_entry(cert, line, token):
    for e in cert["ledger"]:
        if e["line"] == line and e["token"] == token:
            return e
    return None


def main() -> int:
    arms = {}
    for arm, source, certifier in (("internal", internal_tokens, certify_internal),
                                   ("external", external_tokens, certify_external)):
        cells = collections.Counter()
        unresolved = 0
        cert_cache = {}
        detail = []
        for repo, line, token, verdict in source():
            if repo not in cert_cache:
                try:
                    cert_cache[repo] = certifier(repo)
                except Exception:
                    cert_cache[repo] = None
            cert = cert_cache[repo]
            e = find_entry(cert, line, token) if cert else None
            if e is None or e["status"] != "VERIFIED":
                unresolved += 1
                continue
            ob = "obligated" if e["epistemics"]["obligated"] else "volunteered"
            cells[(ob, verdict)] += 1
            detail.append({"repo": repo, "line": line, "token": token,
                           "obligation": ob, "panel": verdict})
        def share(ob):
            c = cells[(ob, "CLAIM")]
            n = c + cells[(ob, "NOT_A_CLAIM")]
            return {"n": n, "claims": c, "claim_share": round(c / n, 4) if n else None}
        arms[arm] = {"obligated": share("obligated"), "volunteered": share("volunteered"),
                     "unresolved": unresolved, "cells": {f"{k[0]}|{k[1]}": v
                                                         for k, v in cells.items()},
                     "detail": detail}
        print(f"[{arm}] obligated {arms[arm]['obligated']}   "
              f"volunteered {arms[arm]['volunteered']}   unresolved {unresolved}")

    payload = {
        "join": "panel-adjudicated claimhood x epistemics obligation, per VERIFIED token",
        "question": ("among VERIFIED tokens the blind panels adjudicated, is the claim-share of "
                     "volunteered oaths lower than that of obligated oaths?"),
        "no_new_judgement": ("panel verdicts are the committed 2026-08-27 majority verdicts; "
                             "obligation tags come from live re-certification at the pinned "
                             "verifier. Nothing here was re-adjudicated."),
        "arms": {k: {kk: vv for kk, vv in v.items() if kk != "detail"} for k, v in arms.items()},
        "per_token_detail": {k: v["detail"] for k, v in arms.items()},
        "caveats": (
            "Panel = three LLM seats of one family; unanimity ~0.96-0.98 is a correlated-error "
            "ceiling. Sample sizes are small and no interval or significance is asserted. "
            "External resolution depends on the frozen fetch cache; unresolved counts are "
            "reported, not dropped. The internal sample was drawn uniformly from VERIFIED tokens, "
            "so its obligated/volunteered mix estimates the corpus mix independently of the "
            "census."),
    }
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"-> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
