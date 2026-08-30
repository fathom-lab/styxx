"""The external half of the unobligated-oath census: how do foreign certificates compose?

`RESULT_unobligated_oath_2026_08_28.md` measured our own corpus: 0.5811 of verifications
volunteered. `RESULT_obligation_predicts_claimhood_2026_08_30.md` measured what that costs abroad
on the panel sample: volunteered oaths collapse to 0.3654 claim-share. What nobody has computed is
the corpus-wide composition of the external certificates themselves — how much of what the
verifier swears to on other people's documents is volunteered.

Method: rebuild each CERTIFIED external repository's document and receipts from the frozen fetch
cache (every blob hash-verified against the committed manifest; a missing or corrupt blob reports
the repository unresolved rather than substituting a live fetch), re-certify at the pinned
annotated verifier, and fold the epistemics_summary blocks.

  python papers/closed-model-frontier/oath_external_epistemics_census.py
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx.certify import certify_doc                                   # noqa: E402

OUT = HERE / "oath_external_epistemics_census.json"
CACHE = Path(os.environ.get("OATH_EXT_CACHE",
                            Path(os.environ.get("TEMP", "/tmp")) / "oath_ext_corpus_cache"))


def rebuild_and_certify(rec: dict):
    stage = Path(tempfile.mkdtemp())
    doc_path, rpaths = None, []
    for f in rec["files"]:
        key = hashlib.sha256(f"{rec['repo']}@{rec['sha']}/{f['path']}".encode()).hexdigest()[:32]
        blob = CACHE / key
        if not blob.exists():
            return None, "cache_missing"
        raw = blob.read_bytes()
        if hashlib.sha256(raw).hexdigest() != f["sha256"]:
            return None, "cache_corrupt"
        if f["role"] == "document":
            doc_path = stage / "DOC.md"
            doc_path.write_bytes(raw)
        else:
            p = stage / f"r{len(rpaths)}_{Path(f['path']).name}"
            p.write_bytes(raw)
            rpaths.append(p)
    if not doc_path or not rpaths:
        return None, "incomplete"
    try:
        return certify_doc(doc_path, rpaths), None
    except Exception as e:
        return None, f"certify_error:{type(e).__name__}"


def main() -> int:
    manifest = json.loads((HERE / "oath_external_corpus.json").read_text(encoding="utf-8"))
    vm_total = collections.Counter()
    branches = collections.Counter()
    sources = collections.Counter()
    derived = collections.Counter()
    unresolved = collections.Counter()
    per_repo = []
    done = 0

    for rec in manifest["per_repo"]:
        if rec.get("status") != "CERTIFIED":
            continue
        cert, err = rebuild_and_certify(rec)
        if cert is None:
            unresolved[err] += 1
            continue
        done += 1
        s = cert["epistemics_summary"]
        for k, v in s["by_branch"].items():
            branches[k] += v
        for k, v in s["obligation_sources"].items():
            sources[k] += v
        for k, v in s["verified"]["value_match"].items():
            vm_total[k] += v
        for k, v in s["verified"]["derived"].items():
            derived[k] += v
        tot = s["verified"]["total"]
        unob = (s["verified"]["value_match"]["unobligated_integer_filter_ran"]
                + s["verified"]["value_match"]["unobligated_integer_filter_na"]
                + s["verified"]["derived"]["unobligated"])
        if tot:
            per_repo.append({"repo": rec["repo"], "verified": tot, "unobligated": unob,
                             "share": round(unob / tot, 4)})

    per_repo.sort(key=lambda r: (-r["share"], -r["verified"]))
    ver_total = sum(vm_total.values()) + sum(derived.values())
    unob_total = (vm_total["unobligated_integer_filter_ran"]
                  + vm_total["unobligated_integer_filter_na"] + derived["unobligated"])
    weakest = vm_total["unobligated_integer_filter_na"]

    internal = json.loads((HERE / "oath_unobligated_oath_census.json").read_text(encoding="utf-8"))
    payload = {
        "census": "external certificates' epistemic composition, folded from epistemics_summary",
        "repos_certified_in_manifest": sum(1 for r in manifest["per_repo"]
                                           if r.get("status") == "CERTIFIED"),
        "repos_recertified": done,
        "unresolved": dict(unresolved),
        "by_branch": dict(branches.most_common()),
        "obligation_sources": dict(sources.most_common()),
        "verified_value_match": dict(vm_total),
        "verified_derived": dict(derived),
        "headline": {
            "verified_total": ver_total,
            "unobligated_oaths": unob_total,
            "unobligated_oath_rate_external": round(unob_total / ver_total, 4)
            if ver_total else None,
            "weakest_attestations": weakest,
            "weakest_share": round(weakest / ver_total, 4) if ver_total else None,
            "internal_comparison": {
                "unobligated_oath_rate_internal":
                    internal["headline"]["unobligated_oath_rate"],
                "weakest_share_internal": internal["headline"]["weakest_share_of_verified"],
            },
        },
        "repos_most_exposed": per_repo[:10],
        "companion_quality_number": (
            "The join RESULT measured volunteered external oaths at 0.3654 claim-share on the "
            "panel sample. This census gives the corpus-wide DENOMINATOR those volunteers sit "
            "in; multiplying the two is an estimate, not a measurement, and is left to the "
            "reader with that label."),
    }
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    h = payload["headline"]
    print(f"recertified {done} external repos   unresolved {dict(unresolved) or 0}")
    print(f"verified split: {dict(vm_total)}  derived: {dict(derived)}")
    print(f"\nEXTERNAL UNOBLIGATED OATH RATE : {h['unobligated_oath_rate_external']}  "
          f"({h['unobligated_oaths']} of {h['verified_total']})")
    print(f"internal, for comparison       : {h['internal_comparison']['unobligated_oath_rate_internal']}")
    print(f"weakest attestations external  : {h['weakest_share']}  (internal "
          f"{h['internal_comparison']['weakest_share_internal']})")
    print(f"-> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
