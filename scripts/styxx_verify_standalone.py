#!/usr/bin/env python3
"""Standalone, trust-minimized verifier for styxx attestations.

THIS FILE IMPORTS NOTHING FROM styxx. Standard library only (hashlib, json,
sys, argparse). It is the executable form of the styxx content-addressing
spec (docs/attestation-content-address.md): an independent reimplementation
that lets you verify the STRUCTURAL integrity of a styxx attestation or chain
without trusting — or even installing — styxx.

What it verifies (structure only):
  * per-attestation digest = sha256(canonical_payload)
        canonical_payload = json.dumps(core, sort_keys=True,
                                        separators=(",",":"), ensure_ascii=False)
        core = artifact minus the "generated_at" and "digest" keys
  * chain link digest = sha256(f"{prev}|{att_digest}")
        genesis prev = "styxx-attestation-chain-v1"
        head = last link's chain_digest
  * optional: --expected-head anchors the chain so a re-sealed tamper is caught

What it does NOT verify (honest scope — needs styxx + the repo):
  * claim verdicts (need git + the pinned repo tree)
  * cognometric vitals scores (need styxx's scoring instruments)
These are reported as "NOT CHECKED", never asserted true.

Portability boundary (honest): the canonical payload uses Python's json float
repr. Byte-identical agreement holds for an independent PYTHON reimplementation.
A cross-language (e.g. browser/JS) verifier needs a canonical-number scheme
(JCS / RFC 8785); that is documented future work, not implemented here.

Exit code 0 = structural integrity OK, 1 = FAIL, 2 = usage/parse error.
"""

import argparse
import hashlib
import json
import sys

CHAIN_GENESIS = "styxx-attestation-chain-v1"


def canonical_payload(artifact):
    core = {k: v for k, v in artifact.items() if k not in ("generated_at", "digest")}
    return json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_digest(artifact):
    return hashlib.sha256(canonical_payload(artifact).encode("utf-8")).hexdigest()


def chain_digest(prev, att_digest):
    return hashlib.sha256(f"{prev}|{att_digest}".encode("utf-8")).hexdigest()


def _semantic_notice(artifact, out):
    if isinstance(artifact.get("claims"), list) and artifact["claims"]:
        out.append("  semantic (claim verdicts): NOT CHECKED — needs styxx + repo")
    if artifact.get("vitals") is not None:
        out.append("  semantic (vitals scores):  NOT CHECKED — needs styxx instruments")


def verify_attestation(artifact, out):
    """Verify a single attestation's content address. Returns True/False."""
    recorded = (artifact.get("digest") or {}).get("value")
    recomputed = compute_digest(artifact)
    ok = recorded is not None and recorded == recomputed
    out.append(f"  attestation digest: {'OK' if ok else 'FAIL'}")
    if not ok:
        out.append(f"    recorded:   {recorded}")
        out.append(f"    recomputed: {recomputed}")
    _semantic_notice(artifact, out)
    return ok


def verify_chain(artifact, out, expected_head=None):
    """Verify a chain artifact's Merkle linkage + per-link digests."""
    links = artifact.get("links") or []
    ok = True
    prev = CHAIN_GENESIS
    for i, link in enumerate(links):
        att = link.get("attestation", {})
        att_recorded = (att.get("digest") or {}).get("value")
        att_recomputed = compute_digest(att)
        att_ok = att_recorded is not None and att_recorded == att_recomputed
        link_recorded_att = link.get("attestation_digest")
        att_field_ok = link_recorded_att == att_recomputed
        expect_link = chain_digest(prev, att_recomputed)
        link_ok = link.get("chain_digest") == expect_link
        prev_ok = link.get("prev_chain_digest") == prev
        good = att_ok and att_field_ok and link_ok and prev_ok
        ok = ok and good
        out.append(f"  link[{i}]: {'OK' if good else 'FAIL'}")
        if not good:
            if not att_ok:
                out.append("    attestation digest mismatch")
            if not att_field_ok:
                out.append("    attestation_digest field mismatch")
            if not prev_ok:
                out.append("    prev_chain_digest mismatch (reordered/tampered)")
            if not link_ok:
                out.append("    chain_digest mismatch (broken link)")
        _semantic_notice(att, out)
        prev = link.get("chain_digest")

    head = artifact.get("head_chain_digest")
    head_ok = head == (prev if links else CHAIN_GENESIS)
    out.append(f"  head_chain_digest: {'OK' if head_ok else 'FAIL'}")
    ok = ok and head_ok

    if expected_head is not None:
        anchor_ok = head == expected_head
        out.append(
            f"  expected-head anchor: {'OK' if anchor_ok else 'FAIL'} "
            f"(re-seal {'cannot hide from anchor' if anchor_ok else 'or tamper detected'})"
        )
        ok = ok and anchor_ok
    else:
        out.append("  expected-head anchor: NOT PROVIDED — a re-sealed chain "
                    "would pass structure; anchor with --expected-head to catch it")
    return ok


def main(argv=None):
    p = argparse.ArgumentParser(description="Standalone styxx attestation verifier (no styxx import).")
    p.add_argument("path", help="path to an attestation or chain JSON artifact")
    p.add_argument("--expected-head", default=None,
                   help="externally-anchored chain head; catches a re-sealed chain")
    args = p.parse_args(argv)

    try:
        with open(args.path, encoding="utf-8") as fh:
            artifact = json.load(fh)
    except (OSError, ValueError) as e:
        print(f"error: cannot read/parse {args.path}: {e}", file=sys.stderr)
        return 2

    out = []
    is_chain = "links" in artifact and "head_chain_digest" in artifact
    if is_chain:
        out.append(f"styxx attestation CHAIN — {len(artifact.get('links') or [])} link(s)")
        ok = verify_chain(artifact, out, expected_head=args.expected_head)
    else:
        out.append("styxx ATTESTATION")
        ok = verify_attestation(artifact, out)

    print("\n".join(out))
    print(f"\nstructural integrity: {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
