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
  * transparency-log proofs (RFC 6962, string-tagged — docs §7): an inclusion
        proof (kind="inclusion") or a consistency proof (kind="consistency");
        --witnessed-root anchors a consistency proof so a rewritten/suppressed
        past entry is caught

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


# --- transparency log (RFC 6962, string-tagged) — see docs §7 --------------
TLOG_LEAF_TAG = "styxx-tlog-leaf:"
TLOG_NODE_TAG = "styxx-tlog-node:"


def leaf_hash(entry):
    return hashlib.sha256((TLOG_LEAF_TAG + entry).encode("utf-8")).hexdigest()


def node_hash(left, right):
    return hashlib.sha256((TLOG_NODE_TAG + left + ":" + right).encode("utf-8")).hexdigest()


def merkle_tree_hash(leaves):
    n = len(leaves)
    if n == 0:
        return hashlib.sha256(b"").hexdigest()
    if n == 1:
        return leaves[0]
    k = 1
    while (k << 1) < n:
        k <<= 1
    return node_hash(merkle_tree_hash(leaves[:k]), merkle_tree_hash(leaves[k:]))


def _decomp_incl(index, size):
    inner = (index ^ (size - 1)).bit_length()
    border = bin(index >> inner).count("1")
    return inner, border


def _chain_inner(seed, proof, index):
    for i, h in enumerate(proof):
        seed = node_hash(seed, h) if (index >> i) & 1 == 0 else node_hash(h, seed)
    return seed


def _chain_inner_right(seed, proof, index):
    for i, h in enumerate(proof):
        if (index >> i) & 1 == 1:
            seed = node_hash(h, seed)
    return seed


def _chain_border_right(seed, proof):
    for h in proof:
        seed = node_hash(h, seed)
    return seed


def verify_inclusion(proof, out, root=None):
    index, size = proof.get("leaf_index"), proof.get("tree_size")
    leaf, path = proof.get("leaf_hash"), proof.get("audit_path") or []
    expected = root if root is not None else proof.get("root")
    if expected is None or not (isinstance(index, int) and 0 <= index < size):
        out.append("  inclusion: FAIL (malformed proof)")
        return False
    inner, border = _decomp_incl(index, size)
    if len(path) != inner + border:
        out.append("  inclusion: FAIL (wrong audit-path length)")
        return False
    res = _chain_border_right(_chain_inner(leaf, path[:inner], index), path[inner:])
    ok = res == expected
    out.append(f"  leaf {index} of {size}: inclusion {'OK' if ok else 'FAIL'}")
    return ok


def verify_consistency(proof, out, first_root=None):
    size1, size2 = proof.get("first_size"), proof.get("second_size")
    path = list(proof.get("proof") or [])
    r1 = first_root if first_root is not None else proof.get("first_root")
    r2 = proof.get("second_root")
    if r1 is None or r2 is None or size1 > size2:
        out.append("  consistency: FAIL (malformed proof)")
        return False
    if size1 == size2:
        ok = r1 == r2 and not path
    elif size1 == 0:
        ok = not path
    else:
        inner, border = _decomp_incl(size1 - 1, size2)
        shift = (size1 & -size1).bit_length() - 1
        inner -= shift
        if size1 == (1 << shift):
            seed, start = r1, 0
        elif path:
            seed, start = path[0], 1
        else:
            out.append("  consistency: FAIL (empty proof)")
            return False
        if len(path) != start + inner + border:
            out.append("  consistency: FAIL (wrong proof length)")
            return False
        path = path[start:]
        mask = (size1 - 1) >> shift
        h1 = _chain_border_right(_chain_inner_right(seed, path[:inner], mask), path[inner:])
        h2 = _chain_border_right(_chain_inner(seed, path[:inner], mask), path[inner:])
        ok = h1 == r1 and h2 == r2
    out.append(f"  size {size1} -> {size2}: consistency {'OK' if ok else 'FAIL'}"
               + ("" if ok else " — a past entry was edited/deleted/reordered"))
    if first_root is None:
        out.append("  witnessed first-root: NOT PROVIDED — pass --witnessed-root to detect a rewrite")
    return ok


def main(argv=None):
    p = argparse.ArgumentParser(description="Standalone styxx attestation verifier (no styxx import).")
    p.add_argument("path", help="path to an attestation, chain, or transparency-log proof JSON")
    p.add_argument("--expected-head", default=None,
                   help="externally-anchored chain head; catches a re-sealed chain")
    p.add_argument("--witnessed-root", default=None,
                   help="witnessed earlier root for a consistency proof; catches a rewritten history")
    args = p.parse_args(argv)

    try:
        with open(args.path, encoding="utf-8") as fh:
            artifact = json.load(fh)
    except (OSError, ValueError) as e:
        print(f"error: cannot read/parse {args.path}: {e}", file=sys.stderr)
        return 2

    out = []
    kind = artifact.get("kind")
    if kind == "inclusion":
        out.append("styxx transparency-log INCLUSION proof")
        ok = verify_inclusion(artifact, out)
    elif kind == "consistency":
        out.append("styxx transparency-log CONSISTENCY proof (append-only check)")
        ok = verify_consistency(artifact, out, first_root=args.witnessed_root)
    elif "links" in artifact and "head_chain_digest" in artifact:
        out.append(f"styxx attestation CHAIN — {len(artifact.get('links') or [])} link(s)")
        ok = verify_chain(artifact, out, expected_head=args.expected_head)
    else:
        out.append("styxx ATTESTATION")
        ok = verify_attestation(artifact, out)

    print("\n".join(out))
    label = "proof" if kind in ("inclusion", "consistency") else "structural integrity"
    print(f"\n{label}: {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
