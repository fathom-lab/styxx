"""Live attestation-chain dogfood over styxx's OWN history (P4, prereg'd).

Builds a 2-link chain:
  link 0: "The version is 7.7.10." pinned to the v7.7.10 commit  -> PASS as-of-then
  link 1: "The version is 7.7.11." pinned to HEAD                 -> PASS as-of-now

Then exercises the kill-gate live:
  * end-to-end verify (per-link reproduction + intact Merkle links)  [K3, P3]
  * determinism: rebuild yields the same head_chain_digest           [K1, P1]
  * naive reorder is caught at the first divergent link              [K2, P2]
  * the SAME claim, pinned to the OTHER commit, flips verdict        [P4 / commit-pin]

Writes a frozen receipt next to this script. Reports whichever way it lands.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from styxx.attestation import attest_chain, verify_chain

REPO = Path(__file__).resolve().parents[2]
RECEIPT = Path(__file__).resolve().parent / "chain_self_history_2026_05_28.json"


def _rev(ref: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", f"{ref}^{{commit}}"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def main() -> int:
    v7710 = _rev("v7.7.10")
    head = _rev("HEAD")
    print(f"v7.7.10 -> {v7710}")
    print(f"HEAD     -> {head}")

    items = [
        ("The version is 7.7.10.", v7710),
        ("The version is 7.7.11.", head),
    ]

    chain = attest_chain(items, REPO, id_prefix="SELF")
    res = verify_chain(chain, REPO)

    # per-link verdicts (PASS/FAIL as-of each pinned commit)
    link_verdicts = []
    for link in chain.artifact["links"]:
        att = link["attestation"]
        claims = att["claims"]
        verdict = "PASS" if all(c["verdict"] == "PASS" for c in claims) else "FAIL"
        link_verdicts.append({
            "seq": link["seq"],
            "claim": claims[0]["text"] if claims else "(none)",
            "pinned_commit": att["substrate"]["commit"],
            "verdict": verdict,
        })

    # K1 determinism: rebuild, compare head
    chain2 = attest_chain(items, REPO, id_prefix="SELF")
    determinism_ok = chain.head == chain2.head

    # K2 naive reorder: swap links without re-sealing, expect BROKEN
    import copy
    tampered = copy.deepcopy(chain.artifact)
    tampered["links"][0], tampered["links"][1] = (
        tampered["links"][1], tampered["links"][0],
    )
    from styxx.attestation import AttestationChain
    res_tampered = verify_chain(AttestationChain(tampered), REPO)
    reorder_caught = not res_tampered.ok

    # commit-pin flip: claim "7.7.10" against HEAD must FAIL; against v7.7.10 PASS
    flip_chain = attest_chain(
        [("The version is 7.7.10.", head)], REPO, id_prefix="FLIP",
    )
    flip_claims = flip_chain.artifact["links"][0]["attestation"]["claims"]
    flip_verdict = "PASS" if all(c["verdict"] == "PASS" for c in flip_claims) else "FAIL"
    flip_ok = flip_verdict == "FAIL"  # 7.7.10 claim should NOT hold at HEAD (7.7.11)

    receipt = {
        "dogfood": "attestation-chain over styxx own history (P4)",
        "generated_at": chain.artifact["generated_at"],
        "v7.7.10_commit": v7710,
        "head_commit": head,
        "head_chain_digest": chain.head,
        "n_links": chain.artifact["n_links"],
        "link_verdicts": link_verdicts,
        "verify_ok": res.ok,
        "verify_links_ok": res.links_ok,
        "verify_head_ok": res.head_ok,
        "K1_determinism_ok": determinism_ok,
        "K2_naive_reorder_caught": reorder_caught,
        "K2_reorder_broken_at": res_tampered.broken_at,
        "commit_pin_flip_ok": flip_ok,
        "commit_pin_flip_verdict_at_HEAD": flip_verdict,
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(receipt, indent=2))
    all_ok = (
        res.ok and determinism_ok and reorder_caught and flip_ok
        and link_verdicts[0]["verdict"] == "PASS"
        and link_verdicts[1]["verdict"] == "PASS"
    )
    print("\nP4 RESULT:", "SURVIVED" if all_ok else "FALSIFIED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
