"""Live cognometric-attestation dogfood on styxx's OWN self-report (prereg'd).

Attests a real (task -> self-report) pair with embedded, re-derivable vitals,
chained over styxx's history, then exercises the kill-gate live:

  * verify re-derives the scores from the recorded (prompt, response)   [P2]
  * K1: rebuild is deterministic (same head + same scores)              [K1]
  * K2 (decisive): flip an embedded score AND re-seal the digest ->
        digest_ok True but vitals_ok False (recomputation catches it)   [K2]
  * P3: the artifact carries a machine-readable register-not-honesty
        boundary + reference-less-deception caveat                       [P3]
  * chain vitals reproduce per link                                      [P4]

Writes a frozen receipt next to this script. Reports whichever way it lands.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

from styxx.attestation import (
    _compute_digest,
    attest_chain,
    verify_attestation,
    verify_chain,
)

REPO = Path(__file__).resolve().parents[2]
RECEIPT = Path(__file__).resolve().parent / "cognometric_attestation_self_2026_05_28.json"

# A real (task, self-report) pair from this session's work. The report is the
# agent's own words; the vitals score its REGISTER, not its truth.
PROMPT = "Report on the attestation work you shipped this session."
REPORT = (
    "The version is 7.7.11. I shipped commit-pinned attestation and "
    "tamper-evident chains, each pre-registered with a kill-gate before the "
    "code was written, and I report the result whichever way it lands. The "
    "vitals measure register, not whether this report is true."
)


def main() -> int:
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD^{commit}"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    chain = attest_chain([(REPORT, head, PROMPT)], REPO, id_prefix="COGN", vitals=True)
    link = chain.artifact["links"][0]["attestation"]
    vitals = link["vitals"]

    res = verify_chain(chain, REPO)
    att_res = verify_attestation(link, REPO)

    # K1 determinism
    chain2 = attest_chain([(REPORT, head, PROMPT)], REPO, id_prefix="COGN", vitals=True)
    determinism_ok = (
        chain.head == chain2.head
        and vitals["scores"] == chain2.artifact["links"][0]["attestation"]["vitals"]["scores"]
    )

    # K2 decisive: flip a score AND re-seal the digest
    tampered = copy.deepcopy(link)
    axis = next(iter(tampered["vitals"]["scores"]))
    orig = tampered["vitals"]["scores"][axis]
    tampered["vitals"]["scores"][axis] = 0.0 if orig > 0.5 else 1.0
    tampered["digest"]["value"] = _compute_digest(tampered)  # re-seal
    tamper_res = verify_attestation(tampered, REPO)
    k2_caught = tamper_res.digest_ok and not tamper_res.vitals_ok and not tamper_res.ok

    receipt = {
        "dogfood": "cognometric attestation on styxx's own self-report",
        "generated_at": chain.artifact["generated_at"],
        "head_commit": head,
        "prompt": PROMPT,
        "response": REPORT,
        "vitals_scores": vitals["scores"],
        "vitals_tier": vitals["tier"],
        "vitals_measures": vitals["measures"],
        "deception_caveat": vitals["caveats"].get("deception"),
        "axes_undefined_without_prompt": vitals["axes_undefined_without_prompt"],
        "verify_attestation_ok": att_res.ok,
        "verify_attestation_vitals_ok": att_res.vitals_ok,
        "verify_chain_ok": res.ok,
        "K1_determinism_ok": determinism_ok,
        "K2_resealed_score_tamper": {
            "axis_flipped": axis,
            "digest_fooled": tamper_res.digest_ok,
            "caught_by_recompute": not tamper_res.vitals_ok,
            "overall_ok": tamper_res.ok,
            "mismatches": tamper_res.vitals_mismatches,
        },
        "K2_caught": k2_caught,
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))

    all_ok = att_res.ok and res.ok and determinism_ok and k2_caught
    print("\nRESULT:", "SURVIVED" if all_ok else "FALSIFIED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
