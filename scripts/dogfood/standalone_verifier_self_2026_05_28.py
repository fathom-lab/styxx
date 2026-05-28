"""Live dogfood: verify a REAL styxx attestation with the standalone, zero-styxx
verifier, and cross-check it byte-for-byte against the library (prereg'd).

Builds a real (task -> self-report) chain over styxx's own HEAD with embedded
vitals, writes it, then:

  * loads scripts/styxx_verify_standalone.py as a stdlib-only module        [K3]
  * recomputes every per-attestation + chain-link digest and matches the
    library byte-for-byte                                                   [K1]
  * standalone verify_chain reports OK, anchored to the library head        [K1]
  * a flipped digest -> standalone reports FAIL                             [K2]
  * a re-sealed chain passes structure but is caught with the anchor        [P4]

Reports whichever way it lands. Frozen receipt next to this script.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from styxx.attestation import attest_chain

REPO = Path(__file__).resolve().parents[2]
STANDALONE = REPO / "scripts" / "styxx_verify_standalone.py"
RECEIPT = Path(__file__).resolve().parent / "standalone_verifier_self_2026_05_28.json"

PROMPT = "Report on the attestation work you shipped this session."
REPORT = (
    "The version is 7.7.12. I shipped a standalone, stdlib-only verifier that "
    "imports nothing from styxx and re-derives the content address from a "
    "published spec, so a styxx attestation can be verified without trusting "
    "styxx. It checks structure only; semantic claims and vitals are reported "
    "NOT CHECKED, and a re-sealed chain is caught only with an external anchor."
)


def load_standalone():
    spec = importlib.util.spec_from_file_location("sv", STANDALONE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    sv = load_standalone()

    # K3: the standalone verifier imports nothing from styxx.
    tree = ast.parse(STANDALONE.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    no_styxx = "styxx" not in imported and imported <= {"argparse", "hashlib", "json", "sys"}

    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD^{commit}"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    chain = attest_chain([(REPORT, head, PROMPT)], REPO, id_prefix="SV", vitals=True).artifact

    # K1: byte-for-byte digest agreement over the chain.
    prev = sv.CHAIN_GENESIS
    per_link = []
    k1_ok = True
    for link in chain["links"]:
        att = link["attestation"]
        att_d = sv.compute_digest(att)
        att_match = att_d == link["attestation_digest"]
        link_match = link["chain_digest"] == sv.chain_digest(prev, att_d)
        k1_ok = k1_ok and att_match and link_match and link["prev_chain_digest"] == prev
        per_link.append({
            "att_digest_agrees": att_match,
            "chain_digest_agrees": link_match,
        })
        prev = link["chain_digest"]
    head_match = chain["head_chain_digest"] == prev
    k1_ok = k1_ok and head_match

    out = []
    standalone_chain_ok = sv.verify_chain(chain, out, expected_head=chain["head_chain_digest"])

    # K2: flip an embedded digest -> standalone catches it.
    import copy
    tampered = copy.deepcopy(chain)
    tampered["links"][0]["attestation"]["digest"]["value"] = "0" * 64
    out2 = []
    k2_caught = sv.verify_chain(tampered, out2) is False

    # P4: re-seal the whole chain after editing substrate -> structure passes,
    # anchor catches it.
    resealed = copy.deepcopy(chain)
    good_head = resealed["head_chain_digest"]
    lk = resealed["links"][0]
    lk["attestation"]["report"] = "The version is 9.9.9.\n"
    ad = sv.compute_digest(lk["attestation"])
    lk["attestation"]["digest"]["value"] = ad
    lk["attestation_digest"] = ad
    cd = sv.chain_digest(sv.CHAIN_GENESIS, ad)
    lk["chain_digest"] = cd
    resealed["head_chain_digest"] = cd
    out3 = []
    reseal_passes_structure = sv.verify_chain(resealed, out3) is True
    out4 = []
    reseal_caught_by_anchor = sv.verify_chain(resealed, out4, expected_head=good_head) is False

    receipt = {
        "dogfood": "standalone (zero-styxx) verifier on a real styxx chain",
        "generated_at": chain["generated_at"],
        "head_commit": head,
        "standalone_path": str(STANDALONE.relative_to(REPO)),
        "K3_standalone_imports": sorted(imported),
        "K3_no_styxx_import": no_styxx,
        "K1_per_link_digest_agreement": per_link,
        "K1_head_agrees": head_match,
        "K1_all_digests_agree": k1_ok,
        "K1_standalone_chain_ok_anchored": standalone_chain_ok,
        "K2_flipped_digest_caught": k2_caught,
        "P4_reseal_passes_structure": reseal_passes_structure,
        "P4_reseal_caught_by_anchor": reseal_caught_by_anchor,
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))

    all_ok = (
        no_styxx and k1_ok and standalone_chain_ok and k2_caught
        and reseal_passes_structure and reseal_caught_by_anchor
    )
    print("\nRESULT:", "SURVIVED" if all_ok else "FALSIFIED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
