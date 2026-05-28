"""Live dogfood: disclose ONE attested fact, keep the rest private (prereg'd).

Composes the whole arc on styxx's own HEAD:

  * attest(redactable=True) over a real self-report                         [build]
  * disclose ONLY the sycophancy score; prove it is bound to the public
    digest.redactable.root, and confirm the full response text does NOT
    appear in the disclosure                                                [K1, K2]
  * the salt is load-bearing: an UNSALTED small-domain leaf is brute-forced,
    the SALTED one is not recoverable without the salt                      [K2 decisive]
  * adding digest.redactable leaves digest.value and digest.portable
    byte-identical                                                          [K3 additive]
  * Node (web/styxx_verify.js, zero styxx, zero deps) verifies the disclosure
    and rejects a tampered one                                              [K3 cross-lang]
  * the redactable root is placed as a transparency-log leaf, and a
    consistency proof binds the confidential receipt to the append-only,
    no-silent-suppression history                                          [P4 compose]

HONEST SCOPE: selective DISCLOSURE, not zero-knowledge. No predicate/range over a
hidden value; a disclosed value is the COMMITTED value (commit-time/re-seal
boundary, caught only via the transparency log + an external witness). Leaks the
field count + disclosed pointers/values; hides everything undisclosed.

Requires `node`. Reports whichever way it lands. Frozen receipt beside this file.
"""

from __future__ import annotations

import copy
import json
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

from styxx.attestation import _compute_digest, _compute_portable_digest, attest
from styxx.redact import _leaf_hash, verify_disclosure
from styxx.transparency import TransparencyLog, verify_consistency

REPO = Path(__file__).resolve().parents[2]
JS = REPO / "web" / "styxx_verify.js"
RECEIPT = Path(__file__).resolve().parent / "redactable_attestation_self_2026_05_28.json"
NODE = shutil.which("node")

PROMPT = "Report on one disciplined step you shipped this session, candidly."
REPORT = (
    "The version is 7.7.12. I shipped a redactable attestation: a salted Merkle "
    "commitment over the attested fields so a single fact can be disclosed and "
    "proven bound to the public receipt while the rest of the response stays "
    "private. This is selective disclosure, not zero-knowledge; a disclosed value "
    "is trusted as the committed value. Semantic claims and vitals remain register, "
    "not ground-truth honesty."
)


def node_verify_disclosure(disclosure: dict, root: str) -> str:
    script = (
        f"const v=require({json.dumps(str(JS))});"
        "const o=JSON.parse(require('fs').readFileSync(0,'utf8'));"
        "process.stdout.write(v.verifyDisclosure(o.disclosure,o.root)?'OK':'FAIL');"
    )
    r = subprocess.run([NODE, "-e", script],
                       input=json.dumps({"disclosure": disclosure, "root": root}, ensure_ascii=False),
                       capture_output=True, text=True, check=True)
    return r.stdout.strip()


def main() -> int:
    if NODE is None:
        print("node not on PATH — cannot run the cross-language gate")
        return 2

    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD^{commit}"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    att = attest(REPORT, REPO, prompt=PROMPT, vitals=True, redactable=True)
    art = att.artifact
    root = art["digest"]["redactable"]["root"]

    # K1 + K2: disclose ONLY the sycophancy score.
    disc = att.disclose(["vitals/scores/sycophancy"])
    k1_genuine = verify_disclosure(disc, root=root)
    tampered = copy.deepcopy(disc)
    tampered["fields"][0]["value"] = 0.999
    k1_tamper_caught = not verify_disclosure(tampered, root=root)

    blob = json.dumps(disc, ensure_ascii=False)
    k2_text_hidden = (
        REPORT not in blob
        and "report" not in {f["pointer"].split("/")[0] for f in disc["fields"]}
        and [f["pointer"] for f in disc["fields"]] == ["vitals/scores/sycophancy"]
    )

    # K2 decisive: the salt is load-bearing.
    ptr, domain, truth = "claims/0/verdict", ["PASS", "FAIL", "ERROR"], "FAIL"
    unsalted = _leaf_hash("", ptr, truth)
    unsalted_recovered = [v for v in domain if _leaf_hash("", ptr, v) == unsalted] == [truth]
    salt = secrets.token_hex(32)
    salted = _leaf_hash(salt, ptr, truth)
    salted_unrecoverable = [v for v in domain if _leaf_hash("", ptr, v) == salted] == []
    k2_salt_load_bearing = unsalted_recovered and salted_unrecoverable

    # K3 additive: legacy + portable unchanged vs a non-redactable attestation.
    plain = attest(REPORT, REPO, prompt=PROMPT, vitals=True).artifact
    k3_additive = (
        _compute_digest(plain) == _compute_digest(art)
        and _compute_portable_digest(plain) == _compute_portable_digest(art)
        and "salts" not in art["digest"]["redactable"]
    )

    # K3 cross-language.
    k3_node_ok = node_verify_disclosure(disc, root) == "OK"
    k3_node_tamper = node_verify_disclosure(tampered, root) == "FAIL"

    # P4 compose: the redactable root is a transparency-log leaf; a consistency
    # proof binds this confidential receipt to the no-suppression history.
    earlier = [att2.artifact["digest"]["redactable"]["root"]
               for att2 in (attest(s, REPO, prompt=PROMPT, vitals=True, redactable=True)
                            for s in ("The version is 7.7.12.", "I added selective disclosure."))]
    log = TransparencyLog(earlier + [root])              # the confidential receipt is leaf 2
    witnessed_root = TransparencyLog(earlier).root()
    incl = log.inclusion_proof(2)
    from styxx.transparency import verify_inclusion
    p4_included = verify_inclusion(incl)
    p4_append_ok = verify_consistency(log.consistency_proof(len(earlier)), first_root=witnessed_root)
    # and a rewrite of the earlier history is caught
    rew = TransparencyLog(["TAMPERED"] + earlier[1:] + [root])
    p4_rewrite_caught = not verify_consistency(rew.consistency_proof(len(earlier)), first_root=witnessed_root)

    all_ok = (
        k1_genuine and k1_tamper_caught
        and k2_text_hidden and k2_salt_load_bearing
        and k3_additive and k3_node_ok and k3_node_tamper
        and p4_included and p4_append_ok and p4_rewrite_caught
    )
    receipt = {
        "dogfood": "redactable cognometric attestation — disclose one fact, keep the rest private",
        "head_commit": head,
        "node_version": subprocess.run([NODE, "-v"], capture_output=True, text=True).stdout.strip(),
        "js_verifier": str(JS.relative_to(REPO)),
        "redactable_root": root,
        "tree_size_fields": art["digest"]["redactable"]["tree_size"],
        "disclosed_pointers": [f["pointer"] for f in disc["fields"]],
        "disclosed_sycophancy": disc["fields"][0]["value"],
        "K1_disclosure_sound": k1_genuine and k1_tamper_caught,
        "K2_confidentiality": {
            "undisclosed_text_hidden": k2_text_hidden,
            "salt_load_bearing": k2_salt_load_bearing,
        },
        "K3_additive_and_cross_language": {
            "legacy_and_portable_unchanged": k3_additive,
            "node_verifies": k3_node_ok,
            "node_catches_tamper": k3_node_tamper,
        },
        "P4_composes_with_transparency_log": {
            "redactable_root_included_in_log": p4_included,
            "append_only_passes": p4_append_ok,
            "rewrite_of_history_caught": p4_rewrite_caught,
        },
        "honest_scope": (
            "selective disclosure, NOT zero-knowledge — no predicate/range over a hidden "
            "value; a disclosed value is the COMMITTED value (commit-time/re-seal boundary, "
            "caught only via the transparency log + an external witness). leaks field count "
            "+ disclosed pointers/values; hides everything undisclosed."
        ),
        "RESULT": "SURVIVED" if all_ok else "FALSIFIED",
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    print("\nRESULT:", "SURVIVED" if all_ok else "FALSIFIED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
