"""Live dogfood: prove a styxx attestation verifies in a SECOND LANGUAGE (prereg'd).

Builds a real (task -> self-report) chain over styxx's own HEAD with vitals,
then for each artifact shape:

  * Python computes digest.portable (RFC 8785 / JCS canonical)             [build]
  * Node (web/styxx_verify.js, zero styxx, zero deps) recomputes it and the
    two agree byte-for-byte — INCLUDING the saturating coverage=1.0 token
    that diverged under the legacy scheme                                  [K1]
  * the legacy digest.value is byte-identical with portable added         [K2]
  * Node alone catches a tampered portable digest                          [K3]

Requires `node` on PATH. Reports whichever way it lands. Frozen receipt next
to this script.
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from styxx.attestation import (
    _compute_digest,
    attest,
    attest_chain,
)

REPO = Path(__file__).resolve().parents[2]
JS = REPO / "web" / "styxx_verify.js"
RECEIPT = Path(__file__).resolve().parent / "portable_attestation_self_2026_05_28.json"
NODE = shutil.which("node")

PROMPT = "Report on the attestation work you shipped this session."
REPORT = (
    "The version is 7.7.12. I shipped a portable, cross-language content address: "
    "an additive digest.portable over an RFC 8785 canonical form that reproduces "
    "byte-for-byte in JavaScript, so a styxx attestation can be verified in a "
    "browser in any language with zero install and zero trust. The legacy digest "
    "is unchanged; semantic claims and vitals remain NOT CHECKED by structure."
)


def node_portable_digest(artifact: dict) -> str:
    script = (
        f"const v=require({json.dumps(str(JS))});"
        "const a=JSON.parse(require('fs').readFileSync(0,'utf8'));"
        "process.stdout.write(v.portableDigest(a));"
    )
    r = subprocess.run([NODE, "-e", script], input=json.dumps(artifact, ensure_ascii=False),
                       capture_output=True, text=True, check=True)
    return r.stdout.strip()


def node_verify(artifact: dict) -> int:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
        fh.write(json.dumps(artifact, ensure_ascii=False))
        path = fh.name
    r = subprocess.run([NODE, str(JS), path], capture_output=True, text=True)
    Path(path).unlink(missing_ok=True)
    return r.returncode


def main() -> int:
    if NODE is None:
        print("node not on PATH — cannot run the cross-language gate")
        return 2

    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD^{commit}"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    plain = attest(REPORT, REPO).artifact
    vitals = attest(REPORT, REPO, prompt=PROMPT, vitals=True).artifact
    chain = attest_chain([(REPORT, head, PROMPT)], REPO, id_prefix="PORT", vitals=True).artifact
    # A single TRUE sentence saturates coverage to exactly 1.0 — the token that
    # diverges under the legacy scheme (python "1.0" vs js "1").
    sat = attest("The version is 7.7.12.", REPO, prompt=PROMPT, vitals=True).artifact

    # K1: Python portable == Node portable, byte-for-byte.
    k1 = {
        "plain": plain["digest"]["portable"]["value"] == node_portable_digest(plain),
        "vitals": vitals["digest"]["portable"]["value"] == node_portable_digest(vitals),
        "saturating_coverage_1_0": (
            sat["summary"]["coverage"] == 1.0
            and sat["digest"]["portable"]["value"] == node_portable_digest(sat)
        ),
        "chain_link0": (
            chain["links"][0]["attestation"]["digest"]["portable"]["value"]
            == node_portable_digest(chain["links"][0]["attestation"])
        ),
    }

    # K2: legacy digest untouched by the additive portable field.
    no_portable = copy.deepcopy(vitals)
    del no_portable["digest"]["portable"]
    k2_legacy_unchanged = (
        _compute_digest(no_portable) == vitals["digest"]["value"]
        and _compute_digest(vitals) == vitals["digest"]["value"]
    )

    # K3: Node alone catches a tampered portable digest.
    tampered = copy.deepcopy(plain)
    tampered["digest"]["portable"]["value"] = "0" * 64
    k3_node_catches = node_verify(tampered) == 1 and node_verify(plain) == 0

    receipt = {
        "dogfood": "portable (cross-language) attestation — verified in Node",
        "generated_at": chain["generated_at"],
        "head_commit": head,
        "node_version": subprocess.run([NODE, "-v"], capture_output=True, text=True).stdout.strip(),
        "js_verifier": str(JS.relative_to(REPO)),
        "python_portable_digest_plain": plain["digest"]["portable"]["value"],
        "node_portable_digest_plain": node_portable_digest(plain),
        "K1_python_eq_node": k1,
        "K1_all": all(k1.values()),
        "K2_legacy_digest_unchanged": k2_legacy_unchanged,
        "K3_node_catches_portable_tamper": k3_node_catches,
        "saturating_token_present": sat["summary"]["coverage"] == 1.0,
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))

    all_ok = all(k1.values()) and k2_legacy_unchanged and k3_node_catches
    print("\nRESULT:", "SURVIVED" if all_ok else "FALSIFIED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
