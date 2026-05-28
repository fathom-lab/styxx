"""Live dogfood: prove no styxx receipt can be SILENTLY SUPPRESSED (prereg'd).

Builds a Cognometric Transparency Log over a real sequence of styxx attestations
(leaves = each attestation's digest.portable.value, over styxx's own HEAD), then:

  * every leaf gets an inclusion proof that verifies; a tampered leaf fails  [K1]
  * a witness pins the tree head at size m; then for the SAME later log:
      - a pure append (first m leaves untouched) PASSES consistency m->n
      - an edit / delete / reorder / TRUNCATE of any of the first m witnessed
        leaves FAILS consistency m->n                                        [K2, decisive]
  * Node (web/styxx_verify.js, zero styxx, zero deps) reproduces the root and
    verifies every inclusion + consistency proof byte-for-byte                [K3]

K2 is the whole thesis: silent suppression of a PAST receipt is detectable to
anyone holding the earlier tree head. P4 (honest boundary, NOT a kill): this
needs a WITNESSED tree head — the data structure alone does not stop an operator
who never publishes one from equivocating; that needs tree-head gossip, as in CT.

Requires `node` on PATH. Reports whichever way it lands. Frozen receipt next to
this script.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from styxx.attestation import attest
from styxx.transparency import TransparencyLog, verify_consistency, verify_inclusion

REPO = Path(__file__).resolve().parents[2]
JS = REPO / "web" / "styxx_verify.js"
RECEIPT = Path(__file__).resolve().parent / "transparency_log_self_2026_05_28.json"
NODE = shutil.which("node")

PROMPT = "Report on one disciplined step you shipped this session."
# A real sequence of self-reports over styxx's own repo. Each becomes one leaf.
REPORTS = [
    "The version is 7.7.12.",
    "I shipped a stdlib-only standalone verifier with zero styxx imports.",
    "I added digest.portable, an RFC 8785 cross-language content address.",
    "I shipped a zero-dependency JavaScript verifier that runs in the browser.",
    "I pre-registered the transparency log before writing its code.",
    "I built inclusion and consistency proofs over attestation digests.",
    "I cross-validated the Merkle roots Python against Node byte-for-byte.",
    "The legacy digest.value is unchanged; nothing already issued is invalidated.",
]


def node_emit(payload: dict, script: str) -> str:
    full = (
        f"const v=require({json.dumps(str(JS))});"
        "const I=JSON.parse(require('fs').readFileSync(0,'utf8'));"
        + script
    )
    r = subprocess.run([NODE, "-e", full], input=json.dumps(payload, ensure_ascii=False),
                       capture_output=True, text=True, check=True)
    return r.stdout.strip()


def main() -> int:
    if NODE is None:
        print("node not on PATH — cannot run the cross-language gate")
        return 2

    # Build the real log: one portable digest per attestation, over HEAD.
    entries: list[str] = []
    for text in REPORTS:
        art = attest(text, REPO, prompt=PROMPT, vitals=True).artifact
        entries.append(art["digest"]["portable"]["value"])
    n = len(entries)
    log = TransparencyLog(list(entries))
    root_n = log.root()

    # K1: every leaf includes; a tampered leaf does not.
    k1_all_include = all(verify_inclusion(log.inclusion_proof(i)) for i in range(n))
    bad = dict(log.inclusion_proof(3), leaf_hash="0" * 64)
    k1_tamper_caught = not verify_inclusion(bad)

    # Witness a tree head at size m (BEFORE the later entries existed).
    m = 5
    witness = TransparencyLog(list(entries[:m]))
    witnessed_head = witness.tree_head(timestamp="2026-05-28T00:00:00+00:00")
    witnessed_root = witnessed_head["root"]

    # K2 (decisive): genuine append PASSES; any rewrite of the first m FAILS.
    append_ok = verify_consistency(log.consistency_proof(m), first_root=witnessed_root)

    def rewrite(entries_mut: list[str]) -> bool:
        """Return True iff the rewrite is CAUGHT (consistency fails)."""
        mutated = TransparencyLog(entries_mut)
        c = mutated.consistency_proof(m)
        return not verify_consistency(c, first_root=witnessed_root)

    edited = list(entries); edited[2] = "deadbeef" * 8
    deleted = [e for i, e in enumerate(entries) if i != 2]          # drop a past receipt
    reordered = list(entries); reordered[1], reordered[3] = reordered[3], reordered[1]
    truncated = entries[: m - 1] + entries[m:]                       # suppress the m-1 receipt

    k2 = {
        "genuine_append_passes": append_ok,
        "edit_caught": rewrite(edited),
        "delete_caught": rewrite(deleted),
        "reorder_caught": rewrite(reordered),
        "truncate_caught": rewrite(truncated),
    }

    # K3: Node reproduces the root and verifies inclusion + consistency.
    node_root = node_emit(
        {"entries": entries},
        "const ls=I.entries.map(v.leafHash);process.stdout.write(v.merkleTreeHash(ls));",
    )
    incl_proofs = [log.inclusion_proof(i) for i in range(n)]
    node_incl_bad = node_emit(
        incl_proofs,
        "let b=0;for(const p of I)if(!v.verifyInclusion(p))b++;process.stdout.write(String(b));",
    )
    cons_payload = {"good": log.consistency_proof(m), "edited": TransparencyLog(edited).consistency_proof(m),
                    "witnessed": witnessed_root}
    node_cons = node_emit(
        cons_payload,
        "let b=0;if(!v.verifyConsistency(I.good,I.witnessed))b++;"
        "if(v.verifyConsistency(I.edited,I.witnessed))b++;process.stdout.write(String(b));",
    )
    k3 = {
        "node_root_eq_python": node_root == root_n,
        "node_all_inclusions_ok": node_incl_bad == "0",
        "node_consistency_good_pass_bad_fail": node_cons == "0",
    }

    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD^{commit}"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    all_ok = (
        k1_all_include and k1_tamper_caught
        and all(k2.values())
        and all(k3.values())
    )
    receipt = {
        "dogfood": "cognometric transparency log — no silent suppression (verified in Node)",
        "head_commit": head,
        "node_version": subprocess.run([NODE, "-v"], capture_output=True, text=True).stdout.strip(),
        "js_verifier": str(JS.relative_to(REPO)),
        "log_size": n,
        "root": root_n,
        "witnessed_size_m": m,
        "witnessed_root": witnessed_root,
        "witnessed_tree_head_digest": witnessed_head["digest"]["value"],
        "K1_inclusion_sound_complete": k1_all_include and k1_tamper_caught,
        "K2_consistency_catches_rewrite": k2,
        "K2_all": all(k2.values()),
        "K3_cross_language": k3,
        "K3_all": all(k3.values()),
        "P4_honest_boundary": (
            "append-only-ness is proven RELATIVE to the witnessed tree head above; "
            "defeating equivocation by an operator who never publishes a head needs "
            "tree-head gossip/witnessing, exactly as in CT — out of scope, not claimed."
        ),
        "RESULT": "SURVIVED" if all_ok else "FALSIFIED",
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    print("\nRESULT:", "SURVIVED" if all_ok else "FALSIFIED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
