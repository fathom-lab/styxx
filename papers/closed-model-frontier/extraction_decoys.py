"""Build the extraction-panel decoy set — the repair owed by
ADDENDUM_extraction_ceiling_gate_unsatisfiable_2026_09_01.md.

WHY THIS FILE EXISTS. PREREG_extraction_ceiling_2026_09_01.md froze a reliability gate its own
packet could not supply: it assumed v14_packet.json's 30 sealed decoys split 15 claim /
15 not-a-claim. They do not. `decoy_verified` are real claims the gate verified;
`decoy_synthetic_contradiction` are the same real claims with the PATH perturbed to a `zz_`
name. The perturbation operates on the path, never on the speech act, so all 30 answer CLAIM to
the extraction question — 30 positives, zero negatives. The panel was voided and no E reported.

WHAT THIS BUILDS. A two-sided decoy set for the substituted question:

    Ignoring entirely whether the gate's verdict was correct, and reading only the author's
    summary: is the author making a claim about this path — asserting that THIS change created,
    deleted or touched it?  CLAIM / NOT-A-CLAIM / UNREADABLE

  * 15 CLAIM decoys — taken UNMODIFIED from v14_packet.json's `decoy_verified`, which are real
    claims the gate verified. Nothing is synthesised on this side.
  * 15 NOT-A-CLAIM decoys — synthesised from real VERIFIED path claims drawn from the
    DEVELOPMENT split only, by one of three frames committed here in a 5/5/5 ratio. The
    development restriction is SPLIT_external_corpus_2026_08_31.md rule 1: held-out prose is not
    consumed to build the instrument that scores held-out prose.

The frames are authorship, and authorship is judgment. That is disclosed in the addendum and in
any result that quotes a number from this set. What the frames buy is not neutrality; it is that
the construction is STATED, mechanical given the source sentence, and auditable.

The perturbation is deliberately the same SHAPE as the packet's own
`decoy_synthetic_contradiction` — a stated string transform of a real item — so this side of the
decoy set is no more manufactured than the side that already shipped.

    python papers/closed-model-frontier/extraction_decoys.py build

Writes extraction_decoys.json and extraction_decoys_digest.txt. Runs NO panel and computes NO E.
"""
from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import styxx.diffgate as DG                                    # noqa: E402
from external1_harness import reconstruct                      # noqa: E402

DB = HERE / "external1_shelf.sqlite"
V14_PACKET = HERE / "v14_packet.json"
V14_KEY = HERE / "v14_key_SEALED.json"
OUT = HERE / "extraction_decoys.json"
DIGEST = HERE / "extraction_decoys_digest.txt"

SEED = 20260901
N_PER_FRAME = 5                      # 5 + 5 + 5 = 15 NOT-A-CLAIM
N_CLAIM = 15


def bucket(repo_url: str) -> int:
    n = (repo_url or "").strip().rstrip("/").lower()
    return int(hashlib.sha256(n.encode("utf-8")).hexdigest()[:8], 16) % 10


# ──────────────────────────────────────────────────────────────────────────
# The three frames. Committed here, in this ratio, before any panel is prompted.
# Each takes a real claim sentence and the path it claims, and returns prose in
# which the author is NOT asserting that this change touched that path.
# ──────────────────────────────────────────────────────────────────────────

def frame_negation(text: str, path: str) -> str:
    return (f"To be explicit about scope: this change does not touch `{path}`. "
            f"That file is unchanged here and stays exactly as it is on main.")


def frame_quotation(text: str, path: str) -> str:
    one = " ".join(str(text).split())
    if len(one) > 160:
        one = one[:157].rstrip() + "..."
    return (f"For reference, the linked issue asks for the following: \"{one}\" "
            f"That work is tracked separately and is not part of this pull request.")


def frame_comparative(text: str, path: str) -> str:
    return (f"The approach here follows the same pattern `{path}` already uses, which was "
            f"settled in an earlier pull request. Nothing in this change modifies that file.")


FRAMES = [("negation", frame_negation),
          ("quotation", frame_quotation),
          ("comparative_reference", frame_comparative)]


def build() -> int:
    t0 = time.time()
    DG.WITHHOLD_PATH_ACCUSATION = False        # same posture v14_packet.py builds under

    # ---- 15 CLAIM decoys: unmodified, straight out of the sealed V14 packet ----
    pkt = json.loads(V14_PACKET.read_text(encoding="utf-8"))
    key = json.loads(V14_KEY.read_text(encoding="utf-8"))
    by_id = {it["id"]: it for it in pkt["items"]}
    claim_ids = sorted(i for i, v in key.items() if v.get("truth") == "decoy_verified")
    if len(claim_ids) < N_CLAIM:
        print(f"REFUSED: only {len(claim_ids)} decoy_verified items available")
        return 1
    claim_side = []
    for iid in claim_ids[:N_CLAIM]:
        src = by_id[iid]
        claim_side.append({
            "id": f"XD-C{len(claim_side):02d}",
            "expected": "CLAIM",
            "construction": "unmodified decoy_verified from v14_packet.json",
            "source_v14_id": iid,
            "agent": src["agent"], "url": src["url"],
            "claim_kind": src["claim_kind"], "claim_text": src["claim_text"],
            "claim_detail": src["claim_detail"], "changed_files": src["changed_files"],
        })

    # ---- 15 NOT-A-CLAIM decoys: synthesised from DEVELOPMENT verified claims ----
    con = sqlite3.connect(DB)
    ver, facts, seen_pr = [], {}, 0
    for pid, agent, title, body, url in con.execute(
            "SELECT id, agent, title, body, html_url FROM pr "
            "WHERE body IS NOT NULL AND body != ''"):
        if bucket("/".join((url or "").split("/")[:5])) >= 3:
            continue                            # HELD-OUT — not touched to build this
        rows = con.execute("SELECT filename, status, patch FROM f WHERE pr_id=?",
                           (pid,)).fetchall()
        if not rows:
            continue
        seen_pr += 1
        if seen_pr % 2000 == 0:
            print(f"   ...{seen_pr} development PRs scanned, {len(ver)} verified claims "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        diff, implied = reconstruct(rows)
        parsed, _ = DG.parse_unified_diff(diff)
        if parsed != implied:
            continue
        g = DG.gate_diff_text(f"{title or ''}\n\n{body}", diff, run=None, strict=False)
        st = {}
        for fn, s, _p in rows:
            if fn and fn not in st:
                st[fn] = (s or "").lower() or "modified"
        for i, c in enumerate(g.claims):
            if not c.kind.startswith("file_") or c.verdict != "VERIFIED":
                continue
            if not (c.detail or {}).get("path"):
                continue
            ver.append({"pr_id": pid, "agent": agent, "url": url, "claim_index": i,
                        "kind": c.kind, "text": c.text, "detail": c.detail})
            facts[pid] = st
    con.close()
    print(f"DEVELOPMENT population: {seen_pr} PRs, {len(ver)} verified path claims "
          f"[{time.time()-t0:.0f}s]", flush=True)

    need = N_PER_FRAME * len(FRAMES)
    if len(ver) < need:
        print(f"REFUSED: need {need} development verified claims, have {len(ver)}")
        return 1

    rng = random.Random(SEED)
    picked = rng.sample(ver, need)
    notclaim_side = []
    for k, src in enumerate(picked):
        fname, fn = FRAMES[k // N_PER_FRAME]
        path = src["detail"]["path"]
        notclaim_side.append({
            "id": f"XD-N{k:02d}",
            "expected": "NOT-A-CLAIM",
            "construction": f"synthesised by frame '{fname}' from a DEVELOPMENT verified claim",
            "frame": fname,
            "source_url": src["url"], "source_claim_text": src["text"],
            "agent": src["agent"], "url": src["url"],
            "claim_kind": src["kind"],
            "claim_text": fn(src["text"], path),
            "claim_detail": {"path": path},
            "changed_files": [{"path": p, "status": s}
                              for p, s in facts[src["pr_id"]].items()],
        })

    items = claim_side + notclaim_side
    rng.shuffle(items)

    out = {
        "what": ("two-sided decoy set for the EXTRACTION question — 15 CLAIM taken unmodified "
                 "from the sealed V14 packet, 15 NOT-A-CLAIM synthesised by three committed "
                 "frames from DEVELOPMENT-split verified claims"),
        "repairs": "ADDENDUM_extraction_ceiling_gate_unsatisfiable_2026_09_01.md",
        "question": ("Ignoring entirely whether the gate's verdict was correct, and reading only "
                     "the author's summary: is the author making a claim about this path — "
                     "asserting that THIS change created, deleted or touched it? "
                     "CLAIM / NOT-A-CLAIM / UNREADABLE"),
        "seed": SEED,
        "split_of_notaclaim_sources": "DEVELOPMENT only (bucket < 3), per SPLIT rule 1",
        "frames": {n: f.__doc__ or n for n, f in FRAMES},
        "frame_ratio": {n: N_PER_FRAME for n, _ in FRAMES},
        "authorship_disclosure": (
            "The NOT-A-CLAIM side is authored by us. The frames are judgment, stated in advance "
            "and mechanical given the source sentence. Any result quoting this set must say so "
            "in the same breath as the number."),
        "n_claim": len(claim_side), "n_notaclaim": len(notclaim_side),
        "items": items,
    }
    body = json.dumps(out, indent=2, ensure_ascii=False) + "\n"
    OUT.write_text(body, encoding="utf-8")
    dg = hashlib.sha256(body.encode("utf-8")).hexdigest()
    DIGEST.write_text(
        f"sha256 = {dg}\nitems = {len(items)}\n"
        f"claim = {len(claim_side)}  not_a_claim = {len(notclaim_side)}\n"
        f"seed = {SEED}\nbuilt = extraction_decoys.py\n"
        "committed BEFORE any panel is prompted\n", encoding="utf-8")
    print(f"\nwrote {OUT.name}  sha256 {dg[:16]}...  "
          f"{len(claim_side)} CLAIM / {len(notclaim_side)} NOT-A-CLAIM")
    print("NO panel was run and no E was computed by this file.")
    return 0


if __name__ == "__main__":
    sys.exit(build())
