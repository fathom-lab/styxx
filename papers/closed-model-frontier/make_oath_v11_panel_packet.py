"""Build the BLIND adjudication packet for OATH v0.11 gate G4'b.

The prereg (PREREG_oath_v11_row_ordinal_retraction_2026_08_25.md, §Battery + gates, G4'b)
requires a SECOND adjudicator, not the author of `oath_v10_panel_isclaim.json`, to
re-adjudicate the full 11-token PROSPECTUS roster PLUS a fresh draw of 10 non-PROSPECTUS
tokens from the 150-token class roster, reading OFF-arm statuses only, blind to the prior
panel's calls and to the clause's reason codes.

What the blinding actually is, stated plainly: the full 11-token roster goes in, so the
adjudicator cannot tell the four RETRACTION TARGETS from the seven collateral tokens, and the
10 fresh non-PROSPECTUS cases are interleaved by a pre-committed shuffle so document identity
does not sort the packet. The packet carries no target flag, no reason code, no prior call,
and no prereg text.

Draw parameters are pre-committed HERE, before the packet is generated:
  sort_key   = ('doc', 'line', 'token')      -- same key the prior panel used, for comparability
  fresh rng  = random.Random(1111).sample(sorted_non_prospectus_roster, 10)
  shuffle    = random.Random(2111).shuffle(cases)

Seed 1111 is deliberately NOT the prior panel's 11: this is a FRESH draw, and re-drawing the
prior sample would make "blind to the prior panel's calls" a formality. Overlap with the prior
30 is computed and disclosed rather than engineered to zero.

Writes: oath_v11_panel_packet.json (the blind input) and oath_v11_panel_key.json (the
un-blinding key, which the adjudicator never reads).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
CENSUS = HERE / "oath_v10_ordinal_census.json"
PRIOR_PANEL = HERE / "oath_v10_panel_isclaim.json"
PROSPECTUS = "papers/agent-conscience/PROSPECTUS_knowsay_2026_07_27.md"

SORT_KEY = ("doc", "line", "token")
FRESH_SEED = 1111
FRESH_N = 10
SHUFFLE_SEED = 2111

# The frozen CLAIM/LABEL definition, carried verbatim from the prior panel's `purpose` so the
# second adjudicator scores under the SAME definition (the prereg's requirement), without
# seeing any of the prior panel's calls.
DEFINITION = (
    "For each token: does the number assert something about the world that a receipt could "
    "confirm or contradict (CLAIM), or is it a row number / list marker / identifier with no "
    "truth condition (LABEL)? A LABEL has no truth condition, so neither VERIFIED nor "
    "UNGROUNDED is meaningful for it and ABSTAIN is the only defensible status. "
    "TIES RESOLVE TOWARD CLAIM."
)

# Fields copied into a blind case. `status` is the OFF-arm shipped status, which the prereg
# explicitly permits. Everything that could identify a retraction target is excluded.
BLIND_FIELDS = (
    "rel", "doc", "line", "token", "value", "first_col_header", "sole_token_in_cell",
    "status", "receipt_ref", "context",
)


def _leaf_view(row: dict) -> dict | None:
    """The receipt leaf the OFF-arm verifier bound this token to, if any.

    Included because the prior panel read it ("where I quote a leaf I opened the receipt
    file") and withholding it would make the two adjudications answer different questions.
    """
    leaf = row.get("leaf")
    if not leaf:
        return None
    return {
        "receipt": leaf.get("receipt"),
        "path": leaf.get("path"),
        "leaf_value": leaf.get("leaf_value"),
        "terminal_segment": leaf.get("terminal_segment"),
        "terminal_is_index_name": leaf.get("terminal_is_index_name"),
        "leaf_value_equals_its_own_subscript": leaf.get("value_equals_subscript"),
    }


def main() -> int:
    census = json.loads(CENSUS.read_text(encoding="utf-8"))
    roster = census["roster"]
    if len(roster) != 150:
        raise SystemExit(f"roster is {len(roster)} tokens, expected the frozen 150")

    ordered = sorted(roster, key=lambda r: tuple(r[k] for k in SORT_KEY))
    prospectus = [r for r in ordered if r["rel"] == PROSPECTUS]
    others = [r for r in ordered if r["rel"] != PROSPECTUS]
    if len(prospectus) != 11:
        raise SystemExit(f"{len(prospectus)} PROSPECTUS roster tokens, expected 11")

    fresh = random.Random(FRESH_SEED).sample(others, FRESH_N)

    # Disclosure, not engineering: how much of the fresh draw the prior panel already saw.
    prior = json.loads(PRIOR_PANEL.read_text(encoding="utf-8"))
    prior_coords = {(c["rel"], c["line"], c["token"]) for c in prior["cases"]}
    overlap = sorted(f"{r['doc']}:L{r['line']}:{r['token']}" for r in fresh
                     if (r["rel"], r["line"], r["token"]) in prior_coords)

    selected = prospectus + fresh
    random.Random(SHUFFLE_SEED).shuffle(selected)

    cases, key = [], []
    for i, row in enumerate(selected, 1):
        cid = f"C{i:02d}"
        case = {"case": cid}
        case.update({k: row.get(k) for k in BLIND_FIELDS})
        case["receipt_leaf"] = _leaf_view(row)
        cases.append(case)
        key.append({
            "case": cid, "rel": row["rel"], "line": row["line"], "token": row["token"],
            "arm": "prospectus_roster" if row["rel"] == PROSPECTUS else "fresh_draw",
            "off_arm_status": row["status"],
        })

    packet = {
        "gate": "G4'b — second blind adjudicator",
        "prereg": ("papers/closed-model-frontier/"
                   "PREREG_oath_v11_row_ordinal_retraction_2026_08_25.md"),
        "lens": "IS THIS TOKEN A CLAIM AT ALL?",
        "definition": DEFINITION,
        "instructions": [
            "Call every case CLAIM or LABEL under the definition above. Ties resolve toward CLAIM.",
            "Record a confidence (HIGH / MEDIUM / LOW) and a one-sentence reason for each call.",
            "`status` is the status the CURRENT shipped verifier assigns with the candidate "
            "clause OFF. It is context, not an answer: the question is whether the token is a "
            "claim at all, not whether the verifier got it right.",
            "You may open the cited document and the cited receipt JSON. Do not read any "
            "PREREG_oath_v11* file, oath_v10_panel_isclaim.json, or any v0.11 battery or "
            "implementation code.",
        ],
        "sampling": {
            "population": "the census roster: 150 candidate first-cell markdown-table tokens "
                          "in the 140-document fully-resolvable certified frame",
            "sort_key": list(SORT_KEY),
            "prospectus_roster": len(prospectus),
            "fresh_draw": FRESH_N,
            "fresh_rng": f"random.Random({FRESH_SEED}).sample(sorted_non_prospectus_roster, {FRESH_N})",
            "presentation_shuffle": f"random.Random({SHUFFLE_SEED}).shuffle(cases)",
            "examined": len(cases),
            "fresh_draw_overlap_with_prior_panel": overlap,
            "fresh_draw_overlap_count": len(overlap),
            "overlap_note": "Disclosed, not engineered. Seed 1111 was pre-committed in this "
                            "script before the draw was inspected; a redraw to force zero "
                            "overlap would be selection on an observed value.",
        },
        "cases": cases,
    }

    (HERE / "oath_v11_panel_packet.json").write_text(
        json.dumps(packet, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (HERE / "oath_v11_panel_key.json").write_text(
        json.dumps({"note": "UN-BLINDING KEY. The adjudicator must not read this file.",
                    "key": key}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"packet: {len(cases)} cases ({len(prospectus)} roster + {FRESH_N} fresh)")
    print(f"fresh-draw overlap with the prior panel's 30: {len(overlap)} {overlap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
