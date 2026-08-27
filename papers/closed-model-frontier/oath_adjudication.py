"""Build blind adjudication packets over the external corpus, and score them.

Protocol: `PROTOCOL_oath_external_corpus_2026_08_27.md`, ground-truth section.

**Frozen before the collected ledger was read.** This file was written and committed while the
collection was still running, so the decoy sampling rule, the sample sizes, the seed, the packet
size and the question text below were all fixed without their author having seen a single accused
token. That ordering is the only thing that makes the blinding claim worth anything.

## Why decoys exist

The protocol's first draft said adjudicators "are not told" what the verifier decided. That is
worthless when every item in the packet is an accused token: membership alone leaks the verdict,
and a panel that knows the answer is measuring its own agreeableness. So packets are salted with
tokens the verifier ABSTAINED on and tokens it VERIFIED, shuffled together and presented
identically. Inclusion no longer carries status.

The decoys are not padding. They buy the measurement the pilot never had:

* judgements on **accused** tokens give the **false-accusation rate** — of what it flagged, how
  much was not a claim at all;
* judgements on **abstained** tokens give the **miss rate** — of what it declined to check, how
  much was a checkable claim. On an external corpus where the verifier abstains on roughly
  nineteen tokens in twenty, this is the more important number, and it is the one that separates
  *calibrated restraint* from *inertness*. An instrument that accuses nothing because it checks
  nothing is not accurate;
* judgements on **verified** tokens are a sanity arm: if adjudicators call verified tokens
  non-claims at a high rate, the panel disagrees with the instrument about what a claim IS, and
  every other number here is suspect.

## The frozen sampling rule

* **Every** accused (UNGROUNDED) token. Not sampled.
* Abstained decoys: uniform sample without replacement, `n = min(150, available)`.
* Verified decoys: uniform sample without replacement, `n = min(75, available)`.
* One deterministic seed for both draws and for the shuffle, recorded in the output.
* Packets of 25 items, in shuffled order, each judged independently by three adjudicators.

Ties and unanimity failures resolve **against the instrument**: a token no majority calls a real
claim counts as a false accusation when it was accused, and a token no majority calls a claim does
NOT count as a miss when it was abstained. Both directions make the instrument look worse.

  python papers/closed-model-frontier/oath_adjudication.py build
  python papers/closed-model-frontier/oath_adjudication.py score judgements.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

LEDGER = HERE / "oath_external_corpus_ledger.jsonl"
PACKETS = HERE / "oath_adjudication_packets.json"
KEY = HERE / "oath_adjudication_key.json"
RESULT = HERE / "oath_adjudication_result.json"

SEED = 20260827
N_ABSTAIN_DECOYS = 150
N_VERIFIED_DECOYS = 75
PACKET_SIZE = 25
PANEL = 3

QUESTION = (
    "Is this token a CLAIM whose truth a reader could check against the cited receipts?\n"
    "\n"
    "Answer CLAIM if the number asserts something about a measurement, result, dataset, or "
    "outcome that the document is reporting — something a reader could look up in the receipts "
    "and find to be right or wrong.\n"
    "\n"
    "Answer NOT_A_CLAIM if the number is anything else: a version, a date, a hyperparameter or "
    "configuration value, a file path or index, a count of items in a list, a citation, a URL "
    "fragment, a table row number, a quantity quoted FROM somewhere else rather than asserted, "
    "or a number inside a formula or code snippet that names no measured quantity.\n"
    "\n"
    "Answer UNSURE only if you genuinely cannot tell from the line and its context. UNSURE is "
    "counted against the instrument, so use it honestly rather than defensively."
)


def load_ledger() -> list[dict]:
    if not LEDGER.exists():
        raise SystemExit(f"no ledger at {LEDGER}; run oath_external_corpus.py first")
    return [json.loads(ln) for ln in LEDGER.read_text(encoding="utf-8").splitlines() if ln.strip()]


def build() -> dict:
    rows = load_ledger()
    rng = random.Random(SEED)

    accused = [r for r in rows if r["status"] == "UNGROUNDED"]
    abstained = [r for r in rows if r["status"] == "ABSTAIN"]
    verified = [r for r in rows if r["status"] == "VERIFIED"]

    dec_a = rng.sample(abstained, min(N_ABSTAIN_DECOYS, len(abstained)))
    dec_v = rng.sample(verified, min(N_VERIFIED_DECOYS, len(verified)))

    items, key = [], {}
    for src in (accused, dec_a, dec_v):
        for r in src:
            iid = f"T{len(items):04d}"
            # Presentation is identical across arms. `status` and `receipt_ref` are the answer and
            # are written to the KEY file only — they never enter a packet.
            items.append({
                "id": iid,
                "repo": r["repo"],
                "line": r["line"],
                "token": r["token"],
                "context": r["context"],
                "obligating_words": r.get("obligating_words", []),
            })
            key[iid] = {"status": r["status"], "repo": r["repo"], "line": r["line"],
                        "token": r["token"], "receipt_ref": r.get("receipt_ref")}
    rng.shuffle(items)
    for i, it in enumerate(items):
        it["id_shuffled_position"] = i

    packets = [items[i:i + PACKET_SIZE] for i in range(0, len(items), PACKET_SIZE)]
    payload = {
        "protocol": "papers/closed-model-frontier/PROTOCOL_oath_external_corpus_2026_08_27.md",
        "frozen_before_ledger_was_read": True,
        "seed": SEED, "packet_size": PACKET_SIZE, "panel": PANEL,
        "question": QUESTION,
        "composition": {"accused": len(accused), "abstain_decoys": len(dec_a),
                        "verified_decoys": len(dec_v), "total_items": len(items),
                        "packets": len(packets)},
        "available": {"accused": len(accused), "abstained": len(abstained),
                      "verified": len(verified)},
        "blinding": ("Items from all three arms are shuffled together and presented identically. "
                     "An adjudicator cannot infer the verifier's decision from membership."),
        "packets": packets,
    }
    PACKETS.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
                       encoding="utf-8", newline="\n")
    KEY.write_text(json.dumps(key, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8", newline="\n")
    print(f"accused {len(accused)}  abstain-decoys {len(dec_a)}  verified-decoys {len(dec_v)}")
    print(f"-> {PACKETS.name} ({len(packets)} packets)  /  {KEY.name} (withheld)")
    return payload


def _verdict(votes: list[str]) -> str:
    """Majority of three. No majority resolves AGAINST the instrument, i.e. to NOT_A_CLAIM."""
    for v in ("CLAIM", "NOT_A_CLAIM"):
        if sum(1 for x in votes if x == v) * 2 > len(votes):
            return v
    return "NOT_A_CLAIM"


def score(judgements_path: Path) -> dict:
    key = json.loads(KEY.read_text(encoding="utf-8"))
    judged = json.loads(Path(judgements_path).read_text(encoding="utf-8"))

    by_item: dict[str, list[str]] = {}
    for j in judged:
        by_item.setdefault(j["id"], []).append(j["verdict"])

    arms = {"UNGROUNDED": [], "ABSTAIN": [], "VERIFIED": []}
    unanimous, split = 0, 0
    for iid, votes in by_item.items():
        if iid not in key:
            continue
        v = _verdict(votes)
        arms[key[iid]["status"]].append({"id": iid, "verdict": v, "votes": votes, **key[iid]})
        if len(set(votes)) == 1:
            unanimous += 1
        else:
            split += 1

    def share(rows, want):
        return round(sum(1 for r in rows if r["verdict"] == want) / len(rows), 4) if rows else None

    out = {
        "panel_size": PANEL,
        "items_scored": sum(len(v) for v in arms.values()),
        "agreement": {"unanimous": unanimous, "split": split,
                      "unanimity_share": round(unanimous / (unanimous + split), 4)
                      if (unanimous + split) else None,
                      "note": ("Adjudicators are LLM agents of one family; high unanimity is NOT "
                               "evidence of correctness, it is the correlated-error ceiling the "
                               "protocol discloses.")},
        "false_accusation_rate": {
            "n": len(arms["UNGROUNDED"]),
            "rate": share(arms["UNGROUNDED"], "NOT_A_CLAIM"),
            "reading": "of what the verifier ACCUSED, the share no majority called a real claim",
        },
        "miss_rate": {
            "n": len(arms["ABSTAIN"]),
            "rate": share(arms["ABSTAIN"], "CLAIM"),
            "reading": ("of what the verifier DECLINED to check, the share a majority called a "
                        "checkable claim — restraint if low, blindness if high"),
        },
        "verified_arm_sanity": {
            "n": len(arms["VERIFIED"]),
            "rate": share(arms["VERIFIED"], "CLAIM"),
            "reading": ("of what the verifier VERIFIED, the share the panel agrees is a claim. "
                        "If this is low the panel and the instrument disagree about what a claim "
                        "IS, and every other number here is suspect."),
        },
        "per_arm_detail": arms,
    }
    RESULT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n",
                      encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in out.items() if k != "per_arm_detail"}, indent=1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "score"))
    ap.add_argument("judgements", nargs="?", default=None)
    a = ap.parse_args()
    if a.cmd == "build":
        build()
    else:
        if not a.judgements:
            raise SystemExit("score needs a judgements JSON path")
        score(Path(a.judgements))
    return 0


if __name__ == "__main__":
    sys.exit(main())
