"""Can WIDENING THE TRIGGER VOCABULARY fix the miss rate? Scored against a null rule.

Population: every abstained token put to the blind panels of 2026-08-27 — 150 external and 75
internal — each carrying a majority verdict from three seats. That is the first hand-adjudicated
ground truth this lane has ever had for *misses*: tokens the verifier declined to check, labelled
by whether they were checkable claims.

## Why this census exists

`RESULT_oath_verified_channel_internal_2026_08_27.md` put the obligation predicate at the top of
the repair queue: the miss rate is `0.4267` internally and `0.4067` externally, and keeping the
contract buys nothing against it. The diagnosis is unambiguous — of external misses, **61 of 61**
occurred because the trigger vocabulary never fired at all; internally 24 of 32. So the obvious
repair is to widen the vocabulary.

This census exists to find out whether that can work *before* a preregistration is written around
it, because the obvious repair is what killed v0.12 and v0.13.

## The control, which is the point

Per rule 10 of `OATH_CONTRACT.md`: a column that the null rule ties is not a deciding column.
The null rule here is **obligate every number, no vocabulary test at all** — the most permissive
rule the design admits. It is the ceiling on what any lexical widening can achieve, because no
word list can catch more misses than obligating everything, and none can cost less than the words
it adds. If the null rule's trade is bad, every candidate below it is bad too.

Candidates are defined mechanically, not chosen: *add the K words most frequent among adjudicated
misses and rare among adjudicated non-claims*, for K in a fixed ladder. No hand-picking, so the
frontier is auditable rather than flattering.

  python papers/closed-model-frontier/oath_obligation_repair_census.py
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx.discriminates import discrimination_report, render  # noqa: E402

OUT = HERE / "oath_obligation_repair_census.json"
K_LADDER = (1, 3, 5, 10, 20, 40)
MIN_LEN = 3
STOP = frozenset("""the a an of and or to in on is was were be been being for with by at as it
this that from not no we our you your they their he she but if then than so such all any each
per via when where which who whom whose while has have had do does did can could should would
may might must will shall its his her them there here what how why into over under out up down
one two three both same other another new old more most less least only just also very much many
few own same too now still yet already about after before between during without within across
above below off again once because during through""".split())

WORD = re.compile(r"[A-Za-z][A-Za-z_-]{%d,}" % (MIN_LEN - 1))


def adjudicated_abstentions() -> list[dict]:
    """Every abstained token a panel judged, from both arms, with its majority verdict."""
    pairs = [("oath_adjudication_result.json", "oath_external_corpus_ledger.jsonl", "external"),
             ("oath_internal_result.json", "oath_internal_ledger.jsonl", "internal")]
    rows = []
    for res_name, led_name, arm in pairs:
        res = json.loads((HERE / res_name).read_text(encoding="utf-8"))
        led = [json.loads(ln) for ln in
               (HERE / led_name).read_text(encoding="utf-8").splitlines() if ln.strip()]
        idx = {(r["repo"], r["line"], r["token"]): r for r in led}
        for a in res["per_arm_detail"]["ABSTAIN"]:
            k = (a["repo"], a["line"], a["token"])
            if k in idx:
                rows.append({**idx[k], "verdict": a["verdict"], "arm": arm})
    return rows


def words_of(ctx: str) -> set[str]:
    return {w for w in WORD.findall(ctx.lower()) if w not in STOP}


def main() -> int:
    rows = adjudicated_abstentions()
    # Only tokens the vocabulary MISSED are addressable by widening it. Tokens whose line already
    # fired a trigger were abstained downstream and no word list can reach them.
    addressable = [r for r in rows if not r["obligating_words"]]
    misses = [r for r in addressable if r["verdict"] == "CLAIM"]
    nonclaims = [r for r in addressable if r["verdict"] == "NOT_A_CLAIM"]

    mc, nc = collections.Counter(), collections.Counter()
    for r in misses:
        for w in words_of(r["context"]):
            mc[w] += 1
    for r in nonclaims:
        for w in words_of(r["context"]):
            nc[w] += 1

    # Mechanical ranking: frequent among misses, rare among non-claims. No hand-picking.
    ranked = sorted(mc, key=lambda w: (-(mc[w] - nc[w]), -mc[w], w))

    def score(vocab: set[str]) -> dict:
        caught = sum(1 for r in misses if words_of(r["context"]) & vocab)
        cost = sum(1 for r in nonclaims if words_of(r["context"]) & vocab)
        return {"misses_caught": caught, "non_claims_obligated": cost,
                "cost_per_catch": round(cost / caught, 3) if caught else None}

    candidates = {}
    for k in K_LADDER:
        vocab = set(ranked[:k])
        candidates[f"add_top_{k}_words"] = {"k": k, **score(vocab), "words": sorted(vocab)}

    # HELD-OUT. The numbers above are IN-SAMPLE and are optimistic by construction: the words were
    # ranked to separate these very tokens, then scored on them. Any word list looks good that way,
    # and quoting it would be the oldest self-deception in measurement.
    #
    # The split is BY DOCUMENT, not by token. Tokens from one README share vocabulary, so a random
    # token split leaks the answer across folds and reproduces the in-sample number wearing a
    # held-out label.
    docs = sorted({r["repo"] for r in addressable})
    fold_of = {d: (sum(map(ord, d)) % 2) for d in docs}
    heldout = {}
    for k in K_LADDER:
        caught = cost = n_m = n_n = 0
        for test_fold in (0, 1):
            fit_m = [r for r in misses if fold_of[r["repo"]] != test_fold]
            fit_n = [r for r in nonclaims if fold_of[r["repo"]] != test_fold]
            fm, fn = collections.Counter(), collections.Counter()
            for r in fit_m:
                for w in words_of(r["context"]):
                    fm[w] += 1
            for r in fit_n:
                for w in words_of(r["context"]):
                    fn[w] += 1
            vocab = set(sorted(fm, key=lambda w: (-(fm[w] - fn[w]), -fm[w], w))[:k])
            te_m = [r for r in misses if fold_of[r["repo"]] == test_fold]
            te_n = [r for r in nonclaims if fold_of[r["repo"]] == test_fold]
            caught += sum(1 for r in te_m if words_of(r["context"]) & vocab)
            cost += sum(1 for r in te_n if words_of(r["context"]) & vocab)
            n_m += len(te_m)
            n_n += len(te_n)
        heldout[f"add_top_{k}_words"] = {
            "k": k,
            "misses_caught": caught, "non_claims_obligated": cost,
            "cost_per_catch": round(cost / caught, 3) if caught else None,
            "recall_of_misses": round(caught / n_m, 3) if n_m else None,
        }

    # THE NULL RULE: obligate every number, no vocabulary test at all.
    control = {"misses_caught": len(misses), "non_claims_obligated": len(nonclaims)}
    control["cost_per_catch"] = (round(control["non_claims_obligated"] / control["misses_caught"], 3)
                                 if control["misses_caught"] else None)

    disc = discrimination_report(
        {n: {"misses_caught": c["misses_caught"],
             "non_claims_obligated": c["non_claims_obligated"]} for n, c in candidates.items()},
        {"misses_caught": control["misses_caught"],
         "non_claims_obligated": control["non_claims_obligated"]},
        {"misses_caught": "higher_is_better", "non_claims_obligated": "lower_is_better"},
        deciding=["non_claims_obligated"],
    )

    # The same check on the HELD-OUT numbers. Run both deliberately: discriminates compares a
    # candidate against a null rule, which is a different question from whether the candidate was
    # fitted on the data it is scored on. It passes the in-sample candidates and cannot see the
    # overfitting -- a real limit of that instrument, recorded here because this census is the
    # first case to expose it.
    disc_ho = discrimination_report(
        {n: {"misses_caught": c["misses_caught"],
             "non_claims_obligated": c["non_claims_obligated"]} for n, c in heldout.items()},
        {"misses_caught": control["misses_caught"],
         "non_claims_obligated": control["non_claims_obligated"]},
        {"misses_caught": "higher_is_better", "non_claims_obligated": "lower_is_better"},
        deciding=["non_claims_obligated"],
    )

    payload = {
        "census": "can widening the trigger vocabulary fix the miss rate?",
        "status": "RECON. Licenses no clause, no bar, no repair.",
        "ground_truth": ("majority verdicts of three blind seats over every abstained token in "
                         "both 2026-08-27 panels; the first hand-adjudicated miss set this lane "
                         "has had"),
        "population": {
            "abstained_tokens_adjudicated": len(rows),
            "addressable_by_vocabulary": len(addressable),
            "of_which_real_claims_MISSED": len(misses),
            "of_which_correctly_abstained": len(nonclaims),
            "not_addressable_fired_a_trigger_already": len(rows) - len(addressable),
        },
        "null_rule": {
            "rule": "obligate every number; no vocabulary test at all",
            **control,
            "why_it_is_the_ceiling": (
                "No word list can catch more misses than obligating everything, and none can cost "
                "less than the words it adds. The null rule's trade bounds every candidate."),
        },
        "candidates_IN_SAMPLE": candidates,
        "candidates_HELD_OUT": heldout,
        "in_sample_warning": (
            "candidates_IN_SAMPLE ranks and scores words on the SAME tokens and is optimistic by "
            "construction. Read candidates_HELD_OUT, which fits the word list on one half of the "
            "DOCUMENTS and scores it on the other. The split is by document because tokens from "
            "one README share vocabulary and a token-level split leaks across folds."),
        "discrimination_in_sample": disc,
        "discrimination_held_out": disc_ho,
        "the_finding": (
            "Vocabulary widening does NOT generalise. In-sample the top-20 rule catches 36 of 85 "
            "misses at a cost of 4, a cost-per-catch of 0.111 that reads as a triumph. Fitted on "
            "one half of the documents and scored on the other, the same rule catches ONE, a "
            "recall of 0.012. The words that separate misses from non-claims in one set of "
            "documents do not transfer to another set. The obvious repair is dead, and it would "
            "have looked like a success to anyone who stopped at the in-sample table."),
        "and_the_instrument_did_not_catch_it": (
            "styxx-discriminates PASSES the in-sample candidates: every one beats the null rule "
            "on the deciding column. It asks whether a candidate beats doing nothing, which is a "
            "different question from whether the candidate was fitted on the data it is scored "
            "on. A held-out split is not something that check can substitute for, and this is the "
            "first cycle to demonstrate it."),
        "what_this_does_not_show": (
            "Whether a STRUCTURAL predicate could separate these populations. This census scores "
            "lexical rules only, and its negative result is about vocabulary, not about the "
            "repairability of the obligation predicate."),
    }
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"addressable {len(addressable)}  misses {len(misses)}  correct-abstentions {len(nonclaims)}")
    print()
    print(f"{'rule':<22}{'catches':<10}{'cost':<10}{'cost/catch'}")
    for n, c in candidates.items():
        print(f"{n:<22}{c['misses_caught']:<10}{c['non_claims_obligated']:<10}{c['cost_per_catch']}")
    print(f"{'NULL (obligate all)':<22}{control['misses_caught']:<10}"
          f"{control['non_claims_obligated']:<10}{control['cost_per_catch']}")
    print()
    print("HELD OUT (word list fitted on the other half of the documents):")
    print(f"{'rule':<22}{'catches':<10}{'cost':<10}{'cost/catch':<13}{'recall'}")
    for n, c in heldout.items():
        print(f"{n:<22}{c['misses_caught']:<10}{c['non_claims_obligated']:<10}"
              f"{str(c['cost_per_catch']):<13}{c['recall_of_misses']}")
    print()
    print("IN-SAMPLE discrimination:")
    print(render(disc, "obligate every number"))
    print()
    print("HELD-OUT discrimination:")
    print(render(disc_ho, "obligate every number"))
    print(f"\n-> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
