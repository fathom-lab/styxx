"""Can a STRUCTURAL predicate separate missed claims from correctly-abstained non-claims?

Follows `RECON_obligation_repair_is_not_lexical_2026_08_27.md`, which established that widening
the trigger vocabulary does not generalise: fitted on one half of the documents and scored on the
other, the best word list catches one missed claim in eighty-five. Its conclusion was that a
trigger word list IS a marker standing in for "this sentence asserts a measurement", and widening
a marker cannot repair a predicate whose defect is that it is a marker.

This census scores the alternative. It freezes nothing and proposes no clause.

## The candidates, and why they were chosen before the data was consulted

Each rule below was written from a general argument about how measurements differ TYPOGRAPHICALLY
from configuration, not by inspecting which tokens the panel called claims. That ordering is the
only thing that makes the held-out numbers worth reading, and it is stated so a reader can check
it against the commit history.

The argument in each case:

* **precision** — a measured quantity is reported at the precision it was measured to (`0.4267`);
  a configuration value is chosen by a human and tends to be round (`256`, `20`, `1.0`).
* **emphasis** — authors bold the number that carries their result and not the seed they used.
* **table position** — a results table's first cell names the condition; the cells after it carry
  the measurements.
* **code span** — a number inside backticks is usually being shown as literal text: a flag, a
  path, a command.
* **reporting verb / comparative** — these two are LEXICAL and are included deliberately as a
  control on the census's own thesis. They are small fixed English lists written a priori rather
  than fitted, so if the structural rules generalise and these do not, the distinction is doing
  work; if all of them fail together, the failure is about held-out generalisation and not about
  structure versus vocabulary.

## Method, with yesterday's lesson built in rather than bolted on

Ground truth is the majority verdict of three blind seats over every abstained token in both
2026-08-27 panels. Every rule is scored IN SAMPLE and HELD OUT, split **by document**, because
tokens from one document share structure as well as vocabulary. Every rule is scored against the
null rule — obligate every number — and `share_of_control` is reported per candidate, because a
rule that does nothing also has a very low cost.

  python papers/closed-model-frontier/oath_structural_obligation_census.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx.discriminates import discrimination_report, render  # noqa: E402

OUT = HERE / "oath_structural_obligation_census.json"

REPORTING_VERBS = frozenset("""achieves achieved achieve reaches reached reach scores scored
yields yielded gives gave obtains obtained attains attained records recorded measured
improves improved outperforms outperformed""".split())
COMPARATIVES = frozenset("""better worse higher lower faster slower stronger weaker versus
than baseline improvement gain drop increase decrease""".split())


def _find(ctx: str, tok: str) -> int:
    """Where the token sits in its (stripped, capped) context line, or -1."""
    for m in re.finditer(re.escape(tok), ctx):
        a, b = m.start(), m.end()
        if (a == 0 or not ctx[a - 1].isdigit()) and (b == len(ctx) or not ctx[b].isdigit()):
            return a
    return -1


def _code_spans(ctx: str) -> list[tuple[int, int]]:
    spans, open_at = [], None
    for m in re.finditer("`", ctx):
        if open_at is None:
            open_at = m.end()
        else:
            spans.append((open_at, m.start()))
            open_at = None
    return spans


# --- the frozen candidate set -------------------------------------------------------------------

def r_precision(ctx: str, tok: str) -> bool:
    """>= 2 decimal places: reported at a precision a human did not choose."""
    return "." in tok and len(tok.split(".")[-1]) >= 2


def r_emphasis(ctx: str, tok: str) -> bool:
    """Wrapped in markdown emphasis: the author pointed at it."""
    i = _find(ctx, tok)
    if i < 0:
        return False
    before, after = ctx[:i], ctx[i + len(tok):]
    return bool(re.search(r"\*\*?\s*$", before) and re.match(r"\s*\*\*?", after))


def r_table_not_first_cell(ctx: str, tok: str) -> bool:
    """In a pipe table, past the first cell: the condition is named, this is a measurement."""
    if not ctx.lstrip().startswith("|"):
        return False
    i = _find(ctx, tok)
    return i > 0 and ctx.count("|", 0, i) >= 2


def r_outside_code(ctx: str, tok: str) -> bool:
    """Not inside backticks: not being shown as literal text."""
    i = _find(ctx, tok)
    if i < 0:
        return False
    return not any(a <= i < b for a, b in _code_spans(ctx))


def r_reporting_verb(ctx: str, tok: str) -> bool:
    return bool(REPORTING_VERBS & {w.lower() for w in re.findall(r"[A-Za-z]+", ctx)})


def r_comparative(ctx: str, tok: str) -> bool:
    return bool(COMPARATIVES & {w.lower() for w in re.findall(r"[A-Za-z]+", ctx)})


def r_precision_outside_code(ctx: str, tok: str) -> bool:
    """The two strongest structural signals conjoined."""
    return r_precision(ctx, tok) and r_outside_code(ctx, tok)


def r_precision_or_emphasis(ctx: str, tok: str) -> bool:
    return r_precision(ctx, tok) or r_emphasis(ctx, tok)


RULES = {
    "STRUCT_precision_2dp": r_precision,
    "STRUCT_markdown_emphasis": r_emphasis,
    "STRUCT_table_not_first_cell": r_table_not_first_cell,
    "STRUCT_outside_code_span": r_outside_code,
    "STRUCT_precision_and_outside_code": r_precision_outside_code,
    "STRUCT_precision_or_emphasis": r_precision_or_emphasis,
    "LEXICAL_reporting_verb": r_reporting_verb,
    "LEXICAL_comparative": r_comparative,
}


def adjudicated_abstentions() -> list[dict]:
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


def score(rule, claims, nonclaims) -> dict:
    caught = sum(1 for r in claims if rule(r["context"], r["token"]))
    cost = sum(1 for r in nonclaims if rule(r["context"], r["token"]))
    return {"misses_caught": caught, "non_claims_obligated": cost,
            "recall": round(caught / len(claims), 4) if claims else None,
            "precision": round(caught / (caught + cost), 4) if (caught + cost) else None}


def main() -> int:
    rows = adjudicated_abstentions()
    addressable = [r for r in rows if not r["obligating_words"]]
    claims = [r for r in addressable if r["verdict"] == "CLAIM"]
    nonclaims = [r for r in addressable if r["verdict"] == "NOT_A_CLAIM"]

    in_sample = {n: score(f, claims, nonclaims) for n, f in RULES.items()}

    # Held out by DOCUMENT. These rules carry no fitted parameters, so the split cannot change
    # what they do -- it changes which tokens they are scored on. Both halves are reported so a
    # reader can see the rules are stable across documents rather than take it on assertion.
    docs = sorted({r["repo"] for r in addressable})
    fold_of = {d: (sum(map(ord, d)) % 2) for d in docs}
    by_fold = {}
    for f in (0, 1):
        c = [r for r in claims if fold_of[r["repo"]] == f]
        n = [r for r in nonclaims if fold_of[r["repo"]] == f]
        by_fold[f"fold_{f}"] = {"n_claims": len(c), "n_nonclaims": len(n),
                                "rules": {nm: score(fn, c, n) for nm, fn in RULES.items()}}

    control = {"misses_caught": len(claims), "non_claims_obligated": len(nonclaims),
               "recall": 1.0,
               "precision": round(len(claims) / len(addressable), 4) if addressable else None}

    disc = discrimination_report(
        {n: {"misses_caught": c["misses_caught"],
             "non_claims_obligated": c["non_claims_obligated"]} for n, c in in_sample.items()},
        {"misses_caught": control["misses_caught"],
         "non_claims_obligated": control["non_claims_obligated"]},
        {"misses_caught": "higher_is_better", "non_claims_obligated": "lower_is_better"},
        deciding=["non_claims_obligated"],
    )

    # SCALE. Precision on a 212-token adjudicated sample says nothing about how many obligations
    # a rule would actually create. A rule firing on every table cell in every document could add
    # thousands, most of them never adjudicated. This projects each rule over EVERY currently
    # abstained token in both corpora, and multiplies by the measured precision to estimate how
    # many would be claims. The estimate inherits all the sample's uncertainty and is labelled.
    all_abstained = []
    for led_name in ("oath_external_corpus_ledger.jsonl", "oath_internal_ledger.jsonl"):
        for ln in (HERE / led_name).read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r["status"] == "ABSTAIN" and not r["obligating_words"]:
                all_abstained.append(r)
    scale = {}
    for nm, fn in RULES.items():
        fires = sum(1 for r in all_abstained if fn(r["context"], r["token"]))
        prec = in_sample[nm]["precision"]
        scale[nm] = {
            "fires_on_abstained_corpus_wide": fires,
            "share_of_abstained_corpus": round(fires / len(all_abstained), 4)
            if all_abstained else None,
            "projected_claims_recovered": round(fires * prec) if prec is not None else None,
            "projected_non_claims_obligated": round(fires * (1 - prec)) if prec is not None
            else None,
        }

    payload = {
        "census": "can a STRUCTURAL predicate separate missed claims from non-claims?",
        "status": "RECON. Licenses no clause, no bar, no repair.",
        "follows": "RECON_obligation_repair_is_not_lexical_2026_08_27.md",
        "candidates_were_written_before_the_data_was_consulted": True,
        "population": {"adjudicated_abstentions": len(rows), "addressable": len(addressable),
                       "claims_MISSED": len(claims), "correctly_abstained": len(nonclaims)},
        "null_rule": {"rule": "obligate every number", **control},
        "in_sample": in_sample,
        "by_fold": by_fold,
        "discrimination": disc,
        "scale_projection": {
            "abstained_tokens_corpus_wide": len(all_abstained),
            "per_rule": scale,
            "caveat": ("Projected counts multiply a corpus-wide fire count by a precision measured "
                       "on 212 adjudicated tokens. They are an order-of-magnitude estimate of what "
                       "a rule would DO, not a measurement, and the precision's fold-to-fold swing "
                       "is the honest error bar on them."),
        },
        "how_to_read_this": (
            "These rules carry no fitted parameters, so unlike a word list they cannot overfit a "
            "training half. The two folds are reported to show stability across documents, not as "
            "a train/test split. What would still sink a rule is a recall or precision that "
            "swings between folds, or a precision at or below the null rule's, which is the base "
            "rate of claims among abstained tokens."),
        "what_this_does_not_show": (
            "Whether any of these rules is implementable inside certify.py without destroying "
            "existing verifications. This census scores them only against adjudicated abstentions "
            "-- tokens the verifier already declined. It says nothing about what they would do to "
            "the tokens it currently obligates, which is where every previous widening died."),
    }
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"claims MISSED {len(claims)}   correctly abstained {len(nonclaims)}   "
          f"base rate {control['precision']}")
    print()
    print(f"{'rule':<38}{'catch':<7}{'cost':<7}{'recall':<9}{'prec':<8}"
          f"{'prec f0':<9}{'prec f1'}")
    for n, c in in_sample.items():
        f0 = by_fold["fold_0"]["rules"][n]["precision"]
        f1 = by_fold["fold_1"]["rules"][n]["precision"]
        print(f"{n:<38}{c['misses_caught']:<7}{c['non_claims_obligated']:<7}"
              f"{str(c['recall']):<9}{str(c['precision']):<8}{str(f0):<9}{str(f1)}")
    print(f"{'NULL obligate everything':<38}{control['misses_caught']:<7}"
          f"{control['non_claims_obligated']:<7}{control['recall']:<9}{control['precision']}")
    print()
    print()
    print(f"SCALE over {len(all_abstained)} abstained tokens corpus-wide:")
    print(f"{'rule':<38}{'fires':<8}{'share':<9}{'~claims':<10}{'~non-claims'}")
    for nm, sc in scale.items():
        print(f"{nm:<38}{sc['fires_on_abstained_corpus_wide']:<8}"
              f"{str(sc['share_of_abstained_corpus']):<9}"
              f"{str(sc['projected_claims_recovered']):<10}"
              f"{sc['projected_non_claims_obligated']}")
    print()
    print(render(disc, "obligate every number"))
    print(f"\n-> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
