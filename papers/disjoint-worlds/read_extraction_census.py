# -*- coding: utf-8 -*-
"""EXTRACTION census for the cross-model READ battery: where did the 462 concepts
come from, and what was the pool they survived out of?

WHY THIS FILE EXISTS
--------------------
A peer session proposed a decomposition that this lab has adopted:

    P = E x A

  P  precision as published
  E  EXTRACTION — the share of cases where the thing the instrument was pointed at
     was actually the thing it claims to adjudicate
  A  ADJUDICATION — given the pointing was right, the share where the verdict was
     right

Every number this lab publishes is P and has been read as if it were A. For the
diffgate path-claim class, P was measured at 0.16 held-out (`RESULT_v14`) and
`extraction_census.py` opened the E-term. For the calibrated instruments
(HaluEval-QA 0.998, TruthfulQA 0.994) the benchmark HANDS the span to the
instrument, so E = 1 by construction and those numbers are pure A-terms.

The cross-model read has the second shape. `read_top1` is an index-matched argmin
over the held-out target centroids (`run_b31v2.py:90-93`, `run_b34v3.py:46-49`):
the truth is in the candidate array with probability 1, there is no threshold and
no reject option. So the published read figures are A-terms, and the question this
file was written to answer is what E is for them.

The proposed answer was: E is a SELECTION RATIO — the share of candidate concepts
that survived into the committed battery — computable from committed artifacts with
no new experiment. THIS FILE REPORTS THAT IT IS NOT COMPUTABLE, and reports the
whole construction chain so a reader can see why.

WHAT THIS FILE IS
-----------------
A CENSUS, in the tradition of `papers/closed-model-frontier/extraction_census.py`
and `RESULT_collateral_census_2026_08_31.md`: descriptive, report-only, no
hypothesis test, no gate, no verdict token. NO PREREGISTRATION COVERS IT, so
nothing it emits may carry a headline finding.

It loads no model, runs no experiment, and reads no GPU artifact. It parses
committed Python source with `ast`, replays two splits with numpy, and reconciles
its counts against receipts it does not write.

It writes exactly one file: `read_extraction_census.json`. It never edits a receipt.

    python papers/disjoint-worlds/read_extraction_census.py
"""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

OUT = HERE / "read_extraction_census.json"
OUT_MARKER = "EXTRACTION census of the cross-model read battery"

G0CLEAR = HERE / "run_g0clear.py"
PARENT = HERE / "run_thought_transfer.py"
B31V2 = HERE / "run_b31v2.py"
B34V3 = HERE / "run_b34v3.py"

# Receipts READ for reconciliation. None of these is written by this file.
R_G0CLEAR = HERE / "g0clear_result_llama3b.json"
R_B31V2 = HERE / "b31v2_result.json"
R_B34V3 = HERE / "b34v3_result.json"
R_B34V3_ADD = HERE / "b34v3_fresh_split_addendum.json"
R_B35C = HERE / "b35c_result.json"

# The marker lives in the KEY NAMES, not in a sibling status field. The lesson is
# `extraction_census.py`'s: a consumer who indexes a bare key never touches the
# sibling that says the number decides nothing. A reader cannot reach these
# integers without typing the warning.
UNCHECKABLE = (
    "UNCHECKABLE — the pre-filter candidate pool was never recorded as an "
    "artifact, so no survival ratio exists to compute. This is a verdict, not a "
    "failure, and it is not a claim that the ratio is low. Absence of evidence is "
    "never a contradiction.")
DEDUP_WARNING = (
    "NOT A SELECTION RATIO and NOT AN E-TERM. This is the share of the "
    "hand-typed literal that survived order-preserving deduplication. It measures "
    "that the author typed three words twice while balancing category blocks. The "
    "literal is the OUTPUT of the selection, not its input; quoting this as E "
    "would be exactly the error the P = E x A decomposition exists to name.")


# --------------------------------------------------------------------------
# MECHANICAL — pure functions of committed bytes. Every rule is stated in RULES.
# --------------------------------------------------------------------------

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def literal_split_words(path: Path, name: str):
    """The words of a module-level `NAME = <str literal>.split()` assignment.

    Parsed with `ast` from the committed source rather than by importing the
    module, because both modules import torch/transformers at module scope. The
    string is recovered by `ast.literal_eval` of the receiver of `.split()`, so
    implicit concatenation of adjacent literals is handled the way Python handles
    it, and no regex approximates the parse.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            continue
        v = node.value
        if (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute)
                and v.func.attr == "split" and not v.args):
            return (ast.literal_eval(v.func.value).split(),
                    node.lineno, node.value.end_lineno)
    raise LookupError(f"{name} not found as a `.split()` literal in {path.name}")


def dedup_order_preserving(words):
    """The comprehension at run_g0clear.py:66-67, reproduced exactly."""
    seen = set()
    return [c for c in words if not (c in seen or seen.add(c))]


def duplicates_with_positions(words):
    first, dups = {}, []
    for i, w in enumerate(words):
        if w in first:
            dups.append({"word": w, "first_index": first[w], "repeat_index": i})
        else:
            first[w] = i
    return dups


def split_concepts_replay(concepts, seed=0):
    """run_g0clear.py:99-108, reproduced. numpy only; no model, no GPU."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(concepts))
    n = len(concepts)
    n_tr, n_sel = int(0.70 * n), int(0.15 * n)
    return ([concepts[i] for i in idx[:n_tr]],
            [concepts[i] for i in idx[n_tr:n_tr + n_sel]],
            [concepts[i] for i in idx[n_tr + n_sel:]])


def b34v3_split_replay(concepts, seed=343, n_fin=70):
    """run_b34v3.py:64-72, reproduced. NOT split_concepts — its own permutation."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(concepts))
    n_tr = len(concepts) - n_fin
    return ([concepts[i] for i in idx[n_fin:n_fin + n_tr]],
            [concepts[i] for i in idx[:n_fin]])


def loop_skip_branches(path: Path, func_name: str):
    """Count `continue`/`break` statements inside a named function.

    A per-concept filter in these extractors could only be expressed as a skip
    inside the concept loop. Zero skips is therefore mechanical evidence that no
    concept is ever dropped, stronger than a keyword grep.
    """
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            ifs = [n for n in ast.walk(node) if isinstance(n, ast.If)]
            return {
                "found": True,
                "continue_statements": sum(1 for n in ast.walk(node) if isinstance(n, ast.Continue)),
                "break_statements": sum(1 for n in ast.walk(node) if isinstance(n, ast.Break)),
                "if_statements": len(ifs),
                # every branch quoted, so a reader can check that none of them is a
                # per-concept filter without opening the file
                "if_statement_lines": [
                    {"line": n.lineno, "source": lines[n.lineno - 1].strip()} for n in ifs],
            }
    return {"found": False}


FILTER_TOKENS = ("vocab", "tokeniz", "frequency", "freq_", "filter", "exclude",
                 "drop", "reject", "discard", "skip", "min_len", "stopword")


def token_scan(path: Path):
    """Which filter-shaped identifiers appear anywhere in the file's bytes.

    Reported as a DIAGNOSTIC, not as a category: a hit is not a filter and a miss
    is not proof of absence. The load-bearing evidence is loop_skip_branches.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    out = {}
    for i, raw in enumerate(lines, 1):
        low = raw.lower()
        for t in FILTER_TOKENS:
            if t in low:
                out.setdefault(t, []).append({"line": i, "source": raw.strip()[:120]})
    return out


def files_referencing(pattern):
    """Committed .py files in this directory whose source contains `pattern`.

    This census file excludes itself: it references both patterns only in order to
    reproduce them, and counting itself would inflate the reuse figure.
    """
    hits = []
    for p in sorted(HERE.glob("*.py")):
        if p.name == Path(__file__).name:
            continue
        try:
            t = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if pattern in t:
            hits.append(p.name)
    return hits


def git_provenance(path: Path):
    try:
        r = subprocess.run(
            ["git", "log", "--follow", "--format=%H|%ad|%s", "--date=iso", "--", str(path)],
            cwd=str(ROOT), capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return {"status": "UNCHECKABLE_git_returned_nonzero", "stderr": r.stderr.strip()[:300]}
        lines = [l for l in r.stdout.splitlines() if l.strip()]
        commits = []
        for l in lines:
            h, d, s = l.split("|", 2)
            commits.append({"commit": h, "date": d, "subject": s})
        return {"status": "read", "n_commits": len(commits), "commits": commits}
    except Exception as e:                                    # noqa: BLE001
        return {"status": f"UNCHECKABLE_{type(e).__name__}", "detail": str(e)[:300]}


RULES = {
    "raw_tokens":
        "the module-level `_BANK = \"\"\"...\"\"\".split()` literal in "
        "run_g0clear.py, recovered by ast.literal_eval of the receiver of "
        ".split(), then split on whitespace exactly as Python's str.split does",
    "deduplicated":
        "the order-preserving comprehension at run_g0clear.py:66-67, reproduced "
        "character-for-character in dedup_order_preserving",
    "split_seed0":
        "run_g0clear.py:99-108 replayed with numpy default_rng(0): "
        "n_tr = int(0.70*N), n_sel = int(0.15*N), fin = the remainder",
    "split_seed343":
        "run_b34v3.py:64-72 replayed with numpy default_rng(343): fin = first 70 "
        "of the permutation, tr = the next N-70. This is NOT split_concepts",
    "anchors_392":
        "run_b31v2.py:104-105 concatenates TRAIN and SEL_dirs after calling "
        "split_concepts(seed=0); the concatenation is the consumer's, not the "
        "splitter's",
    "loop_skip_branches":
        "ast walk of the named extraction function counting `continue` and "
        "`break`. A per-concept filter in these extractors could only be a skip "
        "inside the concept loop, so zero skips is evidence that no concept is "
        "ever dropped by measurement",
    "candidate_pool":
        "NO RULE EXISTS. There is no committed corpus, vocabulary slice, dataset "
        "or generator script from which the battery was drawn, so there is no "
        "denominator to compute against",
}


def main() -> int:
    # Refuse to clobber anything that is not this file's own previous emission.
    if OUT.exists():
        try:
            prev = json.loads(OUT.read_text(encoding="utf-8"))
        except Exception:                                     # noqa: BLE001
            print(f"REFUSING to overwrite unparseable {OUT.name}", file=sys.stderr)
            return 2
        if not str(prev.get("what", "")).startswith(OUT_MARKER):
            print(f"REFUSING to overwrite {OUT.name}: not this census's output",
                  file=sys.stderr)
            return 2

    # ---- the literal ----
    raw, bank_start, bank_end = literal_split_words(G0CLEAR, "_BANK")
    concepts = dedup_order_preserving(raw)
    dups = duplicates_with_positions(raw)

    # ---- the parent bank ----
    parent_raw, p_start, p_end = literal_split_words(PARENT, "CONCEPTS")
    parent = dedup_order_preserving(parent_raw)
    parent_in_bank = [w for w in parent if w in set(concepts)]
    parent_missing = [w for w in parent if w not in set(concepts)]
    net_new = [w for w in concepts if w not in set(parent)]

    # ---- the splits ----
    tr0, sel0, fin0 = split_concepts_replay(concepts, seed=0)
    tr343, fin343 = b34v3_split_replay(concepts, seed=343, n_fin=70)
    overlap = sorted(set(fin343) & set(fin0))
    anchors_392 = list(tr0) + list(sel0)

    # ---- receipts, read only ----
    r_g0 = json.loads(R_G0CLEAR.read_text(encoding="utf-8"))
    r31 = json.loads(R_B31V2.read_text(encoding="utf-8"))
    r34 = json.loads(R_B34V3.read_text(encoding="utf-8"))
    r34a = json.loads(R_B34V3_ADD.read_text(encoding="utf-8"))
    r35c = json.loads(R_B35C.read_text(encoding="utf-8"))

    recon = {
        "n_concepts_receipt": r_g0["n_concepts"],
        "n_concepts_recomputed": len(concepts),
        "n_concepts_reconciles": r_g0["n_concepts"] == len(concepts),
        "b31v2_n_heldout_receipt": r31["n_heldout"],
        "b31v2_n_heldout_recomputed": len(fin0),
        "b31v2_heldout_reconciles": r31["n_heldout"] == len(fin0),
        "b34v3_n_tr_receipt": r34["n_tr"],
        "b34v3_n_tr_recomputed": len(tr343),
        "b34v3_n_heldout_receipt": r34["n_heldout"],
        "b34v3_n_heldout_recomputed": len(fin343),
        "b34v3_split_reconciles": (r34["n_tr"] == len(tr343)
                                   and r34["n_heldout"] == len(fin343)),
        "b34v3_addendum_overlap_receipt": r34a["held_out_overlap_with_v1v2"],
        "b34v3_overlap_recomputed": len(overlap),
        "b34v3_overlap_reconciles": r34a["held_out_overlap_with_v1v2"] == len(overlap),
        "b34v3_genuinely_fresh_receipt": r34a["genuinely_fresh"],
        "b34v3_genuinely_fresh_recomputed": len(fin343) - len(overlap),
        "b35c_vocab_size_receipt": r35c["vocab_size"],
        "b35c_vocab_is_the_same_battery": r35c["vocab_size"] == len(concepts),
    }
    recon["all_reconcile"] = all(
        recon[k] for k in recon if k.endswith(("_reconciles", "_battery")))

    # ---- the filter chain ----
    chain = [
        {"stage": "hand-typed literal `_BANK`",
         "citation": f"run_g0clear.py:{bank_start}-{bank_end}",
         "in": None, "removed": 0, "out": len(raw),
         "note": "the list was born whole; there is no upstream artifact"},
        {"stage": "order-preserving deduplication",
         "citation": "run_g0clear.py:66-67",
         "in": len(raw), "removed": len(raw) - len(concepts), "out": len(concepts),
         "note": "the ONLY filter in the entire chain",
         "removed_words": [d["word"] for d in dups]},
    ]
    absent = [
        ("tokenization / single-token constraint",
         "run_g0clear.py extract_multi; run_thought_transfer.py extract"),
        ("presence-in-both-models / shared-vocabulary check",
         "no target tokenizer is consulted anywhere; concepts are embedded through "
         "the 12 CONCEPT_TEMPLATES at introspection_gate.py:26-38"),
        ("frequency threshold", "no frequency table is imported anywhere in the chain"),
        ("part-of-speech filter", "no POS tagger is imported anywhere in the chain"),
        ("representation-quality gate / norm or outlier screen",
         "no per-concept statistic is ever compared against a threshold"),
        ("downstream re-filtering by the read experiments",
         "run_b31v2.py:34,103 and run_b34v3.py:26,70 both import the list wholesale"),
    ]
    for name, cite in absent:
        chain.append({"stage": name, "citation": cite, "in": len(concepts),
                      "removed": 0, "out": len(concepts),
                      "note": "NO SUCH FILTER EXISTS"})

    skips = {
        "run_g0clear.extract_multi": loop_skip_branches(G0CLEAR, "extract_multi"),
        "run_thought_transfer.extract": loop_skip_branches(PARENT, "extract"),
    }

    # ---- reuse of the same held-out set ----
    reuse = files_referencing("split_concepts")
    bank_consumers = files_referencing("from run_g0clear import")

    # ---- line structure of the literal, with the honest caveat ----
    src_lines = G0CLEAR.read_text(encoding="utf-8").splitlines()
    lit_lines = []
    for ln in range(bank_start, bank_end + 1):
        body = src_lines[ln - 1]
        if '"""' in body:          # the opening and closing delimiter lines
            continue
        words = body.split()
        if words:
            lit_lines.append({"line": ln, "tokens": len(words),
                              "first": words[0], "last": words[-1]})

    payload = {
        "what": (OUT_MARKER + ": the construction chain of the N=462 concept "
                 "battery that b31v2, b34v3 and b35c are scored over"),
        "kind": ("CENSUS — descriptive, report-only. No hypothesis test, no gate, "
                 "no verdict token, NO PREREGISTRATION. Nothing here may carry a "
                 "headline finding."),
        "companion_paper": "CENSUS_read_extraction_2026_09_01.md",
        "loads_no_model": True,
        "writes": [OUT.name],
        "edits_no_receipt": True,
        "rules": RULES,

        "provenance": {
            "run_g0clear_py_sha256": sha256(G0CLEAR),
            "git": git_provenance(G0CLEAR),
            "recorded_pool_size": None,
            "recorded_pool_size_status": UNCHECKABLE,
            "design_targets_not_measurements": {
                "run_g0clear.py:29 docstring": "~480 single-word concepts",
                "PREREG_thought_transfer_g0clear_2026_06_20.md:25,43": "N~480",
                "note": ("both were written before the list existed; they are "
                         "design targets, not counts of anything"),
            },
        },

        "battery": {
            "raw_tokens_in_literal": len(raw),
            "raw_tokens_recorded_in_any_receipt": False,
            "deduplicated": len(concepts),
            "deduplicated_recorded_in": "g0clear_result_llama3b.json .n_concepts",
            "duplicates": dups,
            "distinct_after_dedup_equals_len_set": len(concepts) == len(set(raw)),
            "line_structure": {
                "note": ("token counts per source line of the literal. THE BYTES "
                         "CARRY NO CATEGORY LABELS: there are no comments, no "
                         "blank lines and no markers between the apparent "
                         "category blocks, so any category attribution is a "
                         "reader's inference and is NOT mechanically derivable "
                         "from this file. Reported as structure, not as taxonomy."),
                "lines": lit_lines,
            },
        },

        "parent_bank": {
            "citation": f"run_thought_transfer.py:{p_start}-{p_end}",
            "raw_tokens": len(parent_raw),
            "deduplicated": len(parent),
            "recorded_as": {
                "run_g0clear.py:5 docstring": 110,
                "g0clear_result_llama3b.json .parent_baseline.N": r_g0["parent_baseline"]["N"],
            },
            "recorded_figure_matches_committed_literal":
                r_g0["parent_baseline"]["N"] == len(parent),
            "drift_unrecorded": len(parent) - r_g0["parent_baseline"]["N"],
            "parent_words_present_in_battery": len(parent_in_bank),
            "parent_words_absent_from_battery": parent_missing,
            "net_new_words_in_battery": len(net_new),
            "how_the_net_new_were_chosen": UNCHECKABLE,
        },

        "chain": chain,
        "loop_skip_branches": skips,
        "filter_token_scan_DIAGNOSTIC_not_a_category": {
            "note": ("a hit is not a filter and a miss is not proof of absence; "
                     "the load-bearing evidence is loop_skip_branches"),
            "run_g0clear.py": token_scan(G0CLEAR),
            "run_b31v2.py": token_scan(B31V2),
            "run_b34v3.py": token_scan(B34V3),
            "run_thought_transfer.py": token_scan(PARENT),
        },

        "splits": {
            "split_concepts_seed0": {
                "citation": "run_g0clear.py:99-108",
                "train": len(tr0), "sel_dirs": len(sel0), "fin_dirs": len(fin0),
            },
            "b31v2_anchors": {
                "citation": "run_b31v2.py:104-105",
                "anchors": len(anchors_392), "held_out": len(fin0),
                "note": "train+sel concatenated by the CONSUMER, not the splitter",
            },
            "b34v3_seed343": {
                "citation": "run_b34v3.py:32,64-72",
                "train": len(tr343), "held_out": len(fin343),
                "note": ("b34v3 does NOT use split_concepts(seed=0). Same battery, "
                         "own permutation, different membership."),
            },
            "held_out_overlap_seed343_vs_seed0": {
                "n": len(overlap),
                "words": overlap,
                "genuinely_fresh": len(fin343) - len(overlap),
                "preregistered_as": "disjoint",
                "falsified_by": "b34v3_fresh_split_addendum.json",
            },
            "scripts_referencing_split_concepts": reuse,
            "scripts_importing_from_run_g0clear": bank_consumers,
            "held_out_ness_note": (
                "the seed-0 FIN-70 is never in a fit within any single run, but it "
                "has been SCORED by every script listed above. Its held-out-ness "
                "has been spent by repetition, which no single receipt records."),
        },

        # The A-terms this census exists to scope. Read only; not recomputed here.
        "published_read_figures_are_A_terms": {
            "why": ("read_top1 is an index-matched argmin over the held-out target "
                    "centroids (run_b31v2.py:90-93, run_b34v3.py:46-49). The truth "
                    "is in the candidate array with probability 1; there is no "
                    "threshold and no reject option. E = 1 by construction on every "
                    "trial these scripts have ever run."),
            "b31v2_gemma_M1_mlp_top1": r31["targets"]["gemma_2b"]["M1_mlp_top1"],
            "b31v2_gemma_requires": "392 TRUE cross-model concept pairs (supervised)",
            "b34v3_gemma_read_top1_full70": r34["targets"]["gemma_2b"]["read_top1"],
            "b34v3_gemma_read_top1_fresh57": r34a["gemma_read_fresh_only"],
            "b34v3_llama_read_top1_full70": r34["targets"]["llama_1b"]["read_top1"],
            "b34v3_llama_read_top1_fresh57": r34a["llama_read_fresh_only"],
            "quote_both_or_neither": (
                "the fresh-57 recompute is the honest figure for b34v3; the full-70 "
                "headline includes 13 concepts scored before"),
        },

        # The closed-set penalty, which is a separate term and is not E.
        "closed_set_penalty_SEPARATE_TERM_not_E": {
            "receipt": "b35c_result.json",
            "verdict_in_receipt": r35c["verdict"],
            "status": ("UNLICENSED — b35c returned INVALID__null_artifact on a "
                       "null-model error (G2_null false). These retention figures "
                       "are reported as observations and decide nothing."),
            "vocab_size": r35c["vocab_size"],
            "gemma_read70": r34["targets"]["gemma_2b"]["read_top1"],
            "gemma_read462": r35c["targets"]["gemma_2b"]["read462"],
            "llama_read70": r34["targets"]["llama_1b"]["read_top1"],
            "llama_read462": r35c["targets"]["llama_1b"]["read462"],
            "gemma_retention_70_to_462": round(
                r35c["targets"]["gemma_2b"]["read462"]
                / r34["targets"]["gemma_2b"]["read_top1"], 4),
            "llama_retention_70_to_462": round(
                r35c["targets"]["llama_1b"]["read462"]
                / r34["targets"]["llama_1b"]["read_top1"], 4),
            "retention_is_derived_from_two_receipts": [
                "b34v3_result.json .targets.*.read_top1",
                "b35c_result.json .targets.*.read462"],
            "note": ("widening an ALREADY-CLOSED candidate set 70 -> 462 cost most "
                     "of the read. 462 is still the same hand-typed list, so this "
                     "bounds nothing about an open vocabulary."),
        },

        "dedup_survival_ratio_NOT_A_SELECTION_RATIO": {
            "warning": DEDUP_WARNING,
            "value": round(len(concepts) / len(raw), 6),
            "numerator": len(concepts),
            "denominator": len(raw),
            "what_it_measures": "deduplication of a hand-typed literal, nothing else",
        },

        "E_term_for_the_read_UNCHECKABLE": {
            "status": UNCHECKABLE,
            "question_it_would_answer": (
                "was the activation the reader was handed in fact a "
                "single-concept-dominated state, isolable at the committed "
                "extraction site — last token, layer 11 locked by the g0clear "
                "sweep, mean-pooled over the 12 CONCEPT_TEMPLATES, differenced "
                "against the fixed 'object' baseline — at the moment it was read?"),
            "why_it_cannot_be_measured_from_committed_artifacts": [
                "the candidate pool was never written down; the selection happened "
                "at authoring time and left no rejection log",
                "no generator script and no source corpus is cited anywhere in the "
                "repository",
                "the apparatus has never once been pointed at a state it did not "
                "itself manufacture from 12 experimenter-written sentences, so no "
                "committed run can answer the question even in principle",
            ],
            "what_would_have_to_be_re_run": [
                "commit a candidate pool as a mechanically-derived artifact with a "
                "stated rule and a hashed file, not a docstring summary",
                "express each intended filter as a committed function and record "
                "the survivor count after each stage in the result JSON",
                "sample the battery from the survivors under a committed seed",
                "re-extract on GPU across all four models — the _b31v2_pts_*.npz "
                "caches are keyed positionally to the current 462-word list "
                "(run_b34v3.py:37-38) and cannot be reused for a different battery",
                "re-run the b31v2 M0/M1/N1 cells and the b34v3 label-free read",
            ],
            "what_that_would_still_not_buy": (
                "a perfectly recorded word-list ratio is an E-term for single-word "
                "English noun concepts, not for the states a model can hold. That "
                "second gap survives the repair and no construction over a word "
                "list closes it."),
        },

        "reconciliation": recon,
    }

    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    # ---------------- table ----------------
    print(f"\nREAD EXTRACTION CENSUS - {G0CLEAR.name} _BANK")
    print(f"  sha256 {payload['provenance']['run_g0clear_py_sha256'][:16]}...")
    g = payload["provenance"]["git"]
    if g.get("status") == "read":
        print(f"  git: {g['n_commits']} commit(s) --follow; the list was born whole")
    print("\nCONSTRUCTION CHAIN")
    for st in chain:
        print(f"  {st['stage']:<52}{str(st['in'] or '-'):>6} -{st['removed']:<4}-> {st['out']:>4}"
              f"   {st['citation']}")
    print("\nSPLITS")
    print(f"  split_concepts(seed=0)   TRAIN {len(tr0)} / SEL {len(sel0)} / FIN {len(fin0)}")
    print(f"  b31v2 anchors            {len(anchors_392)} anchors, {len(fin0)} held-out")
    print(f"  b34v3 seed 343           {len(tr343)} train, {len(fin343)} held-out")
    print(f"  held-out overlap         {len(overlap)} of {len(fin343)}  "
          f"(genuinely fresh {len(fin343) - len(overlap)})")
    print("\nRECONCILIATION")
    for k, v in recon.items():
        if k.endswith(("_reconciles", "_battery", "all_reconcile")):
            print(f"  {k:<40}{'OK' if v else 'MISMATCH - investigate'}")
    print("\nDEDUP SURVIVAL RATIO (NOT a selection ratio, NOT an E-term)")
    print(f"  {len(concepts)}/{len(raw)} = "
          f"{payload['dedup_survival_ratio_NOT_A_SELECTION_RATIO']['value']}")
    print("\nE-TERM FOR THE READ")
    print("  UNCHECKABLE - no candidate pool was ever recorded. Not estimated here.")
    print(f"\n-> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
