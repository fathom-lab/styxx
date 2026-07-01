"""Dataset loaders + mechanical grading/normalization for the keystone depth-vs-truth harness.

Implements PREREG.md (papers/depth-truth/PREREG.md):
  - Section 3 (Datasets): exact HF sources, field VERIFICATION at load, seeded shuffle/sample,
    tercile selection for PopQA-rare, tri-state carry for TruthfulQA.
  - Section 3 + Section 5 (grading/normalization): official-style normalization
    (lowercase, strip articles a/an/the, strip punctuation, collapse whitespace) and
    normalized-exact-match grading against an item's full alias list. Mechanical, no model judge.

Contract note: loaders return item dicts consumed by the generation/scoring modules; this file
performs NO model loading, NO GPU work, and NO network access at import time. `datasets.load_dataset`
is imported lazily INSIDE each loader so `import datasets` (this module) works fully offline.

Seeds: dataset-shuffle seed is passed explicitly by the caller (PREREG pins it to 7).
"""

from __future__ import annotations

import json
import re
import string
from typing import Any

import numpy as np

# --- normalization / grading (PREREG Section 3) --------------------------------------------------

# Whole-word articles stripped after punctuation removal (SQuAD/TriviaQA official style).
_ARTICLES = ("a", "an", "the")
_ARTICLE_RE = re.compile(r"\b(?:%s)\b" % "|".join(_ARTICLES))
_WS_RE = re.compile(r"\s+")
# Translation table that maps every ASCII punctuation char to a space (so word boundaries survive).
_PUNCT_TABLE = {ord(c): " " for c in string.punctuation}


def normalize(s: str) -> str:
    """Official-style normalization (PREREG Section 3).

    Order: lowercase -> strip punctuation -> strip standalone articles (a/an/the) -> collapse
    whitespace. Punctuation is removed BEFORE articles so tokens like ``"the,"`` still match, and
    articles are stripped as whole words only (``"theatre"`` is untouched). Returns a single-spaced,
    edge-trimmed string.
    """
    if s is None:
        return ""
    s = str(s).lower()
    s = s.translate(_PUNCT_TABLE)          # punctuation -> spaces
    s = _ARTICLE_RE.sub(" ", s)            # drop whole-word articles
    s = _WS_RE.sub(" ", s).strip()         # collapse + trim
    return s


def grade(answer: str, gold: list[str]) -> bool:
    """True iff the normalized answer exactly equals the normalized form of ANY gold alias.

    Mechanical normalized-exact-match (PREREG Section 3) — not substring, no model judge. An empty
    normalized answer never matches (even against an empty gold alias); the empty-answer case is a
    ``nonanswer`` exclusion handled upstream per PREREG Section 5, not a grade.
    """
    norm_ans = normalize(answer)
    if not norm_ans:
        return False
    for g in gold or []:
        if norm_ans == normalize(g):
            return True
    return False


# --- shared loader helpers -----------------------------------------------------------------------

def _require_fields(feature_names, required, source: str) -> None:
    """Raise RuntimeError('field mismatch: ...') if any required top-level field is absent.

    PREREG Section 3: loaders VERIFY names/fields at load; a mismatch stops the run and reports —
    we never silently guess an alternate field name.
    """
    have = set(feature_names)
    missing = [f for f in required if f not in have]
    if missing:
        raise RuntimeError(
            "field mismatch: %s missing field(s) %s; present=%s"
            % (source, missing, sorted(have))
        )


def _coerce_str_list(value: Any) -> list[str]:
    """Normalize a gold/alias container into a flat list[str].

    Handles list/tuple, plain strings, and JSON-encoded string lists (PopQA stores
    ``possible_answers`` as a JSON string like ``'["Paris", "City of Light"]'``).
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except (ValueError, TypeError):
                pass
        return [value]
    return [str(value)]


# --- ID: TriviaQA (PREREG Section 3) -------------------------------------------------------------

def load_id_triviaqa(n: int, seed: int, skip: int = 0) -> list[dict]:
    """Load the in-distribution TriviaQA items (PREREG Section 3).

    HF ``trivia_qa`` config ``rc.nocontext``, validation split; seeded shuffle; then items
    ``[skip:skip+n]``. ``skip`` lets the pilot (§6) draw 20 items OUTSIDE the n_ID window
    (call with ``skip=n_ID``). Fields VERIFIED: ``question`` and ``answer`` (with
    ``answer.aliases`` / ``answer.value``) — mismatch raises ``RuntimeError('field mismatch: ...')``.

    Each returned dict: ``{id, question, gold: list[str]}``. ``gold`` = the item's full alias list
    (``answer.aliases`` plus ``answer.value`` and ``answer.normalized_aliases`` when present),
    de-duplicated while preserving order.
    """
    from datasets import load_dataset  # lazy: keeps import-time offline & GPU-free

    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    _require_fields(ds.features.keys(), ["question", "answer"], "trivia_qa[rc.nocontext]")

    # Verify the nested answer sub-fields exist (do not guess if renamed).
    answer_feature = ds.features["answer"]
    answer_subfields = getattr(answer_feature, "keys", lambda: [])()
    if "aliases" not in answer_subfields or "value" not in answer_subfields:
        raise RuntimeError(
            "field mismatch: trivia_qa answer struct missing aliases/value; present=%s"
            % sorted(answer_subfields)
        )

    ds = ds.shuffle(seed=seed)
    lo, hi = skip, skip + n
    sel = ds.select(range(lo, min(hi, len(ds))))

    items: list[dict] = []
    for i, row in enumerate(sel):
        ans = row["answer"]
        gold: list[str] = []
        for key in ("value", "aliases", "normalized_aliases"):
            gold.extend(_coerce_str_list(ans.get(key)) if isinstance(ans, dict) else [])
        # De-dup preserving order.
        seen, gold_uniq = set(), []
        for g in gold:
            if g not in seen:
                seen.add(g)
                gold_uniq.append(g)
        qid = row.get("question_id") if isinstance(row, dict) else None
        items.append({
            "id": str(qid) if qid else "triviaqa-%d-%d" % (seed, lo + i),
            "question": row["question"],
            "gold": gold_uniq,
        })
    return items


# --- OOD-1: PopQA-rare (PREREG Section 3) --------------------------------------------------------

def load_ood1_popqa_rare(n: int, seed: int) -> list[dict]:
    """Load the rare-entity out-of-distribution PopQA items (PREREG Section 3).

    HF ``akariasai/PopQA``. VERIFIES the popularity field name is ``s_pop`` (raises
    ``RuntimeError('field mismatch: ...')`` if absent — no guessing an alternate). Ranks all items
    by ``s_pop`` ascending, takes the bottom tercile (rarest third), then draws a seeded sample of
    the first ``n`` from that tercile. ``gold`` comes from ``possible_answers``.

    Each returned dict: ``{id, question, gold: list[str], s_pop: float}``.
    """
    from datasets import load_dataset  # lazy

    ds = load_dataset("akariasai/PopQA")
    # PopQA ships a single split; take it whichever way it is keyed.
    if hasattr(ds, "features"):
        split = ds
    else:
        split_name = "test" if "test" in ds else next(iter(ds.keys()))
        split = ds[split_name]

    _require_fields(
        split.features.keys(),
        ["question", "possible_answers", "s_pop"],
        "akariasai/PopQA",
    )

    s_pop = np.asarray([float(v) for v in split["s_pop"]], dtype=float)
    order = np.argsort(s_pop, kind="stable")          # ascending: rarest first
    tercile_size = max(1, len(order) // 3)
    bottom_idx = order[:tercile_size]                 # rarest third

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(bottom_idx))           # seeded sample within the tercile
    chosen = bottom_idx[perm][:n]

    questions = split["question"]
    answers = split["possible_answers"]
    ids = split["id"] if "id" in split.features else None

    items: list[dict] = []
    for rank, idx in enumerate(chosen):
        idx = int(idx)
        gold = _coerce_str_list(answers[idx])
        items.append({
            "id": str(ids[idx]) if ids is not None else "popqa-%d-%d" % (seed, idx),
            "question": questions[idx],
            "gold": gold,
            "s_pop": float(s_pop[idx]),
        })
    return items


# --- OOD-2: TruthfulQA-gen (PREREG Section 3, SECONDARY / disclosed-noisy) ------------------------

def load_ood2_truthfulqa(n: int = 250) -> list[dict]:
    """Load the TruthfulQA-generation items for tri-state grading (PREREG Section 3).

    HF ``truthful_qa`` config ``generation`` split; first ``n`` (n_OOD2 = 250 fixed, no shuffle —
    the deterministic first-250 window). VERIFIES ``question``, ``correct_answers``,
    ``incorrect_answers`` — mismatch raises ``RuntimeError('field mismatch: ...')``.

    Carries BOTH answer lists so the caller can apply §3 tri-state grading (match correct ⇒ correct;
    match incorrect ⇒ incorrect; neither ⇒ excluded_flag=grade_ambiguous). Each returned dict:
    ``{id, question, correct_answers: list[str], incorrect_answers: list[str]}``.
    """
    from datasets import load_dataset  # lazy

    ds = load_dataset("truthful_qa", "generation", split="validation")
    _require_fields(
        ds.features.keys(),
        ["question", "correct_answers", "incorrect_answers"],
        "truthful_qa[generation]",
    )

    sel = ds.select(range(min(n, len(ds))))
    items: list[dict] = []
    for i, row in enumerate(sel):
        items.append({
            "id": "truthfulqa-%d" % i,
            "question": row["question"],
            "correct_answers": _coerce_str_list(row["correct_answers"]),
            "incorrect_answers": _coerce_str_list(row["incorrect_answers"]),
        })
    return items
