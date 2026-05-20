# -*- coding: utf-8 -*-
"""
Bet 0 Phase 1 — executable scaffold.

Runs the H1 hypothesis test once operator decisions in
`papers/grounded-arc/preregistration_2026_05_19.md` §4 are filled in
and committed via amendment. Until then, this script is a SHAPE — it
prints what would happen, what's missing, and aborts safely without
touching any holdout.

The script is designed so the agent picking it up next session can:

1. Run it once to see what's missing (it prints a structured plan).
2. Fill in the locked-pending §4 fields via amendment commit.
3. Re-run it; the script then proceeds through corpus construction,
   embedding, validity computation, instrument scoring, ρ analysis,
   and prints the pre-registration result.

No holdout data is touched until step 3, and the pre-registration's
abandon condition is enforced mechanically — if Spearman ρ_refusal
< 0.40, the script exits with a non-zero code and writes
``papers/grounded-arc/H1_FAILED.md`` for the closed-negative paper.

This is the executable record of the discipline. The bar lives in code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# Windows console: reconfigure stdio to utf-8 so the box-drawing chars in
# the report render without raising UnicodeEncodeError on cp1252. Matches
# styxx/__init__.py's _auto_reconfigure_stdio pattern.
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    _reconf = getattr(_stream, "reconfigure", None) if _stream else None
    if _reconf is not None:
        try:
            _enc = (getattr(_stream, "encoding", "") or "").lower()
            if _enc and "utf" not in _enc:
                _reconf(encoding="utf-8", errors="replace")
        except Exception:
            pass


# ════════════════════════════════════════════════════════════════════
# Pre-registration LOCKED constants
# ════════════════════════════════════════════════════════════════════
# These must match the values in
# papers/grounded-arc/preregistration_2026_05_19.md. Any deviation
# means the script is NOT honoring the pre-registration and must
# refuse to run. The check is performed at startup.
PREREG_PATH = Path("papers/grounded-arc/preregistration_2026_05_19.md")
H1_BAR_RHO = 0.40                          # do NOT lower
H1_BAR_P = 0.01                            # permutation null
H1_MIN_N = 400                             # per-instrument holdout size
H1_N_BINS = 4                              # overlap-bin stratification
H1_PERMUTATIONS = 10_000                   # permutation null sample size
INSTRUMENTS = [
    "refusal", "sycophancy", "overconfidence",
    "hallucination", "drift",
]
HEADLINE_INSTRUMENT = "refusal"            # H1 abandons on this one
RNG_SEED = 20260519


# ════════════════════════════════════════════════════════════════════
# Operator-decision schema (locked at amendment commit)
# ════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class OperatorDecisions:
    """The three decisions from preregistration §4 that must be locked
    before Phase 1 begins. Read from a JSON file the operator commits
    via amendment. If the file doesn't exist or fields are missing,
    Phase 1 cannot proceed."""
    embedding_model: str
    h1_abandon_rho: float
    ship_target: str
    amendment_commit_hash: Optional[str] = None
    locked_at: Optional[str] = None

    @classmethod
    def load(cls, path: Path) -> "OperatorDecisions":
        if not path.exists():
            raise FileNotFoundError(
                f"operator decisions not yet committed at {path}. "
                "fill in papers/grounded-arc/preregistration_2026_05_19.md §4 "
                "and commit the amendment to lock the pre-registration."
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        required = {"embedding_model", "h1_abandon_rho", "ship_target"}
        missing = required - set(data.keys())
        if missing:
            raise ValueError(
                f"operator decisions JSON is incomplete; missing: "
                f"{sorted(missing)}"
            )
        if float(data["h1_abandon_rho"]) < H1_BAR_RHO:
            raise ValueError(
                f"operator-set h1_abandon_rho={data['h1_abandon_rho']} "
                f"is BELOW the pre-registered floor {H1_BAR_RHO}. "
                "operators may RAISE the bar; lowering it would "
                "invalidate the pre-registration."
            )
        return cls(**data)


# ════════════════════════════════════════════════════════════════════
# Holdout corpora identification (locked at construction)
# ════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class HoldoutCorpus:
    instrument: str
    source: str
    n_prompts: int
    sha256: str
    bins: List[str]  # overlap-bin labels

    def verify(self, prompts: List[str]) -> bool:
        """Verify that the loaded prompts match the locked hash."""
        h = hashlib.sha256(
            "\n".join(sorted(prompts)).encode("utf-8")
        ).hexdigest()
        return h == self.sha256


def load_holdout_corpora(
    path: Path,
) -> Dict[str, HoldoutCorpus]:
    """Read the locked holdout-corpora identification from a JSON file.
    File format::

        {
          "refusal":     {"source": "...", "n_prompts": 400,
                          "sha256": "...", "bins": ["low","med","high","vhigh"]},
          "sycophancy":  {...},
          ...
        }
    """
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        inst: HoldoutCorpus(instrument=inst, **fields)
        for inst, fields in data.items()
    }


# ════════════════════════════════════════════════════════════════════
# Validity function (LOCKED at threshold-law paper values)
# ════════════════════════════════════════════════════════════════════
def threshold_law_validity(
    *,
    distance: float,
    tau: float,
    alpha: float,
) -> float:
    """Validity in [0, 1] given prompt-to-calibration distance and the
    locked threshold-law curve parameters from
    Zenodo `10.5281/zenodo.20278945`. Sigmoid form locked per §6 of
    the pre-registration."""
    return 1.0 / (1.0 + math.exp(-alpha * (tau - distance)))


# ════════════════════════════════════════════════════════════════════
# Per-instrument scoring (loaded lazy — heavy deps)
# ════════════════════════════════════════════════════════════════════
def score_instrument(
    instrument: str,
    prompt: str,
    response: str,
    correct_reference: Optional[str] = None,
) -> float:
    """Returns the cognometric score for one (prompt, response) pair
    on the given instrument. Routes through styxx.preflight() / styxx
    .guardrail for the instrument-specific scorer."""
    from styxx.preflight import preflight
    result = preflight(prompt, response,
                       correct_reference=correct_reference,
                       persist=False)
    return float(result.scores.get(instrument, 0.0))


# ════════════════════════════════════════════════════════════════════
# Spearman ρ + permutation null
# ════════════════════════════════════════════════════════════════════
def spearman_rho(a: List[float], b: List[float]) -> float:
    """Spearman rank correlation. Pure-python; no scipy required."""
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    ra = _rank(a)
    rb = _rank(b)
    n = len(a)
    mean_a = sum(ra) / n
    mean_b = sum(rb) / n
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb))
    den_a = math.sqrt(sum((x - mean_a) ** 2 for x in ra)) or 1e-12
    den_b = math.sqrt(sum((y - mean_b) ** 2 for y in rb)) or 1e-12
    return num / (den_a * den_b)


def _rank(xs: List[float]) -> List[float]:
    """Average-rank ties; matches scipy.stats.rankdata's default behavior."""
    sorted_pairs = sorted(enumerate(xs), key=lambda p: p[1])
    ranks: List[float] = [0.0] * len(xs)
    i = 0
    n = len(xs)
    while i < n:
        j = i
        while j + 1 < n and sorted_pairs[j + 1][1] == sorted_pairs[i][1]:
            j += 1
        avg = (i + j) / 2.0 + 1  # 1-indexed
        for k in range(i, j + 1):
            ranks[sorted_pairs[k][0]] = avg
        i = j + 1
    return ranks


def permutation_p(
    a: List[float],
    b: List[float],
    *,
    n_permutations: int,
    rng_seed: int,
) -> float:
    """Two-sided permutation null on Spearman ρ."""
    import random
    rng = random.Random(rng_seed)
    observed = spearman_rho(a, b)
    n_at_least = 0
    b_perm = list(b)
    for _ in range(n_permutations):
        rng.shuffle(b_perm)
        if spearman_rho(a, b_perm) >= observed:
            n_at_least += 1
    return n_at_least / n_permutations


# ════════════════════════════════════════════════════════════════════
# Phase 1 main
# ════════════════════════════════════════════════════════════════════
def run_phase_1(*, decisions_path: Path, corpora_path: Path,
                dry_run: bool = False) -> int:
    """Execute Phase 1. Returns exit code: 0 = H1 cleared, 1 = H1
    failed (closed-negative paper drafted), 2 = preconditions not met
    (operator decisions / corpora not committed yet)."""
    print("══════════════════════════════════════════════════════════════")
    print(" styxx 8.0 grounded-arc · bet 0 · phase 1")
    print(f" pre-registration: {PREREG_PATH}")
    print(f" timestamp: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    print("══════════════════════════════════════════════════════════════")
    print()

    # ── 1. Load operator decisions ─────────────────────────────────
    try:
        decisions = OperatorDecisions.load(decisions_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"  [PRECONDITION FAIL] {e}")
        print()
        print("  → fill in papers/grounded-arc/preregistration_2026_05_19.md §4")
        print("  → commit an amendment JSON at:", decisions_path)
        print("  → re-run this script")
        return 2

    print(f"  ✓ operator decisions LOCKED:")
    print(f"      embedding_model = {decisions.embedding_model}")
    print(f"      h1_abandon_rho  = {decisions.h1_abandon_rho}  "
          f"(pre-reg floor: {H1_BAR_RHO})")
    print(f"      ship_target     = {decisions.ship_target}")
    print(f"      amendment_hash  = {decisions.amendment_commit_hash}")
    print()

    # ── 2. Load and verify holdout corpora ─────────────────────────
    corpora = load_holdout_corpora(corpora_path)
    missing = [i for i in INSTRUMENTS if i not in corpora]
    if missing:
        print(f"  [PRECONDITION FAIL] holdout corpora not constructed yet")
        print(f"  missing for: {missing}")
        print(f"  → construct holdouts per pre-reg §5")
        print(f"  → write {corpora_path} with hashes")
        return 2

    for inst, corpus in corpora.items():
        print(f"  ✓ holdout[{inst:>15}]  n={corpus.n_prompts:>4}  "
              f"sha256={corpus.sha256[:16]}…  bins={corpus.bins}")
    print()

    if dry_run:
        print("  [DRY RUN] preconditions met. would now embed prompts and "
              "score instruments. exiting.")
        return 0

    # ── 3. The actual run (placeholders — implement when reaching here) ──
    # The structure below is the LOCKED-IN execution path. The
    # implementations marked TODO are pure compute on already-locked
    # corpora and decisions; nothing in them changes the experimental
    # design. They are the parts that will land in the next session
    # after operator decisions are filled in.
    print("  [NOT IMPLEMENTED] Phase 1 execution body — fills in next session")
    print(
        "  the structure: for each instrument I,\n"
        "    1. embed all holdout prompts via decisions.embedding_model\n"
        "    2. compute k-NN distance to calibration corpus (k=10)\n"
        "    3. apply threshold_law_validity() per prompt\n"
        "    4. score each (prompt, response) with score_instrument(I)\n"
        "    5. compute error = |score - gold|\n"
        "    6. ρ = spearman_rho(validity, -error)\n"
        "    7. p = permutation_p(validity, -error, n=H1_PERMUTATIONS)\n"
        "  decision rule:\n"
        f"    if ρ_refusal < {H1_BAR_RHO}: write H1_FAILED.md, return 1\n"
        "    else: write H1_PASSED.md, return 0; continue to H2/H3/H4."
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decisions",
        default="papers/grounded-arc/operator_decisions.json",
        help="path to the locked operator-decisions JSON",
    )
    parser.add_argument(
        "--corpora",
        default="papers/grounded-arc/holdout_corpora.json",
        help="path to the locked holdout-corpora identification JSON",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="exit after precondition checks; do not embed or score",
    )
    args = parser.parse_args(argv)
    return run_phase_1(
        decisions_path=Path(args.decisions),
        corpora_path=Path(args.corpora),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
