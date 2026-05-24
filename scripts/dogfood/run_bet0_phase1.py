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

    # ── 3. The actual run — uses the validated validity engine ─────
    # (papers/grounded-arc/validity_engine.py, self-tested 2026-05-24).
    # Pure compute on already-locked corpora + decisions; the experimental
    # design is fixed above. Unreachable until corpora + calibration params
    # are committed — the precondition gates above return 2 before here.
    import importlib.util as _ilu
    _ep = Path(__file__).resolve().parents[2] / "papers" / "grounded-arc" / "validity_engine.py"
    _spec = _ilu.spec_from_file_location("validity_engine", _ep)
    eng = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(eng)  # type: ignore[union-attr]

    grounded = corpora_path.parent
    cal_params_path = grounded / "calibration_params.json"
    if not cal_params_path.exists():
        print(f"  [PRECONDITION FAIL] validation-slice calibration not fit: {cal_params_path}")
        print("  → fit α per instrument on the validation slice (τ locked at the")
        print("    threshold-law paper value; τ_distance = 1 − 0.31 = 0.69) and")
        print("    commit calibration_params.json — never fit α on the holdout.")
        return 2
    cal_params = json.loads(cal_params_path.read_text(encoding="utf-8"))

    results: Dict[str, Dict[str, float]] = {}
    for inst in INSTRUMENTS:
        corpus = corpora[inst]
        cal_prompts = [
            ln for ln in (grounded / "calibration" / f"{inst}.txt")
            .read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        hold = [
            json.loads(ln) for ln in (grounded / "holdout" / f"{inst}.jsonl")
            .read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        # Locked-hash verification — refuse to score a mutated holdout.
        if not corpus.verify([h["prompt"] for h in hold]):
            print(f"  [HASH MISMATCH] holdout[{inst}] != locked sha256 — refusing to run")
            return 2
        tau = float(cal_params[inst].get("tau", eng.TAU_DISTANCE))
        alpha = float(cal_params[inst]["alpha"])
        cal_emb = eng.embed(cal_prompts, decisions.embedding_model)
        hold_emb = eng.embed([h["prompt"] for h in hold], decisions.embedding_model)
        validities: List[float] = []
        neg_err: List[float] = []
        for item, q in zip(hold, hold_emb):
            d = eng.knn_distance(q, cal_emb, k=10)
            validities.append(eng.validity(d, tau=tau, alpha=alpha))
            s = score_instrument(inst, item["prompt"], item["response"],
                                 item.get("reference"))
            neg_err.append(-abs(s - float(item["gold"])))
        rho = spearman_rho(validities, neg_err)
        p = permutation_p(validities, neg_err,
                          n_permutations=H1_PERMUTATIONS, rng_seed=RNG_SEED)
        results[inst] = {"rho": rho, "p": p, "n": float(len(hold))}
        print(f"  ρ[{inst:>15}] = {rho:+.3f}  (p={p:.4f}, n={len(hold)})")

    rho_ref = results[HEADLINE_INSTRUMENT]["rho"]
    p_ref = results[HEADLINE_INSTRUMENT]["p"]
    bar = decisions.h1_abandon_rho
    cleared = rho_ref >= bar and p_ref < H1_BAR_P
    summary = (
        f"# H1 {'PASSED' if cleared else 'FAILED'}\n\n"
        f"headline instrument: {HEADLINE_INSTRUMENT}\n"
        f"rho = {rho_ref:+.4f} (bar {bar}); permutation p = {p_ref:.4f} "
        f"(bar {H1_BAR_P})\n\nall instruments:\n"
        + "\n".join(f"- {i}: rho={r['rho']:+.4f} p={r['p']:.4f} n={int(r['n'])}"
                    for i, r in results.items())
        + f"\n\nembedding_model: {decisions.embedding_model}\n"
        f"amendment_commit: {decisions.amendment_commit_hash}\n"
    )
    (grounded / ("H1_PASSED.md" if cleared else "H1_FAILED.md")).write_text(
        summary, encoding="utf-8")
    print()
    if cleared:
        print(f"  H1 CLEARED: ρ_{HEADLINE_INSTRUMENT}={rho_ref:+.3f} ≥ {bar} "
              f"(p={p_ref:.4f}). → continue to H2/H3/H4.")
        return 0
    print(f"  H1 FAILED: ρ_{HEADLINE_INSTRUMENT}={rho_ref:+.3f} < {bar}. "
          f"→ arc abandoned; closed-negative paper (H1_FAILED.md).")
    return 1


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
    parser.add_argument(
        "--self-test", action="store_true",
        help="validate the validity engine on synthetic + real-embedding data "
             "(touches NO holdout); proves the kill-gate machinery is runnable",
    )
    args = parser.parse_args(argv)
    if args.self_test:
        import importlib.util as _ilu
        _ep = Path(__file__).resolve().parents[2] / "papers" / "grounded-arc" / "validity_engine.py"
        _spec = _ilu.spec_from_file_location("validity_engine", _ep)
        _eng = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_eng)  # type: ignore[union-attr]
        return _eng.self_test()
    return run_phase_1(
        decisions_path=Path(args.decisions),
        corpora_path=Path(args.corpora),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
