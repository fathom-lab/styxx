# -*- coding: utf-8 -*-
"""styxx.audit_confound — confound-robustness audit + auto-guard for ANY text-scoring instrument.

The question every deployed classifier / guardrail / probe should answer but almost none do: *is my score
secretly riding a confound?* A toxicity classifier that keys on length, a sentiment model that keys on
punctuation, an overconfidence guardrail that keys on terseness — each separates its training corpus while
silently failing in deployment whenever the confound and the construct come apart.

This tool answers it the only way you can answer it cleanly: with a corpus where the construct and the suspected
confound are ORTHOGONAL (decorrelated by design — usually frontier-generated, see `build_confound_grid`). On that
corpus it measures three things and, if needed, hands back a fix:

  1. DISCRIMINATION — does the instrument still separate the construct *within* each confound stratum? (robust)
     or does it need the confound to discriminate at all? (broken)
  2. SCORE BIAS — holding the construct fixed, how much does the confound move the score? (OLS coef + bootstrap CI)
  3. DEPLOYMENT HARM — at a fixed threshold, the false-positive / false-negative rate swing across confound levels.
  4. GUARD — if the bias is at the score level (discrimination intact), an operating-point-preserving correction
     that removes it, validated 5-fold OUT-OF-SAMPLE, returned as a ready-to-use `report.guard(score, confound)`.

This generalizes the validated overconfidence red-team (a 4%->46% length-driven error swing) + its deployment
guard into one instrument-agnostic primitive. Companion to ``styxx.validate_probe`` (is a probe tracking the
concept?) — this asks: is a *score* tracking the concept, or a confound?

  from styxx import audit_confound
  rows = [{"text": t, "label": 0|1, "confound": c}, ...]   # label = construct class; confound = the cue to test
  report = audit_confound(rows, score_fn=my_instrument)     # or scores=[...] precomputed
  if report.verdict.startswith("THRESHOLD"):
      fair_score = report.guard(raw_score, confound_value)  # length/whatever-fair score
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import math

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

ORTHO_BAR = 0.20   # |corr(label, confound)| must be <= this for the grid to count as orthogonal
AUC_BAR = 0.70     # within-stratum discrimination floor to call the instrument "robust"


@dataclass
class ConfoundAuditReport:
    """Result of :func:`audit_confound`. ``guard`` is a ready-to-use length/confound-fair score correction."""
    instrument: str
    confound: str
    n: int
    gate_ok: bool
    orthogonality_corr: float
    overall_auc: float
    within_stratum_auc: Dict[str, float]
    confound_score_coef: float
    confound_score_coef_ci95: Tuple[float, float]
    harm: Dict[str, Any]
    guard_auc_raw: float
    guard_auc_adj_oos: float
    guard_disparity_raw: float
    guard_disparity_adj_oos: float
    verdict: str
    guard_slope: float = 0.0
    guard_ref: float = 0.0
    construct_recoverable_auc: Optional[float] = None

    def guard(self, raw_score: float, confound_value: float) -> float:
        """Operating-point-preserving confound correction: at the reference confound level the score is
        unchanged; deviations are corrected. Only meaningful when the verdict is THRESHOLD-biased."""
        return float(raw_score - self.guard_slope * (confound_value - self.guard_ref))

    def summary(self) -> str:
        return self.verdict


def _wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (round(max(0.0, c - h), 3), round(min(1.0, c + h), 3))


def _boot_coef_ci(D: np.ndarray, S: np.ndarray, col: int, reps: int = 2000, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed); out = []
    n = len(S)
    for _ in range(reps):
        ix = rng.integers(0, n, n)
        try:
            out.append(np.linalg.lstsq(D[ix], S[ix], rcond=None)[0][col])
        except Exception:
            pass
    return (round(float(np.percentile(out, 2.5)), 3), round(float(np.percentile(out, 97.5)), 3))


def audit_confound(rows: List[Dict[str, Any]], score_fn: Optional[Callable[[str], float]] = None, *,
                   scores: Optional[List[float]] = None, label_key: str = "label",
                   confound_key: str = "confound", text_key: str = "text",
                   instrument: str = "instrument", confound: str = "confound",
                   construct_recoverable_auc: Optional[float] = None) -> ConfoundAuditReport:
    """Audit whether ``score_fn`` (or precomputed ``scores``) rides ``confound`` rather than the construct.

    rows: each dict has the construct ``label`` (0/1), the ``confound`` value (numeric), and the ``text``.
    Provide EITHER ``score_fn`` (called on each text) OR ``scores`` (precomputed, aligned with rows).
    The corpus should already decorrelate label and confound (use :func:`build_confound_grid`); the report's
    ``gate_ok`` reports whether that orthogonality actually held.

    ``construct_recoverable_auc`` (optional, RECOMMENDED): the CV-AUC of a fresh refit on this corpus's text
    (e.g. a BoW or feature refit). It disambiguates a CONFOUND-DEPENDENT verdict — if the construct is
    recoverable from the text (refit AUC high) but the instrument's score isn't tracking it, the INSTRUMENT is
    broken (keying on the confound); if the construct is NOT recoverable (refit AUC ~chance), the corpus didn't
    instantiate it and the audit is INCONCLUSIVE. Without it, this tool audits the SCORE only and cannot tell
    a broken instrument from a degenerate corpus.
    """
    if scores is None:
        if score_fn is None:
            raise ValueError("provide score_fn or scores")
        scores = [float(score_fn(r[text_key])) for r in rows]
    S = np.asarray(scores, float)
    y = np.asarray([int(r[label_key]) for r in rows])
    C = np.asarray([float(r[confound_key]) for r in rows], float)
    n = len(S)
    if n < 20 or len(np.unique(y)) < 2:
        raise ValueError("need >=20 rows and both classes present")

    ortho = float(np.corrcoef(y, C)[0, 1]) if C.std() > 0 else float("nan")
    is_high = (C > np.median(C)).astype(int)
    gate_ok = bool(C.std() > 0 and abs(ortho) <= ORTHO_BAR
                   and is_high.sum() >= 2 and (1 - is_high).sum() >= 2)

    overall_auc = float(roc_auc_score(y, S))

    def _stratum_auc(mask):
        ys, ss = y[mask], S[mask]
        if len(np.unique(ys)) < 2:
            return float("nan")
        return float(roc_auc_score(ys, ss))
    auc_low = _stratum_auc(is_high == 0); auc_high = _stratum_auc(is_high == 1)

    # score bias holding construct fixed: S ~ 1 + label + is_high
    D = np.column_stack([np.ones(n), y.astype(float), is_high.astype(float)])
    beta = np.linalg.lstsq(D, S, rcond=None)[0]
    coef = float(beta[2]); coef_ci = _boot_coef_ci(D, S, 2)

    # deployment harm at a fixed threshold = median(S): FP (label 0 above) / FN (label 1 below) per stratum
    thr = float(np.median(S))

    def _err(label_val, hi, above):
        m = (y == label_val) & (is_high == hi); s = S[m]
        k = int((s > thr).sum()) if above else int((s < thr).sum())
        nn = int(m.sum())
        return {"k": k, "n": nn, "rate": round(k / nn, 3) if nn else None, "ci95": list(_wilson(k, nn)) if nn else None}
    harm = {"fp_high": _err(0, 1, True), "fp_low": _err(0, 0, True),
            "fn_high": _err(1, 1, False), "fn_low": _err(1, 0, False)}
    fp_swing = abs((harm["fp_high"]["rate"] or 0) - (harm["fp_low"]["rate"] or 0))
    fn_swing = abs((harm["fn_high"]["rate"] or 0) - (harm["fn_low"]["rate"] or 0))
    harm["max_swing"] = round(max(fp_swing, fn_swing), 3)

    # guard: pure confound effect (label ⟂ confound by design) = slope of S on C; recenter on mean C.
    slope = float(np.linalg.lstsq(np.column_stack([np.ones(n), C]), S, rcond=None)[0][1])
    ref = float(C.mean())
    # OOS validation: fit slope/ref on train, apply to held-out
    S_adj = np.zeros(n)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(S):
        b = np.linalg.lstsq(np.column_stack([np.ones(len(tr)), C[tr]]), S[tr], rcond=None)[0]
        S_adj[te] = S[te] - b[1] * (C[te] - C[tr].mean())
    auc_adj = float(roc_auc_score(y, S_adj))

    def _fp_disp(score):
        t = np.median(score); cal = (y == 0)
        return float((score[cal & (is_high == 1)] > t).mean() - (score[cal & (is_high == 0)] > t).mean())
    disp_raw = _fp_disp(S); disp_adj = _fp_disp(S_adj)

    coef_sig = (coef_ci[0] > 0 or coef_ci[1] < 0)
    construct_absent = construct_recoverable_auc is not None and construct_recoverable_auc < 0.60
    recov_note = ("" if construct_recoverable_auc is None
                  else f" (construct IS recoverable from text, refit AUC {construct_recoverable_auc:.2f} -> the "
                       f"INSTRUMENT is keying on '{confound}', not that the construct is undetectable; fix = "
                       f"retrain/reground, NOT a threshold guard)")
    if not gate_ok:
        verdict = (f"INCONCLUSIVE — corpus is not confound-orthogonal (corr(label,{confound})={ortho:+.2f}, "
                   f"need |.|<={ORTHO_BAR}); cannot separate construct from confound. Rebuild the grid.")
    elif construct_absent:
        verdict = (f"INCONCLUSIVE — the construct is not recoverable from the text even by a refit "
                   f"(AUC {construct_recoverable_auc:.2f} ~ chance); the corpus did not instantiate the construct. "
                   f"Fix the grid (stronger stances) before trusting any confound verdict.")
    elif (np.isnan(auc_low) or np.isnan(auc_high)):
        verdict = "INCONCLUSIVE — a confound stratum lacks both classes; widen the grid."
    elif auc_low >= AUC_BAR and auc_high >= AUC_BAR and coef_sig:
        _gh = (auc_adj >= overall_auc - 0.005 and abs(disp_adj) < abs(disp_raw))
        _gtxt = ("a guard (residualize on confound) keeps AUC and cuts the FP length-disparity "
                 f"{disp_raw:+.2f}->{disp_adj:+.2f} (5-fold OOS) -> use report.guard(score, confound)") if _gh else (
                 f"a guard reduces the FP length-disparity {disp_raw:+.2f}->{disp_adj:+.2f} but trades AUC "
                 f"{overall_auc:.2f}->{auc_adj:.2f} (5-fold OOS) — worth it only if length-fairness outweighs the AUC cost")
        verdict = (f"THRESHOLD-BIASED (discrimination robust) — '{instrument}' separates the construct within "
                   f"each '{confound}' stratum (AUC {auc_low:.2f}/{auc_high:.2f}) but the SCORE shifts with the "
                   f"confound (coef {coef:+.2f}, 95% CI {list(coef_ci)}). At a fixed threshold the error rate swings "
                   f"~{harm['max_swing']:.0%} across confound levels. {_gtxt}. Fix is a deployment threshold, not a retrain.")
    elif auc_low < AUC_BAR or auc_high < AUC_BAR:
        verdict = (f"CONFOUND-DEPENDENT (broken) — within-stratum discrimination drops (AUC {auc_low:.2f}/"
                   f"{auc_high:.2f}); '{instrument}' needs '{confound}' to discriminate. A threshold guard will NOT "
                   f"fix this — the instrument is keying on the confound, not the construct." + recov_note)
    else:
        verdict = (f"ROBUST — '{instrument}' separates the construct (AUC {overall_auc:.2f}) and the confound's "
                   f"score effect is not significant (coef {coef:+.2f}, CI {list(coef_ci)}); swing "
                   f"{harm['max_swing']:.0%}. No confound problem detected.")

    return ConfoundAuditReport(
        instrument=instrument, confound=confound, n=n, gate_ok=gate_ok, orthogonality_corr=round(ortho, 3),
        overall_auc=round(overall_auc, 3), within_stratum_auc={"low": round(auc_low, 3), "high": round(auc_high, 3)},
        confound_score_coef=round(coef, 3), confound_score_coef_ci95=coef_ci, harm=harm,
        guard_auc_raw=round(overall_auc, 3), guard_auc_adj_oos=round(auc_adj, 3),
        guard_disparity_raw=round(disp_raw, 3), guard_disparity_adj_oos=round(disp_adj, 3),
        verdict=verdict, guard_slope=slope, guard_ref=ref,
        construct_recoverable_auc=construct_recoverable_auc)


def build_confound_grid(items: List[str], pos_prompt: str, neg_prompt: str,
                        confound_rules: Dict[str, str], generate_fn: Callable[[str, str], str], *,
                        confound_value_fn: Callable[[str], float] = lambda t: math.log1p(len(t.split())),
                        ) -> List[Dict[str, Any]]:
    """Generate an orthogonal construct×confound grid for :func:`audit_confound` using any text generator.

    items: the prompts/questions to answer. pos_prompt/neg_prompt: the two construct stance system-prompts
    (label 1 / label 0). confound_rules: {confound_level_name: instruction_suffix} (e.g. {"short": "...one
    sentence...", "long": "...5 sentences..."}). generate_fn(system_prompt, user_item) -> text — bring your own
    vendor (frontier recommended; weak models fail to hold stance AND confound, which the report's gate catches).
    confound_value_fn(text) -> numeric confound (default = log word count).
    Crosses construct × confound so the two are orthogonal by construction.
    """
    rows: List[Dict[str, Any]] = []
    for it in items:
        for label, stance in ((0, neg_prompt), (1, pos_prompt)):
            for level, rule in confound_rules.items():
                text = generate_fn((stance + " " + rule).strip(), it)
                if not text:
                    continue
                rows.append({"text": text, "label": label, "confound": float(confound_value_fn(text)),
                             "confound_level": level, "item": it})
    return rows
