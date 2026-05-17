"""
corpus_coverage_law_fine.py — high-resolution replication of the
corpus-coverage law + sufficiency-threshold localization.

The law was supported on 5 corpus levels (Spearman +0.83). A 5-point
relationship is suggestive; a fine monotone curve with a located
saturation threshold is a *quantitative, operational* law. This is a
higher-resolution REPLICATION of an already-preregistered result — not
a new claim. Reuses the validated code + the FIXED cached behavioral
labels (no new generation; embeddings only).

PREREGISTERED (before run):
  ~12 size-controlled (n=360) label-free corpora, domain-fraction
  finely graded 0.00→1.00 plus a pure-narrative far anchor; overlap
  measured (mean-max-cosine, home space).
  H1': cross-family (mpnet) Spearman(overlap,AUC) ≥ 0.60 at higher
       resolution (replication).
  H2': same-family (te3-small) |Spearman| < 0.40 (control still flat).
  THRESHOLD := smallest measured overlap with cross-family mean AUC
       ≥ 0.80 (operational sufficiency point).
  Replication holds iff H1' ∧ H2'. Honest negative otherwise.

Caveats unchanged: lexical refusal labels; OpenAI-only (NOT
cross-vendor — the real paper gate, blocked on a 2nd-vendor key);
single overlap metric; single seed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from corpus_coverage_law import (  # noqa: E402  — reuse validated code
    _generic_pool, _narrative_pool, _domain_pool, eval_prompts,
    embed_oai, embed_mpnet, auc, spearman, overlap_metric, OBV, CACHE,
    MODELS, Transport, CognometricInstrument, transported_score,
)

SEED = 20260517
RNG = np.random.default_rng(SEED)
S = 360
DOMAIN_FRACS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40,
                0.55, 0.70, 0.85, 1.00]


def _take(pool, k):
    out, i = [], 0
    while len(out) < k:
        out.append(pool[i % len(pool)]); i += 1
    return out[:k]


def make_fine_levels():
    G = list(RNG.permutation(_generic_pool()))
    N = list(RNG.permutation(_narrative_pool()))
    D = list(RNG.permutation(_domain_pool()))
    ev = set(eval_prompts())
    levels = {"Lnarr": _take(N, S)}
    for f in DOMAIN_FRACS:
        nd = int(round(S * f))
        levels[f"f{f:0.2f}"] = _take(G, S - nd) + _take(D, nd)
    for k, v in levels.items():
        assert len(v) == S and not (set(v) & ev), f"bad level {k}"
    return levels


def main():
    if not CACHE.exists():
        print("FATAL: behavioral cache missing — run corpus_coverage_law.py "
              "first (need identical fixed labels). Aborting; no new "
              "generation in this replication by design.")
        return
    beh = {k: np.array(v, float)
           for k, v in json.loads(CACHE.read_text()).items()}
    prompts = eval_prompts()
    obv_txt = [t for t, _ in OBV]
    obv_y = np.array([v for _, v in OBV])
    print(f"reused fixed labels ({len(prompts)} prompts, "
          f"{len(beh)} models); replication, no new generation")

    levels = make_fine_levels()
    A_obv = embed_oai("text-embedding-3-large", obv_txt)
    A_p = embed_oai("text-embedding-3-large", prompts)
    A_lvl = {k: embed_oai("text-embedding-3-large", v)
             for k, v in levels.items()}
    foreigns = {"text-embedding-3-small":
                lambda t: embed_oai("text-embedding-3-small", t),
                "all-mpnet-base-v2": embed_mpnet}
    B_obv = {f: fn(obv_txt) for f, fn in foreigns.items()}
    B_p = {f: fn(prompts) for f, fn in foreigns.items()}

    rows = []
    print(f"\n{'level':<9}{'overlap':>9}{'cf_AUC(mpnet)':>15}"
          f"{'sf_AUC(ctrl)':>14}")
    # stable order by measured overlap
    ov_by = {k: overlap_metric(A_lvl[k], A_p) for k in levels}
    for lvl in sorted(levels, key=lambda k: ov_by[k]):
        ov = ov_by[lvl]
        rec = {"level": lvl, "overlap": round(ov, 4)}
        for fname, fn in foreigns.items():
            t = Transport.fit(A_lvl[lvl], fn(levels[lvl]),
                              method="procrustes")
            instr = CognometricInstrument.from_labeled(
                t.home_repr(A_obv), obv_y)
            sc = transported_score(instr, t, B_p[fname])
            ms = float(np.nanmean([auc(sc, beh[m]) for m in MODELS]))
            rec[fname] = round(ms, 4)
        rows.append(rec)
        print(f"{lvl:<9}{ov:>9.3f}"
              f"{rec['all-mpnet-base-v2']:>15.3f}"
              f"{rec['text-embedding-3-small']:>14.3f}")

    cf_ov = np.array([r["overlap"] for r in rows])
    cf_au = np.array([r["all-mpnet-base-v2"] for r in rows])
    sf_au = np.array([r["text-embedding-3-small"] for r in rows])
    rho_cf = spearman(cf_ov, cf_au)
    rho_sf = spearman(cf_ov, sf_au)

    above = [r["overlap"] for r in sorted(rows, key=lambda r: r["overlap"])
             if r["all-mpnet-base-v2"] >= 0.80]
    threshold = round(min(above), 4) if above else None

    H1 = rho_cf >= 0.60
    H2 = abs(rho_sf) < 0.40
    ok = H1 and H2

    out = {
        "ts": "2026-05-17",
        "experiment": "corpus-coverage law — high-resolution replication "
                       "+ sufficiency-threshold localization",
        "preregistered": {"H1p": "cf Spearman>=0.60", "H2p": "|sf Spearman|<0.40",
                          "threshold": "min overlap with cf mean AUC>=0.80"},
        "n_levels": len(rows), "size_controlled_n": S,
        "n_eval": len(prompts), "models": MODELS,
        "labels": "FIXED, reused from _behavior_cache_75.json (no new gen)",
        "rows": rows,
        "stats": {"spearman_crossfamily": round(rho_cf, 4),
                  "spearman_samefamily": round(rho_sf, 4),
                  "sufficiency_threshold_overlap": threshold,
                  "cf_AUC_at_min_overlap": round(float(cf_au[np.argmin(cf_ov)]), 4),
                  "cf_AUC_at_max_overlap": round(float(cf_au[np.argmax(cf_ov)]), 4),
                  "H1p_replicates": bool(H1), "H2p_control_flat": bool(H2),
                  "REPLICATION_HOLDS": bool(ok)},
        "caveats": ["lexical refusal labels",
                    "OpenAI-only — NOT cross-vendor (real paper gate, "
                    "blocked on a 2nd-vendor key)",
                    "single overlap metric (mean-max-cosine); single seed"],
    }
    op = HERE / "out_corpus_coverage_law_fine.json"
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved: {op}")

    print("\n" + "=" * 60)
    print(f"resolution: {len(rows)} levels  (was 5)")
    print(f"cross-family Spearman(overlap,AUC) = {rho_cf:+.3f}  "
          f"H1'={'PASS' if H1 else 'FAIL'}")
    print(f"same-family  Spearman(overlap,AUC) = {rho_sf:+.3f}  "
          f"H2'={'PASS' if H2 else 'FAIL'} (control)")
    print(f"cross-family AUC: {out['stats']['cf_AUC_at_min_overlap']:.3f} "
          f"(min overlap) -> {out['stats']['cf_AUC_at_max_overlap']:.3f} "
          f"(max overlap)")
    print(f"sufficiency threshold: corpus↔domain overlap >= "
          f"{threshold} -> cross-family AUC >= 0.80")
    print(f"\nVERDICT: {'LAW REPLICATES AT RESOLUTION — operational '
                        'threshold located' if ok else 'REPLICATION FAILED '
                        '— honest negative'}")


if __name__ == "__main__":
    main()
