# -*- coding: utf-8 -*-
"""Repro for the styxx 7.23.0 SUBSTRATE GATE — the confound auditor now refuses to trust its own
synthetic corpus. Backs FINDING_groundtruth_substrate_artifact_2026_06_27.md by showing the tool
change that operationalizes it. Key-free, no transformers/network (audits via injected score_fn).

    python papers/grounded-honesty-axis/substrate_gate_repro.py

Shows:
  1. the bundled (LLM-generated) boundary corpora are lexically ENTANGLED with length — a
     label-trained bag-of-words margin rides the confound within class (permutation p < 0.05);
  2. audit_hf_model on a length-biased scorer now emits SYNTHETIC-ARTIFACT RISK by default;
  3. validate_against_ground_truth refutes that verdict when real-data behavior is clean.
"""
import numpy as np

import styxx
from styxx.confound_audit import _lexical_entanglement
from styxx.hf_audit import _load_corpus


def main() -> None:
    print("styxx", styxx.__version__, "— substrate gate repro\n")

    print("[1] lexical-entanglement fingerprint on the BUNDLED synthetic corpora")
    for construct in ("sentiment", "toxicity"):
        rows = _load_corpus(construct)
        y = np.array([r["label"] for r in rows])
        C = np.array([r["confound"] for r in rows], float)
        texts = [r["text"] for r in rows]
        corr, p = _lexical_entanglement(texts, y, C)
        flag = "ENTANGLED (artifact fingerprint)" if (p is not None and p < 0.05) else "n.s."
        print(f"    {construct:9s}: within-label |corr(BoW-margin, length)| = {corr}  perm p = {p}  -> {flag}")

    print("\n[2] audit_hf_model on a length-biased scorer (bundled synthetic corpus)")
    rows = _load_corpus("sentiment")
    by = {r["text"]: r for r in rows}
    cmean = sum(r["confound"] for r in rows) / len(rows)
    biased = lambda t: 0.5 + 0.30 * (2 * by[t]["label"] - 1) + 0.25 * (by[t]["confound"] - cmean)
    rep = styxx.audit_hf_model("demo/length-biased", construct="sentiment", score_fn=biased)
    print(f"    verdict head            : {rep.verdict.split('—')[0].strip()}")
    print(f"    corpus_provenance       : {rep.corpus_provenance}")
    print(f"    synthetic_artifact_warning: {rep.synthetic_artifact_warning}")
    print(f"    lexical_confound_corr/p : {rep.lexical_confound_corr} / {rep.lexical_confound_p}")

    print("\n[3] validate_against_ground_truth — real-data behavior is clean -> refuted")
    real_rows = [{"text": f"real{i}", "label": int(rows[i]["label"]), "confound": rows[i]["confound"]}
                 for i in range(len(rows))]
    by_real = {r["text"]: r for r in real_rows}
    clean = lambda t: 0.5 + 0.40 * (2 * by_real[t]["label"] - 1)  # no length term on real data
    real, rec = styxx.validate_against_ground_truth(rep, real_rows, score_fn=clean, confound="log_words")
    print(f"    real verdict head       : {real.verdict.split('—')[0].strip()}  (provenance {real.corpus_provenance}, n={real.n})")
    print(f"    reconciliation          : {rec.split('—')[0].strip()}")


if __name__ == "__main__":
    main()
