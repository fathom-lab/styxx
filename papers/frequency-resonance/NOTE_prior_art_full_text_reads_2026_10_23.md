# NOTE — the prior-art full texts the permuted-MNIST robustness RESULT owes: LRU, S4D, LinOSS and D-LinOSS on real versus complex eigenvalues — 2026-10-23

**status: the full-text reads that `RESULT_pmnist_robustness_2026_10_23.md` l.431-433 owes before any public
sentence, made from arXiv's HTML on 2026-10-23, each paper by one reader and again by an independent re-reader. It
edits no committed document: not that RESULT, not `RESULT_pmnist_ablation_2026_07_23.md` (which
`CERT_pmnist_ablation.json` names), not `PAPER_oscillation_causal_map_2026_07_23.md`, not `papers/INDEX.md`. It
swears no spans, because its sources are outside this repository. It states which of those documents' sentences
stand, which need a pointer or a banner, and which wording the reads show to be wrong, contradicted in part or not
supportable as written; how any of that lands is the operator's call.**

The debt, in the robustness RESULT's words (l.431-433): "Nothing beyond a three-block complex-diagonal linear
recurrence (LRU-style; `CLRU` in `run_pmnist_ablation.py`): the architecture is named by its code, and the prior-art
reads `ATTACK.md` §5 names (the LRU and S4D real-versus-complex rows) are owed before any public sentence; no priority
claim is made." `ATTACK.md` is the red team of the robustness design; it is not in this repository. Its §5 names five
reads: the LRU paper and S4D (the two the RESULT names), and TIDES, D-LinOSS and LinOSS. This NOTE reads four of
them. TIDES is not read (see "What was not read").

## How the reads were made, and what this NOTE takes from them

- Each paper was read in full from arXiv's HTML rendering, with the abs page for the version list. Nothing was
  downloaded and no PDF was opened. The LRU, LinOSS and D-LinOSS texts came through a fetch layer that summarizes
  what it fetches, so their table cells were confirmed in separate targeted passes, not copied as raw bytes. S4D's
  Tables 2 and 3 are SVG images in HTML v2 (Table 3's carry glyph paths and no text), so text extraction misses
  them; both reads rendered them in a browser pane and read them by eye.
- The re-reader checked each report against the paper and listed corrections and omissions. **This NOTE states only
  what the two reads agree on.** Where they differ, the section "Where the two reads differ" under each paper gives
  both. Items that only one read reports are left out, with one exception: three items that only the re-reader
  reported and that a third look, during review, confirmed in the paper (two S4D table cells and one D-LinOSS table
  row) appear marked as such.
- At most one quotation from each paper, under fifteen words. Section headings, table row labels and the papers' own
  names for things appear in quotation marks as locators, not as quotations. Lab sentences are quoted exactly, with
  file and line, as they read on origin/main `2e218601`.
- Every "the paper does not contain" below comes after both reads went through the paper's full table list, figure
  list and appendices; the place is named under "Where both reads looked".

## 1. The LRU paper — Orvieto et al., arXiv 2303.06349, v1

**Citation.** Orvieto, Smith, Gu, Fernando, Gulcehre, Pascanu and De, "Resurrecting Recurrent Neural Networks for
Long Sequences", arXiv 2303.06349. Version read: v1, the only version (submitted 2023-03-11; the abs page's comment
gives 30 pages and 9 figures). Read at https://arxiv.org/html/2303.06349v1, version list from
https://arxiv.org/abs/2303.06349.

**Real versus complex: no real-eigenvalue arm anywhere in the paper.** No table, figure or appendix trains a model
whose eigenvalues are held real, or whose phase is held at zero. `ATTACK.md` §5 recalled a "Λ real" row against a
"Λ complex" row on Long Range Arena. No row carries either label, and that recollection is refuted.

What the paper does contain:

- **Table 2 (main; sCIFAR, ListOps, Pathfinder) and its full copy, Table 7 (Appendix C.3, all six LRA tasks).** The
  rows build the LRU step by step. Test accuracy, mean of 3 runs, in the order sCIFAR / ListOps / Text / Retrieval /
  Pathfinder / PathX ("fails" where the table marks a failure):

  | row | sCIFAR | ListOps | Text | Retrieval | Pathfinder | PathX |
  |---|---|---|---|---|---|---|
  | Dense A | 72.2 | 50.4 | 89.1 | 89.1 | fails | fails |
  | Λ Real + Im | 86.5 | 58.8 | 87.4 | 87.8 | fails | fails |
  | Λ Exp | 85.4 | 60.5 | 86.5 | 89.4 | 65.4 (9.0) | fails |
  | Λ Stable Exp | 87.2 | 59.4 | 87.6 | 89.1 | 93.5 | fails |
  | + Ring Init | 88.1 | 59.4 | 89.4 | 90.1 | 94.4 | fails |
  | + γ normalization (the LRU; Table 8) | 89.0 | 60.2 | 89.4 | 89.9 | 95.1 | 94.2 |

  "Λ Real + Im" is a complex diagonal Λ learned through its real and imaginary parts, set against the polar
  `exp(−ν + iθ)` form. It is not a real-only row. The paper never defines the label in prose; both reads take it as
  that Cartesian form, and Section 3.3 contrasts the exponential form with learning the real and imaginary parts of
  Λ. "Dense A" is a dense real-valued linear RNN whose Glorot-initialized matrix has a complex spectrum (Figure 2
  plots it on the complex plane), so it rotates too. Table 8 repeats the same rows under other names ("Linear Dense
  RNN", "Diagonal Complex RNN", "Stable Exp Param w/ Ring Init", "+γ Normalization (LRU)").
- **Table 1 and Table 6 (real dense RNNs).** RNN-Lin 72.2 / 50.4 / 89.1 / 89.1, RNN-ReLU 69.7 / 37.6 / 88.0 / 88.5,
  RNN-Tanh 69.9 / 43.9 / 87.2 / 88.9 on the four tasks in the order above; all three fail Pathfinder and PathX.
- **Table 5 (Appendix C.2).** An `r_min` / `r_max` sweep on sCIFAR, for the complex LRU only.
- **Section 3.4 and Figure 5: phase range at initialization, PathX only.** Initializing the phase in [0, π/10]
  instead of [0, 2π], together with γ normalization, is what lets the LRU train on PathX (94.2% in the text; Figure
  5 gives curves, not numbers). Per the Table 8 caption, PathX uses [0, π/10] and every other task [0, 2π]. The phase
  stays learnable in both, so this is an initialization range, not a clamp.
- **Section 3.2.1 ("Are complex numbers really necessary?") and Appendix E.3 ("On alternatives to using complex
  numbers").** Derivation only, no experiment. The real alternative discussed is a real canonical form, algebraically
  equivalent to the complex diagonal and keeping its rotations. It is not a model with real eigenvalues.

**Tasks.** The benchmarks are the six Long Range Arena tasks. sMNIST, psMNIST and permuted MNIST appear nowhere in
the paper. Neither read finds a real-eigenvalue or zero-phase arm in any experiment; whether the paper also has
synthetic experiments is under "Where the two reads differ".

**Matched-ness.** Not matched in the way the lab's clamp is. Dense A against the diagonal rows changes the structure
(dense real against diagonal complex) and the parameter count. The paper states no parameter counts, and Section 2.2
says hyperparameters such as the learning rate were tuned per model on a log grid. Across Λ Real + Im, Λ Exp and Λ
Stable Exp the structure is fixed and only the parameterization moves, and every one of those arms keeps complex
eigenvalues.

**The authors' conclusion.** Section 4: the performance of deep SSMs comes from linear recurrence, complex diagonal
matrices, and the normalization and initialization that discretization induces, not from parameter sharing or the
continuous-time view. The polar form helps optimization by decoupling magnitude from oscillation frequency (Section
3.3). On complex numbers themselves the paper takes a representational line; Appendix E.3 closes on "complex numbers
are not per-se needed", meaning a real block form can carry the same rotations.

**Where the two reads differ.**

- *S4D variants named in the paper.* Reader: Section 4 mentions S4D-Lin only. Re-reader: Section 4 names S4D-Lin, and
  footnote 11 names S4D-Inv and S4D-LegS; Section 2.1 and Appendix B define S4D-Lin and S4D-Inv. Both: S4D-Real is
  not mentioned.
- *Figure 4.* Reader: Figures 4 and 5 together carry the PathX phase-range result. Re-reader: Figure 4 only
  illustrates a three-component state's response to an impulse over a long sequence; Figure 5 is the only PathX
  training comparison (γ normalization on or off, crossed with full or reduced phase).
- *How much of the LRU the lab's `CLRU` shares.* Reader: `CLRU` uses this paper's parameterization (`nu =
  log(−log r)`, `γ = √(1−|λ|²)` on the input, ring init `r` in [0.9, 0.999]). Re-reader: only `nu = log(−log r)` and
  the ring idea match. In the paper `θ = exp(θ_log)` with `θ_log` trained, so θ is strictly positive; γ is its own
  trained parameter, initialized at `√(1−|λ|²)`, and a Section 3.4 footnote says recomputing it each step works about
  as well, which is what the lab does; Appendix A's output is `Re(Cx) + Du`; and the ring radii are tuned per task
  (Table 8). Both: `CLRU` is LRU-style.
- *Appendix E.3's real form.* Reader: real 2×2 rotation blocks. Re-reader: block diagonal, 1×1 entries for the real
  eigenvalues of a general real matrix and 2×2 blocks for the conjugate pairs. Both: a derivation, with no model
  trained with real eigenvalues only.
- *Synthetic experiments.* Reader: the tasks are the six LRA tasks only. Re-reader: the paper also has three
  synthetic settings, a one-layer linear-versus-tanh RNN learning a length-100 convolution kernel (Figure 7,
  Appendix E.2), a single-eigenvalue toy that moves the target phase toward π/2 and compares the Cartesian and
  exponential parameterizations (Figure 8, Appendix E.2), and a numerical check of Proposition 3 (Figure 9). Both:
  none of them has a real-eigenvalue or θ = 0 arm. Figure 8 is phase-related, so the "one phase experiment" in §5
  below is the one on a long-range task.
- *Venue.* Reader: ICML 2023. Re-reader: the abs page names no venue. This NOTE names none.
- *Parameter counts.* Reader: not stated (medium confidence). Re-reader: not stated (high confidence; searched both
  windows of the text for the phrases).

**Where both reads looked.** The abs page; the whole HTML of v1 (174,553 characters, in two windows); Abstract;
Sections 1, 2.1, 2.2, 3.1, 3.2.1, 3.2.2, 3.3, 3.4, 4, 5; Tables 1-9 (Tables 1-3 in the main text, 4-9 in the
appendix); Figures 1-9 (captions); Appendices A, B, C.1-C.3, D, E.1-E.3, F; the reference list. Text searches for
MNIST, S4D-Real, "real diagonal" and parameter-count phrases found none.

## 2. S4D — Gu, Gupta, Goel and Ré, arXiv 2206.11893, v2

**Citation.** Albert Gu, Ankit Gupta, Karan Goel, Christopher Ré, "On the Parameterization and Initialization of
Diagonal State Space Models", arXiv 2206.11893. Version read: v2 (2022-08-05), the current one; v1 is dated
2022-06-23. Read at https://arxiv.org/html/2206.11893v2; Table 3(a) at
https://arxiv.org/html/2206.11893v2/diag_init.svg, Table 3(b) at `.../ablations_trainable.svg`, Table 2 at
`.../diag_real.svg` and `.../diag_compute.svg` (checked against appendix Tables 7 and 8); version list from
https://arxiv.org/abs/2206.11893.

**Real versus complex: yes, as an initialization row.** Table 3(a), which ablates the initialization of S4D's
diagonal A matrix (§5.2, paragraph "Ablation: Other S4D matrices"), carries a row "S4D-Real", placed in the S4D-Inv
block. In the paper's words, "S4D-Real uses a particular real initialization with A_n = −(n+1)" (§5.2). That is
the HiPPO-LegS real spectrum: the imaginary parts start at zero and the real parts move too, from −1/2 to −(n+1).

**Tasks and numbers (Table 3(a)).** Best validation accuracy for sCIFAR (length 1024, bidirectional) and 35-way
autoregressive Speech Commands on raw audio (length 16000); RMSE for BIDMC SpO2 (length 4000; lower is better). The
S4D-Real deltas are read against the S4D-Inv row by the table's layout; the caption does not say so in words.

| row (S4D-Inv block) | sCIFAR | Speech Commands | BIDMC RMSE |
|---|---|---|---|
| S4D-Inv (baseline) | 84.79 | 90.27 | 0.114 |
| S4D-Real | −5.45 (about 79.34) | −10.17 (about 80.10) | +0.066 (about 0.180) |
| S4D-Random | −6.32 | −1.95 | +0.034 |
| Scale 0.01 | −5.03 | −0.08 | +0.028 |
| Scale 100 | −7.77 | −52.31 | +0.034 |
| S4D-Inv2 | −2.62 (re-reader; third look) | −39.84 | +0.005 (re-reader; third look) |

The S4D-Lin block (baseline 85.12 / 90.66 / 0.128) has Scale 0.01 at −7.27 / −1.92 / +0.040 and Scale 100 at
−7.91 / −4.04 / +0.077, and no real-eigenvalue row. The "Scale 0.01" rows shrink every imaginary part 100× and keep
the real part at −1/2: the nearest thing in the paper to a phase-only knob, though not a clamp to zero, and Scale 100
hurts as well. Table 3(a) prints no standard deviations; Appendix B says results average 2 or 3 seeds. Appendix A's
Figure 6(b) reports the opposite extreme, real part 0 (oscillation with no decay), as performing poorly, with no
numbers.

No real-eigenvalue row appears in Table 3(b) (frozen against trained A and B), Table 4 (larger models), Table 5 (Long
Range Arena), Table 6 (Path-X parameterization; its "Identity" and "ReLU" Re(Λ) entries constrain the real part) or
Table 9 (Speech Commands). MNIST and permutation appear nowhere in the paper.

**Matched-ness.** One architecture and one protocol for every Table 3(a) row: 4 layers, H = 128, about 100K
parameters, learning rate 0.004, weight decay 0.01, no augmentation, 2-3 seeds, best validation accuracy (§5
Methodology; Appendix B). The S4D parameterization is fixed after §5.1, so only the initialization of the diagonal A
varies. It is not a single knob in the lab's sense: S4D-Real moves the real parts along with the phase, and the paper
nowhere says the imaginary parts are held at zero during training, so it does not show the eigenvalues stay real. It
is a real-versus-complex initialization contrast, not a clamp. The model is a convolution-mode S4D stack, not the
lab's three-block `CLRU`, and no row has a training-budget sweep.

**The authors' conclusion.** About diagonal SSMs, not oscillation as a mechanism. S4D-Real has the same spectrum as
S4-LegS, so diagonalization alone is not what matters; complex-valued matrices are needed to use diagonal SSMs at all
(§4, an expressivity argument, Proposition 2); and the imaginary parts set the oscillation frequencies of the basis
functions, which should be spread out (§4).

**Where the two reads differ.**

- *Whether A was trained in Table 3(a).* Reader: unknown from the paper alone. Re-reader: the text does not say, but
  the numbers do. Table 3(a)'s S4D-Inv baseline (SC 90.27, BIDMC 0.114) sits with Table 3(b)'s trained S4D-Inv (SC
  89.69, BIDMC 0.1132), not the frozen one (SC 76.53, BIDMC 0.0995), and the paper's parameterization keeps the
  imaginary part of A as its own trained tensor (§3.3, and the Figure 2 code). On that reading S4D-Real's eigenvalues
  were probably free to leave the real axis in training. Both: the paper does not show S4D-Real stays real; it is an
  initialization contrast, not a clamp.
- *Schedule.* Reader: cosine schedule for every row. Re-reader: cosine for sCIFAR and Speech Commands, a multistep
  scheduler for BIDMC (Appendix B).
- *How much the S4D-Real row says.* Reader: complex beats real on all three tasks; within the S4D-Inv block S4D-Real
  has the largest RMSE penalty and the third-largest Speech Commands drop. Re-reader: true only against the S4D-Inv
  baseline. Other complex rows of the same table do as badly or worse: on sCIFAR, Inv Scale 100 (−7.77) and
  S4D-Random (−6.32) drop more than S4D-Real (−5.45), and S4D-Lin Scale 0.01 and Scale 100 land at about 77.85 and
  77.21 against S4D-Real's 79.34; on Speech Commands, Inv Scale 100 (−52.31) and S4D-Inv2 (−39.84); on BIDMC, S4D-Lin
  Scale 100 reaches about 0.205 against S4D-Real's 0.180. So across the whole table S4D-Real is not uniquely worst on
  any task, and the table supports "this tuned complex spectrum beats this real spectrum, and many other complex
  spectra", not oscillation as such, even in direction.
- *Table 6's layout.* Reader: "Identity" and "ReLU" Re(Λ) are rows. Re-reader: Table 6 is a one-row table and they
  are columns. Both: they constrain the real part, not the imaginary part.

**Where both reads looked.** The abs page; the whole HTML of v2: Abstract, §1-§6, acknowledgments, references,
Appendix A (A.1, the low-rank perturbation section, Figures 5-6) and Appendix B (ablation protocol, full
parameterization and large-model results, the LRA setup, Table 10); Tables 1-10, with Tables 2 and 3 rendered from
their SVG images; the captions of Figures 1-8. Text searches for MNIST and "permut" found nothing.

## 3. LinOSS — Rusch and Rus, arXiv 2410.03943, v3

**Citation.** T. Konstantin Rusch, Daniela Rus, "Oscillatory State-Space Models", arXiv 2410.03943. Version read: v3
(2025-06-18), the latest; v1 is dated 2024-10-04 and v2 2025-02-12; the abs comment reads ICLR 2025 Oral. Read at
https://arxiv.org/html/2410.03943v3 (112,708 characters, in two passes), with the v1 and v2 HTML checked for
structure, tables and ablations; version list from https://arxiv.org/abs/2410.03943.

**Real versus complex, or oscillation off: no such arm, in any of v1, v2 or v3.** No experiment sets A = 0, uses real
eigenvalues, or runs a damped-only arm without oscillation. Every ablation keeps the oscillation on:

- Table 6 (§4.4, Appendix C.1): ReLU against squared parameterization of A (UEA average 67.8 against 67.1), and an
  initialization change (64.0, with EigenWorms at 74.4 ± 31.3).
- Figure 2 (§4.4, Appendix C.2): LinOSS-IM (dissipative) against LinOSS-IMEX (conservative) on a synthetic
  harmonic-motion task (2000/500/500 sequences, dt = 0.1, 1000 steps). The IM error grows over time and the IMEX error
  stays flat; §4.4 says IMEX ends more than 8× better. Both arms oscillate; this compares discretizations.
- Figure 3 (§4.4, Appendix C.3): a dt sweep from 1e-3 to 1 for LinOSS-IM on SCP1, Heartbeat and Motor, 5 seeds,
  variation described as not substantial, no numbers in the text.
- Appendix D: a velocity-Verlet scheme, derived and never evaluated.

v1 has only the ReLU-against-squared ablation (then Table 5, Appendix A.1).

**Tasks and numbers.** Six UEA classification sets from the Walker et al. 2024 long-range benchmark (EigenWorms
length 17,984; SCP1 896; SCP2 1,152; Ethanol 1,751; Heartbeat 405; Motor 3,000), PPG-DaLiA heart-rate regression
(length 49,920) and weather forecasting (720 in, 720 out). Sequential MNIST, permuted MNIST, sCIFAR and Long Range
Arena appear in no version.

- Table 1 (UEA test accuracy, average of the six): LinOSS-IM 67.8, LinOSS-IMEX 65.0, Log-NCDE 64.3, S6 62.0, S5
  61.8, LRU 61.7, NRDE 60.6, NCDE 60.2, Mamba 58.6. EigenWorms: LinOSS-IM 95.0 ± 4.4, LRU 87.8, S5 81.1, Mamba 70.9.
- Table 2 (PPG-DaLiA, test MSE ×1e-2): LinOSS-IM 6.4 ± 0.23, LinOSS-IMEX 7.5, Log-NCDE 9.56, NRDE 9.90, Mamba 10.65,
  LRU 12.17, S5 12.63, S6 12.88, NCDE 13.54. The v3 abstract rounds this to nearly 2× against Mamba and LRU; the
  table ratios are 1.66× against Mamba and 1.90× against LRU.
- Table 3 (weather MAE): LinOSS-IMEX 0.508, LinOSS-IM 0.528, S4 0.578, Informer 0.731, LSTnet 0.757, LogTrans 0.773,
  LSTMa 1.109, Reformer 1.575.

**Matched-ness.** No matched-structure control. The comparison models are separate architectures with widely
different parameter counts (Table 5: on EigenWorms LinOSS-IM 134,279 and LRU 101,129; on SCP1 LinOSS-IM 991,240, LRU
25,892, S5 226,328).

**The model, and why the lab's clamp is not its ablation.** By Proposition 3.1, LinOSS-IM's eigenvalues are
`(1 ± i·dt·√A)/(1 + dt²·A)`, with `|λ|² = S = 1/(1 + dt²·A)`. A is learned, diagonal and independent of the input;
dt is described as a hyperparameter (dt = 1 in the main runs). The reader derives from Proposition 3.1, and the
re-reader checks the algebra (trace 2S, determinant S); it is their derivation, not a sentence in the paper: per mode
the phase is `θ = arctan(dt·√A)` and the magnitude is `|λ| = cos θ`. Magnitude and phase are tied together by the
one parameter A, so in LinOSS-IM every bit of damping comes with oscillation. Driving θ to 0 forces `|λ|` to 1; at
A = 0 the mode is a repeated eigenvalue 1, an integrator, with no decay. The lab's CLAMPED arm (real, with a
learnable magnitude below 1) is not reachable inside LinOSS-IM or LinOSS-IMEX, and the FREE arm (magnitude and phase
independent) matches neither.

**The authors' conclusion.** As whole models, LinOSS beats LRU, S5, S6, Mamba and the NCDE family on the UEA average
(Table 1) and on PPG-DaLiA (Table 2), and S4, Informer and the other weather rows on weather (Table 3); LinOSS-IM is
stable for A ≥ 0 (Proposition 3.1), and the model is universal for continuous causal
operators (Theorem 3.3, a constructive proof). Its case for oscillation is stated with a hedge: forced harmonic
oscillators "seem to ensure expressive representations" (§1, Introduction). It is never tested by removing the
oscillation inside the model.

**Where the two reads differ.**

- *LinOSS-IMEX's modulus.* Reader: `|λ| = 1` for every A > 0. Re-reader: Proposition E.1 proves unit modulus only for
  `0 < dt ≤ 2/√A_kk` (the HTML prints a max over k where a min seems needed); above that bound the IMEX transition
  matrix (trace `2 − dt²·A`, determinant 1) has real eigenvalues, one of modulus above 1, a regime the unbounded ReLU
  parameterization can reach. Both: neither lab arm is reachable inside IMEX.
- *A_kk = 0.* Reader: the case is never discussed (while noting that the ReLU parameterization lets a dimension be
  switched off). Re-reader: §3.1 says ReLU lets dimensions be switched off completely, which is A_kk = 0, and Appendix
  C.1 says ReLU with a standard-normal init would switch off about half of them; what the paper never analyses is the
  dynamics at A_kk = 0. Both: never ablated.
- *The A = 0 matrix.* Reader: `[[1, dt], [0, 1]]`, no decay at all. Re-reader: in the paper's `[z, y]` ordering it is
  `[[1, 0], [dt, 1]]`, the same Jordan block transposed: a double integrator whose state can grow linearly, which does
  not match the paper's description of such a dimension as switched off. Both: repeated eigenvalue 1, no decay.
- *Table 6's initialization rows.* Reader: uniform against standard-normal initialization, all rows LinOSS-IM.
  Re-reader: the standard-normal row (64.0) uses the squared parameterization, so the clean initialization contrast
  is squared-uniform 67.1 against squared-normal 64.0.
- *Where the comparison numbers come from.* Reader: each comparison model tuned by its own grid search under the Walker
  et al. 2024 protocol (Appendix B.1). Re-reader: Appendix B.1 describes LinOSS's own search; §4.1 says all other
  results are taken from Walker et al. (2024); the weather rows cite their own papers.
- *The abstract by version.* Reader: nearly 2× against Mamba and LRU. Re-reader: that is v3; the v1 and v2 abstracts
  say nearly 2× against Mamba and 2.5× against LRU, with Table 2's LRU at 15.64 (2.44×); Mamba's 10.65 (1.66×) is the
  same in all three versions.
- *Table 5's label.* Reader: Table 5a. Re-reader: v3 renders Table 5 as one table. The numbers agree.

**Where both reads looked.** The abs page (v1-v3); the whole v3 HTML: Abstract, §1, §2.1-2.4, §3.1 (Proposition 3.1,
Remark 1, the parameterization paragraph, Proposition 3.2), §3.2 (Theorem 3.3), §4.1-4.4, §5, references,
Appendices A, B.1-B.2, C.1-C.3, D, E.1-E.3; Tables 1-6, the complete list; Figures 1-3, the complete list. v1 and v2:
headings, tables, figures. Text searches for MNIST, CIFAR, LRA, ListOps, Pathfinder, "real eigenvalues" and S4D-Real
found none.

## 4. D-LinOSS — Boyer, Rusch and Rus, arXiv 2505.12171, v2

**Citation.** Jared Boyer, T. Konstantin Rusch, Daniela Rus, "Learning to Dissipate Energy in Oscillatory
State-Space Models", arXiv 2505.12171. Version read: v2 (2025-09-30), the current one; v1 is dated 2025-05-17 and was
read for structure. The abs comment gives 18 pages and 4 figures and names no venue. Read at
https://arxiv.org/html/2505.12171v2, with https://arxiv.org/html/2505.12171v1 and the version list at
https://arxiv.org/abs/2505.12171.

**Real versus complex, or oscillation off: no such arm, in v1 or v2.** Nothing sets A = 0, zeroes a frequency or a
phase, or sets a matched real-diagonal arm against an oscillatory one. The nearest items:

- Table 1 (§3.2, Appendix B.1): LinOSS-IMEX, LinOSS-IM and D-LinOSS learning a synthetic exponential decay with one
  real eigenvalue of 0.8 (length 1000, 3 seeds; Appendix B.1 writes the system as A = 0.8, and the paper does not
  reconcile the two). RMSE as v2's header prints it (×1e-3): IMEX 24.5 ± 2.3, IM 8.0 ± 1.7, D-LinOSS 0.8 ± 0.1. All
  three arms are oscillator-family models; the variable is damping and discretization.
- Figure 2 (§4.4, Appendix B.2; v2 only): the adding task at lengths 500, 2000 and 5000. D-LinOSS reaches validation
  MSE 1e-2 about 1.7× faster than LinOSS-IM at 500 and 2000 and about 1.4× at 5000; LinOSS-IMEX does not beat the
  0.167 random-guess level.
- Figure 4 (Appendix B.5): four initialization schemes within D-LinOSS; radially uniform `|λ|` with uniform phase
  and `r_min = 0.9` does best. The phase is drawn over [0, π); no scheme confines it near zero.

**Tasks and numbers.** No psMNIST, sMNIST, sCIFAR or Long Range Arena in either version.

- Table 2 (UEA, average): D-LinOSS 68.0, LinOSS-IM 67.8, LinOSS-IMEX 65.0, Log-NCDE 64.3, S6 62.0, S5 61.8, LRU 61.7,
  NRDE 60.6, NCDE 60.2, Mamba 58.6. D-LinOSS against LinOSS-IM by dataset: EigenWorms 93.9 / 95.0, SCP1 88.9 / 87.8,
  SCP2 58.6 / 58.2, Ethanol 29.9 / 29.9, Heartbeat 75.8 / 75.8, Motor 61.1 / 60.0.
- Table 3 (PPG-DaLiA, MSE ×1e-2, though the caption says accuracies): D-LinOSS 6.16 ± 0.73, LinOSS-IM 6.4 ± 0.23,
  LinOSS-IMEX 7.5 ± 0.46, Log-NCDE 9.56, NRDE 9.90, Mamba 10.65, LRU 12.17, S5 12.63, S6 12.88, NCDE 13.54.
- Table 4 (weather MAE, no variance): D-LinOSS 0.486, LinOSS-IMEX 0.508, LinOSS-IM 0.528, S4 0.578, Informer 0.731,
  Informer† 0.741 (re-reader; third look), LSTnet 0.757, LogTrans 0.773, LSTMa 1.109, Reformer 1.575. The reader
  lists nine rows and omits Informer†.
- Table 6 (compute): D-LinOSS and LinOSS-IM have the same parameter count on EigenWorms (134,279) and different
  counts on Ethanol (71,112 against 6,728), Heartbeat (4,356 against 10,936) and Motor (20,598 against 91,844).

**Matched-ness, and the lab's arm.** No matched structure and no oscillation-off arm. Each D-LinOSS mode is a damped,
forced second-order oscillator (`x'' = −A·x − G·x' + B·u`, A and G diagonal and non-negative), discretized
implicit-explicitly into a 2×2 block per mode. Proposition 3.2 gives every mode modulus `1/√(1 + Δt·G)`; the §3.3
parameterization clamps A between bounds derived from the stability inequality, which keeps the modes in
complex-conjugate pairs; Proposition 3.3 says the reachable set is the closed unit disk, and Proposition 3.4 that
LinOSS's sets have measure zero. The reader derives, and the re-reader's own algebra of the clamp bounds confirms (it
is not a sentence in the paper), that A = 0 forces G = 0, which gives the integrator with a repeated eigenvalue 1. So
"frequency off" in D-LinOSS's own parameterization is not the lab's CLAMPED decaying real mode. In its eigenvalues,
the lab's arm corresponds to the θ = 0 slice of D-LinOSS's disk, which the paper never isolates; whether it
corresponds in dynamics is under "Where the two reads differ".

**The authors' conclusion.** A learnable damping matrix G decouples dissipation from frequency and lets the
eigenvalues fill the unit disk. D-LinOSS scores slightly better than LinOSS-IM on all three real-data benchmarks (UEA
average accuracy 68.0 against 67.8; PPG-DaLiA MSE 6.16 against 6.4, lower is better; weather MAE 0.486 against
0.528, lower is better), with most or all of the UEA per-dataset gaps inside the reported standard deviations (the
reads differ, below), and it converges faster on the adding task. Every comparison is between whole models or
within the LinOSS family (damping, discretization, initialization). Section 6 asserts that "D-LinOSS is inherently
well-suited to represent temporal relationships with oscillatory structure", with no ablation behind it, names the
time-invariant layer as a limitation, and leaves selective variants to future work.

**Where the two reads differ.**

- *Table 4's rows.* Reader: nine. Re-reader: ten, the tenth being Informer† at 0.741 (also in v1); a third look in
  review confirmed it in v2.
- *The θ = 0 slice.* Reader: the lab's arm corresponds to it. Re-reader: inside the §3.3 clamp that slice is
  critically damped, a defective 2×2 Jordan block whose impulse response carries both r^n and n·r^n terms, so it
  matches the lab's single real decaying mode in eigenvalue, not in dynamics (the re-reader's derivation from the
  §2.2 blocks).
- *Table 6's parameter matching.* Reader: matched on EigenWorms only, of the four datasets it lists. Re-reader: also
  SCP1 (992,776 against 991,240) and SCP2 (399,496 against 399,112), each difference being state dimension times
  blocks, that is the added G parameters at the same configuration; and the EigenWorms row is identical in all three
  columns, which Table 5's configuration would not give (the re-reader's inference).
- *Where the comparison numbers come from.* Reader: most are carried from prior work. Re-reader: §4.1 says the
  LinOSS-IM and LinOSS-IMEX scores come from Rusch and Rus 2025 and all others from Walker et al. 2024, so every Table
  2 row except D-LinOSS is copied; the only head-to-head LinOSS runs are Table 1 and Figure 2.
- *The UEA gaps.* Reader: most are inside the reported standard deviations. Re-reader: all six are within one
  standard deviation.
- *Citations of the LRU paper.* Reader: only for random initialization matching HiPPO (§5). Re-reader: §3.4 cites it
  twice more, and Table 3 cites its LRU row.
- *Real eigenvalues.* Reader: a decaying real eigenvalue exists only as a critically damped double root. Re-reader:
  that holds inside the §3.3 clamp; the oscillator equation itself admits overdamped modes with two distinct real
  eigenvalues, which the clamp excludes, and Proposition 3.2's condition is sufficient, not necessary (Appendix A.2).
- *The A bounds.* Reader: their exact form is not given. Re-reader: the paper names them only; its own algebra gives
  `L = (√(1 + G·Δt) − 1)²/Δt²` and `U = (√(1 + G·Δt) + 1)²/Δt²`, so L = 0 at G = 0.
- *Which comparison models are real- or complex-diagonal.* The reader labelled S6 and Mamba real-diagonal and LRU and
  S5 complex-diagonal; the re-reader finds no such statement in the paper. This NOTE drops the labels.

**Where both reads looked.** The abs page; the whole v2 HTML: §1, §2.1-2.3, §3.1-3.4, §4.1-4.4, §5, §6; Tables 1-6;
Figures 1-4 (captions and text; Figures 2 and 4 give no numbers); Appendices A.1-A.5 and B.1-B.5, B.5 being the last
heading. v1: outline, tables, figures. Text searches for MNIST, psMNIST, "permuted", CIFAR, LRA, "real eigenvalue" and
"non-oscillatory" found none.

## 5. What this means for the lab's sentences

Read together, the four papers give this:

- **A real-versus-complex contrast inside one diagonal SSM is published in S4D (2022)** (Table 3(a), S4D-Real
  against S4D-Inv at matched size, on sCIFAR, Speech Commands and BIDMC). It is an initialization contrast that also
  moves the real part, not a clamp, and the two reads differ on how much it says about oscillation (§2). The LRU
  paper has no real arm. LinOSS and D-LinOSS have no oscillation-off arm.
- **The lab's own part is a measurement, stated as one.** In a three-block LRU-style complex-diagonal recurrence
  (`CLRU` in `run_pmnist_ablation.py`), the phase θ was held at 0 for the whole of training while the magnitude
  parameterization and initialization stayed unchanged, on permuted MNIST with permutation 1234: a gap of +0.3122 at
  4000 steps on two seeds (the July receipt), and 0.1097 at 12,000 steps under the sealed re-run (the robustness
  RESULT). None of the four papers read here has permuted MNIST, and none reports an arm that holds the phase at zero
  through training with the magnitude parameterization and initialization unchanged. That describes four papers. It
  is not a statement about the literature: TIDES, S5 and the PAPER's other related work are unread.

### `RESULT_pmnist_ablation_2026_07_23.md` (the July RESULT; `CERT_pmnist_ablation.json` names it)

- **Stands, as a measurement:** l.25, "`FREE − CLAMPED = +0.3122`. Matched-param (101,002 vs 100,810) and
  RNG-matched; the single knob is θ." No read paper has permuted MNIST, so none confirms or refutes the number. Its
  forward pointer to the robustness RESULT is owed, and open PR #226 (branch `papers/pmnist-robustness-pointers`, not
  merged when this NOTE was written) adds it as a NOTE beside the July RESULT. "Matched-param" is approximate: in
  `run_pmnist_ablation.py` l.55-58, FREE registers θ as a parameter and CLAMPED as a buffer, so FREE trains 192 more
  parameters (64 modes × 3 blocks), which is the l.25 difference itself.
- **Known wrong, needs a banner:** l.6, "`run_pmnist_ablation.py` (deep S5/LinOSS-class classifier", and l.49-51,
  "Combined with Appendix A of the paper (the phase-clamp is LinOSS's oscillation ablation by the math), this is a
  causal statement about the LinOSS/S5 class on real long-range data that the whole-architecture benchmarks cannot
  make." The LinOSS half fails by LinOSS's own Proposition 3.1: magnitude and phase are tied (§3), so the CLAMPED
  arm is not reachable inside LinOSS, and D-LinOSS does not change that (§4). The architecture is LRU-style, as both
  LRU reads find. The S5 half was not read. Within-architecture real-versus-complex comparisons exist in the
  literature (S4D, §2), so "that the whole-architecture benchmarks cannot make" needs this pointer beside it.
- **Exceeds every read, needs the same banner:** l.15-16, "This is the sharpest confirmation that oscillation is
  what makes an SSM a long-range model."; l.39, "the oscillation *is* the long-range mechanism"; l.47-48, "load-bearing
  for long-range modeling itself". The measurement is one architecture, one permutation, two seeds and 4000 steps,
  and the lab's own re-run reads `BUDGET_CONFOUND__gap_does_not_survive_3x_steps`. None of the four papers supports
  the general sentence. The LRU paper's one phase experiment on a long-range task points elsewhere: PathX needed
  the phase initialized in a thin slice near zero, with θ still learnable and γ normalization on (§1). That neither
  confirms nor refutes the July gap, and it does not support the sentence.

### `PAPER_oscillation_causal_map_2026_07_23.md` (draft 2026-07-23)

- **Stands, with version pins:** l.64-65, "LinOSS (2410.03943) and D-LinOSS use *static* (fixed-after-training)
  frequencies and report no within-architecture oscillation-vs-decay ablation." Both reads of each paper agree (LinOSS
  v1-v3; D-LinOSS v1-v2). l.195-196, "the within-architecture ablation the LinOSS paper does not report", is
  literally true for the same reason, but see Appendix A below: the lab's clamp is not LinOSS's ablation.
- **Stands as a paraphrase of LinOSS's abstract, needs the table beside it:** l.34-35, "LinOSS (Rusch & Rus, ICLR
  2025) builds a bank of forced harmonic oscillators and beats Mamba ~2× on 50k-length sequences". Table 2 gives
  1.66× against Mamba (§3).
- **Contradicted in part as general sentences, needs a banner:** l.9-10, "Yet the field evaluates them by comparing
  whole architectures (e.g. LinOSS vs Mamba), which cannot isolate what the oscillation *itself* contributes";
  l.36-38, "But *why* they work — whether the oscillation is causally essential or an incidental feature of a good
  architecture — is untested, because the standard comparison is between whole models."; l.38, "We supply the
  missing causal instrument". S4D's Table 3(a)
  is a within-architecture real-versus-complex comparison at matched size from 2022, and the LRU paper compares
  parameterizations within the complex diagonal family. These sentences are true of LinOSS and D-LinOSS, not of the
  field.
- **True as written, implies more than it says, needs the S4D pointer:** l.91-92, "This is the controlled
  single-knob within-architecture ablation the whole-model benchmarks cannot run." Whole-model benchmarks cannot run
  a within-architecture ablation, so the sentence is literally true. What it implies, that no within-architecture
  real-versus-complex comparison exists, does not hold: S4D published one (§2).
- **Priority-shaped, not supportable as written, should not be printed:** l.72-73, "Across all of these, the move no
  one makes — clamp the oscillation off, hold everything else fixed, measure Δ — is ours." It is shaped as a
  priority claim, four reads cannot carry a universal sentence, and no read supports it. Its nearest relative in the
  four papers is S4D-Real, which starts the imaginary parts at zero, holds architecture and protocol fixed and
  reports Δ. S4D-Real is an initialization rather than a phase held at zero through training, and it moves the real
  part too, so it is not the move as the sentence states it; nor do the four papers show anyone making that move.
  What separates the lab's arm (θ held at 0 through training, magnitude parameterization and initialization
  unchanged) from S4D-Real is a design difference, not priority. The PAPER's two ordinals belong here too and
  are filed the same way: l.32-34, where the intuition "becomes, for the [ordinal] time, a *controlled
  experiment*", and the ordinal adjective l.39 puts on the lab's within-architecture map (ordinal elided in both
  quotes). S4D's Table 3(a) (2022) is a controlled real-versus-complex comparison inside one model, and the LRU
  paper's PathX phase-range runs (2023) are controlled phase experiments. The Related Work section (l.62-73) also omits the
  LRU paper, whose magnitude parameterization (`nu = log(−log r)`) and ring initialization `CLRU` uses, and S4D.
- **Known wrong, needs a banner:** l.47, "**FREE**: θ learnable → the oscillatory (S5/LinOSS-class) model.", and
  l.79, "A deep S5/LinOSS-class classifier", for the reason given under the July RESULT.
- **Known wrong, needs a banner on Appendix A:** l.173, "## Appendix A: the phase-clamp is LinOSS's oscillation
  ablation (equivalence)"; l.176-177, "so the flagship result is a statement about LinOSS itself, not a proxy";
  l.192-193, "Clamping the phase `θ → 0` (our FREE→CLAMPED knob) is identical to clamping the restoring force `A → 0`".
  The appendix's algebra holds: its `tr = 1 + S − Δt²AS` simplifies to `2S`, with `det = S`. Its own l.189-190 says
  that at A = 0 the mode is "a pure **integrator, no rotation**", with magnitude 1, while CLAMPED keeps a learnable
  magnitude `exp(−exp(nu)) < 1` (`run_pmnist_ablation.py` l.65). In LinOSS-IM, `|λ| = √S = cos θ`, so the
  knob moves magnitude with phase. "Identical" and "not a proxy" do not follow.
- **Exceeds every read, needs a banner:** l.18-19, "the oscillation *is* the long-range mechanism", and
  l.106-107, "On genuinely long-range data the oscillation is not merely helpful — it *is* the long-range mechanism;
  a decay model of equal budget cannot substitute for it.", for the reason given under the July RESULT.
- **Outside every read:** l.15 and l.89 on sequential MNIST ("causally load-bearing"). No read paper has sMNIST.

### `papers/INDEX.md` row 51

"The single-knob phase clamp shows oscillation is causally load-bearing in SSMs (+0.3122 on permuted MNIST); every
proposed extension — adaptive frequency, nested coupling, real-model transfer — died." Status "LIVE (extensions
NEGATIVE)", as the row reads on origin/main `2e218601`.

- "in SSMs" exceeds the measurement: one LRU-style architecture. No read paper supports the general sentence: S4D
  shows a real initialization losing to S4D-Inv at matched size (the reads differ on what that says about
  oscillation, §2), the LRU paper has no real arm, and LinOSS and D-LinOSS have no oscillation-off arm.
- The row can carry no priority, implied or stated, because a real-versus-complex contrast within one diagonal SSM
  is published in S4D (2022).
- What the row needs is a named architecture ("a three-block LRU-style complex-diagonal recurrence, `CLRU`"), and
  pointers to the robustness RESULT, which says the row's status changes with its verdict (l.449-450), and to this
  NOTE. The INDEX "carries no finding" (its l.3-4), so the edit is cheap; this NOTE does not make it.
- Open PR #226 (branch `papers/pmnist-robustness-pointers`, not merged when this NOTE was written) rewrites this
  row: it names the architecture as LRU-style `CLRU`, drops "in SSMs", points to the robustness RESULT and its
  verdict, and sets the status to SUPERSEDED. It does not point to this NOTE.

### `RESULT_pmnist_robustness_2026_10_23.md`

- **Stands:** l.431-433. Its naming, "LRU-style; `CLRU` in `run_pmnist_ablation.py`", holds under both readings of
  how much of the LRU `CLRU` shares (§1). "No priority claim is made" stands, as does l.423's "on priority, or on
  what the lab added that others did not". The two reads it owes are done here. The LinOSS and D-LinOSS reads that
  `ATTACK.md` §5 adds are done too, and TIDES is not.
- **No edit:** that RESULT is byte-pinned by its sidecar, so this NOTE stands beside it. A forward pointer, if one is
  wanted, is the operator's call.

### `ATTACK.md` §5 (outside this repository), for the record

- The LRU recollection ("Λ real" against "Λ complex" on LRA): refuted by both LRU reads.
- The S4D recollection holds in substance, with three corrections. The real row is set against S4D-Inv only (the
  S4D-Lin block has none). The tasks are sCIFAR, Speech Commands and BIDMC, not LRA. S4D-Real moves the real part too.
- "predates the lab's July clamp by one to two years": S4D's v1 (2022-06-23) is about four years before 2026-07-23,
  and the LRU paper (2023-03-11) about three years and four months.
- Its conditional ("If either recollection holds, ... the knob itself is not new") holds through S4D, in this form.
  A real-versus-complex initialization contrast inside one architecture at matched size is published. The specific
  form the lab ran, phase held at zero for all of training with the magnitude parameterization and initialization
  unchanged, is not reported in any of the four papers read.

### Other lab files the reads touch (no edit made)

- `PREREG_smnist_ablation_2026_07_23.md` l.4 ("The oscillation-vs-decay causal ablation LinOSS never published, on a
  REAL long-range benchmark."), `run_smnist_ablation.py` l.5, `fig_flagship_sharpening.py` l.18 ("the test LinOSS
  never published") and `fig_smnist_flagship.py` l.17 ("the oscillation-vs-decay ablation LinOSS never published"):
  literally true, since LinOSS v1-v3 and D-LinOSS v1-v2 have no such arm. What they imply, that the lab ran LinOSS's
  ablation, is wrong for the Appendix A reason. The PREREG is frozen, so a pointer is the route.
- `run_pmnist_ablation.py` l.8 docstring, "deep S5/LinOSS-class classifier": the same mislabel.
- `FRONTIER_due_diligence_2026_06_04.md` l.21 gives arXiv 2505.12171 the title "Oscillatory State-Space Models", and
  l.58 lists it as "(oscillatory SSMs)". 2505.12171 is D-LinOSS; that title is LinOSS's (2410.03943).
- `POSITIONING_entrainment_lit.md` l.19 ("~2× Mamba, ~2.5× LRU on a 50k-length task") matches LinOSS's v1 and v2
  abstract, per the re-reader; the v3 abstract says nearly 2× against both, and Table 2 gives 1.66× and 1.90× (§3).
  l.20 credits D-LinOSS with a "cell-swap ablation vs S5/LRU/Mamba"; the paper presents those as comparison rows,
  not as an ablation, with their scores taken from prior work (most of them or all of them; the reads differ, §4).
  The l.51-53 residual (read TIDES and D-LinOSS "to confirm neither runs a slow-tracking frequency loop on drifting
  periods") is discharged for D-LinOSS, whose layer is time invariant (§4), and not for TIDES.

## 6. What was not read

- **TIDES** (arXiv 2605.09742), which `ATTACK.md` §5 lists as an owed read and which the PAPER (l.65-67) and
  `POSITIONING_entrainment_lit.md` (l.21, l.51-53) cite. Its half of the POSITIONING residual stays open.
- **Le, Jaitly and Hinton**, arXiv 1504.00941, which `ATTACK.md` §5 cites for the one-fixed-permutation psMNIST
  convention. The permutation-averaging search stays where `ATTACK.md` §7 left it, at abstract level.
- **The S5 paper**, the "S5" in the lab's "S5/LinOSS-class" label. Nothing here says whether `CLRU` is S5-class.
- **The PAPER's other related work**: "Tuning Frequency Bias of SSMs" (2410.02035), AOSNET (2606.06010), AKOrN
  (2410.13821) and "Task-Level Insights from Eigenvalues" (2510.09379).
- **Within the four papers:** no PDF and no released code; whatever only one of the two reads reports; and the
  plotted values of figures that give no numbers in the text (LRU Figures 1 and 5, LinOSS Figures 2-3, D-LinOSS
  Figures 2 and 4).
