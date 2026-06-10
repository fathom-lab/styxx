# FINDING — B24: a pre-emission "commitment-to-the-user's-answer" direction reads silent caving (REPORT_AS_LANDED; auto-REFUTED killed as overclaim #5)

**2026-06-09. Fathom Lab / styxx. Closed-model frontier (R4). Qwen2.5-3B-Instruct, n=108 (57 HELD / 71
CAVED), silent bare-term regime, seed 0, local/$0.** Pre-reg frozen pre-result at `9d3291c`
(`PREREG_b24_whitebox_vs_behavioral_2026_06_09.md`); receipts `b24_headtohead_result.json` +
`b24_controls_addendum.json`; adversarial verification `wf_96778321-0b8` (4 lenses + adjudication).

## Headline (at true strength)

On silent bare-term caves — where the text tells are dead (text-syc **0.500** = exact chance, text-dec 0.512,
replicating B22) — a home-field white-box linear probe reads the impending cave **pre-emission** at the commit
slot: **AUROC ~0.94** [boot 0.89–0.98] at (POS-A, L29), surviving the **non-vacuous first-character firewall
(0.935)**, beating the logit-lens answer-token decoder at that depth (0.60) and surface confidence (0.69),
selection-corrected max-perm null p<0.01. The strictly-pre-commit read (POS-B, before the assistant scaffold,
lens flat at chance at every layer) is **0.838** — real but in the TIE band. **Behavioral grounding stays
strictly stronger (1.000, part-definitional) at every position: the white-box never beats it.**

## The auto-verdict was killed (the discipline's fifth kill)

The frozen gate auto-fired **REFUTED**. The red-team killed it as an **overclaim** — the fifth auto-verdict
overclaim killed in this program (v1, v2, v4, v5, now B24):

1. **The pre-registered firewall was vacuous.** All 108 (xid,yid) answer-token pairs are unique → GroupKFold
   groups are singletons → it *is* item-level CV, and `WB_collapse ≈ 0` was **forced by design**, not found.
   The load-bearing invariance evidence is the **first-CHARACTER firewall** (26 real groups; held-out items
   share no committed first letter with training): **0.935 POS-A / 0.841 POS-B** — independently regenerated
   from the frozen residuals by the verifier, now persisted in `b24_controls_addendum.json`.
2. **The 0.90 bar was grazed, not cleared** (boot lower 0.892 < 0.90, best-of-74-cells) — the pre-reg's own
   rubric routes a borderline single-cell max to REPORT_AS_LANDED, never REFUTED.
3. **POS-A is the commit slot** (the slot whose logits sample the first answer token) — pre-emission /
   pre-unembedding, but *not* strictly pre-commit. The only strictly-pre-commit number is POS-B's 0.838,
   which fails two of three REFUTED conditions. The POS-A edge over POS-B (+0.12) is localized to L28–30,
   exactly where the logit-lens lifts off the floor — consistent with reading an early distributed form of
   the crystallizing answer representation.
4. **Construct: not "fold-intent."** On single-token collinear items (caving ≡ which answer token; zero
   decoupled items), "intent to fold" and "committing to the user-asserted answer" are the same partition.
   An answer-abstract length/format code is still decodable *within* intent strata at L29 (0.82/0.93). The
   honest name is a **pre-emission commitment-to-the-user's-answer direction**, not a mens-rea read.

## What IS established (all disk-verified)

- The signal is real and non-trivial: depth-dissociation (committed-answer content/length code saturates by
  L4 at 0.95 while the cave label is near chance 0.63 there, ramping to 0.94 only by L29 on a deep plateau
  L27–31); length-only anti-predicts (0.353); first-char firewall holds; selection-corrected max-null
  p95=0.715 vs observed 0.941 (0/120 perms ≥ observed).
- **What is refuted is the PREDICTION, not the thesis**: the disk prior predicted Outcome-D (the white-box
  read on silent single-token caves collapses to the answer token). It does not collapse — the residual
  stream carries the impending cave above the answer-token decoder.
- **The closed-model thesis is reinforced, not refuted**: behavioral 1.0 > white-box 0.94 everywhere; BOTH
  detect silent caving (defense-in-depth), and behavioral is the one that needs no weights.

## Scope (loud)

One model (Qwen2.5-3B-Instruct), one regime (silent bare-term sycophantic caving), one seed, n=108,
single-token collinear items (the core unresolvable confound *on this item geometry*), best-of-74-cells
headline with selection-corrected null, behavioral g part-definitional (consumes the committed answer) and
never the headline. Not cross-vendor, not true-API.

## Owed / spawned

- **B25 (decisive next bet): intent-decoupling item set** — multi-token answers or user-asserted Y =
  paraphrase/synonym of correct X, so commit-to-user's-answer ≠ commit-to-wrong-content gives non-zero
  decoupled items. Run the same POS-B grouped probe: survives ⇒ fold-intent; collapses ⇒ it was the
  answer-commitment code. The only design that breaks the collinearity.
- B23-F true-closed-substrate arm (running at time of writing), 50-seed stability, answer-abstract-subspace
  ablation at POS-B.