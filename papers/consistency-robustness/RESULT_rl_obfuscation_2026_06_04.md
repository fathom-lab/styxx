# RESULT — the probe resisted; "consistency > signature" is NOT supported — they have complementary surfaces

**Date:** 2026-06-04 · **Verdict: INCONCLUSIVE** (the owed RL-obfuscation arm). Frozen:
`PREREG_rl_obfuscation_2026_06_04.md`. Qwen2.5-1.5B-Instruct, adversarial LoRA (L_task + λ·L_evade +
λ·L_preserve).

## Numbers
- Probe AUROC: **1.000 → 1.000** (P1 FALSE — did NOT collapse). `L_evade → 0.000` during training (the
  probe was fooled on the *training* residuals) but the evasion **did not generalize** to held-out items.
- Still deceptive: 52% lying (P2 TRUE). Stateless belief 96%, grounded_honesty AUROC **0.966** (P3 TRUE)
  — the `L_preserve` term kept the attack surgical (the model stayed capable + deceptive).

## What it means (honest, and it revises the thesis)
The gradient-LoRA evasion **overfit the training residuals and the probe stayed robust on held-out data.**
This is consistent with *every* probe attack tried today:
- learned minimal per-item perturbation — did not collapse it (cost 76%);
- partial steering — only the maximal α=1 shift collapsed it;
- adversarial LoRA — evaded on-train, did not generalize.

**The activation probe resisted all my feasible attacks.** Meanwhile the *consistency* check
(grounded_honesty) is the one I actually **cracked** (18% via question-framing). So **in these
experiments the signature was MORE robust than the consistency check** — the opposite of the
"consistency > signature" thesis I started with. I'm not going to keep that thesis as stated.

## The mature conclusion: complementary surfaces, defense-in-depth
The two signals fail to **different** attacks, and neither dominates:
- **Activation probe (signature):** robust to representation perturbation / gradient evasion here;
  the literature collapses it with heavier *full RL-obfuscation* (arXiv 2505.27958), which I could not
  reproduce with a feasible gradient-LoRA. Its real weak point is documented but expensive.
- **grounded_honesty (consistency):** immune to representation attacks by construction, but has a
  **belief-level** weak point — cheap question-framing fools it 18% (and the defense closes it).

The honest, useful answer is **defense-in-depth**: pair a signature probe + a consistency check +
injection detection; they cover each other's surfaces. "One beats the other" was too clean.

## Honest scope
- Gradient-LoRA ≠ full RL-obfuscation; one model, one probe layer, one λ schedule. A *negative* on
  cheap probe-collapse, NOT proof the probe is unbreakable (the literature breaks it with more compute).
- What stands, fully: grounded_honesty is **attack-tested and defended, cross-vendor** (the strong,
  complete result). What does NOT stand: my framing of it as *beating* signatures.
