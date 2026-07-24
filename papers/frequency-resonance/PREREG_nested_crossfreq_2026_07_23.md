# PREREG — NEST-OSS: nested cross-frequency (theta-gamma) coupling kill-gate — 2026-07-23

**FROZEN before any confirmatory data** (smoke/plumbing only so far). Runner: `run_nest_capacity.py`.
1× RTX 4070 Laptop, torch 2.5.1+cu121. Governs the theta-gamma moonshot in
`~/.claude/plans/sunny-toasting-toast.md`.

## Question

Neuroscience's working-memory code is **theta-gamma cross-frequency coupling**: a slow theta rhythm
frames fast gamma sub-cycles, each a memory SLOT holding one ordered item at a distinct phase (the
7±2 bound; Lisman & Jensen 2013). **No state-space model implements true cross-frequency coupling** —
they use a FLAT bank of independent oscillators. Decisive question, frozen:

> Does explicit nested coupling — a slow clock that gates fast modes into ordered phase-slots — beat a
> flat oscillatory bank of equal budget at holding multiple ordered items (the theta-gamma function)?

## Task (frozen)

Ordered copy: present K symbols, then K GO tokens, recall in order at the GO slots. Mixed K per step,
`KGRID=[2,3,4,5,6,8,10]`. Metric: mean recall accuracy over K (seed-averaged, `EVAL_N=512`, 3 seeds
`[0,1,2]`, 3000 steps), and kcap = largest K solved at ≥0.80.

## Arms (frozen)

- **FLAT** — free oscillatory bank, learnable θ per mode, **uniform** input (the LinOSS-style
  competitor; the baseline to beat).
- **NEST** (the invention) — same modes; input to mode j gated by a slow clock
  `g_j(t)=(1−α)+α·0.5·(1+cos(ω_slow·t−ψ_j))`, learned `ω_slow, ψ_j, α`. Items at different slow-phases
  route to different modes = phase-multiplexed slots. **Single knob:** `α=0` reduces NEST to FLAT
  **bit-for-bit** (red-team verified `max|Δ|=0.0`; nest-only params drawn after the read head).
- **WIDE** = FLAT at **2× modes** — the **capacity-headroom positive control**. More modes provably
  buy more capacity; if WIDE does not beat FLAT, the task has no headroom and the test ABSTAINS.
- **ORACLE** (per-item temporal slotting: item t → mode t mod d) — **diagnostic only, NOT the positive
  control**: smoke found hard input-gating is *lossy* (it destroys the flat bank's distributed
  encoding), so it is reported to show that slotting-via-gating is the wrong mechanism, not to gate.
- **CLAMPED** (θ≡0, decay) floor; **TRANSFORMER** (attention) context, **not** gated.

Matched-param FLAT vs NEST (differ only by `ω_slow, ψ, α` = d+2 params; exact counts reported) and
matched-compute (same scan recurrence). Recurrence via the parallel `lin_scan`, red-team-verified
`scan==seq`.

## Frozen kill-gate (primary D=8; sweep D∈{8,16})

`adv = NEST−FLAT`, `head = WIDE−FLAT` (positive control), on primary-D mean accuracy.

- **ABSTAIN** iff `head < 0.05` — doubling modes buys no capacity, so the task cannot detect a nesting
  gain. No conclusion; redesign (larger K / smaller D) and re-freeze. (The "positive control must fire"
  rule — no verdict from a test that cannot see the effect.)
- **GREENLIGHT** iff `head ≥ 0.05` **and** `adv ≥ 0.05` **and** `adv ≥ 0.5·head` (nesting gives a
  d-mode bank ≥half the capacity a 2×-wider bank buys). → scale-up.
- **KILL** iff `head ≥ 0.05` **and** `adv < 0.02` — headroom exists but explicit nesting does not beat
  the flat bank's own implicit phase multiplexing (the flat oscillatory bank is already at the
  theta-gamma ceiling). → honest negative.
- **WEAK** otherwise — real but sub-threshold; not a greenlight.

## Honest prior & discipline

This arc's own multiplexing result ([[project_frequency_resonance_2026_06_04]]) found a flat
oscillatory bank already realizes a **constant ~1.8× phase-multiplexing** multiplier, so the honest
prior is **NEST ≈ FLAT → KILL**, with the finding being "the flat bank is already at the theta-gamma
ceiling; explicit coupling is the redundant thing no SSM implements *because* it is redundant." The
frozen gate, a firing positive control, and no post-hoc selection decide it. Result → `nest_capacity_result.json`
+ `RESULT_`, grounded by `python -m styxx.certify` (expect OATH-HELD).
