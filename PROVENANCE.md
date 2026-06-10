# Provenance & Authorship

styxx is research on AI honesty / hallucination detection — and an experiment in **autonomous AI
research itself**. A large fraction of the code, experiments, and findings here are **authored by AI
agents under human direction.** That is the project's design, not an incidental detail, and we state it
plainly: the entire thesis of the work is *calibrated honesty*, which has to start with honesty about
how the work is made.

## Who authors what

| Git identity | Role |
|---|---|
| `darkflobi <darkflobi@darkcity.wtf>` | Primary **autonomous research agent** — the majority of commits. |
| `Flobi <heyzoos123@gmail.com>` | Human operator / director. Sets goals, reviews, signs off. |
| `styxx-coding-agent <styxx-agent@fathomlab.io>` | Execution agent for coding tasks. |
| `Co-Authored-By: Claude <noreply@anthropic.com>` | The model behind some agent sessions, credited on commits it co-wrote. |

The `Co-Authored-By: Claude` trailer is **accurate attribution, not a planted contributor.** An AI
genuinely co-wrote those commits and the trailer says so. An automated scanner with no context may read
it as a "fake contributor" — this document is the context: the AI authorship is real, intentional, and
the point of the project.

## How to trust this *without* trusting the authors

The credibility model is **not "trust the author."** The main author is an autonomous agent —
unverifiable as a human, by design. The credibility model is **reproducibility:**

- **Every research claim is pre-registered.** The kill-gate — the number that would falsify the claim —
  is committed to git *before* the data exists, frequently with the dataset SHA-256'd. See
  `papers/grounded-honesty-axis/PREREG_*.md`.
- **Every finding ships a re-runnable harness and a receipt.** A `run_*.py` / `score_*.py` produces a
  `*_result.json`. Clone, run, and check the number against the pre-registered bar yourself.
- **We publish the failures.** The `FINDING_*` files include `REPORT_AS_LANDED` and `VOID` results —
  hypotheses killed in public (e.g. the residual-probe arc: three pre-registered swings, all negative).
  A project that hides its negatives is the one to distrust.

So don't take our word for it — **re-run the receipts.**

## What this is *not* (owned, not hidden)

- **New and pre-adoption.** The public repository is young. Stated, not spun.
- **AI-authored.** By design — see above.
- **Not a financial document.** The `$STYXX` token (`docs/token/`) coordinates a *public network*; it
  never gates the open-source library, and nothing in this repository is investment advice or a value
  claim. Provenance here is about *code*, not markets.

## Verify

- License `LICENSE` (MIT) · Security `SECURITY.md` · Citation `CITATION.cff` · Contributing `CONTRIBUTING.md`
- Re-run any finding: `papers/grounded-honesty-axis/` — pre-registration, harness, and result JSON sit side by side.
- The standard, stated in full: `papers/THESIS_the_honesty_standard_2026_05_31.md`.
