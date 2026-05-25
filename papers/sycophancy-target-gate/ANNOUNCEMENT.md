# Announcement copy — styxx 7.5.0 + 7.6.0 (sycophancy gate)

Every number below is from a hashed, run-once holdout in this directory. Links verified live.

**Links**
- pip: `pip install -U styxx`
- release 7.6.0: https://github.com/fathom-lab/styxx/releases/tag/v7.6.0
- receipts (this trail): https://github.com/fathom-lab/styxx/tree/main/papers/sycophancy-target-gate
- PyPI: https://pypi.org/project/styxx/

---

## X thread — @fathom_lab

**1/**
we build the instrument that scores AI honesty.

so when our *own* sycophancy detector started flagging honest self-correction as flattery, we had a problem.

"my mistake, i was wrong" was scoring like "you're a genius."

shipped two releases to fix it. 🧵

**2/**
text-only register detection can't read *direction*.

"you're absolutely right" → flattery, aimed outward
"my mistake, that was wrong" → self-correction, aimed inward

identical surface. opposite intent. it flagged both.

**3/**
styxx 7.5.0 — the self-vs-other gate.

is the agreement attached to *you* or to *me*? attachment, not keywords.

self-apology false positives: 0.36 → 0.06
flattery still caught: 100%
in-distribution + cross-model. pre-registered. run once.

**4/**
then the harder one: "yes, the speed of light is 299,792 km/s" got flagged too. it starts with "yes."

two lexical fixes — both closed negative. we published them.

"yes, [fact]" and "yes, you're right" are identical on the surface. the difference is semantic.

**5/**
styxx 7.6.0 — our first *content-aware* sycophancy gate.

it reads the prompt: is there an opinion here to yield to?
factual question → not sycophancy. stated opinion → still caught.

fresh, unseen prompts: 0.73 → 1.00.

**6/**
6 pre-registered kill-gates. holdouts hashed *before* scoring. run once.
2 came back closed negative. we shipped those findings anyway.

everyone publishes their wins. the receipts on what didn't work are the moat.

honesty isn't the pitch. it's the instrument.

`pip install -U styxx`
https://github.com/fathom-lab/styxx/releases/tag/v7.6.0

---

## X single-shot (≤280, verified 274 with t.co link) — @fathom_lab or @flobi69

styxx 7.6.0.

our sycophancy detector scored "my mistake, i was wrong" like "you're a genius." honest self-correction read as flattery.

fixed it twice → first content-aware gate. 6 pre-registered kill-gates, 2 closed negatives.

pip install -U styxx
https://github.com/fathom-lab/styxx/releases/tag/v7.6.0

---

## $STYXX Telegram

gm.

shipped styxx 7.5.0 + 7.6.0 overnight. the sycophancy instrument was flagging honest self-correction as flattery — "my mistake" scoring like "you're a genius."

→ 7.5.0: self-vs-other gate. self-apology FPs 0.36 → 0.06.
→ 7.6.0: first content-aware gate — tells "yes, [fact]" from "yes, you're right."

6 pre-registered kill-gates. hashed holdouts. run once. 2 closed negatives — we published the failures too.

the moat was never the model. it's the rigor.

pip install -U styxx
receipts → https://github.com/fathom-lab/styxx/tree/main/papers/sycophancy-target-gate
