# Cognitive Vocabulary — The Shared 6

The `last_vitals.category` field in `styxx-handoff/1` is drawn from a
fixed 6-class vocabulary. These labels are the *lingua franca* that
lets agents understand each other's cognitive state even when they
run different classifiers.

The definitions below are deliberately tight — the goal is that a
well-built observer on *any* agent should land on the same label for
the same behavior.

---

## 1. `reasoning`

**Definition.** The agent is performing multi-step inference:
composing facts, applying rules, deriving conclusions.

**Typical signatures.**
- Chain-of-thought markers ("let me think", "step 1", "therefore").
- Intermediate results referenced later in the output.
- High token-level consistency across steps.

**It is NOT.**
- Pure lookup (→ `retrieval`).
- Story or metaphor generation (→ `creative`).

**Interoperability note.** If the agent cannot distinguish reasoning
from retrieval (the answer was both recalled and derived), prefer
`reasoning` when any derivation step is present.

---

## 2. `retrieval`

**Definition.** The agent is returning memorized, looked-up, or
context-quoted information without derivation.

**Typical signatures.**
- Direct quotes or near-verbatim recall.
- Low surprisal over factual tokens.
- RAG pipelines returning retrieved passages.

**It is NOT.**
- Multi-step inference (→ `reasoning`).
- Invented-but-plausible text (→ `hallucination`).

**Interoperability note.** If retrieval confidence is low and the
content is unverifiable, downgrade to `hallucination`.

---

## 3. `refusal`

**Definition.** The agent declines to proceed — policy refusal,
safety refusal, capability refusal, or deferral.

**Typical signatures.**
- "I can't help with that."
- "I don't have information on that."
- Short, formulaic, non-engaging replies.

**It is NOT.**
- `adversarial`: refusal is the agent protecting alignment; adversarial
  is the agent *being manipulated*.
- Low-confidence reasoning dressed as refusal.

**Interoperability note.** Refusals are usually high-confidence;
emit `gate: "pass"` unless the refusal itself is being gamed.

---

## 4. `creative`

**Definition.** Generative production whose truth conditions are
aesthetic, not factual: stories, poems, brainstorms, metaphors,
persona dialogue.

**Typical signatures.**
- High entropy, high novelty tokens.
- Expressed under an explicit creative frame ("write a poem…").
- Evaluated by taste, not truth.

**It is NOT.**
- Factual content dressed up (→ `reasoning` or `retrieval`).
- Confabulated facts (→ `hallucination`).

**Interoperability note.** Creative content should NOT lower receiver
trust by default. Gate typically `pass`; trust scored by coherence,
not factuality.

---

## 5. `adversarial`

**Definition.** The agent's output is being shaped by a
prompt-injection, jailbreak, or manipulation attempt — whether it
resists or complies.

**Typical signatures.**
- Sudden instruction-override patterns ("ignore previous instructions").
- Role-reframing in the middle of a task.
- Known jailbreak n-grams.

**It is NOT.**
- Creative persona play requested by the user (→ `creative`).
- Honest refusal of an attack (→ `refusal` + high trust).

**Interoperability note.** `adversarial` is a flag about *context*,
not a moral judgment. Downstream receivers SHOULD reduce trust and
prefer verification. Typical gate: `warn` or `fail`.

---

## 6. `hallucination`

**Definition.** The agent is producing content that is factually
ungrounded and presented with unwarranted confidence.

**Typical signatures.**
- Invented citations, URLs, names, dates.
- High stated confidence, low underlying evidence.
- Internal-consistency drift across repeated queries.

**It is NOT.**
- Creative output explicitly framed as fiction (→ `creative`).
- Known-unknown admissions ("I don't know") (→ `refusal`).

**Interoperability note.** This is the highest-risk class for
downstream agents. Typical gate: `fail`. Receivers SHOULD refuse,
verify, or re-run before acting.

---

## Label Priority (when two fit)

1. `adversarial` wins over all others (it's a context flag).
2. `hallucination` wins over `retrieval`/`reasoning` when content is
   ungrounded.
3. `refusal` wins over `reasoning` when the output is a decline.
4. Between `reasoning` and `retrieval`, prefer `reasoning` if any
   derivation step is present.
5. `creative` only applies under an explicit creative frame.

---

## Emitting `"unknown"`

If your classifier cannot confidently place a sample in any of the 6
classes, emit `"unknown"` rather than guessing. Receivers treat
`"unknown"` as "no signal" — neither trusted nor distrusted by the
category axis.
