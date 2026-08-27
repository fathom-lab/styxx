# The OATH contract — how to write a document that carries its own receipts

**What this is.** A specification you can adopt in your own repository, without adopting anything
else from this one, so that any reader can mechanically check your numbers against your data
without trusting you. Plus the tool that tells you whether you have kept it.

```bash
pip install styxx
python -m styxx.oathready YOUR_DOC.md results.json [more_receipts.json ...]
```

It prints every numeric token in your document, says whether it grounds in a receipt, and tells
you what to change where it does not. Exit code is non-zero only if something is *accused* —
silence is honest and never fails the check.

---

## Why a contract, and not a detector

Proof-carrying code ships a program with a machine-checkable proof of its own safety, so you do
not have to trust the author. It does **not** verify arbitrary binaries: it requires a compiler
that emits the proof alongside the program. Retrofitting it onto software that was not built to
carry proofs is not something anyone claims to do.

This is the same move one level up, and it has the same boundary — though the boundary is not
where we first said it was.

On 2026-08-26 we pointed our verifier at twelve public repositories it had never seen, saw it
abstain on 94% of what it read, and reported that **every accusation it made was false**.
**That claim is withdrawn.** Repeated on 2026-08-27 against 140 repositories drawn from seven
filename conventions rather than two, the false-accusation rate is `0.2596` — about three quarters
of what it accuses outside this lab are genuine claims. The original finding replicates almost
exactly on its own query and nowhere else: it was a fact about one filename, not about external
writing. The measurement that withdraws it is
[`RESULT_oath_external_corpus_2026_08_27.md`](papers/closed-model-frontier/RESULT_oath_external_corpus_2026_08_27.md).

The same cycle found something worse, and you should know it before adopting anything here. Of
external tokens the verifier **verified** — the affirmative attestation, the thing this whole
document is asking you to rely on — a blind panel judged only about half to be claims at all. The
rest are command-line flags, link labels, hardware specifications and numerals inside HTML tags,
each carrying `OATH-VERIFIED` because its value happened to match a receipt field. The instrument
is not nearly silent outside its contract. It is noisy in both directions, and the direction that
should worry you is the one where it speaks confidently.

None of that is a claim about those repositories, all of which are doing what essentially the
whole field does. It is a claim about the instrument, and the rules below are what keep you on
the side of it that works.

So this document is the contract. It is short, every rule below was learned by getting it wrong,
and each names the cycle that taught it.

---

## The contract

### 1. Cite your receipts, and ship them

A claim document names the JSON files its numbers came from, and those files are committed next
to it. Receipts are the explicit set you cite — nothing is discovered, nothing is inferred. If a
number's receipt is not in the repository, the number is not checkable, and the honest status is
abstention rather than a quiet pass.

### 2. Persist SUMMARY fields, not just per-item arrays

Write `{"recall": 0.82, "n_held": 27}`, not only `{"per_item": [{...}, {...}]}`.

Bulk per-item arrays are deliberately excluded from the checkable surface. A receipt carrying a
thousand row values covers most two-decimal numbers in `[0,1]` by coincidence, so matching
against it verifies nothing at all — a corrupted number would "verify" just as readily as a
correct one. *(Learned in v0.1, where exactly that happened.)*

### 3. Name the quantity on the same line as the number

`recall reached 0.82` grounds. A bare `0.82` in a sentence that never says what it measures does
not, because an integer or a float only binds to a receipt field whose **path** shares a word
with the claim's own line. This is what stops `27` from grounding in some unrelated experiment's
`n_held`. *(v0.3 count-binding.)*

### 4. Never let a claim sit on a truncated line

We found one of our own published findings failing months later because line 13 ended
mid-sentence — `n=48 → 43 scored (27 HELD, 16 CAVED, 4` — and the dangling `4` had lost the words
that bound it to `n_nogate`. The value was in the receipt the whole time. The verifier was right
and nothing could see it. *(Found 2026-08-26; recorded in `tests/test_certificate_reproduces.py`.)*

### 5. Know that quoting a number is treated as claiming it

**This is the sharpest limit of the current instrument.** A figure you quote — from another
paper, from a console transcript, from an error you are reporting — is treated as a figure you
assert. There is one narrow escape (disclosure phrasing such as *originally printed* or
*superseded*) and it does not reach quotation in general.

The document announcing this limitation is itself accused on the numbers it quotes as examples.
We published it that way rather than rewording it until it passed. If you write error reports,
literature reviews, or anything that repeats other people's numbers, expect false accusations and
plan to disclose them.

### 6. Keep configuration values off measurement lines

A number is required to ground when its line carries measurement vocabulary — `rate`, `recall`,
`mean`, `accuracy`, `delta`, `score` and their relatives. So *"the learning rate was tuned over
100,000 steps"* obligates `100,000`, because the line contains `rate`. Hyperparameters, seeds,
step counts and API constants sharing a line with that vocabulary will be accused.

Put configuration in its own sentence, or persist it as a receipt field and let it ground
honestly. *(Measured on a real README; see the recon.)*

### 7. Numbers inside formulas and tables are not exempt

The purest false accusation we have produced is the literal `1` inside
`\left(1 \pm \frac{\Delta \sigma^2}{\sigma^2}\right)` — a mathematical constant, accused because
the formula contains a `\Delta`. Markdown table row ordinals under an ordinal header are
explicitly silenced *(v0.11)*, but no comparable rule exists for formula constants or for label
cells. Until one does, keep claims out of rendered mathematics.

### 8. Print a number at full precision and it is obligated regardless

A value printed at seven or more fractional digits was copied out of a computation rather than
typed by a person, so it must ground no matter what its line says. Quote the receipt value
verbatim and it verifies; round it and it may not. *(v0.7.)*

### 9. Check what your numbers bind TO, not just that they bind

This is the rule most likely to save you from public embarrassment, and it is the one no
pass/fail verdict will tell you.

A number can "verify" against a receipt leaf that merely happens to hold its value — an array
index, a seed, a step counter, a loss that coincidentally equals your layer count. That is
arithmetic accident wearing a receipt, and it is *worse* than abstaining, because it looks like
evidence. `styxx.oathready` reports these separately as **coincident** for exactly this reason.

In our own flagship result note, nine bindings are coincident. We report the number rather than
suppress it. *(The path-binding repair that would fix this was measured across five design
families in v0.8 and closed NEGATIVE — none beat parity. It is an open problem, not an oversight.)*

### 10. If a number decides something, score the null rule too

This rule is not about certifying a document. It is about the measurements you take *before*
freezing a bar, and it is here because it is the one we broke most recently and most stupidly.

A census scored candidate rules for a proposed clause and reported one column as "the column that
decides". An adversary scored the rule that does nothing at all, and it scored the same. The best
candidate and the worst rule available were indistinguishable, so nothing downstream of that
column was evidence for anything, and the cycle was cancelled rather than shipped
(`papers/closed-model-frontier/RECON_v13_not_frozen_2026_08_27.md`).

The countermeasure costs one extra row in your table. **Score the most permissive rule your design
admits — the one that does no work — and put it in the receipt beside the candidates.** If it ties
your best candidate, the column is not measuring your idea; it is measuring your frame.
`styxx-discriminates` does the comparison and names the two ways it fails: `DEGENERATE`, where
everything including the control shares one value, and `NULL_TIES_BEST`, where candidates differ
from each other but none beats doing nothing — which looks alive and is not.

Necessary, not sufficient. A column can separate cleanly and still measure the wrong thing.

---

## What a kept contract does and does not buy you

**Does.** Any reader can re-run the check against your committed data and get the same answer you
did, without trusting you, without contacting you, and without a GPU. Numbers that drift away
from their receipts become visible instead of silently ageing. That is a real property and
almost nothing in the field currently has it.

**Does not.** It does not mean your numbers are correct. The check binds a claim to a receipt; it
cannot tell you the receipt describes the experiment you say it does, that the experiment was
well designed, or that the conclusion follows. **A document can keep this contract perfectly and
be completely wrong.** Anyone who tells you otherwise — including us — is overclaiming, and you
should say so publicly.

## When not to bother

If your document quotes other people's numbers more than it reports your own, the current
instrument will fight you (rule 5) and the report will be mostly false accusations. If your
results are inherently non-numeric, there is nothing here for you: this checks numeric claims
only, with no semantic entailment of any kind.

## Tell us when it breaks

A replication that *fails* is worth more to us than one that matches, and
[`REPLICATIONS.md`](REPLICATIONS.md) says how to file one. If `oathready` accuses a number that
is plainly not a claim, that is a defect in our instrument and we want the case — the row-ordinal
class became a whole cycle because four such accusations turned out to be false, and the four
were ours.
