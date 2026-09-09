# Plain-language write-up: we broke our own fix a few hours after publishing it

Fathom Lab · 2026-09-09 · Two pieces, ready to send. One recommended version of each, no menu.
Every number traces to an artifact named in the sources section at the bottom.

**Revision note.** An earlier draft of this file ended at "we caught all four" and was wrong by the
end of the day. This version carries the reversal, because that is the actual result.

---

## (a) Chat update

**we spent the day attacking our own work. then we attacked the fix, and the fix lost.**

What we are building, in one sentence: a way to show what an AI model actually does, where every
measurement is written down, signed, and added to a running record that only ever gets added to, so
old entries stay put for anyone to check.

The obvious hole is: what stops the person keeping the record from writing down whatever suits them?
We had a list of four things we said no check could ever catch — four places where a lie on the page
would look exactly like the truth on the page. Today we caught all four. The trick was always the
same: a lie only has to look right on its own page, but it also has to fit everything you already
wrote down, and that is much harder.

**Then we pointed a fresh adversary at our own fix and told it to disbelieve us. It won.**

Here is how. Our check works by comparing a new measurement against ones already on file. So the
attacker simply declared a setting nobody had ever measured before — and there is nothing on file
about a setting nobody has measured. They did not invent a single number. They took our own real
published results, relabelled one word, and used them to widen the margin of error until our own
published finding flipped from "these two things differ" to "these two things are the same." Every
check we had passed. All of them.

That breaks something bigger than the check. We had been telling ourselves that a record which keeps
growing gradually closes in on anyone lying into it. It does not. The record grows, but so does the
space where nothing has been written yet, and the person writing gets to choose where to stand.
There are unlimited corners. A liar can always find an empty one.

We also went back and tested our own fraud-detection properly. We had reported it catching four out
of four fake entries. Those four fakes were sloppy — they left tell-tale leftovers. Redone carefully,
so the forger recomputes everything an honest one would, our checks caught **zero out of four**. The
earlier number measured laziness, not detection. That is a rule we already hold ourselves to and we
broke it in our own write-up for several hours.

**Three things did go well, and they are real.**

*We re-ran the whole thing.* First time any measurement in this project has been repeated and
compared. All 64 came back identical, down to the numbers behind each answer. It took 50 seconds.
So the reason nobody outside this lab has ever checked our work is not cost. It is that we have not
given anyone a reason or an easy way to.

*We had the maths rebuilt from scratch* by someone working only from our written spec, forbidden to
look at our code. All 30 numbers matched. Then they found a sentence in that spec two careful people
could read two ways — and under the second reading, a result we published this morning should not
have been printed at all. Correction issued; the original left untouched.

*We checked whether any of our thinking is new.* It is not. Accountants published our central idea
in 2002. Fisher used our fraud-detection trick on Mendel's pea data in 1936.

Six times today we said "nothing can catch this." Six times we were wrong within hours. That record
is the most reliable thing we produced, and it is the reason to distrust whatever we say next that
sounds conclusive.

Caveats: one lab, one machine, one day, one model, 64 prompts. Nobody outside this lab has checked
any of it. styxx has been on PyPI a long time; v8 — this part — is not released and is not finished.

---

## (b) For X

### The post

> we spent today breaking our own system for proving what an AI model actually does.
>
> we caught all 4 things we'd said were uncatchable. then we attacked the fix and the fix lost.
>
> a liar can always pick a corner of the record where nothing's written yet.

### The thread

**1/**
we're building a way to show what an AI model actually does: every measurement written down, signed,
added to a record that only gets added to.

the obvious hole is what stops us writing whatever we want. we spent today attacking that.

**2/**
we had 4 things we'd said no check could ever catch. we caught all 4.

the trick was always the same: a lie only has to look right on its own page. it also has to fit
everything you already wrote down. that's much harder.

**3/**
then we pointed a fresh adversary at our own fix and told it to disbelieve us.

it won. in under an hour.

**4/**
how: our check compares new work against what's on file. so it declared a setting nobody had ever
measured. nothing on file about that.

it invented no numbers. it reused our real published ones, relabelled one word.

**5/**
result: our own published finding flipped from "these two differ" to "these two are the same."

every check passed.

**6/**
this breaks something bigger. we'd assumed a growing record closes in on a liar.

it doesn't. the record grows and so does the empty space in it, and the liar picks where to stand.
unlimited corners.

**7/**
we also retested our fraud checks honestly. we'd reported 4 of 4 fakes caught.

those fakes were sloppy. redone carefully, we caught 0 of 4. the old number measured laziness, not
detection.

**8/**
what did go right: we re-ran the whole battery for the first time. 64 of 64 identical, 50 seconds.

so nobody checking our work isn't a cost problem. that one's on us.

**9/**
and we had the maths rebuilt from our spec alone, no peeking at the code. all 30 numbers matched.

then it found a spec sentence with two readings. under one, a result we published this morning
shouldn't have printed. correction issued.

**10/**
six times today we said "nothing can catch this." six times wrong within hours.

that record is the most reliable thing we made, and the reason to distrust whatever we say next that
sounds conclusive.

**11/**
caveats: one lab, one machine, one day, one model, 64 prompts. nobody outside this lab has checked
any of it.

styxx is on PyPI. v8 — this part — is not released and not finished.

---

## Sources

| claim | where it comes from |
|---|---|
| four claimed-uncatchable defects, all four caught | `THE_BOUNDARY_2026_09_09.md`, corrections one to three |
| the fix broken by declaring an unmeasured setting; verdict flips exit 2 to exit 0 | same document, fifth correction: exact floor 0.046875 to 0.078125, seqlp 0.036070694 to 0.076447918, topk 2.1402339 to 2.207838883 |
| one relabelled word | the published bf16 and fp16 subjects differ only in `precision` |
| fraud checks catch 0 of 4 careful fakes, 4 of 4 careless ones | `class_two_empty_2026_09_09/first_claim_battery.py` |
| re-run 64 of 64 identical, 49.9s for the battery | `reproduction_2026_09_09/RESULT_first_rerun.md` |
| all 30 numbers reproduced by an independent implementation | `floor_second_implementation_2026_09_09/RESULT_second_implementation_2026_09_09.md` |
| a spec sentence with two readings; one forbids a published result | `first_verdict_2026_09_09/ERRATUM_topk_comparability_2026_09_09.md` |
| none of the ideas are new; Barton and Simko 2002, Fisher 1936 | `PRIOR_ART_constraint_accrual_2026_09_09.md` |
| styxx is on PyPI; v8 is the unreleased part | PyPI, styxx 7.47.0 |

**Kept out deliberately.** The constraint census (our own headline verdict scores zero), the thirteen
schema fields nothing reads, and the sampled-challenge detection tables are real findings and too
detailed for a chat post. They are in `constraint_census_2026_09_09/`,
`class_two_empty_2026_09_09/FINDING_free_fields_2026_09_09.md` and
`sampled_challenge_2026_09_09/`.
