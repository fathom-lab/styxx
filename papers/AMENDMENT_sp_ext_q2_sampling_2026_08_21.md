# AMENDMENT — SP-EXT, Q2 standalone: sampling rule, declared before inspection

Frozen and committed **before any of the 140 Q2 candidates was read**. Only the
per-repository counts have been seen, which are in this document.

Parent prereg: `PREREG_sp_external_corpus_2026_08_21.md` (`38b8428`).

---

## what changed since the parent prereg

Q2 was frozen but **never ran**: its efficient form is `git log -S`, which against
`--filter=blob:none` clones timed out after 10 minutes on the smallest repository
in the set. All 14 repositories were re-cloned with full blobs, after which the
identical query takes **13 seconds**. Nothing about the query was altered — the
regexes in `scripts/sp_ext_q2.py` are copied verbatim from the parent prereg, and
the module says so.

## Q2's yield

2,914 commits examined by pickaxe across all 14 repositories → **140 candidates**.

| repo | Q2 | | repo | Q2 |
|---|---:|---|---|---:|
| inspect_ai | 34 | | trulens | 14 |
| great_expectations | 28 | | whylogs | 7 |
| deepeval | 18 | | deepchecks | 5 |
| lm-evaluation-harness | 18 | | garak | 4 |
| giskard | 4 | | evidently | 3 |
| ragas | 3 | | pandera | 2 |
| alibi-detect | 0 | | cleanlab | 0 |

For comparison, the intersection actually adjudicated in
`RESULT_sp_ext_2026_08_21.md` was **8** under the same frozen regexes. **Q2 alone
finds 17.5× more**, which confirms in numbers what that result could only assert:
requiring the commit message to match as well was throwing away most of the
signal.

## the sampling rule, frozen

420 agents (140 × 3 lenses) will not complete — a 52-agent run was already cut off
by a session limit today, and a partial adjudication reported as a whole one is
the defect this corpus catalogues.

So: **a random sample of 40 candidates, `seed = 20260821`**, drawn with
`random.Random(seed).sample(candidates, 40)` over the list as written in
`out_sp_ext_q2.json`, **each adjudicated under the unchanged 3-lens protocol.**

- The seed and the draw are published; the sample is reproducible from the raw
  file by anyone.
- The **100 undrawn candidates are reported as UNADJUDICATED**, never as
  rejected, and never as absent.
- **No candidate is swapped out of the sample for any reason.** If a drawn
  candidate is inconvenient, unparseable, or dull, it is adjudicated anyway and
  the outcome recorded.

## gates

G1, G2, G3 and G5 from the parent prereg apply unchanged, computed **over the
sample of 40** and reported as such. In particular:

- **G2 remains two-sided.** An accept rate above 80% goes in the title; below 20%
  goes in the title.
- **G1's yield threshold of 12 is not rescaled.** It was set for a corpus, not
  for a sample, and moving it because the denominator changed would be exactly
  the post-hoc adjustment this project forbids.

## what this cannot fix

The parent prereg's G5 stands: recall is unknown. Q2 finds commits whose **diff
took a particular shape**. A silent-pass fix that removed no flattering constant —
one that added a guard *above* an unchanged return, say — is invisible to it, and
that is a large and unmeasured class. **SP-EXT remains a lower bound and is never
quoted as a rate.**
