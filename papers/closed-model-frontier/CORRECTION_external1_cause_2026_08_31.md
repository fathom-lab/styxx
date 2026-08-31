# CORRECTION — EXTERNAL-1's cause analysis was partly wrong, and part of the fault was ours

Fathom Lab · 2026-08-31 · Corrects the "why it failed" section of
`RESULT_external1_the_gate_fails_in_the_wild_2026_08_31.md`, published hours earlier the
same day. Amends `PREREG_v13_repair_2026_08_31.md` before any repair was written.
Receipt: `external1_counterfactual.json`.

**The measured precision stands. The explanation does not.**

## What was claimed, and what is true

The RESULT named four defects, and led with this one: *bare names never met their full
paths* — a claim naming `glob.ts` accused while the diff carried `src/node/glob.ts`. That
was wrong. The gate's path lookup already matches on directory-suffix **and** on basename;
it has done so all along. The sentence should never have been written, and it was written
because the taxonomy was inferred from the adjudication panel's prose ("same file, fuller
path") instead of read from the instrument's own recorded reasons, which were sitting in
the ledger the whole time.

Reading those reasons instead shows the real distribution: the largest single cause of
rejected accusations was a **status mismatch** — the gate found the file and objected to
what had happened to it.

## And the status mismatches were substantially our fault

The corpus stores one row per file **per commit**. Across pull requests, a large minority
of file records disagree between commits — most often a file `added` in one commit and
`modified` in a later one. The harness emitted a diff hunk per row, and the diff parser
keeps the last header it reads, so a file the pull request genuinely created was handed to
the gate as merely modified. The gate then correctly objected that a creation claim did not
match a modification — against a status the harness had invented.

That is not the instrument being imprecise. That is the instrument being fed a false
account of the diff, by us.

The harness now folds each file's status across all of its commits before writing the diff:
added and never removed is a creation, removed and never added is a deletion, everything
else is a modification. The rule is order-free, documented in the code, and stated here.

## The counterfactual, measured rather than argued

Running the original accusing behaviour over the corrected reconstruction isolates the two
causes. The harness defect accounted for about a sixth of the path accusations — a real share, and smaller than a first reading of the panel transcript suggested. The rest
survive the fix, and they exhibit exactly the remaining patterns the panel described:
*"Removed the helper from `mantineTheme.ts`"* accusing the file of not being deleted, and
`Next.js` extracted as a filename and then accused of not existing.

## What this changes

- **The disabling stands.** Precision 0.23 was measured over the pipeline as it actually
  ran, and the preregistered consequence attaches to the measurement, not to the
  explanation. The accusation stays withheld until a held-out panel licenses its return.
  The withholding is now a documented flag rather than a deletion, precisely so
  counterfactuals like this one remain measurable.
- **V13's first repair is struck.** Suffix matching needs no repair because it was never
  broken. Implementing it would have "fixed" a defect that did not exist and claimed credit
  for the harness fix.
- **V13's remaining repairs are unchanged and now better evidenced**: the verb bound to the
  wrong object, prose nouns passing the extension whitelist, and negation read as
  assertion. The dominant one is the first of those.
- **A repair is added that the correction itself exposes**: the harness's status folding is
  now part of the measured pipeline, and any re-measurement must run over the corrected
  reconstruction or it is not comparable to anything.

## The rule this cost us, stated so it is not re-learned

An instrument's failure has two possible authors, and the one holding the pen is the more
likely of the two. Before attributing a measurement's failure to the thing being measured,
read the thing being measured's own words — not the summary of a summary. Both misreadings
here were preventable by opening a file we had already written.

---

*Second correction of the day, same discipline as the first: the number survives, the story
about the number does not, and the story is replaced in public before anyone builds on it.*
