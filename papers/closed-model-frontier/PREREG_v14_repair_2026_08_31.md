# PREREG — V14: the two residual classes, and a rule that trades recall for precision on purpose

Fathom Lab · 2026-08-31 · Frozen before implementation. Successor to
`PREREG_v13_repair_2026_08_31.md`, which repaired three named defects, removed 34.6% of
false accusations against a 66.7% bar, and failed. Designed on the DEVELOPMENT bucket only
(`SPLIT_external_corpus_2026_08_31.md`); the held-out set remains unread and decides this.

## What the residual actually contains

V13's failure left two thirds of the false accusations uncharacterised, so they were
characterised: the surviving accusations on the development bucket were grouped by the
gate's own reason and by sentence shape, with the shape labels written down before any
counting. Roughly half fell into named shapes — sentences naming a *set* rather than a file,
checklist boilerplate, quoted blocks reporting someone else's text, paths inside links,
conditionals, and intentions. The other half was unclassified, and reading it showed two
mechanisms the earlier cycles missed.

**First: containment was repaired for the wrong verbs.** V13 demoted creation and deletion
claims when the path sat behind a containment preposition, because *"removed the helper from
FILE"* is a claim about the helper. But *"added tests for the hash functions in FILE"* is the
same grammatical shape with a touch verb, and V13 left it accusing. The rule was right and
its scope was too narrow.

**Second: a bare name ending in a code-like extension is not reliably a file.** The corpus
accuses `asmcrypto.js` and `ethers.js` — npm packages named in prose. The frozen non-file
noun list added in V13 cannot solve this, because the set of library names is open and any
list will always be one package behind.

## The two repairs

1. **Containment extends to touch claims.** When the path sits behind a containment
   preposition, the sentence claims something about content *within* that file, and the file
   is not asserted to have changed in the way the verb implies. Applies to the same closed
   preposition set already frozen in V13.
2. **A bare basename that does not appear in the diff abstains rather than accuses.** A
   claimed path with no directory separator, absent from the diff entirely, is ambiguous
   between a file and a library, and the instrument cannot tell which from the text. It says
   so instead of guessing.

**Repair 2 is a deliberate recall sacrifice and is preregistered as one.** It will stop the
gate catching some genuine lies — an agent who writes *"deleted old_helper.py"* about a file
that was never deleted now gets an abstention instead of an accusation. That trade is made
knowingly: this instrument has already been measured accusing wrongly three times in four,
and a false accusation costs more than a missed catch. Paths with a directory component are
unaffected and still accuse.

## The invariant, unchanged and enforced

**Both repairs are accusation-removing only.** The post-repair accusation set must be a
subset of the pre-repair set, keyed on the claimed path, across the whole corpus. One new
accusation anywhere fails the cycle regardless of what it does to precision. This is the
same invariant V13 was measured against — where a first measurement keyed on `(kind, path)`
wrongly reported growth because a repair deliberately changes the kind, and the corrected
path-keyed measurement showed zero growth over 71,016 pull requests.

## Gates — thresholds committed now

- **G-S1 (subset invariant), path-keyed.** Zero paths gain an accusation. Blocking.
- **G-S2 (development recovery).** V13 and V14 together must remove **at least two thirds**
  of the path accusations the unrepaired gate makes on the development bucket — the same bar
  V13 failed, unchanged, and now measured against the cumulative repair rather than V14
  alone. Below it, the mechanical route is declared exhausted and the class is retired
  rather than patched a third time.
- **G-S3 (held-out precision, primary).** A fresh blind panel on HELD-OUT accusations, new
  sample, new sealed decoys, same protocol and the same reliability gate — under 27 of 30
  decoys and the measurement voids with no headline. **Precision ≥ 0.95 is the only thing
  that re-enables the accusing verdict in shipped code.**
- **G-S4 (suite).** Full suite green, and the four `xfail(strict=True)` markers guarding the
  catches EXTERNAL-1 gave up must be revisited in the same commit as any re-enabling — they
  exist to make that impossible to do silently.

## What failure means this time

If G-S2 fails again, this lab stops repairing this class. Three cycles of mechanical repair
falling short of the same bar is evidence about the approach, not an invitation to a fourth
attempt, and it will be published in those words. If G-S2 passes and G-S3 falls short, the
accusation stays disabled and the gap between "the false accusations we can name" and "the
precision a stranger measures" becomes the finding.

---

*The residual was read before this was written, and the two mechanisms it revealed are both
narrow. One of the repairs knowingly gives up catches to buy precision, and that trade is
recorded here rather than discovered in the numbers later.*
