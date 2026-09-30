# NOTE — PATH-2a: main's diff gate, unchanged, plus an overlay that only abstains where #97, #121 or #101 can have made a verdict wrong

## 0. Status

2026-09-30. Branch `fix/diffgate-abstain-where-wrong`, based on `origin/main` `1cde8b82`. On that base
`styxx/diffgate.py` hashes to `9b620e00a19464589308a987819894ae7cc3c111c66a5f8a457a84b8a6c604eb` and
`web/gate/diffgate.js` to `06688702999cdabe763265722a0ac14d4b9ffb40d0efcbb32339eba89f00c141` (sha256, LF).

**This note is written before the code and committed alone, ahead of it.** It states what PATH-2a is, the rules it
runs, why the two ports decide alike, what it covers and what it does not, and one falsifiable prediction. The
measurements are taken on the commits that follow and are recorded in `web/gate/README.md` and the CHANGELOG entry,
not here. Each later review pass gets its own new note, committed alone the same way; no note is edited.

PATH-2a stands beside, and does not replace, pull request #161 (branch `fix/diffgate-path-resolution`, head
`a1e85d6d`). The documents that frame the three defects live on that branch, not on `main`, and none of them is
edited here:

- `papers/closed-model-frontier/PREREG_path2_resolution_2026_09_17.md` (the defects, the repairs R-97, R-121 and R-101,
  and the unit gates G-P1 to G-P5);
- `papers/closed-model-frontier/AMENDMENT_path2_resolution_2026_09_17.md` and
  `papers/closed-model-frontier/ERRATUM_path2_amendment_2026_09_17.md`;
- `papers/closed-model-frontier/NOTE_path1_pairs_repinned_by_path2_2026_09_25.md` and the third to fifteenth-pass notes
  `papers/closed-model-frontier/NOTE_path2_*_pass_2026_09_*.md`.

#161 tried, over fifteen review passes, to *repair* the three defects with a reimplemented reader and a per-claim guard
against a vendored copy of `main`. At the fifteenth pass its #97 and #121 licences were withdrawn, so its output is
only ever `main`'s verdict or UNCHECKABLE, and its remaining blockers all come from the reimplementation. PATH-2a takes
the conclusion that history points to and builds only that: `main`'s own reader, unchanged in both ports and at both
doors, and an overlay that can do one thing to a record.

## 1. The three defects on `1cde8b82`, verbatim

Run on the base, at the raw door (`gate_diff_text`) and at the git door (`gate_diff` on a two-commit repository,
`HEAD~1..HEAD`).

**#97 — the earliest entry in diff order matching by exact path, suffix or base name.** A diff that modifies
`README.md` and creates `integrations/git/README.md`:

    "Created integrations/git/README.md."
    gate_diff_text                     PASS
      UNCHECKABLE  file_created  'integrations/git/README.md' is status 'M', claim wants 'A' — accusation WITHHELD pending the EXTERNAL-1 repair
    gate_diff (git, HEAD~1..HEAD)      PASS
      UNCHECKABLE  file_created  'integrations/git/README.md' is status 'M', claim wants 'A' — accusation WITHHELD pending the EXTERNAL-1 repair

The same two files in the other order (a hand-ordered raw diff; git always sorts `README.md` ahead):

    "Modified README.md."
    gate_diff_text                     PASS
      VERIFIED     file_touched  diff status 'A' for 'integrations/git/readme.md'

A claim naming a directory, over a diff that only creates the root `README.md`:

    "Created integrations/git/README.md."
    gate_diff_text                     PASS
      VERIFIED     file_created  diff status 'A' for 'readme.md'
    gate_diff (git, HEAD~1..HEAD)      PASS
      VERIFIED     file_created  diff status 'A' for 'readme.md'

**#101 — a changed `def` counts as added.** `backoff(n)` gains a parameter; `test_a` and `test_b` gain a trailing
comment; nothing is added. "Adds function backoff with jitter. Added 2 tests.":

    gate_diff_text                     PASS
      VERIFIED     symbol_added  added lines do define function 'backoff'
      VERIFIED     tests_added   diff adds 2 test functions, claim says 2
    gate_diff (git, HEAD~1..HEAD)      PASS
      VERIFIED     symbol_added  added lines do define function 'backoff'
      VERIFIED     tests_added   diff adds 2 test functions, claim says 2

**#121 — the path key's `lstrip("./")` merges a dotfile with its undotted twin.** A diff deleting `pr_agent.toml`,
creating `.pr_agent.toml` and modifying `.github/workflows/ci.yml`. "3 files changed. Created .pr_agent.toml. Deleted
pr_agent.toml. Only touches src/.":

    gate_diff_text                     FAIL
      CONTRADICTED files_changed_count  diff changes 2 files, claim says 3
      VERIFIED     file_created         diff status 'A' for 'pr_agent.toml'
      UNCHECKABLE  file_deleted         'pr_agent.toml' is status 'A', claim wants 'D' — accusation WITHHELD pending the EXTERNAL-1 repair
      CONTRADICTED only_touches         paths outside 'src': ['pr_agent.toml', 'github/workflows/ci.yml']
    gate_diff (git, HEAD~1..HEAD)      FAIL
      CONTRADICTED files_changed_count  diff changes 2 files, claim says 3
      UNCHECKABLE  file_created         '.pr_agent.toml' is status 'D', claim wants 'A' — accusation WITHHELD pending the EXTERNAL-1 repair
      VERIFIED     file_deleted         diff status 'D' for 'pr_agent.toml'
      CONTRADICTED only_touches         paths outside 'src': ['github/workflows/ci.yml', 'pr_agent.toml']

(The git door lists `--name-status` in git's sorted order, so the two twins register in the other order and the
last writer of the shared key differs.)

## 2. What PATH-2a is

**An overlay applied after `main`'s reader.** Both Python doors keep calling `main`'s `_gate` unchanged. The only edit
to `main`'s text at each door is `return _gate(` becoming `g = _gate(`, plus one appended line
`return _p2a_abstain(g, strict, lambda: _P2aFacts(...))  # PATH-2a`. In the port, `main`'s `gateDiffText` is renamed
`_gateDiffTextMain` (its definition and its DECLARE-1 self-call, two lines), and a wrapper named `gateDiffText` calls it
and then the overlay; the export lines and `build_bookmarklet.py` are untouched. DECLARE-1's recursion stays on `main`'s
reader, and the overlay runs once, on the outside. Everything the overlay adds sits between two marker comments,
`=== PATH-2a abstain-only overlay: BEGIN ===` and `... END ===`, in each file.

**The single change it may make to a record.** For a claim whose `(kind, verdict)` is in REACH (below), it may set
`verdict` to `UNCHECKABLE` and `why` to

    {V} withheld by PATH-2a ({defect}): {phrase}. main's reading: {main_why}

where `{V}` is `main`'s verdict word, `{defect}` is `#97`, `#121`, `#97, #121` or `#101`, `{phrase}` is one of the fixed
phrases of section 3, and `{main_why}` is `main`'s reason, verbatim. It then recomputes the gate verdict with `main`'s
own formula (`FAIL` if any claim is CONTRADICTED, or if `--strict` and any claim is UNCHECKABLE; else `PASS`). The
claim list — order, `kind`, `text`, `detail` — and every other field (`measured`, `why_unmeasured`,
`uncovered_sentences`, `uncovered_texts`, `sentences_total`, `unparsed_claims`, `base`, `head`) are `main`'s.

**The reconstruction property.** Remove the marked block, turn the two `g = _gate(` back into `return _gate(` and
delete the two lines ending in `# PATH-2a`, and the result is `main`'s `styxx/diffgate.py`, byte for byte (`9b620e00…`).
In the port: remove the block and turn the two `_gateDiffTextMain(` back into `gateDiffText(`, and the result is
`main`'s `web/gate/diffgate.js` (`06688702…`). A committed test performs both reconstructions and uses the
reconstructed `main` as the reference for every differential below, so no vendored copy of `main` is committed.

**The error fallback.** If the overlay itself raises while reading a diff, every claim in REACH abstains with the
phrase "this overlay failed while reading the diff". The abstain-only relation therefore holds by construction even
under a bug in the overlay; the tests assert that the phrase never appears on a committed or fuzz input.

**REACH** — the `(kind, verdict)` pairs the overlay may move:

| kind | VERIFIED | CONTRADICTED | why |
|---|---|---|---|
| `file_created`, `file_deleted`, `file_touched` | yes | not reachable | the path accusation is withheld on `main` (`WITHHOLD_PATH_ACCUSATION = True`) |
| `files_changed_count` | yes | yes | #121 can hide a file, and can make a truthful count read short |
| `only_touches` | yes | yes | a dropped dot can move a path in or out of the prefix |
| `tests_added` | yes | yes | #101 inflates the count |
| `symbol_added` | yes | no | #101 only adds hits, so a CONTRADICTED symbol is never #101's doing |

`tests_pass`, `compat_claim`, `declaration_problem` and every UNCHECKABLE claim are never touched. A test pins
`WITHHOLD_PATH_ACCUSATION is True`, with a message saying that a path CONTRADICTED is outside REACH and must be added
to it before that licence returns.

## 3. The rules

The overlay reads, from the door's own bytes: the paths `main` registered, *as written* (a line-for-line mirror of
`parse_unified_diff`'s loop that reuses `main`'s `_Pending`, or of `gate_diff`'s `--name-status` loop); the added and
removed lines under both ports' line splitting; and whether a file header holds a character the two ports split or
strip differently. A lockstep test pins the mirror: rebuilt with `main`'s `_norm`, it must reproduce
`parse_unified_diff`'s status map exactly, in both ports.

**Path claims** (`file_created`, `file_deleted`, `file_touched`; VERIFIED only). A decided path claim is kept only if
three readers without the mechanism also verify it: **V97** (exact path, then suffix, then base name, and base name only
for a bare claim), **V121** (keys keep their leading dots: `^(?:\.?/)+` removed instead of `lstrip("./")`), and **V97
and V121 together**. Before that the overlay abstains where it cannot compute these readers exactly: a divergent header
(`divergent`), a drive-like path or a final `.` segment where base names are read differently (`odd`), a comparison
that turns on case outside ASCII for a path the claim may match (`case`), or a reading that does not reproduce
`main`'s (`unreproduced`). Example: "Created integrations/git/README.md." over a diff that creates only `README.md`
abstains (`dir`); "Modified github/workflows/ci.yml." over `.github/workflows/ci.yml` abstains (`dot_tier`).

**`files_changed_count`** (VERIFIED or CONTRADICTED). When two registered paths differ only by a leading dot run, the
dot-kept count lies in a range `[lo, hi]` (the two bounds differ only where case outside ASCII could merge keys). A
CONTRADICTED count abstains only when `lo <= n <= hi`; a VERIFIED count abstains unless `lo == hi == n`. On the #121
reproduction "3 files changed." abstains, and "9 files changed." stays CONTRADICTED.

**`only_touches`** (VERIFIED or CONTRADICTED). Abstains where, for either candidate prefix set, "every changed path lies
under the prefix" reads differently with leading dots kept than with them dropped, or where V121's own choice of
whether the second prefix is path-shaped reads otherwise. "Only touches github/." over `.github/…` (a false VERIFIED)
and "Only touches .pr_agent.toml" over `config/.pr_agent.toml` (a false CONTRADICTED) both abstain; "Only touches src
and github." over `src/a.py`, `.github/x.yml` and `docs/x.md` stays CONTRADICTED, because `docs/x.md` is outside under
every reading.

**`tests_added`** (VERIFIED or CONTRADICTED). `chg` is the number of `def test_` sites `main` may count whose name a
removed line of the diff also defines (a `def` followed by any run of "coarse" characters: code point at most 0x20,
0x7F, or at least 0x80). The claim abstains iff `chg >= 1` and `got - chg <= n <= got`, PREREG R-101's interval. A
count outside it is wrong under every reading and stays as `main` gave it. Pairing is across the whole diff, not per
file: a superset of R-101's same-file rule that needs no per-file keys.

**`symbol_added`** (VERIFIED only). Abstains iff a removed line defines the claimed name (`def` or `class`, the same
coarse separators).

**Reasons.** The phrase keys and phrases:

| key | phrase |
|---|---|
| dir | the claim names a directory, and only a file of the same name elsewhere matches it |
| tier | a changed path that matches the claim more closely than the one main resolved it to reads otherwise |
| dot | with leading dots kept, the changed path the claim resolves to reads otherwise |
| dot_tier | with leading dots kept and the closest match taken, the claim reads otherwise |
| count | two changed paths differ only by a leading dot, which the path key drops, and counted apart the claim reads otherwise |
| only | with leading dots kept, whether every changed path lies under the prefix reads otherwise |
| tests | a test the added lines count is also defined in the removed lines, and a changed test is not an added one |
| symbol | the removed lines define this name too, and a changed definition is not an added one |
| divergent | a file header of this diff holds a character that the Python and JavaScript readers split or strip differently |
| odd | a path here has a drive-like prefix or a final '.' segment, where base names are read differently |
| case | a path here compares only where case outside ASCII is folded, which this overlay does not do |
| unreproduced | this overlay does not reproduce main's reading of the diff |
| unparsed | main's reason does not have the form this overlay reads |
| error | this overlay failed while reading the diff |

## 4. Why the two ports decide alike

The decision is a pure function of `main`'s record for the claim plus three facts: the registered paths, the two line
views, and the divergence flag. The line views and the flag are computed from the bytes with constant expressions in
both ports. The registered paths are the same strings in both ports whenever no file header holds a divergent
character (`\x0b \x0c \x1c \x1d \x1e \x1f \x85 U+2028 U+2029 U+FEFF`): CPython's `splitlines()` breaks lines beyond
`\r\n|\r|\n` only at those characters, and `str.strip()` and JavaScript's `trim()` differ only on them. When the flag is
set, every decided path, count and scope claim abstains with the same key in both ports.

Every operation is structural or ASCII: `startswith`, `endswith`, slicing, `find` of ASCII literals, strip and split
with explicit sets, a per-code-point fold (A–Z to a–z, U+212A to `k`, U+0130 to `i` + U+0307) and a per-code-point
"wild" form that replaces every code point at or above 0x80 by one placeholder. No `lower()`, `toLowerCase`, `\s`,
`\w`, `\b`, `\d`, `unicodedata`, `pathlib` or locale call appears in either block; a static self-check enforces that.
The lemma the case rule rests on — `fold(x) == fold(y)` implies `lower(x) == lower(y)` implies `wild(x) == wild(y)` —
holds because U+0130 and U+212A are the only non-ASCII code points whose `lower()` contains ASCII, and U+0130 is the
only one whose `lower()` has more than one code point. The tables the overlay depends on (the `splitlines` and
`isspace` sets, the `trim` and `\s` set, the two `lower()` facts) are pinned by enumeration on every runtime the tests
run on, so a newer Unicode on CI fails loudly rather than drifting.

Where `main`'s own two ports already disagree on a claim, the overlay follows each port's `main`; those records differ
anyway, so they are reported, not asserted.

## 5. Coverage

**"Because of" is defined by counterfactual variants of `main`.** V97 replaces `find_path`'s loop with the tiered loop;
V121 replaces `_norm`'s body with the dot-keeping key; V101 subtracts, from the counts and hits, the names the removed
lines define; and a fourth variant applies all three. A decided claim is **attributable** when `main`'s verdict is false
by truth and some variant's verdict for the same claim is not false. A committed truth test (an oracle over base and
head file models: git-style changed paths, CPython's `ast` for definitions) asserts that every attributable claim
abstains, on #161's reproductions and on generated families for the three defects, and pins the counts.

**Known gaps, with their shapes.** PATH-2a does not abstain on these; each is outside the three mechanisms or outside
what the bytes can tell apart:

- case-only merges with no dot (`README.md` and `readme.md`, or a claim in another case), which PREREG_path2 also
  leaves;
- names that `strip()` merges, such as a trailing space in a `+++` value;
- multi-commit renderings where a file created and later renamed has no dot twin (#161's R14.2);
- mnemonic, GNU or no-prefix renderings over real `a/` or `b/` directories, where `main` strips a real directory;
- joint shapes where #121 is one of two causes, so that V121 alone is still false;
- `async def` tests, which `main` does not count;
- `def` lines in non-Python files or inside strings, which are `main`'s #110 class;
- records with the right verdict and the wrong file in the reason: the reversed "Modified README.md." above, whose
  reason names `integrations/git/readme.md`, and "Created .pr_agent.toml." over the twins, whose reason names
  `pr_agent.toml`. Both are kept (option R in section 9).

## 6. Recall

Measured by a committed script, `web/gate/differential/path2a_recall.py`: per file and in total, the decided claims
of the reconstructed `main` and how many of them PATH-2a abstains on, by kind, phrase key and defect, with each corpus
file's sha256; with `--truth`, on cases that carry models, the right, false and undecided verdicts abstained. The
figures go into the README and the CHANGELOG at the head they were measured on.

**The prediction.** Of `main`'s 46 decided claims on its six committed pinned files, exactly one moves:
`path1_pairs.json`, pair `path1:unrepaired-typo`, claim 0 — ".githiub/workflows/dependabot.yml" resolved by base name
to `.github/workflows/dependabot.yml`, a directory claim matched only by base name and a false VERIFIED. #161's head
also abstains there. The pinned files are `main`'s records and are not edited; the move is recorded in a new
`web/gate/differential/path2a_moves.json`, which the pinned-pair checks read.

## 7. What this does not do

- **No repair.** PATH-2a never gives VERIFIED where `main` was wrong. The three defects are not repaired; they no longer
  produce `main`'s false verdicts on the shapes above, and the record says which verdict was withheld and why.
- **G-P1 is not met.** PREREG_path2's G-P1 expects VERIFIED on the #97 and #121 reproductions (for example
  `VERIFIED diff status 'A' for 'integrations/git/readme.md'`, and the count VERIFIED at 3). PATH-2a gives UNCHECKABLE
  where `main` was wrong, or keeps `main`'s record. Whether that is acceptable in place of G-P1 is **the operator's
  decision**, not this branch's.
- **#161's licences are not restored.** The repairs belong to #161's re-licensing work.
- **The path accusation stays withheld**, and COMPAT-2 is not affected.

## 8. Consequences

- **`--strict` fails on every new abstention**, exactly as it fails on any other UNCHECKABLE.
- **Capsules.** A v0.2 capsule minted on `main` over bytes where the overlay abstains will no longer reproduce on this
  branch. The two committed v0.2 diffgate capsules (DOGFOOD_session_2026_08_31 and HANDOFF_capsule_v02_2026_08_31) and
  charon's two capsule-diffgate lines are expected to be unaffected, and a test checks them. The diffgate record version
  stays `"v0"`.
- **Receipts are not regenerated.** Scripts whose re-run could read differently are the bench scripts, the BIN, COMPAT,
  DECLARE and SCOPE gates, the EXTERNAL-1 to EXTERNAL-6 harnesses, capsule mint and verify, and charon.
- **The bookmarklet grows** by the port block (about 8 KB); the exact size is recorded in the README.
- **Per-call cost** is expected to be within noise; it is measured and recorded with the figures.

## 9. Options left to the operator

- **O-1: the directory base-name rule** (`P2A_DIRECTORY_BASENAME_ABSTAINS = True`, a flag in the block). V97 does not
  resolve a claim naming a directory by base name alone; this is #97's second suggestion, which PREREG_path2 declined
  for its repair. It carries the predicted pinned move and most of the committed corpus's abstentions, and it is what
  catches "Modified github/workflows/ci.yml." over `.github/…`. Striking it keeps those false VERIFIEDs.
- **O-2: option R.** Also abstain on right-verdict records whose reason names another file (section 5).
- **O-3: whole-diff #101 pairing.** It costs recall on tests moved between files or classes.
- **O-4: strict users.** Each abstention fails `--strict`.
- **O-5: capsules.** Keep the diffgate version `"v0"` and disclose, or bump it, which would turn every older capsule
  into INSTRUMENT SKEW.
- **G-P1**, as section 7 says.

## 10. Review protocol

One note per review pass, written before that pass's code and committed alone. No note, PREREG, AMENDMENT, ERRATUM,
RESULT, receipt, certificate or sworn file is edited; a correction to this note goes in the next one.
