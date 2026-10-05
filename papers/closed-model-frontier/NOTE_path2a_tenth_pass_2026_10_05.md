# NOTE — PATH-2a, tenth pass: DECIDE and APPLY, the static checks named as lints, the shelf paragraph corrected, and each finding of the reviews of `c69b161b`

## 0. Status

2026-10-05. Branch `fix/diffgate-abstain-where-wrong`, head `c69b161b` (pass nine). `main`'s two reader files are
unchanged (sha256, LF: `9b620e00…` and `06688702…`). The earlier notes are listed in the README's *PATH-2a* section;
none is edited.

What this pass had to read:

- four finished reviews of `c69b161b`: by construction and cost (fix before shipping: two majors, five minors and a
  process entry); coverage and recall (ship: one record, three minors); cross-port (ship: one record, five minors);
  integration, CI and docs (ship, on a condition outside the tree: six minors);
- the lead's direction of 2026-10-05 (§1), which outranks the usual order of work where the two differ.

**This note is written before this pass's code and committed alone, ahead of it.** Nothing below was prototyped in
the working tree. A figure in this note is a reviewer's, measured at `c69b161b`, and is named as such; §9 says what is
measured again at the head that carries the code. If the committed code departs from this note, the departure goes
into a note of its own. Finding identifiers are this note's: A-n, B-n, C-n and I-n number each lens's findings in the
order its review lists them.

Dispositions: **fixed** (a change in this pass, with a test where the finding is a defect); **fixed by construction**
(the property now holds at run time, whatever the block's other code does); **disclosed** (kept, and stated in the
README, the CHANGELOG or here); **operator** (left to the operator); **recorded**.

## 1. The lead's direction of 2026-10-05

1. **Stop extending the static record-write checks.** Pass nine's construction review found, again, ways an edited
   block could slip past the Python self-check and the port's store scan. A source scan cannot close that. Bar A is
   to hold by construction at run time instead, in both ports and at both doors: the block is split into DECIDE and
   APPLY (§2).
2. **The static checks that remain are lints against an honest future edit, not proofs against a hostile one.** The
   README says so in those words; the escape-inside-an-identifier hole is closed by decoding; obfuscation is not
   chased; what the lints do not cover is listed (§3).
3. **The README's EXTERNAL-1 shelf paragraph is wrong in two ways** (§5).
4. From the coverage review's minors: the full limit of the `redefined` rule, the decorated-world exchange figure,
   and the complete per-rule table (§6).
5. The import cost is measured again and stated as a ratio (§7).
6. **A correction to the lead's own decision text of 2026-10-04**, which the ninth note and the README repeat (§4).

## 2. Bar A by construction at run time: DECIDE and APPLY

Up to pass nine one function, `_p2a_abstain` (`_p2aAbstain` in the port), held `main`'s live record, called the rules
on it and wrote the abstentions; that the rules wrote nothing else was the business of a source scan. From this pass:

**DECIDE** (`_p2a_decisions(seen, facts)`; `_p2aDecisions` in the port) holds every rule and every reader. It is
given `seen`, a copy made for it, and the door's bytes through `facts`; it returns plain data, a list of decisions,
each `(claim index, phrase key, defect tag)`. It is never given the record.

**What the copy is.** The lead's words are "a deep copy of main's result". The block imports nothing, and the port
must run in browsers that have no `structuredClone`, so the copy is built field by field, and it is narrower than a
deep copy: a fresh object holding a fresh list with one fresh claim per claim of the record, in order; each holds the
claim's kind, verdict and reason (three strings) and a fresh detail holding, of the six fields a rule reads (`path`,
`n`, `name`, `prefix`, `prefix2`, `declared`), those whose value is a string or a boolean. Strings and booleans cannot
be changed in place, so nothing DECIDE can reach through its argument is shared with the record. The claim's text is
not copied: bar C's premise, that a decision never reads it, now holds because DECIDE is not given it. No rule reads
any other field, and `main` writes no other type into those six.

**APPLY** (`_p2a_apply(g, strict, decide)`; `_p2aApply`) is the only code of the block that touches the record. In
order:

1. It lists the claims in reach, by index, from the record itself. If there is none it returns the record as it came:
   DECIDE is not called and no copy is made.
2. It calls `decide` on the copy and reads what comes back, inside one `try`. A decision is taken only if it is a
   3-tuple (a 3-element array in the port) of an integer that is the index of a claim in reach, a string that is a key
   of the fixed phrase table, and a string that is one of the fixed defect tags of that claim's kind. What is kept of
   it is the record's own claim, the table's own phrase text and the tag; no object DECIDE made is kept. Everything
   else DECIDE returned is ignored.
3. **The fallback rule is the existing one**: if `decide` raises, or anything raises while its result is read, every
   claim in reach is withheld with the phrase `error` ("this overlay failed while reading the diff") and its kind's
   tag. If `decide` returns something that is not a list (an array, in the port), every claim in reach is withheld
   the same way with a phrase of its own, `malformed`, so that the record says which of the two happened. `malformed`
   is new; with `unreproduced`, `unparsed` and `error` it is a fallback phrase, and the tests assert that none of the
   four is written on the committed inputs.
4. Only then does it write, and nothing in this step runs code DECIDE supplied: for each decision taken, if the claim
   is still decided and in reach (so a second decision for one index is ignored), it stores the reason, in the fixed
   form `{verdict} withheld by PATH-2a ({tag}): {phrase}. main's reading: {main's reason}`, and then the verdict
   `UNCHECKABLE`. Exactly those two fields.
5. If a claim moved it recomputes the gate verdict by `main`'s own formula over the final claims; otherwise the
   record is `main`'s object, untouched. It returns the record.

`_p2a_abstain(g, strict, facts)` stays as the doors' one call, so the two hook lines of the Python file and the port's
`gateDiffText` do not move and the reconstruction test (cut the block, revert the hooks, `main`'s bytes come back) is
unchanged: it is now `_p2a_apply(g, strict, <DECIDE over facts>)` and nothing else.

**What follows, and what does not.** Whatever `decide` is — the block's own, an edited one, or one written to do
harm — the record APPLY returns is `main`'s but for claims in reach turned UNCHECKABLE with a reason of the fixed
form, and the gate verdict is `main`'s formula over the final claims: bar A as the bar words it. It does not follow
that the right claims are withheld (bar B: the truth tests) or that the two ports decide alike (bar C). An ignored
decision keeps `main`'s verdict, so a DECIDE that returned a decision APPLY refuses would be a silent miss: a
committed test asserts that on the committed inputs APPLY takes every decision the block's DECIDE returns. And
APPLY itself is code: an edit of APPLY, of `_p2a_abstain` or of the hooks is outside the construction. Those few lines
are pinned as text by a committed test, one pin per port, so that an edit of them fails a test that names what moved
(the byte pins of the whole file fail on a comment anywhere, which is why the ninth review did not count them).

**Tests.**
- Hostile DECIDE functions handed to APPLY, in both ports, over #161's reproductions in both strict modes: ones that
  change, empty or grow the copy and each claim of it; return junk (nothing, a number, a string, a mapping, a tuple, a
  list of junk); return indices out of range, negative, fractional, boolean or as strings; return one index twice
  with different phrases; return phrases or tags outside the fixed sets, or another kind's tag; return decisions for
  claims outside reach or already UNCHECKABLE; return a valid decision for every index; raise; and, in Python,
  return look-alike objects (a list or tuple subclass that lies about its length or raises when read). None may
  produce a record outside the abstain-only relation, and the ones whose whole output APPLY ignores must leave the
  record equal to `main`'s.
- The ninth construction review's plants (five in Python: a default argument, a lambda default, `*args`, `**kwargs`
  and a class attribute; nine in the port: a member as a `for … of` or `for … in` target, an optional call of `pop`,
  `push` and `reverse`, an escaped member store, an escaped mutator, an escaped `Object.assign`, a helper that hides
  `.claims`), each planted where the decisions are computed and aimed at that function's claims. At `c69b161b` that
  function held the live record and each plant moved it while every static check passed; the test asserts that the
  record stays inside the relation, with `main`'s claim count, on inputs that fire the plant.
- The pins of APPLY's text; the relation, bar C and truth tests as they are.

## 3. The static checks, named for what they are

**Deleted, with their plants**: in the Python block `_p2a_holder` (the alias reader of pass eight),
`_p2a_shape_problems` (the `todo` / `hits` shape of pass nine), and the store, item-store, augmented-store and
mutator rules of the self-check; in the test module the port's store scan (`js_store_problems`, `js_abstain_problems`
and their helpers). They guarded a property that no longer rests on them, and each pass found another way round them.

**Kept, as lints against an honest future edit**: the rules that keep a runtime Unicode question out of the block,
which bar C(i) leans on. Python: no import; only listed names and attributes; no `lower`, `casefold`, `is*`,
`splitlines`, `normalize`, `encode`, `int`, `repr`, `format`, `getattr`, `eval`; every strip and split with its
characters; every regex static, with no class escape, flag or unescaped `.`; ASCII source. Port: the token scan's
banned names and words, regex rules, loose equality and unary sign rules, and the rule on computed member access.
The Python function is renamed for what it checks, `selfcheck_p2a_asks_no_runtime`; a name that said "only abstains"
would now claim what a source scan cannot show.

**Closed in this pass**: a Unicode escape inside an identifier (`toLowerCase` written with `a`, or the braced
form) hid a name from every rule of the port's scan. The scan now decodes escapes in code before it reads names, and
refuses a backslash left in code. The `for … of`, `for … in`, destructuring and optional-call forms of A-2 were ways
to store or to call a mutator unseen; with the store scan gone there is nothing for them to slip past, and the name
rules read a banned name whatever the form of the call (two plants pin that an optional call and an optional chain of
a banned method are refused). The port's scan also refuses `~`, `|`, `&`, `^` and `<<`, which the block does not
use, and `>>` anywhere but after `.length` (its two uses there); each converts a string as a unary `+` does (C-6).

**Not covered, and not chased**: a name built from strings at run time; `eval`-like indirection beyond the names
refused; an alias of a refused function carried through data; in Python an f-string of a list (it reads `repr`); in
the port a binary `*`, `-`, `%` or relational operator, or `++` and `--`, on a string (each converts with the
engine's white space; the block gives them numbers only); the numeric parameters of built-in methods. The README
lists these.

## 4. A correction to the decision text of 2026-10-04, and the count seam

The lead's decision, as the ninth note (§1) and the README record it, says the `apart` switch dates from pass five
and that every blocker of passes five to eight came from it. The notes show otherwise, and the lead has confirmed it:

- the fourth and fifth notes do not mention `apart`. Pass five's gate-agreement rule was the **count seam** (fifth
  note, C-1), written against the blocker the review of pass four's head found: a count one port reads and the other
  does not;
- the review of pass five's head found the next blocker, a tests claim standing where one of the seam's two counts
  stood; `apart` entered at the **sixth** pass (sixth note, C-1) to clear it;
- the blockers the reviews of passes six, seven and eight found (the case tables of two Unicode versions, a `def`
  that ends its line, a regex backtrack in the scope window) came from `apart`.

So: the switch caused the blockers of passes six, seven and eight, and the count seam the blocker of pass five. The
ninth note is not edited; the README, the CHANGELOG and the block's comments are corrected in this pass.

**The count seam is still in the layer.** It is the phrase `seam` (`_p2a_seam`, `_p2aSeam`): over a diff with a dot
twin, a count claim read from the summary (not a declared one) is withheld in both ports when the summary holds a
white space only one port's `\s` reads between a character that can end a count and one that can begin one, or `file`
spelled with a letter only CPython folds. It **withholds**; it never keeps a verdict, so it is not a switch of the kind
the lead removed, and it cannot leave a false accusation standing. What it does now, under the restated bar: nothing
for C(i) or C(ii) (C-5 below), which hold with or without it; it moves C(iii) figures, in both directions, at a cost
in recall.

## 5. The EXTERNAL-1 shelf paragraph

Found on 2026-10-05 by a checker of the lab's chat update and confirmed by the lead against the shelf:

1. **69,058 counts runs of rows, not pull requests.** The ninth pass's tally walked table `f` in row order and took a
   change of `pr_id` for a new pull request. Nine ids are split across runs of rows that do not touch (eight of them
   among the bodies read), so some pull requests were read twice, each time with part of their files.
2. **The truth source was misnamed, and misused.** The paragraph says the verdicts were "judged against GitHub's own
   file list". Table `f` is the AIDev dataset's `pr_commit_details`: one row per file **per commit** (228,693 of
   1,409,108 pull-request-and-file pairs have more than one row). The tally built one diff group per row and took the
   last row's status as the file's. `external1_harness.py` records (`_fold_statuses`, its CORRECTION of 2026-08-31)
   that this was a defect the lab had already corrected once.

**What this pass does.** The measurement is run again: per pull request id, through the index on `pr_id`; the diff
rebuilt by the committed harness's own `reconstruct` (one group per file, with the net status of `_fold_statuses`,
every commit's patch under it); both ports. Reported: pull requests read, decided claims in reach, claims withheld by
phrase, records outside the relation, the bar C row, and how many rebuilt diffs `main`'s parser reads back as the
folded listing (the harness's own check of its reconstruction).

**The judged figures are removed** (146 false verdicts, 115 withheld; 6,793 right, 2 withheld; 4,920 unjudged). The
fold does not make that judgement sound. It is order-free because the shelf carries no commit order: a file added
and later removed nets to "modified" by rule, a merge commit's rows list files the pull request does not change, and a
commit the dataset lacks cannot be seen. The shelf therefore holds no pull-request file list to judge a verdict
against, only the listing the diff itself is rebuilt from, and a verdict that agrees with its own input is not thereby
right. In their place the paragraph gives a figure that needs no truth: for every decided claim in reach, whether any
of the four committed variants of `main` (V97, V121, V101, all three) reads it otherwise, and how many of those the
overlay keeps. Under bar B's but-for reading a kept claim that no variant reads otherwise cannot be an attributable
miss, whatever the truth is. The three-matching reading against the folded listing is kept in the scratch output and
is not printed in the README.

## 6. From the coverage review's minors

- **The limit of `redefined`** (B-2). The README names `-U0`. The limit is wider: a test defined again is invisible
  to the overlay whenever its earlier definition lies outside the hunk's context lines, at any context width; `-U0` is
  the extreme. The reviewer's count on the committed truth world, at git's default three lines: 8 false tests verdicts
  of this kind kept (4 CONTRADICTED, which are false accusations, and 4 VERIFIED), 6 more at context 0 and 1, against
  12 withheld as `redefined`. They are not misses under the bar's four-variant attribution (the committed V101 pairs
  by removed lines) and the fact is not in the diff; withholding every tests claim on a modified file would cost the
  recall the rule exists to keep. The README says all of this; no rule changes.
- **The decorated-world exchange** (B-3). The README gave what the removed switch cost as one ratio (20 false
  CONTRADICTEDs now withheld against 8 right ones lost, on 26 pinned cases). On a truth-judged world under seventeen
  decorations (20,400 inputs, Python raw door, the reviewer's figures) the removal withholds 903 more false
  CONTRADICTEDs and loses 1,812 more right ones; the parent head kept 796 of 12,559 attributable false verdicts
  there and the head keeps none. The operator confirms the lead's decision of 2026-10-04 on both numbers.
- **The per-rule table** (B-4) gets its decorated-world half, and one sentence for a loss older than pass nine: a
  scope claim that shares its sentence with an accented letter, a CJK character or an emoji outside O-11's ranges after
  the word `only` is withheld in both ports whatever its truth.

## 7. The findings of the reviews of `c69b161b`

### By construction and cost

| id | severity | finding | disposition |
|---|---|---|---|
| A-1 | major | The Python self-check does not hold the block to abstention: `_p2a_holder` does not carry a record container through a default argument, `*args`, `**kwargs` or a class attribute. Three plants gated on a claim count no committed input has drop or add claims and pass every committed test but the byte pins a comment trips too. | **fixed by construction** (§2): the rules run on a copy and APPLY alone writes. The alias reader is deleted, not extended. Each plant is a committed case that moved the record at `c69b161b` and cannot now. |
| A-2 | major | The port's token and store scan does not hold the port's block to abstention, and one hole (a Unicode escape inside an identifier) also defeats the name rules that guard bar C. | **fixed by construction** for the record (§2; the store scan is deleted); **fixed** for the name rules (§3: escapes are decoded before names are read). The nine plants are committed cases. |
| A-3 | minor | Neither static check constrains what the three allowed stores write, when, or what the function returns or raises. | **fixed by construction**: the two claim stores and the gate verdict store are APPLY's, written from the tables and the record; APPLY returns the record it was given; APPLY's text is pinned. |
| A-4 | minor | Four README sentences about the static checks are false as written. | **fixed**: the section is written again (§3) and no longer says a check refuses every way to write the record. |
| A-5 | minor | The README's import cost (27 to 30 ms against `main`'s 6) is not what the reviewer measures (44 ms on 3.12.10, 37 ms on 3.14.2; a fresh interpreter 31 to 38 ms → 106 to 158 ms). | **fixed**: measured again at the head, with its method and as a ratio (§9). The block's patterns are not made lazy in this pass: that moves no decision but is a change of its own. |
| A-6 | minor | Record: no superlinear cost found. Two notes for the README's table: the largest constant factor is on a diff of many unchanged `def` lines with one added test; the eighth review's figure does not say which diff it was measured on. | **disclosed**: the context-line row is added and the figure's diff named. |
| A-7 | minor | Record: bar A held on every run. One note: with a `strict` object whose truth raises or changes between reads, the Python door reads it once more than `main` when the overlay withholds `main`'s only CONTRADICTED. | **disclosed, operator**: APPLY evaluates `main`'s formula over the final claims, as bar A words it; no bool or int is affected. |
| A-8 | minor | The reviewer's process notes (one `sed -i` on its own scratch script, heredocs that lost backslashes). | **recorded**. |

### Coverage and recall

| id | severity | finding | disposition |
|---|---|---|---|
| B-1 | record | Bar B holds on every set read; no committed #161 reproduction is missed but JOINT_121; recall reproduced. The fifth review's generator reproduces only under `PYTHONHASHSEED=0`. | **recorded**. |
| B-2 | minor | `-U0` understates the limit of `redefined`. | **fixed** in the README (§6). |
| B-3 | minor | The README quantifies the switch's removal with one ratio; on a decorated truth world the exchange runs the other way. | **disclosed** (§6), for the operator. |
| B-4 | minor | Five rule-and-verdict cells withhold more right than false verdicts; the README's table is the plain world's only, and one loss is named nowhere. | **disclosed** (§6). No rule change: `dot_earliest` and `shape` are required by bar B as worded; O-6, O-8, O-10, O-12, O-14 and O-15 stand. |

### Cross-port

| id | severity | finding | disposition |
|---|---|---|---|
| C-1 | record | C(i) and C(ii) hold on every input measured; the README's C(iii) figures reproduce. | **recorded**. |
| C-2 | minor | Of 33 one-edit asymmetries planted in one port, 11 pass every committed behaviour test, and each breaks C(i) or C(ii) on a two-line input. | **fixed**: the reviewer's inputs become cross-port cases with both ports' decisions pinned, and each edit a one-port plant the tests must refuse. |
| C-3 | minor | On a runtime the bar C pins were not measured on, `_bar_c` compares with the row sorting yields, so the newer-engine set fails on an engine below Unicode 16 for no fault of the code. | **fixed**: the reference row is the one for the running interpreter's Unicode version where there is one; the newer-engine set is keyed by what the two runtimes fold. |
| C-4 | minor | C(i) and C(ii) hold only on an engine whose white space is the port's static table; in a browser running the bookmarklet nothing checks that. | **disclosed** in the README beside the other caveats: a fallback phrase in a record is the sign. The panel is not edited. |
| C-5 | minor | The ninth note's sentence that `case_count`, `seam`, `extract`, `divergent` and O-11 stay because "each still carries C(i)" rests on dropping a rule from one port. Dropped from both, the summary-side guards (`seam`, and `extract` for an ASCII path, name, number or prefix in odd company) leave C(i) and C(ii) as they are. | **corrected here**: those guards are kept for the C(iii) figures and, for paths outside ASCII, for coverage; C(i) and C(ii) do not depend on them. Keeping or dropping the summary-side part is listed as operator option **O-16**, not taken, with the reviewer's figures (committed inputs: claims withheld 2,627 → 2,319 in Python; gates apart only under the overlay 6 → 11). `divergent` is different: the registrations come from each port's own line split. |
| C-6 | minor | The port's scan passes a unary `~`, a bitwise `\|` and a postfix `++` on a string. | **fixed** for `~` and the bitwise operators (refused; `>>` allowed only after `.length`); **disclosed** for `++` and `--`. |

### Integration, CI and docs

| id | severity | finding | disposition |
|---|---|---|---|
| I-1 | minor | With the pass-8 body still on the pull request, pushing this head turns the repository's own body-against-diff job red: the diff is now under GitHub's line limit, and that body holds a sentence the gate reads as a tests claim. | **operator**: replace the body before or with the push. This pass writes a new body in scratch and checks it with `gate_diff_text` against `git diff origin/main...HEAD`. |
| I-2 | minor | C-3 again; and where CI's Node carries Unicode 17 no CI job asserts a bar C(iii) figure exactly. | **fixed** as C-3; **operator**: add the pins for CI's Node after the earliest CI run, or pin Node in the workflow. |
| I-3 | minor | The C(i) comparison does not read the defect tag: `_seen` is (verdict, phrase key). | **fixed**: the tag is compared too, with a one-port plant that swaps a tag. |
| I-4 | minor | The record disagrees with itself on when the switch entered. | **corrected** (§4). |
| I-5 | minor | The CHANGELOG gives the Windows flavour's figures without saying so, and "22 to 24 ms" where the README's own numbers give 21 to 24. | **fixed** in the new entry. |
| I-6 | minor | The reviewers' scratch root is 150 characters long, so five test modules cannot run in a clone under it on Windows. | **recorded**, for the operator. |

## 8. What this pass does not change, and what the operator decides

- **No decision changes.** The rules, REACH, the phrases but `malformed`, the reason form and the two hooks are as
  they were; the records at this head equal `c69b161b`'s on every input measured (§9).
- **No repair.** PATH-2a never gives VERIFIED where `main` was wrong. **G-P1** is not met; that is the operator's
  decision.
- **`main`'s reader** is untouched in both ports.
- **Nothing committed is edited:** no receipt, certificate, sworn file, PREREG, RESULT, ANALYSIS, AMENDMENT, ERRATUM,
  earlier NOTE or the charon log.
- **Operator options** O-1 to O-15 stand as disclosed, O-13 (the five joint #121 reproductions) included, and the
  behaviour under `--strict`; O-16 is added and not taken. For the operator at merge: the lead's decision of
  2026-10-04 with the corrected history (§4) and both exchange figures (§6); the pull-request body (I-1); the CI pins
  (I-2).

## 9. Measured again at the head

Written into the README with what each figure measures: the relation on the committed inputs in both ports and at
the git door; that the records equal `c69b161b`'s on the committed inputs and on a seeded adversarial set, in both
ports; the hostile DECIDE runs; the truth figures and the abstention pins; bar C on the committed inputs and the
seeded sets; recall with `path2a_recall.py`; the committed timing and memory cases, and the cost of the copy; the
import cost; the shelf as §5 describes it; the decorated-world exchange and the per-rule table; the bookmarklet's size
and its equality with the port; the size of the PR's diff. A figure quoted from a review and not measured again is
named as the reviewer's.
