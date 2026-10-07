# NOTE — PATH-2a, eleventh pass: DECIDE handed nothing of `main`'s, the boundary of bar A restated, APPLY reaching no prototype after DECIDE, and each finding of the reviews of `16daa725`

## 0. Status

2026-10-05. Branch `fix/diffgate-abstain-where-wrong`, head `16daa725` (pass ten). `main`'s two reader files are
unchanged (sha256, LF: `9b620e00…` and `06688702…`). The earlier notes are listed in the README's *PATH-2a* section;
none is edited.

What this pass had to read:

- four finished reviews of `16daa725`, each "fix first", each for one shared blocker (by construction and cost: one
  blocker, one major, five minors; coverage and recall: one blocker, one major, two minors, two records; cross-port:
  one blocker, one major, four minors; integration, CI and docs: one blocker, five minors);
- the lead's direction of 2026-10-05 for this pass (§1), which outranks the usual order of work where the two differ;
  the direction for pass ten still stands.

**This note is written before this pass's code and committed alone, ahead of it.** A figure in this note is a
reviewer's, measured at `16daa725`, and is named as such; §9 says what is measured again at the head that carries the
code. If the committed code departs from this note, the departure goes into a note of its own. Finding identifiers are
this note's: A-n, B-n, C-n and I-n number each lens's findings in the order its review lists them.

Dispositions: **fixed** (a change in this pass, with a test that fails on `16daa725` where the finding is a defect);
**disclosed** (kept, and stated in the README, the CHANGELOG or here); **operator** (left to the operator);
**recorded**.

## 1. The lead's direction for this pass

1. **Blocker A, reach through the copy (Python).** DECIDE's copy was built from `main`'s own `DiffClaim`, so
   `type(seen.claims[0])` was `main`'s class, and the copy's container had a Python-level `__init__` whose
   `__globals__` is the module. Fix: a class of the block's own with `__slots__` and no method for each claim, a
   container with no `__init__`, so that nothing reachable from DECIDE's first argument as data is `main`'s; drop
   `DiffClaim` from the names the block may use if nothing else needs it. The reviewers' class-property and
   class-setattr functions become committed cases, asserted inside the relation. In the port, if it costs little,
   APPLY uses built-ins captured when the module loads rather than ones reached through the copy's prototype, with
   the cost measured; otherwise that is the stated limit.
2. **The boundary, restated by the lead.** Bar A by construction is claimed for a DECIDE that uses what it is handed
   as data and calls the facts function it is given, whatever data it returns. It is not claimed for code that reaches
   around that through reflection: a class reached by `type()`, a function's `__globals__` or closure, a frame,
   `sys.modules`, or a built-in patched through a prototype. No guard inside one Python interpreter or one JavaScript
   realm can close those, and DECIDE is the lab's own code, so the hostile tests guard against an honest mistake, not
   an attacker. This is said in the README's section on what holds bar A, in the docstrings of DECIDE and APPLY and
   here, replacing every sentence that says the record leaves APPLY as `main`'s whatever DECIDE does. The
   `facts.__globals__` route is committed as a hostile case pinned as the stated limit: asserted to leave the
   relation, so that a later change that closes it is noticed.
3. **Major B, the single-prefix scope gap: no rule change in this pass.** The README sentence and the code comment
   that say a kept CONTRADICTED there needs two rendering faults are corrected: `git diff --no-prefix` over a
   repository whose top directory is `a/` or `b/` and holds its dot twin is enough. The reviewer's four cases are
   pinned in the truth module as known kept attributable claims, counted and asserted. The case goes to the operator
   under O-10 with the reproduction and the narrow rule the reviewer names, unmeasured.
4. **Minors taken.** APPLY ignores a decision whose phrase is `error` or `malformed` (those are APPLY's own), in both
   ports, with a hostile case. The Python APPLY tests exact types (`type(x) is list`, `tuple`, `int`, `str`) instead of
   `isinstance`. The seam row and the `case_count` CONTRADICTED figures go into the decorated per-rule table, or the
   README says that its world does not exercise them, and they are carried into O-16.
5. The lead pushes; this pass does not. The worktree is left clean with everything committed.

## 2. What DECIDE is handed, and what bar A by construction covers

**At `16daa725`.** APPLY built each copy as `DiffClaim(kind, "", {detail fields}, verdict, why)`, and the container
as `_P2aSeen(claims)`. Both premises of the tenth note's sentence "nothing DECIDE can reach through its argument is
shared with the record" (§2 there) were false: `type(seen.claims[0])` was `main`'s `DiffClaim`, the class of every
claim in the record, so a DECIDE that set a property or a `__setattr__` on it made APPLY's own two stores
(`c.why = …`, `c.verdict = …`) run its code on the record; and `type(seen).__init__.__globals__` was the module, with
APPLY's own tables in it. The reviewers' figures: a `__setattr__` on the class put 580 of 958 raw-door runs and 42 of
192 git-door runs outside the relation (integration); CONTRADICTED became VERIFIED and FAIL became PASS at both doors,
and the patch persisted into later calls, `main`'s own reader included (construction). The committed hostile tests
could not see it: their records came from the reference module, whose `DiffClaim` is another class. The tenth note is
not edited; this paragraph is its correction.

**From this pass, in Python.** Each copy is an instance of `_P2aClaim`, a class of the block with
`__slots__ = ("kind", "verdict", "why", "detail")` and no method; the container is an instance of `_P2aSeen`, with
`__slots__ = ("claims",)` and no method; both are built by attribute stores, so no function of the module is
reachable from them. What DECIDE's first argument reaches as data is then a `_P2aSeen`, a list, `_P2aClaim`s, dicts,
strings and booleans. A DECIDE that patches `_P2aClaim` or `_P2aSeen` (both reached by `type()`) changes only DECIDE's
own copies, in this call and later ones; APPLY reads nothing of them after DECIDE returns, and builds the next copy
inside its `try`, so a patch that raises there withholds with `error`. `DiffClaim` leaves the lint's list of names the
block may read: nothing in the block uses it now.

**The boundary, as the README, the docstrings and the CHANGELOG will state it** (§1, item 2). Covered: a DECIDE that
uses what it is handed as data and calls the `facts` function it is given, whatever data it returns. Not covered:
reflection (a class reached by `type()`, a function's `__globals__` or closure, a frame, `sys.modules`, `gc`, a
built-in patched through a prototype). The hostile tests guard against an honest mistake in the lab's own DECIDE, not
an attacker. The second argument is the door's `facts`, a function whose `__globals__` is the module: that route is
committed as a case that leaves the relation (it patches `main`'s `DiffClaim` through `facts.__globals__` and the
test asserts that records move), so that a change that closes it is noticed and the README sentence moves with it.

**In the port** (§1, item 1; C-2, I-2). The copy was already plain objects, but APPLY, after DECIDE returned, read the
phrase table with an inherited lookup (`_P2A_PHRASES[key]`), checked tags with `Array.prototype.includes`, kept its
plan with `push`, iterated with `for … of` and destructuring, read the claims in reach through a `Map`, and computed
the gate verdict with `some`. A DECIDE reaches `Object.prototype` and `Array.prototype` from its argument, so in one
realm (as in a page) it could write a phrase outside the table, accept any tag, or receive the record's claims as
`this`. The cross-port reviewer's figures over 962 runs: the inherited phrase broke the relation in 604, an
`includes` patch in 634, a `some` patch in 950. From this pass, after DECIDE returns APPLY reaches no built-in through
a prototype: it reads by index (`k < x.length`, `x[k]`) on arrays it or `main` built, compares with `===`, tests an
array with `Array.isArray` captured when the module loads, finds a phrase key in a list of the table's own keys read
once at load, compares tags by an index loop over the kind's own tag list, and writes its picks into arrays it filled
before DECIDE ran, so that no store after DECIDE meets a setter on a prototype. The port's lint refuses the words
`Object`, `prototype` and `__proto__`, so the tables are not frozen; they are not reachable from DECIDE's argument.
What stays outside, and is said: a prototype patch that persists into a later call (it reaches `main`'s own reader of
that call before APPLY does), and the function constructor reached through `constructor`, which is `eval` by another
name. The cost of the change is measured (§9).

**Tests that fail on `16daa725`.**
- Python: DECIDE functions that set a property on `verdict` of their claims' class, a `__setattr__` on that class, a
  descriptor that swallows the reason, and a `__setattr__` on the container's class; each through APPLY on records
  built by the module's own classes (not the reference module's), and through both doors with the module's own
  DECIDE replaced, in both strict modes; each asserted inside the relation. One case leaves the class patched, then
  asserts that the next call returns `main`'s record where nothing is in reach and stays inside the relation where
  something is. The `facts.__globals__` case is asserted to leave the relation. Every patch is undone after its test.
- The port: in one realm (the record built by the port's own `main` reader, in the realm DECIDE runs in), a phrase put
  on `Object.prototype`, an `includes` that accepts any tag, a `some` that answers false, a `push` that rewrites what it
  is handed, and an `Array.isArray` that says yes to anything; each asserted inside the relation, with what APPLY must
  then do (ignore, decide, or fall back with `malformed`), and each prototype restored after its run.

## 3. Exact types, and the two phrases that are APPLY's own

**Exact types (Python).** APPLY takes a result only if `type(got) is list`, a decision only if `type(d) is tuple`
and `len(d) == 3`, and its fields only if `type(i) is int`, `type(key) is str` and `type(tag) is str`. A subclass of
any of them is not taken: a list subclass is `malformed`, a tuple, int or str subclass is ignored. `type()` reads the
object's own type, not a `__class__` it may claim. Four committed hostile cases change what they assert, each now
stricter than before: a list that raises when read (`error` → `malformed`); a tuple that lies about its length, a key
whose hash raises, and an index that says it equals every index (each → ignored, the record `main`'s). The lint
allows `type` only as `type(<one argument>) is` (or `is not`) a builtin type, and `int` only there.

**`error` and `malformed`** (A-4). They say APPLY's own fallback fired, and a reader takes them as that signal; at
`16daa725` DECIDE could return them as ordinary decisions (the construction reviewer counted 6,446 claims carrying
`malformed` beside 1,262 honest decisions in one hostile run). From this pass APPLY ignores a decision whose phrase is
either, in both ports; `unreproduced` and `unparsed` stay DECIDE's, which returns them. The hostile case "one index
twice, with two phrases", whose leading phrase was `error`, now withholds with its second phrase, `unparsed`; a new
case returns only those two phrases and must leave the record `main`'s.

## 4. The lints

Lints against an honest future edit, as the tenth note named them; three changes and one disclosure.
- **`%`-formatting** (A-6). `'%s' % ([claimed],)` reads `repr` as an f-string of a list does. The block formats with
  `%` nowhere, so the Python lint refuses `%` whose left side is a string literal, whatever the right side.
- **Template interpolation** (C-3, I-4). The port's scan read a template literal whole as a string, so code inside
  `${…}` met no rule. It now reads each `${…}` as code (nested templates too). The reviewers' plants (`Number`,
  `parseInt`, a unary `+`, `==`, `Math.max`, `| 0`, a computed call, `Object.keys`, a regex literal, each inside an
  interpolation) are committed refused cases.
- **`type`** as above (§3).
- **Disclosed** (C-4): the port's scan has no allow-list of names, unlike the Python lint, so a call of one of
  `main`'s own helpers that asks the engine (`_hasRealExtension`, `_norm`, `_prefixIsPathShaped`) passes it; the
  README lists that among what the lints do not cover. The f-string of a list stays disclosed.

## 5. The single-prefix scope gap (B-2, B-3; operator option O-10)

The coverage reviewer built, with git, four repositories whose top directory is `b/` (or `a/`) and holds `.b/` (or
`.a/`) beside one other directory, and read `git diff --no-prefix`: `main` strips the real `b/`, keys `.b/z.py` as
`b/z.py` (#121), reads `b` as a path and accuses "paths outside 'b'" where every changed file lies under `b/`; V121 says
the prefix is not a path, so the false CONTRADICTED is attributable under the committed but-for definition, and the
head keeps it, in both ports (`np-b-dotb`, `np-b-dotb-created`, `np-b-dotb-sentence`, `np-a-dota-deleted`). The
README's sentence and the comment in `_p2a_only` said such a verdict "needs two independent faults in how the diff was
rendered"; one is enough. They are corrected, and the README stops filing renderings over real `a/` or `b/`
directories among gaps outside the three mechanisms. No rule changes: the four cases are pinned in the truth module as
known kept attributable claims, counted (four in Python, four in the port) and asserted, so that a change that closes
or widens the gap moves the pin. For the operator, unmeasured: the reviewer's narrow rule (`shape` also for a single
prefix that is a path for `main` and not for V121 when `main`'s key for the prefix is `a` or `b`), or a trigger on
`diff --git` headers that carry one path on both sides with no `a/` `b/` split.

## 6. The seam figures (B-4; operator option O-16)

The decorated per-rule table has no `seam` row because its world holds no seam: at `16daa725` the rule fired there 0
times (that world's summaries are decorated with characters both ports read alike as white space, or that no count
template meets); this is to be confirmed at the head and said in the README. The reviewer's worlds do exercise it,
and are given as the reviewer's: on the builder's four families under seventeen other decorations (seed 51005, 4,080
inputs) `seam` on CONTRADICTED withholds 89 right against 54 false in Python and 50 against 32 in the port; on real
git renderings (seed 2027) 80 against 60, 40 against 20, and 36 against 27 at the git door; `redefined` on VERIFIED
there 39 right against 0 false. A right CONTRADICTED count is withheld whatever number is claimed, which can turn a
right FAIL into PASS ("7 files changed." over three files, beside U+001F between two words). The `case_count`
CONTRADICTED row is in the table already (110 false against 27 right in Python). O-16 (dropping the summary-side
guards from both ports) carries these figures, and the reviewer's narrower form of `seam` (only for a claimed number
between `main`'s count and the count with the twins apart, the range the count rule already reads) is listed beside
it. Neither is taken.

## 7. The findings of the reviews of `16daa725`

### By construction and cost

| id | severity | finding | disposition |
|---|---|---|---|
| A-1 | blocker | DECIDE's copy is made of `main`'s `DiffClaim`; a DECIDE that uses only its argument patches that class and moves the record at both doors. | **fixed** (§2): block-private slotted classes with no method; committed cases at APPLY and at both doors on the module's own records. |
| A-2 | major | The README, the CHANGELOG, four docstrings and the tenth note say nothing DECIDE does reaches the record. | **fixed** (§2): every such sentence replaced by the lead's boundary; the tenth note corrected here. |
| A-3 | minor | APPLY trusts tables that are mutable module state, and the checks read the live tables. | **fixed in part**: in the port APPLY reads the phrase keys from a list taken at load, and the tables are not reachable from DECIDE's argument; the committed relation and the port's hostile harness compare against copies of the tables taken before any DECIDE runs. In Python the tables are reachable only through module globals, which is the stated limit (§2); freezing them needs an import the block does not make. **disclosed**. |
| A-4 | minor | APPLY accepts `error` and `malformed` from DECIDE. | **fixed** (§3), with a hostile case in each port. |
| A-5 | minor | Cost: linear against `main` on about 70 families; two notes for the table (a single-line `def` row at ×10.5 and ×17; `zone_regime` drifting from 0.08 to 0.13 of `main`). | **disclosed**: the row and the drift are added to the README, as the reviewer's figures. |
| A-6 | minor | `'%s' % ([claimed],)` passes the Python lint and reads `repr`. | **fixed** (§4). |
| A-7 | record | Bar A holds with the shipped DECIDE everywhere measured; no trace of the switch. | **recorded**. |

### Coverage and recall

| id | severity | finding | disposition |
|---|---|---|---|
| B-1 | blocker | The same as A-1, with the port's prototype route as a note. | **fixed** as A-1 and C-2. |
| B-2 | major | Truth-judged misses of a single-prefix `only_touches` CONTRADICTED over `--no-prefix` output of a repository whose top directory is `a/` or `b/`. | **operator** (O-10), no rule change by the lead's direction (§5); the four cases pinned as known kept, both ports. |
| B-3 | minor | "Needs two independent rendering faults" is wrong: one suffices. | **fixed** in the README and the comment (§5). |
| B-4 | minor | `seam` withholds a right CONTRADICTED whatever number is claimed; two cells outside the README's five withhold more right than false on the reviewer's worlds; no `seam` row. | **disclosed** (§6), carried into O-16; the narrower `seam` listed for the operator, not taken. |
| B-5 | record | Bar B holds but for B-2; records equal `c69b161b`'s. | **recorded**. |
| B-6 | record | Recall reproduces; every abstention on `main`'s corpora withholds a VERIFIED the diff's own listing shows false. | **recorded**. |

### Cross-port

| id | severity | finding | disposition |
|---|---|---|---|
| C-1 | blocker | The same as A-1, and `_P2aSeen.__init__.__globals__` reaches the module. | **fixed** (§2): no `__init__`; the `facts.__globals__` route pinned as the stated limit. |
| C-2 | major | The port's APPLY checks through prototypes DECIDE reaches. | **fixed** for one call (§2): nothing after DECIDE goes through a prototype; committed one-realm cases. A patch that persists into a later call is the stated limit. |
| C-3 | minor | The port's scan does not read code inside `${…}`. | **fixed** (§4). |
| C-4 | minor | The port's scan has no allow-list; a reuse of `main`'s helpers passes. | **disclosed** (§4). |
| C-5 | minor | A non-`Exception` `BaseException` from DECIDE leaves the Python door, where the port withholds. | **disclosed**; whether APPLY should catch more is the **operator**'s call. The README says "if DECIDE raises an `Exception`" and names what propagates. |
| C-6 | minor | `_nearest_pin` falls back to the row that sorts ahead for an unpinned Unicode version. | **operator**: no CI interpreter is affected; noted for when a Unicode 17 CPython is in CI. |

### Integration, CI and docs

| id | severity | finding | disposition |
|---|---|---|---|
| I-1 | blocker | The same as A-1. | **fixed** as A-1. |
| I-2 | minor | The CHANGELOG's list of what the construction does not cover leaves out the prototype route. | **fixed** (§2), the CHANGELOG and README restated together. |
| I-3 | minor | "If DECIDE raises, every claim in reach is withheld" holds for `Exception` only. | **fixed** wording (C-5). |
| I-4 | minor | The same as C-3. | **fixed** (§4). |
| I-5 | minor | "26 in Python and 24 in the port" counts the block's own DECIDE in each list. | **fixed**: the README and CHANGELOG count the hostile functions apart from the block's own. |
| I-6 | minor | CI notes; a house-rule word in a committed note of the fifth pass. | **recorded**; the note is committed and not edited (the CHANGELOG already says so). |

## 8. What this pass does not change, and what the operator decides

- **No decision changes.** The rules, REACH, the phrases, the reason form and the two hooks are as they were. APPLY
  refuses only what the block's own DECIDE never returns (a fallback phrase, a value of a subclass), so the records at
  this head are to equal `16daa725`'s on every input measured (§9).
- **No repair.** PATH-2a never gives VERIFIED where `main` was wrong. **G-P1** is not met; that is the operator's
  decision.
- **`main`'s reader** is untouched in both ports.
- **Nothing committed is edited:** no receipt, certificate, sworn file, PREREG, RESULT, ANALYSIS, AMENDMENT, ERRATUM,
  earlier NOTE or the charon log.
- **Operator options** O-1 to O-16 stand as disclosed, O-13 included, and the behaviour under `--strict`. For the
  operator at merge: the boundary of bar A as restated (§2); O-10 with the reviewer's four cases and narrow rule (§5);
  O-16 with the seam figures (§6); whether APPLY should catch a `BaseException` (C-5); the CI pins (C-6, and I-2 of the
  tenth note).

## 9. Measured again at the head

Written into the README with what each figure measures: the relation on the committed inputs in both ports and at
the git door; that the records equal `16daa725`'s on the committed inputs and on seeded adversarial sets, in both
ports; the hostile DECIDE runs in both ports, the class and prototype cases and the pinned limit; the truth figures,
the four known kept cases and the abstention pins; bar C on the committed inputs; recall with `path2a_recall.py`; the
committed timing cases, the cost of the port's APPLY before and after this pass's change, and the import cost; the
seam count on the decorated world; the bookmarklet's size and its equality with the port; the size of the PR's diff.
A figure quoted from a review and not measured again is named as the reviewer's.
