# FINDING — thirteen certificate fields that no code reads

Fathom Lab · 2026-09-09 · **A screening result, not a defect list.** Produced by
`free_field_census.py` against git commit `8ee23691`, reading schemas and source from HEAD rather
than the working tree. Not sworn. Two of the thirteen were then checked by hand and are reported
separately below because they are worth more than the rest.

## Why this exists

`THE_BOUNDARY_2026_09_09.md` placed four defects in a class it called *reachable by nothing*, and
all four turned out to be reachable. Every time, the shape was the same: a field was written into a
certificate, some other logged byte constrained it, and no code compared them. The author examined
one certificate, found nothing contradicting the field, and concluded nothing could.

Four misses by inspection is an argument for doing it mechanically. A field that no code reads can
hold anything, and that is where the next misclassification will come from.

## The number, with the tool's own precision attached

| | |
|---|---|
| declared fields across 12 schemas | 227 |
| read somewhere in `styxx/` | 214 |
| no read found by the first screen | 18 |
| of those, actually read (found by the recheck) | 5 |
| **surviving candidates** | **13** |
| screen precision on its own candidate list | 13/18 = **0.72** |

The precision line is the point. The first screen looks for `"name"` or `.name` and misses a field
read as a bare identifier — a keyword argument, a local, a destructured key — so it called
`certified`, `grader`, `items_blob`, `loader` and `top_p` free when they are read. The tool reports
that against itself rather than publishing 18. A third failure mode was found by hand and fixed: it
was reporting `$defs.alias.observed_at` as a top-level `observed_at`, which sends a reader to the
wrong part of the schema.

**A fourth failure mode, found by pinning the finding as a test.** `tests/test_v8_free_fields.py`
asserts the desired state under `xfail(strict=True)`, so a repair turns the suite red and announces
itself. Within minutes of being written it fired on `observed_at` — and not because anything had
started reading the alias subject's field. A new module took `observed_at` as its own parameter name
for an unrelated quantity, and leaf-name matching cannot tell two fields of the same name in
different types apart. This is the worst of the four modes, because it reports a field as **read**
when nothing reads it, hiding gaps rather than inventing them. The pin now covers
`observed_model_id`, which is distinctive enough to survive the method; `observed_at` is checked
only for still being declared. Every count in the table above inherits this mode and is therefore an
**upper bound on coverage**: the true number of unread fields is at least thirteen.

**This is a screen, not a measurement.** Text search cannot see a field read through a variable, a
schema walk, or a canonicaliser that digests whole objects, so a name here is a question — *what, if
anything, would contradict a forged value?* — and today's record says the answer is usually
something. The real measurement is mutation: change each field, run the suite, record what goes red.
That needs a stable tree and is owed.

## The two that matter

**An alias subject's observed identity is declared and read by nothing.** `subject.json` supports two
subject kinds, `weights` and `alias`, and applies `$defs/alias` when the kind is `alias`. That block
declares `observed_model_id` and `observed_at`. Neither appears in any source file under `styxx/`,
by either search.

An alias subject is a hosted model behind an API, which is exactly the case where identity **cannot**
be established by hashing weights, so the observed fields are the only identity evidence there is.
This is the same defect as the dead subject guard repaired earlier today — a verifier comparing a
certificate to itself because nothing reported what was actually loaded — left unrepaired on the
subject kind where it does more damage. The earlier repair made `Runner.subject()` a required
protocol member for weights subjects. It did not reach here.

**A sublog's chaining fields are declared and read by nothing.** All five of `sublog_id`,
`prev_root_hash`, `prev_tree_size`, `entries_sha256` and `count_since_prev` are unread. Those names
describe a sublog's linkage to its parent log; unread, a sublog chains to nothing that is checked,
which is the same family as the truncation attack already open against the main log.

**That question is now answered, and the answer is sharper than the screen suggested.** The type is
declared live: it is in the type set in `consts.py`, its permitted reference roles are set in
`cert.py`, it has a schema, and it appears in the test fixtures. The schema **requires all eight**
body fields — `sublog_id`, `tree_size`, `root_hash`, `prev_tree_size`, `prev_root_hash`,
`consistency_proof`, `count_since_prev`, `entries_sha256` — which together specify a complete
RFC 6962 chaining structure from a previous head, with an entry count and a digest over the entries.
It is a well-designed mechanism, fully specified, and **nothing anywhere produces one or verifies
one**. A certificate type with neither a producer nor a consumer.

This matters more after today's truncation work than it would have this morning. A verifiable chain
from a previous head is precisely the missing piece that attack turned on, and the schema has been
carrying the design for it the whole time. The resolution is a decision, not a repair: implement the
type or remove it. A required field nobody writes is worse than an absent one, because a reader who
sees the schema reasonably concludes the guarantee exists.

## The rest

`action.json`: `body.action_kind`, `body.action_sha256`, `body.context_sha256`.
`promotion.json`: `body.code_sha256`, `body.scope.task_family`.
`recipe.json`: `materials.chat_template_source`.

And one field read by code with no test mentioning it, `recipe.json` `sae.attribution` — a read that
can be deleted without anything going red.

## Limits

One commit, one repository, one search method with a measured precision of 0.72 on its own output
and at least one failure mode it cannot measure. Nothing here has been reviewed outside this
session. The alias finding and the sublog finding were confirmed by hand; the other eleven were not,
and some of them are probably false.
