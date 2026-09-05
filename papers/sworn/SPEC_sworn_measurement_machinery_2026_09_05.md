# SPEC — sworn measurement machinery v0.1: packets, sealed keys, canary twins, seat runners and a scorer, committed before any seat runs, and running nothing as a measurement

Fathom Lab · 2026-09-05 · **A spec, not a result, and not a preregistration.** Frozen in its own
commit before any code. Leg 2 of `papers/PLAN_the_next_level_2026_09_02.md`, built to
`papers/sworn/DESIGN_sworn_measurement_v2_2026_09_02.md`, which is not edited — a design is
history too. The design's bars are copied here verbatim and every one of them is marked
*proposed, unsigned*: the operator's signature turns them into a `PREREG_sworn_measurement_<date>.md`
in its own commit, and no seat reads a real document before that commit exists. This document
makes no numeric claim. The only numbers in it are the design's own bars, the machinery's
parameters, and the counts a synthetic dry run will produce; none of them is a measurement of
sworn output, and none may be quoted as one.

## Why this exists

The design names the week of engineering that needs no decision: *the packets, the sealed keys,
the scorer committed before any seat runs, and the canary inserter*. Each of those has a
pattern in this tree already — the sealed-key packets of `agent_claim_packets_build.py`, the
two-sided decoys of `extraction_decoys.py`, the scorer-before-the-panel of
`score_extraction_panel.py`, the receipt-seeing packet of `make_oath_v11_panel_packet.py`, the
`claude -p` clean-config transport of `run_b23_fable.py` — and none of them reads a sworn
document. This spec fixes the file formats, the question texts, the decoy and canary
constructions, the statistics and the gates so that the code can be written once, tested on
synthetic items, and locked by the preregistration without a line of it being chosen after a
seat has spoken.

The reason the scorer is committed before the panel is the reason `score_extraction_panel.py` states in
its opening lines: a scorer authored after seeing the answers is a scorer with a thumb on it.
The reason the keys are sealed outside the repository is that a packet whose truth sits beside
it in the tree is a packet the seats could be handed the target of. The reason the canaries are
planted by the harness into a twin, never into the committed document, is that a committed
receipt is history and a sworn document's bytes are pinned by its sidecar.

## What the machinery is, and the boundary it keeps

Everything under `papers/sworn/measurement/` is an **adapter or a harness**: it reads the
repository through git plumbing, calls model transports, writes files. None of it is imported
by `styxx/sworn.py`, and `styxx/sworn.py` is not modified by this leg. The verifier stays a pure
function of bytes; the machinery hands it bytes and reads what it prints.

Two things follow. The scorer (`score.py`) obtains every verdict it uses by calling
`styxx.sworn.verify` on a sidecar at the commit that sidecar names, or by reading a committed
verdict receipt; it never adjudicates a span itself. And the canary builder establishes that a
planted span is false **without calling the verifier** — with the standard library's `Decimal`
and `bytes.find` — because a canary whose falsehood was established by the instrument under
test would make the canary gate a tautology.

## The directory and the ladder

```
papers/sworn/measurement/
  README.md                 opening line: no measurement runs before the PREREG commit
  common.py                 shared pure helpers: units, projection, majority, kappa, wilson, LF writes
  population.py             the population is a script → population.json
  build_packets.py          canonical text → packet_L.json, packet_R.json, decoys; keys to SEALED
  seal_key.py               the one script that knows the salt → keys/*.sha256
  canaries.py               the canary inserter → twins in SEALED, twins/canary_digest.txt
  seat_claude.py            Claude family, claude -p clean-config transport
  seat_local.py             local family, Qwen2.5-7B-Instruct bf16 on CPU, 3B fallback
  twin_trivial.py           the trivially-swearing twin for G-G1
  score.py                  the scorer, committed before any seat → measurement_result.json
  synthetic.py              synthetic documents, receipts and canned seat answers for the dry run
  dry_run.py                population → packets → canaries → seats (canned) → score, synthetic only
  keys/                     committed salted digests only — never a key
  twins/                    committed digests only — never a twin
  seat_outputs/<family>/    seat transcripts and the per-family ledger (written by seat runs)
  dryrun/                   the synthetic run's inputs and its result, suffixes .syn.json / .json
```

The ladder, in order, each rung a commit:

1. this spec;
2. `common.py`, `population.py`, `population.json`, tests;
3. `build_packets.py`, `seal_key.py`, `canaries.py`, and their in-repo outputs — packets,
   decoys, `keys/*.sha256`, `twins/canary_digest.txt`; the keys and the twins go to the sealed
   directory and never to the tree;
4. `score.py`, `synthetic.py`, `dry_run.py`, the seat runners with their refusal, `twin_trivial.py`;
5. the dry run over synthetic items, its output committed as a receipt, and a short sworn RESULT
   bound only to that receipt's machinery counts;
6. **STOP.** The next commit is the operator's: `PREREG_sworn_measurement_<date>.md` naming the
   signed bars, the pooled `n` for G-C, the local substrate, the cross-family label rule, the
   Claude model alias, and a lock hash over `score.py`, `packet_L.json`, `packet_R.json`,
   `twins/canary_digest.txt` and `keys/*.sha256`. Seats run after that commit and not before.

Every seat runner refuses to run without `--dry-run` unless `git ls-files
papers/sworn/PREREG_sworn_measurement_*.md` is non-empty at HEAD and every `keys/*.sha256` the
packets name is committed at HEAD. The refusal is a `SystemExit` whose message starts with
`REFUSED:`.

## Parameters

Parameters of the machinery, not bars and not measurements. Each is a module constant, and a
change to any of them after the PREREG commit is a moved bar.

| name | value | where |
|---|---|---|
| `SEED` | 20260905 | every shuffle: `random.Random(SEED + k)` with `k` named per use |
| `WINDOW_MAX_UNITS` | 40 | units per Panel L item; windows cut only at blank-line boundaries |
| `N_DECOYS_PER_SIDE` | 15 | two sides, so thirty decoys per panel per family |
| `SEATS_PER_FAMILY` | 3 | seats per packet per family |
| `LEAF_VIEW_MAX_CHARS` | 2000 | a whole-receipt view is cut here and marked `truncated` |
| `EDGE_WORDS` | 3 | words a seat quotes to locate a bracket's start and end |
| `CANARY_RATE` | `all` | every hostable site is planted; the rate is printed as capacity |
| `WILSON_Z` | 1.96 | the interval's `z`; `styxx.mind.wilson` |
| `LOCAL_MAX_NEW_TOKENS` | 1200 | generation cap for the local family |
| `CLAUDE_TIMEOUT_S` | 600 | per call; `run_b23_fable.py` used 120 for one-line answers |

## The population is a script

`population.py` applies DESIGN v2 row 7 mechanically at a pinned commit: a document is in the
in-house arm iff its sidecar is tracked under `papers/`, its stem starts with `RESULT_`,
`FINDING_` or `DECLARATION_`, and its path is not under `papers/sworn/`. The builder's format
documents, the PLAN, the SYNTHESIS, the EXPLORATORY receipt and the DESIGNs are excluded by that
rule and listed with the reason. Nothing is sampled: the rule selects, the script counts.

```
{
 "schema": "styxx-sworn/measurement-population/v1",
 "pinned_commit": "<40-hex: the commit whose tree the rule was applied to>",
 "rule": "<the sentence above, verbatim>",
 "seed": 20260905,
 "documents": [
  {"doc_id": "D01", "stem": "papers/<arc>/<name>", "role": "design_eight" | "prospective",
   "sidecar_commit": "<40-hex from the sidecar>", "document_sha256": "<sidecar document.sha256>",
   "receipt_digest": "<committed .sworn-receipt.json digest>",
   "document_verdict": "<from the committed receipt>", "sworn_total": N, "counts": {...},
   "narrative_sentences": N, "units": N, "fragments": N}
 ],
 "excluded": [{"stem": "...", "reason": "..."}],
 "what_this_is_not": ["not a sample: the rule selects", "no number here is a measurement"]
}
```

`doc_id`s are assigned by shuffling the selected stems with `random.Random(SEED)`, so document
order leaks nothing about arc or date. `role` is `design_eight` for a document whose
sidecar was tracked at the design's commit and `prospective` otherwise. `units` and `fragments`
are defined below and reconciled: the script refuses if `units − sworn_total` does not equal
the committed receipt's `coverage.narrative_sentences`, because a unit set that disagrees with
the receipt is a different splitter, and the join would be measuring it.

The population script refuses to overwrite an existing `population.json`; a re-pin is a new
file at a new commit. A document whose committed receipt no longer re-derives at its commit is
recorded with `document_verdict` as the receipt printed it and flagged `receipt_moved: true` —
a moved receipt is a finding, not an input to hide.

## Units, fragments and windows

The **unit set** of a document is the set of sentence-sized things a label can attach to. It is
computed from the sidecar at its commit and from nothing else:

- every sworn span is one unit, `sworn: true`, carrying its `span_index`;
- the canonical text with every sworn span and every fenced region masked to spaces is split by
  the diffgate splitter `(?<=[.!?])\s+|\n+` — `styxx.sworn._SENTENCE_SPLIT`, byte for byte — and
  every non-empty stripped piece is one unit, `sworn: false`, at its canonical byte range;
- a narrative unit containing no ASCII letter or digit is a **fragment** (a dash left between
  two chained spans, a bare punctuation mark): it stays in the reconciled count, it is excluded
  from every Q1 numerator and denominator, and its count is printed.

Units are byte ranges into the canonical text (UTF-8 bytes, never a Python `str` index). The
count of narrative units must equal the committed receipt's `coverage.narrative_sentences`; the
population script asserts it, and the tests assert it on every population document.

A **window** is a byte range of the canonical text cut only at blank-line boundaries (a run of
two or more `\n`, which the splitter already treats as a boundary), packed greedily so that no
window holds more than `WINDOW_MAX_UNITS` units; a single paragraph larger than that is its own
window and the packet records `oversize: true` on it. The window is the item the Panel L seats
read. Its text is the canonical bytes of the range, tags already stripped — the seats never see
a tag, a receipt, a document name, or which items are decoys.

## The packets

**Panel L** (`packet_L.json`) asks whether a sentence is load-bearing. **Panel R**
(`packet_R.json`) asks whether a leaf evidences a sentence. Both carry the question and the seat
instructions verbatim, so a re-run cannot be steered by a changed prompt, and both carry only
opaque item ids: what an id is — which document, which window, which span, whether it is a
decoy — lives in the sealed key.

```
{
 "schema": "styxx-sworn/measurement-packet/v1",
 "panel": "L",
 "question": "<QUESTION_L, verbatim below>",
 "instructions": "<INSTRUCTIONS_L, verbatim below>",
 "output_schema": {... the JSON schema a seat's answer must satisfy ...},
 "built_from": {"population_sha256": "<sha256 of population.json>", "seed": 20260905,
                "decoys_sha256": "<sha256 of decoys_L.json>"},
 "key_digest_file": "keys/sworn_measurement_key_L.sha256",
 "items": [{"id": "L-0001", "text": "<window text>"}]
}
```

A Panel R item carries the sentence, the declared kind, and a **leaf view** of the receipt the
author bound it to, resolved at the sidecar's commit through `styxx.sworn.GitTree` and a
standard-library pointer walk. The verifier's verdict never enters the packet.

```
{"id": "R-0001", "sentence": "<the span's inner text, tags stripped>", "kind": "numeric",
 "leaf": {"receipt_name": "<basename only>", "pointer": "/a/b" | null, "lines": "L3-L5" | null,
          "value": "<leaf as JSON text | line slice | receipt text | sha256 hex>",
          "value_kind": "leaf" | "slice" | "receipt_text" | "sha256",
          "truncated": false}}
```

The leaf-view rules, by receipt form: a pointer fragment shows the leaf serialised as JSON text
with numbers as printed in the file; a line anchor shows the slice; a whole-receipt `numeric`,
`quote` or `absent` shows the receipt's text up to `LEAF_VIEW_MAX_CHARS` and marks `truncated`
when cut; a `hash` shows the receipt's sha256 as hex. A truncated view is disclosed in the
instructions as a ceiling the seat may answer `UNSURE` against.

**Panel L decoys** are two-sided and authored: `N_DECOYS_PER_SIDE` LOAD-BEARING and as many
NOT passages, each two or three sentences with one keyed sentence, drawn only from documents
the population rule excludes (the builder's format documents and the PLAN), committed in
`decoys_L.json` with a selection rule and the authorship disclosure the extraction
preregistration prescribes. **Panel R decoys** are built, not authored: the YES side is HELD
spans from the excluded documents with the leaf the author named; the NO side is the same
spans retargeted by the canary constructions below to a leaf that holds a different value or a
slice that lacks the needle. Their expected answers live in the sealed key only. Every decoy is
shuffled into the item order with `random.Random(SEED + 1)` for L and `SEED + 2` for R; a seat
cannot tell a decoy from a document item by id, position or shape.

Packet digests are written to `packets_digest.txt` (`sha256  filename` per line) and the
builder prints `NO seat was run and no number was computed by this file.`

## The sealed keys

A key is a JSON object `{item_id: meta}`, serialised with `sort_keys=True`, `indent=1`,
`ensure_ascii=False` and a trailing LF, written to `$STYXX_SEALED_DIR` (default
`C:\Users\heyzo\clawd\styxx-sealed`) under a name that begins `sworn_measurement_`. The salt is
`sworn_measurement_salt.txt` in the same directory, minted by `seal_key.py new-salt` from
`os.urandom`. The committed digest is `sha256(key_bytes + salt_utf8)`, written in-repo to
`keys/<key name>.sha256` as `<hex>  <key name>\n`. `seal_key.py check <name>` recomputes it and
exits 1 on a mismatch; `score.py` refuses to fold if any digest it reads does not match.

| key | meta per item |
|---|---|
| `sworn_measurement_key_L.json` | `{"kind": "document", "doc_id", "window": {"start", "end"}, "oversize"}` or `{"kind": "decoy", "decoy_id", "side": "LOAD-BEARING" \| "NOT", "keyed": {"start", "end"}}` — `keyed` is the byte range of the keyed sentence inside the item text |
| `sworn_measurement_key_R.json` | `{"kind": "document", "doc_id", "span_index"}` or `{"kind": "decoy", "decoy_id", "side": "YES" \| "NO", "source_stem", "construction"}` |
| `sworn_measurement_canary_key.json` | `{doc_id: {"stem", "commit", "twin_sha256", "canaries": [record]}}` — the canary record below |

The plaintext keys and the salt are released into `keys/` in a commit made after every seat
output has been recorded (`seal_key.py release`, which refuses without an explicit flag whose
name says what it asserts), never before.

## The canary rule

A **canary** is a well-formed sworn span whose receipt is known, by construction and by a
standard-library check, not to hold what the sentence says — *a leaf holding a different value;
a quote that is not there*. Canaries are planted in a **twin** of the document: a sidecar with
the same `text`, the same `document.sha256`, the same `commit` and the same manifest, whose
`spans` are the original spans with the canaries inserted or substituted. The committed
document, sidecar and receipt are never touched. The twin is a valid sidecar with no extra key
(`load_sidecar` refuses unknown keys); its relation to the original is recorded in the canary
key and in `twins/canary_digest.txt`, and its file name ends `.canary-twin.json` — never
`.sworn.json`, which `tests/test_sworn_dogfood.py` and `tests/test_sworn_eol.py` sweep.

Three constructions, all leaving `text` byte-identical:

- **A — retarget.** An existing span with a `path:<file>#/pointer` receipt is rewritten to a
  sibling scalar leaf of the same cited file at the same commit. For `numeric`, the sibling must
  be a JSON number whose `Decimal` value differs from the printed token by at least
  `10 ** (−fractional_digits)`, so the verifier's `ROUND_HALF_EVEN` at printed precision cannot
  reconcile them (the margin is the rounding rule's own quantum). For `quote`, the sibling must
  be a JSON string that does not contain the needle's bytes.
- **B — quote that is not there.** A narrative unit containing exactly one backtick needle of
  at least `styxx.sworn.SHORT_NEEDLE_BYTES` bytes becomes a `quote` span over a receipt the
  document already cites, chosen so that `receipt_bytes.find(needle) == -1`.
- **C — number that is not there.** A narrative unit containing exactly one number token
  becomes a `numeric` span over a cited leaf whose `Decimal` value differs by the A margin.

Falsehood is established with `subprocess` git plumbing at the sidecar's commit,
`json.loads(parse_float=Decimal)`, a pointer walk with `~0`/`~1` unescaping, and `bytes.find` —
never with `styxx.sworn.verify`. **Form** — not truth — is checked with the lexer and the loader:
the twin must pass `styxx.sworn.load_sidecar`, and `styxx.sworn.scan(styxx.sworn.render(twin))`
must reproduce its spans exactly; a candidate that would overlap an existing span, a fenced
region or an inline-code run, sit off a UTF-8 boundary, exceed the code-point cap, or carry a
pointer that trips the receipt grammar is rejected before it is planted. A canary that the
verifier later reports MALFORMED or UNRESOLVED **counts in `n` and not in `k`**: it is a planted
falsehood the verifier did not FAIL, whatever the reason, and dropping it would be the scorer
choosing its denominator.

The canary record:

```
{"twin_span_index": 7, "construction": "A" | "B" | "C", "original_span_index": 7 | null,
 "kind": "numeric" | "quote", "receipt": "path:<file>#/<pointer>",
 "printed": "0.55" | null, "needle_sha256": "<hex>" | null,
 "leaf_value": "0.2" | null, "margin": "1E-2" | null, "falsehood_check": "decimal_margin" | "bytes_find"}
```

The **rate**: `CANARY_RATE = all` plants every hostable site — every retargetable span, every
quote-hostable and number-hostable narrative unit — and prints the capacity per document as
counts (A, B, C, total). The design's rule reads *n ≥ 30* per document and the bar reads
*Wilson lower bound ≥ 0.95*. These two clauses cannot both hold per twin: with `k = n = 30` the
Wilson lower bound at `z = 1.96` lies below 0.95 (the formula is below; the test file pins the
inequality), and a document can host only as many canaries as its spans and its hostable
narrative units allow, which for some documents is fewer than thirty without editing their
text, and editing the text would break the offsets the packets share. The scorer therefore
evaluates G-C **pooled across the arm** — `k` and `n` summed over every twin — and prints the
per-twin counts beside it with no per-twin bar. Which `n` the bar applies to is the operator's
to sign; the machinery prints the smallest `n` at which `k = n` clears the bar so the signature
can be informed.

Twins go to `$STYXX_SEALED_DIR/sworn_measurement_twins/<doc_id>.canary-twin.json`; the
committed `twins/canary_digest.txt` lists `sha256  <doc_id>.canary-twin.json  canaries=<n>`
per twin and the salted digest of the canary key is `keys/sworn_measurement_canary_key.sha256`.
The positions are sealed; the counts are capacity and are not sealed.

## The trivially-swearing twin (G-G1)

DESIGN v0.1 item 4, inherited by v2's G-G1: for each document, a second version by the same
family under the instruction *swear only what is trivially true*. `twin_trivial.py` prompts the
family with the canonical text, the tag grammar of `SPEC_sworn_output_v02_2026_09_02.md`, the
list of receipt names the document already cites, and that instruction; the returned inline
document is canonicalised with `styxx.sworn.to_sidecar`, which refuses a document that cannot
round-trip. The twin's canonical text must equal the original's byte for byte — the twin
re-tags, it does not rewrite — or the pair is recorded `twin_text_changed` and excluded with
its count printed. The floor of each side is `styxx.sworn.verify(...)["coverage"]["sentence_share"]`
at the document's commit.

Stated plainly, because the README and the scorer's output must say it: under v0.2,
`sentence_share = sworn_total / (sworn_total + narrative_sentences)` does not know which
sentences were sworn. A pair in which the trivial twin prints the lower floor is a pair in which
the twin swore fewer sentences, or masked fewer of the narrative. G-G1 as designed is a
comparison of that floor, implemented exactly as written and titled as what it is.

## The questions, verbatim

`QUESTION_L`:

> For each sentence in the passage below, decide whether it is LOAD-BEARING — a claim the
> document's conclusion depends on, which could turn out to be right or wrong — or NOT
> load-bearing — context, method, a date, a version, a file name, a restatement, a hedge, a
> sentence about what the document does not say — or UNSURE. Draw the sentence boundaries
> yourself.

`INSTRUCTIONS_L` (the three blocks; the local family's seats see them in the three rotations
of block order, seat 1 as written, seat 2 starting from FORMAT, seat 3 starting from CAUTIONS;
the Claude family's seats see them as written in three fresh sessions):

> TASK. You are reading one passage from a technical report. Label every sentence.
>
> FORMAT. Answer with one JSON object and nothing else:
> `{"brackets": [{"opening_words": "...", "closing_words": "...", "label": "LOAD-BEARING"}, ...]}`.
> `opening_words` is the opening three words of the sentence and `closing_words` is its closing three
> words, copied exactly as they appear in the passage including punctuation and capitalisation;
> a sentence shorter than six words may repeat words in both. `label` is exactly one of
> `LOAD-BEARING`, `NOT`, `UNSURE`. Cover every sentence once; do not skip, merge or reorder.
>
> CAUTIONS. A sentence that reports a number, a quoted string, a hash, or the absence of
> something can still be NOT load-bearing if the conclusion would stand without it. A sentence
> with no number can be LOAD-BEARING. Use UNSURE honestly; it is counted against the panel, not
> against you. Do not reproduce the passage; quote only the opening and closing three words.

`QUESTION_R`:

> Here is one sentence from a technical report, the kind of check its author declared for it,
> and the receipt leaf the author bound it to. Does the leaf evidence the sentence — is what the
> leaf holds what the sentence says? Answer YES, NO or UNSURE.

`INSTRUCTIONS_R` (same rotation rule):

> TASK. Compare one sentence with one leaf. `numeric`: the sentence's one number should be the
> leaf's value at the precision the sentence prints. `quote`: the text between backticks in the
> sentence should occur in the leaf. `absent`: it should not. `hash`: the sentence's 64-hex
> digest should equal the leaf's digest.
>
> FORMAT. Answer with one JSON object and nothing else: `{"answer": "YES"}`, `"NO"` or
> `"UNSURE"`.
>
> CAUTIONS. Judge the pairing, not the sentence's truth in the world. If the leaf view is marked
> truncated and the answer would depend on the cut part, answer UNSURE. Do not explain.

The per-seat variation for a greedy local model is the block rotation and nothing else; three
seats of one deterministic model under three prompt orders are not three judgements, and the
RESULT that reports them says so. The Claude family's three seats are three fresh sessions under
the transport's default sampling; that is disclosed in the same sentence.

## Seat transcripts and the ledger

`seat_outputs/<family>/<panel>-seat<s>.json`:

```
{
 "schema": "styxx-sworn/measurement-seat/v1",
 "family": "claude" | "local", "panel": "L" | "R", "seat": 1,
 "substrate": {"model": "...", "transport": "claude-cli clean-config" | "transformers",
               "dtype": "bf16" | null, "device": "cpu" | "cuda" | null, "quant": null | "nf4",
               "named_in_design": true | false, "block_order": ["TASK", "FORMAT", "CAUTIONS"]},
 "packet_sha256": "<sha256 of the packet file read>",
 "prereg": "<path at HEAD>" | null, "dry_run": false,
 "contamination_probe": {"asked": "...", "answer": "...", "ok": true} | null,
 "items": [{"id": "L-0001", "raw_sha256": "<sha256 of the transport's raw bytes>",
            "parsed": true, "brackets": [...]} | {"id": "R-0001", "raw_sha256": "...",
            "parsed": true, "answer": "YES"}],
 "unparsed": ["L-0007"], "errors": [{"id": "...", "error": "..."}],
 "verdict": "RECORDED" | "VOID-CONTAM" | "DRY-RUN"
}
```

Every transport call appends one line to `seat_outputs/<family>/ledger.jsonl`:
`{"item_id", "panel", "seat", "raw_sha256", "ts", "error"}` — the raw bytes are hashed before
any decoding, so the hash is of what the transport wrote and not of what Python read. The Claude
family runs the contamination probe of `run_b23_fable.py` before any item (the question
names this lab's operator and agent; any answer other than `NO` writes the seat file with
`VOID-CONTAM` and no items). An answer the parser cannot read is recorded `parsed: false`,
listed in `unparsed`, and projects as NO-LABEL on every unit of its item.

## Projection, majority and the cross-family label

A Panel L bracket is **located** by exact byte search of `opening_words` in the item text: a
unique occurrence anchors the start; `closing_words` is then searched from that anchor and its
earliest occurrence ends the bracket. A `opening_words` that is absent or occurs more than once, or a
`closing_words` absent after the anchor, leaves the bracket `unlocated` and counted. No
normalisation is applied on the initial pass; a second pass with runs of whitespace collapsed is
applied only to brackets the initial pass left unlocated, and the count of brackets located by the
second pass is printed.

Each unit takes, per seat, the label of the located bracket with the **largest byte overlap**
with it; zero overlap is `NO-LABEL`; a tie between brackets carrying different labels is
`NO-LABEL`. The **family label** of a unit is the strict majority of its three seats' projected
labels over `{LOAD-BEARING, NOT, UNSURE}` — at least two seats agreeing — else `NO-MAJORITY`;
`NO-LABEL` is not a vote. The **final label** is the two families' labels when they agree and
`FAMILY-SPLIT` when they do not. `UNSURE`, `NO-MAJORITY` and `FAMILY-SPLIT` are excluded from
every numerator and denominator and each is printed as its own count. Panel R answers project
without location: the family answer is the strict majority of `{YES, NO, UNSURE}` and the final
answer is the agreement rule above.

The design's open alternative — a pooled six-seat majority — changes every denominator; the
scorer implements the per-family rule and the PREREG names which rule is signed.

**Bindability** is a byte predicate over three kinds, applied to a narrative unit's text:
`numeric` iff `styxx.sworn._number_token` finds exactly one number token; `quote` iff
`styxx.sworn._needle_in` finds exactly one backtick needle of at least `SHORT_NEEDLE_BYTES`;
`hash` iff exactly one 64-hex run. `absent` has no byte predicate and is reported as
unbindable-by-predicate, disclosed. A sworn unit is bindable by construction. Q1's cells are
separated by this predicate, not by a question to the seats: cell 1 is a final LOAD-BEARING
unit that is bindable, cell 2 a final LOAD-BEARING unit that is not, cell 3 a final NOT unit.

## Statistics

`wilson(k, n, z=WILSON_Z)` is `styxx.mind.wilson`: with `p = k/n`,
`centre = (p + z²/2n) / (1 + z²/n)`, `half = z·√(p(1−p)/n + z²/4n²) / (1 + z²/n)`, the interval
is `[max(0, centre − half), min(1, centre + half)]`; `n = 0` gives `NaN` on both sides. The
scorer prints `k of n` and both bounds; a bar applies to the lower bound.

`cohen_kappa(a, b)` over paired labels with `UNSURE`, `NO-LABEL`, `NO-MAJORITY` and
`FAMILY-SPLIT` excluded from both lists pairwise: `po` is the agreement share, `pe` is
`Σ_label p_a(label)·p_b(label)`, `kappa = (po − pe) / (1 − pe)`, `NaN` when `pe = 1` or the
paired count is zero; the excluded count is printed. Two kappas are reported: over the
splitter's unit set with projected labels (the design's rule), and a panel-boundary variant
over the Claude family's seat-1 located brackets with the local family's seat-1 bracket
projected by largest overlap — a second number only.

`majority(votes)` is `score_extraction_panel.majority`: the modal vote when it is strict, else
none.

## The gates, each as a function of committed inputs

Inputs: `population.json`; `packet_L.json`, `packet_R.json`, `decoys_L.json`; the sealed keys,
whose salted digests must equal `keys/*.sha256`; `seat_outputs/*/*.json`; the twins, whose
digests must equal `twins/canary_digest.txt`; the committed sidecars and receipts at the commits
they name; `twin_trivial` sidecars where they exist. Bars are copied verbatim from DESIGN v2 and
are **proposed, unsigned**.

| gate | function of the inputs | bar (proposed, unsigned) | a miss is titled |
|---|---|---|---|
| G-D | per family, per panel: the family label of each decoy's keyed unit (L) or the family answer (R) against the sealed side; `correct_overall / 30` and `correct_side / 15` for each side | ≥ 27/30 overall and ≥ 9/15 per side, per family | void panel; the void is the result |
| G-F | the number of families clearing G-D on a panel | two, each clearing its decoys | one-family: counts only, no precision |
| G-S1 | over non-fragment units with a final label: `sworn ∧ LOAD-BEARING ∧ bindable` over `LOAD-BEARING ∧ bindable` | ≥ 0.70 | *authors leave bindable sentences unbound* |
| G-S1X | the same quantity on the external arm | ≥ 0.50 | as above, with the receipt-availability ceiling beside it |
| G-U | count of final LOAD-BEARING units that are not bindable | reported, no bar | the format's ceiling, priced for v0.3 |
| G-S2 | sworn units with final NOT over sworn units with a final label in `{LOAD-BEARING, NOT}` | ≤ 0.25 | *the numerator is padding* |
| G-P | Panel R items of kind document with final NO over those with final in `{YES, NO}` | ≤ 0.10 | *the author named the wrong leaf* |
| G-C | over every canary twin verified live by `styxx.sworn.verify` at its commit: `k` = canaries whose verdict is FAILED, `n` = canaries planted; `wilson(k, n)` | lower bound ≥ 0.95 | *the verifier misses planted falsehoods* |
| G-R | count of Panel R document items whose committed receipt verdict is HELD and whose final answer is NO | reported as its own cell | the forgotten term, printed |
| G-G1 | pairs with a trivial twin whose text matches: pairs where `twin.sentence_share < original.sentence_share` over pairs | ≥ 0.80 of pairs | *the floor cannot price gaming*; the floor leaves the headline |

**Q3**, per document, no gate: `sworn_total / (sworn_total + final-LOAD-BEARING narrative
units)` minus the committed receipt's `coverage.sentence_share`, expected positive by
construction and printed as how much the floor understates.

**The WITHHELD rule.** When G-D voids a family on a panel, that family's labels on that panel
enter no final label. When fewer than two families clear a panel, every share on that panel is
the literal string `WITHHELD` and the counts stay; with exactly one family, the counts are
labelled `one-family` and no share is printed. `G-C` and `G-G1` do not depend on a panel and
are printed regardless. A gate that fails prints its title from the last column beside its
numbers; nothing is re-run.

`measurement_result.json`:

```
{
 "schema": "styxx-sworn/measurement-result/v1",
 "prereg": "<path at HEAD>" | null, "dry_run": false, "quotable": true | false,
 "lock": {"schema": "styxx-sworn/measurement-inputs/v1", "head": "<40-hex>",
          "inputs": [{"path", "raw_sha256", "content_sha256", "blob", "committed"}]},
 "population": {"documents", "sworn_spans", "units", "fragments"},
 "families": {"claude": {...substrate...}, "local": {...substrate...}},
 "gates": {"G_D": {...}, "G_F": {...}, "G_S1": {...}, "G_U": {...}, "G_S2": {...},
           "G_P": {...}, "G_C": {"k", "n", "wilson95": [lo, hi], "per_twin": {...},
                                  "smallest_n_clearing_bar_at_k_eq_n", "bar", "pass"},
           "G_R": {...}, "G_G1": {"pairs", "lower", "share", "note", "bar", "pass"}},
 "cells": {"final_labels": {...}, "family_split", "unsure", "no_majority", "no_label",
           "unlocated_brackets", "located_by_second_pass", "unparsed_items"},
 "kappa": {"splitter": {...}, "panel_boundary": {...}},
 "q3": {doc_id: {"panel_coverage", "sentence_share", "difference"}},
 "disclosure": ["<the sentences the README lists, verbatim>"],
 "withheld": ["G_S1", ...]
}
```

Every gate object carries `"proposed_unsigned": true` until a PREREG names it; the scorer reads
the bar values from its own module constants and, when a PREREG exists at HEAD, refuses to fold
if the PREREG's lock hash does not cover `score.py`'s committed blob.

## The dry run

`dry_run.py` runs the whole ladder — population, packets, decoys, keys, canaries, three canned
seats per family per panel, a trivial twin, the scorer — over a **synthetic** population on a
`styxx.sworn.MemoryTree` at a nominal commit, in one process, writing under `dryrun/` and to a
temporary sealed directory. It exists so every code path — projection, decoy gating including a
family built to fail, canary verification including one canary planted to be MALFORMED, kappa,
the WITHHELD path — is exercised before any real seat runs, and so the machinery counts can be
sworn without any real document being read by a model.

What it **may** print and write: counts of items built, windows, decoys, canaries planted per
construction, canaries the verifier FAILED / MALFORMED in the synthetic twins, seats exercised,
brackets located and unlocated, units labelled; the sha256 of every file it wrote; the
smallest-`n` table for the canary bar; the words `DRY RUN — no quotable number`.

What it **may not** print or write: any share, interval, kappa or Q3 value (each is replaced by
the literal `DRYRUN-NO-RATE`); any real document path, stem, or receipt name; any file whose
name ends `.sworn.json`, `.sworn-receipt.json` or `PREREG_*.md`; any number described as a
measurement of sworn output. Its result file is `dryrun/dry_run_result.json` with
`"dry_run": true, "quotable": false` as the two keys after the schema.

Refusals, each a `SystemExit` beginning `REFUSED:`: a population entry whose stem resolves to
a file in the repository, or whose `doc_id` does not begin `SYN-`; a sealed directory equal to
`$STYXX_SEALED_DIR`; an output path outside `dryrun/`.

## The seat runners

`seat_claude.py` copies `run_b23_fable.py`'s `cli()` — `claude --model <alias>
--setting-sources "" --tools "" --system-prompt <s> -p <prompt> --output-format json`, plus
`--json-schema <schema>` — with `CLAUDE_TIMEOUT_S`, raw stdout captured as bytes and hashed
before decoding, `is_error` retried twice then recorded as an error. The model alias is a module
constant the PREREG names; the transport check in the dry run records whether the alias answers
on this box and is not a seat run.

`seat_local.py` loads `Qwen/Qwen2.5-7B-Instruct` with `torch_dtype=torch.bfloat16` and
`device_map="cpu"` under `torch.set_num_threads`, applies the chat template with the rotated
instruction blocks as the system turn and the item as the user turn, and generates greedily
with `max_new_tokens=LOCAL_MAX_NEW_TOKENS`; the answer is the earliest balanced JSON object in the
generated text, else `parsed: false`. `--model Qwen/Qwen2.5-3B-Instruct` is the design's named
fallback; `--device cuda` and `--quant nf4` are substrates the design does not name, and a seat
file written under either carries `"named_in_design": false`. `--throughput-probe` generates a
fixed number of tokens on a synthetic prompt and prints tokens per second and peak resident
memory; it writes no seat file.

Both runners: the PREREG refusal above; `--dry-run` substitutes `synthetic.canned_answer` and
makes no transport call; one item per call, no session reuse; the seat file is written once,
complete, with `newline="\n"`.

## Tests this spec commits to

`tests/test_sworn_measurement_machinery.py`: the unit set of every population document
reconciles with its committed receipt; `bindable` on fixtures of each kind and a fragment;
`locate` and `project_labels` on hand-built windows including the tie and the duplicate
`opening_words`; `majority` and the family/final rules; `cohen_kappa` on a hand-computed table;
`wilson` equal to the `run_b23_fable.py` formula on a grid without importing that module, and
the inequality at `k = n = 30` against the bar with the smallest clearing `n` above it; packets
are deterministic — two builds from the same inputs are byte-identical; the sealed key digest
file exists before any seat file and a seat runner refuses without it; a canary twin on a
synthetic sidecar leaves `text` and `document.sha256` unchanged, keeps spans disjoint and
ordered, reproduces under `scan(render(...))`, and — in the test only, on synthetic bytes —
verifies FAILED at exactly the canary indices and HELD everywhere else; a MALFORMED canary
counts in `n`; the scorer reproduces every gate from a fixture, prints `WITHHELD` when a family
fails G-D, and prints counts only with one family; the dry run refuses a real document path and
a real sealed directory; no file under `papers/sworn/measurement/` ends `.sworn.json`; every
JSON the machinery writes is LF-only.

## What ships with v0.1

The directory above with every script, `population.json` at a pinned commit, `packet_L.json`,
`packet_R.json`, `decoys_L.json`, `packets_digest.txt`, `keys/*.sha256`,
`twins/canary_digest.txt`, `dryrun/dry_run_result.json`, the test file, a README whose opening
line is the PREREG sentence, a CHANGELOG entry, and a short sworn RESULT bound only to the dry
run's machinery counts. No seat output over a real document. No `measurement_result.json`
that is not the dry run's.

## What this spec does not say

That any bar is signed: every bar is the design's proposal. That the machinery measures
anything: it will not read a real document into a model before the PREREG commit, and its dry
run is over synthetic bytes. That two families on one machine are independent judges: the design
says correlated error is the ceiling and this spec inherits the sentence. That the decoys are
not the builder's: the L side is authored and the R side is constructed, both disclosed. That
`sentence_share` prices gaming: G-G1 is a floor comparison and is titled as one. That the
canary gate measures the verifier's recall on falsehoods in the wild: it measures recall on
falsehoods of three named constructions, planted by the builder of the verifier. That the
Claude transport works on this box today: the dry run records what it answered. That the local
substrate is decided: the throughput probe informs a signature that has not happened.

## Owed after v0.1, recorded as owed

1. The PREREG commit — the operator's, never the builder's — naming the signed bars, the pooled
   `n` for G-C, the local substrate, the cross-family rule, the Claude alias, and the lock hash.
2. The external-arm packet builder over AIDev HELD-OUT pull requests, titled by its receipt pool.
3. The CI-bound pilot's minting job, which waits on the sworn action being merged into a workflow.
4. A prospective canary rate for the twelve documents not yet written, enforced by a harness that
   runs before the author sees a verdict.
5. The release of the keys after the seats, and the RESULT that reports the panel under the
   design's titles.

---

*The measurement's whole discipline is that nothing in it is chosen after a seat has spoken.
This spec chooses everything that can be chosen now — formats, questions, decoys, canaries,
gates — and leaves the one thing it may not choose, the bars, to a signature.*
