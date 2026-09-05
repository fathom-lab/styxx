# sworn measurement machinery v0.1

No measurement runs before a `papers/sworn/PREREG_sworn_measurement_<date>.md` is committed with
the lock hash of `score.py`, `packet_L.json`, `packet_R.json`, `twins/canary_digest.txt` and
`keys/*.sha256`. Every seat runner in this directory refuses to run without `--dry-run` until that
file is tracked at HEAD, and `--dry-run` is accepted only over a packet built from a synthetic
population.

Built to `papers/sworn/SPEC_sworn_measurement_machinery_2026_09_05.md`, frozen in its own commit
before any of this code; the design whose bars await the operator's signature is
`papers/sworn/DESIGN_sworn_measurement_v2_2026_09_02.md`. Neither is edited by this directory.

## The ladder

```
population.py     the population rule at a pinned commit         -> population.json
build_packets.py  canonical text -> Panel L windows, Panel R leaf views, decoys, sealed keys
seal_key.py       new-salt | seal | check | release             -> keys/<name>.sha256
canaries.py       canary twins into the sealed directory         -> twins/canary_digest.txt
                  [ the PREREG commit — the operator's, not the builder's ]
seat_claude.py    Claude family, claude -p clean-config, one call per item, ledger of raw bytes
seat_local.py     local family, Qwen2.5-7B-Instruct bf16 on CPU; 3B is the design's fallback; cuda and nf4 are not named
twin_trivial.py   the trivially-swearing twin for G-G1
score.py          every gate as a function of committed inputs  -> measurement_result.json
synthetic.py      synthetic documents, receipts, decoys and canned seat answers
dry_run.py        the whole ladder over synthetic items, in one process, under dryrun/
```

The dry run: `python papers/sworn/measurement/dry_run.py`. It writes `dryrun/dry_run_result.json`
with `"dry_run": true, "quotable": false`, prints counts and digests, and replaces every share,
interval, kappa and Q3 value with the literal `DRYRUN-NO-RATE`. It refuses a population entry
that resolves to a file in the repository, a doc_id that does not begin `SYN-`, a sealed directory
equal to `$STYXX_SEALED_DIR`, and an output path outside `dryrun/`.

Tests: `python -m pytest tests/test_sworn_measurement_machinery.py -q -p no:cacheprovider`.

## What is sealed and what is committed

Keys (`{item_id: meta}`) and canary twins live under `$STYXX_SEALED_DIR` (default
`C:\Users\heyzo\clawd\styxx-sealed`) and never in the tree. The tree carries only salted
digests: `keys/<name>.sha256` is `sha256(key_bytes + salt)`, and `twins/canary_digest.txt` lists
each twin's sha256 and its canary count. `seal_key.py release` moves the plaintext into the tree
in a commit made after every seat output is recorded, and refuses without the flag whose name
says what it asserts.

## The disclosures a RESULT of the panel must carry

Copied into `measurement_result.json` under `disclosure`, verbatim:

1. The authors of the in-house documents knew the bars before writing them; the in-house arm is
   bar-aware by construction.
2. Both families ran on one machine, and one of them is the family that wrote the documents;
   correlated error is the ceiling, and kappa between families is a description, never validity.
3. The Panel L decoys were authored by the builder from documents the population rule excludes;
   the Panel R decoys were constructed by the same code that plants the canaries.
4. Under v0.2, `sentence_share = sworn_total / (sworn_total + narrative_sentences)` does not know
   which sentences were sworn; a G-G1 pass says the trivial twin swore fewer sentences, and
   nothing about whether the floor prices gaming.
5. The canaries for the nine existing documents were planted after authorship; a prospective
   rate for documents not yet written is owed and enforced by a harness that runs before the
   author sees a verdict.
6. Three seats of one deterministic local model under three instruction-block orders are not
   three judgements; the Claude family's three seats are three fresh sessions under the
   transport's default sampling.

## Two arithmetic facts the signature has to know

With `z = 1.96`, `wilson(30, 30)` has a lower bound below the design's 0.95 bar; the smallest `n`
at which `k = n` clears it is printed by `canaries.py` and pinned by the tests. Five of the nine
in-house documents cannot host thirty canaries without editing their text, and editing the text
breaks the offsets the packets share; the scorer therefore pools `k` and `n` across the arm and
prints per-twin counts beside the pooled interval with no per-twin bar. Which `n` the bar applies
to is the operator's to sign.

## What this directory does not say

That any bar is signed. That anything here measures sworn output: no seat reads a real document
before the PREREG commit, and the dry run is over synthetic bytes. That two families on one
machine are independent judges. That the canary gate measures recall on falsehoods in the wild:
it measures recall on three named constructions planted by the builder of the verifier.
