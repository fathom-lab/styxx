# STRANGER — how someone who does not trust this lab checks the sand, in seven commands

Fathom Lab · 2026-09-14 · Everything the lab says about the checksum, the plate, the ferry log,
the seals and the bounty is supposed to be re-derivable from bytes in this repository by a person
who has never spoken to the lab. This page is the list of commands that person runs, what each
one proves, and what it does not. Where a value must come from outside the repository (a chain
head, a beacon), the page says where the lab publishes it and why it cannot live only in the tree.

## 0. the checkout

    git clone https://github.com/fathom-lab/styxx && cd styxx
    git log -1 --format=%H          # the commit you are checking; every command below is about this commit

A commit id is the only name a claim has here. A claim that names no commit is not a claim.

## 1. the tests

    pip install -e ".[test]"
    python -m ruff check styxx && python -m pytest tests -q

What it proves: the verifiers (`styxx.sworn`, `styxx.charon`, `styxx.checksum`, `styxx.beacon`,
`styxx.clock`, `styxx.portability`) behave as their tests say on your machine, including the tests
that pin the canary pool to the hash the beacon-draw PREREG froze (`tests/test_beacon_draw.py`)
and the tests that bind every band in `papers/checksum/score.py` to the frozen PREREG text
(`tests/test_checksum_score.py`). What it does not prove: anything about a model. Linux catches
what Windows hides (`.github/workflows/test.yml` runs 3.9–3.12); if the suite is red on your box
the lab wants the log.

## 2. the ferry log

    python -m styxx.charon status --log papers/charon/charon.log.jsonl
    python -m styxx.charon verify --log papers/charon/charon.log.jsonl --repo . --expect-head <head>

What it proves: every line of the log — every sworn document, capsule and certificate the lab has
published — re-derives from the bytes at the commit the line names, and the chain
is intact up to the head you were given. The head must come from outside the log (a log can be
truncated and re-chained); the lab prints it in each operator REPORT (`papers/chat/`), in the
pull-request comments, and in any post that cites the log. Today's head is
`cb549dd61b6de2842b5d3722e8a74a2e08a925dccd14195713da4e3c644cf71b` (251 lines). What it does not
prove: that a HELD line is true about the world — only that its numbers are the numbers in its
receipt at its commit.

## 3. one sworn document, by hand

    python -m styxx.sworn verify papers/checksum/PREREG_checksum_beacon_draw_2026_09_14.md --repo . --commit b8205b1c

What it proves: every number in that document is the number in the file it cites, at that commit
(SWORN-HELD, 24 spans); the same for any `*.md` beside a `*.sworn-receipt.json`. Without
`--commit` the verifier reads the working tree and every span is UNRESOLVED; that is the documented
operator-gated behaviour, not a pass. What it does not prove: that the cited file is honest — for
that, re-run the recipe that wrote it (below).

## 4. the seals, when they exist

    python -m styxx.clock verify papers/charon/anchors.jsonl

What it proves: each recorded transaction is a confirmed memo from the creator wallet carrying the
digest it claims, is the earliest such memo, and the digest is the sha256 of the git blob named
(`git show <commit>:<path> | sha256sum`). The line for a `sealed-prereg` prints the beacon: the
block hash of that slot. As of this page no anchor exists (`SEALS_2026_09_13.md` lists the digests
that will be sealed and the memo text for each); until then this command has nothing to verify
and says so. What it does not prove: anything about when a run happened — except for the
beacon-draw run, whose canaries did not exist until that block did.

## 5. the draw

    python -m styxx.beacon <beacon> 48

What it proves: the 48 canary ids and the canary hash that the beacon-draw certs and every
fingerprint under it carry are the ones this pool and this beacon produce, by sha256 arithmetic
anyone can redo in a shell (`styxx/beacon.py`, `select`, a few lines). The pool is 778 items, hash
`9e450999977a274fe63f1f7358378a7b42b710c54572dbc0e72bd2b45ab9906f`, frozen in the PREREG and
pinned by a test. What it does not prove: that the run used those items — the certs' draw record
and `checksum.check_draw_record` do that, and the scorer runs it.

## 6. the reading

    python papers/checksum/score.py --prereg beacon_draw papers/checksum/beacon_draw_certs.json \
        --expect-beacon <beacon from step 4> --expect-blob d6a98261f44f31664cdedbc24769532d5a701bf46d2c8b9ce30cf6a227fb5519

What it proves: the hypotheses and kill gates the PREREG froze, read against the certs by code,
clause by clause — predicted band, observed value, holds — with K5 refusing a run whose beacon is
not the seal's or whose draw does not re-derive, and K1 evaluating nothing else if it fired. The
RESULT the lab writes swears to this scorecard; your scorecard must match it. Until the sealed run
exists, run it on the instrument check the lab committed:

    python papers/checksum/score.py --prereg beacon_draw papers/checksum/beacon_draw_certs_dryrun_qwen0.5b.json

and read `INSTRUMENT CHECK; gates fired: K5` — the scorer refuses to call a 0.5B check the
experiment. What it does not prove: that the bands were sensible; that is what the PREREG's
"what the instrument check showed" section is for, and it is sworn.

## 7. the recipe itself

    python papers/checksum/run_smollm_quant.py            # CPU, minutes: the sworn RESULT's recipe
    python -m styxx.portability papers/checksum/smollm_quant_certs.json <your certs> ...

What it proves: on your machine, the verdicts of `RESULT_checksum_smollm_quant_2026_09_13.md`
reproduce (SAME / DRIFT / DRIFT); the magnitudes do not, by a measured amount —
`RESULT_portability_smollm_quant_2026_09_13.md` puts the lab's own two-machine width at 0.10
nats/token on the int8 arm and 0.22 on the random arm, and `BOUNTY.md` says what a re-run outside
that width is worth. What it does not prove: anything about the deploy-scale run, which needs a GPU
and the seal.

## what the lab claims, and where each claim's receipt is

The only positioning sentence the lab may say is the one `SURVEY_sand_neighbours_pass3_2026_09_14.md`
(sworn) licenses: "we know of no lab that" does four things at once, each with its neighbours
named inside the sentence. Thirty-seven sources across three frozen-protocol passes; two clauses
retired (the plate is Perrig & Song 1999 applied to receipts; a seal on a public chain is
timestamping, 1991); none of the four survivors is unoccupied. The words "first", "novel" and
"revolutionary" are not licensed by anything in this repository.

## if something does not re-derive

Open an issue with the command, the commit, and the output. `BOUNTY.md` says which disagreements
are paid and how; the lab's own second-machine re-run is why the tolerance rule exists.
