# STRANGER — how someone who does not trust this lab checks the sand, in seven commands

Fathom Lab · 2026-09-14 · Everything the lab says about the checksum, the plate, the ferry log,
the seals and the bounty is supposed to be re-derivable from bytes in this repository by a person
who has never spoken to the lab. This page is the list of commands that person runs, what each
one proves, and what it does not. Where a value must come from outside the repository (a chain
head, a beacon), the page says where the lab publishes it and why it cannot live only in the tree.
Revised the same evening after the lab's own red team: two sentences on this page claimed more
than the commands prove, and the page now says what they prove.

## 0. the checkout

    git clone https://github.com/fathom-lab/styxx && cd styxx
    git log -1 --format=%H          # the commit you are checking; every command below is about this commit
    git status --untracked-files=no # empty, or you are checking bytes the commit does not carry

A commit id is the only name a claim has here. A claim that names no commit is not a claim.

## 1. the tests

    pip install -e ".[test]"
    python -m ruff check styxx && python -m pytest tests -q

What it proves: the verifiers (`styxx.sworn`, `styxx.charon`, `styxx.checksum`, `styxx.beacon`,
`styxx.clock`, `styxx.portability`, `styxx.stranger`) behave as their tests say on your machine,
including the tests that pin the canary pool to the hash the beacon-draw PREREG froze
(`tests/test_beacon_draw.py`) and the tests that parse every band out of the frozen PREREG text and
pin every comparator on its edge (`tests/test_checksum_score.py`). What it does not prove: anything
about a model. Linux catches what Windows hides (`.github/workflows/test.yml` runs 3.9–3.12); if the
suite is red on your box the lab wants the log.

## 2. the ferry log

    python -m styxx.charon status --log papers/charon/charon.log.jsonl
    python -m styxx.charon verify --log papers/charon/charon.log.jsonl --repo . --expect-head <full 64-hex head>

What it proves: every line of the log — every sworn document, capsule and certificate the lab has
published — re-derives from its sidecar at the commit the line names, and the chain is intact up to
the head you were given. The head must come from outside the log (a log can be truncated and
re-chained), and it must be the full 64 hex: `charon status` prints only a prefix. The lab prints the
full head in each operator REPORT (`papers/chat/`), in the pull-request comments, and in any post that
cites the log. What it does not prove: that a HELD line is true about the world — only that its
numbers are the numbers in its receipt at its commit; and it does not read the `.md` in your working
tree (step 3 does).

## 3. one sworn document, by hand

    python -m styxx.sworn check  papers/checksum/PREREG_checksum_beacon_draw_2026_09_14.sworn-receipt.json \
                                 papers/checksum/PREREG_checksum_beacon_draw_2026_09_14.sworn.json --repo .
    python -m styxx.sworn verify papers/checksum/PREREG_checksum_beacon_draw_2026_09_14.md --repo . --commit b8205b1c

What it proves: the first re-derives the committed receipt exactly (VERIFIED: the digest matches and
the verdict reproduces; `same-build=False` on an older receipt only says the verifier's bytes moved
since; `document=` names the verdict the receipt re-derives, which can be SWORN-FAILED — a receipt
that re-derives is not a document that held). The second re-reads the document from scratch — every
number in it is the number in the file it cites, at that commit (SWORN-HELD, 24 spans). Hand `check`
the `.sworn.json` sidecar, not the `.md`: the sidecar carries the commit and the manifest binding,
and a receipt whose spans cite a harness manifest reads FAILED without it. But `check` on a sidecar
renders the document from the sidecar and never opens the `.md` beside it, so a `.md` edited after
its receipt still reads VERIFIED; the one-command form below compares them byte for byte, and
`verify --commit` above re-reads the committed bytes rather than your working copy. Without `--commit`
the verifier reads the working tree and every span is UNRESOLVED; that is the documented
operator-gated behaviour, not a pass. What it does not prove: that the cited file is honest — for
that, re-run the recipe that wrote it (below).

## 4. the seals, when they exist

    python -m styxx.clock verify papers/charon/anchors.jsonl

What it proves: each recorded transaction is a confirmed transaction the creator wallet signed, with a
memo instruction exactly `styxx <kind> <digest>`, and — for a seal — it is the earliest such
transaction: a memo carrying the digest that another key sent the wallet does not count and is listed
as foreign (until 2026-09-14 it counted, and anyone could have made a real seal read
EARLIER_MEMO_EXISTS). The digest is the sha256 of the git blob named (`git show <commit>:<path> |
sha256sum`). A `sealed-prereg` line that reads ANCHORED prints the beacon, the block hash of that
slot; any other status prints no beacon. As of this page no anchor exists (`SEALS_2026_09_13.md` lists
the digests that will be sealed and the memo text for each); until then this command has nothing to
verify and says so. What it does not prove: that anything happened after the seal. For the
beacon-draw run it proves that WHICH 48 items were graded was fixed by a value no one could choose
before that block; it does not prove that the per-item values were computed after it — the pool is
public, and a fingerprint over all its items can be computed by anyone beforehand and subset to any
draw (CORRECTION_prereg_beacon_draw_2026_09_14.md).

## 5. the draw

    python -m styxx.beacon <beacon> 48

What it proves: the 48 canary ids and the canary hash that the beacon-draw certs and every
fingerprint under it carry are the ones this pool and this beacon produce, by sha256 arithmetic
anyone can redo in a shell (`styxx/beacon.py`, `select`, a few lines). The pool is 778 items, hash
`9e450999977a274fe63f1f7358378a7b42b710c54572dbc0e72bd2b45ab9906f`, frozen in the PREREG and
pinned by a test. What it does not prove: that the run used those items — the certs' draw record,
every arm's cert and `checksum.check_draw_record` do that, and the scorer and the one-command form
run them.

## 6. the reading

    python papers/checksum/score.py --prereg beacon_draw papers/checksum/beacon_draw_certs.json \
        --expect-beacon <the beacon step 4 prints on the ANCHORED line>

What it proves: the hypotheses and kill gates the PREREG froze, read against the certs by code,
clause by clause — predicted band, observed value, holds. The scorer re-derives what it could trust:
K1 from the recorded floor, not the runner's flag; the sealed PREREG digest, which it freezes itself
and compares whether or not you pass `--expect-blob`; the draw, with n = 48; and that every arm's cert
grades the same drawn set. K5 refuses a run whose beacon is not the seal's. A valid sealed run whose
K1 fired is a result, INCONCLUSIVE; anything else invalid is an INSTRUMENT CHECK. The RESULT the lab
writes swears to this scorecard (`styxx.checksum/scorecard/v2`); your scorecard must match it. Until
the sealed run exists, run it on the instrument check the lab committed:

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

## all of it in one command

    python -m styxx.stranger --repo . --expect-head <full 64-hex head>   # steps 0, 2, 3, 5, 6; a table and an exit code
    python -m styxx.stranger --repo . --with-tests --network              # steps 1 and 4 as well (minutes; the chain)
    python -m styxx.stranger --repo . --only ferry_log,draw,reading --json stranger_report.json

`styxx.stranger` runs the steps above and prints PASS / FAIL / SKIP per step with the detail a
dispute needs, then writes a report (`styxx.stranger/report/v2`) naming the commit it ran on. It adds
no verdict of its own: the ferry log is `styxx.charon.verify_log`, every receipt in the tree is
`python -m styxx.sworn check`, the draw is `styxx.beacon.select` and `checksum.check_draw_record`, the
reading is `papers/checksum/score.py`. What it does that the manual steps would not make obvious:

- it FAILS when tracked files differ from the commit, because every step reads the working tree
  (`--allow-dirty` checks the working tree knowingly);
- it refuses an `--expect-head` that is not the full 64 hex before spending minutes on the log;
- it hands `check` the sidecar and then compares the `.md` on disk byte for byte with the document the
  sidecar renders, so a document edited after its receipt FAILS;
- it tallies the verdicts of the documents the receipts re-derive, and prints every one that is not
  SWORN-HELD, so a re-derived FAILED document is never read as a held one;
- it reports the sworn-action samples, which were issued against temporary files, as not checkable
  here rather than as failures;
- it re-checks every fingerprint beside a beacon-drawn certs file with `check_draw_record`;
- it compares only committed scorecards of the scorer's current schema, and FAILS one written for
  other certs bytes or not matching what the scorer reads today.

SKIP is not a pass, and the table says why each step was skipped. Expect a few minutes: the ferry log
re-derives every document and the receipts run one verifier process each.

## what the lab claims, and where each claim's receipt is

The only positioning sentence the lab may say is the one `SURVEY_sand_neighbours_pass3_2026_09_14.md`
(sworn) licenses, read with `CORRECTION_sand_neighbours_pass3_2026_09_14.md` beside it: "we know of no
lab that" does four things at once, each with its neighbours named inside the sentence. Thirty-seven
sources across three frozen-protocol passes; two clauses retired (the plate is Perrig & Song 1999
applied to receipts; a seal on a public chain is timestamping, 1991); none of the four survivors is
unoccupied. The fingerprint clause survives on what it measures — teacher-forced log-probabilities —
more than on its floor or its interval: the correction shows the nearest source (ChatLog) carries the
clause's four object elements under one reading of a term the protocol left undefined, and not the
log-probabilities. The words "first", "novel" and "revolutionary" are not licensed by anything in this
repository.

## if something does not re-derive

Open an issue with the command, the commit, and the output. `BOUNTY.md` says which disagreements
are paid and how; the lab's own second-machine re-run is why the tolerance rule exists.
