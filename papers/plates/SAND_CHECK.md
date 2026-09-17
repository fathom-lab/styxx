# the sand check — how a stranger reproduces the lab, and gets paid when it can't

Three ways in, from easiest to strictest. Every one ends with something you can post. (Revised
2026-09-13 after the lab's own red team walked it: the earlier text said a plate's lines were a
digest and a record could not be edited; neither was true, and both sentences are gone.)

## 1. on a phone (no install)

Open `papers/plates/plate.html` from a clone and paste a receipt digest — the 64-character string
under any plate the lab posts. You get the plate for that digest. The plate is a reading aid for
the 64 characters, not a substitute for them: about 2^23 distinct mode captions and far fewer
distinguishable pictures stand for 2^256 possible digests, so two plates with the same lines are
two digests that agree on the bits the picture draws, and nothing more. Compare the full string,
then enjoy the picture. This is not yet a reproduction; it is a way to read a receipt without
reading json. The mapping is `plate/v1`; the page runs the same parameter derivation as
`styxx.plate` (a test executes the page's own script against the python plate), and before
2026-09-13 the page drew the vertical mirror — if you saved a plate from the page before that,
flip it.

## 2. reproduce a post (one command, any laptop)

    git clone https://github.com/fathom-lab/styxx && cd styxx
    git checkout <the commit the post names>
    pip install -e '.[plate]'            # git and python 3.9+; matplotlib and scipy come with the extra
    python -m styxx.sworn verify <the post's .md> --repo . --commit <that commit> --out mine.json
    python -m styxx.plate <the digest printed by verify> mine.png

Compare the digest `verify` printed with the digest under the lab's plate. If they are equal you
reproduced the post's receipt — the plate you just drew is that comparison as sand, nothing else.
If they differ, keep going to 3 — you may be holding a bounty. (`mine.png` is the output path;
without one the plate is written to the current directory as `plate_<12 hex>.png`.)

## 3. file the record (this is what gets paid)

    python -m styxx.challenge <the .md> <the lab's .sworn-receipt.json> --repo . --out challenge.json

You need a full clone at the commit the receipt names: a shallow clone or a zip cannot re-derive
a receipt, and the tool refuses rather than write a record that would read as a bounty claim.
Keep the document's file name — the receipt digests the name. The command writes two files: the
record (`challenge.json`) and your own receipt beside it (`challenge.mine.sworn-receipt.json`).
Attach both.

The record says `agree: true` (a replication) or `agree: false` (a challenge), and `why` says
which half disagreed — the verdict, the digest, or the verifier build. A record is a self-report:
`record_sha256` names the record's own bytes and nothing signs it, so the lab settles a record by
re-running your stated steps, never by reading it. That is why the record carries the commit, the
document name, both verifier builds and the hash of your receipt.

- a challenge: open an issue titled `challenge: <document>` with the two files attached;
- a replication: open an issue titled `replication: <document>` with the two files; the lab
  re-runs and adds the row to `REPLICATIONS.md` (sworn-document section) with your handle and
  your `record_sha256`.

For a *result* rather than a post, re-run its recipe (the RESULT names it — e.g.
`python papers/checksum/run_smollm_quant.py`) and compare the numbers the RESULT swears to,
against the tolerance and environment the RESULT or a correction beside it states. A different
number outside that tolerance, reproducibly, with your environment stated, is a challenge too.

## what earns a bounty

- a challenge record with `agree: false` and `same_build: true` that the lab cannot explain by a
  documented environment difference;
- a number in a sworn RESULT that does not reproduce from its own recipe on a second machine,
  outside the tolerance the RESULT (or a correction committed beside it) states — a RESULT that
  states none is not covered until one does (see `BOUNTY.md`, Tolerance);
- a sworn span that is HELD by the verifier but false about the world (the receipt is right and
  the sentence is wrong) — say which sentence and why.

## what does not

- a plate that looks different because your matplotlib differs (compare `coefficients_sha256` in
  the json — that is the target, not the png bytes, and the raw floats in the json can differ in
  their last digits across machines too);
- a challenge against a document at a commit it does not name, or from a checkout that does not
  have that commit;
- a record produced by a modified verifier (`same_build: false`): the digest differs by
  construction; check out the commit and run again;
- disagreements about what a number *means*. The instrument reports magnitude and resolution;
  meaning is a preregistered decision, and you can challenge the prereg by proposing a better one.

Amounts, payment and timing are set by the lab in `BOUNTY.md`. The verifier decides whether a
record disagrees. The lab decides how much that is worth. Those are two different decisions, on
purpose.
