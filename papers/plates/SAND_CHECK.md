# the sand check — how a stranger reproduces the lab, and gets paid when it can't

Three ways in, from easiest to strictest. Every one ends with something you can post.

## 1. on a phone (no install)

Open `papers/plates/plate.html` (or the hosted copy) and paste a receipt digest — the 64-character
string under any plate the lab posts. You get the plate for that digest. Same lines as the lab's
plate means the same digest. That is not yet a reproduction; it is a way to read a receipt
without reading json. The mapping is `plate/v1`; the page and `styxx.plate` derive the same modes
from the same bytes.

## 2. reproduce a post (one command, any laptop)

    git clone https://github.com/fathom-lab/styxx && cd styxx
    git checkout <the commit the post names>
    pip install -e '.[plate]'
    python -m styxx.sworn verify <the post's .md> --repo . --commit <that commit> --out mine.json
    python -m styxx.plate $(python -c "import json;print(json.load(open('mine.json'))['digest'])")

Post the picture. If it matches the lab's plate for that post, you reproduced the post's
receipt. If it doesn't, keep going to 3 — you may be holding a bounty.

## 3. file the record (this is what gets paid)

    python -m styxx.challenge <the .md> <the lab's .sworn-receipt.json> --repo . --out challenge.json

The record says `agree: true` (a replication — open a PR adding one line to `REPLICATIONS.md`
with your `record_sha256`) or `agree: false` (a challenge — open an issue titled `challenge:
<document>` with `challenge.json` attached). The record hashes both receipts; it cannot be edited
afterwards to say something else.

For a *result* rather than a post, re-run its recipe (the RESULT names it — e.g.
`python papers/checksum/run_smollm_quant.py`) and compare the numbers the RESULT swears to. A
different number, reproducibly, with your environment stated, is a challenge too.

## what earns a bounty

- a challenge record with `agree: false` and `same_build: true` that the lab cannot explain by a
  documented environment difference;
- a number in a sworn RESULT that does not reproduce from its own recipe on a second machine;
- a sworn span that is HELD by the verifier but false about the world (the receipt is right and
  the sentence is wrong) — say which sentence and why.

## what does not

- a plate that looks different because your matplotlib differs (compare `coefficients_sha256` in
  the json — that is the target, not the png bytes);
- a challenge against a document at a commit it does not name;
- disagreements about what a number *means*. The instrument reports magnitude and resolution;
  meaning is a preregistered decision, and you can challenge the prereg by proposing a better one.

Amounts, payment and timing are set by the lab in `BOUNTY.md`. The verifier decides who is right.
The lab decides how much that is worth. Those are two different decisions, on purpose.
