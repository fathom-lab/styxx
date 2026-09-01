# RECON — what was actually true when charon work started

Fathom Lab · 2026-09-01 · Deliverable 1 of the charon brief. Every line below is command output
taken from the working tree, not from the brief. **The brief was written from outside — the
shipped wheel, the changelog, the public pages. Where the tree disagrees, the tree wins and the
disagreement is recorded here.** No code was written before this document.

---

## Confirmed as the brief describes

```
$ pip show styxx | head -3
Name: styxx
Version: 7.47.0

$ python -c "import styxx.migrate"
ModuleNotFoundError: No module named 'styxx.migrate'

$ python -c "import pkgutil,styxx;print('anthropic_hack' in [m.name for m in pkgutil.iter_modules(styxx.__path__)])"
True

$ python -m styxx publish --help
usage: styxx publish [-h] --name NAME [--dry-run] [--endpoint ENDPOINT]
  --endpoint ENDPOINT  custom endpoint URL (default: fathom.darkflobi.com)

$ ruff check styxx
All checks passed!

$ python scripts/rigor_gate.py
rigor-gate: scanned 734 result JSON(s) - 734 clean, 0 FLAGGED
```

`migrate` is absent, `anthropic_hack` still ships under that name, `publish` is agent
personality/telemetry against `fathom.darkflobi.com` and is not a certificate client. Rule 3 of
the brief is satisfied at the start of work: ruff clean, rigor gate green.

`styxx --help` lists 40 subcommands. **There is no `migrate` and no `charon`.**

---

## Where the tree disagrees with the brief

### 1. The corpus is larger and one certificate has drifted

```
$ python -m styxx.corpus_audit papers/
corpus papers: 208 certificates | HELD 200  FAILED 8  unresolved 0
               verdict-drift 1  receipt-drift 0  incomplete 1  receipt-changed 1
  epistemics: 6330 verified | obligated 2660 unobligated 3670 (rate 0.5798)
              | weakest 2116 (0.3343) | 0 pre-v1
```

The brief expected **202 certificates, HELD 195, FAILED 7**. Observed **208 / 200 / 8**.

**The three drift flags matter more than the counts and the brief does not mention them.**
`verdict-drift 1` is a certificate whose verdict has moved since it was issued; `incomplete 1`
and `receipt-changed 1` are the other two. Under the charon spec these are not curiosities —
`verdict-drift` is the live, already-existing instance of the `SKEW` status the brief asks the
spec to define. The seed should be built against these three by name.

**A second denominator exists and must not be conflated.** A survey run today counted **241**
`*.certificate.json` + `CERT_*.json` files under `papers/`, where `corpus_audit` reports 208
certificates. The two numbers count different things — files on disk versus certificates the
auditor resolves to a document — and any charon corpus page must say which it is using.

### 2. GitHub releases are NOT stuck at v7.24.3

```
$ gh release list --limit 5
v7.47.0   Latest   2026-09-01
v7.46.0            2026-08-31
v7.45.0            2026-08-22
v7.44.2            2026-08-21
v7.44.1            2026-08-21
```

The brief's phase-0 item — *"the releases page shows v7.24.3 as latest while pypi is at
7.47.0"* — **is already fixed.** GitHub and PyPI agree at 7.47.0 and `git tag` ends
`v7.45.0 / v7.46.0 / v7.47.0`. That debt item is closed before it was opened, and phase 0 should
drop it rather than re-do it. The README byte-comparison remains unchecked and stays in scope.

### 3. arXiv: the "nobody has done this" claim is partly already done here

`papers/arxiv/SUBMIT.md` exists and stages three submissions. Submission 1 (Frame-Locality) is
described as:

> `main.tex` + `anc/` holding the OATH certificate and all 17 receipts — arXiv publishes
> ancillary files alongside the source, **so the paper ships self-verifying**

**Phase 4's central idea is already implemented for at least one paper**, verified to compile
with a 60/60 fidelity gate. Phase 4 is therefore not "do it four times" from zero; it is "finish
and capsule what is staged." The brief's framing overstates the remaining work.

The endorsement blocker is confirmed on file (cs.LG). `SUBMIT.md` states submission requires
account login and that **credential entry is agent-prohibited** — consistent with the brief's
rule that Alex submits. No credential file was opened.

### 4. The `charon` grep hits are all false positives

`grep -ril charon` returns 5 files, none of them a prior implementation or concept note:

- `papers/agent-conscience/pr_strata.json` — *"during mutual eclipses of Pluto and Charon"*, an
  astronomy question in a QA corpus.
- `papers/dogfood-self-audit/CONFAB_DOSE_gemini_L3_all_day.json` — *"basic `charon` functionality
  implemented"*, which is **a model confabulating in a confabulation test**, not a record of
  work.

Charon does not exist in this tree. The brief's expectation holds; the hits are noise.

---

## The design finding that came out of today's dogfood, and that the brief does not cover

**Good news first, and it is load-bearing.** A v0.1 capsule **embeds its receipts**. The payload
carries `spec, created, document, receipts, certificate, verifier`, and the receipt entries are
`{name, b64}` — base64 bytes, not paths. Layer-2 (`python -m styxx.capsule verify`) is therefore
**a pure function of the capsule bytes**, exactly as the brief's rule 1 and admission rule
require. Charon's core premise is sound and I verified it rather than assuming it.

**The attack the spec must name.** `DOGFOOD_2026_09_01_the_instruments_on_todays_work.md`,
written today, established that an OATH verdict on **fixed document bytes** moves with the
receipt list the author supplies:

| receipts supplied | UNGROUNDED | verdict |
|---|---|---|
| the 5 the document cites | 3 | OATH-FAILED |
| 8 | 1 | OATH-FAILED |
| the full pool of both arcs | 0 | **OATH-HELD** |

and that the token which flipped it (`55`) had **257 candidate leaves in the widened pool, 256 of
them array indices, line numbers and column numbers**, because the verifier value-matches without
comparing the receipt path to the claim.

Since the minter chooses which receipts to embed at mint time, **a submitter can shop for
HELD by embedding a larger receipt set.** Charon will faithfully reproduce that verdict and mark
it HELD, because charon's job is reproduction, not adjudication — and it will be correct to do so
while being useless as a signal.

This does not break charon. It bounds what an entry means, and the bound is sharper than the
brief's *"it does not know whether receipts are truthful, only that the document matches them."*
The sharper statement is:

> **A HELD entry means: this document matched the receipt set its author chose to embed. A larger
> embedded receipt set makes HELD strictly easier to obtain.**

**Recommended, for the spec rather than for code:** record `n_receipts` and the sha256 of each
embedded receipt in the entry line, so receipt-set growth is visible in the log and a reader can
see when a HELD was bought with volume. This costs nothing at ingest and changes no verdict.

---

## State at start of work

| | |
|---|---|
| version | 7.47.0, PyPI and GitHub agree |
| ruff | clean |
| rigor gate | 734 result JSONs, 0 flagged |
| corpus | 208 certificates, 200 HELD, 8 FAILED, 3 drift flags |
| obligation rate | 0.5798 unobligated |
| `migrate` | absent, referenced publicly |
| `anthropic_hack` | present under that name |
| charon | does not exist |
| open PR | #50, 10 commits, all checks green |

## Not done in this recon

- The README byte-comparison between the shipped wheel and GitHub main.
- `oathready` over `web/` and the netlify site source.
- Reading `styxx/handshake.py`, `styxx/seal.py`, `styxx/protocol.py`, `scripts/rigor_gate.py`
  line by line; only their entry points and exit status were checked.
- Any decision about `migrate` — that is phase 0 and it is a ship-or-purge decision, not a recon
  finding.
