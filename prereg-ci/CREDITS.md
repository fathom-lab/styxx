# prereg-ci: credit

A NOTE on the work prereg-ci follows, as its design credits it (the design's section 2.1 and the prior-art notes'
addendum, both in the lab's working tree). Each entry says what the work already does, what prereg-ci adopts from
it, and what was read. Nothing here was run against their code: what each does on the four mutations of
[`README.md`](README.md) is expected from its documentation, and that run is owed. Its result goes to the authors
before any sentence comparing the tools is published.

## reproducible-science (Elliot Tower, 2026)

GitHub `elliottower`, MIT, https://github.com/elliottower/reproducible-science. Read: the root README and the
`packages/prereg` README (the design's revision), pull requests #94 and #101 (its red team).

- **What it does.** `prereg freeze` records a whole-file sha256, the commit, a UTC time and an access level, sends
  the digest to OpenTimestamps calendars with no account, adds `-text` for the frozen file to `.gitattributes`, and
  refuses a second freeze; `prereg check` exits 1 on a changed file, and `--staged` refuses an index that changes
  one; `prereg amend` writes a numbered amendment file with four required fields, one of them what had been seen;
  `results seal` hashes inputs before a run. Per its README, the engine computes no verdict.
- **What prereg-ci adopts.** The `seen` field on an amendment line and its four access levels (`nothing run`, `no
  results seen`, `results not opened`, `results seen`); OpenTimestamps as an outside time rung (printed, T3b);
  `-text` for sealed files, which `seal` prints as advice. Its refusal of a second freeze is, in v0, a note: `seal`
  prints a line that records bytes another line already records with the line it repeats. The refusal of a
  changed sealed decision (`DECISION_EDITED_AFTER_SEAL`) is the one its `CHANGED` already gives.

## Panossian (2026)

Andre Panossian, "Does the way we write a theory change the program an LLM builds from it? A prospective randomized
study of renderer format in LLM theory-to-program translation", arXiv 2608.10314, 2026-08-10,
https://arxiv.org/abs/2608.10314. Read: the abstract (the design's revision), the full text (its red team).

- **What it does.** One preregistered study with an outcome and claim matrix: support criteria, thresholds and the
  sentence published under each verdict, sealed with an RFC 3161 token and three OpenTimestamps calendars, and a
  driver that refuses unless the seal is strictly earlier; a deterministic evaluator emitted the registered verdict
  token, `NOT_SUPPORTED`, on both hypotheses.
- **What prereg-ci adopts.** The sentence per verdict, as the `withdraws` key on an outcome row and the licence
  check over shipped surfaces; RFC 3161 and OpenTimestamps as the outside time rungs (T3a, verified offline, and
  T3b); an invalidation record for an abandoned seal, as the manifest's `withdrawn` kind.

## honest-signal (Alessandro Cardurani, 2026)

https://github.com/alexcard3/honest-signal, code MIT, docs CC-BY-4.0. Read: the README and `firewall.py`, through a
fetch summary (the design's scout).

- **What it does.** A required status check over paired preregistration and result files: the preregistration
  exists; the commit that added it is a strict ancestor of the result's commit (the same commit fails, a rename
  fails closed, a shallow clone aborts); its blob is unchanged at HEAD and at the result's commit; the kill
  criterion is not vacuous. Its README says it does not read the result file.
- **What prereg-ci adopts.** Strict ancestry for order (`SEAL_NOT_BEFORE_RECEIPT`), a shallow clone read as
  UNCHECKABLE, renames and deletions of a sealed PREREG failing closed. Its vacuity rule is already credited in the
  lab's gates scanner (`styxx/protocol.py`, the vacuity check), which `lint` reads as `GATE_VACUOUS`.

## preregister (Nick Harris, 2026)

https://github.com/nickharris808/preregister, Apache-2.0, v0.1.0. Read: the README (the design's scout).

- **What it does.** A JSON decision rule over metrics with declared supports; each branch is labelled reachable
  with a witness, unreachable or unknown, and a plan whose finding or null can never fire is not sealed; `score`
  refuses on a digest mismatch, a missing metric or a value outside its declared support.
- **What prereg-ci adopts.** Declared supports and branch reachability, for v1 (not in v0). v0's `lint` asks
  whether every assignment of the gates reaches a row; it does not ask whether each gate can take both values,
  which is the question preregister answers.

## scienceverse (Daniël Lakens and Lisa M. DeBruine, 2021)

"Improving Transparency, Falsifiability, and Rigor by Making Hypothesis Tests Machine-Readable", Advances in Methods
and Practices in Psychological Science 4(2), 2021, doi 10.1177/2515245920970949. Read: the abstract and the
package's demo (the design's scout; the package is marked experimental).

- **What it does.** Hypotheses as criteria plus boolean evaluation rules that give a conclusion string, saved as
  JSON before analysis and after it.
- **What prereg-ci adopts.** The format precedent: a gates table and its outcome rows are criteria and evaluation
  rules in another syntax, published by these authors in 2021. prereg-ci's part is the enforcement around them:
  the seal in git, the re-derivation at merge, the refusal.

## also read by the design

Each is in the design's section 2.1 with its source: experimentation platforms that map a rule written before launch
to a ship decision (GrowthBook, Eppo, Spotify Confidence, with Schultzberg, Ankargren and Frånberg, arXiv 2402.11609,
2024); RegCheck (Cummins, Clarke, Hussey and Elson, arXiv 2601.13330, 2026), an LLM-assisted comparison of a
registration with its paper, with a human judging each discrepancy; Popper and Aver (Jimenez et al., 2016),
validation statements run in CI; OSF Registries and AsPredicted, third-party time custody; in-toto (Torres-Arias et
al., 2019), whose expiry idea is `close_after`; ICH E9 (1998), whose two-timestamp discipline is the time ladder;
promptfoo and CML, thresholds that gate CI.
