# CORRECTION — what this demo directory is, and why it no longer verifies

Fathom Lab · 2026-09-13 · appended beside the demo, which is left as committed.

- **The three dates are labels.** `run_observatory_demo.py` passed `when` = 2026-09-13, -14 and
  -15 to `observe()`; the fingerprint files' own `created` fields read 2026-09-13T15:43:31Z,
  15:43:49Z and 15:44:07Z. Three days were thirty-six seconds on one machine, and neither
  `STATUS.md` nor `log.jsonl` said so. Day 3 is the same weights int8-quantized, which the
  table did not say either.
- **The logged coefficient hashes never matched the files.** `observe()` hashed the in-memory
  float64 geometry while the fingerprint file stores values rounded to 1e-6; a stranger
  re-hashing the file gets a different value for all three lines. The logged distances were
  likewise computed against an in-memory fingerprint, so day 2 read 2.67e-7 nats against a floor
  of 0 while the bytes read exactly 0.
- **`verify()` v0 checked only the chain and the fingerprint bytes.** A forgery of every verdict
  re-hashed forward, a truncated log, and a replaced plate all verified. The head hash was pinned
  nowhere outside the file.

Observatory v1 (`styxx/observatory.py`, same day) writes `taken` beside `when`, records the floor
applied beside the floor measured, derives coefficients and verdicts from the files a stranger
has, refuses an empty rebaseline reason, and takes a pinned head and count in `verify()`. Under
v1 this directory fails verification on every line (coefficients, and the missing applied floor),
which is the finding, not a bug: a v0 log is history. A fresh demo under v1 lives in
`observatory_demo_v1/`.
