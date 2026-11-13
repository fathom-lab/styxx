# PROGRESS

- started fresh 2026-11-13; no prior files of mine found.
- phase: reading the text and the schema (vectors NOT opened yet).
- checker/: json.go canon.go decimal.go doc.go refs.go manifest.go check.go unicode15.go written from the text.
- Go 1.25.6 unicode tables reproduce BOTH 7.2 digests (L/N and Nd) -> host tables are 15.0.0; self-test runs at start.
- next: schema.go (mini JSON Schema for 11.3), receipt.go, GUESSES.md, then inspect vector file SHAPE only, write the runner, first run.
- GUESSES.md written (33 guesses) BEFORE vectors. Now: vector file shape inspection (paths/kinds only).
- FIRST RUN 2026-11-13 ~14:03 UTC: 2193 of 2193 pass (conformance 1718/1718, hand 416/416, jcs 27/27, recheck 32/32). saved first_run.txt; never overwrite.
- next: power check of the harness + per-guess alternate readings (ALT env flag) to learn which guesses the vectors settle.
- alternate readings run (alt_runs.txt): 24 guesses switched, 16 settled by vectors, 8 open (G1 G2 G3 G8 G10 G17 G19 G30); controls C1/C2/C3 fail 11/8/155 vectors.
- final_run.txt: 2193 of 2193 (default build, RV0_ALT unset). GUESSES.md settled section written.
- DONE. Not implemented: the 3.3 `type` code (no byte input reaches it), a verifier object in emitted receipts, legacy-projection profile (text: no v0 checker emits it).
