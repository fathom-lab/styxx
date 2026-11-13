# cleanroom-go: evidence, not an implementation the lab supports

A Go checker for RECEIPT v0, written on 2026-11-13 by one agent of the lab's own workflow, from
`RECEIPT_v0.md`, `receipt-v0.schema.json` and the vector file, without the lab's code. It is kept
here as the record of that test. The lab does not maintain it, does not fix it and does not call it
an implementation of the text. Its bugs stay in. The full account is
`../RESULT_cleanroom_receipt_v0.md`.

## What is here, and how it was copied

Every file below was copied byte for byte from the clean-room folder
`wt/_receipt_v0_cleanroom_2026_11_13`. The sha256 of each file was compared before and after the copy,
and the lists are equal.

- `checker/`: 15 Go files, `go.mod` and the schema the build embeds. Standard library only.
- `GUESSES.md`: the author's 33 guesses, written before the vectors were opened, and the run of 24 of
  them both ways.
- `files_read.log`: the author's own list of the files it opened.
- `first_run.txt`: the output of its first full run over the vectors.
- `alt_runs.txt`: one summary line for each alternate reading (`RV0_ALT=Gn`).
- `PROGRESS.md`: the author's progress notes.

Two files were left out. The first is the built `checker.exe` (sha256
`8ae3e399fbedaec3815687c351150ccbfdec7b6954c2a62bf1a76d420f5ea138`), which is a Windows binary. The
second is `final_run.txt`, which is byte-identical to `first_run.txt` (sha256
`a376211f87c2f68c405004da408287bc3ff7258a3399d568a5cf0b53ce2ab8e5`).

## How it was made

The given files were the text, the schema and the vector file, placed at 13:50:32Z. The `checker/`
folder was created at 13:56:03Z, `first_run.txt` was written at 14:03:42Z and the final run at
14:08:30Z, all UTC and all read from file times. So 13 min 10 s passed from the text to a run that
passed every vector. The author's log says it read only the shape of the vector file (key paths and
value kinds) before it wrote the runner, and read the expected answers only through the runner.

## What the audit of its reading found

The lab's measurer checked the build against the lab's code. Nothing it found shows that lab code was
copied. Nothing it found shows that the build never saw lab material either.

- 16 of the 96 Go function names equal a name in `check_v0.js` once case and underscores are ignored
  (`main` is one of the 16). For comparison, `check_v0.js`, which was written with `check_v0.py`
  open, shares 57 of its 83 names with it. Most of the shared Go names follow from the text's own
  words. One does not: `isHex64Any` is exactly a `check_v0.js` name, and the text gives no source
  for it.
- One comment phrase is shared with `check_v0.py`: "keys written more than once".
- The session scratchpad the author used was shared with every agent of the workflow. During the
  build it held lab material from other agents (a derive/v0-draft row, a lab pytest temp folder).
  No `check_v0` or `sworn` source was in it. The clean room was an instruction plus a self-report on
  a shared disk. It was not an isolation, and no access log of the operating system exists.
- It is one agent from the same model family as the lab's own authors. It is not an outside
  implementer.

## Pass counts

- The first run (`first_run.txt`) passed 2193 of 2193 vectors of `../vectors/receipt_v0_vectors.json`.
  A build of this copy reproduces that output byte for byte.
- On `../vectors/receipt_v0_vectors_fold.json` (62 vectors, written in the fold round from the
  measurer's gap list), it passes 59 of 62. The 3 it fails are the order-of-faults vectors of 3.3
  (gap G2). The text now settles that order, and the Go build took another reading of it.
- At a nesting depth of 3,000,000 it dies with a fatal goroutine stack overflow. Its parser is
  recursive, and the crash cannot be recovered. At 1,000,000 it answers. 6.13 now says a checker
  must not stop without an answer at any depth, so this is a bug in the build. It stays unfixed.

`tests/test_receipt_v0_fold.py` builds this copy with `go build` when go is on PATH (with
`GOTOOLCHAIN=local GOPROXY=off GOSUMDB=off`), runs both vector files through it and requires
exactly these counts. When go is missing, the test says so loudly and skips.
