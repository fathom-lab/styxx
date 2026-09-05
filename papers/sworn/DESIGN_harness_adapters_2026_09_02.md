# DESIGN — harness adapters at L1 and L2: manifest minters that are not a recorder

Fathom Lab · 2026-09-02 · **A design, frozen before the code it describes.** Leg 3, item 3 of
`papers/PLAN_the_next_level_2026_09_02.md`. It builds on `SPEC_sworn_output_v02_2026_09_02.md`
(rule R6, the trust ladder) and on `styxx.evidence` (`styxx-evidence/v0.2`), and it edits neither.
It makes no numeric claim: the sworn spans below bind sentences to the verifier's and the
evidence reader's bytes at the commit the sidecar names, and nothing else in it is a number.

**Re-sworn 2026-09-05 at f28d35a, before any code.** The draft of 2026-09-02 was never committed.
Between the draft and this commit the verifier's bytes moved under the charon adversarial pass
(`GitTree._index` now streams `cat-file`); the manifest API the adapters call did not move, and the
hash sentence below names the blob as it is at the commit the sidecar names. The Claude Code
section is rewritten against two things read off the live hook documentation and this box's own
transcripts: the hook fires for parallel tool calls and inside subagents under one session id, and
importing `styxx.sworn` runs the package's eager imports on every tool call. The section on
decisions at the end records what the draft left open and how each was closed.

**Prior art, credited where the code will credit it.** in-toto Witness records what a step did
and signs it; we adjudicate what an author *said* against what a harness recorded, and sign
nothing. The asymmetry — the producer carries the burden of a checkable commitment, the consumer
runs a cheap check — is Necula's. The ladder is SLSA-shaped. `authored_sha256` is in-toto's
*products* set, unsigned. These adapters add no rung, no kind, no verdict and no signature to any
of that; they mint the one object the verifier already reads.

## What an adapter is, and is not

An adapter turns bytes a harness already holds — a test report, a CI event payload and a diff, a
tool-hook payload — into a `sworn/manifest/0.2` file: receipts with a `kind_of_source` from the
closed external vocabulary, a `harness_note` naming where each came from, an `authored_sha256`
set where the harness could see what the agent wrote, and a `rung`. The verifier
(`styxx.sworn`) resolves `rN` spans against that file and prints the rung and the harness string
beside every verdict.

An adapter is **not a recorder**: it does not observe a process, does not sign, does not verify
a signature, does not fetch, and does not decide whether the bytes it was handed are genuine.
It is **not a verifier**: it produces no verdict and has no accusing branch. It is **not a gate**:
every entry point exits zero on every input it can write a manifest for, and a hook that cannot
write one says so on stderr and still exits zero, because a hook that blocks the agent has become
a gate without a measured precision.

<sworn r="path:styxx/sworn.py" k="quote">The verifier's ladder is the closed tuple `RUNGS = ("L1", "L2")`, and nothing this leg adds may widen it.</sworn>
<sworn r="path:styxx/sworn.py" k="quote">The verifier refuses the signed rung by name, `RUNGS_RESERVED = ("L3",)`, so no adapter here may declare it.</sworn>
<sworn r="path:styxx/evidence.py" k="quote">The evidence reader's vocabulary is `VERDICTS = ("VERIFIED", "UNCHECKABLE")` and the JUnit adapter adds no word to it.</sworn>
<sworn r="path:styxx/sworn.py" k="hash">At the commit this design names the verifier the adapters target hashes to 110f3461bb6772248c987f9707ac7ef3a0d5df4d37b4d26124b7aa82551c04c9.</sworn>
<sworn r="path:styxx/evidence.py" k="hash">At the same commit the evidence reader hashes to 598406796b6d550011ad3ce4d6c4c0c33328f3fc25166b9b5e39d154c84e8657.</sworn>

## The ladder, as R6 states it, and what each adapter can reach

| rung | who minted | what it is worth | which adapter may declare it |
|---|---|---|---|
| L0 | the author, or bytes in `authored_sha256` | MALFORMED `receipt_author_minted` | none; an adapter that is handed bytes the caller says the agent wrote records them into `authored_sha256`, and the verifier does the rest |
| L1 | a local hook sharing a filesystem and a shell with the agent | **weak** — the agent's shell can write what the hook later reads | Claude Code hooks (always); JUnit (when the caller declares it); GitHub (when the caller declares it) |
| L2 | a runner that minted after the agent's turn ended and that the agent could not write to | the strong rung this version reaches | JUnit and GitHub, **only on the caller's declaration** that the run was after the turn on a machine or branch the agent could not write to |
| L3 | a signed attestation whose signature the verifier checked | reserved | none |

**The rung is declared, never detected.** No adapter can see whether its inputs were produced
after the agent's turn or on a machine the agent could not reach. The JUnit adapter takes `rung`
as a required argument and its docstring says the adapter cannot know which is true. The GitHub
adapter prints L2 only when the caller also declares `ran_after_turn_on_base`, and even then it
reads one thing off the event that can contradict the declaration (below). The Claude Code
adapter declares L1 and offers no way to declare higher, because a hook is by construction the
thing R6 calls weak. Every adapter refuses L3 at its argument parser, as the verifier's own
`manifest new` already does.

## The fork caveat, stated once and printed into every GitHub manifest

On a `pull_request` event from a fork, the job that mints the manifest runs the workflow file
*as it exists in the pull request's head*, on a runner the claimant's changes configured. The
manifest is then minted by a party the claimant controls, and "a party other than the claimant"
— the sentence L2 rests on — does not hold. It holds only for a workflow pinned to the base
branch (`pull_request_target`, which runs the base's workflow file with the base's secrets, and
which therefore must not check out and execute head content) or for a manifest attested outside
the job. The GitHub adapter therefore:

- reads `head.repo.full_name` against `base.repo.full_name` (and `head.repo.fork`) from the
  event payload and records `fork: true|false|unknown` into the manifest's `harness` string, with
  this caveat verbatim, on every manifest it mints, fork or not; `unknown` is the case where the
  head repository is absent from the payload (a deleted fork), and it is treated as a fork for
  every refusal below;
- refuses to print L2 for a fork `pull_request` event unless the caller separately declares
  `base_pinned_workflow`, which is the caller asserting something the event bytes cannot show
  — the refusal is a `ValueError` at mint time, not a verdict, and the caller may mint at L1
  instead;
- never lowers or raises a rung on its own; it refuses or it prints what was declared.

The Action that would consume this adapter (leg 3, item 4) is not this design; its README owes
the same sentence.

## The three adapters, receipt by receipt

### `styxx.harness.junit` — a test report through `styxx.evidence`

Input: one file of JUnit XML or in-toto test-result bytes, read from disk at the path the caller
gave; `rung` (required, `L1` or `L2`); an optional `turn` id; an optional list of byte-objects
the caller knows the agent wrote this turn. The counts come from `styxx.evidence.load_evidence`,
which resolves them from `<testcase>` children and never from root attributes, and whose
vocabulary has no accusing word. The adapter reads the report-only band and copies numbers out
of it; it derives nothing.

| id | bytes | kind | complete | note |
|---|---|---|---|---|
| r1 | the passed count as ASCII digits | `test_report` | true | "passed count resolved by styxx.evidence from PATH" |
| r2 | the failures count as ASCII digits (failed testcases; harness errors are kept apart, as the reader keeps them) | `test_report` | true | likewise |
| r3 | the report bytes, whole | `test_report` | true | the path as given |
| r4 | `load_evidence`'s returned object as canonical JSON (RFC 8785, one trailing LF) — so `r4#/totals/errors`, `r4#/outcome`, `r4#/sources/0/producer_guess` are addressable leaves | `test_report` | true | "styxx.evidence.load_evidence over PATH, byte for byte" |

The path enters r4: `load_evidence` records `paths_requested` and each source's `path` as the
caller gave them, and the adapter passes the path through unchanged so that a reader handed the
same path re-derives r4 byte for byte. A caller who does not want a machine path inside a
receipt gives a path relative to the repository root, which is what the lab's own RESULT does. A
report that cannot be read is a usage error at the command line (exit code two), not a manifest
with an `unreadable` reason carrying the path of a file that was not there.

When the reader could not parse the file (`unparsed` non-empty, no source), **r1 and r2 are not
minted**. A count of zero from a report that did not parse is absence printed as a number — the
defect `styxx.evidence` calls M7 — and a span that names `r1` then resolves UNRESOLVED
`manifest_id_missing`, which is the verifier saying it could not see, not a lie made to pass. r3
and r4 are minted regardless; r4 carries the parse failure by name.

What it cannot attest: that the report describes a run of the code at any commit (the reader
says as much under `binding`); that the run was complete; that the agent did not write the
report. The last is why the `authored` argument exists: bytes the caller hands it are recorded
into `authored_sha256`, and a report the agent wrote then resolves MALFORMED
`receipt_author_minted` by set membership, end to end. If the caller hands nothing, the adapter
records nothing and says so in its harness string.

### `styxx.harness.github` — an event payload and a diff the caller fetched

Input: the parsed `GITHUB_EVENT_PATH` JSON and its raw bytes; the event name
(`pull_request`, `pull_request_target`, `push` — anything else is refused); the diff bytes, or
`None`; `diff_complete`, which the adapter sets on r1 only when the caller asserts it (a caller
that streamed a truncated diff and says nothing gets `complete: false`, so `absent` over r1 is
MALFORMED rather than a hollow oath); `rung`; `ran_after_turn_on_base`; `base_pinned_workflow`.
No network is opened inside the module: whatever fetched the diff is the caller's, and the
adapter records only that the caller said it was whole. The module reads no environment
variable; the command line reads `GITHUB_EVENT_PATH` and `GITHUB_EVENT_NAME` as argument
defaults and nowhere else.

| id | bytes | kind | complete | note |
|---|---|---|---|---|
| r1 | the diff bytes (omitted when `None`) | `http_fetch` | as the caller asserted | "diff supplied by the caller; completeness asserted by the caller, not observed" |
| r2 | the base sha as ASCII | `harness_note` | true | "base sha from EVENT_NAME event (pull_request.base.sha or push.before)" |
| r3 | the head sha as ASCII | `harness_note` | true | likewise for head / after |
| r4 | the event name as ASCII | `harness_note` | true | "GITHUB_EVENT_NAME as given" |
| r5 | the event payload bytes, whole | `harness_note` | true | "GITHUB_EVENT_PATH bytes as given" — so `r5#/pull_request/number`, `r5#/pull_request/head/repo/fork` are leaves |

What it cannot attest: that the diff is the diff of head against base (the caller fetched it);
that the event was delivered by GitHub (the bytes were read from a path); that anything ran.

### `styxx.harness.claude_code` — a PostToolUse stager and a Stop finaliser, at L1

**Two events, two costs, one manifest.** A PostToolUse hook runs once per tool call, for parallel
tool calls in one assistant message, and inside subagents, all under one `session_id`. Importing
`styxx.sworn` runs the package's eager imports on each of those processes, and a read-modify-write
of one manifest file by concurrent hook processes drops receipts or corrupts the file. So the
PostToolUse path imports nothing from `styxx`: it stages one JSON file per event, atomically
(write beside, then rename), named by the event's `tool_use_id`, and the Stop finaliser — which
runs once per turn, after the tools have returned — folds the staged events into a manifest,
deterministically: receipt ids are assigned in the order of (capture time, `tool_use_id`), never
as "next free id" at staging time, so a re-fold of the same events is byte-identical, and the
manifest's `minted_at` is the capture time of the last event folded. Stop fires on every turn and
again when a Stop hook has blocked (ours never does); the finaliser is therefore idempotent and
refolds from the staged set each time. `claude --resume` reuses the `session_id`, so a manifest
accretes across resumed sessions; every receipt's note carries the `prompt_id` when the payload
has one, so a reader can cut by turn. The staging module is stdlib-only at import; the entry
script under `integrations/` loads it by file path so that `styxx/__init__.py` never runs on the
PostToolUse path.

**Where the files go.** A directory the hook configuration names (`--dir`, else
`STYXX_SWORN_MANIFEST_DIR`, else a per-user data directory) that **must lie outside the
workspace**: if it resolves under the payload's `cwd` or under `CLAUDE_PROJECT_DIR` (the
session's start directory, which does not follow a worktree the way `cwd` does), the adapter
writes nothing and says why on stderr — a manifest the agent can `Read` into a receipt of itself
is the L0 case wearing an L1 label. The test is `realpath` plus case folding plus a path-prefix
check on path components, on both roots, because a string prefix wrongly treats a sibling
directory whose name extends the workspace's as inside it; a directory that is not a native
absolute path (a `\\?\` prefix, a POSIX-style spelling of a Windows path) is refused rather than
guessed at. The rule is hygiene, not trust: the agent's shell can write to that directory too,
and that is what L1 means. The `session_id` becomes a path component only after it matches a
strict pattern (`^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`, and no Windows device name); a payload
whose `session_id` fails it writes nothing.

**What the tools return, read tolerantly.** The live hook documentation and this box's own
transcripts disagree on the shape of `tool_response` (the documentation gives `{success,
content}` for Read and `{success, filePath}` for Write and Edit; the transcripts give `{file:
{content, startLine, numLines, totalLines}}` for Read, `{content, originalFile}` for Write,
`{oldString, newString, originalFile, replaceAll}` for Edit, and a Bash response without an exit
code). The adapter reads both shapes, keys nothing on the response for authorship, and takes what
the agent wrote from `tool_input` (`content`, `new_string`), which both shapes carry, plus the
bytes on disk at `tool_input.file_path` right after the tool ran.

| tool | receipt minted | kind | complete | note |
|---|---|---|---|---|
| Bash | `tool_response.stdout` (or the string response) | `tool_stdout` | false when the response says `interrupted`, or the text carries a truncation marker or a persisted-output stub; true otherwise | the command |
| Bash, stderr non-empty | `tool_response.stderr` | `tool_stderr` | as above | the command |
| Read | `tool_response.file.content`, else `tool_response.content`, else the string response | `file_read` | true only when the response reports the whole file (window starts at line one and spans every line, no token-cap truncation) and `tool_input` set no offset or limit; false otherwise; an image response mints nothing | the file path |
| WebFetch | `tool_response.result`, else `.content`, else the string response | `http_fetch` | **false, always** — the tool returns a rendering of the page, not the page's bytes | the URL |
| Write | none | — | — | `record_authored` over `tool_input.content` and over the file on disk |
| Edit | none | — | — | `record_authored` over `tool_input.new_string`, over the post-edit text reconstructed from `tool_response.originalFile` by the edit's own replacement (`replace_all` honoured), and over the file on disk |
| Write or Edit whose response is a string | none | — | — | the tool failed; nothing is recorded |
| anything else (Grep, Glob, Task, TodoWrite, MultiEdit, NotebookEdit, every MCP tool) | none | — | — | ignored, and the docstring lists the names |

The Stop payload's `last_assistant_message`, when present, is recorded into `authored_sha256`
as well: over-recording can only turn HELD into MALFORMED, and a message the agent wrote is a
byte-object the harness saw the agent write. Every receipt has a byte cap (`RECEIPT_BYTE_CAP`
in the module); a response over the cap is staged as its sha256 alone and folded as a hash-only
entry, which the verifier resolves for `hash` spans and reports `manifest_bytes_absent` for the
rest. A manifest holds every tool response verbatim, so a `Read` of a secrets file or a `Bash`
that printed a token lands in it at rest; the README says to handle the manifest as the
transcript is handled. The finaliser prints nothing on stdout — Stop's stdout is parsed for a
decision — and the manifest's path on stderr.

**This adapter is blind, permanently, to files written by shell commands.** `cat > file`,
`python -c 'open(...).write(...)'`, a heredoc, `git apply`: the hook sees a Bash tool call and its
stdout, never the file. Those bytes never enter `authored_sha256`, and a later `Read` of that file
mints a `file_read` receipt the verifier will accept. The gap is structural — the hook is not a
filesystem monitor and must not pretend to be one — and it is printed into the manifest's
`harness` string on every manifest this adapter mints, so a reader of any verdict at L1 sees it
beside the rung. A second, smaller gap is printed with it: a tool's response can differ from the
bytes on disk (a trimmed trailing newline, a preserved CRLF, an edit matched after whitespace
normalisation), so set membership over exact bytes can miss a file the agent did write; reading
the file from disk after the tool narrows that gap and does not close it. Both gaps say the same
thing R6 says: L1 is weak.

## The purity boundary

`styxx/sworn.py` and `styxx/evidence.py` import nothing from `styxx.harness`, and neither does
`styxx/__init__.py`. The adapters import from them; never the reverse.
`tests/test_harness_purity_boundary.py` walks those modules' ASTs and fails on any `import` or
`from … import` naming `styxx.harness` or `harness`, on any string constant spelling the package
name (an `importlib` smuggle), and on the package appearing in `sys.modules` after the verifier
and the reader are imported in a fresh interpreter. In the other direction the JUnit and GitHub
modules reference none of the ambient-state modules `styxx.evidence` forbids below their
command-line layer, and the PostToolUse entry script imports nothing from `styxx`. The verifier
and the reader stay pure functions of bytes; an adapter is where the ambient world (a path, an
environment variable, stdin, the clock) is allowed in, and it is kept on its own side of the
line so that adding an adapter can never change a verdict.

## Entry points

- `python -m styxx.harness junit REPORT.xml --rung L1|L2 [--turn ID] [--authored FILE …] --out M.json`
- `python -m styxx.harness github --event EVENT.json --event-name NAME [--diff DIFF] [--diff-complete] --rung L1|L2 [--after-turn-on-base] [--base-pinned-workflow] [--turn ID] --out M.json`
- `python -m styxx.harness claude-code post-tool [--dir DIR]` and `… stop [--dir DIR]`, reading
  stdin; `python -m styxx.harness.claude_code` accepts the same words
- `integrations/claude-code/sworn-hooks/post-tool.py` and `stop.py`: thin entry scripts for a
  `settings.json` hook block, with a README whose opening paragraph says this is the weak rung and
  why. The block is documented for a user's own settings and is never committed into this
  repository's `.claude/settings.json`, where it would run for every contributor and every lab
  session.

The JUnit and GitHub commands exit zero when they wrote a manifest and two on a usage error (a
report that is not there, a rung not in the ladder, a declaration the adapter refuses) — a usage
error is the caller declaring something inconsistent, not a verdict. The Claude Code commands
exit zero on every input, including stdin they cannot read, because a hook that blocks is a gate.

## Tests this design commits to

Each adapter mints a manifest that `python -m styxx.sworn verify` resolves against a canned
document, in-process and through the CLI; every manifest prints its rung and the receipt's
`rungs` count agrees; a receipt whose bytes are in `authored_sha256` verifies MALFORMED
`receipt_author_minted` end to end for every adapter; the JUnit adapter withholds r1 and r2 on
an unparsed report, keeps failures apart from harness errors, and writes r4 as the canonical form
a reader can re-derive from the reader's own output; the GitHub adapter refuses L2 without the
declaration, refuses L2 on a fork `pull_request` (and on a `pull_request` whose head repository
is absent) without the base-pinned declaration, accepts `pull_request_target`, reads `push`
events, and refuses any other event name; `absent` over an incomplete diff is MALFORMED; the
Claude Code adapter refuses a directory inside `cwd` (including a case variant, a `..` spelling
and a directory under `CLAUDE_PROJECT_DIR`) and accepts a sibling directory whose name extends
the workspace's, refuses a `\\?\` or POSIX-spelled directory, writes nothing for a `session_id`
that is not a safe path component, marks a truncated or interrupted Bash response incomplete,
marks a windowed Read incomplete, reads both documented and observed response shapes, records
Write and Edit content and the on-disk file into `authored_sha256` and nothing for a failed tool,
stages parallel events without loss, folds them idempotently, and returns zero for every payload
including stdin it cannot read; the Stop path writes nothing to stdout; the README's settings
block parses as JSON; the purity boundary holds in both directions; `pyproject.toml` ships the
package. The outside-workspace rule is tested with Windows paths where the interpreter runs on
Windows. No real Claude Code session is run.

## Decisions taken at re-swearing, recorded so they can be argued with

- The PostToolUse path stages per-event files and the Stop path folds them; the draft's "state
  is the manifest file itself" is withdrawn for the two reasons given above (import cost per
  tool call; concurrent hook processes under one session).
- The default manifest directory is a per-user data directory, reachable by the agent's shell;
  the README says so and offers `STYXX_SWORN_MANIFEST_DIR` for an operator who wants to choose.
- `last_assistant_message` enters `authored_sha256`; a byte cap with a hash-only fallback is
  adopted; MultiEdit, NotebookEdit and MCP tools stay ignored and are named.
- The RESULT that ships with the code swears to a manifest the JUnit adapter itself mints over a
  `--junitxml` run of the adapter tests, at L1, on the author's box: the adapter is dogfooded by
  the document that announces it, and the rung printed is the honest one.
- The design keeps its filename and date and is re-sworn under this header rather than issued as
  a new document, so the plan's reference to it stays true.

## What this design does not say

That any of these rungs has been verified: R6 prints, nothing checks, and L3 is reserved. That an
L2 manifest is trustworthy: it is as trustworthy as the caller's declaration and the runner it
describes. That the Claude Code adapter records what the agent wrote: it records what the Write
and Edit tools reported and what was on disk after them, and is blind to the shell. That a
manifest minted here has been used by anyone outside this lab, or that the format is thereby a
standard of any kind. That the format has been measured — that remains
`DESIGN_sworn_measurement_v2_2026_09_02.md`, waiting on a signature. That the JUnit adapter's
counts are true of any run: they are what the bytes say, resolved by a reader whose own docstring
lists the shards and suites that leave no mark. That the hook payload shapes read here are stable:
they are what the documentation and one machine's transcripts showed on the day, and the adapter
tolerates both because neither is a contract.

---

*A harness is where the world gets in. These three let it in through a named door, write down
which door, and print the rung on the way out. None of them can tell whether the world was
honest; they can only refuse to say it was.*
