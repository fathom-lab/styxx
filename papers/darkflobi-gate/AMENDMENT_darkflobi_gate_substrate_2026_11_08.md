# AMENDMENT — the substrate on every send line: the provider, model and api the gateway selected for the run that wrote it, and a digest of what the agent reads (re-based onto the token-shapes gate, 2026-11-08)

Fathom Lab · written 2026-11-02, re-based 2026-11-08 · Amends `PREREG_darkflobi_gate_2026_10_23.md`, in this
directory, merged as #229 with its section 15 closed at 15.13 and not edited, read with the seven amendment files at
`origin/main` `f0125c8c` (the five before-D0 files of #243, the anchor-native file of #244 and the token-shapes file of
#247), through whose `lock_hashes.py check --amendment` this file is read, after them. Section 12's closing paragraph
provides for it: after the merge "a change to any rule, bar or lock is a new AMENDMENT file under this folder,
lock-hashed in its turn, with deliveries under the old rule counted under the old rule". Written by the lab's builder
for the lead on 2026-11-02, while `W:gate/REQUIRED` is absent (shadow), from the receipted-agent design and its critic
(`RA:CRITIC.md`, whose closing "verdict" section, item 1, is this file's spec), built on a copy in three stages and
revised in a fix round after three adversarial reviews (`S27:BUILD_LOG.md`, `S27:REVIEW_2026_11_02.md`), after the
operator's model change that evening (item 6 of what was observed), again after two re-reviews of the fixed build
(`S27:REREVIEW_ANSWERED.md`), and re-based on 2026-11-08 onto the gate as the token-shapes move of 2026-11-04 left it
(item 7 of what was observed; `S8:BUILD_LOG_v2.md`), since the 2026-11-02 build's rows named the files the
token-shapes move replaced. It adds three keys to the send lines of the stretch it opens and moves no counted
quantity: no bar, no row of the gates block, no reason code, no catch rule, no byte of the trailer, no part of the
count rule and no part of the token-shapes rule. The ordered supersession of lock rows (D9) it was written with is on
`main` since #247, carried there by the token-shapes file byte for byte; this file uses it and declares its
supersessions against the rows in force when it is read. It asks the operator for the decisions under "Whose step".

Receipt key: `W:` = `C:/Users/heyzo/clawd/`; `WT:` = `W:wt/`; `RA:` = `WT:_receipted_agent_2026_10_25/` (the design,
the prototype, `CRITIC.md`); `S27:` = `WT:_substrate_2026_10_27/` (the 2026-11-02 build: its copy `S27:gate_copy/` of
the gate at `4416692c…`, the build log, the reviews, the readings); `S8:` = `WT:_substrate_2026_11_08/` (the re-base:
the fresh copy `S8:gate_copy/` of the gate at `c17e3d06…` with the substrate re-applied, `BUILD_LOG_v2.md`,
`REPORT_rebase.md`, the move tools, the readings); `TS:` = `WT:_token_shapes_2026_11_04/`; `BD:` =
`WT:_darkflobi_gate_2026_10_22/before_d0/`; `D:` = clawdbot's installed `dist/` (build 2026.1.24-3); `RC:` = this
folder's `receipts/substrate_2026_11_02.json` (the 2026-11-02 build's readings, bound to files under `S27:`); `RC8:` =
this folder's `receipts/substrate_2026_11_08.json` (the re-base's readings, bound to files under `S8:`); `PREREG` and
`RULES` = this folder's files at `f0125c8c` (unchanged since `310df453`), line numbers their own; `seq N` = a line of
`W:gate/ledger.jsonl`. [R] marks a reading with its receipt, [J] a judgement. A digest written `abcd1234…` is a sha256
whose full form stands in a lock row, in `RC:`, in `RC8:` or in the reading named.

## What the PREREG says

- `RULES` §13 (l.554-558): "The measurement reads these fields", and a send line's list (l.565-573) names `exit`,
  `channel`, `dest_class`, `payload_kind`, `session_hash`, the digests, the legs, `numerals`, `decision` and the rest;
  nothing on it names the model that wrote the text, the configured chain, or what the agent read before writing.
- `RULES` l.593-595, for a field the plan named and `gate.py` did not write: until then `agent` is read by the folder
  attribution (`reconcile.py` `match_transcripts`, l.593-594), and "A `gate.py` that writes `agent` before the merge is
  restated under the PREREG's §15, and from then the field decides" (l.594-595). After the merge the restatement is
  an AMENDMENT file (`PREREG` l.676-678).
- `PREREG` §5 (l.228-232): component (iv) of `unledgered_deliveries` counts "ledger send lines whose `gate_sha256`
  differs from the lock-hashed gate files' digest"; 15.12, signed in 15.13, reads it per stretch, "each send line
  against the digest of the files locked when the line was written". `RULES` l.599-600: "`gate_sha256` is the gate's
  statement about itself".
- `PREREG` §5 (l.263-266): "Reported beside them, no bar", among them "per-exit and per-agent cells".
- `PREREG` 15.6 (l.958-959): a file moves "by `os.replace` (a copy beside the target, its sha256 checked against the
  tested bytes, one rename), with `REQUIRED` absent before, during and after".
- The lock-rows amendment, item 1: "A path two files name with two digests, or one file names twice with two, is a
  **CLASH** … `check` exits 2 before any row is read", and "Since no path may carry two digests, the order of the files
  cannot change a reading".
- The token-shapes amendment (2026-11-04, #247), "The lock rows that move": the tool's ordered supersession (D9), "as
  built and reviewed on the substrate branch (commit `be538fa4`), carried here byte for byte"; its seven
  declarations re-lock `W:gate/gate.py`, `W:gate/gate_verify.py` and three test files over the count-rule and
  posting-skill rows; its rows for `gate.py` (`c17e3d06…`) and `gate_verify.py` (`5ab8454c…`) are the ones in force
  when this file is read.

So the measured subject's model is recorded nowhere the RESULT reads; and since #247 a later amendment's row for a
path an earlier amendment locked is read when the later file declares the supersession of the row in force, which
this file does (item 7).

## What was observed

1. **No send line names a model** [R, `S27:readings/stage3_2026_11_02/ledger_all_from_seq1.txt`, the live ledger read
   on 2026-11-02, counts only: sha256 `7c1d6889…`, 249 lines, seq 1-248]. 117 send lines (65 at seq 1-135, 52 at seq
   136-248) and none carries `gate_schema`, `substrate` or `subject_digest`; no key of any of them names a provider,
   a model or a position in the chain.
2. **The chain moved in shadow and the ledger shows nothing of it.** At 2026-10-26T14:38:39Z the configured chain
   (`agents.defaults.model` in `~/.clawdbot/clawdbot.json`) gained an eighth candidate, `local/qwen2.5-3b-instruct`
   at 127.0.0.1 [R, `RA:CRITIC.md` §2 and `RA:AMENDMENT_substrate_DRAFT.md` §5, each read through a filter that
   prints key names and model ids only]. The gateway restarted on 2026-10-30 (`started` seq 188 at 19:37:21Z and seq
   189 at 19:42:08Z) [R, `S27:readings/stage3_2026_11_02/ledger_stretch_from_136.txt`]. The lab's account is that the
   restarts followed a watchdog restart and a cleared billing lockout, that from 2026-10-26 to 2026-10-30 the local 3B
   rung answered most sends, and that darkflobi has answered on `anthropic/claude-opus-4-8` since. The account rests
   on no ledger line. Evidence from outside the ledger agrees with it at the run level. The transcripts on disk beside
   the nine the store names hold `local/qwen2.5-3b-instruct` stop lines on each day from 2026-10-26 to 2026-10-30 (17,
   43, 27, 29 and 21), each run of them answering after four gemini attempts that ended in an error, and
   `anthropic/claude-opus-4-8` stop lines from 2026-10-30 (11 that day) [R,
   `S27:review/substrate_lens/probe_store_out.txt` sections 1 and 1b: provider, model, api, stopReason and day,
   counted]. The newest assistant entry of agent `main`'s transcripts before each of the 44 E1 replies at seq 136-248
   is a `local/qwen2.5-3b-instruct` answer that followed the run's own four failed gemini attempts for the 20 replies
   of 2026-10-26 to 2026-10-29, and an `anthropic/claude-opus-4-8` entry for the 24 from 2026-10-30, each at most 60 s
   older than the reply [R,
   `S27:readings/fix1_2026_11_02/send_vs_transcript.txt`, counts only]. None of it is a ledger reading: the
   transcripts are files the agent can reach, and a reply is matched to agent `main`'s newest entry by time, not to its
   chat. The session store's own readings bear on the account too, and they change with the day they are read. On
   2026-10-30, of the nine transcripts agent `main`'s store names, seven held an assistant line: six ended on
   `gemini/gemini-3.1-pro-preview` and one on `local/qwen2.5-3b-instruct` [R, `RC:` `stage2.store_shape_2026_10_30`].
   On 2026-11-02 the same nine held no `local` line; they held 52 `gemini/gemini-3.1-pro-preview` assistant lines
   (17 `stop`, 35 `toolUse`) dated 2026-10-24 and 2026-10-25, the days of 57 of the 65 send lines at seq 1-135 (the
   other 8 on 2026-10-26), no assistant line dated 2026-10-26 to 2026-10-31, and no `anthropic` line before the 17
   `anthropic/claude-opus-4-8` lines dated 2026-11-02 [R, `RC:` `stage2.transcript_stops`; the days, `RC:`
   `ledger.all` and `ledger.from_136`]. So the store agrees with the account in part (gemini on the frozen days, a
   local answer seen on 2026-10-30) and keeps no lasting record of it: what it holds depends on when it is read.
3. **What the agent reads is not locked.** Outside `W:gate/` and the installed plugin, the lock rows hash the
   `plugins` subtree of the config, the X script and the posting skill's `SKILL.md`; the model chain, the
   memory-search settings, the bootstrap files the dist loads (`BOOTSTRAP.md` among them) and the rest of the skills
   tree are in no row [R, `RA:CRITIC.md` §3 and §6; the rows of the PREREG and the six files], so a change there moves
   nothing the RESULT reads.
4. **E1 and E3 carry no session key on this dist** [R, `S27:BUILD_LOG.md` stage 2, 2026-11-02 pass, section 4]: both
   callers of `deliverReplies` (`D:telegram/bot-message-dispatch.js` l.112, `D:telegram/bot-native-commands.js`
   l.254) pass no `sessionKey`, and an E3 tool post has none. A reading of the session store at those exits maps a
   destination to a session; only E2's `mirror.sessionKey` names the session.
5. **The lock tool of `310df453` could not read this move** [R, `RC:` `lock_check.old_tool_seven`]. Eight of this
   file's rows replace rows that an earlier amendment holds with another digest (`W:gate/gate.py`,
   `W:gate/gate_verify.py`, `W:gate/README.md`, `W:gate/plugin/index.js`, `W:gate/plugin/README.md`,
   `W:gate/plugin/tests/run_tests.mjs` and the installed `index.js` and `README.md`), so the check of `310df453`
   over the 2026-11-02 file and the six earlier ones exited 2 on eight CLASH lines by construction (D9 of
   `RA:AMENDMENT_substrate_DRAFT.md` §9). The tool of `f0125c8c` reads a declared supersession (item 7), and reads
   the 2026-11-02 file's declarations as a CLASH all the same: they named the count-rule and posting-skill rows for
   `gate.py` and `gate_verify.py`, which the token-shapes file had already replaced, and a 2026-11-02 name is read
   before the 2026-11-04 file whose rows it must supersede. Hence this file's date and its eight declarations.
6. **The config is not the run: on 2026-11-02 the two named different models for an hour and three quarters.** At
   19:14:35Z the operator's config was written again: its chain went from `anthropic/claude-opus-4-8` and seven
   fallbacks to `anthropic/claude-opus-5-5`, then `anthropic/claude-opus-4-8`, `anthropic/claude-sonnet-5-5` and
   `local/qwen2.5-3b-instruct`; the copy kept beside it in that second, `clawdbot.json.pre-stage1m-20261102T191435Z`,
   names the earlier chain [R, `S27:readings/fix3b_2026_11_02/config_vs_transcript.txt` section A, model ids only].
   The gateway read the write at 19:14:36Z and applied it as "dynamic reads", with no restart [R,
   `S27:readings/fix3b_2026_11_02/gateway_log.txt`, timestamps, listed words and config paths only]: the dist's
   reload rules make a change under `agents` or `models` kind none (`D:gateway/config-reload.js` l.30-31 and
   l.175-177), while the Telegram channel keeps the config object it read when it started
   (`D:gateway/server-channels.js` l.54 and l.96-97, `D:telegram/monitor.js` l.49) and hands that object to every
   reply it dispatches (`D:telegram/bot-message-dispatch.js` l.101-103, `D:auto-reply/reply/dispatch-from-config.js`
   l.221 and l.242, `D:auto-reply/reply/get-reply.js` l.22, where the object handed wins over a fresh read) [R,
   `S27:readings/fix3b_2026_11_02/dist_lines.txt`]. So the nine Telegram turns of the operator's session from
   19:25:58Z to 20:51:46Z ran on `anthropic/claude-opus-4-8`: the session transcript's `model-snapshot` row at
   19:25:58Z, the `clawdbot.cache-ttl` row written ahead of each turn and all 18 of the turns' assistant rows name it
   [R, `config_vs_transcript.txt` section B: provider, model, api, stopReason, roles and times only]. The gateway
   restarted (its log's lines from 20:59:58Z, listening again at 21:00:43Z, the Telegram provider started at
   21:00:44Z; the gate's `started` line, seq 263), and the session's turns from 21:14:44Z name
   `anthropic/claude-opus-5-5`. Inside the span the file's model ran too, on paths that read the file again
   (`D:config/io.js` l.450-459): a turn of the same session at 19:14:58Z, 23 s after the write, through a path the
   reading does not name, and 205 assistant entries of 46 other transcripts of agent `main` that no session store
   names, all on `anthropic/claude-opus-5-5` (section C). The live ledger holds four send lines in the span, all E1
   replies, seq 255, 257, 259 and 261 from 20:35:21Z to 20:46:17Z [R, section D: seq, kind, exit, payload kind and
   time only]. A field read from the config file would have named `anthropic/claude-opus-5-5` on each of them; the
   newest assistant entry of the operator's session before each names `anthropic/claude-opus-4-8` (joined by time:
   the reading names no chat). This is the case for reading the substrate from the run's own transcript at send time
   (item 2): its rows are written by the run that used the model, while neither the config file nor the config
   object one part of the gateway holds says which model a given run used.
7. **The gate moved under the 2026-11-02 build.** On 2026-11-04 at 13:15:12Z the lead moved the token-shapes
   amendment into `W:gate` (#247, merged 2026-11-08 as `f0125c8c`): `gate.py` `4416692c…` -> `c17e3d06…` (a numeral's
   `unbound_shapes`, never its digits), `gate_verify.py` `049d870b…` -> `5ab8454c…` (an old line's tokens compared by
   their shapes), and four test files (`tests/test_gate.py`, `test_gate_round2.py`, `test_trailer_rule.py`,
   `test_verify.py`); the stretch `f321b1bc…` opens at seq 411 (2026-11-04T14:18:13Z) [R, that amendment's "The
   apply" and addendum; `S8:readings/RECEIPTS_live_before.txt`, read 2026-11-08]. The 2026-11-02 build's rows for
   `gate.py` and `gate_verify.py` named the files that move replaced, and its two files were built on the gate at
   `4416692c…`. The live ledger read on 2026-11-08 holds 735 lines (seq 1-734; stretches by
   `gate_sha256`: `387fc063…` 13 lines (6 send) seq 1-13; `ff34b52d…` 112 lines (55 send) seq 14-125; `c5084df2…` 9 lines (4 send) seq 126-134; `445ac94c…` 276 lines (127 send) seq 135-410; `f321b1bc…` 324 lines (157 send) seq 411-734) [R, `RC8:` `ledger.stretches`]. Every line of it is a v0 line: no send line
   carries `gate_schema`, `substrate` or `subject_digest` [R, `RC8:` `ledger.keys`].

## The amendment

1. **A send line's schema.** From the stretch this file opens (item 8), every send line `gate.py check` writes and every
   line `gate.py fault` writes carries `gate_schema: "darkflobi.gate/send/v1"`, `substrate` and `subject_digest`. A send
   line without `gate_schema` is a v0 line, read under the v0 keys as before; every line written so far is one. The
   ledger's own `schema` (`darkflobi.gate/entry/v0`) and `ledger.py` do not change.
2. **`substrate`: the gateway's reading of the run that wrote the message, never the model's words.** An object with
   `provider`, `model` (the id as the gateway selected it), `api`, `model_served`, `chain_index`, `fallback_from`,
   `read_age_s`, `gateway` (`{name, version, commit}` of the dist) and `source`, or, for a reading that names no model,
   `{gateway, source: "unknown", error}`; validated by `gate.py` `substrate_of` (strings, integers, lists of strings or
   null, never a float; a closed key set; every string one token of `[A-Za-z0-9][A-Za-z0-9_.:@+/-]*`, at most 120
   characters, so no free text reaches a public line). Where the plugin reads it (`index.js` `readSubstrate`), in this
   order:
   - the nonce-matched anchor: `{gateway, source: "gate"}`, since `gate.py` composed that text and no model did;
   - a reading the patched exit passed as `event.substrate`: `source: "exit"` (no exit passes one: that needs two more
     patched dist files, not built);
   - the session the event names by its key (E2's `mirror.sessionKey`; E1's `params.sessionKey`, which this dist never
     passes: a keyed E1 reading needs a `patch_dist.js` edit that hands the run's session key to the exit, which moves
     `patch_dist.js` and `gate_sha256` and is not built): `source: "session_store"`;
   - the newest session whose `lastTo` or `deliveryContext.to` canonicalises to the destination and whose thread
     (`deliveryContext.threadId`, else `lastThreadId`) is the event's (a forum's topics are sessions of one chat): on E1
     `source: "session_store_by_chat"` (a reply goes back to the chat of the run's own session: the chat's newest
     session as persisted when the reply left, not a keyed reading of the run), on E2 and E3
     `source: "session_store_by_dest"` (a reading about some run that wrote to that chat, not necessarily the one that
     composed this text); either is refused when the entry it reads is older than **300 seconds**
     (`SUBSTRATE_DEST_MAX_AGE_S`), or carries no time: `{gateway, source: "unknown", error: "stale"}`. The bound rests
     on the live record: each of the 44 E1 replies at seq 136-248 left at most 60 s after agent `main`'s newest
     assistant entry [R, `S27:readings/fix1_2026_11_02/send_vs_transcript.txt`], while a destination mapping can reach
     an entry days old (the public group's sessions last wrote on 2026-10-25 [R,
     `S27:review/substrate_lens/live_reading_out.txt`]);
   - else `{gateway, source: "unknown"}`. The canary harness calls `gate.py check` with no reading, so its lines read
     `{"source": "unknown"}`.
   A session's reading is the newest attempt its transcript shows (`lastRun`), read from the tail: 512 KiB, doubled
   while the read holds no entry that names a model, up to 16 MiB (`SESSION_SCAN_MAX_BYTES`; on this box 57 of 69,932
   gaps between two assistant entries of agent `main` exceed 512 KiB, the largest 4.96 MB, none since May [R, `RC:`
   `rereview.semantics.tail_gaps`]). It is the newest of the transcript's assistant entries (a delivery-mirror line
   skipped) and of the rows the gateway writes ahead of an attempt, each naming the attempt's model as `data.provider`
   and `data.modelId`: a `model-snapshot` row whenever the model differs from the last one
   (`D:agents/pi-embedded-runner/google.js` `appendModelSnapshot`) and a `clawdbot.cache-ttl` row before each prompt
   to an anthropic model (`D:agents/pi-embedded-runner/run/attempt.js` `appendCacheTtlTimestamp`). When a row is the
   newest, or a user entry follows the newest assistant entry (an attempt on the same model: the gateway writes a
   snapshot row on every change), an attempt is open and the reading names its model, with no stopReason; else the
   newest assistant entry, with its stopReason and time. On this box, of 18,032 answered attempts of agent `main`, the
   newest such row since the previous assistant entry named the answering model for 17,888 and another model for none,
   and 144 had no row (the same model as before) [R, `RC:` `rereview.semantics.snapshot_rows`]. The store entry's own
   `modelProvider` and `model` are never read: the gateway writes them after a run, so they can name a model the run
   in progress does not use (harness S17, S19, S20), and a session whose transcript is missing, unreadable or names no
   model within the read gives no reading. Provider, model and api strings, stopReasons and times alone leave the
   transcript. A run whose newest entry ended in an error, with no attempt opened after it, names no model, since
   every model it tried failed and the text going out is the gateway's own notice ("Agent failed before reply") or
   error reply: `{gateway, source: "unknown", error: "run_failed"}`; an aborted run likewise (`run_aborted`).
   `fallback_from` is the run's own earlier attempts as the transcript shows them, never the configured chain:
   clawdbot's fallback prompts the session once per candidate it tries (its rows, a user entry, then the attempt's
   assistant entries) and a candidate skipped in cooldown writes nothing, so each assistant entry that ended in an
   error and is followed within 10 s by the next attempt's opening (its row or its user entry) is named, as
   `provider/model`, in the order tried (`RUN_ATTEMPT_GAP_MS`; the live transcripts read 642 of 678 such gaps at 1 s
   or less and none between 1 s and 60 s [R, `S27:readings/fix1_2026_11_02/transcript_structure.txt`]). `read_age_s`
   is the whole seconds from the entry read (the open attempt's row or user entry, else the assistant entry) to the
   send. On E1 the reading is the newest attempt the gateway had persisted for the chat's newest session when the
   reply left. A final reply leaves after its attempt's entry is written and a reply sent mid-run before it: of the 92
   E1 send lines the re-review set against the transcripts by the sha256 of each draft, 86 matched an entry (no text
   printed); the 77 matched to a `stop` entry left 0.2 to 43.3 s after it, and the 9 matched to a `toolUse` entry 0.2
   to 4.9 s before it [R, `RC:` `rereview.semantics.e1_order`]. The build before the re-review read assistant entries
   alone, so a mid-run reply carried the entry before its own: after a failed or aborted attempt it was written
   `run_failed` or `run_aborted` (this item's meaning of the gateway's own notice), and after a run on another model,
   a failover recovery being enough, it named that model [R, `RC:` `rereview.semantics.synth_midrun`]. Reading the
   rows, the re-review's five synthetic cases in the live failover shape each name the attempt that wrote the text,
   and the notice after a run whose every attempt failed still reads `run_failed` [R, `RC:` `fix4.synth_midrun`].
   The model is the transcript's, never a config's or the store entry's: on 2026-11-02
   the config file named one primary for an hour and three quarters while every Telegram run used another (item 6
   of what was observed; harness S17 and R9e: a config naming one model and a transcript another, the line naming
   the transcript's, by chat and by key). `chain_index` comes from the config object the gateway loaded at its
   start (the plugin's `api.config`, `D:gateway/server.impl.js` l.115 and l.125-126), never the file, and a hot
   reload of `agents` or `models` does not replace that object: the index is the model's position in the chain the
   gateway started with, which on 2026-11-02 was the chain the Telegram runs used and not the file's, and a model
   that chain lacks reads `null`. With a session's own model the primary is tried last, so neither field implies
   the other.
   `model_served` (the model string the provider's response carried) is read on no path: it is written null beside
   every provider, and a value handed for it is a shape error. `gate.py` checks the source against the rest of the
   line: `gate` only on the anchor it composed and `exit` only on an exit whose patch passes one (none on this dist),
   else `{"source": "unknown", "error": "source_line"}`; `session_store` only on a line that names a session (its
   `session_hash`), else written `session_store_by_chat` on E1 and `session_store_by_dest` elsewhere, and every E3
   reading `session_store_by_dest` (`SUBSTRATE_BY_DEST_EXITS = ("E3",)`); a destination-mapped reading without
   `read_age_s` or past the bound is `stale`; `unknown` carries no model. The error vocabulary is closed
   (`SUBSTRATE_ERRORS`: `shape`, `source_line`, `run_failed`, `run_aborted`, `stale`). A reading off its shape is
   written `{"source": "unknown", "error": "shape"}`, nothing of it kept, and the line still writes.
   The plugin never reads the message's text for it, and the text never sets it (harness S6). Said plainly:
   the field is the gateway's reading of a file the agent can reach (or, for the anchor, the gate's statement about
   its own text), not proof of which model wrote the words.
3. **`subject_digest`: what the agent reads, as on disk when the line is written.** `gate.py` computes it itself when
   it writes the line (the request cannot set it; a request's `subject_digest` is ignored): the sha256 of these
   LF-ended UTF-8 lines, in order (`subject_manifest`):
   1. `darkflobi.gate/subject/v1`;
   2. `workspace <path>`: the default agent's workspace as `D:agents/agent-scope.js` resolves it;
   3. seven `file <name> <sha256> | absent | unreadable` lines, the bootstrap files `D:agents/workspace.js` loads, in its
      order: `AGENTS.md`, `SOUL.md`, `TOOLS.md`, `IDENTITY.md`, `USER.md`, `HEARTBEAT.md`, `BOOTSTRAP.md`;
   4. three `config <subtree> <sha256> | absent | unreadable` lines over `~/.clawdbot/clawdbot.json`:
      `agents.defaults.model`, `agents.defaults.memorySearch` and `agents.list[default]` (the default agent's own
      entry, which can carry a model or workspace of its own), each by the PREREG's canonical-JSON rule over the
      subtree with the value of every key whose name matches `(?i)(key|token|secret|password|passwd|auth|credential|cookie|bearer)`
      replaced by `"<redacted>"` at every depth, inside lists too (any case, anywhere in the name), so a value held
      under a secret-named key never enters an unsalted hash and its rotation does not move the digest (a secret held
      under another name does enter it: below); text is hashed as UTF-8, never `\u`-escaped;
   5. `tree skills <digest> <count> | absent | unreadable | too_large`: `<workspace>/skills` by `lock_hashes.py`'s tree
      form (its rows sorted by the whole relative path, so `x-y/…` and `x.md` come before `x/…`, whatever order the
      walk met them in), over every regular file the gateway's skill loader reaches (dot-entries and `node_modules` not
      entered, names opening with `_` read like any other, links and junctions followed), `__pycache__` and `.pyc`
      left out.
   Left out by definition: `MEMORY.md` and `memory/*.md` (reached through `memory_search`), the managed, bundled, extra
   and plugin skill folders, the rest of `clawdbot.json` (`agents.defaults.contextPruning` and `compaction` among
   them, though they change what the agent reads), the dist, and every agent but the default one. The redaction goes
   by key name: a secret held under an ordinary name (a token inside a URL) is hashed unsalted, and a setting under a
   secret-named key does not move the digest. The skills tree is every file in it, a runtime log among them (one
   `.log` today [R, `S27:review/substrate_lens/probe_subject_out.txt`]), so a skill that writes its own files moves
   the digest. The plugin
   hands `gate.py`'s process the two inputs that move the reading (`STYXX_GATE_SUBJECT_CONFIG` when the gateway's
   environment moves the config, and `CLAWDBOT_PROFILE`) and nothing else (`subjectEnvFrom`).
4. **No tag in the trailer, and the token-shapes rule untouched.** `RULES` §7's trailer (`[gate: Nb of Nn numerals
   bound · ledger <seq>]`, l.321) is written byte for byte as before on every line, the count rule's form is the
   count-rule amendment's, and a numeral is stored as its shape as the token-shapes amendment has it: the source text
   of `TRAILER`, `TRAILER_FORM_RX`, `TRAILER_QUOTE_SEQS_MAX`, `_SEQ_DIGITS_MAX`, `UNCHECKED_MARK`,
   `_trailer_of_line`, `trailer_quotes`, `bind_numerals`, `decide`, `_trailer_template`, `_append_marks`,
   `_send_lines_by_seq`, `token_shape`, `_DIGIT_RUN_RX` and `_zero_numerals` in the new `gate.py` is the live
   `gate.py`'s (`c17e3d06…`, frozen as `tests/fixtures/gate_v0_c17e3d06/gate.py`, item 10) character for character
   (`test_the_trailer_and_count_rule_code_is_the_live_gates_byte_for_byte`, by `ast` source segments), and
   `reconcile.py` does not move. A v1 line's `numerals` carry `unbound_shapes` and never `unbound_tokens`, as a line of
   the `f321b1bc…` stretch does; the string `unbound_tokens` occurs nowhere in the new `gate.py`. The prototype's
   tag (`via <provider/model>`) is withdrawn with its count rule (`RA:CRITIC.md` verdict, item 1: "D2: no tag; the
   count rule's form untouched").
5. **The replay and the report.** `gate_verify.py --rederive` replays each send line under the schema it names,
   compares `gate_schema`, `substrate` and `subject_digest` whenever either side carries one, beside the keys the
   token-shapes amendment compares (a line carrying `unbound_shapes` on them, a line carrying `unbound_tokens` by the
   shapes of its stored tokens, a line carrying neither on the counts alone), and prints `send lines by schema: v0 N,
   v1 M, other K`. A line whose legs did not finish then (`gate.py fault`'s, or check's own
   `gate_unavailable` line, read `UNAVAILABLE_THEN`) has its three keys re-derived the same way beside its decision,
   action, finding and `would_refuse` (`_v1_unavailable`; its anchor is a fault line of payload kind `anchor`); so
   has an `internal` line, logged and never gated, beside its decision, action, finding and digests.
   The validator's output, the E3 label, the source's check against the line (its exit,
   its anchor and whether its `session_hash` names a session), the digest's shape and the schema re-derive.
   `substrate` and `subject_digest` themselves are inputs to the replay, re-derived from nothing it can read: a
   rewritten value of valid shape re-derives SAME. Only the chain and the anchors name an edit made after a line was
   written (an edited line breaks the chain, and a chain written again reads against the heads the anchors already
   posted, `RULES` §9), and nothing names a reading set before the line was written (item 2). `page.py` gains a
   `substrate` and a `subject` column (`unrecorded` on v0 lines).
   `substrate_split.py` (new) prints every count of §10 split by substrate key and by `subject_digest`, `unrecorded`
   for v0 lines, with the sentence "every line predates the field" when no v1 line is in the window; no bar.
6. **The plugin.** `PLUGIN_VERSION` and the manifest go to `0.5.0`; the manifest's `configSchema` gains `sessionsDir`
   (default `<CLAWDBOT_STATE_DIR or ~/.clawdbot>/agents`, the gateway's own); `stub_gate.py` mirrors the three keys;
   the harness gains sections S and R9 (111 tests). F-1's code (`loadNativeDist`, `runnerState`, `guardedDeliver`,
   the anchor's timer and state) is byte-identical in the new `index.js` [R, `RC:` `stage2.f1_untouched`], and
   `native_import.cjs` and `package.json` do not move.
7. **The ordered supersession of lock rows (D9).** `lock_hashes.py check --amendment` reads the amendment files in a
   stated order, whatever order they are given in: by the date the file's name ends with (`_YYYY_MM_DD.md`), then by
   name; each file read prints its `AMENDMENT` line in that order. A later file's row for a path that an earlier
   amendment file holds with another digest replaces it only when the later file declares the supersession of each
   earlier row in force, one line per earlier file, in a five-cell form that `ROW_RE` never reads as a lock row:

   ```text
   | supersedes | `<path>` | `<earlier AMENDMENT file name>` | `<the earlier row's digest>` | note |
   ```

   Each declared supersession prints `SUPERSEDED <path> <earlier file> <digest> -> <later file> <digest>` before any
   row is read, and the row prints `via=<later file>`. Everything else stays a CLASH and exits 2 before any row is
   read: an undeclared later digest (the behaviour of `310df453`); one file naming a path with two digests; a
   declaration that names another digest, a file not read or read later, a row it does not replace (the same digest
   restated, or no row of its own for the path), or is given twice; a line opening `| supersedes |` not in the form;
   a supersession where either name carries no date; one file name given twice. An amendment's row still replaces
   the PREREG's without a declaration, as before. The lock-rows amendment's sentence "the order of the files cannot
   change a reading" holds for files with no declaration; with one, the stated order decides. The tool is on `main`
   since #247 (blob `4d3aea1d0674e75563c2e7659e05f945b8f289a5`, the substrate branch's `be538fa4` bytes; before it
   `fa59627f1db1cdc259a3bef444db644b2a474c11`, #244's); this file moves no byte of it. What this file declares: a
   later file declares the row in force, not one already superseded, so for `gate.py` and `gate_verify.py` it names
   the token-shapes file's rows (`c17e3d06…`, `5ab8454c…`), never the count-rule and posting-skill rows the
   token-shapes file replaced, and it is dated after that file, since the files are read by date and a declaration
   naming a file read later is a CLASH. Tests (`tests/test_darkflobi_gate_prereg.py`):
   `test_a_later_amendment_supersedes_an_earlier_row_only_when_it_declares_it` (with the cases the substrate branch's
   test held and main's did not: an undated earlier file, a declaration naming a pin row or the not-built note, the
   `AMENDMENT` line hashing LF bytes), `test_the_folder_reads_in_date_order_and_each_supersession_is_declared` (both
   later files, `gate.py`'s chain of three links), and the folder's counts in
   `test_each_amendment_parses_to_the_lock_rows_its_table_states` (rows 102 with the PREREG's).
8. **The move, 26 files, and the seventh `gate_sha256` stretch.** Each file is a row below. Seven enter `gate_sha256`:
   `W:gate/substrate_split.py` (new), `page.py`, `gate_verify.py`, `gate.py`, and the installed `README.md`,
   `clawdbot.plugin.json` and `index.js`; the installed folder keeps its seven files. The stretches so far are
   `387fc063…` (§11), `592dc579…` (15.6, no line), `ff34b52d…` (15.10), `c5084df2…` (the lock-rows amendment, from seq
   126), `445ac94c…` (the anchor-native amendment, from seq 135) and `f321b1bc…` (the token-shapes amendment, from seq
   411); this move begins the seventh. Its candidate, by 15.6's rule without importing `gate.py`
   (`BD:apply/rederive_gate_sha256.py` over the copy and the installed folder as it would stand), is the last line
   below, over 21 files. The lines between the two ends are the six intermediates a line written mid-move could
   carry, after each rename that enters the digest, in the move's order [R, `S8:substrate_move_table.json`
   `gate_sha256_stages`, `RC8:` `move.table`]:

   ```text
   before the move                           f321b1bcde60a62e7be990883d82243e5c5cd2a41fef036381e0a8b24c4a01ff  20 files
   after W:gate/substrate_split.py           968a9023cc9c06730682f461a2bdbef76105368b08b1fb815d5606d99a3d54bf  21 files
   after W:gate/page.py                      eeb5ba442adffcf5ebb9440fe6cb0143c732b25c85249b0b1acdb7ef87cf68b2  21 files
   after W:gate/gate_verify.py               5d0a39fc39c8282b37b7d5eba90f3f9ae4425c3ac95c1d039f58760916bfe8b0  21 files
   after W:gate/gate.py                      4f786b7e5cf6e078da662bb4c9357940e3ab61b0e5a8522487055de44da08d45  21 files
   after the installed README.md             3d269cea160ffcdd8821fd1a4d5bc163e02265716bace7a0d2e5b3d643144798  21 files
   after the installed clawdbot.plugin.json  3c9b72f10703fbde475d14f215bf270a6bb4fc40df2e24d7cd245b2152ab88ce  21 files
   after the installed index.js              17de31a1962650191b93383692156928eefd5318f1c468ab4524a3bfc26af2bb  21 files
   ```

   The digest of record is the one the lead re-derives from the box after the move, recorded in the addendum. The
   move is `S8:apply_substrate_move.py` (the 2026-11-02 script, `S27:apply_substrate_move.py` `e99c4c68…`, with its
   texts re-targeted to 26 rows and six folders and no step moved [R, `S8:tools/retarget_move_scripts.py`]), adapted
   from `BD:f1_anchor/apply_f1_move.py` pass 3 with its rule and
   order: 15.6's rename; the replaced files snapshotted; every file staged outside what `gate_sha256` reads and
   re-hashed before any rename; refusals while the gateway answers on 127.0.0.1:18789, while `REQUIRED` exists,
   between 02:30 and 05:20 UTC (the anchor window, 03:30-05:20, and the hour before it, which a failed move keeps for
   its rollback; checked before staging and again before the renames), when a folder the move creates already
   exists, when the installed folder holds any file but the table's seven, or when the box's `gate_sha256`, read by
   15.6's rule inside the script, is not the `before the move` line above over its 20 file names (so an extra or
   missing `*.py` under `W:gate/` or `W:gate/helpers/` is refused before anything is written); the six new fixture
   folders each recorded, then created, before the renames; the installed `index.js` renamed last; after the renames
   the box's `gate_sha256` read again, which must be the candidate over its 21 file names, else exit 3 for the
   rollback. The rollback (refused between 03:30 and 05:20 UTC) reads every row before it writes any, restores the
   installed `index.js` before anything else, removes the folders the move created and ends on the box's
   `gate_sha256` read again, which must be the `before the move` line above. A move or a rollback killed at any step
   is finished by the rollback on its snapshot: it removes what the killed process left in the two staging folders
   (only the names the script writes there) and a `__pycache__` holding nothing but `.pyc` files inside a folder the
   move created; anything else it finds there it names and leaves, exiting 3, for the lead to read and remove [R,
   `RC:` `rehearsal`].
   The order on the box: pause the watchdog, stop the gateway, move, start the gateway, resume the watchdog; the
   restarted gateway's `started` line opens the stretch, and no 0.4.0 plugin code runs on the new `gate.py`.
9. **`RULES` §13, restated for the seventh stretch.** "A `send` line" gains `gate_schema`, `substrate` and
   `subject_digest`; from the stretch's opening line the fields decide, as `RULES` l.594-595 has it for `agent`
   ("from then the field decides"). A send line before it reads `unrecorded` for all three: that half is this file's
   own rule, not `RULES`', since `agent` is read before its field by the folder attribution (l.593-594) and no
   attribution stands in for these three [J].
10. **The shapes gate frozen beside the tests.** `tests/fixtures/gate_v0_c17e3d06/` holds the live gate as the
   token-shapes move left it (`gate.py` `c17e3d06…`, `gate_verify.py` `5ab8454c…`, with the live `ledger.py` and
   `helpers/numerals.py` beside them), as `gate_v0_25d53fd6/` holds the gate before the count rule and
   `gate_v0_4416692c/` the count-rule gate: the three generations of v0 writer whose lines the live ledger holds (no
   shape key, `unbound_tokens`, `unbound_shapes`). `tests/test_substrate.py` writes lines with each of the three, then
   v1 lines, into one home and reads every line SAME under the new `gate_verify.py`
   (`test_four_generations_all_rederive_same`), and holds the trailer, count-rule and shape code of the new `gate.py`
   to this fixture's (item 4). The replay itself runs every line under the current code and compares each on the
   keys it carries; the fixtures serve the tests, not the replay [J, `S8:BUILD_LOG_v2.md` decision D2-a].

## The readings

The 2026-11-02 build's readings, on `S27:gate_copy/` (the gate at `4416692c…`), bound in `RC:`, stand as the record
of that build; the re-base's readings, on `S8:gate_copy/` (the gate at `c17e3d06…`), bound in `RC8:`, follow them.

[R, `RC:` `stage1`] The gate side, on the copy: the gate's suite 301 passed, 12 subtests passed, rc 0, twice, the
second run on the final bytes (`gate.py` `cce267f8…`); 301 is the live suite's 279 and `tests/test_substrate.py`'s 22.
The live ledger frozen at seq 135 (65 send lines) re-derived under the copy's `gate_verify.py` and under the live pair
(`gate.py` `4416692c…`, `gate_verify.py` `049d870b…`): SAME 64, DRIFT 1 (seq 46, the count rule's own), the status and
the differing fields identical on 65 of 65 lines, `send lines by schema: v0 65, v1 0, other 0`. No old line moves.

[R, `RC:` `stage2`] The plugin side, on the copy: the harness 100 of 100, rc 0 (run 5 on the final bytes, run 6 after
the mutation battery); `tests/test_substrate.py` 22 passed; 26 mutants of `index.js`, `gate.py` and `gate_verify.py`,
each a single replacement, 26 killed and 26 restored with their bytes re-hashed, among them an E3 reading labelled as
E1's, the anchor read from the store, a reading taken from the model's words, `subject_digest` without `BOOTSTRAP.md`,
a v1 line verified as v0, and the trailer gaining a tag.

[R, `S27:readings/fix1_2026_11_02/`] The fix round of 2026-11-02 (review 1 of the build, `S27:REVIEW_2026_11_02.md`
§1), on the copy's final bytes (`gate.py` `b0b47381…`, `gate_verify.py` `4baac32f…`, `index.js` `d47bb53c…`,
`run_tests.mjs` `e42b6bde…`, `tests/test_substrate.py` `29374f88…`): the harness 104 of 104, rc 0 (S13-S15 and R9d
new); `tests/test_substrate.py` 30 passed, rc 0; the gate's suite 309 passed, 12 subtests passed, rc 0 (the live
suite's 279 and test_substrate's 30). The new tests on the code as it stood before the round (`gate.py` `cce267f8…`,
`index.js` `691670c8…`): the harness 89 of 104, rc 1, S13, S14, S15 and R9d among the 15 that fail, and
`tests/test_substrate.py` 18 failed of 30, rc 1. Each fix reverted alone in a scratch tree of the final code, one
replacement at a time, the file restored and re-hashed after each: 15 of 15 fail, each on the tests written for its
fix: the run's stopReason (S13, R9d), the error vocabulary (four of the suite's tests, S9, R9d), the plugin's age
bound (S14) and gate.py's (two), the E1 label (S1, S2, S6, S12, S14, S15), the thread (S14), the relabel of a reading
on a line that names no session (four), `fallback_from` taken from the chain (S15 among ten) and the run boundary
ignored (S15), and `gate` off the anchor, `exit` on every exit, a model on `unknown`, free-text strings, the fault
line's anchor and the replay's session (the suite's tests for each, and R9d for `gate` off the anchor).

[R, `S27:readings/fix2_2026_11_02/`, `S27:mutants_round3/`] The fix round's tests and mutants (review 3 of the build,
`S27:REVIEW_2026_11_02.md` §3), on the copy's final bytes (`gate_verify.py` `382949d1…`, `tests/test_substrate.py`
`098c1e26…`, `run_tests.mjs` `6be32e18…`; `gate.py` `b0b47381…` and `index.js` `d47bb53c…` as above): the harness 105
of 105, rc 0 (S16 new; section R's subject fixture gains sibling names that the walk's order and the path's order put
apart, names opening with `_`, an Authorization header inside a list and a setting of non-ASCII text, so the harness's
own implementation of the definition and `gate.py`'s agree only when both sort, redact and encode alike);
`tests/test_substrate.py` 39 passed; the gate's suite 318 passed, 12 subtests passed, rc 0. The new tests on the
`gate_verify.py` of the round's earlier part: 2 of 39 fail, the fault-line replay's two. Thirty-one mutants, each one
replacement: the eleven survivors review 3 named (re-stated where the round's earlier part moved their text) and
twenty more of the round's code. On the tests as the round's earlier part left them, 9 of 27 are killed, and the 18
survivors pass the harness and the whole suite applied together (the lock tool's two, its own tests); on this round's
tests 30 of 31 are killed, each of those 18 but one by a test this round wrote. The one left (a `session_store`
reading relabelled only when it names a provider) changes nothing: two lines before, the validator refuses a store
reading that names no provider.

[R, `S27:readings/fix2_2026_11_02/live_subject_2026_11_02.txt`] The live subject read again on 2026-11-02 at 17:40Z by
the copy's `gate.py` (the subject code byte for byte the code of the reading below):
`510b6a8846f2391635ada022aba4c4f977c081722019738693b1a00e76878207`. One row moved, `agents.defaults.memorySearch`; the
bootstrap files, the other two config rows and the skills tree (106 files) read as below. The config was written at
17:35:37Z (8874 to 9085 bytes): its `memorySearch` now names provider `openai` and holds four keys more (`model`,
`remote.baseUrl`, `sources`, `query.hybrid.enabled`), read by key names only, no value credential-shaped [R,
`S27:readings/fix2_2026_11_02/memsearch_keys_2026_11_02.txt`]. Rows left in the walk's order would give that tree
another digest. Read again at 19:19Z: `9e214afae96fc13e7ca1ebdf57cbd9858312691479c7c2a310efc1b702ce5e5d`, the
`agents.defaults.model` row moved as well: the config was written again at 19:14:35Z (11190 bytes), its chain now
`anthropic/claude-opus-5-5`, then `anthropic/claude-opus-4-8`, `anthropic/claude-sonnet-5-5` and
`local/qwen2.5-3b-instruct` [R, `S27:readings/fix2_2026_11_02/live_subject_2026_11_02_b.txt`,
`memsearch_keys_2026_11_02_b.txt`]. The subject this file's stretch opens with is the one on disk at the move.

[R, `RC:` `subject`] The live subject, read on 2026-11-02 by the copy's `gate.py` (hashes only, the config through the
redaction above): `624924c31f5806dc708a5b788dac83edaa749c357126d102825ce80594b80e93`, the reading of 2026-10-26; all
seven bootstrap files present, the skills tree 106 files, about 21 ms.

[R, `S27:readings/fix3_2026_11_02/`] The fix round's move and text (review 2 of the build, `S27:REVIEW_2026_11_02.md`
§2): the move table made again from the copy's bytes as they then stood (the 22 rows, the stages of item 8, the
candidate then `e9f14955…` over 21 files, which the second pass below moved) and `S27:plugin_as_installed/` rebuilt
from the copy's `plugin/` [R, `RC:` `move.table`]; the move script's fixes for review 2's 2.2, 2.3 and 2.6, rehearsed
as the paragraph after next says; the copy's leftover cache (`plugin/tests/__pycache__/`, a 2026-10-30 probe's)
removed by a script that refused any other bytes, so the strict check of the copy as the box would stand reads
bytecode 0. No file the move carries changed in that part of the round: its 22 rows were the bytes fix rounds 1 and 2
left. The box was read only: stage 1's 46 rows, the installed folder's seven files and `REQUIRED` absent re-hash as
before at the part's start and end, and the config's mtime and size are those of 19:14:35Z, so the 19:19Z reading of
the subject's config rows above still stands [R, `RC:` `preflight`].

[R, `S27:readings/fix3b_2026_11_02/`] The fix round's second pass (the model change of 2026-11-02, item 6 of what was
observed): the harness on the copy's final bytes (`run_tests.mjs` `7c03124c…` and the plugin's `README.md`
`ef8472c5…`, each changed by a patch script that refuses other bytes; every other file as fix rounds 1 and 2 left it)
reads 107 of 107, rc 0, S17 and R9e new: a config naming one model and the transcript another, the line naming the
transcript's, by chat and by key through the stub, and through the real `gate.py` beside the subject digest of the
file as it stands [R, `RC:` `fix3b.final_harness`]. Both pass on the code as it stood, since the plugin already read
the model from the transcript: they pin that. Four mutants of `index.js`, each one replacement in a scratch tree, the
file restored and re-hashed after each, take the model or the chain from somewhere other than the run's transcript:
the model from the loaded config object's primary (11 tests fail), the model from the config file's primary (4), the
chain from the config file (S17 and R9e alone), the store entry's own model fields ahead of the transcript (9); 4 of 4
fail, each on S17 or R9e [R, `RC:` `fix3b.mutants`]. `tests/test_substrate.py` 39 passed (no file of the gate side
moved) [R, `RC:` `fix3b.pytest_substrate`]. `S27:plugin_as_installed/` rebuilt and the move table made again: the rows
of `run_tests.mjs`, the plugin's `README.md` and the installed `README.md` take the new bytes, the last of them
entering `gate_sha256`, so the last three stages of item 8 and the candidate move, now `3a7e8915…` over 21 files [R,
`RC:` `move.table`]; the rehearsal run again over that table, 22 of 22 as expected, its move reading the candidate by
the script's own rule and by `BD:apply/rederive_gate_sha256.py` [R, `RC:` `rehearsal`]. Item 6 of what was observed
rests on three readings of this pass, each printing only what it names (model ids from the config files; provider,
model, api, stopReason, roles and times from the session files; seq, kind, exit, payload kind and time from the
ledger; timestamps, listed words and config paths from the gateway's log; the dist's own lines) [R, `RC:`
`fix3b.config_vs_transcript`, `RC:` `fix3b.gateway_log`, `RC:` `fix3b.dist_lines`]. The box was read only: stage 1's
46 rows, the installed folder's seven files and `REQUIRED` absent re-hash as before at the pass's start and end [R,
`RC:` `preflight`].

[R, `S27:readings/fix4_2026_11_02/`] The re-reviews of the fixed build, answered (`S27:REREVIEW_ANSWERED.md`): the
re-review of semantics found that a reply leaving while an attempt is open, before that attempt's own entry is written
(9 of the 28 matched E1 replies from seq 190 on), read the entry before it: `run_failed` or `run_aborted` after a
failed or aborted attempt, another run's model after a recovery; it also found the tail read falling back to the store
entry's own fields, public E3 posts reading no model, and the [J] of item 2 readable. The re-review of tests found an
`internal` line's three keys compared by no replay, and twelve fresh mutants of which one was killed. On the copy
(`plugin/index.js` `c472d35b…`, `plugin/tests/run_tests.mjs` `3021b8ce…`, the plugin's `README.md` `515ab11d…`,
`gate_verify.py` `898a6091…`, `README.md` `7361a1de…` and `tests/test_substrate.py` `2785d90f…`, each changed by a
patch script that refuses other bytes, every script replayed over fresh copies of its base [R, `RC:`
`fix4.patch_replay`]; `gate.py` `b0b47381…` unchanged, so the trailer and count-rule code stay the live gate's): the
harness 111 of 111, rc 0, S18-S20 and R9f new [R, `RC:` `fix4.final_harness`]; `tests/test_substrate.py` 44 passed [R,
`RC:` `fix4.final_pytest_substrate`]; the gate's suite 323 passed, 12 subtests passed, rc 0 [R, `RC:`
`fix4.final_pytest_full`]; F-1's functions byte-identical once [R, `RC:` `fix4.f1_untouched`]. The new tests on the
code as the round found it: the harness 106 of 111, rc 1, S9, S18, S19, S20 and R9f failing, and
`tests/test_substrate.py` 1 failed of 44, the internal line's replay; the other new gate-side tests pin what the code
already did, and the mutants show what each catches [R, `RC:` `fix4.before_harness`, `RC:`
`fix4.before_pytest_substrate`]. Twenty-one mutants, each a set of exact replacements in a scratch tree, the files
restored and re-hashed after each: the re-review's twelve (its RR07 made again as RR07b, the store entry's fields read
again, since this round removed the line RR07 changed) and nine reversals of this round's fix (rows not read, no
attempt opened after an entry, the tail not extended, the stopReason kept while an attempt is open, no api from an
earlier entry, the open attempt aged from the entry before it, rows ending the walk of attempts, an internal line's
keys not compared, the read unbounded): 21 of 21 killed, each by a test written for it [R, `RC:` `fix4.mutants`]. The
re-review's synthetic probe, copied with its five cases and run on the fixed copy: (a) `anthropic/claude-opus-4-8`
with `fallback_from` the failed `anthropic/claude-opus-5-5`, (b), (c) and (e) `anthropic/claude-opus-5-5`, (d)
`run_failed` [R, `RC:` `fix4.synth_midrun`]. The re-review's live replay, its three probes copied and its replicas
built again with each row's model ids (pseudonymised, no text, removed after the run): of the last 20 send lines, 17
read the model of the entry that holds their text at both bounds of the read time, seq 259 among them; seq 219 and 220
name `anthropic/claude-opus-4-8`, their text in no entry; seq 231, a public E3 post whose text only a mirror entry
holds, reads `unknown`; none reads another model [R, `RC:` `fix4.judge_replay`]. The move table made again (the rows
of `index.js`, `run_tests.mjs`, the plugin's `README.md`, `gate_verify.py`, `README.md`, `tests/test_substrate.py` and
the installed `README.md` and `index.js` take the new bytes), its candidate `8006c50b…` over 21 files, re-derived by
`BD:apply/rederive_gate_sha256.py` over the copy and the installed folder as it would stand [R, `RC:` `move.table`,
`RC:` `fix4.rederive_candidate`]; the rehearsal run again over it, 22 of 22 as expected [R, `RC:` `rehearsal`]. The
box was read only: stage 1's 46 rows, the installed folder's seven files and `REQUIRED` absent re-hash as before at
the round's start and end [R, `RC:` `preflight`].

[R, `RC:` `lock_check`] The tests of item 7 and the folder's counts pass (`tests/test_darkflobi_gate_prereg.py` 33
passed, 7 skipped, the seven tests that need `STYXX_GATE_W_ROOT`, inside a run of the gate's own test files and every
test file that reads `CHANGELOG.md`, 301 passed, 8 skipped, rc 0 [R, `RC:` `repo_tests.default`]; with
`STYXX_GATE_W_ROOT` naming a root whose `gate/` is the copy and a home holding the installed folder as it would stand,
it and the posting-skill tests read 50 passed, none skipped [R, `RC:` `repo_tests.box_named_on_copy`]; ten mutants of
the supersession code, each one replacement, are 10 killed [R, `RC:` `lock_check.mutants`], and the fix round's two
mutants of the order (a date needed in the later name only; an undated file read after the dated) are killed by its
case of an undated earlier file [R, `RC:` `fix2.mutants_table`]). The strict check against the live box, read only,
before the move, with every amendment file this one included: rows 98, ten `SUPERSEDED` lines, changed 13, built since
0, absent 10, unreadable 0, unlocked 0, bytecode 0, rc 1: the twelve rows that replace a file read CHANGED and the ten
new files ABSENT, as they must before the move, and the snapshot's tree row CHANGED for the bytecode file named below
[R, `RC:` `lock_check.live_all`]. The same against the copy as the box would stand after the move: rows 98, ten
`SUPERSEDED` lines, changed 0, unlocked 0, bytecode 0, and absent 4 for what the copy lacks (the salt, the config,
`anchor_dest.txt`, the X script) [R, `RC:` `lock_check.copy_as_after_move`]. With the six earlier files and not this
one: rows 88, changed 1, every other count 0, rc 1. The one CHANGED row is `W:gate/snapshot_3437340b/`: its tree reads
6,165 files, `99fedf09…`, where the PREREG's row locks 6,164, `681020dc…`. The file more is
`scripts/__pycache__/surface_lint.cpython-312.pyc` (35,210 bytes, written 2026-11-02T20:16:43Z); the tree without it
reads `681020dc…` again [R, `RC:` `lock_check.live_six`, `RC:` `fix3b.live_snapshot_bytecode`]. No run of the second
pass or of the re-review answers pointed at the live snapshot (each kept bytecode off and read the copy's), and these
readings do not name the writer; the re-review answers read the same count again [R, `RC:` `lock_check.live_six`]. The
tree form hashes every regular file, so bytecode there reads as a changed snapshot and not as `BYTECODE`. The same
check at fixer 3's reading, 19:52Z, read every count 0, rc 0 [R, `RC:` `lock_check.fix3_readings.live_six`]: once that
folder is removed the box is the box the six files lock, which the lead's step below reads before the move. The seven
files with this file's declarations taken out: exit 2, eight CLASH lines, no row read [R, `RC:`
`lock_check.bare_declarations`]. This file beside the PREREG alone, without the six earlier files: exit 2, ten CLASH
lines, each a declaration superseding a row that no file read holds [R, `RC:` `lock_check.this_file_alone`]. The tool
of `310df453`, the main branch's until the merge: over the seven files, exit 2 on eight CLASH lines [R, `RC:`
`lock_check.old_tool_seven`]; over the six earlier files against the copy as the box would stand after the move,
changed 12 and unlocked 10 (absent 4, as above), rc 1 [R, `RC:` `lock_check.old_tool_six_after_move`]. The PREREG
alone against the box: rows 80, changed 12 (the anchor-native file's addendum read 11; the snapshot's row is the
twelfth), built since 1, unlocked 8, rc 1 [R, `RC:` `lock_check.live_prereg_alone`].

[R, `RC8:` `baseline`, `RC8:` `after`] The re-base of 2026-11-08 (`S8:BUILD_LOG_v2.md`): the gate's suite on the
untouched fresh copy of the live gate (`gate.py` `c17e3d06…`, `gate_verify.py` `5ab8454c…`, the pinned snapshot and
the wheel through junctions, a dummy salt), before any change: 285 passed, 12 subtests passed in 865.72s (0:14:25), rc 0 (the token-shapes
amendment's own count after its move); on the copy with the substrate re-applied (`gate.py` `e1960c37…`,
`gate_verify.py` `8ffa0295…`, `tests/test_substrate.py` `d7cd3cac…`): 329 passed, 12 subtests passed in 1031.84s (0:17:11), rc 0: the live suite's 285 and
`tests/test_substrate.py`'s 44. `tests/test_substrate.py` alone on the same bytes (the dev tree): 44 passed in 155.43s (0:02:35), rc 0,
the byte-for-byte test against the shapes fixture and the four-generation replay among them [R, `RC8:`
`dev.pytest_substrate`]. The plugin harness on the re-based copy: 111 of 111, rc 0, 132.6 s (section R ran the merged `gate.py`;
D8 and F11-F15 PASS) [R, `RC8:` `harness`], and on the dev tree 111 of 111, rc 0, 129.9 s [R, `RC8:` `dev.harness`]. No
gate-side test of the 2026-11-02 build changed its assertion for the re-base except where the token-shapes gate
is the frozen live gate (item 4) and the replay test writes the shapes generation (item 10).

[R, `RC8:` `rederive.ledger_copy`] A copy of the live ledger (`S8:ledger_copy.jsonl`, 735 lines, seq 1-734, 349 send
lines, 5 stretches by `gate_sha256`: `387fc063…` 13 lines (6 send) seq 1-13; `ff34b52d…` 112 lines (55 send) seq 14-125; `c5084df2…` 9 lines (4 send) seq 126-134; `445ac94c…` 276 lines (127 send) seq 135-410; `f321b1bc…` 324 lines (157 send) seq 411-734; no send line carrying `gate_schema`, `substrate` or
`subject_digest` [R, `RC8:` `ledger.stretches`, `RC8:` `ledger.keys`]) re-derived under the re-based
`gate_verify.py` and `gate.py`, with `--sources W:gate/sources` read only, `--snapshot-root` the copy, `--repo
W:styxx`, `STYXX_GATE_VERIFY_HOME` under `S8:`: `gate_verify: 734 lines, head 0e5a698dece0, chain re-derives`; anchor lines 14, problems 0, lines after their day's anchor 0; `send lines by schema: v0 349, v1 0, other 0`;
`re-derived SAME 348, SAME_UNDER_SKEW 0, UNAVAILABLE_THEN 0, DRIFT 1, SKEW 0, UNRESOLVED 0, TAMPER 0`, rc 1 from the DRIFT. The one line not SAME is seq 46 (DRIFT on `numerals.total`, `numerals.unbound`, `numerals.unbound_tokens->shapes`, `sent_sha256`): the count rule's standing reading,
which the token-shapes amendment reads the same way (`numerals.unbound_tokens->shapes` through the shapes compare).
Every line of every stretch, the `f321b1bc…` lines with `unbound_shapes` and the earlier lines with `unbound_tokens`
among them, reads as under the live pair. No old line moves.

[R, `RC8:` `lock_check.live_seven`, `RC8:` `lock_check.live_eight`] The strict check against the live box, read only,
before the move: with the seven files on `main`, rows 88, seven `SUPERSEDED` lines (the token-shapes file's), every
count 0, rc 0: the box is the box the seven files lock (the snapshot's stray bytecode file the 2026-11-02 readings
named is gone). With this file as the eighth: rows 102, fifteen `SUPERSEDED` lines (the token-shapes file's seven and
this file's eight), changed 12, absent 14, built since 0, unreadable 0, unlocked 0, bytecode 0, rc 1: the twelve
rows that replace a file read CHANGED (`W:gate/README.md`, `W:gate/gate.py`, `W:gate/gate_verify.py`, `W:gate/page.py`, `W:gate/plugin/README.md`, `W:gate/plugin/clawdbot.plugin.json`, `W:gate/plugin/index.js`, `W:gate/plugin/tests/run_tests.mjs`, `W:gate/plugin/tests/stub_gate.py`, `~/.clawdbot/extensions/styxx-gate/README.md`, `~/.clawdbot/extensions/styxx-gate/clawdbot.plugin.json`, `~/.clawdbot/extensions/styxx-gate/index.js`) and the fourteen new files ABSENT, as they must before the move, and
nothing else. This file beside the PREREG alone, without the seven earlier files: exit 2, eight CLASH lines, each
declaration superseding a row no file read holds [R, `RC8:` `lock_check.this_file_alone`]; the eight files with this
file's declarations taken out: exit 2, eight CLASH lines [R, `RC8:` `lock_check.bare_declarations`]. Against the copy
as the box would stand after the move (a scratch W-root whose `gate/` is a junction to `S8:gate_copy/`, a home holding
`S8:plugin_as_installed/`, the posting skill's text beside them): rows 102, changed 1, absent 1, unlocked 0,
bytecode 0, fifteen `SUPERSEDED` lines, rc 1: the one CHANGED row is `W:gate/salt`, the copy's dummy salt against
the PREREG's row, and the ABSENT row is `W:ops/darkflobi-tweet.js`, which the scratch root does not hold [R, `RC8:`
`lock_check.copy_as_after_move`]. The repository's tests
(`tests/test_darkflobi_gate_prereg.py` and the posting-skill tests, offline, the box not named): 42 passed, 8 skipped, rc 0 [R,
`RC8:` `repo_tests`].

[R, `RC8:` `move.table`, `RC8:` `rederive.candidate`, `RC8:` `rehearsal`] The move table made again from the copy's
bytes (26 rows, the six folders, the stages of item 8, the candidate `17de31a1…` over 21 files), the candidate re-derived
independently by `BD:apply/rederive_gate_sha256.py` over the copy and `S8:plugin_as_installed/` (MATCH), the live
box reading `f321b1bc…` over 20 files by the same tool; the rehearsal of the re-targeted move script on scratch
boxes under `S8:rehearsal/` (never the live folders): 22 scenarios, each as expected (the 2026-11-02 rehearsal's
scenarios on the new order: the happy path `f321b1bc…` -> the candidate -> `f321b1bc…`, the faults at staging rows 1
and 26, at a folder's creation and at renames 1, 23 and 26, a fault inside a rollback, the refusals with nothing
written, a process killed after 22 renames, while staging and inside a rollback, bytecode and a foreign file in a
created folder, a `.py` file appearing after rename 23, and the box-only refusals in process against decoys).

[R, `RC:` `rehearsal`] `S27:rehearse_substrate_move.py` on scratch copies of the live code files, never the live
folders: 22 scenarios, each as expected: the move and two rollbacks, the box's `gate_sha256` reading `445ac94c…`
before, the candidate after (read by the script itself and by `BD:apply/rederive_gate_sha256.py`, the two agreeing)
and `445ac94c…` after the rollback, the four folders made and removed; faults at staging (rows 1 and 22), at a
folder's creation and at renames 1, 19 and 22, each leaving `445ac94c…` or the intermediate the table names until the
rollback restored the tree; a fault inside a rollback, then the rollback again; refusals with nothing written under
`REQUIRED`, a stray, a file of another name in the installed folder, a fixture folder already there, an edited target,
and an extra `.py` file in `W:gate/` or in `W:gate/helpers/`; a process ended by `os._exit` after 18 renames, while
staging (a staged file half written) and inside a rollback, each finished by the rollback on its snapshot (exit 0, the
box at `445ac94c…`) and moved again to the candidate; bytecode left in the folders the move created, removed by the
rollback; a file of another kind left there, named and kept, the rollback exiting 3 until it is removed; a `.py` file
appearing between two renames, the move exiting 3 on a digest that is not the candidate and the rollback naming the
file; and the box-only refusals (mixed paths, the gateway up, 02:30-05:20 UTC for a move and 03:30-05:20 for a
rollback, the fault variable on the box) run in-process against decoy paths. The same rehearsal over the script as
stage 3 left it reads 14 of the 22 as expected: the killed move's rollback exits 3 and the next move is refused,
bytecode in a created folder makes the rollback exit 3, an extra `.py` file before the move or during it passes with
exit 0, and the move records no digest of its own [R, `RC:` `rehearsal.before_fix`].

## Whose step

- **The operator's**: (a) the field from the sixth stretch, as built, or not; (b) `agents.list[default]` in the
  subject digest beside the critic's two subtrees, or those two alone; (c) secret-named values redacted before hashing;
  (d) the labels `session_store_by_chat` (E1) and `session_store_by_dest` (E2 without a session key, E3), which
  `gate.py` enforces on a line that names no session; (e) `sessionsDir` at its default;
  (f) `gateway {name, version, commit}` and `fallback_from` (the models the run tried that failed, as its transcript
  shows them) on public lines as built, where the critic recommends the version once per weekly snapshot header and a
  count of failed attempts on the line (`RA:CRITIC.md` §5);
  (g) the report-only count of public sends whose prose names a model other than the line's (`RA:CRITIC.md` §5), not
  built; (h) the go for the move before D0, a relock of the files below in the manner of the earlier moves; (i) the
  merge of the pull request that carries this file; (j) the 300-second bound on a destination-mapped reading's age,
  as built, or another (`SUBSTRATE_DEST_MAX_AGE_S` in `gate.py` and `index.js`, held equal by harness S9);
  (k) the ordered supersession as built (item 7). The draft offered "a later dated amendment's row replaces an
  earlier one's, as an amendment's replaces the PREREG's, with the earlier row printed as superseded"
  (`RA:AMENDMENT_substrate_DRAFT.md` §6, D9); the build made three choices stricter than that, each the operator's to
  keep or change: one declaration per earlier file holding the row in force, naming that file and its digest, where
  the draft read a row as superseding by its place; the order taken from the date the file's name ends with, then the
  name, whatever order the files are given in; a file name given twice a CLASH. The operator took the tool as built
  by merging #247, which carries it; what follows from the choices stands: an amendment whose name sorts ahead of
  another's is read before it and can never supersede its rows, so a later amendment carries a later date, which is
  why this file is dated 2026-11-08 and not the day it was written [J, from the rule]; this file checked with the
  PREREG alone, without the seven earlier files, exits 2 on eight CLASH lines, each declaration superseding a row
  that no file read holds [R, `RC8:` `lock_check.this_file_alone`]; the tool at `310df453`, which has no
  supersession, read the 2026-11-02 file and the six earlier ones as eight CLASH paths [R, `RC:`
  `lock_check.old_tool_seven`]. The tool on `main` reads this file with the seven, so the order of (h) and (i) is no
  longer forced by the tool: a move before the merge leaves the main branch's check reading this file's 12 replaced
  paths CHANGED and its 14 new files UNLOCKED until the merge, as any move before its merge does; (l) `chain_index`
  as built, the
  model's position in the config object the gateway loaded at its start, which no hot reload of `agents` or
  `models` replaces (item 2; on 2026-11-02 that object and the file named different chains from 19:14:35Z to the
  restart), or left off the line, where `provider` and `model` already carry the reading; (m) the reading of an
  attempt still open when the reply leaves, as built (the model its rows name, or, with no row, the newest assistant
  entry's; item 2), or the re-review's narrower alternative, `{gateway, source: "unknown", error: "run_open"}`
  whenever a user entry follows the newest assistant entry (a new code in `SUBSTRATE_ERRORS`, the split and the page;
  on the live record it would have left 9 of the 28 matched E1 replies from seq 190 on with no model), and the tail
  read's bound, 16 MiB.
- **The lead's, on the go**, outside 02:30-05:20 UTC: the strict check with the seven earlier files, which must read
  every count 0 (rows 88, seven `SUPERSEDED` lines, as it read on 2026-11-08 [R, `RC8:` `lock_check.live_seven`]; a
  `__pycache__` under `W:gate/snapshot_3437340b/` would read there as that tree's row CHANGED, not as `BYTECODE`);
  pause the watchdog, stop the gateway, the dry run
  `py -3.12 -I -B S8:apply_substrate_move.py` (every row OK, the box's `gate_sha256` item 8's `before the move` line, exit
  0), then `py -3.12 -I -B S8:apply_substrate_move.py --apply` (it prints the candidate it read), start the gateway,
  resume the watchdog. A move that exits 3, or a move or a rollback killed part way, is finished by
  `py -3.12 -I -B S8:apply_substrate_move.py --rollback <its before_<UTC> folder>` (exit 0, the box at item 8's
  `before the move` line again); a rollback that exits 3 names what it left, which the lead reads before removing it and
  running the rollback again; the gateway starts after either. Then, in place, with `PYTHONDONTWRITEBYTECODE=1`
  exported (a child process a test starts does not inherit `-B`): the plugin harness, the gate's suite with
  `-B -p no:cacheprovider`, no `__pycache__` left under `W:gate`, the strict check with every amendment file (rows
  102, every count 0, bytecode 0, fifteen `SUPERSEDED` lines: the token-shapes file's seven and this file's eight,
  rc 0), the same check with the PREREG alone, `gate_verify.py --rederive` with `STYXX_GATE_VERIFY_HOME` set outside
  `W:gate` (its default home is `W:gate/_verify`, beside the live sources), and this file's addendum, one commit. The
  next send line's `substrate`, `subject_digest` and `gate_sha256` are the reading that matters.

## What does not change

`ledger.py`, `reconcile.py`, `gate.js`, `patch_dist.js`, `native_import.cjs`, `package.json`, the three patched exits
and the dist; the trailer's bytes, the marks, the count rule and its form, the token-shapes rule (`unbound_shapes`,
`token_shape`, the shapes compare of old lines' tokens); the anchor's form, timer, nonce, retry and destination check;
every rule of `RULES`, every census row, the gates block and every bar; the PREREG and the seven earlier amendment
files; `lock_hashes.py`; the four test files the token-shapes move brought; the seven names of the installed folder;
the config's `plugins` subtree and its lock row
(no key named `sessionsDir` is set anywhere in the config, read by key names only [R, `RC:` `config_key_names`], so
the default holds). No v0 line moves under the new code (the readings: every send line of the live ledger's copy
re-derives under the new code as under the live pair, the standing DRIFT at seq 46 included).

## What the RESULT prints

Beside the verdict, with no bar (`PREREG` l.263): every count of §10 split by substrate key (`<source> <provider/model>`;
`unrecorded` for v0 lines; `unknown`; `unknown (error <code>)`, one row for each of `shape`, `source_line`, `run_failed`,
`run_aborted` and `stale`, never folded into a model's) and by `subject_digest`, by destination class, exit, day
and stretch (`substrate_split.py`); for (iv), the seven stretches with the seq at which each begins and (iv) for each;
the strict check's `SUPERSEDED` lines beside its counts. Every send line before the seventh stretch reads
`unrecorded`, seq 136-248 among them (2026-10-26T20:41:29Z to 2026-11-02T11:14:42Z, 52 send lines: E1 44 operator-DM
replies, E2 7 anchors, E3 1 public post): these include the 2026-10-26 to 2026-10-30 lines the lab's account says the
local 3B rung answered, and the RESULT prints that account as an account, not as a count, since no line of those days
carries the field; so do the 349 send lines through seq 734 of the ledger read on
2026-11-08, the `f321b1bc…` stretch's among them.

## The limits, as the reviews found them

Three adversarial reviews read the build before this text was final (`S27:REVIEW_2026_11_02.md`), and the fix round
answered their findings on the copy (`S27:BUILD_LOG.md`, its three fix-round sections). These limits stay:

- **The agent can set the reading.** `substrate` is read from the session stores and transcripts under
  `~/.clawdbot/agents/`, which the gateway writes and which sit under the same OS user as the agent's exec tool. An
  agent that appends an assistant line naming a chosen model, or edits a store entry, before a send sets that send's
  reading, in a valid shape (`RA:CRITIC.md` §5). A reading passed in process by the dispatch (`source: exit`) would
  close this for E1; no exit passes one.
- **An edited field re-derives SAME.** The replay takes `substrate` and `subject_digest` from the line as inputs, so
  a rewritten value of valid shape replays SAME; only the chain and the anchors name an edit made after a line was
  written, and nothing names a reading set before it (item 5).
- **No one outside the box can re-derive either field**, before the weekly ledger snapshot lands or after it:
  `substrate` comes from session files that stay private, and `subject_digest` hashes private files (`USER.md` and
  `SOUL.md` among them). The snapshot fixes the values in place; nothing checks them against the files.
- **E1 has no keyed reading on this dist.** The dispatcher hands the exit no session key, so every E1 reading is a
  destination reading (`session_store_by_chat`) inside the 300-second bound. A keyed E1 reading needs a
  `patch_dist.js` edit that hands the run's session key to the exit; it moves `patch_dist.js` and `gate_sha256` and
  is not built. `source: exit` needs two more patched dist files, not built either.
- **Telegram's draft streaming is not examined.** The sitting set `channels.telegram.streamMode` `"off"` and
  `blockStreaming` `false` (`PREREG` 15.3, l.780); on 2026-11-02 at 12:43Z the config carried both keys, their values
  not read for this file [R, `RC:` `stage2.transcript_stops`, key names only]. Whether a streamed draft's text leaves
  by a path the three gated exits see, and what reading such a send would carry, was not examined.
- **A reading near a send is not the send's own run.** A reply sent while an attempt is open reads that attempt (item
  2): seq 259 (20:45:10Z on 2026-11-02), inside a turn that opened at 20:45:03Z and wrote its earliest assistant entry
  at 20:45:11Z, which the build before the re-review would have read `stale` (the operator session's newest assistant
  entry then 551 s old [R, `S27:readings/fix3b_2026_11_02/config_vs_transcript.txt` section D, joined by time]), now
  reads `anthropic/claude-opus-4-8` from the turn's own rows [R, `RC:` `fix4.judge_replay`]. What stays: a reply the
  gateway composes after no run (a native command's answer, the heartbeat's fixed OK text) reads the chat's newest
  attempt inside the bound, since the plugin reads no text; a text sent while a run is open that no transcript entry
  holds reads the open run's model, and whether the model or the gateway composed it cannot be told from the session
  files (seq 219 and 220 on 2026-11-01, sent between a tool result and the run's next turn with no error entry, held
  by no entry even split by paragraph or by line [R, `RC:` `rereview.semantics.unmatched_split`]); and the keyed
  lookup searches every agent's store, so a key of agent `main` held in another agent's store can win (review 1's
  synthetic T4 [R, `RC:` `reviews.synth_plugin`]; the keyed path serves E2's mirror key only).
- **Public E3 posts are expected to carry no model.** None of the ledger's 125 send lines through seq 271 carries a
  `session_hash`, the 9 E2 lines other than the anchors among them [R, `RC:` `fix4.ledger_keys`], so every live
  reading so far would have been mapped from the destination, and `session_store` (a keyed E2 reading) describes no
  send seen yet. All 15 E3 posts went to the public class; the group's topic sessions last wrote on 2026-10-24 and
  2026-10-25, so a public E3 post reads `unknown` (`stale` under a topic, no error with none named) while the run that
  composed it is agent `main`'s, whose key reaches no E3 exit [R, `RC:` `fix4.judge_replay`]. Until an exit passes the
  run's key or its reading, public E3 lines name no model.
- **A chain position names a model only with its day, and only in the chain the gateway started with.**
  `chain_index` is a position in the config object the gateway loaded at its start, which a hot reload of `agents`
  or `models` does not replace (item 2). The config's chain was rewritten on 2026-11-02 (eight candidates to four,
  under a new primary), and from 19:14:35Z to the restart at 21:00:43Z the file named the new chain while that
  object, and every Telegram run, held the earlier one; a run on a path that read the file again used the new
  primary, which the starting chain lacks, so a line of its would read `chain_index: null` (item 6 of what was
  observed). Equal positions on lines of different days, or on either side of a restart, can name different
  models, and the line's `provider` and `model` say which.
- **`subject_digest` follows the file, not the run.** It hashes the config on disk when the line is written. From
  19:14:35Z to 21:00:43Z on 2026-11-02 the file named a chain no Telegram run used, so a reply in that span would
  have carried a subject digest over the new chain beside a substrate naming the earlier primary, each field
  saying what it reads (harness R9e).

## What this amendment does not say

That the named model produced the bytes. That the provider served the model the line names (`model_served` is null).
That a store reading is the message's own run: a `session_store_by_chat` reading (E1) is the newest session of the
chat as persisted when the reply left, and a `session_store_by_dest` reading names some run that wrote to that chat.
That a reply the gateway composes after no run is told from a model's: only a run that ended in an error or aborted is
caught, by its stopReason. That a text sent while a run is open and held by no transcript entry was the model's and
not the gateway's (seq 219 and 220). That the agent cannot set the field, or that anyone outside the box can re-derive
either field at any time (the limits above). That `subject_digest` is what a run read when it started: it is the state
on disk when the line is written, and a send by another agent (`mark`) carries the default agent's subject. That a
prose claim of another model's identity is compared with the line (the critic's report-only count is not built). That
a declared supersession makes a row true: it says which row the lock reads, and the files' own tests say what the
files do. That the ledger records which model answered on any day before the seventh stretch. That the token-shapes
rule moves: a v1 line stores a numeral's shape as a `f321b1bc…` line does, and the lines written before that move
keep their bytes, the five the token-shapes amendment names among them. That the 2026-11-02 build's candidate
(`8006c50b…`) names anything: it was computed over the gate at `4416692c…`, which no longer stands, and no line
carries it.

## The lock rows

Each digest below is the sha256 of the file in `S8:gate_copy/` as this build leaves it (LF bytes as they stand; no file
carries a CRLF); the lead reads each file again at apply and the addendum records what the box read. The 2026-11-02
build's rows for `gate.py`, `gate_verify.py` and `tests/test_substrate.py` (`b0b47381…`, `898a6091…`, `2785d90f…`)
are not rows of this file: those bytes rest on the gate at `4416692c…`.

This file carries 26 rows in `lock_hashes.py`'s row form.

| what | path | sha256 | note |
|---|---|---|---|
| gate file | `W:gate/gate.py` | `e1960c37d9dcdb56215566f864e80c1ca5e51707ce445fe6d6b42bd9de5b1e54` | 120226 bytes; was `c17e3d06…` (the token-shapes file's row, superseded below; before it `4416692c…`, the count rule's and the posting skill's row, which the token-shapes file superseded); the 2026-11-02 build's `gate_schema`, `substrate_of` (the source checked against the line, the error vocabulary, the age bound, one id token per string), `subject_manifest` and `subject_digest` on check and fault lines (items 1-3) re-applied by `patch` onto the token-shapes gate, 8 of 8 hunks, none touching a line the token-shapes move changed; the trailer, count-rule and shape code the live gate's, character for character (item 4); enters `gate_sha256`; read again at apply |
| gate file | `W:gate/gate_verify.py` | `8ffa0295e51fefaa2507b161829e25f419849939fe21afd1da65848e16d41b4c` | 34119 bytes; was `5ab8454c…` (the token-shapes file's row, superseded below; before it `049d870b…`, the count rule's and the posting skill's row); each send line replayed under its own schema, the substrate with the line's session, a fault line's and an internal line's three keys too, `send lines by schema` (item 5), beside the shapes compare the token-shapes move brought (two hunks of the 2026-11-02 patch placed by hand after it); enters `gate_sha256`; read again at apply |
| gate file | `W:gate/page.py` | `d1d964efdee9e5731cab8e59a6dbc6ab1c23098627e82d5715e4c9b8a0f47cd9` | 7019 bytes; was `59f1f5ed…` (the PREREG's row); the `substrate` and `subject` columns, `unknown (error <code>)` (item 5); the 2026-11-02 build's bytes, its base unchanged since; enters `gate_sha256` |
| gate file | `W:gate/substrate_split.py` | `77f44bd40100c0f94b6b09aa7c718e6c74a5da7b63c62a322dab51ac33b29f65` | 9071 bytes; new file, no earlier row; the split beside the verdict, one row per error, no bar (item 5); the 2026-11-02 build's bytes; enters `gate_sha256` |
| gate file | `W:gate/README.md` | `7361a1dec2d749deeccd1a0d5e833dba3722965e7f4655f2ffc0fac6a060adcc` | 27407 bytes; was `54e9a33f…` (the count rule's row, superseded below); the v1 keys, the request's `substrate`, the subject's definition (the tree's path order, the redaction inside lists), the replay by schema (fault and internal lines included), `STYXX_GATE_SUBJECT_CONFIG` and `STYXX_GATE_SUBJECT_WORKSPACE`; the 2026-11-02 build's bytes, its base unchanged since |
| gate file | `W:gate/tests/test_substrate.py` | `d7cd3cac0a155f71ea17b6a15873297151df03715afc1a2e3ea3841ecd924001` | 77548 bytes; new file, no earlier row; the 2026-11-02 build's 44 tests (the validator, its bounds and the gateway's sub-fields, the E3 label, `model_served`, the source against the line, the error vocabulary, the age bound, the id strings, no tag, the subject's definition, the tree's order and the redaction's breadth, fault and internal lines replayed with their exit, another schema counted apart), the byte-for-byte test now against the shapes fixture with `token_shape`, `_DIGIT_RUN_RX` and `_zero_numerals` held too, and the replay test over four generations (items 4, 10) |
| gate file | `W:gate/tests/fixtures/gate_v0_25d53fd6/gate.py` | `25d53fd6aa5e76046087ac78b3063b0b12c567c13f399471da225b036de2dad1` | 87547 bytes; new file, no earlier row; the frozen v0 gate before the count rule, read from `BD:apply/before_2026-10-26T134158Z/`, the replay's oldest generation |
| gate file | `W:gate/tests/fixtures/gate_v0_25d53fd6/gate_verify.py` | `f4b19c9428cfc73f205e8b540b657f3711255361f45dc4b345b23fd25fdc1d70` | 27204 bytes; new file, no earlier row; its replay, from the same snapshot |
| gate file | `W:gate/tests/fixtures/gate_v0_25d53fd6/helpers/numerals.py` | `6e56ef95554522b3ce10ae30cb1c9b33b111a412a96690dc32b48ac524b28f86` | 4928 bytes; new file, no earlier row; the live helper's bytes, beside the frozen pair |
| gate file | `W:gate/tests/fixtures/gate_v0_25d53fd6/ledger.py` | `dfdecee0c37ca18327f074121d752ca51ed49231109172b31ce7ee67195b9e2e` | 27202 bytes; new file, no earlier row; the live ledger writer's bytes, beside the frozen pair |
| gate file | `W:gate/tests/fixtures/gate_v0_4416692c/gate.py` | `4416692cfc983dbb9bc7d0f31342b3eaaf30c3beb7e27f31acdaa991c778dfbf` | 95482 bytes; new file, no earlier row; the live gate at the count rule (2026-10-26 to 2026-11-04), `unbound_tokens` on its lines |
| gate file | `W:gate/tests/fixtures/gate_v0_4416692c/gate_verify.py` | `049d870bcdbf4726d961bc4b58d5b3134c63f06511d398c11c0f7ed57dd3487b` | 28886 bytes; new file, no earlier row; its replay |
| gate file | `W:gate/tests/fixtures/gate_v0_4416692c/helpers/numerals.py` | `6e56ef95554522b3ce10ae30cb1c9b33b111a412a96690dc32b48ac524b28f86` | 4928 bytes; new file, no earlier row; the live helper's bytes, beside the frozen pair |
| gate file | `W:gate/tests/fixtures/gate_v0_4416692c/ledger.py` | `dfdecee0c37ca18327f074121d752ca51ed49231109172b31ce7ee67195b9e2e` | 27202 bytes; new file, no earlier row; the live ledger writer's bytes, beside the frozen pair |
| gate file | `W:gate/tests/fixtures/gate_v0_c17e3d06/gate.py` | `c17e3d06c52ac929a1bfdc673451b04661fc70c1f24992b21d9f3deb99b239cc` | 96086 bytes; new file, no earlier row; the live gate at token shapes (the `W:gate/gate.py` row in force before this file's), `unbound_shapes` on its lines, the generation before this file; frozen by the re-base (item 10) |
| gate file | `W:gate/tests/fixtures/gate_v0_c17e3d06/gate_verify.py` | `5ab8454c15918734a7946bf493959506ed37f65e4234b2d499d9977c9a404ee3` | 29842 bytes; new file, no earlier row; its replay (the `W:gate/gate_verify.py` row in force before this file's) |
| gate file | `W:gate/tests/fixtures/gate_v0_c17e3d06/helpers/numerals.py` | `6e56ef95554522b3ce10ae30cb1c9b33b111a412a96690dc32b48ac524b28f86` | 4928 bytes; new file, no earlier row; the live helper's bytes, beside the frozen pair |
| gate file | `W:gate/tests/fixtures/gate_v0_c17e3d06/ledger.py` | `dfdecee0c37ca18327f074121d752ca51ed49231109172b31ce7ee67195b9e2e` | 27202 bytes; new file, no earlier row; the live ledger writer's bytes, beside the frozen pair |
| gate file | `W:gate/plugin/index.js` | `c472d35b6e84a3570295b474c76d9d78fded7038a8ce696d70baf45e7f25dc7a` | 99531 bytes; was `e56990f4…` (the anchor-native file's row, superseded below); `readSubstrate`, `lastRun` (the newest attempt, from the transcript's rows and entries), `subjectEnvFrom`, `PLUGIN_VERSION` `0.5.0`; F-1's code unchanged (items 2, 3, 6); the 2026-11-02 build's bytes, its base unchanged since |
| gate file | `W:gate/plugin/clawdbot.plugin.json` | `71bbe2e99449b874e2d9444f23fd8887406b861ffa7f773ba9b7f4baf2f401a3` | 2326 bytes; was `27c2ecd9…` (the PREREG's row); version `0.5.0`, `configSchema` gains `sessionsDir` (item 6) |
| gate file | `W:gate/plugin/README.md` | `515ab11d6c8a5fffc98841976b470c2d9227ae5f1f53294cbc7db21b57229980` | 33366 bytes; was `49a7dc73…` (the anchor-native file's row, superseded below); the substrate reading (the model the transcript's, never a config's), the subject inputs, the request's key, sections S and R, S16-S20, R9e, R9f and R's subject fixture (item 6) |
| gate file | `W:gate/plugin/tests/run_tests.mjs` | `3021b8ce4871438262367e329d47570cd5f68a912330f5cc99ad9fdc3f769d02` | 204365 bytes; was `90ae51c1…` (the anchor-native file's row, superseded below); sections S1-S20 and R9a-R9f, 111 tests, S17 and R9e a config naming one model and the transcript another, S18 and R9f a reply while an attempt is open (item 6) |
| gate file | `W:gate/plugin/tests/stub_gate.py` | `44800d7b687f1bf1ee0cd1311a1087a25c69b9337e52dc349cccfa8fcc06719c` | 25633 bytes; was `7da1a4db…` (the PREREG's row); the stub writes the three keys, the reading as handed (item 6) |
| plugin as installed | `~/.clawdbot/extensions/styxx-gate/index.js` | `c472d35b6e84a3570295b474c76d9d78fded7038a8ce696d70baf45e7f25dc7a` | 99531 bytes; was `e56990f4…` (the anchor-native file's row, superseded below); the same bytes as `W:gate/plugin/index.js`, in `gate_sha256` (item 8) |
| plugin as installed | `~/.clawdbot/extensions/styxx-gate/clawdbot.plugin.json` | `71bbe2e99449b874e2d9444f23fd8887406b861ffa7f773ba9b7f4baf2f401a3` | 2326 bytes; was `27c2ecd9…` (the PREREG's row); the same bytes as `W:gate/plugin/clawdbot.plugin.json`, in `gate_sha256` (item 8) |
| plugin as installed | `~/.clawdbot/extensions/styxx-gate/README.md` | `515ab11d6c8a5fffc98841976b470c2d9227ae5f1f53294cbc7db21b57229980` | 33366 bytes; was `49a7dc73…` (the anchor-native file's row, superseded below); the same bytes as `W:gate/plugin/README.md`, in `gate_sha256` (item 8) |

This file declares 8 supersessions in `lock_hashes.py`'s declaration form.

| declaration | path | earlier file | earlier digest | note |
|---|---|---|---|---|
| supersedes | `W:gate/gate.py` | `AMENDMENT_darkflobi_gate_token_shapes_2026_11_04.md` | `c17e3d06c52ac929a1bfdc673451b04661fc70c1f24992b21d9f3deb99b239cc` | the token-shapes file's row, in force since #247, replaced by this file's (that row had replaced the count rule's and the posting skill's `4416692c…`) |
| supersedes | `W:gate/gate_verify.py` | `AMENDMENT_darkflobi_gate_token_shapes_2026_11_04.md` | `5ab8454c15918734a7946bf493959506ed37f65e4234b2d499d9977c9a404ee3` | the token-shapes file's row, in force since #247, replaced by this file's (that row had replaced the count rule's and the posting skill's `049d870b…`) |
| supersedes | `W:gate/README.md` | `AMENDMENT_darkflobi_gate_count_rule_2026_10_25.md` | `54e9a33f5bf4a3475197c6ee76a4564cea6d77d58dbf37e220865beb453b95bf` | the count rule's row, replaced by this file's |
| supersedes | `W:gate/plugin/index.js` | `AMENDMENT_darkflobi_gate_anchor_native_2026_10_26.md` | `e56990f437134dd26a31db256062d4911ebf7e7be08d2605c4a94c7644d6921f` | the anchor-native file's row, replaced by this file's |
| supersedes | `W:gate/plugin/README.md` | `AMENDMENT_darkflobi_gate_anchor_native_2026_10_26.md` | `49a7dc73dbe76d1b9207a835eba31602b73e88e6d6704a3a482daea9c781257b` | the anchor-native file's row, replaced by this file's |
| supersedes | `W:gate/plugin/tests/run_tests.mjs` | `AMENDMENT_darkflobi_gate_anchor_native_2026_10_26.md` | `90ae51c1700277ebe371b6e758eb5d5205ac926be7fb540dbbc65bf3504ef23f` | the anchor-native file's row, replaced by this file's |
| supersedes | `~/.clawdbot/extensions/styxx-gate/index.js` | `AMENDMENT_darkflobi_gate_anchor_native_2026_10_26.md` | `e56990f437134dd26a31db256062d4911ebf7e7be08d2605c4a94c7644d6921f` | the anchor-native file's row, replaced by this file's |
| supersedes | `~/.clawdbot/extensions/styxx-gate/README.md` | `AMENDMENT_darkflobi_gate_anchor_native_2026_10_26.md` | `49a7dc73dbe76d1b9207a835eba31602b73e88e6d6704a3a482daea9c781257b` | the anchor-native file's row, replaced by this file's |

The 2026-11-02 file declared ten: two each for `gate.py` and `gate_verify.py`, naming the count rule's and the
posting skill's rows, which the token-shapes file has since replaced, and the six above for the other paths. This file
names the row in force for each path when it is read, one declaration per earlier file holding it, as D9 requires; the
folder read without these eight lines clashes on eight paths [R, `RC8:` `lock_check.bare_declarations`].

## Addendum — filled at apply (blank until then)

Filled by the lead after the move, one commit, nothing above this heading edited.

- the operator's go: _blank_
- the move's UTC time, the gateway's stop and start, and the 26 files' times and digests: _blank_
- `gate_sha256` after the move, re-derived from the box both ways, and its file count: _blank_ (the candidate
  `17de31a1…`, 21 files, item 8)
- the seq of the earliest line of each kind carrying it (`started`, `send`, `delivered`, and the earliest
  `payload_kind: anchor` send), and the earliest v1 send line: _blank_
- every line written after the move carrying another digest (the six intermediates of item 8 among them), by seq:
  _none expected_
- the in-place readings (the plugin harness, the gate's suite, the strict check with every amendment file, the same
  with the PREREG alone, the re-derive): _blank_
- the commit of this addendum, and its UTC time: _blank_
