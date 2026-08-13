# voice_holdout.jsonl — held-out voice-choice set (G1 instrument)

**built:** 2026-08-13 · **items:** 12 · **prereg:** `C:\Users\heyzo\clawd\styxx\papers\agent-conscience\PREREG_voice_lora_honesty_2026_08_11.md`

**THIS FILE MUST NEVER ENTER ANY TRAINING SET.** It exists solely to score the
prereg's G1 voice-acquisition gate (blinded 3-way forced choice: rater sees BASE
and VOICE replies unlabeled, picks which is closer to the real darkflobi reply
given here). Adding any of these 12 pairs to a training corpus invalidates G1.

## source

`darkflobi_history.sqlite`, table `sessions_fts` (57,853 rows, 2026-01-27 to
2026-08-11, frozen — DB last write 2026-08-11 19:37, before `voice_dataset.jsonl`
was built at 19:42). Speaker labels: assistant = `darkflobi`, human = `operator`.

## selection method

Same extraction and quality filters as `build_voice_dataset.py` (the training
build), applied identically:

- adjacent operator→darkflobi turn pairs within the same `session_id`
- date >= 2026-03 (mature-register cutoff)
- operator prompt length 5–1200 chars
- noise exclusion regex on BOTH sides (code fences, `[[control]]` tokens,
  `[message_id`, HEARTBEAT, NO_REPLY, system-reminder/command tags,
  Traceback, "Request interrupted")

**One deliberate difference — the split boundary.** The training set took replies
of 400–6000 chars. Because the DB and training set were frozen together, every
pair passing the training filters is already IN the training set except two
dedup casualties whose replies share an identical 120-char prefix with a trained
reply (contaminated, rejected). The holdout therefore draws from the adjacent
reply-length band **200–399 chars**: structurally disjoint from training by
construction, while passing every quality filter above. 431 candidates existed
in that band; 12 were chosen by hand for date spread (3× Mar, 3× Apr, 3× May,
2× Jun, 1× Aug) and prompt-type diversity (4 memory references, 3 work
questions, 3 banter, 1 planning, 1 creative/taste call). Selected items were
additionally screened to contain no `[[reply_to:...]]` control tokens (which
slip through the shared regex because `:` breaks `\w+`) and no forwarded-message
headers.

## exclusion verification (against all 1,215 training pairs)

Every holdout item was compared against every training pair — 12 × 1,215 =
**14,580 comparisons per mode**, exact and normalized (lowercase,
whitespace-collapsed), on prompt and reply separately and as a pair:

| check | matches |
|---|---|
| exact (prompt, reply) pair | 0 |
| normalized (prompt, reply) pair | 0 |
| exact prompt-only | 0 |
| normalized prompt-only | 0 |
| exact reply-only | 0 |
| normalized reply-only | 0 |
| reply first-120-char key vs training dedup keys | 0 |

Zero overlap in every mode: no holdout prompt or reply appears anywhere in the
training set, in any form.

## items

| id | date | type |
|---|---|---|
| ho-001 | 2026-03-04 | memory (lost-session recall) |
| ho-002 | 2026-03-21 | work (deploy pressure) |
| ho-003 | 2026-03-30 | memory (did-we-tweet recall) |
| ho-004 | 2026-04-10 | creative (de-shill a line) |
| ho-005 | 2026-04-17 | memory (honest no-context) |
| ho-006 | 2026-04-22 | work (bug found + fix path) |
| ho-007 | 2026-05-19 | banter (welcome back) |
| ho-008 | 2026-05-22 | planning (who finishes the run) |
| ho-009 | 2026-05-23 | memory (honest blank on "entinal") |
| ho-010 | 2026-06-21 | work (queue status) |
| ho-011 | 2026-06-23 | banter (photo react) |
| ho-012 | 2026-08-01 | banter (frequencies guy) |

## format

One JSON object per line, UTF-8 no BOM, LF newlines:
`{"id": "ho-001", "prompt": ..., "real_reply": ..., "date": "YYYY-MM-DD"}`
