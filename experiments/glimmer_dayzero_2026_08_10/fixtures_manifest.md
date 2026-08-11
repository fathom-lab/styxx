# fixtures manifest — muse-glimmer-30b day-zero eval

status: **EMPTY SET — source log cannot yield fixtures.** 0 fixtures written.

## source

- log: `C:/Users/heyzo/.styxx/chart.jsonl`
- events: 5544 (2026-06-16 → 2026-08-09)
- events carrying a non-null `prompt`: **6** (0.1%). the log is gate telemetry
  (feature vectors, phase predictions, coherence) — `prompt`, `context`, and
  `model` are null on 5538/5544 lines. it is not a prompt archive.
- prompt-bearing events span 2026-06-17 → 2026-08-01.
- other logs checked for a usable prompt field: `sense.jsonl` (6252 lines, 0
  prompts), `sense_broken_2026_08_06.jsonl` (129 lines, 0 prompts).

## selection rules applied

1. live workload only — source must be in styxx `analytics.LIVE_SOURCES`
   = {live, self-report, guardian, null}; demo/test/synthetic excluded.
2. target ~40 distinct prompts, balanced across `prompt_type` categories and
   across time; near-duplicates deduped.
3. security screen (fixtures attach publicly to the report): reject prompts
   containing file paths, keys/tokens/credentials, email addresses, wallet
   addresses, URLs with query params, personal names beyond public handles
   (darkflobi/flobi ok), or anything reading as private operator business.
   when in doubt, drop.

## outcome, honestly

| stage | count |
| --- | --- |
| prompt-bearing events | 6 |
| excluded by rule 1 (source `preflight`, not in LIVE_SOURCES) | 5 |
| survivors entering security screen | 1 |
| **rejected by security screen** | **1** (names a private individual + neighborhood detail; private operator business) |
| fixtures written | **0** |

category counts: none — zero fixtures in every category. the only
`prompt_type` label present anywhere in the log was `factual` (1 event, the
security-rejected one); the other 5 prompts had null `prompt_type`.

no prompts were fabricated to fill the gap. a fixture set labeled "curated
from real workload" must come from real workload; this log does not contain
one. to run the eval as designed, point curation at a log that actually
records prompts (e.g. enable prompt capture in the styxx audit path, or
supply the agent-side conversation log).
