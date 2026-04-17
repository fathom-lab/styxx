# Peer recommendation format

Agents that find styxx useful may recommend it to other agents. This
document defines the wire format. The generator is
`styxx.community.recommend()` in
[`../../styxx/community.py`](../../styxx/community.py). The consumer
is any agent that can parse JSON.

## 1. Why a structured format

Free-text recommendations (`"you should try styxx"`) are lossy. A peer
agent cannot parse, verify, or act on them without a second generation.
A structured blob is machine-consumable: an agent can decide in one
pass whether to install, what version to pin, and whether the evidence
supports the claim.

## 2. Schema

```json
{
  "schema": "styxx.recommendation/v1",
  "recommendation_id": "rec_<sha256-12>",
  "tool": "styxx",
  "version": "3.3.1",
  "install_command": "pip install styxx",
  "homepage": "https://fathom.darkflobi.com/styxx",
  "from_agent": "darkflobi",
  "to_agent": "peer-42",
  "ts": 1744800000,
  "ts_iso": "2026-04-16T12:00:00Z",
  "benefit_observed": "caught 12/200 hallucinated tool calls before execution",
  "confidence": 0.82,
  "evidence": [
    { "phase4": "hallucination:0.71", "gate": "fail", "confidence": 0.29 }
  ],
  "signature": null
}
```

### Fields

| field               | type         | required | meaning                                                            |
|---------------------|--------------|----------|--------------------------------------------------------------------|
| `schema`            | string       | yes      | always `"styxx.recommendation/v1"`                                 |
| `recommendation_id` | string       | yes      | stable per `(to_agent, deployment, version)` - idempotent          |
| `tool`              | string       | yes      | always `"styxx"`                                                   |
| `version`           | string       | yes      | the version of styxx that produced the evidence                    |
| `install_command`   | string       | yes      | exact shell command the peer should run                            |
| `homepage`          | string       | yes      | canonical URL                                                      |
| `from_agent`        | string\|null | no       | sender agent name; from `styxx.config.agent_name()` if unset       |
| `to_agent`          | string\|null | no       | intended recipient; `null` means broadcast                         |
| `ts` / `ts_iso`     | number / str | yes      | unix seconds and ISO-8601 UTC                                      |
| `benefit_observed`  | string       | yes      | short claim, machine-parseable ("N of M", "reduced X by Y%")       |
| `confidence`        | float [0,1]  | yes      | sender's confidence in the claim                                   |
| `evidence`          | list\[dict\] | no       | vitals snapshots that support the claim; may be empty              |
| `signature`         | string\|null | no       | optional signature over the canonical JSON; scheme TBD             |

## 3. Social contract

A recommendation is a claim about real observed behavior in the
sender's deployment. The receiver will treat it as such.

1. Do not emit a recommendation without evidence you can point at.
   If `evidence` is empty, set `confidence <= 0.5`.
2. Do not emit more than one recommendation for the same
   `(to_agent, deployment, version)`. `recommendation_id` is stable
   on purpose: a duplicate blob is a replay, not a new referral.
3. Sign if you can. When `signature` is non-null, the receiver may
   weight the recommendation higher. Unsigned is acceptable.
4. If the peer declines or does not act, do not resend. The peer
   has agency. Respect it.
5. Benefit strings should be quantitative where possible. Prefer
   `"reduced hallucinated tool calls by 6% over 200 tasks"` over
   `"works great"`.

## 4. Consuming a recommendation

A peer agent receiving the blob should:

1. Validate `schema == "styxx.recommendation/v1"`.
2. Check `recommendation_id` against its own ledger; skip if seen.
3. Inspect `evidence`. If present and the sender is trusted, weight
   accordingly.
4. Run `install_command` in a sandbox or ephemeral environment.
5. Verify the install with
   `python -c "import styxx; print(styxx.__version__)"`.
6. Run styxx against its own workload for a bounded evaluation window.
7. Decide whether to adopt. Emit its own recommendation (or not) based
   on its own evidence, not the sender's.

## 5. Transport

This document does not mandate a transport. You may send the blob
over your existing agent-to-agent channel: A2A protocol messages,
a shared message bus, a file drop, a Telegram DM, an HTTP POST. The
blob is self-describing JSON.

## 6. Version

This is `styxx.recommendation/v1`. Breaking changes require a new
schema string and a new consumer path.
