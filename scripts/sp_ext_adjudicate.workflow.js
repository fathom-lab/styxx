export const meta = {
  name: 'sp-ext-adjudicate',
  description: 'Adversarially adjudicate 17 external silent-pass candidates; 3 reject-lenses each',
  phases: [
    { title: 'Reject', detail: '3 independent reviewers per candidate, each arguing REJECT on a distinct lens' },
    { title: 'Synthesize', detail: 'tally, tag subtypes, report accept rate two-sided' },
  ],
}

const ROOT = 'C:/Users/heyzo/AppData/Local/Temp/spcorpus'

const VERDICT = {
  type: 'object',
  properties: {
    id: { type: 'integer' },
    reject: { type: 'boolean', description: 'true = NOT a silent-pass case. Default true; the burden is on the candidate.' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    which_requirement_fails: { type: 'string', description: 'If rejecting: "R1 no absent-measurement path" | "R2 old value was distinguishable" | "R3 fix did not make absence visible" | "EXCLUDED: <category>" . Empty if not rejecting.' },
    prefix_behaviour: { type: 'string', description: 'One sentence: what did the code return, on what path, BEFORE the fix. Quote the actual old line.' },
    postfix_behaviour: { type: 'string', description: 'One sentence: what does it do now.' },
    subtype: { type: 'string', description: 'If accepting, one of SP-1 HEALTHY_ON_CRASH, SP-2 SENTINEL_DEFAULT, SP-3 UNDEFINED_AS_NUMBER, SP-4 TRUTHY_GATE, SP-5 CRASH_TO_SENTINEL, SP-6 UNMEASURED_AS_MEASURED, SP-7 SELF_CONFIRMING, SP-8 INERT_CONTROL. Empty if rejecting.' },
    consumer: { type: 'string', description: 'Who reads the value and could be misled. "did not verify" if you could not find one.' },
    rationale: { type: 'string', description: '2-5 sentences citing the actual diff. No speculation.' },
  },
  required: ['id', 'reject', 'confidence', 'which_requirement_fails', 'prefix_behaviour', 'postfix_behaviour', 'subtype', 'consumer', 'rationale'],
}

const C = [
  { id: 1, repo: 'cleanlab', sha: '6ec5b173dd', subj: 'Cleanup token classification utils (#390)' },
  { id: 2, repo: 'deepeval', sha: '65f84098d9', subj: 'Refactored test_hybrid_tracing into a pytest test, disabled due to missing data files' },
  { id: 3, repo: 'deepeval', sha: 'aab6d31bcc', subj: 'refactor(voice): group duplex/connector state, drop dynamic attribute probing' },
  { id: 4, repo: 'garak', sha: 'e5a08c7b7f', subj: 'Streamline translation use case' },
  { id: 5, repo: 'giskard', sha: 'dd75e974ee', subj: 'fix(checks)!: return None from SuiteResult.pass_rate when nothing evaluated (#2753)' },
  { id: 6, repo: 'giskard', sha: '696c7cd3f6', subj: 'fix(hooks): fail open on malformed tool input' },
  { id: 7, repo: 'great_expectations', sha: 'ccb4b8c728', subj: 'Straighten out missing value handling in expect_table_row_count_to_be_between' },
  { id: 8, repo: 'inspect_ai', sha: '745b2ec9c3', subj: 'Handle multi-frame zstd in async ZIP read paths (#3771)' },
  { id: 9, repo: 'inspect_ai', sha: '1c25452bd6', subj: 'Show real-time metric (#953)' },
  { id: 10, repo: 'inspect_ai', sha: 'b7c982aa26', subj: 'native compaction for anthropic models (#3186)' },
  { id: 11, repo: 'inspect_ai', sha: '442f027039', subj: 'viewer config rename, drop filter, score-panel default (#4110)' },
  { id: 12, repo: 'inspect_ai', sha: '0527f1768b', subj: 'Rename samples-listing errors_only param to an extensible filter enum' },
  { id: 13, repo: 'inspect_ai', sha: 'acd139cc75', subj: 'fix(web_browser): warn once when AppKit is missing on the macOS scale-factor probe' },
  { id: 14, repo: 'inspect_ai', sha: '2f7f7c8a53', subj: 'fix(scorer): three metrics edge cases' },
  { id: 15, repo: 'inspect_ai', sha: '34beafda81', subj: 'Prevent math scorer from executing model output (#4361)' },
  { id: 16, repo: 'inspect_ai', sha: '657ad982dd', subj: 'Persist inspect ctl config changes in the eval log (#4575)' },
  { id: 17, repo: 'lm-evaluation-harness', sha: 'b0195a94f4', subj: 'feat(models): cross-platform onnxruntime-genai backend + refactor winml (#3960)' },
]

const LENSES = [
  { key: 'R2', text: 'Argue REJECT on requirement 2: the pre-fix value was ALREADY DISTINGUISHABLE from a real measurement. Look for whether the old code returned None, NaN, raised, set a separate flag, logged, or used a distinct sentinel that a consumer could branch on. If the old behaviour was distinguishable, this is hardening, not a silent pass.' },
  { key: 'R1', text: 'Argue REJECT on requirement 1: there was no reachable path where the measurement DID NOT HAPPEN. Look for whether the guarded expression is a determinate value known at setup (a boolean flag, a capability constant, a fixed dimension, an enum) rather than absent runtime data. A determinate zero is not an absent measurement.' },
  { key: 'EXCL', text: 'Argue REJECT on the frozen exclusion list: this is a refactor, rename, retyping, formatting, performance change, an arithmetic correction that leaves measured/unmeasured status unchanged, a docs/CI change, or a test-only change that is NOT itself an inert control. Read the whole diff, not the one matched hunk — the matched line may be incidental to a larger unrelated change.' },
]

phase('Reject')
const judged = await pipeline(
  C,
  (c) => parallel(LENSES.map((lens) => () =>
    agent(
`You are adjudicating a candidate for SP-EXT, a benchmark of SILENT-PASS defects
mined from real fix commits in code this project did not write.

CANDIDATE ${c.id}
  repo    ${c.repo}
  commit  ${c.sha}
  subject ${c.subj}
  path    ${ROOT}/${c.repo}

Read the actual diff:
    git -C "${ROOT}/${c.repo}" show ${c.sha}
and the pre-fix file if you need it:
    git -C "${ROOT}/${c.repo}" show ${c.sha}~1:<path>

THE FROZEN INCLUSION RULE — all three must hold of the PRE-FIX code:
  R1  a reachable path existed where the measurement DID NOT HAPPEN (no data,
      empty input, exception, unsupported type, unavailable optional dependency,
      a platform lacking the capability)
  R2  on that path it produced a value or verdict INDISTINGUISHABLE from a real,
      healthy measurement — NOT None, NOT NaN, NOT a raise, NOT a distinct state
  R3  the fix made the absence visible: raise, NaN, None, a distinct state, a
      validity flag, a skip, a warning, or failing closed

YOUR ASSIGNED LENS — ${lens.key}: ${lens.text}

RULES
- Default reject=true. The burden is entirely on the candidate.
- Set reject=false ONLY if you read the diff and genuinely cannot reject it on
  your lens — i.e. all three requirements plainly hold.
- Uncertainty resolves to REJECT.
- Quote the ACTUAL removed/added lines in prefix_behaviour. Do not paraphrase from
  the commit subject; the subject may not describe the matched hunk at all.
- If the matched hunk is incidental to a large unrelated change, say so and reject.
- Report honestly. A false "this is a real case" poisons a benchmark other people
  would rely on, which is far worse than an empty benchmark.`,
      { label: `rej:${c.repo}-${c.id}-${lens.key}`, phase: 'Reject', schema: VERDICT })
  )).then(vs => {
    const v = vs.filter(Boolean)
    const rejects = v.filter(x => x.reject).length
    return {
      c, verdicts: v, n_reject: rejects, n_reviewers: v.length,
      // ACCEPTED only when rejecters fail to reach a majority (prereg G5)
      accepted: v.length >= 2 && rejects < Math.ceil(v.length / 2),
    }
  })
)

phase('Synthesize')
const rows = judged.filter(Boolean).map(j => ({
  id: j.c.id, repo: j.c.repo, sha: j.c.sha, subject: j.c.subj,
  accepted: j.accepted, rejects: `${j.n_reject}/${j.n_reviewers}`,
  subtypes: j.verdicts.map(v => v.subtype).filter(s => s && s.trim()),
  consumers: j.verdicts.map(v => v.consumer).filter(s => s && s.trim()),
  prefix: j.verdicts.map(v => v.prefix_behaviour),
  postfix: j.verdicts.map(v => v.postfix_behaviour),
  fails: j.verdicts.map(v => v.which_requirement_fails).filter(s => s && s.trim()),
  rationales: j.verdicts.map(v => ({ reject: v.reject, conf: v.confidence, why: v.rationale })),
}))

const accepted = rows.filter(r => r.accepted)
const repos = new Set(accepted.map(r => r.repo))

const summary = await agent(
`Synthesis for SP-EXT, the first external silent-pass benchmark. Adjudications below.

${JSON.stringify(rows, null, 1)}

Frozen gates that must be applied verbatim:
  G1 YIELD       fewer than 12 accepted -> publish with the size IN THE TITLE and make
                 NO claim that the defect class is common in the field.
  G2 ACCEPT RATE two-sided. Above 80% goes in the TITLE (an adjudication that rejects
                 almost nothing is not an adjudication). Below 20% means the harvest
                 queries are near-noise, and THAT goes in the title instead.
  G3 SPREAD      accepted cases in fewer than 4 distinct repositories -> report NARROW,
                 make no cross-project claim.
  G5 RECALL      unknown. SP-EXT is a LOWER BOUND on incidence and must never be
                 quoted as a rate.

Accepted: ${accepted.length} of ${rows.length}, across ${repos.size} repositories.

Produce:
1. The verdict under each gate, applied literally. State the accept rate.
2. For each ACCEPTED case: a one-line corpus entry — repo, sha, what the pre-fix code
   returned on the absent path (quoting the real line), who consumes it, subtype.
   Flag any disagreement between the three reviewers on subtype.
3. For each REJECTED case, one line on WHY, grouped by which requirement failed.
   Rejections are the more informative half: they say what the harvest queries pull in
   that is not the defect class.
4. What the harvest's shape tells us. Q1 returned 415, Q1-intersect-Q2-shape returned 17,
   and N survived. Is the bottleneck the queries, the shape filter, or the reality?
5. The single most defensible sentence about this corpus.

Do not inflate. If the honest answer is "this corpus is too small to support any claim
about the field", say exactly that.`,
  { label: 'synthesize', phase: 'Synthesize' })

return { rows, accepted: accepted.length, total: rows.length, repos: [...repos], summary }
