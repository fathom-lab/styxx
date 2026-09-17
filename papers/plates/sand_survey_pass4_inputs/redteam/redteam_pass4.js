export const meta = {
  name: 'redteam-sand-survey-pass4',
  description: 'Red-team sand survey pass 4 (pushed 3060ffd2): seven adversarial finders across prose, builder, search, fetch, timeline and blind re-coding, each finding attacked by skeptics',
  phases: [
    { title: 'Find', detail: 'seven independent finders try to break pass 4' },
    { title: 'Verify', detail: 'skeptics try to refute each finding' },
  ],
}

const REPO = 'C:\\Users\\heyzo\\clawd\\wt\\plate-rep'
const SCRATCH = 'C:\\Users\\heyzo\\AppData\\Local\\Temp\\claude\\C--Users-heyzo--clawdbot\\921a7fa3-f517-462f-9d4d-75165cd9c7e7\\scratchpad'
const TEXTS = SCRATCH + '\\survey4\\texts'
const RAW = SCRATCH + '\\survey4\\search'
const WF = 'C:\\Users\\heyzo\\.claude\\projects\\C--Users-heyzo--clawdbot\\921a7fa3-f517-462f-9d4d-75165cd9c7e7\\subagents\\workflows'
const OUT = SCRATCH + '\\redteam_pass4'

const CONTEXT = `You are red-teaming pass 4 of Fathom Lab's prior-art survey of "the sand" (styxx repository, git worktree ${REPO}, HEAD 3060ffd2, branch plate/the-plate-and-checksum, PR #87). The lab's rule: red-team your own day; every earlier round found a blocker. Your job is to find real defects, not to praise.

What pass 4 is (read these files yourself; do not trust this summary):
- Frozen protocol: papers/plates/PROTOCOL_sand_prior_art_pass4_2026_09_15.md (commit 5d7f39ef). It inherits papers/plates/PROTOCOL_sand_prior_art_2026_09_13.md and papers/plates/PROTOCOL_sand_prior_art_pass3_2026_09_14.md.
- Sworn document: papers/plates/SURVEY_sand_neighbours_pass4_2026_09_14.md (sworn 50/50; receipt .sworn-receipt.json, sidecar .sworn.json). Sworn documents are NEVER edited after their receipt is committed; a defect in one is repaired by a CORRECTION beside it.
- Record: papers/plates/sand_prior_art_survey_pass4_2026_09_14.json, built by papers/plates/build_sand_survey_pass4.py (tests: tests/test_build_sand_survey_pass4.py). Margin record: papers/plates/sand_prior_art_survey_pass4_margins_2026_09_14.json, built by papers/plates/build_sand_survey_pass4_margins.py.
- Inputs: papers/plates/sand_survey_pass4_inputs/ (fetch_pass4.py, record_search.py, search_record.json, list.json, part_a_located.json, fetch_record_run_{A,B,B2,B3}.json, list_B_located_retry.json, list_B08_located_archive.json, fetched_text_scope.json, readings_run_{A,B1,B2,B3}.json, margins_run.json, prompts/*.js).
- Pass 3 record for comparison: papers/plates/sand_prior_art_survey_pass3_2026_09_14.json and its SURVEY, CORRECTION and ERRATUM beside it.
- Operator report addendum: the last section of papers/chat/REPORT_2026_09_13.md. CHANGELOG.md top entry "2026-09-14, later still". PR comment text: ${SCRATCH}\\pr87_comment_pass4.md.
- Extracted full texts (not committed): ${TEXTS}\\<id>.txt. Raw search engine responses: ${RAW}. Workflow journals and per-agent transcripts with timestamps: ${WF}\\wf_* (search wf_fdf4dfce-ffd; readings wf_64806a5e-5d1 (A), wf_d4cb8fe1-186 (B1), wf_947fdf1e-e39 (B2), wf_f9482e9f-e5f (B3); margins wf_0e77527a-e2b). git log in the repo has commit times.
- The charter: never "first", "novel", "revolutionary", "tamper-proof" or "self-verifying" as a claim; every positioning claim cites a receipt.

Rules for you: do NOT edit, create, commit or push anything in the repository. Write any scratch scripts or outputs only under ${OUT} (create it). On this Windows box Bash heredocs collapse backslashes: write scripts with the Write tool and run them. Set PYTHONIOENCODING=utf-8 when running Python. A finding needs evidence a skeptic can re-check: file paths with JSON pointers or line numbers, commands you ran and their output, quotes. Severity: blocker = a sworn or public statement is false, or a status/sentence in the record is wrong under the frozen rules; major = a real defect that weakens a result or a disclosure without flipping it; minor = everything else worth fixing. Report nothing you could not evidence. If you find no defect in your dimension, return an empty findings list and say what you checked in notes.`

const FINDINGS = {
  type: 'object',
  properties: {
    dimension: { type: 'string' },
    checked: { type: 'string' },
    findings: { type: 'array', items: { type: 'object', properties: {
      title: { type: 'string' }, severity: { type: 'string', enum: ['blocker', 'major', 'minor'] },
      claim: { type: 'string' }, evidence: { type: 'string' }, affected: { type: 'array', items: { type: 'string' } },
      sworn_document_affected: { type: 'boolean' }, suggested_repair: { type: 'string' },
    }, required: ['title', 'severity', 'claim', 'evidence', 'affected', 'sworn_document_affected', 'suggested_repair'] } },
  },
  required: ['dimension', 'checked', 'findings'],
}
const VERDICT = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['CONFIRMED', 'PLAUSIBLE', 'REFUTED'] },
    severity_if_real: { type: 'string', enum: ['blocker', 'major', 'minor'] },
    reason: { type: 'string' }, evidence: { type: 'string' },
  },
  required: ['verdict', 'severity_if_real', 'reason', 'evidence'],
}

const DIMENSIONS = [
  { key: 'prose', prompt: `DIMENSION: the words. Check every factual statement in the pass-4 SURVEY that is NOT inside a <sworn> span, and every statement in the REPORT addendum, the CHANGELOG entry and the PR comment, against the records, inputs, git history and texts. Examples to test, not a complete list: "Two screeners decided every work"; "five 2026 intrusion-detection papers" on the list; "Eight were located"; "has no archive capture"; "All twelve passed both proofs"; "each written before the result it could have bent" (commits 450265ff..afc49486); "the citation trail of passes 1 to 3 never reached them" (search pass 1-3 records and SURVEYs for these works by title, arXiv id and author); "None of the nine leads is nearer than distance 2"; "No source in any pass is at distance one without the self-comparison or the fixed item set"; the descriptions of what Gao et al., DiFR and Chauvin et al. do; "None chains verdicts re-derived from bytes" for C2; the claim that the ERROR programme and "two Nature pieces" occupy C5 (check which sources those are); "It is shorter than pass 3's for the first time"; the sentence copied verbatim from the record; the stranger command. Also check the charter words and whether any public wording overstates the result.` },
  { key: 'builder', prompt: `DIMENSION: the builder against the frozen rules. Read build_sand_survey_pass4.py and build_sand_survey_pass4_margins.py line by line against the pass-4 protocol and the inherited 2026-09-13 status and sentence rules. Look for any rule the code applies differently from the text: the RETIRES/DISPUTED/UNPRICED logic; SKIMMED handling (the inherited protocol lets a skimmed source be SILENT or OCCUPIES "from abstract"); UNFETCHABLE/UNCHECKABLE; UNSCORED; the nearness rule ("the listed source (from any pass) at the smallest distance whose only missing element is that one ... at most five names"); the fallback when no distance-one source exists; how pass-3 element keys map onto E1-E5 and whether pass 3's coding under undefined terms should enter the pool at all; the sentence rules (C4a/C5 RETIRED, UNPRICED deletion, the protocol's "a sentence without its fingerprint clause is not the sand's sentence"); the C6 conjunction rule of the 2026-09-13 protocol (does the record compute the conjunction's status at all?); the quote normalisation; the proof-of-reading checks (last three lines, midpoint by line number) and whether they can be passed without reading; the fetch-record merge; the scope file; html unescaping; counts that could be wrong. Write adversarial synthetic inputs under ${OUT} and run the builder on them (copy the pass-3 record into a temp repo layout as tests/test_build_sand_survey_pass4.py does) to demonstrate each defect. Also re-run the real build into a temp copy and confirm the committed record is byte-identical to a rebuild.` },
  { key: 'search', prompt: `DIMENSION: the search and the screen. From papers/plates/sand_survey_pass4_inputs/search_record.json alone, re-derive the list: the rank rule is "(engine, query) pairs desc, then later year, then title", cap 25. Check that B01-B25 and the below-cap ranks follow exactly (including how a null year and title ties are ordered, and whether the workflow's JS localeCompare ordering matches a reasonable reading of "title"). Audit the merge (dedupe) rule's effect: find works that were wrongly merged together, or the same work left as two entries (compare arXiv ids, cleaned titles, URLs). Check that screener decisions exist for every work, that the inclusion rule was applied consistently (look for included works that are only about identifying which of different models produced text, or that are not papers/reports/programme pages, and excluded works that plainly meet (a)-(d)), and that "Part A found by search = 0" is true (search the results for the nine Part A titles). Check the raw file hashes recorded in the search record against the files under ${RAW}. Check the Semantic Scholar handling: which queries were answered, the answered-empty Q11/Q12 claim against the first engine agent's notes in the journal, and whether any answered query was re-sent in run 2 (look at ${RAW}\\s2_retry). Check the arXiv zero-results claim against ${RAW}\\arxiv.` },
  { key: 'fetch', prompt: `DIMENSION: fetch and locate integrity. For every source in list.json, check its fetch record(s): the URL fetched, bytes sha256, the extracted text's sha256 against ${TEXTS}\\<id>.txt, and whether the text is actually the listed work (a title check at 60% of title words can pass on a different paper, an index page, or a review of the work). Check each located retry (list_B_located_retry.json, list_B08_located_archive.json): is the located copy the same work as listed (for B16 the listed source was the ERROR programme's home page and the fetched text is its About page; for B13 an archive capture; for B03-B08 DOI/PMC copies)? Check fetched_text_scope.json: are B04, B05 and B07 really landing pages without article text, and are there other fetched texts that are also only landing pages, abstracts or navigation (look at short texts and at texts whose body is mostly site chrome, e.g. B11, B12, B18, B19, B21, B24) that were not marked and could have entered as READ? Was B23 locatable by an honest open route consistent with "located by title, author and year" (the Internet Archive under another URL form, the DOI of the Science news item, the article syndicated by its publisher) WITHOUT presenting as a browser or bypassing a bot block? You may use curl with the default or the fetcher's identifying user agent and the archive.org availability API; do not spoof a browser user agent and do not try to pass any block. Also judge whether the asymmetry (only failures got a second, located fetch) could bias any status.` },
  { key: 'timeline', prompt: `DIMENSION: result-dependence. The protocol and the SURVEY claim that each rule and repair was written before the result it could bend. Build a timeline from git commit times (git log --format='%h %cI %s' 5d7f39ef..3060ffd2) and from the workflow journals and per-agent transcript timestamps under ${WF} (agent-*.jsonl lines carry timestamps; journal.jsonl result lines are appended when agents finish). For each post-freeze decision, establish what results existed when it was made: the midpoint-by-line rule (after Part A's readings came back); the requirement that the midpoint be checked at all; the fetched_text_scope.json landing-page marks (commit fe94caf7 at 18:21:03 local vs the B2 reading run's first result); the located retry fetches (after the first fetch failures, before or after any reading); the search merge fix and the Semantic Scholar retry; the decision to run a margin check and on which sources; and the margin readers' prompt (prompts/margins_pass4.js) compared with the first readers' prompt (prompts/read_pass4.js): the margin prompt adds "Apply each definition to its letter" and restates E3, E4 and E5 strictly. Could that wording, written after the first readings showed which elements were contested, have pushed margin readers toward coding E4/E5 false, and so make the margin look safer than it is? Is the SURVEY's "each written before the result it could have bent" true for every item in its list? Report every decision whose timing or wording could have bent a result, with times.` },
  { key: 'recode_a', prompt: `DIMENSION: blind re-coding, part 1. The record puts these sources at distance two on C4a: B09 ("An Auditing Test to Detect Behavioral Shift in Language Models", ICLR 2025), B10 (Cai et al., "Are You Getting What You Pay For? Auditing Model Substitution in LLM APIs") and B11 ("Behavioral Fingerprints for LLM Endpoint Stability and Identity"). Without looking at the record's or any reader's coding first, read the protocol's element definitions, then read each text end to end from ${TEXTS}\\B09.txt, B10.txt and B11.txt, and code E1-E6 for each source's central comparison of a model with itself, with verbatim quotes. Only after coding, compare with the record (sources/<id>/verdicts/C4a in the record) and report as a finding any source you code at distance one or zero, or any element where the record's quote does not support its coding, with your quotes. Also note whether B11's text is the paper or only a summary page.` },
  { key: 'recode_b', prompt: `DIMENSION: blind re-coding, part 2. Without looking at the record's coding first, read the protocol's element definitions, then read end to end ${TEXTS}\\B15.txt ("Does quantization affect models' performance on long-context tasks?"), ${TEXTS}\\A04.txt (Russinovich & Salem, Chain & Hash) and ${TEXTS}\\B18.txt ("LLM API Continuous Monitoring", an emergentmind page) and code E1-E6 for each, with verbatim quotes. Then read ${TEXTS}\\B13.txt (the morphllm page) and ${TEXTS}\\B17.txt ("Fingerprinting Inference Systems of Large Language Models") and code them too. Only after coding, compare with the record and report as a finding any source you code nearer than the record (especially distance one or zero), any SILENT that should be OCCUPIES or the reverse, and any element whose quoted evidence does not support the coding.` },
]

phase('Find')
const results = await pipeline(
  DIMENSIONS,
  d => agent(`${CONTEXT}\n\n${d.prompt}\n\nSet dimension to "${d.key}". Return ONLY the structured output.`, { label: `find:${d.key}`, phase: 'Find', schema: FINDINGS }),
  async (found, d) => {
    if (!found) return { dimension: d.key, checked: 'NO RETURN', findings: [] }
    log(`${d.key}: ${found.findings.length} findings (${found.findings.map(f => f.severity).join(', ')})`)
    const verified = await parallel(found.findings.flatMap((f, i) => {
      const n = f.severity === 'minor' ? 1 : 2
      return Array.from({ length: n }, (_, k) => () => agent(`${CONTEXT}\n\nYou are skeptic ${k + 1} for one finding from the "${d.key}" red-team dimension. Try hard to REFUTE it: re-check the evidence yourself from the files, texts and commands, look for a reading of the protocol or the record under which it is not a defect, and check whether its severity is overstated. Return CONFIRMED only if you reproduced the defect yourself, PLAUSIBLE if it may be real but you could not reproduce it, REFUTED if it is not a defect. Default to REFUTED when the evidence does not hold.\n\nFINDING:\n${JSON.stringify(f, null, 1)}\n\nReturn ONLY the structured output.`,
        { label: `verify:${d.key}:${i + 1}:${k + 1}`, phase: 'Verify', schema: VERDICT }).then(v => ({ i, v })))
    }))
    const byFinding = found.findings.map((f, i) => ({ ...f, verdicts: verified.filter(x => x && x.i === i).map(x => x.v).filter(Boolean) }))
    return { dimension: d.key, checked: found.checked, findings: byFinding }
  })
const all = results.filter(Boolean)
const flat = all.flatMap(r => r.findings.map(f => ({ dimension: r.dimension, ...f })))
const confirmed = flat.filter(f => f.verdicts.length && f.verdicts.some(v => v.verdict === 'CONFIRMED') && !f.verdicts.some(v => v.verdict === 'REFUTED'))
const contested = flat.filter(f => f.verdicts.some(v => v.verdict === 'CONFIRMED') && f.verdicts.some(v => v.verdict === 'REFUTED'))
log(`findings ${flat.length}; confirmed ${confirmed.length}; contested ${contested.length}`)
return { dimensions: all.map(r => ({ dimension: r.dimension, checked: r.checked, n: r.findings.length })), findings: flat }