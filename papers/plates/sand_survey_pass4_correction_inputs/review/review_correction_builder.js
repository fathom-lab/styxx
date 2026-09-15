export const meta = {
  name: 'review-pass4-correction-builder',
  description: 'Adversarially review the pass-4 correction builder and its input preparation against the frozen correction protocol, before the builder runs on real returns',
  phases: [{ title: 'Find', detail: 'three finders' }, { title: 'Verify', detail: 'skeptics per finding' }],
}

const REPO = 'C:\\Users\\heyzo\\clawd\\wt\\plate-rep'
const SCRATCH = 'C:\\Users\\heyzo\\AppData\\Local\\Temp\\claude\\C--Users-heyzo--clawdbot\\921a7fa3-f517-462f-9d4d-75165cd9c7e7\\scratchpad'
const OUT = SCRATCH + '\\review_correction_builder'

const CONTEXT = `You are reviewing, adversarially, code the Fathom Lab wrote to correct pass 4 of its prior-art survey of "the sand" (styxx repository, worktree ${REPO}, branch plate/the-plate-and-checksum). A red team found pass 4's builder applied rules the frozen protocol does not say; the lab wrote a correction protocol and a new builder. Your job is to find where the new code still departs from its frozen protocol, or is wrong, before it runs on real reader returns.

Read yourself:
- The correction protocol (frozen, pushed at 9dbf4092): papers/plates/PROTOCOL_sand_pass4_correction_2026_09_15.md
- The pass-4 protocol it builds on: papers/plates/PROTOCOL_sand_prior_art_pass4_2026_09_15.md, and the inherited papers/plates/PROTOCOL_sand_prior_art_2026_09_13.md
- The new builder: papers/plates/build_sand_survey_pass4_correction.py, and the pass-4 builder it imports from: papers/plates/build_sand_survey_pass4.py
- Its tests: tests/test_build_sand_survey_pass4_correction.py
- The input preparation: papers/plates/sand_survey_pass4_correction_inputs/prepare_correction_inputs.py and its outputs beside it (list_correction.json, fetch_record_correction.json, fetched_text_scope_correction.json); texts under ${SCRATCH}\\survey4c\\texts and pass-4 texts under ${SCRATCH}\\survey4\\texts
- The red team's findings the correction answers: papers/plates/sand_survey_pass4_inputs/redteam/redteam_return.json (see "summary")
- The pass-4 inputs: papers/plates/sand_survey_pass4_inputs/ and the pass-2/pass-3 records papers/plates/sand_prior_art_survey_pass2_2026_09_13.json and ..._pass3_2026_09_14.json

Rules for you: do NOT edit, create, commit or push anything in the repository. Scratch files only under ${OUT} (create it). Bash heredocs collapse backslashes on this Windows box: write scripts with the Write tool. Use PYTHONIOENCODING=utf-8. Evidence must be re-checkable: line numbers, commands and their output, a synthetic case that demonstrates the defect (you may copy the test file's build() helper into ${OUT} and run the builder there). Severity: blocker = the code would produce a status, count or sentence the frozen correction protocol forbids, or would refuse or crash on the real inputs; major = a real departure or defect that could matter on real returns; minor = the rest. Report nothing you cannot evidence; an empty list is a fine answer if the code holds.`

const FINDINGS = {
  type: 'object',
  properties: {
    dimension: { type: 'string' }, checked: { type: 'string' },
    findings: { type: 'array', items: { type: 'object', properties: {
      title: { type: 'string' }, severity: { type: 'string', enum: ['blocker', 'major', 'minor'] },
      claim: { type: 'string' }, evidence: { type: 'string' }, suggested_repair: { type: 'string' },
    }, required: ['title', 'severity', 'claim', 'evidence', 'suggested_repair'] } },
  },
  required: ['dimension', 'checked', 'findings'],
}
const VERDICT = {
  type: 'object',
  properties: { verdict: { type: 'string', enum: ['CONFIRMED', 'PLAUSIBLE', 'REFUTED'] }, severity_if_real: { type: 'string', enum: ['blocker', 'major', 'minor'] }, reason: { type: 'string' } },
  required: ['verdict', 'severity_if_real', 'reason'],
}

const DIMENSIONS = [
  { key: 'rules', prompt: 'DIMENSION: rule by rule. For each of the correction protocol\'s eight builder rules (and every rule it marks new), find the code that applies it and test whether it applies exactly that rule and nothing else. Build synthetic cases for the edge of each rule: a SKIMMED all-five C4a coding; a C4a RETIRES not carrying E1-E5; a C2 RETIRES on a READ source with one, two identical, and two distinct confirmers; a DISPUTED on C2 only; an UNSCORED listed candidate; a source read by pass 4 and again under the correction; a correction reading without a located copy for a scope-listed source; a located copy that failed; distance-one ties across elements; no distance-one source; exactly five and six names; leads that name a listed source by a short (2-word) title; the conjunction when C3 and C4b are retired. Check that the builder never reads pass-3 element flags, and that counts match what the protocol says will be recorded.' },
  { key: 'real', prompt: `DIMENSION: the real inputs. The correction reader returns do not exist yet. Run the builder into a temp copy of the repository layout on the real committed pass-4 inputs with a correction inputs directory holding the real list_correction.json, fetch_record_correction.json and fetched_text_scope_correction.json and an EMPTY readings file ({"readings": [], "confirmations": []}), with texts ${SCRATCH}\\survey4\\texts and ${SCRATCH}\\survey4c\\texts. Confirm it runs and explain every difference between its output and the committed pass-4 record papers/plates/sand_prior_art_survey_pass4_2026_09_14.json (counts, statuses, nearness, sentence), classifying each difference as an intended consequence of a correction rule or a defect. Check especially: B11 and B12 now SKIMMED; B04, B05, B07, B11 whose located copies exist but have no correction reading yet (they must not be READ from the landing page, and the earlier pass-4 reading must still be in force); the 23 re-coded sources with no reading (UNREAD, and what that does to C4a's status — the protocol's rule 4 prices an unscored listed candidate as UNPRICED: is listing the 23 as C4a candidates going to unprice C4a if any of them is left unread?); the nearness pool without pass-3 flags; the conjunction; the leads.` },
  { key: 'prepare', prompt: `DIMENSION: the input preparation. Check prepare_correction_inputs.py and its outputs against the correction protocol's sections "Scope: text that is not the article" and "Re-coding the earlier passes' fingerprint sources": every earlier source verified against the right recorded hash (pass 2's fulltext_pdf_sha256 for PDFs, sha256 for HTML; pass 3's fulltext.sha256), the extracted text being the listed work and complete (compare character counts with the pass-2 record's chars_read and pass 3's fulltext.chars), no source silently skipped; the located copies for B04, B05, B07, B11 being the listed works' article text and not landing pages again (compare with the pass-4 texts), B04's handling after the same-page refusal, B12 and B23 recorded as the table says, and whether any located route presented as a browser or passed a block. Also check whether the protocol's table routes were all tried (B04: citation_pdf_url and PubMed Central by PMID; B23: DOIs, Europe PMC, archive captures).` },
]

phase('Find')
const results = await pipeline(
  DIMENSIONS,
  d => agent(`${CONTEXT}\n\n${d.prompt}\n\nSet dimension to "${d.key}". Return ONLY the structured output.`, { label: `find:${d.key}`, phase: 'Find', schema: FINDINGS }),
  async (found, d) => {
    if (!found) return { dimension: d.key, checked: 'NO RETURN', findings: [] }
    log(`${d.key}: ${found.findings.length} findings`)
    const vs = await parallel(found.findings.flatMap((f, i) => Array.from({ length: f.severity === 'minor' ? 1 : 2 }, (_, k) => () =>
      agent(`${CONTEXT}\n\nYou are skeptic ${k + 1} for one finding from the "${d.key}" review. Try hard to REFUTE it: reproduce it yourself, look for a reading of the correction protocol under which it is not a defect, and check whether its severity is overstated. CONFIRMED only if you reproduced it; REFUTED when the evidence does not hold.\n\nFINDING:\n${JSON.stringify(f, null, 1)}\n\nReturn ONLY the structured output.`,
        { label: `verify:${d.key}:${i + 1}:${k + 1}`, phase: 'Verify', schema: VERDICT }).then(v => ({ i, v })))))
    return { dimension: d.key, checked: found.checked, findings: found.findings.map((f, i) => ({ ...f, verdicts: vs.filter(x => x && x.i === i).map(x => x.v).filter(Boolean) })) }
  })
return { results: results.filter(Boolean) }