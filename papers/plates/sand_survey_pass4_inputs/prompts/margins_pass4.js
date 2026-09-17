export const meta = {
  name: 'sand-survey-pass4-margins',
  description: 'Sand survey pass 4 margin check: four blind readers per distance-one source re-code E1-E6 from the full text, outside the frozen protocol, reported beside the record',
  phases: [{ title: 'Margins', detail: 'four independent blind codings per distance-one source' }],
}

const PROTOCOL = args.protocol
const SOURCES = args.sources
const K = args.readers_per_source

const ELEMENT = { type: 'object', properties: { value: { type: 'boolean' }, quote: { type: 'string' } }, required: ['value', 'quote'] }
const ELEMENTS = { type: 'object', properties: { E1: ELEMENT, E2: ELEMENT, E3: ELEMENT, E4: ELEMENT, E5: ELEMENT, E6: ELEMENT }, required: ['E1', 'E2', 'E3', 'E4', 'E5', 'E6'] }
const MARGIN_SCHEMA = {
  type: 'object',
  properties: {
    reader: { type: 'string' }, id: { type: 'string' }, read_end_to_end: { type: 'boolean' },
    last_line_quote: { type: 'string' }, midpoint_quote: { type: 'string' },
    object: { type: 'string' }, elements: ELEMENTS, reason: { type: 'string' },
    hardest_element: { type: 'string' }, hardest_element_why: { type: 'string' },
  },
  required: ['reader', 'id', 'read_end_to_end', 'last_line_quote', 'midpoint_quote', 'object', 'elements', 'reason', 'hardest_element', 'hardest_element_why'],
}

const RULES = `Rules of this reading (Fathom Lab prior-art survey, pass 4). First read the frozen protocol at ${PROTOCOL} in full: the definitions of E1-E6 under "The fingerprint clause's elements, defined before any fetch" are the ONLY definitions you may use.
- Read the text named below END TO END with the Read tool, using offset/limit repeatedly until the end of the file. You are not told how long the file is. To show you reached the end, return last_line_quote = the last non-empty line of the file, verbatim, and midpoint_quote = one complete line, verbatim, from the middle third of the file by line number. A script checks both against the file.
- Code each of E1-E6 true ONLY when the source does it (not when it discusses, cites or proposes it), for the source's own central comparison of a model with itself if it has one, and give a verbatim quote from the text for every element coded true; for an element coded false give the quote that shows what the source does instead, or an empty string. Quotes are checked verbatim against the file after whitespace, ligature and line-end-hyphen normalisation; do not paraphrase inside a quote.
- Apply each definition to its letter. In particular E3 requires a spread measured on identical weights in the comparison's setup AND used to grade the comparison; E4 requires a confidence or credible interval on the compared quantity, or a hypothesis test whose null is "no difference" at a stated level (an alpha, a critical value, or a reported p-value); E5 requires the compared quantity to be computed from token log-probabilities, probabilities or logits.
- Name the element whose coding you found hardest to decide, and say why in one sentence.
- Do not search the web, do not fetch anything, do not read any file other than the protocol and your text, do not write files, and do not look for anyone else's coding of this source.
- Return ONLY the structured output.`

phase('Margins')
const jobs = SOURCES.flatMap(s => Array.from({ length: K }, (_, k) => ({ s, k: k + 1 })))
log(`${SOURCES.length} sources x ${K} blind readers = ${jobs.length} codings`)
const margins = (await parallel(jobs.map(({ s, k }) => () => agent(
  `You are margin reader ${k} of ${K} for source ${s.id}, reading blind and independently. ${RULES}\n\nYour one source: ${s.id}: "${s.title}"${s.year ? ' (' + s.year + ')' : ''}; full text: ${s.text}\n\nSet reader to "margin-${s.id}-${k}" and id to "${s.id}".`,
  { label: `margin:${s.id}:${k}`, phase: 'Margins', schema: MARGIN_SCHEMA })))).filter(Boolean)
const missing = jobs.filter(j => !margins.some(m => m.reader === `margin-${j.s.id}-${j.k}`)).map(j => `${j.s.id}:${j.k}`)
if (missing.length) log(`WARNING: no coding returned for ${missing.join(', ')}`)
return { margins, missing }