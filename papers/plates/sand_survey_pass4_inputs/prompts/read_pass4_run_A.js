export const meta = {
  name: 'sand-survey-pass4-read',
  description: 'Sand survey pass 4: read every listed source end to end under the frozen definitions, code E1-E6 with quotes, and have two blind confirmers re-read any source that would retire the fingerprint clause',
  phases: [{ title: 'Read', detail: 'readers in batches; elements coded with verbatim quotes and proof of reading' }, { title: 'Confirm', detail: 'two blind confirmers per candidate retirement' }],
}

// args: { protocol: <path>, sources: [{ id, title, who, year, text: <path to extracted text>, might_occupy: ['C4a'|'C2'|'C5', ...] }] }
const PROTOCOL = args.protocol
const SOURCES = args.sources

const ELEMENT = { type: 'object', properties: { value: { type: 'boolean' }, quote: { type: 'string' } }, required: ['value', 'quote'] }
const ELEMENTS = { type: 'object', properties: { E1: ELEMENT, E2: ELEMENT, E3: ELEMENT, E4: ELEMENT, E5: ELEMENT, E6: ELEMENT }, required: ['E1', 'E2', 'E3', 'E4', 'E5', 'E6'] }
const READ_SCHEMA = {
  type: 'object',
  properties: {
    reader: { type: 'string' },
    sources: { type: 'array', items: { type: 'object', properties: {
      id: { type: 'string' },
      read_end_to_end: { type: 'boolean' },
      last_line_quote: { type: 'string' },
      midpoint_quote: { type: 'string' },
      verdicts: { type: 'array', items: { type: 'object', properties: {
        clause: { type: 'string', enum: ['C2', 'C4a', 'C5'] },
        verdict: { type: 'string', enum: ['RETIRES', 'OCCUPIES', 'SILENT'] },
        object: { type: 'string' }, reason: { type: 'string' }, neighbour_phrase: { type: 'string' },
        elements: ELEMENTS,
      }, required: ['clause', 'verdict', 'object', 'reason', 'neighbour_phrase'] } },
      bearing_quotes: { type: 'array', items: { type: 'string' } },
      leads: { type: 'array', items: { type: 'string' } },
      notes: { type: 'string' },
    }, required: ['id', 'read_end_to_end', 'last_line_quote', 'midpoint_quote', 'verdicts', 'bearing_quotes', 'leads', 'notes'] } },
  },
  required: ['reader', 'sources'],
}
const CONFIRM_SCHEMA = {
  type: 'object',
  properties: {
    confirmer: { type: 'string' }, id: { type: 'string' }, read_end_to_end: { type: 'boolean' },
    last_line_quote: { type: 'string' }, midpoint_quote: { type: 'string' },
    object: { type: 'string' }, elements: ELEMENTS, reason: { type: 'string' },
  },
  required: ['confirmer', 'id', 'read_end_to_end', 'last_line_quote', 'midpoint_quote', 'object', 'elements', 'reason'],
}

const RULES = `Rules of this reading (Fathom Lab prior-art survey, pass 4). First read the frozen protocol at ${PROTOCOL} in full: the definitions of E1-E6 under "The fingerprint clause's elements, defined before any fetch" are the ONLY definitions you may use, and "RETIRES / OCCUPIES / SILENT" is the pricing rule of the earlier protocols it inherits (RETIRES = the source does the same thing for the same object; OCCUPIES = the same thing for a narrower or different object; SILENT = it does not address the clause).
- Read every text named below END TO END with the Read tool, using offset/limit repeatedly until the end of the file. You are not told how long any file is. To show you reached the end, return last_line_quote = the last non-empty line of the file, verbatim, and midpoint_quote = one complete line, verbatim, from the middle third of the file. A script checks both against the file.
- For C4a, code each of E1-E6 true ONLY when the source does it (not when it discusses, cites or proposes it), and give a verbatim quote from the text for every element coded true; for an element coded false give the quote that shows what the source does instead, or an empty string. Quotes are checked verbatim against the file after whitespace, ligature and line-end-hyphen normalisation; do not paraphrase inside a quote.
- A C4a verdict is RETIRES only when you code E1-E5 all true for the same object (a served model's behaviour against its own earlier or differently-served self). Otherwise OCCUPIES if the source does part of it, SILENT if nothing.
- For C2 (a chained log of re-derived verdicts) and C5 (a standing payment to a stranger whose record shows the lab's own verifier disagreeing with the lab), give a verdict with object, reason and neighbour_phrase; elements may be omitted.
- neighbour_phrase: at most 20 words, of the form "<Authors> <do what> <on what>", accurate to the source, usable inside the lab's positioning sentence.
- Do not search the web, do not fetch anything, do not read any file other than the protocol and your texts, do not write files. Works the text cites that look nearer go into leads, as printed, and are not read.
- Return ONLY the structured output.`

function readerPrompt(batch, i) {
  return `You are reader-${i + 1}. ${RULES}\n\nYour sources:\n${batch.map(s => `- ${s.id}: ${s.who ? s.who + ', ' : ''}"${s.title}"${s.year ? ' (' + s.year + ')' : ''}; might occupy ${s.might_occupy.join(', ')}; full text: ${s.text}`).join('\n')}\n\nReturn one entry in sources for each id above, with one verdict per clause it might occupy (and C4a always, even if SILENT). Set reader to "reader-${i + 1}".`
}

function confirmPrompt(s, k) {
  return `You are confirmer-${k} for source ${s.id}. You read blind: nobody's coding of this source is shown to you, and you must not look for one. ${RULES}\n\nYour one source: ${s.id}: ${s.who ? s.who + ', ' : ''}"${s.title}"${s.year ? ' (' + s.year + ')' : ''}; full text: ${s.text}\n\nCode E1-E6 for C4a under the protocol's definitions, strictly, with quotes, and give the object in one sentence and one sentence of reason. Set confirmer to "confirmer-${k}" and id to "${s.id}". Return ONLY the structured output.`
}

// batches of about four sources, keeping A and B sources in order
const BATCH = 4
const batches = []
for (let i = 0; i < SOURCES.length; i += BATCH) batches.push(SOURCES.slice(i, i + BATCH))
log(`${SOURCES.length} sources in ${batches.length} reader batches`)

phase('Read')
const out = await pipeline(
  batches,
  (batch, _orig, i) => agent(readerPrompt(batch, i), { label: `read:${batch.map(s => s.id).join(',')}`, phase: 'Read', schema: READ_SCHEMA }),
  async (reading, batch) => {
    if (!reading) return { batch: batch.map(s => s.id), reading: null, confirmations: [] }
    const candidates = reading.sources.filter(r => (r.verdicts || []).some(v => v.clause === 'C4a' &&
      (v.verdict === 'RETIRES' || (v.elements && ['E1', 'E2', 'E3', 'E4', 'E5'].every(e => v.elements[e] && v.elements[e].value)))))
    if (candidates.length) log(`candidate retirements to confirm: ${candidates.map(c => c.id).join(', ')}`)
    const confirmations = (await parallel(candidates.flatMap(c => {
      const src = batch.find(s => s.id === c.id)
      return src ? [1, 2].map(k => () => agent(confirmPrompt(src, k), { label: `confirm:${src.id}:${k}`, phase: 'Confirm', schema: CONFIRM_SCHEMA })) : []
    }))).filter(Boolean)
    return { batch: batch.map(s => s.id), reading, confirmations }
  })
const missing = out.filter(o => !o.reading).flatMap(o => o.batch)
if (missing.length) log(`WARNING: no reading returned for ${missing.join(', ')}`)
return { readings: out.map(o => o.reading).filter(Boolean), confirmations: out.flatMap(o => o.confirmations), unread: missing }
