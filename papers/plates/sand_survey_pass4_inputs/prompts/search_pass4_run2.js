export const meta = {
  name: 'sand-survey-pass4-search',
  description: 'Sand survey pass 4, Part B: run the twelve frozen queries on three engines, dedupe, two independent screeners per result under the frozen inclusion rule, rank and cap at 25 (no fetching or reading)',
  phases: [{ title: 'Search', detail: 'one agent per engine, twelve queries, raw responses saved' }, { title: 'Screen', detail: 'two independent screeners per batch of results, title and abstract only' }],
}

const PROTOCOL = 'C:\\Users\\heyzo\\clawd\\wt\\plate\\papers\\plates\\PROTOCOL_sand_prior_art_pass4_2026_09_15.md'
const RAW = 'C:\\Users\\heyzo\\AppData\\Local\\Temp\\claude\\C--Users-heyzo--clawdbot\\921a7fa3-f517-462f-9d4d-75165cd9c7e7\\scratchpad\\survey4\\search'
const QUERIES = [
  ['Q01', 'C4a', 'model equality testing which model is an API serving'],
  ['Q02', 'C4a', 'auditing model substitution in LLM APIs'],
  ['Q03', 'C4a', 'detecting changes in a deployed language model API over time'],
  ['Q04', 'C4a', 'detecting quantization of a served language model from its outputs'],
  ['Q05', 'C4a', 'log probability fingerprint to verify language model identity'],
  ['Q06', 'C4a', 'statistical test whether two language models are the same'],
  ['Q07', 'C4a', 'nondeterminism of LLM inference with identical weights and evaluation variance'],
  ['Q08', 'C4a', 'behavioral drift monitoring of large language models with a fixed prompt set'],
  ['Q09', 'C2', 'tamper-evident hash chained log of machine learning evaluation results'],
  ['Q10', 'C2', 'transparency log of model evaluations or AI claims'],
  ['Q11', 'C5', 'bounty for reproducing or refuting published machine learning results'],
  ['Q12', 'C5', 'paid bug bounty for errors in scientific research or verification tools'],
]
const QTEXT = QUERIES.map(q => `${q[0]} (${q[1]}): ${q[2]}`).join('\n')

const SEARCH_SCHEMA = {
  type: 'object',
  properties: {
    engine: { type: 'string' },
    fetched_at: { type: 'string' },
    raw_files: { type: 'array', items: { type: 'string' } },
    results: { type: 'array', items: { type: 'object', properties: {
      query_id: { type: 'string' }, rank: { type: 'integer' }, title: { type: 'string' }, authors: { type: 'string' },
      year: { type: ['integer', 'null'] }, url: { type: 'string' }, abstract: { type: 'string' } },
      required: ['query_id', 'rank', 'title', 'url'] } },
    failures: { type: 'array', items: { type: 'object', properties: { query_id: { type: 'string' }, error: { type: 'string' } }, required: ['query_id', 'error'] } },
    notes: { type: 'string' },
  },
  required: ['engine', 'fetched_at', 'results', 'failures'],
}

const COMMON = `You are running one engine of the frozen search in pass 4 of Fathom Lab's prior-art survey. First read the frozen protocol at ${PROTOCOL} (section "Part B — the search"). Do exactly what it says for your engine and nothing more: each of the twelve queries below, sent once, first ten results, verbatim query text. Do not add, reword, translate or drop a query; do not open results; do not judge relevance — screening is someone else's job. Save every raw response under ${RAW}\\<engine>\\ (create the directories), one file per query named Q01.<ext> .. Q12.<ext>, and list them in raw_files. Record fetched_at as the UTC time you started (date -u +%Y-%m-%dT%H:%M:%SZ). A query that fails after the retries below is a failure record, never a silent omission. On this Windows box the Bash tool's heredocs collapse backslashes: write any Python you need to a file with the Write tool and run the file. Return ONLY the structured output, results in query order and rank order (rank 1 = first result).

Queries:
${QTEXT}`

const ENGINES = [
  { key: 'arxiv', prompt: `${COMMON}

ENGINE: arXiv API. For each query build search_query as all:W1+AND+all:W2+AND+... over the query's words exactly as written (split on spaces; keep hyphenated words whole; URL-encode each word), and fetch with Bash: curl -s "http://export.arxiv.org/api/query?search_query=<that>&start=0&max_results=10&sortBy=relevance&sortOrder=descending" -o ${RAW}\\arxiv\\Qnn.xml. Wait 3 seconds between requests (arXiv's rule). If a response is not valid Atom XML, retry up to 3 times with 5 s waits, then record a failure. Zero entries is a valid result, not a failure: record it in notes. Parse the saved XML with a Python script (xml.etree): title (whitespace collapsed), authors joined by '; ', year from <published>, url = the entry id (abs link), abstract = <summary> (whitespace collapsed).` },
  { key: 's2', prompt: `${COMMON}

ENGINE: Semantic Scholar Graph API. For each query fetch with Bash: curl -s "https://api.semanticscholar.org/graph/v1/paper/search?query=<URL-encoded query>&limit=10&fields=title,authors,year,abstract,externalIds,url,venue" -o ${RAW}\\s2\\Qnn.json. The unauthenticated API rate-limits: wait 4 seconds between requests; on HTTP 429 or a JSON body with "code":"429" or no "data", wait 15, 30, 60 seconds and retry (up to 5 tries), then record a failure. Parse the saved JSON with a Python script: title, authors (names joined by '; '), year, url (prefer https://arxiv.org/abs/<ArXiv id> from externalIds when present, else the returned url), abstract (empty string when null).` },
  { key: 'web', prompt: `${COMMON}

ENGINE: general web search. Use the WebSearch tool once per query with the query text verbatim, no domain filters. Take the first ten results in the order the tool returns them. Save each tool response's text to ${RAW}\\web\\Qnn.md with the Write tool before parsing it. For each result record title, url, and as abstract whatever snippet or description the tool returned for it (empty string if none); authors empty and year null unless the tool's text states them. Do not open any result.` },
]

const SCREEN_SCHEMA = {
  type: 'object',
  properties: {
    screener: { type: 'string' },
    decisions: { type: 'array', items: { type: 'object', properties: {
      key: { type: 'string' }, include: { type: 'boolean' }, clause: { type: ['string', 'null'] },
      excluded_as: { type: ['string', 'null'] }, reason: { type: 'string' } },
      required: ['key', 'include', 'reason'] } },
  },
  required: ['screener', 'decisions'],
}

const SCORED = `Already sources in passes 1 to 3 (exclude if a result is one of these, including another version or edition of it): Hash Visualization (Perrig & Song 1999); OpenSSH 5.1 release notes; Identicon / 9-block IP identification (Don Park); How To Time-Stamp a Digital Document (Haber & Stornetta); OpenTimestamps; Proof of Existence; Certificate Transparency RFC 6962; Rekor / Sigstore transparency log; The past, present and future of Registered Reports; COS Preregistration (Challenge); How Is ChatGPT's Behavior Changing over Time? (Chen, Zaharia & Zou 2023); Accuracy is Not All You Need (Dutta et al. 2024); Instructional Fingerprinting of Large Language Models; REEF: Representation Encoding Fingerprints; Representational similarity analysis (Kriegeskorte 2008); Defeating Nondeterminism in LLM Inference (Thinking Machines); A Sober Look at Progress in Language Model Reasoning; Improving Reproducibility in Machine Learning Research (Pineau et al.); Proof-of-Learning; Immunefi bug bounty programs; Did the Model Change? Efficiently Assessing Machine Learning API Shifts (Chen, Cai, Zaharia & Zou) and its ICLR version How Did the Model Change?; HAPI: A Large-scale Longitudinal Dataset of Commercial ML API Predictions; ChatLog (Tu et al.); A Fingerprint for Large Language Models (Yang & Wu); Beyond Preserved Accuracy: Evaluating Loyalty and Robustness of BERT Compression (Xu et al.); What Do Compressed Deep Neural Networks Forget? (Hooker et al.); Non-Determinism of "Deterministic" LLM Settings (Atil et al.); Quantifying Variance in Evaluation Benchmarks (Madaan et al.); LLMmap; Hide and Seek: Fingerprinting LLMs with Evolutionary Learning; Dataset Inference: Ownership Resolution; Copy, Right? A Testing Framework for Copyright Protection of Deep Learning Models; SafetyNets; High Accuracy and High Fidelity Extraction of Neural Networks; Is GPT-4 getting worse over time? (Narayanan & Kapoor); ACM Artifact Review and Badging; NeurIPS 2019 Reproducibility Challenge (Sinha et al.).
Part A of this pass (not excluded, but mark excluded_as "part A: A0n" so it takes no B slot): A01 IPGuard (Cao, Jia & Gong); A02 Deep Neural Network Fingerprinting by Conferrable Adversarial Examples (Lukas et al.); A03 Logits of API-Protected LLMs Leak Proprietary Information (Finlayson et al.); A04 Chain & Hash (Russinovich & Salem); A05 Stealing Part of a Production Language Model (Carlini et al.); A06 Can We Trust the Evaluation on ChatGPT? (Aiyappa et al.); A07 Characterising Bias in Compressed Models (Hooker et al. 2020); A08 An Empirical Study of the Non-Determinism of ChatGPT in Code Generation (Ouyang et al.); A09 The Good, the Bad, and the Greedy (Song et al.).`

function screenPrompt(batch, s) {
  return `You are screener ${s + 1} of 2, working independently, in pass 4 of Fathom Lab's prior-art survey. First read the frozen protocol at ${PROTOCOL}, sections "The fingerprint clause's elements" and "Part B — the search" (the inclusion rule). Apply the inclusion rule to each result below using ONLY its title and the abstract or snippet given here: do not search, do not open links, do not use what you remember about a paper beyond its title and abstract. Include a result when it meets (a), (b), (c) or (d); exclude it when it is already a source in passes 1 to 3, when it is only about identifying which of different models produced text (authorship, watermark detection, family or ownership attribution between different models), or when it is not a paper, report or standing programme page. Doubt about (a)-(d) from a thin snippet is not a reason to exclude when the title itself plainly fits; a title that fits nothing is excluded. For each result return: key (verbatim), include, clause (C4a for (a) or (b), C2 for (c), C5 for (d), null when excluded), excluded_as (null, or 'already scored: <which>', or 'part A: A0n', or 'different-models attribution', or 'not a paper'), and one sentence of reason naming the words of the title or abstract that decided it. A Part A match is include=true with excluded_as 'part A: A0n'.

${SCORED}

Results (JSON lines: key, title, year, authors, abstract):
${batch.map(it => JSON.stringify({ key: it.key, title: it.title, year: it.year, authors: (it.authors || '').slice(0, 160), abstract: (it.abstract || '').slice(0, 1500) })).join('\n')}

Return ONLY the structured output with one decision per result, in the order given, screener "${s + 1}".`
}

phase('Search')
// The three engine agents' prompts are unchanged from run wf_fdf4dfce-ffd, so a resume returns their recorded answers
// and no answered query is sent again.
const rawSearches = await parallel(ENGINES.map(e => () => agent(e.prompt, { label: `search:${e.key}`, phase: 'Search', schema: SEARCH_SCHEMA, effort: 'medium' })))
const byEngine = {}
ENGINES.forEach((e, i) => { byEngine[e.key] = rawSearches[i] })
log(`search: ${ENGINES.map(e => byEngine[e.key] ? `${e.key} ${byEngine[e.key].results.length} results, ${byEngine[e.key].failures.length} failures` : `${e.key} NO RETURN`).join('; ')}`)
if (byEngine.arxiv && byEngine.arxiv.results.length === 0) log('arxiv: zero entries for every query under the frozen syntax (every word ANDed); recorded as the frozen procedure\'s outcome, not retried or reworded')

// Semantic Scholar: a query every try of which returned HTTP 429 received no answer. It is re-sent, slowly, until it is
// answered once or fails again; an answered query is never re-sent. Every try is saved.
const s2Failed = byEngine.s2 ? byEngine.s2.failures.map(f => f.query_id) : QUERIES.map(q => q[0])
let s2Retry = null
if (s2Failed.length) {
  const qs = QUERIES.filter(q => s2Failed.includes(q[0])).map(q => `${q[0]}: ${q[2]}`).join('\n')
  s2Retry = await agent(`You are re-sending, for the Semantic Scholar Graph API only, the queries of pass 4 of Fathom Lab's prior-art survey that the first run could not get answered: every try returned HTTP 429 (rate limited), so no answer was received. First read the frozen protocol at ${PROTOCOL} (section "Part B — the search"). Do not add, reword or drop a query; do not open results; do not judge relevance.

Queries to re-send (verbatim):
${qs}

For each query, the same request as the first run: curl -s -w "%{http_code}" "https://api.semanticscholar.org/graph/v1/paper/search?query=<URL-encoded query>&limit=10&fields=title,authors,year,abstract,externalIds,url,venue". Rules: at least 70 seconds between any two requests; on HTTP 429, or a body without "data", wait 60, 90, 120 and 150 seconds and retry (at most 5 tries per query); the moment one response carries "data", that response is the query's answer and the query is never sent again. Save every try's body to ${RAW}\\s2_retry\\Qnn.tryK.json (K = 1..5, never overwrite) and the answer also to ${RAW}\\s2_retry\\Qnn.json; list every saved file in raw_files. A single Bash call is capped at 10 minutes, so write ONE Python script with the Write tool (Bash heredocs collapse backslashes on this Windows box) that takes one query id, sleeps 70 seconds, then does that query's tries, and run it once per query in its own Bash call with timeout 600000. Parse each answer as the first run did: title, authors (names joined by '; '), year, url (https://arxiv.org/abs/<ArXiv id> from externalIds when present, else the returned url), abstract (empty string when null), rank = position in data. A query still unanswered after 5 tries is a failure record whose error lists each try's UTC time and HTTP code. engine = "s2-retry"; fetched_at = the UTC time you started; in notes, one line per try: query id, try, UTC time, HTTP code. Return ONLY the structured output.`, { label: 'search:s2-retry', phase: 'Search', schema: SEARCH_SCHEMA, effort: 'medium' })
  log(`s2 retry of ${s2Failed.join(', ')}: ${s2Retry ? `${s2Retry.results.length} results, ${s2Retry.failures.length} still failing` : 'NO RETURN'}`)
}
const s2Merged = {
  engine: 's2', fetched_at: byEngine.s2 ? byEngine.s2.fetched_at : null,
  results: [...(byEngine.s2 ? byEngine.s2.results : []), ...(s2Retry ? s2Retry.results.filter(r => s2Failed.includes(r.query_id)) : [])],
  failures: s2Retry ? s2Retry.failures.filter(f => s2Failed.includes(f.query_id)) : (byEngine.s2 ? byEngine.s2.failures : QUERIES.map(q => ({ query_id: q[0], error: 'engine agent returned nothing' }))),
}
const searches = [
  { key: 'arxiv', ret: byEngine.arxiv }, { key: 's2', ret: s2Merged }, { key: 'web', ret: byEngine.web },
].map(s => ({ engine: s.key, results: s.ret ? s.ret.results : [], failures: s.ret ? s.ret.failures : QUERIES.map(q => ({ query_id: q[0], error: 'engine agent returned nothing' })) }))

// Merge — plain code, recorded. One result is one work: results are joined when they share an arXiv identifier (from a
// link or from the title) or the same title once a leading [identifier], a trailing (arXiv:identifier) and a trailing
// site label (" | OpenReview", " - Papers with Code", ...) are removed and case and punctuation are ignored. The run
// before this one joined on the raw title only, which let one paper take several slots under its mirror titles.
const DEDUPE_RULE = 'join results sharing an arXiv identifier (link or title) or the same cleaned title (leading [id], trailing (arXiv:id) and a trailing site label of at most six words after | - – — · removed when at least four words remain; a leading (PDF) removed; NFKC; lowercased, non-alphanumerics collapsed), or where one cleaned title of at least three words is a whole-word prefix of another; transitive'
const ARXIV_IN = /(?:arxiv\.org\/(?:abs|pdf|html)\/|\/papers?\/|arxiv[:\s]\s*|\[)(\d{4}\.\d{4,5})(?:v\d+)?/i
const arxivId = r => { for (const s of [r.url || '', r.title || '']) { const m = s.match(ARXIV_IN); if (m) return m[1] } return null }
const displayTitle = t => {
  let s = (t || '').normalize('NFKC').replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim()
  s = s.replace(/^\[[^\]]*\]\s*/, '').replace(/^\(PDF\)\s*/i, '').replace(/\s*\((?:arxiv:\s*)?\d{4}\.\d{4,5}(?:v\d+)?\)\s*/gi, ' ').trim()
  for (let k = 0; k < 2; k++) {
    const m = s.match(/^(.*\S)\s+(?:\||-|–|—|·|\uFFFD)\s+([^|–—·\uFFFD]{1,80})$/)
    if (m && m[1].split(/\s+/).length >= 4 && m[2].trim().split(/\s+/).length <= 6) s = m[1].trim()
    else break
  }
  return s
}
const cleanTitle = t => displayTitle(t).toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim()
const recs = []
for (const s of searches) for (const r of s.results) recs.push({ engine: s.engine, r })
const parent = recs.map((_, i) => i)
const root = i => { while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i] } return i }
const firstByKey = {}
recs.forEach((x, i) => {
  const id = arxivId(x.r), t = cleanTitle(x.r.title)
  x.keys = [id ? `arxiv:${id}` : null, t ? `t:${t}` : null].filter(Boolean)
  for (const k of x.keys) { if (k in firstByKey) parent[root(i)] = root(firstByKey[k]); else firstByKey[k] = i }
})
// a cleaned title of at least three words that is a whole-word prefix of another (a title with its authors or its
// subtitle appended by a web page) joins it
const titled = recs.map((x, i) => ({ i, t: cleanTitle(x.r.title) })).filter(y => y.t)
for (const a of titled) for (const b of titled) {
  if (a.i !== b.i && a.t.split(' ').length >= 3 && b.t.length > a.t.length && b.t.startsWith(a.t + ' ')) parent[root(b.i)] = root(a.i)
}
const groups = {}
recs.forEach((x, i) => { if (x.keys.length) (groups[root(i)] = groups[root(i)] || []).push(x) })
const items = Object.values(groups).map(g => {
  const withId = g.map(x => arxivId(x.r)).find(Boolean)
  const scholarly = g.find(x => x.engine !== 'web') || g[0]
  const it = { key: withId ? `arxiv:${withId}` : g[0].keys.find(k => k.startsWith('t:')), title: displayTitle(scholarly.r.title), authors: '', year: null, urls: [], abstract: '', hits: [] }
  for (const x of g) {
    it.hits.push({ engine: x.engine, query_id: x.r.query_id, rank: x.r.rank, title_as_returned: x.r.title, url: x.r.url || '' })
    if (x.r.url && !it.urls.includes(x.r.url)) it.urls.push(x.r.url)
    if ((x.r.abstract || '').length > it.abstract.length) it.abstract = x.r.abstract
    if (!it.authors && x.r.authors) it.authors = x.r.authors
    if (it.year == null && x.r.year != null) it.year = x.r.year
  }
  return it
})
log(`${recs.length} results merged into ${items.length} distinct works to screen`)

phase('Screen')
const BATCH = 35
const batches = []
for (let i = 0; i < items.length; i += BATCH) batches.push(items.slice(i, i + BATCH))
const screenRuns = await parallel(batches.flatMap((b, bi) => [0, 1].map(s => () =>
  agent(screenPrompt(b, s), { label: `screen:b${bi}:s${s + 1}`, phase: 'Screen', schema: SCREEN_SCHEMA, effort: 'medium' }).then(r => r && { bi, s, r }))))
const decisions = {}
for (const run of screenRuns.filter(Boolean)) {
  for (const d of run.r.decisions) {
    decisions[d.key] = decisions[d.key] || {}
    decisions[d.key][`screener${run.s + 1}`] = d
  }
}
const unscreened = items.filter(it => !decisions[it.key] || !decisions[it.key].screener1 || !decisions[it.key].screener2).map(it => it.key)
if (unscreened.length) log(`WARNING: ${unscreened.length} results lack a decision from one or both screeners`)

const pairs = it => new Set(it.hits.map(h => `${h.engine}|${h.query_id}`)).size
const entered = [], partA = [], excluded = []
for (const it of items) {
  const d = decisions[it.key] || {}
  const ds = [d.screener1, d.screener2].filter(Boolean)
  const a = ds.find(x => x.include && x.excluded_as && /^part A/i.test(x.excluded_as))
  const inc = ds.some(x => x.include && !(x.excluded_as && /^part A/i.test(x.excluded_as)))
  const rec = { ...it, n_engine_query_pairs: pairs(it), screener1: d.screener1 || null, screener2: d.screener2 || null }
  if (a) partA.push({ ...rec, part_a: a.excluded_as })
  else if (inc) entered.push(rec)
  else excluded.push(rec)
}
entered.sort((x, y) => (y.n_engine_query_pairs - x.n_engine_query_pairs) || ((y.year || 0) - (x.year || 0)) || x.title.localeCompare(y.title))
const listB = entered.slice(0, 25).map((it, i) => ({ id: `B${String(i + 1).padStart(2, '0')}`, ...it }))
const belowCap = entered.slice(25).map((it, i) => ({ rank_below_cap: i + 26, ...it }))
log(`included ${entered.length}; list B ${listB.length}; below cap ${belowCap.length}; part A found by search ${partA.length}; excluded ${excluded.length}`)
return { queries: QUERIES, searches, raw_engine_returns: { arxiv: byEngine.arxiv || null, s2: byEngine.s2 || null, s2_retry: s2Retry, web: byEngine.web || null }, dedupe_rule: DEDUPE_RULE, n_results: recs.length, n_distinct: items.length, unscreened, list_b: listB, below_cap: belowCap, part_a_found_by_search: partA, excluded }