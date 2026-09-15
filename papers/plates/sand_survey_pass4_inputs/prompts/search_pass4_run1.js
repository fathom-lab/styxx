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
const searches = (await parallel(ENGINES.map(e => () => agent(e.prompt, { label: `search:${e.key}`, phase: 'Search', schema: SEARCH_SCHEMA, effort: 'medium' })))).filter(Boolean)
log(`search: ${searches.map(s => `${s.engine} ${s.results.length} results, ${s.failures.length} failures`).join('; ')}`)

// dedupe by normalised title — plain code, recorded
const norm = t => (t || '').toLowerCase().replace(/<[^>]+>/g, ' ').replace(/[^a-z0-9]+/g, ' ').trim()
const byKey = {}
for (const s of searches) {
  for (const r of s.results) {
    const k = norm(r.title)
    if (!k) continue
    if (!byKey[k]) byKey[k] = { key: k, title: r.title, authors: r.authors || '', year: r.year ?? null, urls: [], abstract: '', hits: [] }
    const it = byKey[k]
    it.hits.push({ engine: s.engine, query_id: r.query_id, rank: r.rank })
    if (r.url && !it.urls.includes(r.url)) it.urls.push(r.url)
    if ((r.abstract || '').length > it.abstract.length) it.abstract = r.abstract
    if (!it.authors && r.authors) it.authors = r.authors
    if (it.year == null && r.year != null) it.year = r.year
  }
}
const items = Object.values(byKey)
log(`${items.length} distinct results to screen`)

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
return { queries: QUERIES, searches, n_distinct: items.length, unscreened, list_b: listB, below_cap: belowCap, part_a_found_by_search: partA, excluded }