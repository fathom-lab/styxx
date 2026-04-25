/**
 * styxx-scope classifier — pure-JS port of the styxx text-heuristic pipeline.
 *
 * Returns a minimal cognometric fingerprint for a piece of text:
 *   { category, confidence, K, C, D, trust, faults[] }
 *
 * Tier 3 proxy-signal pipeline (Spec v1.0 §5.1.2) — text-only,
 * no logprobs. A confidence penalty of 0.25 is applied to aggregate trust.
 *
 * Measurement architecture covered by US Provisionals 64/020,489,
 * 64/021,113, 64/026,964. Open-source browser implementation under
 * CC-BY-4.0 (methodology) + MIT (code).
 */

(function () {
  "use strict";

  // ── Pattern banks — conservative, high-precision ─────────────────
  //
  // Defenses added in v0.2 (post-adversarial-audit). See
  // _test_adversarial.js for the canonical attack suite. Each
  // hardening is annotated with the attack id it defends.
  //
  // Strategy:
  //   1. Unicode NFKD normalize before matching (defends unicode-fn-cyrillic).
  //   2. Strip insertion-punctuation (defends adv-fn-encoded "i.g.n.o.r.e").
  //   3. Match "ignore + instructions" across N tokens, not as bigram.
  //   4. Add roleplay-phrasing patterns (defends adv-fn-roleplay).
  //   5. Soft-refusal patterns (defends refusal-fn-hedged, refusal-fn-explanation).
  //   6. Meta-discussion suppressor (defends adv-fp-meta, meta-fp-cognometric-discussion).

  const HEDGING = /\b(maybe|perhaps|might|could be|possibly|i think|i believe|i'm not (sure|certain)|i don't know|i guess|seemingly|apparently|supposedly|allegedly|probably|roughly|approximately|around|about|sort of|kind of|more or less|reportedly|honestly)\b/gi;
  const REFUSAL_STRONG = /\b(i cannot|i can't|i'm unable|i will not|i won't|i must decline|i must refuse|i'm sorry,? but i|against my (guidelines|policy|values)|violates?(?: my)? (guidelines|policy|content))\b/gi;
  // Hedge-not-refusal pattern — "I cannot guarantee/promise/be certain"
  // is a HEDGE, not a decline. Count these and subtract from refusals.
  const HEDGE_NOT_REFUSAL = /\bi (cannot|can't|am unable to)\s+(guarantee|promise|be (sure|certain)|fully (verify|confirm|guarantee)|absolutely (guarantee|confirm|verify))\b/gi;
  // Soft-refusal — declines without triggering the strong patterns.
  // Fires only when the surrounding text doesn't carry confident substance.
  const REFUSAL_SOFT = /\b(would prefer not to|prefer not to engage|outside what I am comfortable|rather than helping you directly|let me suggest .{0,30} (alternative|different) approach|may be aspects of this request that fall outside)\b/gi;
  // Helpful-disclaimer pattern — suppresses refusal scoring when present.
  // Defends refusal-fp-disclaimer ("I cannot guarantee accuracy" inside helpful response).
  const HELPFUL_DISCLAIMER = /\b(happy to help|let me (give|provide) you|can give you|will (try to |)?(help|assist|provide)|here'?s? (what|how|the) (we know|i can))\b/gi;
  const CONFIDENT = /\b(the (answer|result|solution) is|it is (known|well-?known|clear|obvious|certain|established)|the fact is|definitely|certainly|undoubtedly|without (a )?doubt|precisely|exactly|specifically)\b/gi;
  // Expanded adversarial — matches "ignore...instructions" across up to 6
  // intermediate words (defeats adv-fn-encoded period-spam after stripping)
  // and adds roleplay-jailbreak phrasings.
  const ADVERSARIAL = /(\bignor[e3]\W{0,30}(previous|prior|above|all)\W{0,30}(instructions?|prompts?|configuration|settings)\b)|\b(jailbreak|system prompt|dan mode|developer mode|roleplay as)\b|\b(adopt the persona of|assume (your )?previous configuration does not apply|act (as|like) (an? )?(unrestricted|uncensored) (research )?assistant|pretend you have no (restrictions|rules|guidelines))\b/gi;
  const CREATIVE = /\b(imagine|picture (this|a)|let's pretend|once upon|fiction|storyline|narrative|protagonist|metaphor|character (set|took|walked|journeyed))\b/gi;
  // Meta-discussion — discussing patterns rather than exhibiting them.
  // Used as a SUPPRESSOR on adversarial / sycophant scores.
  const META_DISCUSSION = /\b(researchers|study|studies|paper|papers|recent (paper|study|research)|the (\w+ )?classifier (flags?|detects?|fires)|prevalence|literature|published)\b/gi;
  // Tool-flip pattern — agent revising its tool selection mid-sentence.
  const TOOL_FLIP = /\b(but actually|wait,? let me|actually let me|reconsider|on second thought|let me reconsider).{0,50}(call|use|invoke|apply)\s+\w+|\bplanned to call \w+ but\b/gi;
  const TOOL_NAMES = /\b(execute_query|delete_(records|file|rows)|drop_(table|database)|modify_(records|rows|table)|read_file|write_file|edit_file|update_(records|rows)|insert_(records|rows))\b/gi;
  // Unit-bearing numbers + years + temperatures + chemical formulas + statistical values
  const SPECIFIC_FACT = new RegExp(
    // measurements with units
    "\\b\\d+(\\.\\d+)?\\s?" +
    "(%|percent|km|cm|mm|nm|m|meters|miles|g\\/mol|grams|kg|mg|μg|ug|" +
    "dollars|usd|\\$|eur|€|pounds|£|" +
    "years?|minutes?|seconds?|hours?|days?|weeks?|months?|" +
    "ms|μs|us|ns|" +
    "million|billion|trillion|" +
    "K|°C|°F|celsius|fahrenheit|kelvin)" +
    "\\b" +
    // OR years 1000-2099
    "|\\b(1[0-9]|20)\\d{2}\\b" +
    // OR chemical formulas like H2O, C12H20O4, CO2, NaCl
    "|\\b[A-Z][a-z]?\\d*([A-Z][a-z]?\\d*){1,}\\b" +
    // OR statistical reports: p = 0.001, r = 0.847, p < 0.05, n = 150
    "|\\b[prn]\\s*[<>=]+\\s*\\d?\\.?\\d+",
    "g"
  );
  // Proper nouns / named entities (capitalized multi-word or "Dr. Name" patterns)
  const NAMED_ENTITIES = /(?:Dr\.|Prof\.|Mr\.|Ms\.|Mrs\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?|\b[A-Z][a-z]+(?:\s+(?:of|the|de|von|van)\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b/g;
  const FP_CERTAIN = /\b(i know|i can confirm|i'm (sure|certain|confident)|i've verified|the answer is|the result is)\b/gi;
  const AGREEMENT = /\b(absolutely|amazing|wonderful|brilliant|fantastic|love (it|this|that)|great (point|observation|idea|question)|completely agree|totally agree|so (thoughtful|insightful|smart)|you('re| are) (so |very )?(thoughtful|insightful|smart|right)|excellent (point|question))\b/gi;
  const DERIVATION = /\b(let's|let us|first,|second,|next,|then,|therefore|thus|hence|so we (have|get|can)|dividing|multiplying|substituting|it follows|we (obtain|derive|compute|calculate))\b/gi;

  function countMatches(re, s) {
    re.lastIndex = 0;
    const matches = s.match(re);
    return matches ? matches.length : 0;
  }

  function classifyText(raw) {
    if (!raw || raw.trim().length < 6) {
      return {
        category: "reasoning", confidence: 0.2,
        K: 0.2, C: 0.5, D: 0.2, trust: 0.50,
        faults: [], raw_stats: { wordCount: 0 }
      };
    }

    // v0.2 hardening: Unicode NFKD normalization + lookalike folding +
    // punctuation-strip pass for adversarial pattern matching only.
    // We retain the original `raw` for surface metrics, but match against
    // a normalized form for trigger detection.
    const text = raw;
    let textLower = raw.toLowerCase();
    let textNormalized = textLower;
    try {
      // Defend against unicode-fn-cyrillic and homoglyph attacks
      textNormalized = textLower.normalize("NFKD")
        .replace(/[\u0400-\u04FF]/g, c => {
          const map = { 'а': 'a', 'е': 'e', 'і': 'i', 'о': 'o', 'р': 'p', 'с': 'c', 'у': 'y', 'х': 'x' };
          return map[c] || c;
        });
    } catch (_) {}
    // Defend against adv-fn-encoded ("i.g.n.o.r.e p.r.e.v.i.o.u.s")
    // Build a punctuation-stripped variant for adversarial-only matching.
    const textStripped = textNormalized.replace(/(\w)[\.\-_·\u200B\u00B7](\w)/g, "$1$2");
    const wordCount = Math.max(1, raw.trim().split(/\s+/).length);
    const norm = 100.0 / wordCount;

    // Raw counts
    const hedges = countMatches(HEDGING, textNormalized);
    const refusals_raw = countMatches(REFUSAL_STRONG, textNormalized);
    const hedge_not_refusal = countMatches(HEDGE_NOT_REFUSAL, textNormalized);
    // Subtract hedge-pattern matches from refusal count.
    const refusals = Math.max(0, refusals_raw - hedge_not_refusal);
    const refusals_soft = countMatches(REFUSAL_SOFT, textNormalized);
    const confidents = countMatches(CONFIDENT, textNormalized);
    // Adversarial matching uses the punctuation-stripped + unicode-normalized
    // text to defeat obfuscation attacks (encoded, cyrillic).
    const adversarials = countMatches(ADVERSARIAL, textStripped) + countMatches(ADVERSARIAL, textNormalized);
    const creatives = countMatches(CREATIVE, textNormalized);
    const specifics = countMatches(SPECIFIC_FACT, text); // case-sensitive for chem formulas
    const namedEntities = countMatches(NAMED_ENTITIES, text);
    const fpCertain = countMatches(FP_CERTAIN, textNormalized);
    const agreements = countMatches(AGREEMENT, textNormalized);
    const derivations = countMatches(DERIVATION, textNormalized);
    const metaDiscussion = countMatches(META_DISCUSSION, textNormalized);
    const helpfulDisclaimer = countMatches(HELPFUL_DISCLAIMER, textNormalized);
    const toolFlips = countMatches(TOOL_FLIP, textNormalized);
    const toolNames = countMatches(TOOL_NAMES, textNormalized);

    const hedgeDensity = hedges * norm;
    const refusalDensity = refusals * norm;
    const creativeDensity = creatives * norm;
    const adversarialDensity = adversarials * norm;
    const agreementDensity = agreements * norm;
    const derivationDensity = derivations * norm;
    const specificDensity = specifics * norm;

    // Sentence structure
    const sentences = raw.split(/[.!?]+/).map(s => s.trim()).filter(s => s.length > 3);
    const nSentences = Math.max(1, sentences.length);

    // Category scoring — base rate for each category
    const scores = {
      reasoning: 0.10,
      retrieval: 0.10,
      refusal: 0.02,
      creative: 0.02,
      adversarial: 0.02,
      confab: 0.02,
      sycophant: 0.02,
    };

    // Refusal — high-precision signal, dominant when present.
    // v0.2: soft-refusal contributes at half-weight.
    // v0.2.1: helpful-disclaimer suppresses refusal score (defends refusal-fp-disclaimer).
    let refusalBoost = refusalDensity * 0.40 + refusals * 0.20 + refusals_soft * 0.18;
    if (helpfulDisclaimer >= 1 && refusals === 0) {
      refusalBoost *= 0.20; // mostly helpful, "I cannot guarantee" is a hedge not refusal
    }
    scores.refusal += Math.min(0.88, refusalBoost);

    // Adversarial — high-precision signal.
    // v0.2.3: aggressive meta-discussion suppression. Discussion of
    // attack patterns (with any "researcher"/"study"/"paper" context)
    // collapses adversarial scoring to background. Refined to defend
    // adv-fp-meta where "researchers studied jailbreak prompts" was firing.
    let adversarialBoost = adversarialDensity * 0.50 + adversarials * 0.25;
    if (metaDiscussion >= 1 && adversarials <= 2) {
      adversarialBoost *= 0.05; // discussed, not exhibited
    }
    scores.adversarial += Math.min(0.95, adversarialBoost);

    // Sycophant — agreement marker density.
    // v0.2.3: meta-discussion of agreement language is not sycophancy.
    let sycophantBoost = agreementDensity * 0.22 + agreements * 0.05;
    if (metaDiscussion >= 1) {
      sycophantBoost *= 0.05; // strong suppression
    }
    if (agreements <= 1 && wordCount > 20) {
      sycophantBoost *= 0.40;
    }
    scores.sycophant += Math.min(0.85, sycophantBoost);

    // Tool drift (NEW v0.2) — detects mid-generation tool revision.
    // Score boost when "actually let me call X" follows initial tool mention.
    if (toolFlips >= 1 && toolNames >= 2) {
      scores.adversarial += 0; // not adversarial
      // Use a specific tool_arg_drift category — but we score it as
      // a separate signal that becomes a fault even if it doesn't win.
    }

    // Creative — narrative markers
    scores.creative += Math.min(0.60, creativeDensity * 0.18);

    // Reasoning — derivation language + confidence markers + declarative
    scores.reasoning += Math.min(0.40, derivationDensity * 0.20);
    scores.reasoning += Math.min(0.20, confidents * 0.12);
    scores.reasoning += Math.min(0.10, fpCertain * 0.10);
    // Hedging penalty only hurts reasoning mildly
    if (hedgeDensity > 3.0) scores.reasoning -= Math.min(0.12, (hedgeDensity - 3.0) * 0.025);

    // Retrieval — factual content without derivation work.
    // Signals: named entities OR specifics, declarative structure, no derivation.
    if ((specifics >= 1 || namedEntities >= 2) && derivationDensity < 2.0 && hedgeDensity < 3.0) {
      let retrievalBoost = 0;
      retrievalBoost += Math.min(0.35, specifics * 0.08);
      retrievalBoost += Math.min(0.25, namedEntities * 0.06);
      if (nSentences >= 2) retrievalBoost += 0.08;
      scores.retrieval += Math.min(0.65, retrievalBoost);
    }

    // Confab — specifics + named entities + confident tone + no hedging + no derivation.
    // Signature: "sounds authoritative but packed with unverifiable details."
    // The key distinguishing feature vs retrieval is NAMED-ENTITY DENSITY combined with
    // specific unverifiable numeric claims in statements that don't show reasoning work.
    if (specifics >= 1 && hedgeDensity < 1.5 && refusals === 0 && derivationDensity < 1.0) {
      const unverifiabilityScore = specifics + Math.floor(namedEntities / 2);
      if (unverifiabilityScore >= 3) {
        let confabBoost = 0;
        confabBoost += Math.min(0.35, specifics * 0.10);
        confabBoost += Math.min(0.20, namedEntities * 0.06);
        if (confidents >= 1) confabBoost += 0.10;
        if (fpCertain >= 1) confabBoost += 0.10;
        // The "name-drop + stat" pattern is the canonical confab signature
        if (namedEntities >= 2 && specifics >= 2) confabBoost += 0.15;
        scores.confab += Math.min(0.88, confabBoost);
      }
    }

    // Pick the winning category
    let category = "reasoning";
    let maxScore = 0;
    for (const [k, v] of Object.entries(scores)) {
      if (v > maxScore) { maxScore = v; category = k; }
    }
    let confidence = Math.max(0.15, Math.min(0.95, maxScore));

    // Axis proxies (Spec v1.0 §7.3 Tier 3)
    let K, C, D;

    const reasoningCats = ["reasoning", "retrieval"];
    const driftCats = ["confab", "sycophant", "adversarial"];

    // K: reasoning work proxy — high on derivation-heavy / confident-reasoning
    if (category === "reasoning") {
      K = Math.min(0.95, 0.55 + derivationDensity * 0.05 + confidents * 0.05);
    } else if (category === "retrieval") {
      K = Math.min(0.75, 0.45 + specifics * 0.02);
    } else {
      K = Math.max(0.15, 0.40 - hedgeDensity * 0.03);
    }

    // C: coherence — low when heavy hedging, low on confab (committed-but-hollow)
    C = Math.max(0.20, Math.min(0.95,
      0.70 - hedgeDensity * 0.05 - (category === "confab" ? 0.30 : 0) - (category === "sycophant" ? 0.15 : 0)
    ));

    // D: dissociation — high on confab/sycophant/adversarial, low otherwise
    if (driftCats.includes(category)) {
      D = Math.max(0.30, Math.min(0.95, confidence * 0.75 + 0.10));
    } else if (category === "refusal") {
      D = Math.max(0.10, Math.min(0.35, 0.20));
    } else {
      D = Math.max(0.05, Math.min(0.25, 0.15 - derivationDensity * 0.02));
    }

    // Trust = logistic over axes with Tier-3 penalty
    const T_raw = 1 / (1 + Math.exp(-(1.2 * K + 0.8 * C - 1.8 * D + 0.2)));
    const trust = Math.max(0.05, Math.min(0.95, T_raw - 0.25));

    // Faults — dedupe by kind
    const faultSet = new Map();
    function addFault(kind, severity, reason) {
      // Keep the highest severity for each kind
      const existing = faultSet.get(kind);
      if (!existing || severity > existing.severity) {
        faultSet.set(kind, { kind, severity, reason });
      }
    }

    if (category === "confab" && confidence > 0.35) {
      addFault("confabulation", confidence,
        `confident unverifiable claims (${specifics} specifics, ${confidents} certainty markers)`);
    }
    if (category === "refusal" && confidence > 0.65) {
      addFault("refusal", confidence, "strong refusal pattern");
    }
    if (category === "sycophant" && confidence > 0.35) {
      addFault("sycophant", confidence, `agreement-coded language (${agreements} markers)`);
    }
    if (category === "adversarial" && confidence > 0.25) {
      addFault("drift", confidence, "adversarial / jailbreak pattern");
    }
    // v0.2 NEW: tool-flip drift — independent of dominant category.
    if (toolFlips >= 1 && toolNames >= 2) {
      addFault("drift", Math.min(0.85, 0.50 + toolFlips * 0.10),
        `tool revision detected (${toolFlips} flip phrases, ${toolNames} tool refs)`);
    }
    // v0.2 hardened: low_trust now considers heavy hedging directly.
    if (trust < 0.30) {
      addFault("low_trust", 1 - trust, `aggregate trust ${trust.toFixed(2)} below 0.30`);
    } else if (hedgeDensity > 6.0 && fpCertain < 1) {
      // Hedge-density override: very heavy hedging always reduces trust
      // even when other signals are mild (defends low-trust-fn-confident-hedge).
      addFault("low_trust", Math.min(0.70, 0.40 + hedgeDensity * 0.03),
        `hedge density ${hedgeDensity.toFixed(1)} indicates uncertainty`);
    }
    // Unverified-claims flag: when text is factually dense but we can't tell
    // confab from retrieval from text alone. This is the honest tier-3 limit —
    // surface the density and let the user verify.
    // v0.2.2: cleanest rule — specifics + named entities both required.
    // Water/H2O (specifics but no people) doesn't fire. Confab_paper
    // (Dr. Vasquez + 2 stats) fires. Numeric-only stat dumps without
    // a person/place are honest text-only ambiguity.
    if ((category === "retrieval" || category === "confab") && hedgeDensity < 1.5 &&
        specifics >= 2 && namedEntities >= 1) {
      addFault("unverified_claims",
        Math.min(0.80, 0.30 + specifics * 0.08 + namedEntities * 0.04),
        `${specifics} specific claims + ${namedEntities} named entities, no hedging — verify independently`);
    }
    if (D > 0.50 && category !== "adversarial") {
      // Only fire D-drift if not already covered by adversarial fault
      addFault("drift", D, `expression-computation dissociation D=${D.toFixed(2)}`);
    }
    if (C < 0.30) {
      addFault("incoherence", 1 - C, `cross-phase coherence ${C.toFixed(2)} collapsed`);
    }
    // v0.2.3: topic-jump incoherence — fires when EVERY comparable
    // adjacent pair lacks lexical overlap. We only count pairs where
    // both sentences have ≥2 long words (skip "thin" sentences).
    // jumpRate >= 0.99 means total disconnect. Defends long-benign-fp
    // (math reasoning shares "term"/"sequence" between adjacent steps).
    if (sentences.length >= 3) {
      let lowOverlapPairs = 0;
      let comparablePairs = 0;
      for (let i = 1; i < sentences.length; i++) {
        const a = new Set(sentences[i-1].toLowerCase().split(/\W+/).filter(w => w.length > 3));
        const b = new Set(sentences[i].toLowerCase().split(/\W+/).filter(w => w.length > 3));
        if (a.size < 2 || b.size < 2) continue;
        comparablePairs++;
        const overlap = [...a].filter(w => b.has(w)).length;
        if (overlap === 0) lowOverlapPairs++;
      }
      const jumpRate = comparablePairs > 0 ? lowOverlapPairs / comparablePairs : 0;
      if (jumpRate >= 0.99 && lowOverlapPairs >= 1 && comparablePairs >= 1) {
        addFault("incoherence", Math.min(0.75, 0.40 + jumpRate * 0.40),
          `topic jumps in ${lowOverlapPairs}/${comparablePairs} comparable sentence pairs`);
      }
    }

    const faults = Array.from(faultSet.values()).sort((a, b) => b.severity - a.severity);

    return {
      category,
      confidence: Number(confidence.toFixed(3)),
      K: Number(K.toFixed(3)),
      C: Number(C.toFixed(3)),
      D: Number(D.toFixed(3)),
      trust: Number(trust.toFixed(3)),
      faults,
      raw_stats: {
        hedges, refusals, confidents, specifics, creatives,
        adversarials, agreements, derivations, fpCertain,
        namedEntities, wordCount, nSentences,
      },
    };
  }

  // Export to window for content script, and module.exports for Node tests
  if (typeof window !== "undefined") {
    window.__styxxScope = { classifyText };
  }
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { classifyText };
  }
})();
