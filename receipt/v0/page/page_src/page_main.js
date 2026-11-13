"use strict";
/*
 * RECEIPT v0 check page: the page's own code (lab-written, Fathom Lab). It never runs the checker
 * in the page itself. It reads the checker's text from the element #checker-src, hashes it, and
 * runs it only when the hash equals the one recorded when the page was built, only in a fresh
 * Worker made from that text, and only under a time limit. A verdict is shown only for a run that
 * returned one message this code accepts; anything else is "no verdict" with the reason.
 *
 * Every string that comes from a document, a manifest, a receipt or the checker reaches the page
 * through textContent. No innerHTML, no attribute built from input, no URL built from input.
 * Direction control characters are shown as visible markers (forDisplay); copy hands over raw text.
 *
 * Repairs after the attack of 2026-11-13 (findings F1 to F10) are marked with their finding ids.
 */
const ReceiptV0Page = (function () {
  const CHECKER_SHA256_AT_BUILD = "@@CHECKER_SHA256@@";
  const CHECKER_ORIGIN = "@@CHECKER_ORIGIN@@";
  const EXAMPLES = @@EXAMPLES_JSON@@;
  const TIME_LIMIT_MS = 10000;
  const MAX_INPUT_BYTES = 16 * 1024 * 1024;
  // F4: limits on what the page takes and draws. The worker refuses to send a result whose
  // receipt text is longer than MAX_RECEIPT_CHARS (workerMain holds the same number; the build
  // checks that the two agree), and acceptResult refuses one too.
  const MAX_RECEIPT_CHARS = 2000000;
  const ROWS_DRAWN = 200;
  const PRE_CHARS = 200000;

  // RECEIPT_v0.md section 10, the closed lists. A result naming anything else is not accepted.
  const SPAN_VERDICTS = ["HELD", "FAILED", "UNRESOLVED", "MALFORMED"];
  const DOCUMENT_VERDICTS = ["SWORN-HELD", "SWORN-FAILED", "SWORN-UNCHECKED", "UNSWORN"];
  const REASONS = ["tag_syntax", "nesting", "stray_closer", "unclosed", "empty_span",
    "hidden_commitment", "length_cap", "kind_unknown", "kind_reserved", "receipt_form",
    "absent_over_partial", "hash_over_partial", "directional_override", "number_count",
    "number_grammar", "needle_count", "needle_empty", "digest_form",
    "kind_of_source_unknown", "receipt_author_minted", "anchor_out_of_range", "receipt_not_json",
    "pointer_unresolvable", "pointer_ambiguous", "leaf_not_scalar", "leaf_not_numeric",
    "leaf_not_string", "short_needle",
    "manifest_absent", "manifest_integrity", "rung_unknown", "manifest_id_missing",
    "manifest_bytes_absent", "manifest_no_completeness", "no_repository",
    "value_mismatch", "rounds_to_zero", "needle_missing", "needle_present", "digest_mismatch"];
  const REFUSALS = ["nothing_to_verify", "name_form", "commit_form", "manifest_spec",
    "manifest_receipts_shape", "manifest_authored_shape", "manifest_form", "float", "int_range",
    "non_ascii_key", "lone_surrogate", "depth", "type", "receipt_shape", "receipt_digest",
    "attestations_not_empty", "profile_not_checkable", "document_mismatch"];
  const ANSWERS = ["REDERIVES", "DOES-NOT-REDERIVE"];

  /* ------------------------------------------------------------------ sha256 (FIPS 180-4)
   * The page's own, separate from the checker's, so that the checker is hashed before any of
   * its code runs. Input: an array of byte values (a Uint8Array from any realm). */
  const K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2];

  function sha256hex(bytes) {
    const n = bytes.length;
    const blocks = Math.floor((n + 8) / 64) + 1;
    const m = new Uint8Array(blocks * 64);
    for (let i = 0; i < n; i++) m[i] = bytes[i];
    m[n] = 0x80;
    // bit length: n * 8 as two 32-bit words (n stays far below 2^53 / 8 here)
    const hi = Math.floor(n / 0x20000000), lo = (n * 8) >>> 0;
    const L = m.length;
    m[L - 8] = (hi >>> 24) & 255; m[L - 7] = (hi >>> 16) & 255; m[L - 6] = (hi >>> 8) & 255; m[L - 5] = hi & 255;
    m[L - 4] = (lo >>> 24) & 255; m[L - 3] = (lo >>> 16) & 255; m[L - 2] = (lo >>> 8) & 255; m[L - 1] = lo & 255;
    const H = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    const w = new Array(64);
    const rot = (x, r) => (x >>> r) | (x << (32 - r));
    for (let off = 0; off < L; off += 64) {
      for (let i = 0; i < 16; i++) {
        const j = off + i * 4;
        w[i] = ((m[j] << 24) | (m[j + 1] << 16) | (m[j + 2] << 8) | m[j + 3]) >>> 0;
      }
      for (let i = 16; i < 64; i++) {
        const s0 = rot(w[i - 15], 7) ^ rot(w[i - 15], 18) ^ (w[i - 15] >>> 3);
        const s1 = rot(w[i - 2], 17) ^ rot(w[i - 2], 19) ^ (w[i - 2] >>> 10);
        w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
      }
      let a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7];
      for (let i = 0; i < 64; i++) {
        const t1 = (h + (rot(e, 6) ^ rot(e, 11) ^ rot(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i]) >>> 0;
        const t2 = ((rot(a, 2) ^ rot(a, 13) ^ rot(a, 22)) + ((a & b) ^ (a & c) ^ (b & c))) >>> 0;
        h = g; g = f; f = e; e = (d + t1) >>> 0; d = c; c = b; b = a; a = (t1 + t2) >>> 0;
      }
      H[0] = (H[0] + a) >>> 0; H[1] = (H[1] + b) >>> 0; H[2] = (H[2] + c) >>> 0; H[3] = (H[3] + d) >>> 0;
      H[4] = (H[4] + e) >>> 0; H[5] = (H[5] + f) >>> 0; H[6] = (H[6] + g) >>> 0; H[7] = (H[7] + h) >>> 0;
    }
    let out = "";
    for (let i = 0; i < 8; i++) out += ("00000000" + H[i].toString(16)).slice(-8);
    return out;
  }

  function utf8(s) { return new TextEncoder().encode(s); }

  /* F7: direction controls (U+061C, U+200E, U+200F, U+202A to U+202E, U+2066 to U+2069) can
   * reverse how the text after them is drawn. For display only, each becomes a visible marker
   * such as <U+202E>; copy buttons hand over the raw text. */
  const BIDI_RE = /[\u061C\u200E\u200F\u202A-\u202E\u2066-\u2069]/g;
  function forDisplay(v) {
    return String(v).replace(BIDI_RE, (c) => "<U+" + ("0000" + c.charCodeAt(0).toString(16).toUpperCase()).slice(-4) + ">");
  }
  function hasBidi(v) { return forDisplay(v) !== String(v); }

  /* ------------------------------------------------------------------ the worker
   * workerMain runs ONLY inside the worker, after the checker's own text has run there and set
   * self.ReceiptV0. It is shipped by Function.prototype.toString, so it must not use anything
   * from this file's scope. It removes the worker's network and loading globals before it reads
   * any message. That is a second layer under the page's Content-Security-Policy, and a modest
   * one (F9): it cannot remove import(), which is syntax, and it runs after the checker's own
   * top-level code; both are left to the policy the worker inherits. It answers one message with
   * one message and never catches a failure into a verdict: a Refusal is the checker's own answer
   * (section 11); any other error is reported as an error. A result too large for the page (F4)
   * is not sent; only its size is. */
  function workerMain() {
    "use strict";
    const scope = self;
    const C = scope.ReceiptV0;
    const MAX_RECEIPT_CHARS = 2000000;
    const names = ["fetch", "XMLHttpRequest", "WebSocket", "WebSocketStream", "EventSource",
      "importScripts", "Worker", "SharedWorker", "BroadcastChannel", "WebTransport",
      "RTCPeerConnection", "indexedDB", "caches", "FontFace", "fonts"];
    for (const n of names) {
      try { Object.defineProperty(scope, n, { value: undefined, writable: false, configurable: false }); } catch (e) { /* kept: CSP still applies */ }
    }
    const bytes = (x) => (x === null || x === undefined ? null : new Uint8Array(x));
    const tooLarge = (op, chars, spans) => ({ ok: true, op, refusal: null,
      too_large: { receipt_chars: chars, spans, limit: MAX_RECEIPT_CHARS } });
    function handle(job) {
      if (!C) return { ok: false, op: job && job.op, error: "the checker did not load in the worker" };
      if (job.op === "check") {
        const core = C.check(bytes(job.document), job.name, bytes(job.manifest), job.commit);
        const text = C.jcs(C.receipt(core, C.VERIFIER));
        if (text.length > MAX_RECEIPT_CHARS) return tooLarge("check", text.length, core.spans.length);
        const rows = core.spans.map((r) => ({
          at: r.at, start: r.start, end: r.end, receipt: r.receipt, kind: r.kind,
          claim: r.claim, claim_sha256: r.claim_sha256, verdict: r.verdict, reason: r.reason,
          detail_text: C.jcs(r.detail),
          resolved_sha256: r.resolved_sha256 === undefined ? null : r.resolved_sha256,
          provenance_text: r.provenance === undefined ? null : C.jcs(r.provenance),
        }));
        return {
          ok: true, op: "check", refusal: null,
          core_digest: C.digest(core),
          receipt_text: text,
          receipt_sha256: C.sha256(new TextEncoder().encode(text)),
          document_verdict: core.document_verdict,
          document_malformed_text: core.document_malformed === null ? null : C.jcs(core.document_malformed),
          name: core.document.name,
          inline_sha256: core.document.inline_sha256,
          canonical_sha256: core.document.canonical_sha256,
          manifest_digest: core.manifest_digest,
          commit: core.commit,
          counts: { HELD: core.counts.HELD, FAILED: core.counts.FAILED,
            UNRESOLVED: core.counts.UNRESOLVED, MALFORMED: core.counts.MALFORMED },
          sworn_total: core.sworn_total,
          unresolved: core.unresolved,
          rungs_text: C.jcs(core.rungs),
          rows,
        };
      }
      if (job.op === "recheck") {
        return { ok: true, op: "recheck", refusal: null,
          answer: C.recheck(bytes(job.receipt), bytes(job.document), bytes(job.manifest)) };
      }
      if (job.op === "jcs") {
        const canonical = C.jcs(C.parseJson(bytes(job.value)));
        if (canonical.length > MAX_RECEIPT_CHARS) return tooLarge("jcs", canonical.length, 0);
        return { ok: true, op: "jcs", refusal: null, canonical };
      }
      return { ok: false, op: job && job.op, error: "unknown operation" };
    }
    scope.onmessage = function (ev) {
      const job = ev.data;
      let out;
      try {
        out = handle(job);
      } catch (e) {
        if (C && e instanceof C.Refusal) out = { ok: true, op: job && job.op, refusal: String(e.code) };
        else out = { ok: false, op: job && job.op, error: String((e && e.name) || "Error") + ": " + String((e && e.message) || e) };
      }
      scope.postMessage(out);
    };
  }

  function buildWorkerSource(checkerText) {
    // The checker comes first, so its own "use strict" directive governs the whole worker script.
    return checkerText + "\n;\n(" + workerMain.toString() + ")();\n";
  }

  /* ------------------------------------------------------------------ accepting a result */
  const isObj = (v) => typeof v === "object" && v !== null && !Array.isArray(v);
  const isStr = (v) => typeof v === "string";
  const isHex64 = (v) => isStr(v) && /^[0-9a-f]{64}$/.test(v);
  const isNat = (v) => typeof v === "number" && Number.isSafeInteger(v) && v >= 0;
  const strOrNull = (v) => v === null || isStr(v);

  function checkShape(m) {
    const need = ["core_digest", "receipt_text", "receipt_sha256", "document_verdict",
      "document_malformed_text", "name", "inline_sha256", "canonical_sha256", "manifest_digest",
      "commit", "counts", "sworn_total", "unresolved", "rungs_text", "rows"];
    for (const k of need) if (!(k in m)) return "missing field " + k;
    if (!isHex64(m.core_digest)) return "digest is not 64 lowercase hex digits";
    if (!isStr(m.receipt_text)) return "receipt text is not a string";
    if (m.receipt_sha256 !== sha256hex(utf8(m.receipt_text))) return "receipt sha256 does not match the receipt text";
    if (DOCUMENT_VERDICTS.indexOf(m.document_verdict) < 0) return "document verdict not in section 10";
    if (!strOrNull(m.document_malformed_text) || !isStr(m.name) || !isHex64(m.inline_sha256)) return "document fields";
    if (!(m.canonical_sha256 === null || isHex64(m.canonical_sha256))) return "canonical sha256";
    if (!(m.manifest_digest === null || isHex64(m.manifest_digest))) return "manifest digest";
    if (!strOrNull(m.commit) || !isStr(m.rungs_text)) return "commit or rungs";
    if (!isObj(m.counts) || !SPAN_VERDICTS.every((v) => isNat(m.counts[v]))) return "counts";
    if (!Array.isArray(m.rows) || !isNat(m.sworn_total) || m.rows.length !== m.sworn_total) return "rows and sworn_total differ";
    if (m.unresolved !== m.counts.UNRESOLVED) return "unresolved differs from counts";
    const tally = { HELD: 0, FAILED: 0, UNRESOLVED: 0, MALFORMED: 0 };
    for (const r of m.rows) {
      if (!isObj(r) || SPAN_VERDICTS.indexOf(r.verdict) < 0) return "a row verdict not in section 10";
      if (r.verdict === "HELD" ? r.reason !== null : REASONS.indexOf(r.reason) < 0) return "a row reason not in section 10";
      if (!isNat(r.at) || !(r.start === null || isNat(r.start)) || !(r.end === null || isNat(r.end))) return "row offsets";
      if (!strOrNull(r.receipt) || !strOrNull(r.kind) || !strOrNull(r.claim) || !strOrNull(r.claim_sha256)) return "row strings";
      if (!isStr(r.detail_text) || !strOrNull(r.provenance_text) || !(r.resolved_sha256 === null || isHex64(r.resolved_sha256))) return "row detail";
      tally[r.verdict] += 1;
    }
    if (!SPAN_VERDICTS.every((v) => tally[v] === m.counts[v])) return "counts differ from the rows";
    let parsed;
    try { parsed = JSON.parse(m.receipt_text); } catch (e) { return "receipt text is not JSON"; }
    if (!isObj(parsed) || parsed.digest !== m.core_digest || !isObj(parsed.core)) return "receipt digest differs";
    const core = parsed.core;
    if (core.document_verdict !== m.document_verdict || !Array.isArray(core.spans) || core.spans.length !== m.rows.length) return "receipt core differs";
    for (let i = 0; i < m.rows.length; i++) {
      const a = core.spans[i], b = m.rows[i];
      if (!isObj(a) || a.verdict !== b.verdict || a.reason !== b.reason || a.at !== b.at) return "receipt row " + (i + 1) + " differs";
    }
    return null;
  }

  /** acceptResult(job, message) -> {accepted: true, result} | {accepted: false, why} */
  function acceptResult(job, m) {
    if (!isObj(m)) return { accepted: false, why: "the worker answered with something that is not an object" };
    if (m.ok !== true) {
      const e = isStr(m.error) ? m.error.slice(0, 500) : "no message";
      return { accepted: false, why: "the checker stopped with an error: " + e };
    }
    if (m.op !== job.op) return { accepted: false, why: "the answer is for another operation" };
    // F4: a result too large for this page is no verdict, whoever reports its size
    const size = isObj(m.too_large) && isNat(m.too_large.receipt_chars) ? m.too_large.receipt_chars
      : isStr(m.receipt_text) ? m.receipt_text.length : isStr(m.canonical) ? m.canonical.length : 0;
    if ("too_large" in m || size > MAX_RECEIPT_CHARS) {
      return { accepted: false, why: "the result is too large for this page: its receipt text has " + size +
        " characters, and this page takes at most " + MAX_RECEIPT_CHARS + ". The command line checker has no such limit." };
    }
    if (m.refusal !== null) {
      if (REFUSALS.indexOf(m.refusal) < 0) return { accepted: false, why: "a refusal code not in section 10" };
      return { accepted: true, result: { op: m.op, refusal: m.refusal } };
    }
    if (m.op === "recheck") {
      if (ANSWERS.indexOf(m.answer) < 0) return { accepted: false, why: "an answer not in section 11.3" };
      return { accepted: true, result: { op: m.op, refusal: null, answer: m.answer } };
    }
    if (m.op === "jcs") {
      if (!isStr(m.canonical)) return { accepted: false, why: "canonical text is not a string" };
      return { accepted: true, result: { op: m.op, refusal: null, canonical: m.canonical } };
    }
    if (m.op === "check") {
      const bad = checkShape(m);
      if (bad) return { accepted: false, why: "the result failed this page's shape check: " + bad };
      // F2: the result must be for the inputs this run handed over, hashed again here
      const d = job.document;
      if (d === null || d === undefined || m.inline_sha256 !== sha256hex(d)) {
        return { accepted: false, why: "the result names a document sha256 other than the one this run handed over" };
      }
      if (m.name !== job.name) return { accepted: false, why: "the result names a document name other than the one this run handed over" };
      if (m.commit !== (job.commit === undefined ? null : job.commit)) {
        return { accepted: false, why: "the result names a commit label other than the one this run handed over" };
      }
      return { accepted: true, result: m };
    }
    return { accepted: false, why: "unknown operation" };
  }

  /* ------------------------------------------------------------------ one run
   * runJob(env, checkerText, job, {timeoutMs}) -> Promise of
   *   {status: "verdict", result, ms} | {status: "no-verdict", why, ms}
   * env: {makeWorker(source) -> worker, release(), setTimer(f, ms), clearTimer(h), now()}.
   * There is no path that runs the checker without a worker: no worker, no verdict. */
  function copyBytes(x) {
    if (x === null || x === undefined) return null;
    const out = new Uint8Array(x.length);
    for (let i = 0; i < x.length; i++) out[i] = x[i];
    return out;
  }

  function runJob(env, checkerText, job, opts) {
    const limit = opts && opts.timeoutMs ? opts.timeoutMs : TIME_LIMIT_MS;
    const t0 = env.now();
    return new Promise((resolve) => {
      let w = null, timer = null, done = false;
      const finish = (r) => {
        if (done) return;
        done = true;
        if (timer !== null) env.clearTimer(timer);
        if (w) { try { w.terminate(); } catch (e) { /* already gone */ } }
        try { env.release(); } catch (e) { /* nothing to release */ }
        r.ms = Math.round(env.now() - t0);
        r.limitMs = limit;
        resolve(r);
      };
      if (sha256hex(utf8(checkerText)) !== CHECKER_SHA256_AT_BUILD) {
        finish({ status: "no-verdict", why: "the embedded checker does not hash to the sha256 recorded at build; nothing was run" });
        return;
      }
      try {
        w = env.makeWorker(buildWorkerSource(checkerText));
      } catch (e) {
        w = null;
        finish({ status: "no-verdict", why: "no worker could be started (" + String((e && e.message) || e) +
          "). This page never runs the checker without a time limit, so there is no verdict." });
        return;
      }
      w.onmessage = (ev) => {
        const a = acceptResult(job, ev ? ev.data : undefined);
        finish(a.accepted ? { status: "verdict", result: a.result }
          : { status: "no-verdict", why: a.why });
      };
      w.onerror = (ev) => {
        if (ev && typeof ev.preventDefault === "function") ev.preventDefault();
        finish({ status: "no-verdict", why: "the worker failed: " + String((ev && ev.message) || "no message (a blocked or failed worker script)") });
      };
      w.onmessageerror = () => finish({ status: "no-verdict", why: "a message from the worker could not be read" });
      timer = env.setTimer(() => finish({ status: "no-verdict",
        why: "no answer within " + limit + " ms; the worker was stopped" }), limit);
      try {
        w.postMessage({ op: job.op, name: job.name, commit: job.commit, document: copyBytes(job.document),
          manifest: copyBytes(job.manifest), receipt: copyBytes(job.receipt), value: copyBytes(job.value) });
      } catch (e) {
        finish({ status: "no-verdict", why: "the inputs could not be handed to the worker: " + String((e && e.message) || e) });
      }
    });
  }

  /* ------------------------------------------------------------------ showing a result
   * Uses only createElement, className, textContent, appendChild, a fixed set of attributes with
   * fixed values, and click listeners. `doc` is the page's document (or a stand-in in tests).
   * Every value is drawn through forDisplay (F7). At most ROWS_DRAWN span cards and PRE_CHARS
   * receipt characters are drawn (F4); the copy button hands over the whole raw receipt.
   * ctx: {copy(text, button), inputs: [[label, value]], notes: [text]} (F2, F3).
   * Returns what it drew: {status, big, rows: [[verdict, reason]], rows_total} for the example
   * comparison (F1), read back from the elements it made. */
  const VERDICT_CLASS = { HELD: "v-held", FAILED: "v-failed", UNRESOLVED: "v-unresolved", MALFORMED: "v-malformed",
    "SWORN-HELD": "v-held", "SWORN-FAILED": "v-failed", "SWORN-UNCHECKED": "v-unresolved", UNSWORN: "v-malformed" };
  const VERDICT_WORDS = {
    "SWORN-HELD": "at least one span HELD, none FAILED or MALFORMED. UNRESOLVED spans may sit beside it (13.11).",
    "SWORN-FAILED": "a span FAILED or was MALFORMED, or the document itself is malformed (2.1, 2.2).",
    "SWORN-UNCHECKED": "spans were marked but none was compared: no span HELD or FAILED.",
    "UNSWORN": "no span was marked, so nothing was checked.",
    HELD: "one comparison passed. It does not say the sentence is true (section 12).",
    FAILED: "the comparison ran and did not pass.",
    UNRESOLVED: "the checker could not see the evidence; nothing was compared. Never an accusation.",
    MALFORMED: "refused before its comparison ran.",
  };

  function renderOutcome(doc, root, outcome, ctx) {
    let bigEl = null;
    const rvEls = [], rrEls = [];
    const E = (tag, cls, text) => {
      const e = doc.createElement(tag);
      if (cls) e.className = cls;
      if (text !== undefined && text !== null) e.textContent = forDisplay(text);
      if (cls && cls.split(" ")[0] === "big") bigEl = e;
      if (cls === "rv") rvEls.push(e);
      if (cls === "rr") rrEls.push(e);
      return e;
    };
    const copyButton = (text, what) => {
      const b = E("button", "copy", "copy " + what);
      b.setAttribute("type", "button");
      b.addEventListener("click", () => { if (ctx && ctx.copy) ctx.copy(text, b); });
      return b;
    };
    const field = (parent, label, value, mono) => {
      const d = E("div", "field");
      d.appendChild(E("span", "label", label));
      d.appendChild(E(mono ? "code" : "span", mono ? "value mono" : "value", value === null ? "null" : value));
      parent.appendChild(d);
      return d;
    };
    const inputsBlock = () => {
      if (!ctx) return;
      if (ctx.inputs && ctx.inputs.length) {
        const d = E("div", "inputs");
        d.appendChild(E("p", "label", "the inputs this run was given (taken when run was pressed)"));
        for (const pair of ctx.inputs) field(d, pair[0], pair[1], true);
        root.appendChild(d);
      }
      if (ctx.notes) for (const n of ctx.notes) root.appendChild(E("p", "note warn", n));
    };
    const shown = () => {
      const t = (e) => (e ? e.textContent : null);
      return { status: outcome.status, big: t(bigEl), rows: rvEls.map((e, i) => [t(e), t(rrEls[i])]),
        rows_total: outcome.result && outcome.result.rows ? outcome.result.rows.length : 0 };
    };
    root.textContent = "";
    const ran = "ran in a fresh worker made from the embedded checker; " +
      (outcome.ms !== undefined ? outcome.ms + " ms, " : "") + "limit " + outcome.limitMs + " ms";
    if (outcome.status !== "verdict") {
      root.appendChild(E("p", "big v-none", "no verdict"));
      root.appendChild(E("p", "why", outcome.why));
      root.appendChild(E("p", "note", "Nothing above is a verdict. The page shows one only for a run that returned cleanly."));
      inputsBlock();
      return shown();
    }
    const r = outcome.result;
    root.appendChild(E("p", "note", ran));
    if (r.refusal) {
      root.appendChild(E("p", "big v-malformed", "REFUSED " + r.refusal));
      root.appendChild(E("p", "why", "A refusal is not a verdict: the checker emitted nothing and returned this code (RECEIPT_v0.md sections 10 and 11)."));
      inputsBlock();
      return shown();
    }
    if (r.op === "recheck") {
      root.appendChild(E("p", "big " + (r.answer === "REDERIVES" ? "v-held" : "v-failed"), r.answer));
      root.appendChild(E("p", "why", r.answer === "REDERIVES"
        ? "This checker computes the same verdict core from the inputs listed below. It does not say the document held: the document verdict is a different word inside the receipt (11.3)."
        : "This checker computes a different verdict core from the inputs listed below than the one the receipt records (11.3)."));
      inputsBlock();
      return shown();
    }
    if (r.op === "jcs") {
      root.appendChild(E("pre", "mono", r.canonical.length > PRE_CHARS ? r.canonical.slice(0, PRE_CHARS) : r.canonical));
      return shown();
    }
    // check
    const head = E("div", "dv " + VERDICT_CLASS[r.document_verdict]);
    head.appendChild(E("p", "big", r.document_verdict));
    head.appendChild(E("p", "why", VERDICT_WORDS[r.document_verdict]));
    root.appendChild(head);
    inputsBlock();
    const facts = E("div", "facts");
    field(facts, "spans", String(r.sworn_total) + "  (HELD " + r.counts.HELD + ", FAILED " + r.counts.FAILED +
      ", UNRESOLVED " + r.counts.UNRESOLVED + ", MALFORMED " + r.counts.MALFORMED + ")", false);
    field(facts, "document name (inside the digest, 13.6)", r.name, true);
    field(facts, "document sha256", r.inline_sha256, true);
    field(facts, "canonical text sha256", r.canonical_sha256, true);
    field(facts, "manifest digest", r.manifest_digest, true);
    field(facts, "commit label", r.commit, true);
    field(facts, "rungs", r.rungs_text, true);
    if (r.document_malformed_text !== null) field(facts, "document malformed", r.document_malformed_text, true);
    root.appendChild(facts);
    const rows = E("div", "rows");
    r.rows.slice(0, ROWS_DRAWN).forEach((row, i) => {
      const card = E("div", "row " + VERDICT_CLASS[row.verdict]);
      const h = E("p", "rowhead");
      h.appendChild(E("span", "rv", row.verdict));
      h.appendChild(E("span", "rr", row.reason === null ? "" : " " + row.reason));
      h.appendChild(E("span", "ri", "  span " + (i + 1) + ", byte " + row.at));
      card.appendChild(h);
      card.appendChild(E("p", "why", VERDICT_WORDS[row.verdict]));
      field(card, "kind", row.kind, true);
      field(card, "receipt reference", row.receipt, true);
      field(card, "claim (inner text)", row.claim, true);
      field(card, "canonical offsets", row.start === null ? "null" : row.start + " to " + row.end, true);
      field(card, "detail", row.detail_text, true);
      if (row.resolved_sha256 !== null) field(card, "receipt bytes sha256", row.resolved_sha256, true);
      if (row.provenance_text !== null) field(card, "provenance (declared by the manifest, not verified)", row.provenance_text, true);
      rows.appendChild(card);
    });
    if (r.rows.length > ROWS_DRAWN) {
      rows.appendChild(E("p", "note", (r.rows.length - ROWS_DRAWN) + " more spans are not drawn here (this page draws at most " +
        ROWS_DRAWN + "). Every span is in the receipt text; the copy receipt button copies all of it."));
    }
    root.appendChild(rows);
    const rec = E("div", "receipt");
    const dg = E("div", "copyline");
    dg.appendChild(E("span", "label", "digest (sha256 of the canonical verdict core, 9.3)"));
    dg.appendChild(E("code", "value mono", r.core_digest));
    dg.appendChild(copyButton(r.core_digest, "digest"));
    rec.appendChild(dg);
    const rs = E("div", "copyline");
    rs.appendChild(E("span", "label", "sha256 of the receipt text below (its UTF-8 bytes, no final line end)"));
    rs.appendChild(E("code", "value mono", r.receipt_sha256));
    rs.appendChild(copyButton(r.receipt_sha256, "sha256"));
    rec.appendChild(rs);
    const rt = E("div", "copyline");
    rt.appendChild(E("span", "label", "canonical receipt (9.4). The command line checker prints the same text followed by one LF." +
      (hasBidi(r.receipt_text) ? " Direction control characters are drawn as markers such as <U+202E>; the copy button copies the raw text." : "")));
    rt.appendChild(copyButton(r.receipt_text, "receipt"));
    rec.appendChild(rt);
    if (r.receipt_text.length > PRE_CHARS) {
      rec.appendChild(E("p", "note", "Only the first " + PRE_CHARS + " of " + r.receipt_text.length +
        " characters are drawn below; the copy receipt button copies all of them."));
    }
    rec.appendChild(E("pre", "mono receipttext", r.receipt_text.length > PRE_CHARS ? r.receipt_text.slice(0, PRE_CHARS) : r.receipt_text));
    root.appendChild(rec);
    return shown();
  }

  /* F1: the example banner compares what was drawn (the document verdict and every span verdict
   * and reason, read back from the elements) and the receipt text with the command line output
   * recorded in this same file. An edited copy of the file can change both sides, so the banner
   * says that too. Decided from the run's own operation and inputs (F5), never the current mode. */
  function compareWithExample(ex, op, outcome, shown) {
    if (!ex || outcome.status !== "verdict") return null;
    const r = outcome.result;
    let match;
    if (op === "check") {
      const rowsSame = !!shown && shown.rows.length === ex.expected.rows.length && shown.rows_total === ex.expected.rows.length &&
        ex.expected.rows.every((x, i) => shown.rows[i][0] === x[0] && (shown.rows[i][1] || "").trim() === (x[1] === null ? "" : x[1]));
      match = r.op === "check" && !r.refusal && rowsSame && !!shown && shown.big === ex.expected.document_verdict &&
        r.document_verdict === ex.expected.document_verdict &&
        sha256hex(utf8(r.receipt_text + "\n")) === ex.expected.check_stdout_sha256;
    } else {
      match = r.op === "recheck" && !r.refusal && r.answer === ex.expected.recheck && !!shown && shown.big === r.answer;
    }
    return { match, text: (match ? "Same as" : "DIFFERS from") + " the command line checker's recorded answer for this example (" +
      ex.expected.file + ")" + (op === "check" ? ": the receipt text, the document verdict and every span verdict drawn here." : ".") +
      " That record is inside this same file, so an edited copy of the file can change both sides; this line does not show that the file is the lab's." };
  }

  return Object.freeze({
    CHECKER_SHA256_AT_BUILD, CHECKER_ORIGIN, EXAMPLES, TIME_LIMIT_MS, MAX_INPUT_BYTES,
    MAX_RECEIPT_CHARS, ROWS_DRAWN, PRE_CHARS,
    sha256hex, utf8, forDisplay, hasBidi, workerMain, buildWorkerSource, acceptResult, runJob, renderOutcome,
    compareWithExample,
  });
})();

globalThis.ReceiptV0Page = ReceiptV0Page;

/* ==================================================================== the page (browser only) */
if (typeof document !== "undefined") (function ui() {
  const P = ReceiptV0Page;
  const $ = (id) => document.getElementById(id);
  const checkerEl = $("checker-src");
  const checkerText = checkerEl ? checkerEl.textContent : "";
  const checkerBytes = P.utf8(checkerText);
  const nowHash = P.sha256hex(checkerBytes);
  const intact = nowHash === P.CHECKER_SHA256_AT_BUILD;
  $("sha-now").textContent = nowHash;
  $("sha-build").textContent = P.CHECKER_SHA256_AT_BUILD;
  $("sha-bytes").textContent = String(checkerBytes.length);
  $("sha-origin").textContent = P.CHECKER_ORIGIN;
  $("sha-match").textContent = intact ? "the same: the checker may run" : "DIFFERENT: nothing will run";
  $("sha-match").className = intact ? "ok" : "bad";

  // Trusted Types: where the browser enforces them, new Worker() takes only a URL this policy
  // approves, and it approves only the blob: URL this page has just made.
  let allowedUrl = null, policy = null;
  try {
    if (typeof trustedTypes !== "undefined" && trustedTypes.createPolicy) {
      policy = trustedTypes.createPolicy("receipt-v0-worker", {
        createScriptURL: (u) => { if (u !== allowedUrl || u === null) throw new TypeError("not the worker URL this page made"); return u; },
      });
    }
  } catch (e) { policy = null; }

  function browserEnv() {
    let url = null;
    return {
      makeWorker(source) {
        if (typeof Worker !== "function") throw new Error("this browser offers no Worker");
        url = URL.createObjectURL(new Blob([source], { type: "text/javascript" }));
        allowedUrl = url;
        try {
          return new Worker(policy ? policy.createScriptURL(url) : url);
        } finally { allowedUrl = null; }
      },
      release() { if (url !== null) { URL.revokeObjectURL(url); url = null; } },
      setTimer: (f, ms) => setTimeout(f, ms),
      clearTimer: (h) => clearTimeout(h),
      now: () => (typeof performance !== "undefined" ? performance.now() : Date.now()),
    };
  }

  const res = $("result");
  const runBtn = $("run");
  const note = (cls, text) => { const p = document.createElement("p"); p.className = cls; p.textContent = P.forDisplay(text); return p; };

  /* F2, F5: every run gets an id; every input change, mode switch or example load moves `seq` on.
   * An outcome whose id is no longer `seq` is dropped, and a shown result is replaced as soon as
   * any input changes. */
  let seq = 0;          // moves on with every run and every change
  let runningId = 0;    // the run in flight, if any
  let shownId = 0;      // the run whose answer (or "running") is on screen; 0: none
  const slots = [];
  const anyReading = () => slots.some((s) => s.reading);
  function updateRun() {
    if (!intact) return;
    const reading = anyReading();
    runBtn.disabled = reading || (runningId !== 0 && runningId === seq);
    runBtn.textContent = reading ? "waiting: a file is still being read" : "run the checker";
  }
  function invalidate() {
    seq += 1;
    if (shownId !== 0) {
      shownId = 0;
      res.textContent = "";
      res.appendChild(note("note stale", "Inputs changed since this run, so its answer is no longer shown. Run again."));
    }
    updateRun();
  }

  // ---- inputs: each slot holds bytes from a file, from pasted text, or from an example
  function slot(prefix) {
    const ta = $(prefix + "-text"), fi = $(prefix + "-file"), info = $(prefix + "-info"), clr = $(prefix + "-clear");
    const s = { state: null, exampleId: null, onchange: null, onfile: null, reading: false, pending: null, note: null };
    let readToken = 0;
    const span = (cls, text) => { const e = document.createElement("span"); if (cls) e.className = cls; e.textContent = text; return e; };
    const show = () => {
      info.textContent = "";
      if (s.reading) { info.textContent = "reading the file... (run waits until it is read)"; return; }
      if (s.pending !== null) { info.textContent = "changed: read when you stop typing or press run"; return; }
      if (!s.state) { info.textContent = s.note || "empty: nothing is handed to the checker"; return; }
      const st = s.state;
      const tail = ": " + st.bytes.length + " bytes, sha256 " + st.sha;
      if (st.fileName !== undefined) {
        // F7: the file name in its own element, drawn with direction controls as markers
        info.appendChild(span("", "the file \""));
        info.appendChild(span("fname", P.forDisplay(st.fileName)));
        info.appendChild(span("", "\"" + tail));
      } else {
        info.textContent = st.label + tail;
      }
    };
    const cancelPending = () => { if (s.pending !== null) { clearTimeout(s.pending); s.pending = null; } };
    const set = (st, exampleId) => {
      if (st) st.sha = P.sha256hex(st.bytes);
      s.state = st; s.exampleId = exampleId || null; s.note = null;
      show(); if (s.onchange) s.onchange(); invalidate();
    };
    const readBox = () => {
      const v = ta.value;
      if (v === "") { set(null); return; }
      const b = P.utf8(v);
      if (b.length > P.MAX_INPUT_BYTES) {
        s.state = null; s.exampleId = null; s.note = "too large for this page (over " + P.MAX_INPUT_BYTES + " bytes)"; show(); invalidate(); return;
      }
      set({ bytes: b, label: "pasted text (a browser hands it over with LF line ends)", pasted: true, hasLF: v.indexOf("\n") >= 0 });
    };
    s.flush = () => { if (s.pending !== null) { cancelPending(); readBox(); } };
    ta.addEventListener("input", () => {
      readToken += 1; s.reading = false;          // a file still being read is no longer wanted
      fi.value = "";
      s.exampleId = null;
      cancelPending();
      s.pending = setTimeout(() => { s.pending = null; readBox(); }, 150);
      show(); invalidate();                       // F2: at once, not after the pause
    });
    fi.addEventListener("change", () => {
      const f = fi.files && fi.files[0];
      if (!f) return;
      readToken += 1;
      const tok = readToken;
      cancelPending();
      s.state = null; s.exampleId = null; s.note = null;
      if (s.onfile) s.onfile(f);
      if (f.size > P.MAX_INPUT_BYTES) {
        s.reading = false; s.note = "too large for this page (over " + P.MAX_INPUT_BYTES + " bytes)"; show(); invalidate(); return;
      }
      s.reading = true;
      show(); invalidate();
      const done = (buf) => {
        if (tok !== readToken) return;
        s.reading = false; ta.value = "";
        set({ bytes: new Uint8Array(buf), fileName: String(f.name) });
      };
      const fail = () => {
        if (tok !== readToken) return;
        s.reading = false; s.state = null; s.note = "the file could not be read"; show(); invalidate();
      };
      if (typeof f.arrayBuffer === "function") f.arrayBuffer().then(done, fail);
      else { const rd = new FileReader(); rd.onload = () => done(rd.result); rd.onerror = fail; rd.readAsArrayBuffer(f); }
    });
    clr.addEventListener("click", () => { readToken += 1; s.reading = false; cancelPending(); ta.value = ""; fi.value = ""; set(null); });
    s.loadText = (text, label, exampleId) => {
      readToken += 1; s.reading = false; cancelPending(); fi.value = ""; ta.value = text;
      set({ bytes: P.utf8(text), label }, exampleId);
    };
    s.describe = () => {
      if (!s.state) return "none";
      const st = s.state;
      const src = st.fileName !== undefined ? "the file \"" + st.fileName + "\"" : st.label;
      return src + ", " + st.bytes.length + " bytes, sha256 " + st.sha;
    };
    show();
    slots.push(s);
    return s;
  }
  const doc = slot("doc"), man = slot("man"), rec = slot("rec");
  const nameEl = $("name"), commitEl = $("commit"), nameInfo = $("name-info");
  const NAME_CLI = "The command line checker names a document by the path it was given unless it gets --name; " +
    "give it --name with the exact text of this field to get the same digest.";
  const setNameInfo = (t) => { if (nameInfo) nameInfo.textContent = P.forDisplay(t + " " + NAME_CLI); };
  let nameEdited = false;
  nameEl.addEventListener("input", () => { nameEdited = true; setNameInfo("Typed here."); invalidate(); });
  commitEl.addEventListener("input", () => { invalidate(); });
  // F10: a newly chosen document file names the document again, even after an example or an edit
  doc.onfile = () => { nameEdited = false; };
  doc.onchange = () => {
    if (nameEdited || !doc.state || doc.state.fileName === undefined) return;
    const fname = doc.state.fileName;
    if (P.hasBidi(fname)) {
      // F7: a name holding direction controls is not copied into the field
      setNameInfo("The chosen file's name holds a direction control character, so it was not copied into this field; type the name you want.");
      return;
    }
    nameEl.value = fname;
    setNameInfo("Filled from the chosen file's name.");
  };
  setNameInfo("Not taken from a file.");

  // ---- mode
  let mode = "check";
  const setMode = (m) => {
    mode = m;
    $("mode-check").setAttribute("aria-pressed", m === "check" ? "true" : "false");
    $("mode-recheck").setAttribute("aria-pressed", m === "recheck" ? "true" : "false");
    for (const el of document.querySelectorAll(".only-check")) el.hidden = m !== "check";
    for (const el of document.querySelectorAll(".only-recheck")) el.hidden = m !== "recheck";
    invalidate();                                   // F5
  };
  $("mode-check").addEventListener("click", () => setMode("check"));
  $("mode-recheck").addEventListener("click", () => setMode("recheck"));

  // ---- examples
  const exBox = $("examples");
  P.EXAMPLES.forEach((ex) => {
    const b = document.createElement("button");
    b.setAttribute("type", "button");
    b.className = "example";
    b.textContent = ex.title;
    b.addEventListener("click", () => {
      doc.loadText(ex.document, "example document " + JSON.stringify(ex.name), ex.id);
      man.loadText(ex.manifest, "example manifest", ex.id);
      rec.loadText(ex.receipt, "example receipt (printed by the command line checker)", ex.id);
      nameEl.value = ex.name; nameEdited = true; commitEl.value = "";
      setNameInfo("Set by the example.");
      invalidate();
      $("example-note").textContent = "Loaded " + ex.title + ". The command line checker gave " + ex.expected.document_verdict +
        " with digest " + ex.expected.digest + ", and RECHECK of its receipt gave " + ex.expected.recheck + " (" + ex.expected.file + ").";
    });
    exBox.appendChild(b);
  });

  // ---- copy
  function copy(text, button) {
    const said = (t) => { button.textContent = t; setTimeout(() => { button.textContent = "copy again"; }, 1500); };
    const fallback = () => {
      const t = document.createElement("textarea");
      t.className = "offscreen";
      t.value = text;
      document.body.appendChild(t);
      t.select();
      let ok = false;
      try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
      document.body.removeChild(t);
      said(ok ? "copied" : "copy failed: select the text by hand");
    };
    if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(() => said("copied"), fallback);
    else fallback();
  }

  // ---- run
  const PASTE_NOTE = "The document was pasted. A browser hands pasted text to the page with every line end as LF, " +
    "so the file it came from can get a different verdict, not only a different digest: a claim's length or a quoted needle can change. " +
    "For the file's own verdict, choose the file.";
  const LF_NOTE = " This pasted text holds line ends; if the original used CR LF or a lone CR, the result for the original can differ.";
  if (!intact) { runBtn.disabled = true; runBtn.textContent = "will not run: the checker's hash differs"; }
  runBtn.addEventListener("click", () => {
    if (!intact) return;
    for (const s of slots) s.flush();               // F2: take what the boxes hold now
    if (anyReading()) {
      shownId = 0;
      res.textContent = "";
      res.appendChild(note("note", "A chosen file is still being read, so nothing was run. Press run again when its line shows a sha256."));
      updateRun();
      return;
    }
    const op = mode;
    const get = (s) => (s.state ? s.state.bytes : null);
    const job = op === "check"
      ? { op: "check", document: get(doc), manifest: get(man), name: nameEl.value, commit: commitEl.value === "" ? null : commitEl.value }
      : { op: "recheck", receipt: get(rec), document: get(doc), manifest: get(man) };
    // what this run was given, taken now (F2), and whether any of it was pasted (F3)
    const inputs = [["operation", op === "check" ? "CHECK" : "RECHECK"]];
    if (op === "recheck") inputs.push(["receipt", rec.describe()]);
    inputs.push(["document", doc.describe()]);
    inputs.push(["manifest", man.describe()]);
    if (op === "check") {
      inputs.push(["document name", job.name]);
      inputs.push(["commit label", job.commit === null ? "none" : job.commit]);
    }
    const notes = [];
    if (doc.state && doc.state.pasted) notes.push(PASTE_NOTE + (doc.state.hasLF ? LF_NOTE : ""));
    const pastedOthers = [op === "recheck" && rec.state && rec.state.pasted ? "receipt" : null, man.state && man.state.pasted ? "manifest" : null].filter(Boolean);
    if (pastedOthers.length) {
      notes.push("The " + pastedOthers.join(" and the ") + " was pasted. Pasted bytes can differ from the file's (line ends, a byte order mark at the start), and the result can differ with them.");
    }
    // the recorded command line answer applies only when every input is an unchanged example
    const exId = doc.exampleId;
    const ex0 = P.EXAMPLES.find((e) => e.id === exId);
    const ex = ex0 && man.exampleId === exId && (op === "check"
      ? job.name === ex0.name && job.commit === null
      : rec.exampleId === exId) ? ex0 : null;
    seq += 1;
    const id = seq;
    runningId = id; shownId = id;
    res.textContent = "running in a worker, limit " + P.TIME_LIMIT_MS + " ms...";
    updateRun();
    P.runJob(browserEnv(), checkerText, job, { timeoutMs: P.TIME_LIMIT_MS }).then((outcome) => {
      if (runningId === id) runningId = 0;
      if (id !== seq) { updateRun(); return; }     // inputs changed during the run: dropped (F2, F5)
      updateRun();
      const shown = P.renderOutcome(document, res, outcome, { copy, inputs, notes });
      const cmp = P.compareWithExample(ex, op, outcome, shown);
      if (cmp) res.insertBefore(note(cmp.match ? "ok" : "bad", cmp.text), res.firstChild);
    });
  });
  setMode("check");
})();
