"use strict";
/*
 * RECEIPT v0 (DRAFT, unsigned, nothing frozen): a checker written from receipt/v0/RECEIPT_v0.md,
 * in JavaScript. One file, no dependency; it runs unchanged in Node and in a browser.
 *
 * LAB-WRITTEN, NOT INDEPENDENT. Fathom Lab wrote this file, with the text open and with the lab's
 * Python checker of the same text (receipt/v0/check_v0.py) open beside it; its sha256 and its
 * JSON-reading approach were adapted from the lab's older JavaScript verifier
 * (styxx/_data/sworn_verify.js). Agreement between this file and the lab's other code is
 * agreement inside one lab. It is not a standard, nobody outside the lab emits or checks these
 * objects, and a HELD verdict never means a sentence is true (RECEIPT_v0.md sections 12 and 14).
 *
 * The checking code is pure: no clock, no network, no file system, no randomness, no host JSON
 * parser and no binary floating point in any verdict path. Numbers read from JSON keep the digits
 * as written (6.13); integer bounds, rounding and exponents are decided on those digits with
 * BigInt and digit strings (3.1, 7.3). Strings are JavaScript strings (UTF-16 code units); every
 * place where the text counts or orders code points does so by code point, and an unpaired
 * surrogate is found by code unit.
 *
 * The letter, number and digit classes of 7.2 come from the host's regular-expression engine
 * (\p{L}, \p{N}, \p{Nd}); the Unicode version they follow is the host's, which the text does not
 * pin (its open item 13.1).
 *
 *   check(document, name, manifest = null, commit = null) -> verdict core   (sections 2 to 9, 11.1)
 *   receipt(core, verifier = null)                          -> the verdict receipt (9.4)
 *   recheck(receiptBytes, document, manifest = null)        -> "REDERIVES" | "DOES-NOT-REDERIVE" (11.3)
 *   jcs(value), digest(value)                               -> the canonical text (section 3), its sha256
 *   parseJson(bytes)                                        -> 6.13's strict reading
 *
 * `document`, `manifest` and `receiptBytes` are Uint8Array (a Node Buffer is one). A refusal
 * (section 11) throws Refusal with a code from REFUSALS; nothing is emitted.
 *
 * Command line (Node only; the only code here that reads a file, and it reads only its arguments):
 *
 *   node check_v0.js check DOC [--manifest FILE] [--name NAME] [--commit HEX]
 *   node check_v0.js recheck RECEIPT DOC [--manifest FILE]
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module && module.exports) module.exports = api;
  else root.ReceiptV0 = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {

  const SCHEMA_V0 = "receipt/v0-draft";
  const PROFILE_RN = "rn-inline/v0-draft";
  const PROFILE_LEGACY = "legacy-projection/v0-draft";
  const FORMAT = "sworn/0.1";
  const MANIFEST_SPEC = "sworn/manifest/0.2";
  const KINDS = ["numeric", "quote", "hash", "absent"];
  const EXTERNAL = ["tool_stdout", "tool_stderr", "file_read", "http_fetch", "harness_note",
    "test_report", "attestation"];
  const AUTHOR_SIDE = ["agent_output", "agent_file_write", "agent_message"];
  const SPAN_VERDICTS = ["HELD", "FAILED", "UNRESOLVED", "MALFORMED"];
  const DOCUMENT_VERDICTS = ["SWORN-HELD", "SWORN-FAILED", "SWORN-UNCHECKED", "UNSWORN"];
  const CAP = 300;
  const MIN_NEEDLE = 16;
  const INT_MAX = 9007199254740991n;          // 2^53 - 1
  const DEPTH_MAX = 32;
  const ADJUSTED_MAX = "320";

  // section 10, the closed lists
  const MALFORMED_DOCUMENT = ["tag_syntax", "nesting", "stray_closer", "unclosed", "empty_span",
    "hidden_commitment", "length_cap", "kind_unknown", "kind_reserved", "receipt_form",
    "absent_over_partial", "hash_over_partial", "directional_override", "number_count",
    "number_grammar", "needle_count", "needle_empty", "digest_form"];
  const MALFORMED_EVIDENCE = ["kind_of_source_unknown", "receipt_author_minted",
    "anchor_out_of_range", "receipt_not_json", "pointer_unresolvable", "pointer_ambiguous",
    "leaf_not_scalar", "leaf_not_numeric", "leaf_not_string", "short_needle"];
  const UNRESOLVED_REASONS = ["manifest_absent", "manifest_integrity", "rung_unknown",
    "manifest_id_missing", "manifest_bytes_absent", "manifest_no_completeness", "no_repository"];
  const FAILED_REASONS = ["value_mismatch", "rounds_to_zero", "needle_missing", "needle_present",
    "digest_mismatch"];
  const REASONS = MALFORMED_DOCUMENT.concat(MALFORMED_EVIDENCE, UNRESOLVED_REASONS, FAILED_REASONS);
  const LEGACY_REASONS = ["no_commit", "commit_absent", "path_absent", "not_a_blob",
    "receipt_too_large", "git_unavailable", "prereg_not_in_tree"];
  const DOCUMENT_REASONS = ["invalid_utf8", "unbalanced_fences"];
  const REFUSALS = ["nothing_to_verify", "name_form", "commit_form", "manifest_spec",
    "manifest_receipts_shape", "manifest_authored_shape", "manifest_form",
    "float", "int_range", "non_ascii_key", "lone_surrogate", "depth", "type",
    "receipt_shape", "receipt_digest", "attestations_not_empty", "profile_not_checkable",
    "document_mismatch"];
  const REDERIVES = "REDERIVES";
  const DOES_NOT_REDERIVE = "DOES-NOT-REDERIVE";

  const HEX_LOWER = "0123456789abcdef";
  const HEX_ANY = "0123456789abcdefABCDEF";
  const WORD_ASCII = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_";

  class Refusal extends Error {
    constructor(code, message) {
      super(message ? code + ": " + message : code);
      this.name = "Refusal";
      this.code = code;
    }
  }

  class JSONError extends Error {
    constructor(message) { super(message); this.name = "JSONError"; }
  }

  /* ========================================================================== sha256
   * FIPS 180-4, synchronous (crypto.subtle is async-only). Adapted from styxx/_data/sworn_verify.js.
   */
  const _K = new Uint32Array([
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2]);

  function sha256(bytes) {
    const ml = bytes.length;
    const total = (((ml + 9 + 63) / 64) | 0) * 64;
    const padded = new Uint8Array(total);
    padded.set(bytes);
    padded[ml] = 0x80;
    const dv = new DataView(padded.buffer);
    // the bit length as two 32-bit words, computed without a fractional value
    const bitsHi = Number((BigInt(ml) * 8n) >> 32n);
    const bitsLo = Number((BigInt(ml) * 8n) & 0xffffffffn);
    dv.setUint32(total - 8, bitsHi);
    dv.setUint32(total - 4, bitsLo);
    const H = new Uint32Array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]);
    const w = new Uint32Array(64);
    for (let off = 0; off < total; off += 64) {
      for (let i = 0; i < 16; i++) w[i] = dv.getUint32(off + i * 4);
      for (let i = 16; i < 64; i++) {
        const x = w[i - 15], y = w[i - 2];
        const s0 = ((x >>> 7) | (x << 25)) ^ ((x >>> 18) | (x << 14)) ^ (x >>> 3);
        const s1 = ((y >>> 17) | (y << 15)) ^ ((y >>> 19) | (y << 13)) ^ (y >>> 10);
        w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
      }
      let a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7];
      for (let i = 0; i < 64; i++) {
        const S1 = ((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7));
        const ch = (e & f) ^ (~e & g);
        const t1 = (h + S1 + ch + _K[i] + w[i]) >>> 0;
        const S0 = ((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10));
        const mj = (a & b) ^ (a & c) ^ (b & c);
        const t2 = (S0 + mj) >>> 0;
        h = g; g = f; f = e; e = (d + t1) >>> 0;
        d = c; c = b; b = a; a = (t1 + t2) >>> 0;
      }
      H[0] = (H[0] + a) >>> 0; H[1] = (H[1] + b) >>> 0; H[2] = (H[2] + c) >>> 0;
      H[3] = (H[3] + d) >>> 0; H[4] = (H[4] + e) >>> 0; H[5] = (H[5] + f) >>> 0;
      H[6] = (H[6] + g) >>> 0; H[7] = (H[7] + h) >>> 0;
    }
    let out = "";
    for (let i = 0; i < 8; i++) out += H[i].toString(16).padStart(8, "0");
    return out;
  }

  /* ========================================================================== bytes and text */

  const _ENC = new TextEncoder();
  // ignoreBOM: a leading U+FEFF is text, never silently dropped (6.13 refuses it; a span may hold it)
  const _DEC = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true });

  function asBytes(x) {
    if (x instanceof Uint8Array) return new Uint8Array(x);    // a copy, and a plain Uint8Array
    throw new TypeError("bytes (Uint8Array) expected");
  }

  /** UTF-8 of a string that holds no unpaired surrogate (callers check first). */
  function utf8(s) { return _ENC.encode(s); }

  /** Strict decode of bytes known to be well-formed UTF-8, or null. */
  function decodeStrict(bytes) {
    try { return _DEC.decode(bytes); } catch (e) { return null; }
  }

  /** 2.1: the offset of the first byte of the first ill-formed sequence, or -1 (Unicode 3.9,
   *  Table 3-7: no overlong form, no surrogate, nothing above U+10FFFF). */
  function firstInvalidUtf8(b) {
    const n = b.length;
    let i = 0;
    while (i < n) {
      const c = b[i];
      if (c < 0x80) { i++; continue; }
      let need, lo = 0x80, hi = 0xbf;
      if (c >= 0xc2 && c <= 0xdf) need = 1;
      else if (c === 0xe0) { need = 2; lo = 0xa0; }
      else if ((c >= 0xe1 && c <= 0xec) || c === 0xee || c === 0xef) need = 2;
      else if (c === 0xed) { need = 2; hi = 0x9f; }
      else if (c === 0xf0) { need = 3; lo = 0x90; }
      else if (c >= 0xf1 && c <= 0xf3) need = 3;
      else if (c === 0xf4) { need = 3; hi = 0x8f; }
      else return i;
      if (i + 1 >= n || b[i + 1] < lo || b[i + 1] > hi) return i;
      for (let k = 2; k <= need; k++) {
        if (i + k >= n || (b[i + k] & 0xc0) !== 0x80) return i;
      }
      i += need + 1;
    }
    return -1;
  }

  /** True when the string holds an unpaired UTF-16 surrogate. */
  function hasLoneSurrogate(s) {
    for (let i = 0; i < s.length; i++) {
      const c = s.charCodeAt(i);
      if (c >= 0xd800 && c <= 0xdbff) {
        const d = i + 1 < s.length ? s.charCodeAt(i + 1) : 0;
        if (d >= 0xdc00 && d <= 0xdfff) { i++; continue; }
        return true;
      }
      if (c >= 0xdc00 && c <= 0xdfff) return true;
    }
    return false;
  }

  /** The number of code points (an unpaired surrogate counts as one). */
  function cpLen(s) {
    let n = 0;
    for (let i = 0; i < s.length; i++) {
      const c = s.charCodeAt(i);
      if (c >= 0xd800 && c <= 0xdbff && i + 1 < s.length) {
        const d = s.charCodeAt(i + 1);
        if (d >= 0xdc00 && d <= 0xdfff) i++;
      }
      n++;
    }
    return n;
  }

  /** The first n code points. */
  function cpHead(s, n) {
    let i = 0, k = 0;
    while (i < s.length && k < n) {
      const c = s.charCodeAt(i);
      if (c >= 0xd800 && c <= 0xdbff && i + 1 < s.length) {
        const d = s.charCodeAt(i + 1);
        if (d >= 0xdc00 && d <= 0xdfff) i++;
      }
      i++;
      k++;
    }
    return s.slice(0, i);
  }

  /** Comparison of two strings by code point (not by UTF-16 code unit). */
  function cmpCodePoint(a, b) {
    const n = Math.min(a.length, b.length);
    for (let i = 0; i < n; i++) {
      const x = a.codePointAt(i), y = b.codePointAt(i);
      if (x !== y) return x < y ? -1 : 1;
      if (x > 0xffff) i++;
    }
    return a.length === b.length ? 0 : (a.length < b.length ? -1 : 1);
  }

  /** indexOf of a byte sequence in [from, end). */
  function findBytes(hay, pat, from, end) {
    if (end === undefined) end = hay.length;
    const m = pat.length;
    if (m === 0) return from <= end ? from : -1;
    const first = pat[0];
    let i = from;
    while (i <= end - m) {
      i = hay.indexOf(first, i);
      if (i < 0 || i > end - m) return -1;
      let k = 1;
      while (k < m && hay[i + k] === pat[k]) k++;
      if (k === m) return i;
      i++;
    }
    return -1;
  }

  function startsWithBytes(b, pat, p) {
    if (p + pat.length > b.length) return false;
    for (let k = 0; k < pat.length; k++) if (b[p + k] !== pat[k]) return false;
    return true;
  }

  /** Non-overlapping occurrences, counted from the left (needle not empty). */
  function countBytes(hay, needle) {
    let c = 0, i = 0;
    for (;;) {
      const j = findBytes(hay, needle, i);
      if (j < 0) return c;
      c++;
      i = j + needle.length;
    }
  }

  function bytesEqual(a, b) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
  }

  function concatBytes(parts) {
    let n = 0;
    for (const p of parts) n += p.length;
    const out = new Uint8Array(n);
    let o = 0;
    for (const p of parts) { out.set(p, o); o += p.length; }
    return out;
  }

  const _ASCII_WS = new Set([0x20, 0x09, 0x0a, 0x0b, 0x0c, 0x0d]);
  function allAsciiWs(b) {
    for (let i = 0; i < b.length; i++) if (!_ASCII_WS.has(b[i])) return false;
    return true;
  }

  function isHex64Lower(x) {
    if (typeof x !== "string" || x.length !== 64) return false;
    for (let i = 0; i < 64; i++) if (HEX_LOWER.indexOf(x[i]) < 0) return false;
    return true;
  }

  function isHex40Lower(x) {
    if (typeof x !== "string" || x.length !== 40) return false;
    for (let i = 0; i < 40; i++) if (HEX_LOWER.indexOf(x[i]) < 0) return false;
    return true;
  }

  function isHex64Any(x) {
    if (typeof x !== "string" || x.length !== 64) return false;
    for (let i = 0; i < 64; i++) if (HEX_ANY.indexOf(x[i]) < 0) return false;
    return true;
  }

  /* ========================================================================== 6.13: JSON, read strictly
   * Numbers keep their digits (JNum), never converted to binary floating point; NaN, Infinity and
   * -Infinity parse (JConst) and are never numeric; an object keeps the last value of a repeated
   * key and remembers it as repeated (JObj.dup). Iterative: no host recursion limit decides.
   */
  class JNum {
    constructor(text) { this.text = text; }
    isIntegerToken() { return !/[.eE]/.test(this.text); }
  }

  class JConst {
    constructor(name) { this.name = name; }
  }

  class JObj {
    constructor() { this.map = new Map(); this.dup = new Set(); }
    has(k) { return this.map.has(k); }
    get(k) { return this.map.get(k); }
    keys() { return Array.from(this.map.keys()); }
    values() { return Array.from(this.map.values()); }
    get size() { return this.map.size; }
  }

  const _ESC = { '"': '"', "\\": "\\", "/": "/", b: "\b", f: "\f", n: "\n", r: "\r", t: "\t" };

  function _ws(s, i) {
    const n = s.length;
    while (i < n) {
      const c = s.charCodeAt(i);
      if (c === 0x20 || c === 0x09 || c === 0x0a || c === 0x0d) i++;
      else break;
    }
    return i;
  }

  function _hex4(s, i) {
    const h = s.slice(i, i + 4);
    if (h.length !== 4 || !/^[0-9a-fA-F]{4}$/.test(h)) throw new JSONError("bad \\u escape at " + i);
    return parseInt(h, 16);
  }

  function _string(s, i) {
    i += 1;
    const parts = [];
    const n = s.length;
    for (;;) {
      const q = s.indexOf('"', i);
      if (q < 0) throw new JSONError("unterminated string");
      let bs = s.indexOf("\\", i);
      if (bs > q) bs = -1;
      const stop = bs < 0 ? q : bs;
      const chunk = s.slice(i, stop);
      if (/[\u0000-\u001f]/.test(chunk)) throw new JSONError("control character in a string");
      parts.push(chunk);
      if (bs < 0) return [parts.join(""), q + 1];
      i = bs + 1;
      if (i >= n) throw new JSONError("unterminated escape");
      const e = s[i];
      if (Object.prototype.hasOwnProperty.call(_ESC, e)) {
        parts.push(_ESC[e]);
        i += 1;
      } else if (e === "u") {
        const cp = _hex4(s, i + 1);
        i += 5;
        // a \uD8xx\uDCxx pair becomes one code point; in UTF-16 the two units already are one,
        // and an unpaired surrogate stays a single unit
        parts.push(String.fromCharCode(cp));
      } else {
        throw new JSONError("bad escape");
      }
    }
  }

  function _number(s, i) {
    const n = s.length;
    let j = i;
    const dig = (k) => k < n && s.charCodeAt(k) >= 0x30 && s.charCodeAt(k) <= 0x39;
    if (j < n && s[j] === "-") j++;
    if (j < n && s[j] === "0") j++;
    else if (j < n && s[j] >= "1" && s[j] <= "9") { while (dig(j)) j++; }
    else throw new JSONError("bad number at " + i);
    if (j < n && s[j] === ".") {
      j++;
      const k = j;
      while (dig(j)) j++;
      if (j === k) throw new JSONError("bad fraction at " + i);
    }
    if (j < n && (s[j] === "e" || s[j] === "E")) {
      j++;
      if (j < n && (s[j] === "+" || s[j] === "-")) j++;
      const k = j;
      while (dig(j)) j++;
      if (j === k) throw new JSONError("bad exponent at " + i);
    }
    return [new JNum(s.slice(i, j)), j];
  }

  function _parseText(s) {
    const n = s.length;
    const stack = [];               // [container, pending key or null]
    let i = _ws(s, 0);
    for (;;) {
      if (i >= n) throw new JSONError("value expected at end");
      const c = s[i];
      let val = null;
      let complete = true;
      if (c === "{") {
        i = _ws(s, i + 1);
        const obj = new JObj();
        if (i < n && s[i] === "}") { val = obj; i += 1; }
        else {
          if (i >= n || s[i] !== '"') throw new JSONError("key expected at " + i);
          let key;
          [key, i] = _string(s, i);
          i = _ws(s, i);
          if (i >= n || s[i] !== ":") throw new JSONError("colon expected at " + i);
          i = _ws(s, i + 1);
          stack.push([obj, key]);
          complete = false;
        }
      } else if (c === "[") {
        i = _ws(s, i + 1);
        const arr = [];
        if (i < n && s[i] === "]") { val = arr; i += 1; }
        else { stack.push([arr, null]); complete = false; }
      } else if (c === '"') {
        [val, i] = _string(s, i);
      } else if (s.startsWith("-Infinity", i)) {
        val = new JConst("-Infinity"); i += 9;
      } else if (c === "-" || (c >= "0" && c <= "9")) {
        [val, i] = _number(s, i);
      } else if (s.startsWith("NaN", i)) {
        val = new JConst("NaN"); i += 3;
      } else if (s.startsWith("Infinity", i)) {
        val = new JConst("Infinity"); i += 8;
      } else if (s.startsWith("true", i)) {
        val = true; i += 4;
      } else if (s.startsWith("false", i)) {
        val = false; i += 5;
      } else if (s.startsWith("null", i)) {
        val = null; i += 4;
      } else {
        throw new JSONError("unexpected character at " + i);
      }
      if (!complete) continue;
      // a value is complete: attach it, then close as many containers as the text closes
      for (;;) {
        if (stack.length === 0) {
          i = _ws(s, i);
          if (i !== n) throw new JSONError("trailing bytes at " + i);
          return val;
        }
        const frame = stack[stack.length - 1];
        const cont = frame[0];
        const isObj = cont instanceof JObj;
        if (isObj) {
          const key = frame[1];
          if (cont.map.has(key)) cont.dup.add(key);
          cont.map.set(key, val);
        } else {
          cont.push(val);
        }
        i = _ws(s, i);
        if (i >= n) throw new JSONError("unterminated container");
        const d = s[i];
        if (d === ",") {
          i = _ws(s, i + 1);
          if (isObj) {
            if (i >= n || s[i] !== '"') throw new JSONError("key expected at " + i);
            let key;
            [key, i] = _string(s, i);
            i = _ws(s, i);
            if (i >= n || s[i] !== ":") throw new JSONError("colon expected at " + i);
            i = _ws(s, i + 1);
            frame[1] = key;
          }
          break;
        }
        if (d === (isObj ? "}" : "]")) {
          stack.pop();
          val = cont;
          i += 1;
          continue;
        }
        throw new JSONError("unexpected character at " + i);
      }
    }
  }

  /** 6.13 over bytes: UTF-8 without a byte order mark; RFC 8259 plus NaN, Infinity, -Infinity. */
  function parseJson(data) {
    if (!(data instanceof Uint8Array)) throw new TypeError("bytes expected");
    const s = decodeStrict(data);
    if (s === null) throw new JSONError("not UTF-8");
    if (s.length && s.charCodeAt(0) === 0xfeff) throw new JSONError("byte order mark");
    return _parseText(s);
  }

  /** True when any object anywhere in the value had a key written twice. */
  function hasDup(v) {
    const todo = [v];
    while (todo.length) {
      const x = todo.pop();
      if (x instanceof JObj) {
        if (x.dup.size) return true;
        for (const y of x.map.values()) todo.push(y);
      } else if (Array.isArray(x)) {
        for (const y of x) todo.push(y);
      }
    }
    return false;
  }

  /* ========================================================================== section 3: the canonical form */

  function isPlainObject(v) {
    if (v === null || typeof v !== "object") return false;
    const p = Object.getPrototypeOf(v);
    return p === Object.prototype || p === null;
  }

  const _NEEDS_WORK = /[\u0000-\u001f"\\\ud800-\udfff]/;

  function _jcsString(s, out) {
    if (!_NEEDS_WORK.test(s)) { out.push('"' + s + '"'); return; }
    let buf = '"';
    for (let i = 0; i < s.length; i++) {
      const c = s.charCodeAt(i);
      if (c >= 0xd800 && c <= 0xdbff) {
        const d = i + 1 < s.length ? s.charCodeAt(i + 1) : 0;
        if (d >= 0xdc00 && d <= 0xdfff) { buf += s[i] + s[i + 1]; i++; continue; }
        throw new Refusal("lone_surrogate", "U+" + c.toString(16).toUpperCase());
      }
      if (c >= 0xdc00 && c <= 0xdfff) throw new Refusal("lone_surrogate", "U+" + c.toString(16).toUpperCase());
      if (c === 0x08) buf += "\\b";
      else if (c === 0x09) buf += "\\t";
      else if (c === 0x0a) buf += "\\n";
      else if (c === 0x0c) buf += "\\f";
      else if (c === 0x0d) buf += "\\r";
      else if (c === 0x22) buf += '\\"';
      else if (c === 0x5c) buf += "\\\\";
      else if (c < 0x20) buf += "\\u00" + HEX_LOWER[c >> 4] + HEX_LOWER[c & 15];
      else buf += s[i];
    }
    out.push(buf + '"');
  }

  function _jcsInt(n, out) {
    // n is a BigInt
    if (n > INT_MAX || n < -INT_MAX) throw new Refusal("int_range");
    out.push(n.toString());
  }

  function _jcsKeys(keys) {
    for (const k of keys) {
      if (typeof k !== "string") throw new Refusal("type", "object key");
      for (let i = 0; i < k.length; i++) {
        if (k.charCodeAt(i) > 0x7e) throw new Refusal("non_ascii_key");
      }
    }
    return keys.slice().sort(cmpCodePoint);
  }

  function _jcs(v, depth, out) {
    if (v === null) out.push("null");
    else if (v === true) out.push("true");
    else if (v === false) out.push("false");
    else if (typeof v === "number") {
      // a host number only ever comes from this checker's own counts and offsets
      if (!Number.isInteger(v)) throw new Refusal("float", String(v));
      if (!Number.isSafeInteger(v)) throw new Refusal("int_range");
      _jcsInt(BigInt(v), out);
    } else if (typeof v === "bigint") {
      _jcsInt(v, out);
    } else if (v instanceof JNum) {
      // decided on the number as written (3.1): a fraction or an exponent is never admitted
      if (!v.isIntegerToken()) throw new Refusal("float", v.text.slice(0, 40));
      const digits = v.text.replace(/^-+/, "");
      if (digits.length > 16) throw new Refusal("int_range", v.text.slice(0, 40));
      _jcsInt(BigInt(v.text), out);
    } else if (v instanceof JConst) {
      throw new Refusal("float", v.name);
    } else if (typeof v === "string") {
      _jcsString(v, out);
    } else if (Array.isArray(v)) {
      if (depth >= DEPTH_MAX) throw new Refusal("depth");
      out.push("[");
      for (let k = 0; k < v.length; k++) {
        if (k) out.push(",");
        _jcs(v[k], depth + 1, out);
      }
      out.push("]");
    } else if (v instanceof JObj || isPlainObject(v)) {
      if (depth >= DEPTH_MAX) throw new Refusal("depth");
      const isJ = v instanceof JObj;
      const keys = _jcsKeys(isJ ? v.keys() : Object.keys(v));
      out.push("{");
      for (let k = 0; k < keys.length; k++) {
        if (k) out.push(",");
        _jcsString(keys[k], out);
        out.push(":");
        _jcs(isJ ? v.get(keys[k]) : v[keys[k]], depth + 1, out);
      }
      out.push("}");
    } else {
      throw new Refusal("type", typeof v);
    }
  }

  /** The canonical text of `value` (section 3). Refuses with 3.3's code, never approximates. */
  function jcs(value) {
    const out = [];
    _jcs(value, 0, out);
    return out.join("");
  }

  /** 3.4: sha256 over the UTF-8 bytes of the canonical text. */
  function digest(value) {
    return sha256(utf8(jcs(value)));
  }

  /* ========================================================================== 6.10: strict base64 */

  const _B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  const _B64_INDEX = new Int16Array(128).fill(-1);
  for (let i = 0; i < 64; i++) _B64_INDEX[_B64.charCodeAt(i)] = i;

  /** Bytes, or null when `s` is not strict base64. Pad bits that are not zero are ignored. */
  function b64Strict(s) {
    if (typeof s !== "string" || s.length % 4) return null;
    const pad = s.endsWith("==") ? 2 : (s.endsWith("=") ? 1 : 0);
    const m = s.length - pad;
    const vals = new Uint8Array(m);
    for (let i = 0; i < m; i++) {
      const c = s.charCodeAt(i);
      const v = c < 128 ? _B64_INDEX[c] : -1;
      if (v < 0) return null;                  // outside the alphabet, or '=' inside
      vals[i] = v;
    }
    if (pad && m % 4 === 0) return null;
    const full = (m / 4) | 0;
    const rem = m % 4;                          // 0, 2 or 3 (a length that is a multiple of 4)
    const out = new Uint8Array(full * 3 + (rem === 2 ? 1 : rem === 3 ? 2 : 0));
    let o = 0, i = 0;
    for (let g = 0; g < full; g++, i += 4) {
      const x = (vals[i] << 18) | (vals[i + 1] << 12) | (vals[i + 2] << 6) | vals[i + 3];
      out[o++] = (x >>> 16) & 255; out[o++] = (x >>> 8) & 255; out[o++] = x & 255;
    }
    if (rem === 2) {
      out[o++] = ((vals[i] << 2) | (vals[i + 1] >> 4)) & 255;
    } else if (rem === 3) {
      out[o++] = ((vals[i] << 2) | (vals[i + 1] >> 4)) & 255;
      out[o++] = ((vals[i + 1] << 4) | (vals[i + 2] >> 2)) & 255;
    }
    return out;
  }

  /* ========================================================================== section 2: the span grammar */

  const B_COMMENT_OPEN = utf8("<!--");
  const B_COMMENT_CLOSE = utf8("-->");
  const B_OPENER_HEAD = utf8('<sworn r="');
  const B_OPENER_MID = utf8('" k="');
  const B_OPENER_END = utf8('">');
  const B_CLOSER = utf8("</sworn>");
  const B_SWORN = utf8("sworn");

  /** 2.2: { regions: [[start, endExclusive]], lines: [delimiter line numbers] }. */
  function fences(b) {
    const delims = [];
    const n = b.length;
    let pos = 0, lineNo = 1;
    for (;;) {
      const nl = b.indexOf(0x0a, pos);
      const end = nl < 0 ? n : nl;
      let i = pos;
      while (i < end && i - pos < 3 && b[i] === 0x20) i++;
      if (i + 3 <= end && b[i] === 0x60 && b[i + 1] === 0x60 && b[i + 2] === 0x60) {
        delims.push([lineNo, pos, nl < 0 ? n : nl + 1]);
      }
      if (nl < 0) break;
      pos = nl + 1;
      lineNo++;
    }
    const regions = [];
    for (let k = 0; k + 1 < delims.length; k += 2) regions.push([delims[k][1], delims[k + 1][2]]);
    return { regions, lines: delims.map((d) => d[0]) };
  }

  /** Every maximal backtick run: start -> [length, start of the next run on the same line with the
   *  same length, or -1]. */
  function backtickRuns(b) {
    const n = b.length;
    const runs = [];
    let p = b.indexOf(0x60);
    while (p >= 0) {
      let q = p;
      while (q < n && b[q] === 0x60) q++;
      runs.push([p, q - p]);
      p = b.indexOf(0x60, q);
    }
    const lineOf = new Array(runs.length);
    let lineStart = 0, lf = b.indexOf(0x0a);
    for (let k = 0; k < runs.length; k++) {
      const start = runs[k][0];
      while (lf >= 0 && lf < start) { lineStart = lf + 1; lf = b.indexOf(0x0a, lf + 1); }
      lineOf[k] = lineStart;
    }
    const info = new Map();
    const later = new Map();
    for (let k = runs.length - 1; k >= 0; k--) {
      const [start, length] = runs[k];
      const key = lineOf[k] + ":" + length;
      const nx = later.has(key) ? later.get(key) : -1;
      info.set(start, [length, nx]);
      later.set(key, start);
    }
    return info;
  }

  function _attr(b, i) {
    const n = b.length;
    while (i < n) {
      const c = b[i];
      if (c === 0x22) return i;
      if (c === 0x3c || c === 0x3e || c === 0x0d || c === 0x0a) return -1;
      i++;
    }
    return -1;
  }

  /** [REF, KIND, end] when an opener starts at p, else null. */
  function opener(b, p) {
    if (!startsWithBytes(b, B_OPENER_HEAD, p)) return null;
    const r0 = p + 10;
    const r1 = _attr(b, r0);
    if (r1 < 0 || !startsWithBytes(b, B_OPENER_MID, r1)) return null;
    const k0 = r1 + 5;
    const k1 = _attr(b, k0);
    if (k1 < 0 || !startsWithBytes(b, B_OPENER_END, k1)) return null;
    return [decodeStrict(b.subarray(r0, r1)), decodeStrict(b.subarray(k0, k1)), k1 + 2];
  }

  function _isTail(c) {
    return (c >= 0x41 && c <= 0x5a) || (c >= 0x61 && c <= 0x7a) || (c >= 0x30 && c <= 0x39) ||
      c === 0x5f || c === 0x2d;
  }

  function candidate(b, p) {
    let q = p + 1;
    if (q < b.length && b[q] === 0x2f) q++;
    if (q + 5 > b.length) return false;
    for (let k = 0; k < 5; k++) {
      let c = b[q + k];
      if (c >= 0x41 && c <= 0x5a) c += 0x20;
      if (c !== B_SWORN[k]) return false;
    }
    return q + 5 === b.length || !_isTail(b[q + 5]);
  }

  class Decl {
    constructor(at, receipt, kind, mark, raw) {
      this.at = at; this.receipt = receipt; this.kind = kind;
      this.mark = mark === undefined ? null : mark;
      this.raw = raw === undefined ? null : raw;
      this.innerStart = null; this.innerEnd = null; this.closerAt = null;
    }
  }

  /** 2.3 and 2.4 over a well-formed UTF-8 document with balanced fences: the declarations. */
  function scan(b) {
    const regions = fences(b).regions;
    const runs = backtickRuns(b);
    const n = b.length;
    const decls = [], stack = [];
    let unmarkedOpen = [];
    let p = 0, c = 0, ri = 0;
    let nextLt = -2, nextBt = -2;
    while (p < n) {
      // a. inside a fenced region
      while (ri < regions.length && regions[ri][1] <= p) ri++;
      if (ri < regions.length && regions[ri][0] <= p) { p = regions[ri][1]; continue; }
      const byte = b[p];
      // b. a comment opener at or after the comment end
      if (byte === 0x3c && p >= c && startsWithBytes(b, B_COMMENT_OPEN, p)) {
        const e = findBytes(b, B_COMMENT_CLOSE, p + 4);
        c = e < 0 ? n : e + 3;
        p += 4;
        continue;
      }
      // c. a backtick run (p is always at the start of a maximal run here)
      if (byte === 0x60) {
        const [length, nxt] = runs.get(p);
        p = nxt >= 0 ? nxt + length : p + length;
        continue;
      }
      // d. a candidate
      if (byte === 0x3c && candidate(b, p)) {
        const op = opener(b, p);
        if (op !== null) {
          const d = new Decl(p, op[0], op[1]);
          if (stack.length) {
            for (const o of unmarkedOpen) o.mark = "nesting";
            unmarkedOpen = [];
            d.mark = "nesting";
          } else if (p < c) {
            d.mark = "hidden_commitment";
          } else {
            unmarkedOpen.push(d);
          }
          d.innerStart = op[2];
          decls.push(d);
          stack.push(d);
          p = op[2];
          continue;
        }
        if (startsWithBytes(b, B_CLOSER, p)) {
          if (stack.length) {
            const d = stack.pop();
            if (unmarkedOpen.length && unmarkedOpen[unmarkedOpen.length - 1] === d) unmarkedOpen.pop();
            d.innerEnd = p;
            d.closerAt = p;
            if (p === d.innerStart && (d.mark === null || d.mark === "hidden_commitment")) d.mark = "empty_span";
          } else {
            decls.push(new Decl(p, null, null, "stray_closer", "</sworn>"));
          }
          p += 8;
          continue;
        }
        const lf = b.indexOf(0x0a, p);
        const lim = lf < 0 ? n : lf;
        let gt = b.indexOf(0x3e, p);
        if (gt >= lim) gt = -1;
        const end = gt >= 0 ? gt + 1 : lim;
        decls.push(new Decl(p, null, null, "tag_syntax", decodeStrict(b.subarray(p, end))));
        p = end;
        continue;
      }
      // e. advance: jump to the next byte at which a, b, c or d can apply
      if (nextLt < p) { nextLt = b.indexOf(0x3c, p + 1); if (nextLt < 0) nextLt = n; }
      if (nextBt < p) { nextBt = b.indexOf(0x60, p + 1); if (nextBt < 0) nextBt = n; }
      let nxt = Math.min(nextLt, nextBt);
      if (ri < regions.length) nxt = Math.min(nxt, regions[ri][0]);
      p = Math.max(nxt, p + 1);
    }
    for (const d of stack) {
      if (d.mark === null || d.mark === "hidden_commitment") d.mark = "unclosed";
    }
    decls.sort((x, y) => x.at - y.at);
    return decls;
  }

  const _LEXICAL = ["tag_syntax", "nesting", "stray_closer", "unclosed"];

  /** 2.5: [canonical bytes or null, Map decl -> [start, end]]. */
  function canonicalText(b, decls) {
    const offsets = new Map();
    for (const d of decls) if (_LEXICAL.indexOf(d.mark) >= 0) return [null, offsets];
    const parts = [];
    let last = 0, removed = 0;
    for (const d of decls) {
      const openerLen = d.innerStart - d.at;
      parts.push(b.subarray(last, d.at));
      parts.push(b.subarray(d.innerStart, d.innerEnd));
      const start = d.at - removed;
      offsets.set(d, [start, start + (d.innerEnd - d.innerStart)]);
      removed += openerLen + 8;
      last = d.closerAt + 8;
    }
    parts.push(b.subarray(last));
    return [concatBytes(parts), offsets];
  }

  /* ========================================================================== section 4: REF */

  function isPosInt(s) {
    return s.length > 0 && /^[1-9][0-9]*$/.test(s);
  }

  function digitsLt(x, y) {
    // x < y for digit strings without leading zeros
    if (x.length !== y.length) return x.length < y.length;
    return x < y;
  }

  const _PATH_BANNED = new Set(["\\", "*", "?", "[", "]", "\u007f", "\u0085", " ", " ",
    " ", " ", " ", " ", "　"]);
  for (let x = 0; x <= 0x20; x++) _PATH_BANNED.add(String.fromCharCode(x));
  for (let x = 0x2000; x <= 0x200a; x++) _PATH_BANNED.add(String.fromCharCode(x));

  function pathTargetOk(t) {
    if (!t || t[0] === "/" || t[0] === ":") return false;
    for (const ch of t) if (_PATH_BANNED.has(ch)) return false;
    for (const seg of t.split("/")) if (seg === "" || seg === "." || seg === "..") return false;
    return true;
  }

  /** {form, head, frag} with frag null, {lines: [a, b]} or {pointer: tokens}; null when the REF is
   *  receipt_form. a and b are digit strings. */
  function parseRef(ref) {
    const h = ref.indexOf("#");
    const head = h < 0 ? ref : ref.slice(0, h);
    const frag = h < 0 ? null : ref.slice(h + 1);
    if (frag === "") return null;
    let form;
    if (head.length >= 2 && head[0] === "r" && isPosInt(head.slice(1))) form = "rn";
    else if (head.startsWith("path:")) {
      if (!pathTargetOk(head.slice(5))) return null;
      form = "path";
    } else if (head.startsWith("prereg:")) {
      if (!isHex64Any(head.slice(7))) return null;
      form = "prereg";
    } else return null;
    let fragment = null;
    if (frag !== null) {
      if (frag[0] === "/") {
        for (let k = frag.indexOf("~"); k >= 0; k = frag.indexOf("~", k + 2)) {
          const nx = frag.slice(k + 1, k + 2);
          if (nx !== "0" && nx !== "1") return null;
        }
        const tokens = frag.slice(1).split("/").map((t) => t.split("~1").join("/").split("~0").join("~"));
        fragment = { pointer: tokens };
      } else {
        if (frag[0] !== "L") return null;
        const rest0 = frag.slice(1);
        const dash = rest0.indexOf("-");
        const a = dash < 0 ? rest0 : rest0.slice(0, dash);
        if (!isPosInt(a)) return null;
        let bb;
        if (dash >= 0) {
          const rest = rest0.slice(dash + 1);
          if (rest.slice(0, 1) !== "L" || !isPosInt(rest.slice(1))) return null;
          bb = rest.slice(1);
          if (digitsLt(bb, a)) return null;
        } else {
          bb = a;
        }
        fragment = { lines: [a, bb] };
      }
    }
    return { form, head, frag: fragment };
  }

  /* ========================================================================== section 5: the manifest */

  /** 5.5 and 5.6 over the manifest's bytes. Refuses (11.1) or returns what section 6 reads. */
  function readManifest(data) {
    let m;
    try {
      m = parseJson(data);
    } catch (e) {
      if (!(e instanceof JSONError)) throw e;
      // The text names no code for bytes that do not parse; they are not an object (11.1).
      throw new Refusal("manifest_spec", "the manifest bytes are not JSON under 6.13");
    }
    if (!(m instanceof JObj) || m.get("spec") !== MANIFEST_SPEC) throw new Refusal("manifest_spec");
    let receipts = m.get("receipts");
    if (receipts === null || receipts === undefined) receipts = new JObj();
    else if (!(receipts instanceof JObj)) throw new Refusal("manifest_receipts_shape");
    let authored = m.get("authored_sha256");
    if (authored === null || authored === undefined) authored = [];
    else if (!(Array.isArray(authored) && authored.every(isHex64Any))) {
      throw new Refusal("manifest_authored_shape");
    }
    for (const key of ["harness", "turn", "minted_at"]) {
      if (typeof m.get(key) !== "string") throw new Refusal("manifest_form", key + " missing or not a string");
    }
    if (hasDup(m)) throw new Refusal("manifest_form", "a key written twice");
    const rebuilt = authored.map((a) => a.replace(/[A-F]/g, (u) => u.toLowerCase())).sort(cmpCodePoint);
    const rung = m.has("rung") ? m.get("rung") : null;
    const core = {
      spec: m.get("spec"), harness: m.get("harness"), turn: m.get("turn"),
      minted_at: m.get("minted_at"), authored_sha256: rebuilt, receipts: receipts, rung: rung,
    };
    let dg = null;
    try {
      dg = digest(core);
    } catch (e) {
      if (!(e instanceof Refusal)) throw e;
      dg = null;
    }
    const declared = m.has("digest") ? m.get("digest") : null;
    const intact = dg !== null && (declared === null || (typeof declared === "string" && declared === dg));
    return { intact, digest: dg, harness: m.get("harness"), rung, receipts, authored: new Set(rebuilt) };
  }

  /* ========================================================================== section 7: the kinds */

  // 7.2: letters and numbers (\p{L}, \p{N} from the host's Unicode tables), the eight ASCII
  // marks, U+2212, U+00B1 and the 26 dash code points
  const TOKEN_RE = new RegExp("[\\p{L}\\p{N}_.,+\\-%/:\\u2212\\u00B1" +
    "\\u00AD\\u058A\\u05BE\\u1400\\u1806\\u2010\\u2011\\u2012\\u2013\\u2014\\u2015\\u2E17\\u2E1A" +
    "\\u2E3A\\u2E3B\\u2E40\\u2E5D\\u301C\\u3030\\u30A0\\uFE31\\uFE32\\uFE58\\uFE63\\uFF0D\\u{10EAD}]+", "gu");
  const DIGIT_RE = /\p{Nd}/u;

  function numericTokens(text) {
    const out = [];
    for (const m of text.matchAll(TOKEN_RE)) if (DIGIT_RE.test(m[0])) out.push(m[0]);
    return out;
  }

  const isAsciiDigits = (s) => /^[0-9]+$/.test(s);

  /** 7.2: [[reason, detail], null] or [null, {neg, intDigits, fracDigits, token}]. */
  function numericForm(text) {
    const found = numericTokens(text);
    const detail = { digit_bearing_tokens: found };
    if (found.length !== 1) return [["number_count", detail], null];
    let t = found[0];
    while (t.length && ".,:".indexOf(t[t.length - 1]) >= 0) t = t.slice(0, -1);
    const stripped = t;
    let neg = false;
    if (t[0] === "-" || t[0] === "+" || t[0] === "−") {
      neg = t[0] !== "+";
      t = t.slice(1);
    }
    if (t.endsWith("%")) t = t.slice(0, -1);
    const dotAt = t.indexOf(".");
    const ip = dotAt < 0 ? t : t.slice(0, dotAt);
    const fp = dotAt < 0 ? "" : t.slice(dotAt + 1);
    let ok = true;
    if (dotAt >= 0) ok = isAsciiDigits(fp);
    if (ip === "") ok = ok && dotAt >= 0;
    else if (ip.indexOf(",") >= 0) {
      const groups = ip.split(",");
      ok = ok && groups[0].length >= 1 && groups[0].length <= 3 && isAsciiDigits(groups[0]) &&
        groups.slice(1).every((g) => g.length === 3 && isAsciiDigits(g));
    } else ok = ok && isAsciiDigits(ip);
    if (!ok) return [["number_grammar", detail], null];
    return [null, { neg, intDigits: ip.split(",").join(""), fracDigits: fp, token: stripped }];
  }

  /** Plain notation of the integer `digits` scaled by 10^-f: exactly f fractional digits, a zero
   *  result unsigned. */
  function plain(neg, digits, f) {
    digits = digits.replace(/^0+/, "") || "0";
    let text;
    if (f) {
      digits = digits.padStart(f + 1, "0");
      text = digits.slice(0, digits.length - f) + "." + digits.slice(digits.length - f);
    } else {
      text = digits;
    }
    if (neg && /[1-9]/.test(digits)) text = "-" + text;
    return text;
  }

  /** A digit string plus one. */
  function inc(digits) {
    const out = digits.split("");
    let k = out.length - 1;
    while (k >= 0) {
      if (out[k] === "9") { out[k] = "0"; k--; }
      else { out[k] = String.fromCharCode(out[k].charCodeAt(0) + 1); return out.join(""); }
    }
    return "1" + out.join("");
  }

  /** A JSON number as coefficient and exponent (7.3). exp is a BigInt, or null when the written
   *  exponent has more than 4,000 digits (then expNeg and expText say which way and how far). */
  function decimalOf(text) {
    const neg = text[0] === "-";
    const t = neg ? text.slice(1) : text;
    const ei = t.search(/[eE]/);
    const mant = ei < 0 ? t : t.slice(0, ei);
    const ePart = ei < 0 ? "" : t.slice(ei + 1);
    const dot = mant.indexOf(".");
    const ip = dot < 0 ? mant : mant.slice(0, dot);
    const fp = dot < 0 ? "" : mant.slice(dot + 1);
    const d = { neg, coef: (ip + fp).replace(/^0+/, "") || "0", nfrac: fp.length,
      exp: null, expNeg: false, expText: null };
    const eNeg = ePart[0] === "-";
    const eDigits = ei < 0 ? "0" : (ePart.replace(/^[+-]/, "").replace(/^0+/, "") || "0");
    if (eDigits.length <= 4000) {
      const written = BigInt(eDigits);
      d.exp = (eNeg ? -written : written) - BigInt(d.nfrac);
      d.expNeg = d.exp < 0n;
    } else {
      d.expNeg = eNeg;
      d.expText = eDigits;
    }
    return d;
  }

  /** |a| (a digit string longer than 4,000 digits) plus the small integer k, as digits. */
  function strAddSmall(a, k) {
    // the magnitude is at least 10^4000, so adding a value bounded by a document's length never
    // changes sign; BigInt on the last 19 digits, carries into the head as digit strings
    let head = a.slice(0, a.length - 19);
    let tail = BigInt(a.slice(a.length - 19)) + BigInt(k);
    const base = 10n ** 19n;
    while (tail >= base) { head = inc(head); tail -= base; }
    while (tail < 0n) { head = decDigits(head); tail += base; }
    return ((head + tail.toString().padStart(19, "0")).replace(/^0+/, "")) || "0";
  }

  function decDigits(digits) {
    const out = digits.split("");
    let k = out.length - 1;
    while (out[k] === "0") { out[k] = "9"; k--; }
    out[k] = String.fromCharCode(out[k].charCodeAt(0) - 1);
    return out.join("").replace(/^0+/, "") || "0";
  }

  /** [negative, absolute adjusted exponent as digits]. */
  function adjustedText(d) {
    const shift = d.coef.length - 1;
    if (d.exp !== null) {
      const adj = d.exp + BigInt(shift);
      return [adj < 0n, (adj < 0n ? -adj : adj).toString()];
    }
    if (d.expNeg) return [true, strAddSmall(d.expText, d.nfrac - shift)];
    return [false, strAddSmall(d.expText, shift - d.nfrac)];
  }

  function tooLarge(d) {
    const [neg, mag] = adjustedText(d);
    return !neg && digitsLt(ADJUSTED_MAX, mag);
  }

  /** 7.3's `receipt`: the scientific string form of the General Decimal Arithmetic spec. */
  function sci(d) {
    const sign = d.neg ? "-" : "";
    const [adjNeg, adjMag] = adjustedText(d);
    if (d.exp !== null && d.exp <= 0n && (!adjNeg || !digitsLt("6", adjMag))) {
      const k = Number(-d.exp);
      if (k === 0) return sign + d.coef;
      if (d.coef.length > k) return sign + d.coef.slice(0, d.coef.length - k) + "." + d.coef.slice(d.coef.length - k);
      return sign + "0." + "0".repeat(k - d.coef.length) + d.coef;
    }
    const body = d.coef[0] + (d.coef.length > 1 ? "." + d.coef.slice(1) : "");
    return sign + body + "E" + (adjNeg ? "-" : "+") + adjMag;
  }

  /** round_half_even(|value| * 10^f) as a digit string, in exact decimal arithmetic. */
  function roundScaled(d, f) {
    if (d.coef === "0") return "0";
    if (d.exp === null) return "0";             // only reached for a huge negative exponent
    const s = d.exp + BigInt(f);
    if (s >= 0n) return d.coef + "0".repeat(Number(s));
    const kBig = -s;
    if (kBig > BigInt(d.coef.length)) return "0";
    const k = Number(kBig);
    let kept = d.coef.slice(0, d.coef.length - k) || "0";
    const dropped = d.coef.slice(d.coef.length - k);
    const first = dropped[0], rest = dropped.slice(1);
    let up;
    if (first > "5" || (first === "5" && /[1-9]/.test(rest))) up = true;
    else if (first === "5") up = "13579".indexOf(kept[kept.length - 1]) >= 0;
    else up = false;
    if (up) kept = inc(kept);
    return kept.replace(/^0+/, "") || "0";
  }

  function leafText(text) {
    const head = cpHead(text, 80);
    let out = "";
    for (let i = 0; i < head.length; i++) {
      const c = head.charCodeAt(i);
      if (c >= 0xd800 && c <= 0xdbff) {
        const e = i + 1 < head.length ? head.charCodeAt(i + 1) : 0;
        if (e >= 0xdc00 && e <= 0xdfff) { out += head[i] + head[i + 1]; i++; continue; }
        out += "?";
      } else if (c >= 0xdc00 && c <= 0xdfff) out += "?";
      else out += head[i];
    }
    return out;
  }

  function leafType(v) {
    if (v instanceof JObj) return "object";
    if (Array.isArray(v)) return "array";
    if (v === null) return "null";
    return null;
  }

  /** 7.3 and 7.3a: [verdict, reason, detail]. */
  function numericCompare(form, leaf) {
    const lt = leafType(leaf);
    if (lt !== null) return ["MALFORMED", "leaf_not_scalar", { leaf_type: lt }];
    if (typeof leaf === "string") return ["MALFORMED", "leaf_not_numeric", { leaf: leafText(leaf) }];
    if (leaf === true || leaf === false) {
      return ["MALFORMED", "leaf_not_numeric", { leaf: leaf ? "True" : "False" }];
    }
    if (leaf instanceof JConst) return ["MALFORMED", "leaf_not_numeric", { leaf: leaf.name }];
    const d = decimalOf(leaf.text);
    if (tooLarge(d)) return ["MALFORMED", "leaf_not_numeric", { leaf: leafText(sci(d)) }];
    const f = form.fracDigits.length;
    const printed = plain(form.neg, form.intDigits + form.fracDigits, f);
    const rounded = plain(d.neg, roundScaled(d, f), f);
    const detail = { printed_token: form.token, printed, receipt: sci(d), receipt_rounded: rounded,
      fractional_digits: f, rounding: "ROUND_HALF_EVEN" };
    if (printed !== rounded) return ["FAILED", "value_mismatch", detail];
    if (d.coef !== "0" && !/[1-9]/.test(rounded)) return ["FAILED", "rounds_to_zero", detail];  // 7.3a (D2, draft)
    return ["HELD", null, detail];
  }

  /** 2.3c's rule over the inner bytes: the contents of every inline code span, in order. */
  function codeSpans(inner) {
    const out = [];
    const n = inner.length;
    let p = 0;
    while (p < n) {
      if (inner[p] !== 0x60) { p++; continue; }
      let q = p;
      while (q < n && inner[q] === 0x60) q++;
      const run = q - p;
      const lf = inner.indexOf(0x0a, q);
      const end = lf < 0 ? n : lf;
      let r = q, found = -1;
      while (r < end) {
        if (inner[r] === 0x60) {
          const s0 = r;
          while (r < end && inner[r] === 0x60) r++;
          if (r - s0 === run) { found = s0; break; }
        } else r++;
      }
      if (found >= 0) { out.push(inner.subarray(q, found)); p = found + run; }
      else p = q;
    }
    return out;
  }

  function needleForm(inner) {
    const spans = codeSpans(inner);
    if (spans.length !== 1) return [["needle_count", {}], null];
    if (allAsciiWs(spans[0])) return [["needle_empty", {}], null];
    return [null, spans[0]];
  }

  /** 7.6. Hex digits and the glue class are ASCII, so code units decide as code points would. */
  function hashForm(text) {
    const runs = [];
    const n = text.length;
    let i = 0;
    while (i < n) {
      if (HEX_ANY.indexOf(text[i]) >= 0) {
        let j = i;
        while (j < n && HEX_ANY.indexOf(text[j]) >= 0) j++;
        const glued = (i > 0 && WORD_ASCII.indexOf(text[i - 1]) >= 0) ||
          (j < n && WORD_ASCII.indexOf(text[j]) >= 0);
        if (!glued) runs.push([j - i, text.slice(i, j)]);
        i = j;
      } else i++;
    }
    const lengths = runs.map((r) => r[0]);
    const n64 = lengths.filter((x) => x === 64).length;
    if (n64 !== 1 || lengths.some((x) => x === 32 || x === 40 || x === 96 || x === 128)) {
      return [["digest_form", { hex_runs: lengths }], null];
    }
    return [null, runs.filter((r) => r[0] === 64)[0][1].toLowerCase()];
  }

  /** 6.11: the number of lines and the slice [a, b] (digit strings) or null when out of range. */
  function lineSlice(data, a, bb) {
    let lfs = 0;
    for (let i = 0; i < data.length; i++) if (data[i] === 0x0a) lfs++;
    const count = lfs + (data.length && data[data.length - 1] !== 0x0a ? 1 : 0);
    if (bb.length > 18 || BigInt(bb) > BigInt(count)) return null;
    const na = Number(a), nb = Number(bb);
    // start of line na: after the (na-1)th LF; end of line nb: at the nb-th LF or the end
    let start = 0, seen = 0, i = 0;
    while (seen < na - 1) { i = data.indexOf(0x0a, i); seen++; i++; }
    start = i;
    let end = start;
    let left = nb - na + 1;
    let j = start;
    for (;;) {
      const lf = data.indexOf(0x0a, j);
      left--;
      if (left === 0) { end = lf < 0 ? data.length : lf; break; }
      j = lf + 1;
    }
    return data.subarray(start, end);
  }

  /** 6.12: [leaf, null] or [null, reason]. */
  function walk(root, tokens) {
    let v = root;
    for (const t of tokens) {
      if (v instanceof JObj) {
        if (!v.has(t)) return [null, "pointer_unresolvable"];
        if (v.dup.has(t)) return [null, "pointer_ambiguous"];
        v = v.get(t);
      } else if (Array.isArray(v)) {
        if (!(t === "0" || isPosInt(t)) || t.length > 18 || BigInt(t) >= BigInt(v.length)) {
          return [null, "pointer_unresolvable"];
        }
        v = v[Number(t)];
      } else {
        return [null, "pointer_unresolvable"];
      }
    }
    return [v, null];
  }

  /** 7.5: a line fragment narrows when its slice differs from the full-content slice, or when the
   *  whole receipt is under 16 bytes. Only a line fragment can narrow. */
  function narrows(frag, data, sliceBytes) {
    if (frag === null || !frag.lines) return false;
    if (data.length < MIN_NEEDLE) return true;
    let z = data.length;
    while (z > 0 && data[z - 1] === 0x0a) z--;
    if (z === 0) return false;                  // k = 0: nothing narrows but the clause above
    return !bytesEqual(sliceBytes, data.subarray(0, z));
  }

  /* ========================================================================== section 6 and 7.1: one declaration */

  function codePointLabel(ch) {
    return "U+" + ch.codePointAt(0).toString(16).toUpperCase().padStart(4, "0");
  }

  /** [verdict, reason, detail, extra]; extra holds resolved_sha256 and provenance when 6.14 says
   *  the row carries them. */
  function adjudicate(d, b, man) {
    if (d.mark !== null) {                                                          // 7.1a
      return ["MALFORMED", d.mark, d.raw !== null ? { raw: d.raw } : {}, null];
    }
    const inner = b.subarray(d.innerStart, d.innerEnd);
    const text = decodeStrict(inner);
    if (allAsciiWs(inner)) return ["MALFORMED", "empty_span", {}, null];            // 7.1b
    const cps = cpLen(text);
    if (cps > CAP) {                                                                 // 7.1c
      return ["MALFORMED", "length_cap", { code_points: cps, bytes: inner.length, cap: CAP }, null];
    }
    if (d.kind === "exec") return ["MALFORMED", "kind_reserved", { kind: d.kind }, null];   // 7.1d
    if (KINDS.indexOf(d.kind) < 0) return ["MALFORMED", "kind_unknown", { kind: d.kind }, null];
    const ref = parseRef(d.receipt);                                                 // 7.1e
    if (ref === null) return ["MALFORMED", "receipt_form", { receipt: d.receipt }, null];
    const frag = ref.frag;
    if (frag !== null && d.kind === "absent") {                                       // 7.1f
      return ["MALFORMED", "absent_over_partial", { receipt: d.receipt }, null];
    }
    if (frag !== null && d.kind === "hash") {
      return ["MALFORMED", "hash_over_partial", { receipt: d.receipt }, null];
    }
    for (const ch of text) {                                                         // 7.1g
      if (ch === "‭" || ch === "‮") {
        return ["MALFORMED", "directional_override", { code_point: codePointLabel(ch) }, null];
      }
    }
    let bad, formValue;                                                              // 7.1h
    if (d.kind === "numeric") [bad, formValue] = numericForm(text);
    else if (d.kind === "quote" || d.kind === "absent") [bad, formValue] = needleForm(inner);
    else [bad, formValue] = hashForm(text);
    if (bad !== null) return ["MALFORMED", bad[0], bad[1], null];
    if (ref.form !== "rn") return ["UNRESOLVED", "no_repository", {}, null];        // 7.1i, 4.4
    // section 6
    if (man === null) return ["UNRESOLVED", "manifest_absent", {}, null];            // 6.1
    if (!man.intact) return ["UNRESOLVED", "manifest_integrity", {}, null];          // 6.2
    if (!(man.rung === null || man.rung === "L1" || man.rung === "L2")) {            // 6.3
      return ["UNRESOLVED", "rung_unknown", {}, null];
    }
    if (!man.receipts.has(ref.head)) return ["UNRESOLVED", "manifest_id_missing", {}, null];  // 6.4
    const entry = man.receipts.get(ref.head);
    if (!(entry instanceof JObj) || !isHex64Lower(entry.get("sha256"))) {             // 6.5
      return ["UNRESOLVED", "manifest_integrity", {}, null];
    }
    const kos = entry.get("kind_of_source");
    if (typeof kos !== "string") return ["MALFORMED", "kind_of_source_unknown", {}, null];  // 6.6
    if (AUTHOR_SIDE.indexOf(kos) >= 0 || man.authored.has(entry.get("sha256"))) {   // 6.7
      return ["MALFORMED", "receipt_author_minted", {}, null];
    }
    if (EXTERNAL.indexOf(kos) < 0) return ["MALFORMED", "kind_of_source_unknown", {}, null];  // 6.8
    const extra = {                                                                   // 6.9
      resolved_sha256: entry.get("sha256"),
      provenance: { form: "rn", harness: man.harness, rung: man.rung !== null ? man.rung : "undeclared",
        kind_of_source: kos },
    };
    let data = null;                                                                  // 6.10
    if (entry.has("bytes")) {
      data = b64Strict(entry.get("bytes"));
      if (data === null || sha256(data) !== entry.get("sha256")) {
        return ["UNRESOLVED", "manifest_integrity", {}, null];
      }
    } else if (d.kind !== "hash") {
      return ["UNRESOLVED", "manifest_bytes_absent", {}, null];
    }
    let leaf = null, haveLeaf = false, haystack = data;
    if (frag !== null && frag.lines) {                                                // 6.11
      const sl = lineSlice(data, frag.lines[0], frag.lines[1]);
      if (sl === null) return ["MALFORMED", "anchor_out_of_range", {}, null];
      haystack = sl;
    } else if (frag !== null) {                                                       // 6.12
      let rootValue;
      try {
        rootValue = parseJson(data);
      } catch (e) {
        if (!(e instanceof JSONError)) throw e;
        return ["MALFORMED", "receipt_not_json", {}, null];
      }
      const [lf, why] = walk(rootValue, frag.pointer);
      if (why !== null) return ["MALFORMED", why, {}, null];
      leaf = lf;
      haveLeaf = true;
    }
    if (d.kind === "absent") {                                                        // 7.1j
      const complete = entry.get("complete");
      if (complete !== true && complete !== false) return ["UNRESOLVED", "manifest_no_completeness", {}, null];
      if (complete === false) return ["MALFORMED", "absent_over_partial", { complete: false }, extra];
    }
    // 7.1k
    if (d.kind === "numeric") {                                                       // 7.3
      if (!haveLeaf) {
        try {
          leaf = parseJson(haystack);
        } catch (e) {
          if (!(e instanceof JSONError)) throw e;
          return ["MALFORMED", "receipt_not_json", {}, extra];
        }
      }
      const [v, r, det] = numericCompare(formValue, leaf);
      return [v, r, det, extra];
    }
    if (d.kind === "quote") {                                                         // 7.5
      const needle = formValue;
      let hay;
      if (haveLeaf) {
        if (typeof leaf !== "string") return ["MALFORMED", "leaf_not_string", {}, extra];
        if (hasLoneSurrogate(leaf)) {
          return ["MALFORMED", "leaf_not_string", { note: "leaf is not encodable as UTF-8" }, extra];
        }
        hay = utf8(leaf);
      } else {
        hay = haystack;
        if (needle.length < MIN_NEEDLE && !narrows(frag, data, hay)) {
          return ["MALFORMED", "short_needle", { needle_bytes: needle.length, minimum_bytes: MIN_NEEDLE }, extra];
        }
      }
      const occ = countBytes(hay, needle);
      const det = { needle_bytes: needle.length, haystack_bytes: hay.length, occurrences: occ };
      return occ ? ["HELD", null, det, extra] : ["FAILED", "needle_missing", det, extra];
    }
    if (d.kind === "hash") {                                                          // 7.7
      const det = { stated: formValue, receipt_sha256: entry.get("sha256") };
      if (formValue === entry.get("sha256")) return ["HELD", null, det, extra];
      return ["FAILED", "digest_mismatch", det, extra];
    }
    const needle = formValue;                                                          // 7.8
    const occ = countBytes(data, needle);
    const det = { needle_bytes: needle.length, haystack_bytes: data.length, complete: true };
    if (occ) return ["FAILED", "needle_present", det, extra];
    return ["HELD", null, det, extra];
  }

  /* ========================================================================== sections 8, 9 and 11.1: the core */

  function documentVerdict(documentMalformed, counts, total) {
    if (documentMalformed !== null) return "SWORN-FAILED";
    if (total === 0) return "UNSWORN";
    if (counts.FAILED || counts.MALFORMED) return "SWORN-FAILED";
    if (counts.HELD === 0) return "SWORN-UNCHECKED";
    return "SWORN-HELD";
  }

  /** Sections 2 to 9 with 11.1's refusals. Returns the verdict core (a plain object). */
  function check(document, name, manifest, commit) {
    if (manifest === undefined) manifest = null;
    if (commit === undefined) commit = null;
    if (document === null || document === undefined) throw new Refusal("nothing_to_verify");
    const b = asBytes(document);
    if (typeof name !== "string" || hasLoneSurrogate(name)) throw new Refusal("name_form");
    if (!(commit === null || isHex40Lower(commit) || isHex64Lower(commit))) throw new Refusal("commit_form");
    let man = null;
    if (manifest !== null) man = readManifest(asBytes(manifest));
    const rows = [];
    let documentMalformed = null, canonicalSha256 = null;
    const badAt = firstInvalidUtf8(b);
    if (badAt >= 0) {                                                                // 2.1
      documentMalformed = { reason: "invalid_utf8", at: badAt };
    } else {
      const fl = fences(b);
      if (fl.lines.length % 2) {                                                      // 2.2
        documentMalformed = { reason: "unbalanced_fences", delimiter_lines: fl.lines };
      }
    }
    if (documentMalformed === null) {
      const decls = scan(b);
      const [canonical, offsets] = canonicalText(b, decls);
      if (canonical !== null) canonicalSha256 = sha256(canonical);
      for (const d of decls) {
        const [verdict, reason, detail, extra] = adjudicate(d, b, man);
        let start = null, end = null, claimSha = null, claim = null;
        if (offsets.has(d)) {
          [start, end] = offsets.get(d);
          const inner = b.subarray(d.innerStart, d.innerEnd);
          claimSha = sha256(inner);
          const text = decodeStrict(inner);
          claim = cpLen(text) <= CAP ? text : null;
        }
        const row = { at: d.at, start, end, receipt: d.receipt, kind: d.kind, claim_sha256: claimSha,
          claim, verdict, reason, detail };
        if (extra !== null) {
          row.resolved_sha256 = extra.resolved_sha256;
          row.provenance = extra.provenance;
        }
        rows.push(row);
      }
    }
    const counts = { HELD: 0, FAILED: 0, UNRESOLVED: 0, MALFORMED: 0 };
    const rungs = {};
    for (const r of rows) {
      counts[r.verdict] += 1;
      const key = r.provenance ? r.provenance.rung : "unresolved";
      rungs[key] = (rungs[key] || 0) + 1;
    }
    const core = {
      schema: SCHEMA_V0,
      profile: PROFILE_RN,
      format: FORMAT,
      document: { name, inline_sha256: sha256(b), canonical_sha256: canonicalSha256 },
      commit,
      manifest_digest: man !== null ? man.digest : null,
      spans: rows,
      counts,
      sworn_total: rows.length,
      unresolved: counts.UNRESOLVED,
      document_verdict: documentVerdict(documentMalformed, counts, rows.length),
      document_malformed: documentMalformed,
      rungs,
    };
    jcs(core);                                 // 11.2: a core with no canonical form is refused
    return core;
  }

  /** 9.4: the core beside its digest. `verifier` is provenance the digest does not cover. */
  function receipt(core, verifier) {
    const out = { core, digest: digest(core), attestations: [] };
    if (verifier !== undefined && verifier !== null) {
      if (!isPlainObject(verifier)) throw new TypeError("verifier is an object");
      out.verifier = verifier;
    }
    return out;
  }

  /* ========================================================================== 11.3: checking a receipt
   * A shape check written by hand from receipt-v0.schema.json (it does not read the schema file),
   * except `attestations` holding items: that is 11.3's own second refusal.
   */

  function isIntMin(v, lo) {
    if (!(v instanceof JNum) || !v.isIntegerToken()) return false;
    const t = v.text;
    if (t.replace(/^-+/, "").length > 30) return t[0] !== "-";
    return BigInt(t) >= BigInt(lo);
  }

  function isConstInt(v, n) {
    return v instanceof JNum && v.isIntegerToken() && v.text.length < 30 && BigInt(v.text) === BigInt(n);
  }

  function keysOk(obj, allowed, required) {
    if (!(obj instanceof JObj)) return false;
    for (const k of obj.map.keys()) if (allowed.indexOf(k) < 0) return false;
    for (const k of required) if (!obj.has(k)) return false;
    return true;
  }

  const DETAIL_TYPES = {
    raw: "string", code_points: "count", bytes: "count", cap: ["const", 300],
    kind: "string", receipt: "string", complete: "boolean", code_point: "code_point",
    digit_bearing_tokens: "strings", hex_runs: "counts", printed_token: "string",
    printed: "string", receipt_rounded: "string", fractional_digits: "count",
    rounding: ["str", "ROUND_HALF_EVEN"], needle_bytes: "count", minimum_bytes: ["const", 16],
    haystack_bytes: "count", occurrences: "count", leaf_type: "leaf_type", leaf: "leaf",
    note: "string", stated: "hex64", receipt_sha256: "hex64",
  };

  function detailOk(det) {
    if (!(det instanceof JObj)) return false;
    for (const [k, v] of det.map) {
      if (!Object.prototype.hasOwnProperty.call(DETAIL_TYPES, k)) return false;
      const t = DETAIL_TYPES[k];
      let ok;
      if (t === "string") ok = typeof v === "string";
      else if (t === "count") ok = isIntMin(v, 0);
      else if (t === "boolean") ok = v === true || v === false;
      else if (t === "code_point") ok = typeof v === "string" && /^U\+[0-9A-F]{4,6}$/.test(v);
      else if (t === "strings") ok = Array.isArray(v) && v.every((x) => typeof x === "string");
      else if (t === "counts") ok = Array.isArray(v) && v.every((x) => isIntMin(x, 0));
      else if (t === "leaf_type") ok = v === "object" || v === "array" || v === "null";
      else if (t === "leaf") ok = typeof v === "string" && cpLen(v) <= 80;
      else if (t === "hex64") ok = isHex64Lower(v);
      else if (t[0] === "const") ok = isConstInt(v, t[1]);
      else ok = typeof v === "string" && v === t[1];
      if (!ok) return false;
    }
    return true;
  }

  function provenanceOk(p, rnOnly) {
    const four = ["form", "harness", "rung", "kind_of_source"];
    if (keysOk(p, four, four)) {
      const rung = p.get("rung"), kos = p.get("kind_of_source");
      if (p.get("form") === "rn" && typeof p.get("harness") === "string" &&
          (rung === "L1" || rung === "L2" || rung === "undeclared") &&
          typeof kos === "string" && EXTERNAL.indexOf(kos) >= 0) return true;
    }
    if (rnOnly) return false;
    return keysOk(p, ["form", "note"], ["form", "note"]) &&
      (p.get("form") === "path" || p.get("form") === "prereg") && typeof p.get("note") === "string";
  }

  const ROW_KEYS = ["at", "start", "end", "receipt", "kind", "claim_sha256", "claim", "verdict",
    "reason", "detail"];

  function rowOk(r, rn) {
    if (!keysOk(r, ROW_KEYS.concat(["resolved_sha256", "provenance"]), ROW_KEYS)) return false;
    if (!isIntMin(r.get("at"), 0)) return false;
    for (const k of ["start", "end"]) if (!(r.get(k) === null || isIntMin(r.get(k), 0))) return false;
    for (const k of ["receipt", "kind"]) if (!(r.get(k) === null || typeof r.get(k) === "string")) return false;
    if (!(r.get("claim_sha256") === null || isHex64Lower(r.get("claim_sha256")))) return false;
    const claim = r.get("claim");
    if (!(claim === null || (typeof claim === "string" && cpLen(claim) <= CAP))) return false;
    const v = r.get("verdict");
    if (!(typeof v === "string" && SPAN_VERDICTS.indexOf(v) >= 0)) return false;
    const reason = r.get("reason");
    const allowed = rn ? REASONS : REASONS.concat(LEGACY_REASONS);
    if (!(reason === null || (typeof reason === "string" && allowed.indexOf(reason) >= 0))) return false;
    if (!detailOk(r.get("detail"))) return false;
    if (r.has("resolved_sha256") && !isHex64Lower(r.get("resolved_sha256"))) return false;
    if (r.has("provenance") && !provenanceOk(r.get("provenance"), rn)) return false;
    if (v === "HELD") {
      if (reason !== null || !r.has("resolved_sha256") || !r.has("provenance")) return false;
    } else if (typeof reason !== "string") return false;
    if (r.get("start") === null) {
      if (!(r.get("end") === null && r.get("claim_sha256") === null && claim === null)) return false;
    } else if (!(r.get("end") instanceof JNum)) return false;
    if (r.get("claim_sha256") === null && claim !== null) return false;
    return true;
  }

  const CORE_KEYS = ["schema", "profile", "format", "document", "commit", "manifest_digest", "spans",
    "counts", "sworn_total", "unresolved", "document_verdict", "document_malformed", "rungs"];

  function coreOk(c) {
    if (!keysOk(c, CORE_KEYS, CORE_KEYS)) return false;
    if (c.get("schema") !== SCHEMA_V0) return false;
    const profile = c.get("profile");
    if (!(profile === PROFILE_RN || profile === PROFILE_LEGACY)) return false;
    const rn = profile === PROFILE_RN;
    if (c.get("format") !== FORMAT) return false;
    const doc = c.get("document");
    const dk = ["name", "inline_sha256", "canonical_sha256"];
    if (!(keysOk(doc, dk, dk) && typeof doc.get("name") === "string" && isHex64Lower(doc.get("inline_sha256")) &&
          (doc.get("canonical_sha256") === null || isHex64Lower(doc.get("canonical_sha256"))))) return false;
    const commit = c.get("commit");
    if (!(commit === null || isHex40Lower(commit) || isHex64Lower(commit))) return false;
    if (!(c.get("manifest_digest") === null || isHex64Lower(c.get("manifest_digest")))) return false;
    const spans = c.get("spans");
    if (!(Array.isArray(spans) && spans.every((r) => rowOk(r, rn)))) return false;
    const counts = c.get("counts");
    if (!(keysOk(counts, SPAN_VERDICTS, SPAN_VERDICTS) && SPAN_VERDICTS.every((k) => isIntMin(counts.get(k), 0)))) return false;
    if (!(isIntMin(c.get("sworn_total"), 0) && isIntMin(c.get("unresolved"), 0))) return false;
    const dv = c.get("document_verdict");
    if (!(typeof dv === "string" && DOCUMENT_VERDICTS.indexOf(dv) >= 0)) return false;
    const dm = c.get("document_malformed");
    if (dm !== null) {
      if (keysOk(dm, ["reason", "at"], ["reason", "at"])) {
        if (!(dm.get("reason") === "invalid_utf8" && isIntMin(dm.get("at"), 0))) return false;
      } else if (keysOk(dm, ["reason", "delimiter_lines"], ["reason", "delimiter_lines"])) {
        const dl = dm.get("delimiter_lines");
        if (!(dm.get("reason") === "unbalanced_fences" && Array.isArray(dl) && dl.every((x) => isIntMin(x, 1)))) return false;
      } else return false;
    }
    const rg = c.get("rungs");
    if (!(rg instanceof JObj && rg.values().every((x) => isIntMin(x, 0)))) return false;
    if (rn) {
      for (const k of rg.keys()) if (["L1", "L2", "undeclared", "unresolved"].indexOf(k) < 0) return false;
    }
    return true;
  }

  function receiptOk(r) {
    if (!keysOk(r, ["core", "digest", "verifier", "attestations"], ["core", "digest", "attestations"])) return false;
    if (!isHex64Lower(r.get("digest"))) return false;
    if (r.has("verifier") && !(r.get("verifier") instanceof JObj)) return false;
    if (!Array.isArray(r.get("attestations"))) return false;
    return coreOk(r.get("core"));
  }

  /** 11.3: REDERIVES or DOES-NOT-REDERIVE, or a refusal. REDERIVES says the receipt is what this
   *  checker computes; it does not say the document held. */
  function recheck(receiptBytes, document, manifest) {
    if (manifest === undefined) manifest = null;
    if (!(receiptBytes instanceof Uint8Array)) throw new TypeError("the receipt is bytes");
    let r;
    try {
      r = parseJson(asBytes(receiptBytes));
    } catch (e) {
      if (!(e instanceof JSONError)) throw e;
      throw new Refusal("receipt_shape", "not JSON under 6.13");
    }
    if (hasDup(r)) throw new Refusal("receipt_shape", "a key written twice");
    if (!receiptOk(r)) throw new Refusal("receipt_shape");
    if (r.get("attestations").length) throw new Refusal("attestations_not_empty");
    let again;
    try {
      again = digest(r.get("core"));
    } catch (e) {
      if (!(e instanceof Refusal)) throw e;
      throw new Refusal("receipt_digest", "the core has no canonical form");
    }
    if (again !== r.get("digest")) throw new Refusal("receipt_digest");
    const core = r.get("core");
    if (core.get("profile") !== PROFILE_RN) throw new Refusal("profile_not_checkable");
    if (document === null || document === undefined) throw new Refusal("nothing_to_verify");
    const docBytes = asBytes(document);
    if (sha256(docBytes) !== core.get("document").get("inline_sha256")) throw new Refusal("document_mismatch");
    const mine = check(docBytes, core.get("document").get("name"), manifest, core.get("commit"));
    return digest(mine) === r.get("digest") ? REDERIVES : DOES_NOT_REDERIVE;
  }

  const VERIFIER = { name: "receipt/v0/check_v0.js", written_by: "fathom-lab", independent: false,
    status: "draft" };

  return {
    check, receipt, recheck, jcs, digest, parseJson, readManifest, sha256,
    Refusal, JSONError, JNum, JConst, JObj, REFUSALS, REASONS, LEGACY_REASONS, DOCUMENT_REASONS,
    SPAN_VERDICTS, DOCUMENT_VERDICTS, SCHEMA_V0, PROFILE_RN, PROFILE_LEGACY, FORMAT, MANIFEST_SPEC,
    REDERIVES, DOES_NOT_REDERIVE, VERIFIER,
    _internal: { b64Strict, firstInvalidUtf8, numericTokens, decimalOf, sci, roundScaled, TOKEN_RE, DIGIT_RE },
  };
});

/* ============================================================================ command line (Node only)
 * The only code in this file that touches a file, and it reads only the paths it is handed.
 */
if (typeof require === "function" && typeof module === "object" && require.main === module) {
  (function main(argv) {
    const C = module.exports;
    const fs = require("fs");
    const usage = "usage: node check_v0.js check DOC [--manifest FILE] [--name NAME] [--commit HEX]\n" +
      "       node check_v0.js recheck RECEIPT DOC [--manifest FILE]\n";
    const pos = [], opt = {};
    for (let i = 1; i < argv.length; i++) {
      const a = argv[i];
      if (a === "--manifest" || a === "--name" || a === "--commit") {
        if (i + 1 >= argv.length) { process.stderr.write(usage); process.exitCode = 64; return; }
        opt[a.slice(2)] = argv[++i];
      } else pos.push(a);
    }
    const cmd = argv[0];
    if (!((cmd === "check" && pos.length === 1) || (cmd === "recheck" && pos.length === 2)) ||
        (cmd === "recheck" && (opt.name !== undefined || opt.commit !== undefined))) {
      process.stderr.write(usage); process.exitCode = 64; return;
    }
    const read = (p) => new Uint8Array(fs.readFileSync(p));
    try {
      const man = opt.manifest !== undefined ? read(opt.manifest) : null;
      if (cmd === "check") {
        const core = C.check(read(pos[0]), opt.name !== undefined ? opt.name : pos[0], man,
          opt.commit !== undefined ? opt.commit : null);
        process.stdout.write(Buffer.from(new TextEncoder().encode(C.jcs(C.receipt(core, C.VERIFIER)) + "\n")));
        process.exitCode = 0;
        return;
      }
      const answer = C.recheck(read(pos[0]), read(pos[1]), man);
      process.stdout.write(answer + "\n");
      process.exitCode = answer === C.REDERIVES ? 0 : 1;
    } catch (e) {
      if (e instanceof C.Refusal) { process.stdout.write("REFUSED " + e.code + "\n"); process.exitCode = 2; return; }
      throw e;
    }
  })(process.argv.slice(2));
}
