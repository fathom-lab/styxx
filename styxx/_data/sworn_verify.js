"use strict";
/*
 * sworn_verify.js — a second implementation of the sworn verdict core, for a browser.
 *
 * SPEC: papers/sworn/SPEC_sworn_browser_verifier_v01_2026_09_05.md, frozen before this file.
 * Held to conformance/sworn/ — mode `inline`, receipts rN (and rN#/pointer) against an embedded
 * manifest, path:/prereg: with no tree (UNRESOLVED no_repository), all four kinds. It reproduces
 * the core digest or it does not ship; the bar is 1689 vectors and the harness prints the count.
 *
 * The label, from the plan, and it is the whole honest claim for this file:
 *
 *     re-derives sworn span verdicts offline; a forger controlling the whole file passes both
 *     browser layers; the package at the named commit is the check
 *
 * PURE (SPEC B5): one entry point, `swornVerify(documentBytes, manifestObject|null, opts)`.
 * No I/O, no clock, no globals, no network. Byte offsets are computed over Uint8Array, never
 * over JS string indices, because the format's offsets are byte offsets.
 *
 * Every place where Python and JavaScript disagree is marked `[B4]` with the row of the SPEC's
 * table it implements.
 */

/* ============================================================ sha256 (pure, synchronous)
 * crypto.subtle is async-only, and a capsule's layer 1 must return a verdict without awaiting a
 * promise chain through the lexer. 60 lines of FIPS 180-4 is the cheaper dependency.
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

function sha256Bytes(bytes) {
  const ml = bytes.length;
  const withOne = ml + 1;
  const padded = new Uint8Array(Math.ceil((withOne + 8) / 64) * 64);
  padded.set(bytes);
  padded[ml] = 0x80;
  const hi = Math.floor((ml * 8) / 0x100000000);
  const lo = (ml * 8) >>> 0;
  const dv = new DataView(padded.buffer);
  dv.setUint32(padded.length - 8, hi);
  dv.setUint32(padded.length - 4, lo);
  const H = new Uint32Array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                             0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]);
  const w = new Uint32Array(64);
  const rr = (x, n) => (x >>> n) | (x << (32 - n));
  for (let off = 0; off < padded.length; off += 64) {
    for (let i = 0; i < 16; i++) w[i] = dv.getUint32(off + i * 4);
    for (let i = 16; i < 64; i++) {
      const s0 = rr(w[i - 15], 7) ^ rr(w[i - 15], 18) ^ (w[i - 15] >>> 3);
      const s1 = rr(w[i - 2], 17) ^ rr(w[i - 2], 19) ^ (w[i - 2] >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
    }
    let [a, b, c, d, e, f, g, h] = H;
    for (let i = 0; i < 64; i++) {
      const S1 = rr(e, 6) ^ rr(e, 11) ^ rr(e, 25);
      const ch = (e & f) ^ (~e & g);
      const t1 = (h + S1 + ch + _K[i] + w[i]) >>> 0;
      const S0 = rr(a, 2) ^ rr(a, 13) ^ rr(a, 22);
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

/* ============================================================ bytes and text */

const _ENC = new TextEncoder();
const _DEC_STRICT = new TextDecoder("utf-8", { fatal: true });
const _DEC_LOOSE = new TextDecoder("utf-8");

function utf8(s) { return _ENC.encode(s); }

/** Strict decode: returns null when the bytes are not UTF-8 (the document-level MALFORMED). */
function decodeStrict(bytes) {
  try { return _DEC_STRICT.decode(bytes); } catch (e) { return null; }
}
function decodeLoose(bytes) { return _DEC_LOOSE.decode(bytes); }

/** The byte offset of the first invalid UTF-8 sequence — Python's UnicodeDecodeError.start. */
function firstInvalidUtf8(bytes) {
  let i = 0;
  const n = bytes.length;
  while (i < n) {
    const b = bytes[i];
    let need, min, cp;
    if (b < 0x80) { i++; continue; }
    else if (b >= 0xc2 && b <= 0xdf) { need = 1; cp = b & 0x1f; min = 0x80; }
    else if (b >= 0xe0 && b <= 0xef) { need = 2; cp = b & 0x0f; min = 0x800; }
    else if (b >= 0xf0 && b <= 0xf4) { need = 3; cp = b & 0x07; min = 0x10000; }
    else return i;
    if (i + need >= n + 0 && i + need > n - 1) return i;
    let ok = true;
    for (let k = 1; k <= need; k++) {
      const c = bytes[i + k];
      if (c === undefined || (c & 0xc0) !== 0x80) { ok = false; break; }
      cp = (cp << 6) | (c & 0x3f);
    }
    if (!ok) return i;
    if (cp < min || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) return i;
    i += need + 1;
  }
  return 0;
}

function bytesEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

/** indexOf for Uint8Array. */
function findBytes(hay, needle, from) {
  const n = needle.length;
  if (n === 0) return from || 0;
  const end = hay.length - n;
  for (let i = from || 0; i <= end; i++) {
    let ok = true;
    for (let k = 0; k < n; k++) if (hay[i + k] !== needle[k]) { ok = false; break; }
    if (ok) return i;
  }
  return -1;
}
function countBytes(hay, needle) {
  let c = 0, i = 0;
  for (;;) {
    const j = findBytes(hay, needle, i);
    if (j < 0) return c;
    c++;
    i = j + 1;               // Python bytes.count counts NON-overlapping…
    i = j + needle.length;   // …which is what this line does; the line above is superseded.
  }
}

const _WS_BYTES = new Set([0x20, 0x09, 0x0a, 0x0d, 0x0c, 0x0b]);
function stripWsBytes(b) {
  let a = 0, z = b.length;
  while (a < z && _WS_BYTES.has(b[a])) a++;
  while (z > a && _WS_BYTES.has(b[z - 1])) z--;
  return b.subarray(a, z);
}

/* ============================================================ decimal [B4 rows 3, 7, 8]
 * value = sign * coef * 10^exp, with `digits` the coefficient EXACTLY as written, because
 * Python's Decimal keeps them and str() prints them (Decimal("1.50") is "1.50", not "1.5").
 */
class Dec {
  constructor(sign, digits, exp, special) {
    this.sign = sign;            // 1 | -1
    this.digits = digits;        // digit string, as written (may carry leading zeros)
    this.exp = exp;              // integer
    this.special = special || null;   // null | "NaN" | "Infinity"
  }
  static parse(text) {
    const m = /^([-+]?)(?:(\d*)(?:\.(\d*))?|)(?:[eE]([-+]?\d+))?$/.exec(text);
    if (!m) return null;
    const sign = m[1] === "-" ? -1 : 1;
    const ip = m[2] || "";
    const fp = m[3] === undefined ? null : m[3];
    if (ip === "" && (fp === null || fp === "")) return null;
    const digits = ip + (fp || "");
    const exp = (fp ? -fp.length : 0) + (m[4] ? parseInt(m[4], 10) : 0);
    return new Dec(sign, digits === "" ? "0" : digits, exp, null);
  }
  isSpecial() { return this.special !== null; }
  isFinite() { return this.special === null; }
  /** Python's Decimal.adjusted(). */
  adjusted() { return this.exp + this.digits.length - 1; }
  isZero() { return this.special === null && /^0*$/.test(this.digits); }
  coef() { return BigInt(this.digits || "0"); }
  /** Python's str(Decimal) — to-scientific-string. [B4 row 7] */
  toString() {
    const sign = this.sign < 0 ? "-" : "";
    if (this.special === "NaN") return sign + "NaN";
    if (this.special === "Infinity") return sign + "Infinity";
    const int = this.digits;
    const leftdigits = this.exp + int.length;
    let dotplace;
    if (this.exp <= 0 && leftdigits > -6) dotplace = leftdigits;
    else dotplace = 1;
    let intpart, fracpart;
    if (dotplace <= 0) { intpart = "0"; fracpart = "." + "0".repeat(-dotplace) + int; }
    else if (dotplace >= int.length) { intpart = int + "0".repeat(dotplace - int.length); fracpart = ""; }
    else { intpart = int.slice(0, dotplace); fracpart = "." + int.slice(dotplace); }
    let exp = "";
    if (leftdigits !== dotplace) {
      const e = leftdigits - dotplace;
      exp = "E" + (e >= 0 ? "+" : "-") + Math.abs(e);
    }
    return sign + intpart + fracpart + exp;
  }
  /** quantize to 10^-frac with ROUND_HALF_EVEN, then format "f" — Python's _canon. [B4 row 8] */
  canon(frac) {
    if (!this.isFinite()) return null;
    let coef = this.coef();
    const shift = this.exp + frac;
    if (shift >= 0) {
      coef = coef * (10n ** BigInt(shift));
    } else {
      const p = 10n ** BigInt(-shift);
      const q = coef / p;
      const r = coef % p;
      const twice = r * 2n;
      if (twice > p || (twice === p && (q % 2n) === 1n)) coef = q + 1n;
      else coef = q;
    }
    let neg = this.sign < 0;
    if (coef === 0n) neg = false;                 // signed zero folded, as Python does
    let s = coef.toString();
    let body;
    if (frac === 0) body = s;
    else {
      if (s.length <= frac) s = "0".repeat(frac - s.length + 1) + s;
      body = s.slice(0, s.length - frac) + "." + s.slice(s.length - frac);
    }
    return (neg ? "-" : "") + body;
  }
}

/* ============================================================ JSON, Python-strict [B4 rows 3-6]
 * A reader that keeps number text as Dec, accepts NaN/Infinity/-Infinity as Python's json does,
 * remembers duplicate keys per object, refuses a BOM, and refuses raw control characters in
 * strings (Python's json strict=True). Objects are returned as { map: Map, dups: Set }.
 */
class JObj {
  constructor(map, dups) { this.map = map; this.dups = dups; }
}

function jsonStrict(text) {
  if (text.length && text.charCodeAt(0) === 0xfeff) throw new Error("BOM-prefixed JSON");
  let i = 0;
  const n = text.length;
  function ws() { while (i < n && (text[i] === " " || text[i] === "\t" || text[i] === "\n" || text[i] === "\r")) i++; }
  function fail(msg) { throw new Error(msg + " at " + i); }
  function parseValue() {
    ws();
    if (i >= n) fail("Expecting value");
    const c = text[i];
    if (c === "{") return parseObject();
    if (c === "[") return parseArray();
    if (c === '"') return parseString();
    if (text.startsWith("true", i)) { i += 4; return true; }
    if (text.startsWith("false", i)) { i += 5; return false; }
    if (text.startsWith("null", i)) { i += 4; return null; }
    if (text.startsWith("NaN", i)) { i += 3; return new Dec(1, "0", 0, "NaN"); }
    if (text.startsWith("Infinity", i)) { i += 8; return new Dec(1, "0", 0, "Infinity"); }
    if (text.startsWith("-Infinity", i)) { i += 9; return new Dec(-1, "0", 0, "Infinity"); }
    return parseNumber();
  }
  function parseNumber() {
    const m = /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][-+]?\d+)?/.exec(text.slice(i));
    if (!m) fail("Expecting value");
    i += m[0].length;
    const d = Dec.parse(m[0]);
    if (d === null) fail("Expecting value");
    return d;
  }
  function parseString() {
    if (text[i] !== '"') fail("Expecting string");
    i++;
    let out = "";
    for (;;) {
      if (i >= n) fail("Unterminated string");
      const c = text[i];
      if (c === '"') { i++; return out; }
      if (c === "\\") {
        i++;
        const e = text[i];
        i++;
        if (e === '"') out += '"';
        else if (e === "\\") out += "\\";
        else if (e === "/") out += "/";
        else if (e === "b") out += "\b";
        else if (e === "f") out += "\f";
        else if (e === "n") out += "\n";
        else if (e === "r") out += "\r";
        else if (e === "t") out += "\t";
        else if (e === "u") {
          const h = text.slice(i, i + 4);
          if (!/^[0-9a-fA-F]{4}$/.test(h)) fail("Invalid \\uXXXX escape");
          out += String.fromCharCode(parseInt(h, 16));
          i += 4;
        } else fail("Invalid \\escape");
        continue;
      }
      if (c.charCodeAt(0) < 0x20) fail("Invalid control character");
      out += c;
      i++;
    }
  }
  function parseObject() {
    i++;
    const map = new Map();
    const dups = new Set();
    ws();
    if (text[i] === "}") { i++; return new JObj(map, dups); }
    for (;;) {
      ws();
      const k = parseString();
      ws();
      if (text[i] !== ":") fail("Expecting ':'");
      i++;
      const v = parseValue();
      if (map.has(k)) dups.add(k);
      map.set(k, v);
      ws();
      if (text[i] === ",") { i++; continue; }
      if (text[i] === "}") { i++; return new JObj(map, dups); }
      fail("Expecting ',' delimiter");
    }
  }
  function parseArray() {
    i++;
    const arr = [];
    ws();
    if (text[i] === "]") { i++; return arr; }
    for (;;) {
      arr.push(parseValue());
      ws();
      if (text[i] === ",") { i++; continue; }
      if (text[i] === "]") { i++; return arr; }
      fail("Expecting ',' delimiter");
    }
  }
  const v = parseValue();
  ws();
  if (i !== n) fail("Extra data");
  return v;
}

/** The same reader, with numbers as JS numbers — what Python's plain json.loads hands
 *  Manifest.from_dict. The strict reader's Dec is for RECEIPT bytes, never for a manifest. */
function jsonPlain(text) {
  const v = jsonStrict(text);
  const walk = x => {
    if (x instanceof Dec) {
      if (!x.isFinite()) return x.special === "NaN" ? NaN
                                                    : (x.sign < 0 ? -Infinity : Infinity);
      return Number(x.toString());
    }
    if (Array.isArray(x)) return x.map(walk);
    if (x instanceof JObj) {
      const m = new Map();
      for (const [k, val] of x.map) m.set(k, walk(val));
      return new JObj(m, x.dups);
    }
    return x;
  };
  return walk(v);
}

function jsonPlainBytes(bytes) {
  const text = decodeStrict(bytes);
  if (text === null) throw new Error("not UTF-8");
  return jsonPlain(text);
}

function jsonStrictBytes(bytes) {
  const text = decodeStrict(bytes);
  if (text === null) throw new Error("not UTF-8");
  return jsonStrict(text);
}

/* ============================================================ JCS (RFC 8785), styxx domain */

function jcsString(s) {
  let out = '"';
  for (const ch of s) {
    const c = ch.codePointAt(0);
    if (ch === '"') out += '\\"';
    else if (ch === "\\") out += "\\\\";
    else if (c === 0x08) out += "\\b";
    else if (c === 0x0c) out += "\\f";
    else if (c === 0x0a) out += "\\n";
    else if (c === 0x0d) out += "\\r";
    else if (c === 0x09) out += "\\t";
    else if (c < 0x20) out += "\\u" + c.toString(16).padStart(4, "0");
    else out += ch;
  }
  return out + '"';
}

function jcs(obj) {
  if (obj === true) return "true";
  if (obj === false) return "false";
  if (obj === null || obj === undefined) return "null";
  if (typeof obj === "string") return jcsString(obj);
  if (typeof obj === "number") {
    if (!Number.isFinite(obj)) throw new Error("styxx JCS domain: NaN or Infinity");
    return String(obj);                    // ECMAScript Number::toString, as RFC 8785 requires
  }
  if (typeof obj === "bigint") return obj.toString();
  if (Array.isArray(obj)) return "[" + obj.map(jcs).join(",") + "]";
  if (obj instanceof Map) {
    const keys = [...obj.keys()].sort();
    return "{" + keys.map(k => jcsString(k) + ":" + jcs(obj.get(k))).join(",") + "}";
  }
  if (obj instanceof JObj) return jcs(obj.map);
  if (typeof obj === "object") {
    const keys = Object.keys(obj).sort();
    return "{" + keys.map(k => jcsString(k) + ":" + jcs(obj[k])).join(",") + "}";
  }
  throw new Error("not JCS-serializable");
}

/* ============================================================ constants, from the format */

const CERTIFIES =
  "the spans the author bound were checked against bytes the author did not write, at the commit " +
  "or manifest the document names and at the rung the manifest declares — NOT a claim that the " +
  "document is correct, NOT a claim that the right sentences were bound, NOT a check that the tags " +
  "were written at write time, NOT a check of any signature, and only as trustworthy as the harness " +
  "that minted the manifest and the history that holds the commit";
const SPEC = "sworn/0.1";
const RECEIPT_SCHEMA = "styxx.sworn.verdict-receipt/v1";
const MANIFEST_SPECS = ["sworn/manifest/0.1", "sworn/manifest/0.2"];
const SPAN_CAP_CODEPOINTS = 300;
const SHORT_NEEDLE_BYTES = 16;
const RUNGS = ["L1", "L2"];
const RUNG_UNDECLARED = "undeclared";
const KINDS = ["numeric", "quote", "hash", "absent"];
const RESERVED_KINDS = ["exec"];
const VERDICTS = ["HELD", "FAILED", "UNRESOLVED", "MALFORMED", "WITHHELD"];
const ROUNDING = "ROUND_HALF_EVEN";
const SOURCE_KINDS_EXTERNAL = new Set(["tool_stdout", "tool_stderr", "file_read", "http_fetch",
                                       "harness_note", "test_report", "attestation"]);
const SOURCE_KINDS_AUTHOR = new Set(["agent_output", "agent_file_write", "agent_message"]);

const OPENER_RE = /^<sworn r="([^"<>\r\n]*)" k="([^"<>\r\n]*)">/;
const CANDIDATE_RE = /^<(\/?)[sS][wW][oO][rR][nN](?![A-Za-z0-9_\-])/;
const CLOSER = utf8("</sworn>");
const COMMENT_OPEN = utf8("<!--");
const COMMENT_CLOSE = utf8("-->");

// [B4 row 1] Python's \w is Unicode-alphanumeric plus underscore; \d is category Nd.
const TOKEN_RE = /[\p{L}\p{N}_.,+\-−%/±:]+/gu;
const DIGIT_RE = /\p{Nd}/u;
const GRAM_RE = /^[-+−]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?|\.[0-9]+)%?$/;
const HEXRUN_RE = /(?<![A-Za-z0-9_])[0-9A-Fa-f]+(?![A-Za-z0-9_])/g;
const DIGEST_LENGTHS = new Set([32, 40, 96, 128]);
const RN_RE = /^r[1-9][0-9]*$/;
const PATH_SEG_BAD_RE = new RegExp('[\\\\\\t\\n\\v\\f\\r \\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000\\u0000-\\u001f\\u007f*?\\[\\]]');
const HEX64_RE = /^[0-9a-f]{64}$/;
const HEX64_ANY_RE = /^[0-9a-fA-F]{64}$/;

/* ============================================================ the lexer */

function fencedRegions(raw) {
  const regions = [];
  const delims = [];
  let openAt = null;
  let pos = 0, ln = 0;
  const n = raw.length;
  while (pos < n) {
    ln++;
    const nl = findBytes(raw, utf8("\n"), pos);
    const end = nl < 0 ? n : nl + 1;
    // _FENCE = ^ {0,3}`{3,}
    let p = pos, spaces = 0;
    while (p < end && raw[p] === 0x20 && spaces < 3) { p++; spaces++; }
    let ticks = 0;
    while (p + ticks < end && raw[p + ticks] === 0x60) ticks++;
    if (ticks >= 3) {
      delims.push(ln);
      if (openAt === null) openAt = pos;
      else { regions.push([openAt, end]); openAt = null; }
    }
    pos = end;
  }
  if (openAt !== null) return { regions, delims, balanced: false };
  return { regions, delims, balanced: true };
}

function inRegions(p, regions) {
  for (const [a, b] of regions) if (a <= p && p < b) return b;
  return null;
}

/** Length of the run of backticks starting at p. */
function tickRun(raw, p) {
  let k = 0;
  while (p + k < raw.length && raw[p + k] === 0x60) k++;
  return k;
}
/** First backtick run at or after p, before limit: [start, len] or null. */
function nextTicks(raw, p, limit) {
  for (let i = p; i < limit; i++) {
    if (raw[i] === 0x60) {
      let k = 0;
      while (i + k < raw.length && raw[i + k] === 0x60) k++;
      return [i, k];
    }
  }
  return null;
}

function scan(raw) {
  const out = { declarations: [], fenced: [], comments: [], document_malformed: null,
                canonical: null, lexical_ok: true, candidates: 0 };
  if (decodeStrict(raw) === null) {
    out.document_malformed = { reason: "invalid_utf8", at: firstInvalidUtf8(raw) };
    out.lexical_ok = false;
    return out;
  }
  const f = fencedRegions(raw);
  out.fenced = f.regions;
  if (!f.balanced) {
    out.document_malformed = { reason: "unbalanced_fences", delimiter_lines: f.delims };
    out.lexical_ok = false;
    return out;
  }
  const decls = [];
  const stack = [];
  let p = 0;
  const n = raw.length;
  let commentEnd = -1;
  while (p < n) {
    const skipTo = inRegions(p, f.regions);
    if (skipTo !== null) { p = skipTo; continue; }
    const c = raw[p];
    if (c === 0x3c && p >= commentEnd && findBytes(raw.subarray(p, p + 4), COMMENT_OPEN, 0) === 0) {
      const close = findBytes(raw, COMMENT_CLOSE, p + COMMENT_OPEN.length);
      commentEnd = close < 0 ? n : close + COMMENT_CLOSE.length;
      out.comments.push([p, commentEnd]);
      p += COMMENT_OPEN.length;
      continue;
    }
    if (c === 0x60) {
      const run = tickRun(raw, p);
      const nl = findBytes(raw, utf8("\n"), p + run);
      const lineEnd = nl < 0 ? n : nl;
      let q = p + run;
      let closed = null;
      while (q < lineEnd) {
        const m = nextTicks(raw, q, lineEnd);
        if (!m) break;
        if (m[1] === run) { closed = m[0] + m[1]; break; }
        q = m[0] + m[1];
      }
      p = closed !== null ? closed : p + run;
      continue;
    }
    if (c === 0x3c) {
      // the candidate/opener regexes run over a decoded window; the window is ASCII-delimited
      const windowEnd = Math.min(n, p + 4096);
      const text = decodeLoose(raw.subarray(p, windowEnd));
      const cm = CANDIDATE_RE.exec(text);
      if (cm) {
        out.candidates++;
        const om = OPENER_RE.exec(text);
        if (om) {
          const openerEnd = p + utf8(om[0]).length;
          const d = { at: p, opener_end: openerEnd, receipt: om[1], kind: om[2],
                      closer_at: null, closer_end: null, inner: null,
                      start: null, end: null, malformed: null };
          if (stack.length) {
            for (const o of stack) o.malformed = o.malformed || "nesting";
            d.malformed = "nesting";
          }
          if (p < commentEnd) d.malformed = d.malformed || "hidden_commitment";
          stack.push(d);
          decls.push(d);
          p = openerEnd;
          continue;
        }
        if (findBytes(raw.subarray(p, p + CLOSER.length), CLOSER, 0) === 0) {
          if (stack.length) {
            const d = stack.pop();
            d.closer_at = p;
            d.closer_end = p + CLOSER.length;
            d.inner = raw.subarray(d.opener_end, p);
            if (d.inner.length === 0 && (d.malformed === null || d.malformed === "hidden_commitment")) {
              d.malformed = "empty_span";
            }
          } else {
            decls.push({ at: p, receipt: null, kind: null, inner: null, start: null, end: null,
                         malformed: "stray_closer", raw: "</sworn>" });
          }
          p += CLOSER.length;
          continue;
        }
        const gt = findBytes(raw, utf8(">"), p);
        const nl2 = findBytes(raw, utf8("\n"), p);
        const stop = Math.min(gt >= 0 ? gt + 1 : n, nl2 >= 0 ? nl2 : n, n);
        decls.push({ at: p, receipt: null, kind: null, inner: null, start: null, end: null,
                     malformed: "tag_syntax", raw: decodeLoose(raw.subarray(p, stop)) });
        p = Math.max(stop, p + 1);
        continue;
      }
    }
    p++;
  }
  for (const d of stack) {
    if (d.malformed === null || d.malformed === "hidden_commitment") d.malformed = "unclosed";
  }
  out.declarations = decls;
  const lexicalBad = decls.filter(d => ["tag_syntax", "nesting", "stray_closer", "unclosed"]
                                        .includes(d.malformed));
  out.lexical_ok = lexicalBad.length === 0;
  if (out.lexical_ok) {
    const cuts = [];
    for (const d of decls) {
      cuts.push([d.at, d.opener_end]);
      cuts.push([d.closer_at, d.closer_end]);
    }
    cuts.sort((x, y) => (x[0] - y[0]) || (x[1] - y[1]));
    const pieces = [];
    let last = 0, removed = 0;
    const boundaries = new Map();
    for (const [a, b] of cuts) {
      pieces.push(raw.subarray(last, a));
      boundaries.set(a, a - removed);
      removed += b - a;
      boundaries.set(b, b - removed);
      last = b;
    }
    pieces.push(raw.subarray(last));
    let total = 0;
    for (const q of pieces) total += q.length;
    const canonical = new Uint8Array(total);
    let o = 0;
    for (const q of pieces) { canonical.set(q, o); o += q.length; }
    for (const d of decls) {
      d.start = boundaries.get(d.opener_end);
      d.end = boundaries.get(d.closer_at);
    }
    out.canonical = canonical;
  }
  return out;
}

/* ============================================================ the receipt grammar */

function parseReceipt(ref) {
  if (ref === null || ref === undefined) return [null, "receipt_form"];
  const head = ref.split("#")[0];
  let form, target, frag;
  if (RN_RE.test(head)) {
    form = "rn";
    const k = ref.indexOf("#");
    if (k >= 0) { target = ref.slice(0, k); frag = ref.slice(k + 1); }
    else { target = ref; frag = null; }
  } else if (ref.startsWith("path:")) {
    const body = ref.slice(5);
    form = "path";
    const k = body.indexOf("#");
    if (k >= 0) { target = body.slice(0, k); frag = body.slice(k + 1); }
    else { target = body; frag = null; }
    if (!target || target.startsWith("/") || target.startsWith(":") || PATH_SEG_BAD_RE.test(target)) {
      return [null, "receipt_form"];
    }
    for (const seg of target.split("/")) if (seg === "" || seg === "." || seg === "..") {
      return [null, "receipt_form"];
    }
  } else if (ref.startsWith("prereg:")) {
    const body = ref.slice(7);
    form = "prereg";
    const k = body.indexOf("#");
    if (k >= 0) { target = body.slice(0, k); frag = body.slice(k + 1); }
    else { target = body; frag = null; }
    if (!HEX64_ANY_RE.test(target)) return [null, "receipt_form"];
    target = target.toLowerCase();
  } else {
    return [null, "receipt_form"];
  }
  let fragment = null;
  if (frag !== null) {
    if (frag === "") return [null, "receipt_form"];
    if (frag.startsWith("/")) {
      if (/~(?![01])/.test(frag)) return [null, "receipt_form"];
      const toks = frag.split("/").slice(1)
        .map(t => t.replace(/~1/g, "/").replace(/~0/g, "~"));
      fragment = { type: "pointer", tokens: toks };
    } else {
      const m = /^L([1-9][0-9]*)(?:-L([1-9][0-9]*))?$/.exec(frag);
      if (!m) return [null, "receipt_form"];
      const first = parseInt(m[1], 10);
      const last = m[2] ? parseInt(m[2], 10) : first;
      if (last < first) return [null, "receipt_form"];
      fragment = { type: "lines", first, last };
    }
  }
  return [{ form, target, id: form === "rn" ? target : null, fragment,
            partial: fragment !== null }, null];
}

/** 1-based, inclusive, LF-split, CR retained; interior newlines kept, the LAST selected line's
 *  terminating LF excluded. null when a line is past EOF. Only 0x0a ever splits. */
function lineSlice(data, first, last) {
  const positions = [];
  for (let i = 0; i < data.length; i++) if (data[i] === 0x0a) positions.push(i);
  const nLines = positions.length +
                 ((data.length && data[data.length - 1] !== 0x0a) ? 1 : 0);
  if (first > nLines || last > nLines) return null;
  const begin = first === 1 ? 0 : positions[first - 2] + 1;
  const end = (last - 1 < positions.length) ? positions[last - 1] : data.length;
  return data.subarray(begin, end);
}

function walkPointer(obj, tokens) {
  for (const t of tokens) {
    if (obj instanceof JObj) {
      if (!obj.map.has(t)) return [null, "pointer_unresolvable"];
      if (obj.dups.has(t)) return [null, "pointer_ambiguous"];
      obj = obj.map.get(t);
    } else if (Array.isArray(obj)) {
      if (!/^(0|[1-9][0-9]*)$/.test(t) || parseInt(t, 10) >= obj.length) {
        return [null, "pointer_unresolvable"];
      }
      obj = obj[parseInt(t, 10)];
    } else {
      return [null, "pointer_unresolvable"];
    }
  }
  return [obj, "ok"];
}

/* ============================================================ the manifest */

function b64decodeStrict(s) {
  if (typeof s !== "string") return null;
  if (!/^[A-Za-z0-9+/]*={0,2}$/.test(s) || s.length % 4 !== 0) return null;   // validate=True
  const table = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  const out = [];
  for (let i = 0; i < s.length; i += 4) {
    const c = [0, 1, 2, 3].map(k => {
      const ch = s[i + k];
      return ch === "=" ? -1 : table.indexOf(ch);
    });
    if (c[0] < 0 || c[1] < 0) return null;
    out.push((c[0] << 2) | (c[1] >> 4));
    if (c[2] >= 0) out.push(((c[1] & 15) << 4) | (c[2] >> 2));
    if (c[3] >= 0) out.push(((c[2] & 3) << 6) | c[3]);
  }
  return new Uint8Array(out);
}

class Manifest {
  constructor(d) {
    // Mirrors Manifest.from_dict: the same normalisation, the same refusals.
    if (!(d instanceof JObj)) throw new Error("REFUSED: manifest is not an object");
    const g = k => d.map.has(k) ? d.map.get(k) : undefined;
    const spec = g("spec");
    if (!MANIFEST_SPECS.includes(spec)) throw new Error("REFUSED: unknown manifest spec");
    let receipts = g("receipts");
    if (receipts === undefined || receipts === null) receipts = new JObj(new Map(), new Set());
    let authored = g("authored_sha256");
    if (authored === undefined || authored === null) authored = [];
    if (!(receipts instanceof JObj)) throw new Error("REFUSED: manifest receipts must be an object");
    if (!Array.isArray(authored) || !authored.every(x => typeof x === "string")) {
      throw new Error("REFUSED: manifest authored_sha256 must be a list of hex strings");
    }
    this.spec = spec;
    this.harness = g("harness") === undefined ? "" : g("harness");
    this.turn = g("turn") === undefined ? "" : g("turn");
    this.minted_at = g("minted_at") === undefined ? null : g("minted_at");
    this.receipts = receipts;
    this.authored_sha256 = authored.map(x => x.toLowerCase());
    this.rung = g("rung") === undefined ? null : g("rung");
    this.declared_digest = g("digest") === undefined ? null : g("digest");
  }
  core() {
    const m = new Map();
    m.set("spec", this.spec);
    m.set("harness", this.harness);
    m.set("turn", this.turn);
    m.set("minted_at", this.minted_at);
    m.set("authored_sha256", [...this.authored_sha256].sort());
    m.set("receipts", this.receipts);
    if (this.spec === "sworn/manifest/0.2") m.set("rung", this.rung);
    return m;
  }
  digestOrNull() {
    try { return sha256Bytes(utf8(jcs(this.core()))); } catch (e) { return null; }
  }
  intact() {
    const d = this.digestOrNull();
    if (this.declared_digest === null || this.declared_digest === undefined) return d !== null;
    if (d === null) return false;
    return this.declared_digest === d;
  }
  rungStatus() {
    if (this.rung === null || this.rung === undefined) return ["undeclared", null];
    if (RUNGS.includes(this.rung)) return ["ok", this.rung];
    return ["unknown", String(this.rung)];
  }
}

/* ============================================================ resolution */

function resolve(parsed, kind, manifest) {
  if (parsed.form === "rn") {
    if (manifest === null) return { status: "unresolved", reason: "manifest_absent" };
    if (!manifest.intact()) return { status: "unresolved", reason: "manifest_integrity" };
    const [rungState, rung] = manifest.rungStatus();
    if (rungState === "unknown") return { status: "unresolved", reason: "rung_unknown" };
    if (!manifest.receipts.map.has(parsed.id)) {
      return { status: "unresolved", reason: "manifest_id_missing" };
    }
    const entry = manifest.receipts.map.get(parsed.id);
    if (!(entry instanceof JObj)) return { status: "unresolved", reason: "manifest_integrity" };
    const e = k => entry.map.has(k) ? entry.map.get(k) : undefined;
    const sha = e("sha256");
    if (typeof sha !== "string" || !HEX64_RE.test(sha)) {
      return { status: "unresolved", reason: "manifest_integrity" };
    }
    const kos = e("kind_of_source");
    if (typeof kos !== "string") return { status: "malformed", reason: "kind_of_source_unknown" };
    if (SOURCE_KINDS_AUTHOR.has(kos) || manifest.authored_sha256.includes(sha)) {
      return { status: "malformed", reason: "receipt_author_minted" };
    }
    if (!SOURCE_KINDS_EXTERNAL.has(kos)) {
      return { status: "malformed", reason: "kind_of_source_unknown" };
    }
    const provenance = { form: "rn", harness: manifest.harness,
                         rung: rungState === "ok" ? rung : RUNG_UNDECLARED,
                         kind_of_source: kos };
    const completeRaw = e("complete");
    const complete = typeof completeRaw === "boolean" ? completeRaw : null;
    let data = null;
    if (entry.map.has("bytes")) {
      data = b64decodeStrict(e("bytes"));
      if (data === null) return { status: "unresolved", reason: "manifest_integrity" };
      if (sha256Bytes(data) !== sha) return { status: "unresolved", reason: "manifest_integrity" };
    } else if (kind !== "hash") {
      return { status: "unresolved", reason: "manifest_bytes_absent" };
    }
    return finishResolve({ status: "ok", reason: null, bytes: data, sha256: sha, complete,
                           leaf: null, has_leaf: false, slice: null, provenance }, parsed);
  }
  // path: / prereg: with no tree — SPEC B1: the browser has no repository, and says so.
  return { status: "unresolved", reason: "no_repository" };
}

function finishResolve(res, parsed) {
  const frag = parsed.fragment;
  if (frag === null || res.bytes === null) return res;
  if (frag.type === "lines") {
    const sl = lineSlice(res.bytes, frag.first, frag.last);
    if (sl === null) return { status: "malformed", reason: "anchor_out_of_range" };
    res.slice = sl;
    return res;
  }
  let obj;
  try { obj = jsonStrictBytes(res.bytes); }
  catch (e) { return { status: "malformed", reason: "receipt_not_json" }; }
  const [leaf, why] = walkPointer(obj, frag.tokens);
  if (why !== "ok") return { status: "malformed", reason: why };
  res.leaf = leaf;
  res.has_leaf = true;
  return res;
}

/* ============================================================ the kinds */

function numberToken(text) {
  const all = text.match(TOKEN_RE) || [];
  const digitBearing = all.filter(t => DIGIT_RE.test(t));
  if (digitBearing.length !== 1) return ["number_count", null, digitBearing];
  let tok = digitBearing[0];
  tok = tok.replace(/[.,:]+$/, "");                 // Python's rstrip(".,:")
  if (!GRAM_RE.test(tok)) return ["number_grammar", null, digitBearing];
  return [null, tok, digitBearing];
}

function needleIn(inner) {
  const spans = [];
  let p = 0;
  const n = inner.length;
  while (p < n) {
    const m = nextTicks(inner, p, n);
    if (!m) break;
    const [mstart, run] = m;
    const mend = mstart + run;
    const nl = findBytes(inner, utf8("\n"), mend);
    const lineEnd = nl < 0 ? n : nl;
    let q = mend;
    let closed = null;
    while (q < lineEnd) {
      const m2 = nextTicks(inner, q, lineEnd);
      if (!m2) break;
      if (m2[1] === run) { closed = m2; break; }
      q = m2[0] + m2[1];
    }
    if (closed === null) { p = mend; continue; }
    spans.push(inner.subarray(mend, closed[0]));
    p = closed[0] + closed[1];
  }
  if (spans.length !== 1) return [null, "needle_count"];
  if (stripWsBytes(spans[0]).length === 0) return [null, "needle_empty"];
  return [spans[0], "ok"];
}

function printedDecimal(tok) {
  const t = tok.replace(/,/g, "").replace(/−/g, "-").replace(/%+$/, "");
  const d = Dec.parse(t);
  const frac = t.includes(".") ? t.split(".")[1].length : 0;
  return [d, frac];
}

function safeText(x, limit) {
  const s = String(x).slice(0, limit === undefined ? 80 : limit);
  // Python: .encode("utf-8", errors="replace").decode("utf-8") — a lone surrogate becomes U+FFFD
  return s.replace(/[\ud800-\udfff](?![\udc00-\udfff])/g, "�")
          .replace(/(^|[^\ud800-\udbff])[\udc00-\udfff]/g, (m, p1) => p1 + "�");
}

function checkNumeric(innerText, res) {
  const [why, tok, seen] = numberToken(innerText);
  if (why) return ["MALFORMED", why, { digit_bearing_tokens: seen }];
  const [printed, frac] = printedDecimal(tok);
  let leaf;
  if (res.has_leaf) leaf = res.leaf;
  else {
    const source = res.slice !== null ? res.slice : res.bytes;
    try { leaf = jsonStrictBytes(source); }
    catch (e) { return ["MALFORMED", "receipt_not_json", {}]; }
  }
  if (leaf instanceof JObj || Array.isArray(leaf) || leaf === null) {
    // Python's _json_strict builds objects as class _Obj(dict); type(leaf).__name__ is "_Obj"
    const t = leaf === null ? "NoneType" : (Array.isArray(leaf) ? "list" : "_Obj");
    return ["MALFORMED", "leaf_not_scalar", { leaf_type: t }];
  }
  if (!(leaf instanceof Dec)) {
    return ["MALFORMED", "leaf_not_numeric", { leaf: safeText(pyRepr(leaf)) }];
  }
  if (!leaf.isFinite() || leaf.adjusted() > 320) {
    return ["MALFORMED", "leaf_not_numeric", { leaf: safeText(leaf.toString()) }];
  }
  const lhs = leaf.canon(frac);
  const rhs = printed === null ? null : printed.canon(frac);
  if (lhs === null || rhs === null) {
    return ["MALFORMED", "leaf_not_numeric", { leaf: safeText(leaf.toString()) }];
  }
  const detail = { printed_token: tok, printed: rhs, receipt: leaf.toString(),
                   receipt_rounded: lhs, fractional_digits: frac, rounding: ROUNDING };
  if (lhs === rhs) return ["HELD", null, detail];
  return ["FAILED", "value_mismatch", detail];
}

/** Python's str() for the scalars a JSON leaf can hold, for the `leaf` detail string. */
function pyRepr(x) {
  if (x === true) return "True";
  if (x === false) return "False";
  if (x === null) return "None";
  return String(x);
}

function checkQuote(inner, res) {
  const [needle, why] = needleIn(inner);
  if (needle === null) return ["MALFORMED", why, {}];
  let hay;
  if (res.has_leaf) {
    if (typeof res.leaf !== "string") return ["MALFORMED", "leaf_not_string", {}];
    // [B4 row 9] a lone surrogate is not encodable as UTF-8; Python raises and refuses.
    if (/[\ud800-\udbff](?![\udc00-\udfff])|(?:^|[^\ud800-\udbff])[\udc00-\udfff]/.test(res.leaf)) {
      return ["MALFORMED", "leaf_not_string", { note: "leaf is not encodable as UTF-8" }];
    }
    hay = utf8(res.leaf);
  } else {
    hay = res.slice !== null ? res.slice : res.bytes;
    if (res.slice === null && needle.length < SHORT_NEEDLE_BYTES) {
      return ["MALFORMED", "short_needle", { needle_bytes: needle.length,
                                             minimum_bytes: SHORT_NEEDLE_BYTES }];
    }
  }
  const detail = { needle_bytes: needle.length, haystack_bytes: hay.length,
                   occurrences: countBytes(hay, needle) };
  if (findBytes(hay, needle, 0) >= 0) return ["HELD", null, detail];
  return ["FAILED", "needle_missing", detail];
}

function checkAbsent(inner, res) {
  const [needle, why] = needleIn(inner);
  if (needle === null) return ["MALFORMED", why, {}];
  const hay = res.bytes;
  const detail = { needle_bytes: needle.length, haystack_bytes: hay.length, complete: true };
  if (findBytes(hay, needle, 0) >= 0) return ["FAILED", "needle_present", detail];
  return ["HELD", null, detail];
}

function hexRuns(innerText) {
  HEXRUN_RE.lastIndex = 0;
  const runs = [];
  let m;
  while ((m = HEXRUN_RE.exec(innerText)) !== null) runs.push(m[0]);
  return runs;
}

function checkHash(innerText, res) {
  const runs = hexRuns(innerText);
  const sixtyFour = runs.filter(r => r.length === 64);
  const others = runs.filter(r => DIGEST_LENGTHS.has(r.length));
  if (sixtyFour.length !== 1 || others.length) {
    return ["MALFORMED", "digest_form", { hex_runs: runs.map(r => r.length) }];
  }
  const stated = sixtyFour[0].toLowerCase();
  const detail = { stated, receipt_sha256: res.sha256 };
  if (stated === res.sha256) return ["HELD", null, detail];
  return ["FAILED", "digest_mismatch", detail];
}

/* ============================================================ the adjudicator */

function adjudicate(d, manifest) {
  const verdict = { at: d.at, start: d.start === undefined ? null : d.start,
                    end: d.end === undefined ? null : d.end,
                    receipt: d.receipt === undefined ? null : d.receipt,
                    kind: d.kind === undefined ? null : d.kind,
                    verdict: null, reason: null, detail: {} };
  const out = (v, reason, detail, res) => {
    verdict.verdict = v;
    verdict.reason = reason === undefined ? null : reason;
    verdict.detail = detail || {};
    if (res && res.sha256) verdict.resolved_sha256 = res.sha256;
    if (res && res.provenance) verdict.provenance = res.provenance;
    return verdict;
  };
  if (d.malformed) {
    return out("MALFORMED", d.malformed, d.raw ? { raw: d.raw } : {});
  }
  const inner = d.inner;
  if (stripWsBytes(inner).length === 0) return out("MALFORMED", "empty_span");
  const innerText = decodeStrict(inner);
  const codePoints = [...innerText].length;
  if (codePoints > SPAN_CAP_CODEPOINTS) {
    return out("MALFORMED", "length_cap", { code_points: codePoints, bytes: inner.length,
                                            cap: SPAN_CAP_CODEPOINTS });
  }
  const kind = d.kind;
  if (RESERVED_KINDS.includes(kind)) return out("MALFORMED", "kind_reserved", { kind });
  if (!KINDS.includes(kind)) return out("MALFORMED", "kind_unknown", { kind });
  const [parsed, why] = parseReceipt(d.receipt);
  if (parsed === null) return out("MALFORMED", why, { receipt: d.receipt });
  if (kind === "absent" && parsed.partial) {
    return out("MALFORMED", "absent_over_partial", { receipt: d.receipt });
  }
  if (kind === "hash" && parsed.partial) {
    return out("MALFORMED", "hash_over_partial", { receipt: d.receipt });
  }
  // bytes-only form checks run BEFORE any receipt is opened
  if (kind === "numeric") {
    const [nwhy, , seen] = numberToken(innerText);
    if (nwhy) return out("MALFORMED", nwhy, { digit_bearing_tokens: seen });
  } else if (kind === "quote" || kind === "absent") {
    const [needle, nwhy] = needleIn(inner);
    if (needle === null) return out("MALFORMED", nwhy);
  } else {
    const runs = hexRuns(innerText);
    if (runs.filter(r => r.length === 64).length !== 1 ||
        runs.some(r => DIGEST_LENGTHS.has(r.length) && r.length !== 64)) {
      return out("MALFORMED", "digest_form", { hex_runs: runs.map(r => r.length) });
    }
  }
  const res = resolve(parsed, kind, manifest);
  if (res.status === "unresolved") return out("UNRESOLVED", res.reason);
  if (res.status === "malformed") return out("MALFORMED", res.reason, null, res);
  if (kind === "absent") {
    // a negative is only swearable over an object the harness said it captured whole
    if (res.complete === null || res.complete === undefined) {
      return out("UNRESOLVED", "manifest_no_completeness");
    }
    if (res.complete !== true) {
      return out("MALFORMED", "absent_over_partial", { complete: false }, res);
    }
  }
  let v, reason, detail;
  if (kind === "numeric") [v, reason, detail] = checkNumeric(innerText, res);
  else if (kind === "quote") [v, reason, detail] = checkQuote(inner, res);
  else if (kind === "absent") [v, reason, detail] = checkAbsent(inner, res);
  else [v, reason, detail] = checkHash(innerText, res);
  return out(v, reason, detail, res);
}

/* ============================================================ the entry point */

/**
 * Re-derive the verdict core.
 *
 * @param {Uint8Array} documentBytes  the inline document, bytes
 * @param {object|null} manifestObj   the manifest as a JObj (from jsonStrictBytes) or null
 * @param {object} opts               { name, commit }
 * @returns {object} the core, with `spans`, `counts`, `document_verdict` … and no `coverage`
 *                   and no `verifier` — the two blocks the digest excludes.
 */
function swornVerify(documentBytes, manifestObj, opts) {
  const o = opts || {};
  const name = o.name === undefined ? "" : o.name;
  const commit = o.commit === undefined ? null : o.commit;
  if (commit !== null && !(typeof commit === "string" && /^([0-9a-f]{40}|[0-9a-f]{64})$/.test(commit))) {
    throw new Error("REFUSED: commit must be a full lowercase hex object id or None");
  }
  let manifest = null;
  if (manifestObj !== null && manifestObj !== undefined) {
    manifest = manifestObj instanceof Manifest ? manifestObj : new Manifest(manifestObj);
  }
  const sc = scan(documentBytes);
  const verdicts = sc.declarations.map(d => adjudicate(d, manifest));
  verdicts.sort((a, b) => a.at - b.at);
  const counts = {};
  for (const v of VERDICTS) counts[v] = 0;
  for (const v of verdicts) counts[v.verdict]++;
  let swornTotal = 0;
  for (const v of VERDICTS) swornTotal += counts[v];
  const docMalformed = sc.document_malformed;
  let documentVerdict;
  if (docMalformed) documentVerdict = "SWORN-FAILED";
  else if (swornTotal === 0) documentVerdict = "UNSWORN";
  else if (counts.FAILED === 0 && counts.MALFORMED === 0) documentVerdict = "SWORN-HELD";
  else documentVerdict = "SWORN-FAILED";
  const rungs = {};
  for (const v of verdicts) {
    const prov = v.provenance || {};
    let key;
    if (prov.form === "rn") key = prov.rung;
    else if (prov.form === "path" || prov.form === "prereg") key = "committed";
    else key = "unresolved";
    rungs[key] = (rungs[key] || 0) + 1;
  }
  return {
    schema: RECEIPT_SCHEMA,
    format: SPEC,
    document: { name,
                inline_sha256: sha256Bytes(documentBytes),
                canonical_sha256: sc.canonical !== null ? sha256Bytes(sc.canonical) : null },
    commit,
    manifest_digest: manifest !== null ? manifest.digestOrNull() : null,
    spans: verdicts,
    counts,
    sworn_total: swornTotal,
    unresolved: counts.UNRESOLVED,
    document_verdict: documentVerdict,
    document_malformed: docMalformed,
    rungs,
    certifies: CERTIFIES,
  };
}

/** sha256(utf8(jcs(core))) — the number the conformance vectors pin (SPEC B2). */
function coreDigest(core) {
  return sha256Bytes(utf8(jcs(core)));
}

const _api = { swornVerify, coreDigest, jcs, jcsString, sha256Bytes, jsonStrict, jsonStrictBytes,
               jsonPlain, jsonPlainBytes,
               Manifest, Dec, JObj, scan, parseReceipt, utf8, decodeStrict, LABEL:
  "re-derives sworn span verdicts offline; a forger controlling the whole file passes both " +
  "browser layers; the package at the named commit is the check" };

if (typeof module !== "undefined" && module.exports) module.exports = _api;
if (typeof globalThis !== "undefined") globalThis.swornVerifyApi = _api;
