'use strict';

// An independent implementation of three primitives, written from
// RFC 8785 (JCS), RFC 6962 / RFC 9162 (Merkle trees) and RFC 8032 (Ed25519).
// Node built-ins only.
//
// Conventions:
//   - hashes are 32-byte Buffers, hex is lowercase
//   - encoded keys and signatures are "ed25519:<base64url, no padding>"
//   - verify* functions return false on hostile input and never throw
//   - generators throw on invalid arguments

const crypto = require('crypto');

const HASH_LEN = 32;
const PUBLIC_LEN = 32;
const SIGNATURE_LEN = 64;
const ED25519_PREFIX = 'ed25519:';
const SPKI_ED25519_HEADER = Buffer.from('302a300506032b6570032100', 'hex');
const B64URL_CHARS = /^[A-Za-z0-9_-]+$/;
const LEAF_TAG = Buffer.from([0x00]);
const NODE_TAG = Buffer.from([0x01]);

// ---------------------------------------------------------------------------
// byte helpers
// ---------------------------------------------------------------------------

function asBytes(value, where) {
  if (Buffer.isBuffer(value)) return value;
  if (value instanceof Uint8Array) {
    return Buffer.from(value.buffer, value.byteOffset, value.byteLength);
  }
  if (typeof value === 'string') return Buffer.from(value, 'utf8');
  throw new TypeError(where + ': expected bytes');
}

function isHash(value) {
  return (Buffer.isBuffer(value) || value instanceof Uint8Array) &&
    value.length === HASH_LEN;
}

function requireHash(value, where) {
  if (!isHash(value)) {
    throw new TypeError(where + ': expected a ' + HASH_LEN + '-byte hash');
  }
  return asBytes(value, where);
}

function sha256(...parts) {
  const h = crypto.createHash('sha256');
  for (const part of parts) h.update(part);
  return h.digest();
}

function sha256Hex(buf) {
  return crypto.createHash('sha256').update(asBytes(buf, 'sha256Hex')).digest('hex');
}

// ---------------------------------------------------------------------------
// RFC 8785, JSON Canonicalization Scheme
// ---------------------------------------------------------------------------

// RFC 8785 section 3.2.2.3 serializes numbers with the ECMAScript
// Number-to-String algorithm, which is what String(n) and JSON.stringify(n)
// both produce (1e21 -> "1e+21", 1e-7 -> "1e-7", -0 -> "0").
function jcsNumber(n) {
  if (!Number.isFinite(n)) {
    throw new TypeError('canonicalBytes: NaN and Infinity are not JSON numbers');
  }
  if (Object.is(n, -0)) return '0';
  // An integral value beyond the safe range names a double that is not the
  // integer it was written as, so refuse it.  The test is on the printed form:
  // a value that prints in exponential notation (1e+30, 1.7976931348623157e+308)
  // is a magnitude, not an integer literal, and RFC 8785's own Appendix B
  // requires 1E30 to canonicalize to 1e+30 -- so a blanket refusal above 2^53
  // would contradict the normative example.  See NOTE in the task report.
  // Every finite JavaScript number IS an IEEE-754 double, so every one of them is
  // exactly representable and must serialize (RFC 8785 section 3.2.2.3 defers to the
  // ECMAScript Number-to-String algorithm over doubles).  An earlier rule here refused
  // an integral value that was not a *safe* integer; that conflated two questions.
  // "Safe integer" asks whether integer ARITHMETIC is lossy at that magnitude -- whether
  // some other integer shares the same double.  A serializer only cares whether THIS
  // double exists, and it does.  The rule refused 123456789012345680000, which the
  // Python side accepts and prints unchanged, and a disagreement between two
  // canonicalizers is the one defect a content-addressed format cannot carry.
  // Arbitrary-precision integers are a different matter and are refused by type: a
  // Python `int` beyond 2^53 is rejected there, and BigInt is rejected here (see
  // canonicalBytes), because those types can hold values no double can represent.
  return String(n);
}

// RFC 8785 section 3.2.3 orders property names by their UTF-16 code units
// (not by Unicode code point).  Plain "<" on JS strings is exactly that order.
function compareKeys(a, b) {
  if (a === b) return 0;
  return a < b ? -1 : 1;
}

function serialize(value, out, seen) {
  if (value === null) {
    out.push('null');
    return;
  }
  const t = typeof value;
  if (t === 'boolean') {
    out.push(value ? 'true' : 'false');
    return;
  }
  if (t === 'number') {
    out.push(jcsNumber(value));
    return;
  }
  if (t === 'string') {
    // RFC 8785 section 3.2.2.2 escaping is what JSON.stringify emits.
    out.push(JSON.stringify(value));
    return;
  }
  if (t === 'bigint') throw new TypeError('canonicalBytes: bigint is not a JSON value');
  if (t === 'undefined') throw new TypeError('canonicalBytes: undefined is not a JSON value');
  if (t === 'function') throw new TypeError('canonicalBytes: function is not a JSON value');
  if (t === 'symbol') throw new TypeError('canonicalBytes: symbol is not a JSON value');
  if (t !== 'object') throw new TypeError('canonicalBytes: unsupported type ' + t);

  if (seen.has(value)) throw new TypeError('canonicalBytes: cyclic structure');
  seen.add(value);

  if (Array.isArray(value)) {
    out.push('[');
    for (let i = 0; i < value.length; i++) {
      if (i > 0) out.push(',');
      serialize(value[i], out, seen);
    }
    out.push(']');
  } else {
    const proto = Object.getPrototypeOf(value);
    if (proto !== Object.prototype && proto !== null) {
      throw new TypeError(
        'canonicalBytes: only plain objects are JSON values, got ' +
        (value.constructor && value.constructor.name ? value.constructor.name : 'exotic object'));
    }
    const keys = Object.keys(value).sort(compareKeys);
    out.push('{');
    for (let i = 0; i < keys.length; i++) {
      if (i > 0) out.push(',');
      out.push(JSON.stringify(keys[i]));
      out.push(':');
      serialize(value[keys[i]], out, seen);
    }
    out.push('}');
  }

  seen.delete(value);
}

function canonicalBytes(value) {
  const out = [];
  serialize(value, out, new Set());
  return Buffer.from(out.join(''), 'utf8');
}

function digest(value) {
  return sha256Hex(canonicalBytes(value));
}

// ---------------------------------------------------------------------------
// RFC 6962 section 2.1 Merkle tree over already-hashed leaves
// ---------------------------------------------------------------------------

const EMPTY_ROOT = sha256(Buffer.alloc(0));

function leafHash(entry) {
  return sha256(LEAF_TAG, asBytes(entry, 'leafHash'));
}

function nodeHash(left, right) {
  return sha256(NODE_TAG, requireHash(left, 'nodeHash'), requireHash(right, 'nodeHash'));
}

// largest power of two strictly less than n (n >= 2)
function splitPoint(n) {
  let k = 1;
  while (k * 2 < n) k *= 2;
  return k;
}

function checkLeaves(leafHashes, where) {
  if (!Array.isArray(leafHashes)) {
    throw new TypeError(where + ': leafHashes must be an array');
  }
  return leafHashes.map((h, i) => {
    if (!isHash(h)) {
      throw new TypeError(where + ': leaf hash ' + i + ' is not ' + HASH_LEN + ' bytes');
    }
    return asBytes(h, where);
  });
}

// MTH(D[lo:hi]) over leaves that are already hashed
function mth(d, lo, hi) {
  const n = hi - lo;
  if (n === 0) return EMPTY_ROOT;
  if (n === 1) return d[lo];
  const k = splitPoint(n);
  return nodeHash(mth(d, lo, lo + k), mth(d, lo + k, hi));
}

function root(leafHashes) {
  const d = checkLeaves(leafHashes, 'root');
  return mth(d, 0, d.length);
}

// RFC 6962 PATH(m, D[lo:hi])
function path(d, lo, hi, m, out) {
  const n = hi - lo;
  if (n === 1) return;
  const k = splitPoint(n);
  if (m < k) {
    path(d, lo, lo + k, m, out);
    out.push(mth(d, lo + k, hi));
  } else {
    path(d, lo + k, hi, m - k, out);
    out.push(mth(d, lo, lo + k));
  }
}

function inclusionProof(leafHashes, index, treeSize) {
  const d = checkLeaves(leafHashes, 'inclusionProof');
  const size = treeSize === undefined ? d.length : treeSize;
  if (!Number.isInteger(size) || size < 0 || size > d.length) {
    throw new RangeError('inclusionProof: treeSize out of range');
  }
  if (!Number.isInteger(index) || index < 0 || index >= size) {
    throw new RangeError('inclusionProof: index out of range');
  }
  const out = [];
  path(d, 0, size, index, out);
  return out;
}

// RFC 6962 SUBPROOF(m, D[lo:hi], b)
function subproof(d, lo, hi, m, b, out) {
  const n = hi - lo;
  if (m === n) {
    if (!b) out.push(mth(d, lo, hi));
    return;
  }
  const k = splitPoint(n);
  if (m <= k) {
    subproof(d, lo, lo + k, m, b, out);
    out.push(mth(d, lo + k, hi));
  } else {
    subproof(d, lo + k, hi, m - k, false, out);
    out.push(mth(d, lo, lo + k));
  }
}

function consistencyProof(leafHashes, first, second) {
  const d = checkLeaves(leafHashes, 'consistencyProof');
  const to = second === undefined ? d.length : second;
  if (!Number.isInteger(to) || to < 0 || to > d.length) {
    throw new RangeError('consistencyProof: second out of range');
  }
  if (!Number.isInteger(first) || first < 0 || first > to) {
    throw new RangeError('consistencyProof: first out of range');
  }
  if (first === 0 || first === to) return [];
  const out = [];
  subproof(d, 0, to, first, true, out);
  return out;
}

// ---------------------------------------------------------------------------
// RFC 9162 verification algorithms
// ---------------------------------------------------------------------------

function lsb(x) {
  return x % 2 === 1;
}

function shr(x) {
  return Math.floor(x / 2);
}

function proofBuffers(proof) {
  if (!Array.isArray(proof)) return null;
  const out = [];
  for (const p of proof) {
    if (!isHash(p)) return null;
    out.push(asBytes(p, 'proof'));
  }
  return out;
}

function isPowerOfTwo(n) {
  return n > 0 && (n & (n - 1)) === 0;
}

// RFC 9162 section 2.1.3.2
function verifyInclusion(leaf, index, treeSize, proof, expectedRoot) {
  try {
    if (!isHash(leaf) || !isHash(expectedRoot)) return false;
    if (!Number.isInteger(index) || !Number.isInteger(treeSize)) return false;
    if (index < 0 || treeSize < 0) return false;
    if (index >= treeSize) return false;
    const p = proofBuffers(proof);
    if (p === null) return false;

    let fn = index;
    let sn = treeSize - 1;
    let r = asBytes(leaf, 'verifyInclusion');

    for (const c of p) {
      if (sn === 0) return false;
      if (lsb(fn) || fn === sn) {
        r = nodeHash(c, r);
        if (!lsb(fn)) {
          while (!lsb(fn) && fn !== 0) {
            fn = shr(fn);
            sn = shr(sn);
          }
        }
      } else {
        r = nodeHash(r, c);
      }
      fn = shr(fn);
      sn = shr(sn);
    }

    return sn === 0 && r.equals(asBytes(expectedRoot, 'verifyInclusion'));
  } catch (err) {
    return false;
  }
}

// RFC 9162 section 2.1.4.2
function verifyConsistency(first, second, firstRoot, secondRoot, proof) {
  try {
    if (!Number.isInteger(first) || !Number.isInteger(second)) return false;
    if (first < 0 || second < 0 || first > second) return false;
    if (!isHash(firstRoot) || !isHash(secondRoot)) return false;
    const p = proofBuffers(proof);
    if (p === null) return false;

    const fr0 = asBytes(firstRoot, 'verifyConsistency');
    const sr0 = asBytes(secondRoot, 'verifyConsistency');

    if (first === second) return p.length === 0 && fr0.equals(sr0);
    if (first === 0) return p.length === 0 && fr0.equals(EMPTY_ROOT);
    if (p.length === 0) return false;

    const seq = isPowerOfTwo(first) ? [fr0].concat(p) : p.slice();

    let fn = first - 1;
    let sn = second - 1;
    while (lsb(fn)) {
      fn = shr(fn);
      sn = shr(sn);
    }

    let fr = seq[0];
    let sr = seq[0];

    for (let i = 1; i < seq.length; i++) {
      const c = seq[i];
      if (sn === 0) return false;
      if (lsb(fn) || fn === sn) {
        fr = nodeHash(c, fr);
        sr = nodeHash(c, sr);
        if (!lsb(fn)) {
          while (!lsb(fn) && fn !== 0) {
            fn = shr(fn);
            sn = shr(sn);
          }
        }
      } else {
        sr = nodeHash(sr, c);
      }
      fn = shr(fn);
      sn = shr(sn);
    }

    return sn === 0 && fr.equals(fr0) && sr.equals(sr0);
  } catch (err) {
    return false;
  }
}

// ---------------------------------------------------------------------------
// encoded keys and signatures
// ---------------------------------------------------------------------------

function decodeBody(body, expectedLen, what) {
  if (body.length === 0) throw new Error(what + ': empty base64url body');
  if (!B64URL_CHARS.test(body)) {
    throw new Error(what + ': body is not unpadded base64url');
  }
  if (body.length % 4 === 1) {
    throw new Error(what + ': impossible base64url length');
  }
  const buf = Buffer.from(body, 'base64url');
  if (buf.toString('base64url') !== body) {
    throw new Error(what + ': non-canonical base64url (trailing bits set)');
  }
  if (buf.length !== expectedLen) {
    throw new Error(what + ': decoded ' + buf.length + ' bytes, want ' + expectedLen);
  }
  return buf;
}

function decodePrefixed(str, expectedLen, what) {
  if (typeof str !== 'string') throw new TypeError(what + ': expected a string');
  if (!str.startsWith(ED25519_PREFIX)) {
    throw new Error(what + ': missing "' + ED25519_PREFIX + '" prefix');
  }
  return decodeBody(str.slice(ED25519_PREFIX.length), expectedLen, what);
}

function decodePublic(str) {
  return decodePrefixed(str, PUBLIC_LEN, 'decodePublic');
}

function decodeSignature(str) {
  return decodePrefixed(str, SIGNATURE_LEN, 'decodeSignature');
}

function encodePublic(buf) {
  const b = asBytes(buf, 'encodePublic');
  if (b.length !== PUBLIC_LEN) {
    throw new TypeError('encodePublic: expected ' + PUBLIC_LEN + ' bytes');
  }
  return ED25519_PREFIX + b.toString('base64url');
}

function encodeSignature(buf) {
  const b = asBytes(buf, 'encodeSignature');
  if (b.length !== SIGNATURE_LEN) {
    throw new TypeError('encodeSignature: expected ' + SIGNATURE_LEN + ' bytes');
  }
  return ED25519_PREFIX + b.toString('base64url');
}

// RFC 8032 pure Ed25519 (PureEdDSA over edwards25519, empty context)
function verifySignature(public32, message, sig64) {
  try {
    if (!(Buffer.isBuffer(public32) || public32 instanceof Uint8Array)) return false;
    if (public32.length !== PUBLIC_LEN) return false;
    if (!(Buffer.isBuffer(sig64) || sig64 instanceof Uint8Array)) return false;
    if (sig64.length !== SIGNATURE_LEN) return false;
    const msg = asBytes(message, 'verifySignature');
    const spki = Buffer.concat([SPKI_ED25519_HEADER, asBytes(public32, 'verifySignature')]);
    return crypto.verify(
      null,
      msg,
      { key: spki, format: 'der', type: 'spki' },
      asBytes(sig64, 'verifySignature')
    );
  } catch (err) {
    return false;
  }
}

// ---------------------------------------------------------------------------
// domain separation
// ---------------------------------------------------------------------------

function tagged(tag, digest32) {
  if (typeof tag !== 'string') throw new TypeError('tagged: tag must be a string');
  if (tag.length === 0) throw new TypeError('tagged: tag must not be empty');
  for (let i = 0; i < tag.length; i++) {
    const c = tag.charCodeAt(i);
    if (c === 0) throw new TypeError('tagged: tag must not contain NUL');
    if (c > 0x7f) throw new TypeError('tagged: tag must be ASCII');
  }
  const d = requireHash(digest32, 'tagged');
  return Buffer.concat([Buffer.from(tag, 'ascii'), Buffer.from([0x00]), d]);
}

// ---------------------------------------------------------------------------
// Spec section 9 rule 1: is this fingerprint a challenge to that one?
//
// Written from the spec text, not from the Python: "A challenge is valid iff the
// challenger's fingerprint is comparable (section 2.3) AND has the same subject
// identity (every S_identity field equal); clients compute this from the two
// certs -- nothing in the body asserts it. No match, no challenge."
//
// challengeValidity(target, own) returns the sorted list of names that differ;
// [] means `own` is a challenge to `target`.  Entry shapes, from section 2.3:
//   "subject.<field>"        an S_identity field that differs
//   "cross-subject:<field>"  precision or revision -- an S_identity field too, so
//                            section 9 counts it; a cross-subject pair is a diff,
//                            not a challenge
//   "recipe.<field>"         a recipe_core field that differs
//   "body.synthetic"         one side's numbers came from a runner holding no model
//                            and the other's did not -- see styxx/v8/cert.py under
//                            "Decisions" (C-MOCK). Never softened to cross-subject:
//                            a fabrication is not a precision of a model.
// Never throws: hostile input differs on everything rather than crashing.

const IDENTITY_FIELDS = {
  weights: [
    'hf_repo', 'revision', 'weights_sha256', 'config_sha256', 'tokenizer_sha256',
    'generation_config_sha256', 'precision'
  ],
  alias: ['provider', 'alias', 'region']
};
const RECIPE_CORE_FIELDS = [
  'battery', 'decoding', 'chat_template_sha256', 'system_prompt_sha256'
];
const CROSS_SUBJECT_FIELDS = ['precision', 'revision'];

function sameValue(a, b) {
  // Structural equality over JSON values, via the canonicalization above: two
  // values are the same iff their canonical bytes are the same.
  if (a === undefined && b === undefined) return true;
  if (a === undefined || b === undefined) return false;
  let ba;
  let bb;
  try {
    ba = canonicalBytes(a === null ? null : a);
    bb = canonicalBytes(b === null ? null : b);
  } catch (e) {
    return false;
  }
  return ba.equals(bb);
}

function subjectOf(cert) {
  const s = cert && typeof cert === 'object' ? cert.subject : null;
  return s && typeof s === 'object' && !Array.isArray(s) ? s : {};
}

function recipeOf(cert) {
  const r = cert && typeof cert === 'object' ? cert.recipe : null;
  return r && typeof r === 'object' && !Array.isArray(r) ? r : {};
}

// Present and not `false`, not `=== true`: the schema pins the member to `true` where it is
// declared, so any other value is hostile input, and `synthetic: null` must not read as a
// measurement. `styxx.v8.cert.is_synthetic` implements the same rule.
function isSynthetic(cert) {
  const b = cert && typeof cert === 'object' ? cert.body : null;
  if (!b || typeof b !== 'object' || Array.isArray(b)) return false;
  if (!Object.prototype.hasOwnProperty.call(b, 'synthetic')) return false;
  return b.synthetic !== false;
}

function challengeValidity(target, own) {
  const out = [];
  if (isSynthetic(target) !== isSynthetic(own)) out.push('body.synthetic');
  const sa = subjectOf(target);
  const sb = subjectOf(own);
  const ka = typeof sa.kind === 'string' ? sa.kind : null;
  const kb = typeof sb.kind === 'string' ? sb.kind : null;
  if (ka !== kb) out.push('subject.kind');
  const names = [];
  for (const k of [ka, kb]) {
    for (const name of IDENTITY_FIELDS[k] || []) {
      if (!names.includes(name)) names.push(name);
    }
  }
  for (const name of names) {
    if (sameValue(sa[name], sb[name])) continue;
    out.push(
      CROSS_SUBJECT_FIELDS.includes(name) ? 'cross-subject:' + name : 'subject.' + name
    );
  }
  const ra = recipeOf(target);
  const rb = recipeOf(own);
  for (const name of RECIPE_CORE_FIELDS) {
    if (!sameValue(ra[name], rb[name])) out.push('recipe.' + name);
  }
  return out.sort();
}

module.exports = {
  challengeValidity,
  isSynthetic,
  canonicalBytes,
  sha256Hex,
  digest,
  leafHash,
  nodeHash,
  EMPTY_ROOT,
  root,
  inclusionProof,
  verifyInclusion,
  consistencyProof,
  verifyConsistency,
  decodePublic,
  decodeSignature,
  encodePublic,
  encodeSignature,
  verifySignature,
  tagged
};
