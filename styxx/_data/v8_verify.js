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
// Spec section 8.1: the log a cert binds itself to (L7)
//
// A cert MAY carry `body.log_hint = {log_id, locations?}`.  `log_id` is section 8.1's
// identity for a log: "sha256:" + hex(sha256(raw log public key)).  That is the cert-id
// grammar spelled over bytes that are not a cert, which is why the path is exempted by
// name from the rule that every embedded cert id appears in `refs` and resolves.
//
// The field sits inside `body`, so it is inside the signed digest: it cannot be added,
// edited or removed without a fresh signature under a key the log admits.  An advisory
// field outside the signature would be walked through by exactly the attack this exists
// to stop -- entries replayed verbatim into a log built on a different key, appending
// byte for byte and giving the same root under a different `log_id`, so that "this cert
// is in the log" named no log.
//
// The predicate: a log may seat a cert iff the cert names no log, or names THAT log.
//   - no `log_hint` (absent, or null): unbound.  Any log may seat it, and it is as
//     replayable as it was before the field existed.  This is the state of every cert
//     signed before the field, so refusing it would refuse the whole existing corpus.
//   - `log_hint` present but not an object, or without a `log_id` in section 8.1's form:
//     refused rather than ignored.  A binding that cannot be compared with anything is
//     not a weaker binding, it is a claim shaped like one.
//   - `log_hint.log_id` naming another log: refused.
//
// The same rule holds at the gate that appends an entry and for a reader checking a
// clone where that gate never ran; there is one predicate, read twice.
//
// `logBindingReason` never throws on the CERT -- hostile input is refused, not crashed
// on -- and always throws on a malformed `logId`, which is the caller's own pinned key
// rather than anything off the wire.

const LOG_ID_RE = /^sha256:[0-9a-f]{64}$/;

function isPlainObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function typeName(value) {
  if (value === null) return 'null';
  return Array.isArray(value) ? 'array' : typeof value;
}

function bodyOf(cert) {
  const b = isPlainObject(cert) ? cert.body : null;
  return isPlainObject(b) ? b : {};
}

// "sha256:" + hex(sha256(raw public key)), from the 32 raw bytes or from the
// "ed25519:<base64url>" text a log stores in keys/log.pub.
function logIdFromPublic(publicKey) {
  const raw = typeof publicKey === 'string'
    ? decodePublic(publicKey.trim())
    : asBytes(publicKey, 'logIdFromPublic');
  if (raw.length !== PUBLIC_LEN) {
    throw new TypeError('logIdFromPublic: expected ' + PUBLIC_LEN + ' bytes');
  }
  return 'sha256:' + sha256Hex(raw);
}

// Does this cert name a log at all?  A count of bound and unbound entries is what a
// reader of a corpus is owed: an unbound entry is not a fault, it is the state in which
// "this cert is in the log" is a statement about bytes rather than about this log.
function isLogBound(cert) {
  return isPlainObject(bodyOf(cert).log_hint);
}

function logBindingReason(cert, logId) {
  if (typeof logId !== 'string' || !LOG_ID_RE.test(logId)) {
    throw new TypeError('logBindingReason: logId must be "sha256:<64 lowercase hex>"');
  }
  const hint = bodyOf(cert).log_hint;
  if (hint === undefined || hint === null) return null;
  if (!isPlainObject(hint)) {
    return 'log_hint: body.log_hint is ' + typeName(hint) + ', not an object; the binding a ' +
      'cert makes to its log is {log_id, locations?} or it is nothing (section 8.1)';
  }
  const named = hint.log_id;
  if (typeof named !== 'string' || !LOG_ID_RE.test(named)) {
    return 'log_hint: body.log_hint.log_id is ' + JSON.stringify(named === undefined ? null : named) +
      ', which is not sha256:<64 hex>; a cert that names its log names it in the form ' +
      'section 8.1 gives log_id, or the binding cannot be compared with anything';
  }
  if (named !== logId) {
    return 'log_hint: this cert names log ' + named + ' and this log is ' + logId +
      '; a cert bound to a log is not seated in another one (L7, section 8.1)';
  }
  return null;
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

// ---------------------------------------------------------------------------
// The cross-certificate floor predicate
//
// A noise floor cert says "these declared factors move this subject's channels by at
// most this much".  Read alone, a floor of 0.0 over five run certs that all carry the
// same body is byte-identical to a floor of 0.0 over five runs that genuinely agreed:
// the roster of unreachable defects (papers/v8/THE_BOUNDARY_2026_09_09.md, member 1)
// is built on that observation and it is true -- of ONE certificate.
//
// It stops being true the moment the same subject has been measured before, because a
// prior floor in the log already records which declared factor levels SEPARATE this
// subject.  A candidate that reports a zero where the log holds a positive is not
// merely a quieter measurement; it says a factor does nothing, against a logged
// measurement that it does.
//
// The result is three-valued on purpose.  `unconstrained` is not a pass.  A checker
// that answers when it holds no evidence is the failure this project already has a
// receipt for -- the path-claim accusation class ran at 0.23 precision on external
// pull requests and was disabled -- so "there is no comparable prior" has to be
// sayable, and has to be distinguishable from "compared and agreed".
//
// Asymmetry is deliberate.  A candidate positive where the prior held zero is NOT a
// contradiction: hardware drifts, and a rule that demanded equality would refuse
// honest re-measurement.  A candidate zero where the prior held positive is the one
// direction that cannot be explained by noise, because zero distance means the two
// runs produced identical channel values.
// ---------------------------------------------------------------------------

const FLOOR_AGREES = 'agrees';
const FLOOR_CONTRADICTS = 'contradicts';
const FLOOR_UNCONSTRAINED = 'unconstrained';

// Two tiers of evidence, compared independently and reported separately:
//   'factor'      one covered factor, one unordered pair of its levels (batch 1 vs 8).
//                 Generalizes across floors that used different run schedules, which
//                 matters because the published noise plan fixes no schedule.
//   'assignment'  the unordered pair of FULL covered-factor assignments.  Exact and
//                 free of attribution, but only matches a prior that ran the same
//                 assignments.
const FLOOR_TIERS = ['factor', 'assignment'];

function noiseFloorOf(cert) {
  const f = bodyOf(cert).noise_floor;
  return isPlainObject(f) ? f : null;
}

function nuisanceOf(cert) {
  const n = bodyOf(cert).nuisance;
  return isPlainObject(n) ? n : {};
}

// A stable text key for an arbitrary JSON value: its canonical bytes.  Unrepresentable
// values get a key no representable value can collide with, so they compare unequal to
// everything including each other's neighbours rather than throwing here.
function valueKey(value) {
  try {
    return canonicalBytes(value === undefined ? null : value).toString('utf8');
  } catch (e) {
    return '\u0000unrepresentable';
  }
}

function unorderedPairKey(a, b) {
  const ka = valueKey(a);
  const kb = valueKey(b);
  return ka <= kb ? ka + '\u0001' + kb : kb + '\u0001' + ka;
}

// The runs a floor cert names, resolved BY CERT ID out of a supplied corpus.
//
// Resolving by id rather than by scanning a log for `body.run_index` is a deliberate
// choice: run_index is scoped to one floor, so a log holding two subjects' runs has
// several certs claiming index 0, and an index-keyed scan silently mixes them.  The
// floor names its runs; that naming is what is followed.
//
// The published shape (entry 6 of papers/v8/first_verdict_2026_09_09/log) has the floor
// cert carrying run_index 0 itself and naming the other four by id, so a floor that is
// one of its own runs is added to the set.
function resolveFloorRuns(floorCert, certs) {
  const floor = noiseFloorOf(floorCert);
  if (floor === null) {
    throw new TypeError('resolveFloorRuns: cert carries no body.noise_floor');
  }
  if (!Array.isArray(floor.runs)) {
    throw new TypeError('resolveFloorRuns: body.noise_floor.runs is not an array');
  }
  const index = new Map();
  for (const c of Array.isArray(certs) ? certs : []) {
    if (isPlainObject(c) && typeof c.id === 'string' && !index.has(c.id)) index.set(c.id, c);
  }
  const runs = [];
  const missing = [];
  for (const id of floor.runs) {
    const c = typeof id === 'string' ? index.get(id) : undefined;
    if (c === undefined) missing.push(id);
    else runs.push(c);
  }
  const selfNamed = typeof floorCert.id === 'string' && floor.runs.includes(floorCert.id);
  if (!selfNamed && Number.isInteger(bodyOf(floorCert).run_index)) runs.push(floorCert);
  return { runs, missing, named: floor.runs.length };
}

// Turn a floor cert plus its run certs into comparable cells.
//
// The distances a floor stores are an ordered list over the pairs of its runs.  A
// distance is only evidence about a factor once you know which factor levels it
// separated, so the list is re-attributed to run pairs before anything is compared.
// The convention taken here -- pairs in lexicographic order over runs sorted by
// run_index -- is corroborated on the published floor: under it, and only under it,
// the three zero distances land exactly on the three pairs whose batch_size is equal.
function floorObservations(floorCert, runCerts) {
  const defects = [];
  const cells = new Map();
  const floor = noiseFloorOf(floorCert);
  if (floor === null) return { cells, defects: ['the cert carries no body.noise_floor'], runs: 0, pairs: 0, covers: [] };

  const ordered = (Array.isArray(runCerts) ? runCerts.slice() : [])
    .filter(isPlainObject)
    .sort((a, b) => bodyOf(a).run_index - bodyOf(b).run_index);
  const indices = ordered.map((c) => bodyOf(c).run_index);
  for (const i of indices) {
    if (!Number.isInteger(i)) defects.push('a run cert carries a non-integer body.run_index');
  }
  if (new Set(indices).size !== indices.length) {
    defects.push('two run certs share a body.run_index: ' + JSON.stringify(indices));
  }

  const pairs = [];
  for (let i = 0; i < ordered.length; i += 1) {
    for (let j = i + 1; j < ordered.length; j += 1) pairs.push([i, j]);
  }

  const covers = (Array.isArray(floor.covers) ? floor.covers : []).filter((s) => typeof s === 'string');
  const perChannel = isPlainObject(floor.per_channel) ? floor.per_channel : {};
  if (Object.keys(perChannel).length === 0) defects.push('body.noise_floor.per_channel is empty or absent');

  const touch = (channel, tier, label) => {
    const key = channel + '\u0002' + tier + '\u0002' + label;
    let cell = cells.get(key);
    if (cell === undefined) {
      cell = { channel, tier, label, values: new Set(), cleanValues: new Set() };
      cells.set(key, cell);
    }
    return cell;
  };

  for (const channel of Object.keys(perChannel).sort()) {
    const d = perChannel[channel];
    if (!isPlainObject(d) || !Array.isArray(d.distances)) {
      defects.push('channel ' + channel + ': body.noise_floor.per_channel.' + channel +
        '.distances is not an array');
      continue;
    }
    // The floor's own arithmetic, checked against the runs it names.  The Python
    // demonstration skipped a channel whose distance count did not match; a skip is
    // indistinguishable in the result from "this channel was never measured", and a
    // forger who truncates the list would buy silence with it.  Here it is a defect.
    if (d.distances.length !== pairs.length) {
      defects.push('channel ' + channel + ': ' + d.distances.length + ' distances for ' +
        ordered.length + ' runs, which make ' + pairs.length + ' pairs');
      continue;
    }
    if (d.runs !== undefined && d.runs !== ordered.length) {
      defects.push('channel ' + channel + ': declares runs=' + JSON.stringify(d.runs) +
        ' and names ' + ordered.length + ' run certs');
    }
    if (d.pairs !== undefined && d.pairs !== pairs.length) {
      defects.push('channel ' + channel + ': declares pairs=' + JSON.stringify(d.pairs) +
        ' and its runs make ' + pairs.length);
    }
    for (let k = 0; k < pairs.length; k += 1) {
      const v = d.distances[k];
      if (typeof v !== 'number' || !Number.isFinite(v)) {
        defects.push('channel ' + channel + ': distance ' + k + ' is ' + JSON.stringify(v) +
          ', not a finite number');
        continue;
      }
      if (v < 0) {
        defects.push('channel ' + channel + ': distance ' + k + ' is negative (' + v + ')');
      }
      const na = nuisanceOf(ordered[pairs[k][0]]);
      const nb = nuisanceOf(ordered[pairs[k][1]]);

      for (const f of covers) {
        const cell = touch(channel, 'factor', f + '\u0003' + unorderedPairKey(na[f], nb[f]));
        cell.values.add(v);
        // Clean for f iff every OTHER covered factor is held equal across the pair.
        // A pair that moves two factors at once is still evidence that the two runs
        // differ; it is not evidence about which factor did it.
        const confounded = covers.some((g) => g !== f && valueKey(na[g]) !== valueKey(nb[g]));
        if (!confounded) cell.cleanValues.add(v);
      }

      const project = (n) => {
        const o = {};
        for (const f of covers) o[f] = n[f] === undefined ? null : n[f];
        return o;
      };
      const cell = touch(channel, 'assignment', unorderedPairKey(project(na), project(nb)));
      cell.values.add(v);
      cell.cleanValues.add(v);
    }
  }
  return { cells, defects, runs: ordered.length, pairs: pairs.length, covers };
}

function describeCell(cell) {
  if (cell.tier === 'factor') {
    const [name, levels] = cell.label.split('\u0003');
    return 'channel ' + cell.channel + ', factor ' + name + ' at levels ' +
      levels.split('\u0001').join(' vs ');
  }
  return 'channel ' + cell.channel + ', assignment pair ' +
    cell.label.split('\u0001').join(' vs ');
}

const sortedValues = (set) => Array.from(set).sort((a, b) => a - b);

// floorAgreement(candidateCert, candidateRuns, priorCert, priorRuns)
//
//   -> { verdict, reasons, compared, contradictions, defects, comparability }
//
// `verdict` is one of 'agrees' | 'contradicts' | 'unconstrained'.
//
// Throwing policy follows this file's convention.  A candidate that carries no
// body.noise_floor is a caller mistake and throws; a candidate that carries one whose
// arithmetic does not match the runs it names is a claim refuted by logged bytes and
// is a contradiction, prior or no prior.  Anything about the PRIOR that makes it
// unusable -- absent, not a floor, not comparable under section 2.3, internally
// inconsistent -- yields `unconstrained`, never an accusation: a broken prior is not
// evidence, and this predicate never accuses on the strength of one.
function floorAgreement(candidateCert, candidateRuns, priorCert, priorRuns) {
  if (noiseFloorOf(candidateCert) === null) {
    throw new TypeError('floorAgreement: the candidate carries no body.noise_floor');
  }
  const cand = floorObservations(candidateCert, candidateRuns);
  const compared = { factor: 0, assignment: 0, total: 0 };
  const out = (verdict, reasons) => ({
    verdict,
    reasons,
    compared,
    contradictions: [],
    defects: cand.defects,
    comparability: []
  });

  if (cand.defects.length > 0) {
    return Object.assign(out(FLOOR_CONTRADICTS, cand.defects.map(
      (d) => 'the candidate floor contradicts the run certs it names: ' + d)), {});
  }
  if (noiseFloorOf(priorCert) === null) {
    return out(FLOOR_UNCONSTRAINED, ['no prior floor cert on this subject and battery']);
  }
  const differing = challengeValidity(priorCert, candidateCert);
  if (differing.length > 0) {
    return Object.assign(out(FLOOR_UNCONSTRAINED, [
      'the prior floor is not comparable under section 2.3; it differs on ' +
        differing.join(', ')
    ]), { comparability: differing });
  }
  const prior = floorObservations(priorCert, priorRuns);
  if (prior.defects.length > 0) {
    return out(FLOOR_UNCONSTRAINED, [
      'the prior floor is inconsistent with the runs it names, so it is not evidence: ' +
        prior.defects[0]
    ]);
  }

  const contradictions = [];
  for (const [key, cell] of cand.cells) {
    const before = prior.cells.get(key);
    if (before === undefined) continue;
    compared[cell.tier] += 1;
    compared.total += 1;
    const here = sortedValues(cell.values);
    const there = sortedValues(before.values);
    if (here.every((v) => v === 0) && there.some((v) => v > 0)) {
      contradictions.push({
        tier: cell.tier,
        channel: cell.channel,
        clean: cell.cleanValues.size > 0 && before.cleanValues.size > 0,
        why: describeCell(cell) + ': the prior floor measured ' + JSON.stringify(there) +
          ' and this one reports ' + JSON.stringify(here) +
          (cell.cleanValues.size > 0 && before.cleanValues.size > 0
            ? '' : ' (attribution confounded: the contributing pairs also moved another covered factor)')
      });
    }
  }

  if (compared.total === 0) {
    return out(FLOOR_UNCONSTRAINED, [
      'the prior floor exercised no factor level pair this one also exercised'
    ]);
  }
  if (contradictions.length > 0) {
    return Object.assign(out(FLOOR_CONTRADICTS, contradictions.map((c) => c.why)),
      { contradictions });
  }
  return out(FLOOR_AGREES, [
    compared.total + ' cells compared (' + compared.factor + ' factor-level pairs, ' +
      compared.assignment + ' assignment pairs), none contradicted'
  ]);
}

module.exports = {
  FLOOR_AGREES,
  FLOOR_CONTRADICTS,
  FLOOR_UNCONSTRAINED,
  FLOOR_TIERS,
  noiseFloorOf,
  resolveFloorRuns,
  floorObservations,
  floorAgreement,
  challengeValidity,
  isSynthetic,
  logIdFromPublic,
  isLogBound,
  logBindingReason,
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
