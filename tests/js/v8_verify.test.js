'use strict';

// Tests for the independent JS implementation.  Every vector here is either
// computed or taken from an RFC; nothing is copied from the Python side.

const test = require('node:test');
const assert = require('node:assert');
const crypto = require('crypto');

const V = require('../../styxx/_data/v8_verify.js');

const fromHex = (s) => Buffer.from(s, 'hex');
const hex = (b) => Buffer.from(b).toString('hex');

// ---------------------------------------------------------------------------
// 1. RFC 8785 Appendix B
// ---------------------------------------------------------------------------

const RFC_NUMBERS = [
  333333333.33333329,
  1E30,
  4.50,
  2e-3,
  0.000000000000000000000000001
];

// The RFC's JSON input holds the string
//   "\u20ac$\u000F\nA'B\"\\\"\/"
// whose decoded characters are:
//   EURO SIGN, '$', U+000F, LF, 'A', apostrophe, 'B', '"', '\', '"', '/'
const RFC_STRING = '\u20ac$\u000f\nA\'B"\\"/';

const RFC_EXPECTED =
  '{"literals":[null,true,false],' +
  '"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27],' +
  '"string":"\u20ac$\\u000f\\nA\'B\\"\\\\\\"/"}';

test('RFC 8785 Appendix B canonicalizes to the published text', () => {
  const input = {
    numbers: RFC_NUMBERS,
    string: RFC_STRING,
    literals: [null, true, false]
  };
  const bytes = V.canonicalBytes(input);
  const text = bytes.toString('utf8');
  console.log('[1] canonical text   : ' + text);
  console.log('[1] canonical sha256 : ' + V.sha256Hex(bytes));
  console.log('[1] digest()         : ' + V.digest(input));
  assert.strictEqual(text, RFC_EXPECTED);
  assert.strictEqual(V.digest(input), V.sha256Hex(bytes));
});

test('the task spec transcribed the RFC string with one extra backslash', () => {
  // Verbatim JS literal from the task spec:
  const promptString = "\u20ac$\u000F\u000aA'\u0042\u0022\u005c\\\"\u002f";
  // In JS, \u005c is a backslash AND \\ is a second one, so this value carries
  // two backslashes where the RFC's own input carries one.  Recorded, not fixed.
  assert.strictEqual((promptString.match(/\\/g) || []).length, 2);
  assert.strictEqual((RFC_STRING.match(/\\/g) || []).length, 1);
  const text = V.canonicalBytes({
    numbers: RFC_NUMBERS,
    string: promptString,
    literals: [null, true, false]
  }).toString('utf8');
  console.log('[1b] spec-literal text  : ' + text);
  console.log('[1b] spec-literal sha256: ' +
    V.sha256Hex(V.canonicalBytes({
      numbers: RFC_NUMBERS,
      string: promptString,
      literals: [null, true, false]
    })));
});

// ---------------------------------------------------------------------------
// 2. numbers
// ---------------------------------------------------------------------------

test('ES6 Number-to-String number forms', () => {
  const cases = [
    [1e21, '1e+21'],
    [1e-7, '1e-7'],
    [0.000001, '0.000001'],
    [-0, '0'],
    [1.0, '1'],
    [5e-324, '5e-324'],
    [0.1 + 0.2, '0.30000000000000004'],
    [1.7976931348623157e308, '1.7976931348623157e+308']
  ];
  for (const [value, expected] of cases) {
    const got = V.canonicalBytes(value).toString('utf8');
    console.log('[2] ' + expected + ' <- ' + got);
    assert.strictEqual(got, expected);
  }
  // Repinned 2026-09-08: this value is an exactly-representable double, and RFC 8785
  // section 3.2.2.3 requires it to serialize. The earlier RangeError disagreed with the
  // Python canonicalizer on the same input, which a content-addressed format cannot carry;
  // the disagreement and its resolution are in papers/v8/FINDING_number_domain_2026_09_08.md.
  assert.strictEqual(
    Buffer.from(V.canonicalBytes(123456789012345680000)).toString('utf8'),
    '123456789012345680000');
  console.log('[2] 123456789012345680000 -> 123456789012345680000 (agrees with Python)');
});

// ---------------------------------------------------------------------------
// 3. refusals
// ---------------------------------------------------------------------------

test('non-JSON values are refused', () => {
  assert.throws(() => V.canonicalBytes(NaN), TypeError);
  assert.throws(() => V.canonicalBytes(Infinity), TypeError);
  assert.throws(() => V.canonicalBytes(-Infinity), TypeError);
  // Repinned 2026-09-08. In JavaScript, Math.pow(2,53)+1 evaluates to exactly Math.pow(2,53):
  // the value 2^53+1 has no double and cannot be written here at all. Python's int CAN hold it
  // and refuses it, so this is an asymmetry between the languages, not a divergence between the
  // canonicalizers. What is pinned instead: 2^53 itself is representable and must serialize.
  assert.strictEqual(Math.pow(2, 53) + 1, Math.pow(2, 53));
  assert.strictEqual(
    Buffer.from(V.canonicalBytes(Math.pow(2, 53))).toString('utf8'), '9007199254740992');
  assert.throws(() => V.canonicalBytes(10n), TypeError);
  assert.throws(() => V.canonicalBytes(new Date()), TypeError);
  assert.throws(() => V.canonicalBytes(new Map()), TypeError);
  assert.throws(() => V.canonicalBytes(new Set()), TypeError);
  assert.throws(() => V.canonicalBytes(Buffer.alloc(1)), TypeError);
  assert.throws(() => V.canonicalBytes(undefined), TypeError);
  assert.throws(() => V.canonicalBytes(function () {}), TypeError);
  assert.throws(() => V.canonicalBytes(Symbol('x')), TypeError);
  class Thing { constructor() { this.a = 1; } }
  assert.throws(() => V.canonicalBytes(new Thing()), TypeError);
  // nested, not just at the top level
  assert.throws(() => V.canonicalBytes({ a: [1, undefined] }), TypeError);
  assert.throws(() => V.canonicalBytes({ a: new Date() }), TypeError);
  // a null-prototype plain object is accepted
  const bare = Object.create(null);
  bare.b = 2;
  bare.a = 1;
  assert.strictEqual(V.canonicalBytes(bare).toString('utf8'), '{"a":1,"b":2}');
});

// ---------------------------------------------------------------------------
// 4. key order
// ---------------------------------------------------------------------------

test('property names sort by UTF-16 code units', () => {
  // RFC 8785 section 3.2.3 requires sorting on UTF-16 code units, NOT on
  // Unicode code points.
  const o = {};
  o['\ud83d\ude00'] = 4;   // U+1F600, code units d83d de00
  o['\u00e9'] = 3;
  o['z'] = 2;
  o['a'] = 1;
  const text = V.canonicalBytes(o).toString('utf8');
  console.log('[4] ' + text);
  assert.strictEqual(text, '{"a":1,"z":2,"\u00e9":3,"\ud83d\ude00":4}');

  // Note: those four keys do NOT actually separate the two orders -- 0x0061,
  // 0x007a, 0x00e9, 0x1F600 rank the same way under both rules.  A key in
  // U+E000..U+FFFF is what separates them: under UTF-16 code units the
  // surrogate pair (0xd83d) sorts before U+FFFF, while under code points
  // U+FFFF (65535) sorts before U+1F600 (128512).
  const p = {};
  p['\uffff'] = 1;
  p['\ud83d\ude00'] = 2;
  const ptext = V.canonicalBytes(p).toString('utf8');
  console.log('[4] ' + JSON.stringify(ptext));
  assert.strictEqual(ptext, '{"\ud83d\ude00":2,"\uffff":1}');
});

// ---------------------------------------------------------------------------
// 5. Certificate Transparency vectors
// ---------------------------------------------------------------------------

const CT_ENTRIES = [
  '',
  '00',
  '10',
  '2021',
  '3031',
  '40414243',
  '5051525354555657',
  '606162636465666768696a6b6c6d6e6f'
].map(fromHex);

const CT_LEAVES = CT_ENTRIES.map((e) => V.leafHash(e));

const CT_ROOTS = [
  '6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d',
  'fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125',
  'aeb6bcfe274b70a14fb067a5e5578264db0fa9b51af5e0ba159158f329e06e77',
  'd37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7',
  '4e3bbb1f7b478dcfe71fb631631519a3bca12c9aefca1612bfce4c13a86264d4',
  '76e67dadbcdf1e10e1b74ddc608abd2f98dfb16fbce75277b5232a127f2087ef',
  'ddb89be403809e325750d3d263cd78929c2942b7942a34b77e122c9594a74c8c',
  '5dc9da79a70659a9ad559cb701ded9a2ab9d823aad2f4960cfe370eff4604328'
];

const CT_EMPTY_ROOT = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855';

const CT_INCLUSION = [
  {
    index: 0,
    size: 8,
    proof: [
      '96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7',
      '5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e',
      '6b47aaf29ee3c2af9af889bc1fb9254dabd31177f16232dd6aab035ca39bf6e4'
    ]
  },
  {
    index: 5,
    size: 8,
    proof: [
      'bc1a0643b12e4d2d7c77918f44e0f4f79a838b6cf9ec5b5c283e1f4d88599e6b',
      'ca854ea128ed050b41b35ffc1b87b8eb2bde461e9e3b5596ece6b9d5975a0ae0',
      'd37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7'
    ]
  }
];

const CT_CONSISTENCY = [
  {
    first: 2,
    second: 8,
    proof: [
      '5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e',
      '6b47aaf29ee3c2af9af889bc1fb9254dabd31177f16232dd6aab035ca39bf6e4'
    ]
  },
  {
    first: 6,
    second: 8,
    proof: [
      '0ebc5d3437fbe2db158b9f126a1d118e308181031d0a949f8dededebc558ef6a',
      'ca854ea128ed050b41b35ffc1b87b8eb2bde461e9e3b5596ece6b9d5975a0ae0',
      'd37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7'
    ]
  }
];

test('CT roots for sizes 0..8', () => {
  assert.strictEqual(hex(V.EMPTY_ROOT), CT_EMPTY_ROOT);
  assert.strictEqual(hex(V.root([])), CT_EMPTY_ROOT);
  for (let n = 1; n <= 8; n++) {
    const got = hex(V.root(CT_LEAVES.slice(0, n)));
    console.log('[5] root(' + n + ') = ' + got);
    assert.strictEqual(got, CT_ROOTS[n - 1], 'root at size ' + n);
  }
});

test('CT inclusion proofs', () => {
  for (const v of CT_INCLUSION) {
    const got = V.inclusionProof(CT_LEAVES, v.index, v.size).map(hex);
    console.log('[5] inclusion(' + v.index + ',' + v.size + ') = ' + JSON.stringify(got));
    assert.deepStrictEqual(got, v.proof,
      'inclusion proof for leaf ' + v.index + ' of size ' + v.size);
    assert.strictEqual(
      V.verifyInclusion(CT_LEAVES[v.index], v.index, v.size,
        v.proof.map(fromHex), fromHex(CT_ROOTS[v.size - 1])),
      true);
  }
});

test('CT consistency proofs', () => {
  for (const v of CT_CONSISTENCY) {
    const got = V.consistencyProof(CT_LEAVES, v.first, v.second).map(hex);
    console.log('[5] consistency(' + v.first + ',' + v.second + ') = ' + JSON.stringify(got));
    assert.deepStrictEqual(got, v.proof,
      'consistency proof (' + v.first + ',' + v.second + ')');
    assert.strictEqual(
      V.verifyConsistency(v.first, v.second,
        fromHex(CT_ROOTS[v.first - 1]), fromHex(CT_ROOTS[v.second - 1]),
        v.proof.map(fromHex)),
      true);
  }
});

// ---------------------------------------------------------------------------
// 6. properties over sizes 0..64
// ---------------------------------------------------------------------------

const N = 64;

function pseudoEntry(i) {
  return crypto.createHash('sha256').update('v8_verify/leaf/' + i, 'utf8').digest();
}

const BIG_LEAVES = [];
for (let i = 0; i < N; i++) BIG_LEAVES.push(V.leafHash(pseudoEntry(i)));

const BIG_ROOTS = [];
for (let n = 0; n <= N; n++) BIG_ROOTS.push(V.root(BIG_LEAVES.slice(0, n)));

test('every inclusion proof verifies for sizes 0..64', () => {
  let count = 0;
  for (let n = 0; n <= N; n++) {
    for (let i = 0; i < n; i++) {
      const proof = V.inclusionProof(BIG_LEAVES, i, n);
      assert.strictEqual(
        V.verifyInclusion(BIG_LEAVES[i], i, n, proof, BIG_ROOTS[n]), true,
        'inclusion ' + i + '/' + n);
      count++;
    }
  }
  console.log('[6] inclusion proofs verified: ' + count);
});

test('every consistency proof verifies for all pairs first <= second', () => {
  let count = 0;
  for (let second = 0; second <= N; second++) {
    for (let first = 0; first <= second; first++) {
      const proof = V.consistencyProof(BIG_LEAVES, first, second);
      assert.strictEqual(
        V.verifyConsistency(first, second, BIG_ROOTS[first], BIG_ROOTS[second], proof),
        true, 'consistency ' + first + '->' + second);
      count++;
    }
  }
  console.log('[6] consistency proofs verified: ' + count);
});

test('mangled inclusion proofs are rejected', () => {
  for (let n = 2; n <= N; n++) {
    for (let i = 0; i < n; i++) {
      const proof = V.inclusionProof(BIG_LEAVES, i, n);
      if (proof.length === 0) continue;

      const flipped = proof.map((b) => Buffer.from(b));
      flipped[proof.length - 1][7] ^= 0x01;
      assert.strictEqual(
        V.verifyInclusion(BIG_LEAVES[i], i, n, flipped, BIG_ROOTS[n]), false,
        'flipped byte ' + i + '/' + n);

      if (proof.length >= 2) {
        const swapped = proof.slice();
        const t = swapped[0];
        swapped[0] = swapped[1];
        swapped[1] = t;
        if (!swapped[0].equals(swapped[1])) {
          assert.strictEqual(
            V.verifyInclusion(BIG_LEAVES[i], i, n, swapped, BIG_ROOTS[n]), false,
            'swapped pair ' + i + '/' + n);
        }
      }

      assert.strictEqual(
        V.verifyInclusion(BIG_LEAVES[i], i, n, proof.slice(0, -1), BIG_ROOTS[n]), false,
        'truncated ' + i + '/' + n);

      assert.strictEqual(
        V.verifyInclusion(BIG_LEAVES[i], i, n,
          proof.concat([Buffer.alloc(32, 0xab)]), BIG_ROOTS[n]), false,
        'extended ' + i + '/' + n);
    }
  }
});

test('mangled consistency proofs are rejected', () => {
  for (let second = 2; second <= N; second++) {
    for (let first = 1; first < second; first++) {
      const proof = V.consistencyProof(BIG_LEAVES, first, second);
      if (proof.length === 0) continue;

      const flipped = proof.map((b) => Buffer.from(b));
      flipped[0][3] ^= 0x80;
      assert.strictEqual(
        V.verifyConsistency(first, second, BIG_ROOTS[first], BIG_ROOTS[second], flipped),
        false, 'flipped ' + first + '->' + second);

      if (proof.length >= 2) {
        const swapped = proof.slice();
        const t = swapped[0];
        swapped[0] = swapped[1];
        swapped[1] = t;
        if (!swapped[0].equals(swapped[1])) {
          assert.strictEqual(
            V.verifyConsistency(first, second, BIG_ROOTS[first], BIG_ROOTS[second], swapped),
            false, 'swapped ' + first + '->' + second);
        }
      }

      assert.strictEqual(
        V.verifyConsistency(first, second, BIG_ROOTS[first], BIG_ROOTS[second],
          proof.slice(0, -1)),
        false, 'truncated ' + first + '->' + second);

      assert.strictEqual(
        V.verifyConsistency(first, second, BIG_ROOTS[first], BIG_ROOTS[second],
          proof.concat([Buffer.alloc(32, 0xcd)])),
        false, 'extended ' + first + '->' + second);
    }
  }
});

test('short hashes: verifiers return false, generators throw', () => {
  const proof = V.inclusionProof(CT_LEAVES, 5, 8);
  const root8 = fromHex(CT_ROOTS[7]);
  assert.strictEqual(V.verifyInclusion(Buffer.alloc(31), 5, 8, proof, root8), false);
  assert.strictEqual(V.verifyInclusion(CT_LEAVES[5], 5, 8, proof, Buffer.alloc(31)), false);
  assert.strictEqual(
    V.verifyInclusion(CT_LEAVES[5], 5, 8,
      [Buffer.alloc(31)].concat(proof.slice(1)), root8), false);
  assert.strictEqual(V.verifyInclusion(CT_LEAVES[5], 5, 8, 'not-an-array', root8), false);
  assert.strictEqual(V.verifyInclusion(CT_LEAVES[5], 8, 8, proof, root8), false);
  assert.strictEqual(V.verifyInclusion(CT_LEAVES[5], -1, 8, proof, root8), false);

  const cproof = V.consistencyProof(CT_LEAVES, 6, 8);
  const root6 = fromHex(CT_ROOTS[5]);
  assert.strictEqual(V.verifyConsistency(6, 8, Buffer.alloc(31), root8, cproof), false);
  assert.strictEqual(V.verifyConsistency(6, 8, root6, Buffer.alloc(31), cproof), false);
  assert.strictEqual(
    V.verifyConsistency(6, 8, root6, root8, [Buffer.alloc(31)].concat(cproof.slice(1))), false);
  assert.strictEqual(V.verifyConsistency(8, 6, root8, root6, cproof), false);
  assert.strictEqual(V.verifyConsistency(6, 8, root6, root8, []), false);

  // the contract's two degenerate cases
  assert.strictEqual(V.verifyConsistency(4, 4, BIG_ROOTS[4], BIG_ROOTS[4], []), true);
  assert.strictEqual(V.verifyConsistency(4, 4, BIG_ROOTS[4], BIG_ROOTS[5], []), false);
  assert.strictEqual(
    V.verifyConsistency(4, 4, BIG_ROOTS[4], BIG_ROOTS[4], [Buffer.alloc(32)]), false);
  assert.strictEqual(V.verifyConsistency(0, 8, V.EMPTY_ROOT, root8, []), true);
  assert.strictEqual(V.verifyConsistency(0, 8, BIG_ROOTS[1], root8, []), false);
  assert.strictEqual(
    V.verifyConsistency(0, 8, V.EMPTY_ROOT, root8, [Buffer.alloc(32)]), false);

  assert.throws(() => V.root([Buffer.alloc(31)]), TypeError);
  assert.throws(() => V.root('not-an-array'), TypeError);
  assert.throws(() => V.inclusionProof([Buffer.alloc(31)], 0), TypeError);
  assert.throws(() => V.consistencyProof([Buffer.alloc(31)], 1, 1), TypeError);
  assert.throws(() => V.inclusionProof(CT_LEAVES, 8, 8), RangeError);
  assert.throws(() => V.inclusionProof(CT_LEAVES, -1, 8), RangeError);
  assert.throws(() => V.inclusionProof(CT_LEAVES, 0, 9), RangeError);
  assert.throws(() => V.consistencyProof(CT_LEAVES, 5, 3), RangeError);
  assert.throws(() => V.consistencyProof(CT_LEAVES, 0, 9), RangeError);
  assert.throws(() => V.nodeHash(Buffer.alloc(31), Buffer.alloc(32)), TypeError);
});

// ---------------------------------------------------------------------------
// 7. RFC 8032 section 7.1
// ---------------------------------------------------------------------------

const ED_VECTORS = [
  {
    name: 'TEST 1 (empty message)',
    pub: 'd75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a',
    msg: '',
    sig: 'e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb882' +
         '1590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b'
  },
  {
    name: 'TEST 2 (one-byte message)',
    pub: '3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c',
    msg: '72',
    sig: '92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1' +
         'e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00'
  }
];

test('RFC 8032 Ed25519 vectors verify', () => {
  for (const v of ED_VECTORS) {
    const pub = fromHex(v.pub);
    const msg = fromHex(v.msg);
    const sig = fromHex(v.sig);
    assert.strictEqual(V.verifySignature(pub, msg, sig), true, v.name);
    console.log('[7] ' + v.name + ' verified');

    const badSig = Buffer.from(sig);
    badSig[0] ^= 0x01;
    assert.strictEqual(V.verifySignature(pub, msg, badSig), false);

    const badMsg = Buffer.concat([msg, Buffer.from([0x00])]);
    assert.strictEqual(V.verifySignature(pub, badMsg, sig), false);

    const badPub = Buffer.from(pub);
    badPub[31] ^= 0x01;
    assert.strictEqual(V.verifySignature(badPub, msg, sig), false);

    assert.strictEqual(V.verifySignature(pub, msg, sig.subarray(0, 63)), false);
    assert.strictEqual(V.verifySignature(pub.subarray(0, 31), msg, sig), false);
    assert.strictEqual(V.verifySignature(null, msg, sig), false);
    assert.strictEqual(V.verifySignature(pub, msg, null), false);
    assert.strictEqual(V.verifySignature(pub, msg, Buffer.alloc(64)), false);
  }
});

test('generated key pair round-trips through encode/decode', () => {
  const { publicKey, privateKey } = crypto.generateKeyPairSync('ed25519');
  const raw = publicKey.export({ format: 'der', type: 'spki' }).subarray(-32);
  const message = Buffer.from('the ferry does not care who is aboard', 'utf8');
  const sig = crypto.sign(null, message, privateKey);

  assert.strictEqual(sig.length, 64);
  assert.strictEqual(V.verifySignature(raw, message, sig), true);

  const encPub = V.encodePublic(raw);
  const encSig = V.encodeSignature(sig);
  assert.ok(encPub.startsWith('ed25519:'));
  assert.ok(encSig.startsWith('ed25519:'));
  assert.ok(!encPub.includes('='));
  assert.ok(!encSig.includes('='));
  assert.strictEqual(encPub.length, 'ed25519:'.length + 43);
  assert.strictEqual(encSig.length, 'ed25519:'.length + 86);

  const decPub = V.decodePublic(encPub);
  const decSig = V.decodeSignature(encSig);
  assert.ok(decPub.equals(raw));
  assert.ok(decSig.equals(sig));
  assert.strictEqual(V.verifySignature(decPub, message, decSig), true);
  console.log('[7] round trip ok: ' + encPub);
});

test('encoded key and signature strings are strictly decoded', () => {
  const raw = fromHex(ED_VECTORS[0].pub);
  const enc = V.encodePublic(raw);
  const body = enc.slice('ed25519:'.length);

  assert.throws(() => V.decodePublic(body));                       // no prefix
  assert.throws(() => V.decodePublic('ed25519' + body));           // no colon
  assert.throws(() => V.decodePublic('x25519:' + body));           // wrong prefix
  assert.throws(() => V.decodePublic('ed25519:' + body + '='));    // padding
  assert.throws(() => V.decodePublic('ed25519:' + body.slice(0, -1))); // short
  assert.throws(() => V.decodePublic('ed25519:' + body + 'AA'));   // long
  assert.throws(() => V.decodePublic('ed25519:'));                 // empty body
  assert.throws(() => V.decodePublic(null));
  assert.throws(() => V.decodePublic(Buffer.from(raw)));
  // base64url body carrying a '+' or '/' from standard base64
  assert.throws(() => V.decodePublic('ed25519:' + '+' + body.slice(1)));
  assert.throws(() => V.decodePublic('ed25519:' + '/' + body.slice(1)));

  // non-canonical trailing bits: 32 bytes encode into 43 chars whose last
  // character carries 2 unused bits; setting them must be refused.
  const last = body[body.length - 1];
  const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_';
  const idx = alphabet.indexOf(last);
  const dirty = alphabet[idx ^ 0x01];
  assert.notStrictEqual(dirty, last);
  const mutated = 'ed25519:' + body.slice(0, -1) + dirty;
  if (Buffer.from(mutated.slice(8), 'base64url').toString('base64url') !== mutated.slice(8)) {
    assert.throws(() => V.decodePublic(mutated));
  } else {
    // that flip happened to stay canonical; try the other low bit
    const dirty2 = alphabet[idx ^ 0x02];
    const mutated2 = 'ed25519:' + body.slice(0, -1) + dirty2;
    assert.throws(() => V.decodePublic(mutated2));
  }

  const sigRaw = fromHex(ED_VECTORS[0].sig);
  const encSig = V.encodeSignature(sigRaw);
  assert.throws(() => V.decodeSignature(encSig.slice(0, -1)));
  assert.throws(() => V.decodePublic(encSig));       // 64 bytes, want 32
  assert.throws(() => V.decodeSignature(enc));       // 32 bytes, want 64
  assert.throws(() => V.encodePublic(Buffer.alloc(31)), TypeError);
  assert.throws(() => V.encodeSignature(Buffer.alloc(63)), TypeError);
});

// ---------------------------------------------------------------------------
// 8. tagged()
// ---------------------------------------------------------------------------

test('tagged() prefixes an ASCII tag and a NUL', () => {
  const d = fromHex(CT_ROOTS[7]);
  const tag = 'styxx.v8/cert/1';
  const got = V.tagged(tag, d);
  const want = Buffer.concat([Buffer.from(tag, 'ascii'), Buffer.from([0x00]), d]);
  assert.ok(got.equals(want));
  assert.strictEqual(got.length, tag.length + 1 + 32);
  assert.strictEqual(got[tag.length], 0x00);
  console.log('[8] tagged sha256: ' + V.sha256Hex(got));

  const other = V.tagged('styxx.v8/cert/2', d);
  assert.ok(!got.equals(other));
  assert.ok(!V.tagged('a', d).equals(V.tagged('b', d)));
  // a tag and a longer tag cannot collide across the separator
  assert.ok(!V.tagged('ab', d).equals(V.tagged('a', d)));

  assert.throws(() => V.tagged('', d), TypeError);
  assert.throws(() => V.tagged('a\u0000b', d), TypeError);
  assert.throws(() => V.tagged('caf\u00e9', d), TypeError);
  assert.throws(() => V.tagged('\ud83d\ude00', d), TypeError);
  assert.throws(() => V.tagged(tag, Buffer.alloc(31)), TypeError);
  assert.throws(() => V.tagged(tag, null), TypeError);
  assert.throws(() => V.tagged(null, d), TypeError);
});

// ---------------------------------------------------------------------------
// leaf/node/digest basics
// ---------------------------------------------------------------------------

test('leaf and node hashing use the RFC 6962 prefixes', () => {
  assert.strictEqual(hex(V.leafHash(Buffer.alloc(0))), CT_ROOTS[0]);
  assert.strictEqual(
    hex(V.leafHash(Buffer.alloc(0))),
    crypto.createHash('sha256').update(Buffer.from([0x00])).digest('hex'));
  const l = V.leafHash(Buffer.from('l'));
  const r = V.leafHash(Buffer.from('r'));
  assert.strictEqual(
    hex(V.nodeHash(l, r)),
    crypto.createHash('sha256')
      .update(Buffer.concat([Buffer.from([0x01]), l, r])).digest('hex'));
  assert.notStrictEqual(hex(V.nodeHash(l, r)), hex(V.nodeHash(r, l)));
  assert.strictEqual(V.digest({ b: 1, a: 2 }), V.digest({ a: 2, b: 1 }));
  assert.strictEqual(V.digest({}), V.sha256Hex(Buffer.from('{}', 'utf8')));
  console.log('[x] digest({}) = ' + V.digest({}));
});
// ---------------------------------------------------------------------------
// spec section 9 rule 1 -- challenge validity, computed from the two certs
//
// This is the second implementation of the rule.  The defect it exists for is
// C1 of papers/v8/challenge_and_attack_2026_09_09: an fp16 fingerprint filed as
// the `own` half of a challenge against a bf16 target, and every layer -- this
// verifier included -- took it.
// ---------------------------------------------------------------------------

const BF16 = {
  subject: {
    kind: 'weights',
    hf_repo: 'google/gemma-2-2b-it',
    revision: '2'.repeat(40),
    weights_sha256: 'a'.repeat(64),
    config_sha256: 'b'.repeat(64),
    tokenizer_sha256: 'c'.repeat(64),
    generation_config_sha256: 'd'.repeat(64),
    precision: 'bfloat16'
  },
  recipe: {
    battery: 'sha256:' + 'e'.repeat(64),
    decoding: { temperature: 0, top_p: 1, max_new_tokens: 16, batch_size: 1 },
    chat_template_sha256: 'f'.repeat(64),
    system_prompt_sha256: '0'.repeat(64)
  }
};

function withSubject(over) {
  return {
    subject: Object.assign({}, BF16.subject, over),
    recipe: JSON.parse(JSON.stringify(BF16.recipe))
  };
}

test('a reproduction of the same subject and recipe is a valid challenge', () => {
  assert.deepStrictEqual(V.challengeValidity(BF16, JSON.parse(JSON.stringify(BF16))), []);
});

test('an fp16 fingerprint is not a challenge to a bf16 cert', () => {
  const own = withSubject({ precision: 'float16' });
  const reasons = V.challengeValidity(BF16, own);
  console.log('[9] fp16 against bf16: ' + JSON.stringify(reasons));
  assert.deepStrictEqual(reasons, ['cross-subject:precision']);
});

test('a different revision or a different weights hash is not a challenge', () => {
  assert.deepStrictEqual(
    V.challengeValidity(BF16, withSubject({ revision: '3'.repeat(40) })),
    ['cross-subject:revision']);
  assert.deepStrictEqual(
    V.challengeValidity(BF16, withSubject({ weights_sha256: '9'.repeat(64) })),
    ['subject.weights_sha256']);
});

test('a different battery or decoding is not a challenge', () => {
  const own = JSON.parse(JSON.stringify(BF16));
  own.recipe.battery = 'sha256:' + '1'.repeat(64);
  assert.deepStrictEqual(V.challengeValidity(BF16, own), ['recipe.battery']);
  const other = JSON.parse(JSON.stringify(BF16));
  other.recipe.decoding.batch_size = 8;
  assert.deepStrictEqual(V.challengeValidity(BF16, other), ['recipe.decoding']);
});

test('the environment a challenge exists to differ on is not part of validity', () => {
  const own = JSON.parse(JSON.stringify(BF16));
  own.subject.environment = {
    hardware: { gpu: 'NVIDIA RTX 4090', driver: '999.99', count: 1 },
    runtime: { framework: 'transformers', version: '5.0.0', backend: 'torch 2.9' }
  };
  own.recipe.harness = { name: 'styxx', version: '8.0.1' };
  assert.deepStrictEqual(V.challengeValidity(BF16, own), []);
});

test('an alias subject is compared on its own S_identity fields', () => {
  const a = { subject: { kind: 'alias', provider: 'p', alias: 'm', region: 'eu' }, recipe: {} };
  const b = { subject: { kind: 'alias', provider: 'p', alias: 'm', region: 'us' }, recipe: {} };
  assert.deepStrictEqual(V.challengeValidity(a, b), ['subject.region']);
  const c = { subject: { kind: 'weights' }, recipe: {} };
  assert.ok(V.challengeValidity(a, c).includes('subject.kind'));
});

test('a synthetic body is not a challenge to a measured one', () => {
  // C-MOCK of papers/v8/challenge_and_attack_2026_09_09: distances a runner holding no model
  // produced are labelled in the signed body, and the second implementation reads the label too.
  const fake = JSON.parse(JSON.stringify(BF16));
  fake.body = { synthetic: true };
  assert.deepStrictEqual(V.challengeValidity(BF16, fake), ['body.synthetic']);
  assert.deepStrictEqual(V.challengeValidity(fake, BF16), ['body.synthetic']);
  // Two synthetic runs still compare with each other, and the marker is never softened.
  const alsoFake = JSON.parse(JSON.stringify(fake));
  assert.deepStrictEqual(V.challengeValidity(fake, alsoFake), []);
  assert.strictEqual(V.isSynthetic(fake), true);
  assert.strictEqual(V.isSynthetic(BF16), false);
  // Fail closed on hostile values: present and not `false` is the marker, so `null` and a
  // string both read as synthetic rather than as a measurement.
  for (const value of [null, 'no', 0, {}]) {
    const sloppy = JSON.parse(JSON.stringify(BF16));
    sloppy.body = { synthetic: value };
    assert.strictEqual(V.isSynthetic(sloppy), true, JSON.stringify(value));
  }
  const off = JSON.parse(JSON.stringify(BF16));
  off.body = { synthetic: false };
  assert.strictEqual(V.isSynthetic(off), false);
});

test('challengeValidity never throws on hostile input', () => {
  assert.deepStrictEqual(V.challengeValidity(null, null), []);
  assert.deepStrictEqual(V.challengeValidity({}, {}), []);
  assert.deepStrictEqual(V.challengeValidity({ subject: 'x' }, { subject: [] }), []);
  assert.deepStrictEqual(V.challengeValidity(BF16, {}).sort().length > 0, true);
});

// ---------------------------------------------------------------------------
// spec section 8.1 -- the log a cert binds itself to (L7)
//
// The defect this exists for: seven entries of a published log were replayed
// verbatim into a log built on a different key.  All seven appended, byte for
// byte, and gave the same Merkle root under a different `log_id`, so "this cert
// is in the log" was a statement about bytes and not about a log.  Until this
// section existed, the second implementation read a bound cert as unbound, and
// the two disagreed about whether a cert names its log.
// ---------------------------------------------------------------------------

const LOG_A = 'sha256:' + '3'.repeat(64);
const LOG_B = 'sha256:' + '4'.repeat(64);

function boundTo(logId, extra) {
  const cert = JSON.parse(JSON.stringify(BF16));
  cert.body = { log_hint: Object.assign({ log_id: logId }, extra || {}) };
  return cert;
}

test('a cert naming this log is seated here, and one naming another log is not', () => {
  assert.strictEqual(V.logBindingReason(boundTo(LOG_A), LOG_A), null);
  const why = V.logBindingReason(boundTo(LOG_B), LOG_A);
  console.log('[8.1] wrong log: ' + why);
  assert.ok(typeof why === 'string' && why.includes(LOG_B) && why.includes(LOG_A));
  // `locations` is advisory freight beside the binding and decides nothing.
  assert.strictEqual(
    V.logBindingReason(boundTo(LOG_A, { locations: ['https://example.invalid/log'] }), LOG_A),
    null);
  assert.strictEqual(V.isLogBound(boundTo(LOG_A)), true);
});

test('an unbound cert is seated by any log, which is what optional costs', () => {
  // Not a pass mark: it is the pinned LIMIT.  Requiring the field would refuse every
  // cert already signed, and a cert cannot name a log that does not exist yet, so an
  // unbound cert is exactly as replayable as it was before the field.
  assert.strictEqual(V.logBindingReason(BF16, LOG_A), null);
  assert.strictEqual(V.logBindingReason(BF16, LOG_B), null);
  assert.strictEqual(V.isLogBound(BF16), false);
  const nulled = JSON.parse(JSON.stringify(BF16));
  nulled.body = { log_hint: null };
  assert.strictEqual(V.logBindingReason(nulled, LOG_A), null);
  assert.strictEqual(V.isLogBound(nulled), false);
});

test('a malformed binding is refused rather than ignored', () => {
  // A binding that cannot be compared with anything is not a weaker binding; it is a
  // claim shaped like one, and reading it as "unbound" would let an attacker turn a
  // bound cert loose by breaking the field rather than by removing it.
  const shapes = [
    'not-an-object', 42, true, [], ['sha256:' + '3'.repeat(64)],
    { log_id: 'sha256:zz' },
    { log_id: '3'.repeat(64) },
    { log_id: 'sha256:' + '3'.repeat(63) },
    { log_id: 'sha256:' + 'A'.repeat(64) },
    { log_id: null },
    { locations: ['mirror'] },
    {}
  ];
  for (const shape of shapes) {
    const cert = JSON.parse(JSON.stringify(BF16));
    cert.body = { log_hint: shape };
    const why = V.logBindingReason(cert, LOG_A);
    assert.ok(typeof why === 'string' && why.startsWith('log_hint:'), JSON.stringify(shape));
  }
});

test('log_id is the hash of the raw log public key', () => {
  const pair = crypto.generateKeyPairSync('ed25519');
  const raw = pair.publicKey.export({ format: 'der', type: 'spki' }).subarray(-32);
  const expected = 'sha256:' + crypto.createHash('sha256').update(raw).digest('hex');
  assert.strictEqual(V.logIdFromPublic(raw), expected);
  assert.strictEqual(V.logIdFromPublic(V.encodePublic(raw)), expected);
  // A published pair: keys/log.pub and the log_id its own signed tree head carries,
  // both produced by the other implementation.
  assert.strictEqual(
    V.logIdFromPublic('ed25519:FG7r2x9S5_A7hNy4MlGnQQGP1_30lKNSH8v50S9f3OM'),
    'sha256:88300489aaab498ea197b13fae57c7fb7ba9c23d49cd97be59f5deaa9bae7c4f');
  assert.throws(() => V.logIdFromPublic('ed25519:' + 'A'.repeat(10)), /decoded/);
  assert.throws(() => V.logIdFromPublic(Buffer.alloc(31)), TypeError);
});

test('the binding predicate refuses hostile certs and throws on a bad log id', () => {
  // The cert is input from the wire: refused, never crashed on.  The log id is the
  // caller's own pinned key: a malformed one is the caller's defect and is thrown.
  for (const cert of [null, undefined, 42, 'x', [], {}, { body: null }, { body: 'x' },
                      { body: [] }, { body: { log_hint: undefined } }]) {
    assert.strictEqual(V.logBindingReason(cert, LOG_A), null, JSON.stringify(cert) || 'undefined');
    assert.strictEqual(V.isLogBound(cert), false);
  }
  for (const bad of [null, undefined, '', 'sha256:zz', '3'.repeat(64), LOG_A.toUpperCase(), 7,
                     Buffer.alloc(32)]) {
    assert.throws(() => V.logBindingReason(boundTo(LOG_A), bad), TypeError);
  }
});
