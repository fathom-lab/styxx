// Verify a REAL styxx log using only the independent JavaScript implementation.
//
// styxx/_data/v8_verify.js was written from the specification and the RFCs by an author who was
// forbidden to read the Python. Until now it had only ever been run against synthetic test
// vectors: the RFC 8785 example, the Certificate Transparency roots, the RFC 8032 signatures.
// Agreement on published vectors shows the two implementations read the same standards. It does
// not show they produce the same bytes for a real cert, which is a much larger object with real
// unicode, real floats, real nesting, and the domain-tag and tree-head constructions this project
// invented rather than inherited.
//
// This script points the JavaScript at a log the Python produced and asks the questions Appendix D
// asks a stranger to ask, using none of the Python:
//
//   1. does every entry's cert id recompute from its own bytes?
//   2. does every cert's signature verify under its issuer key?
//   3. do the entries reproduce the root the tree head signs?
//   4. does the tree head's signature verify under the pinned log key?
//   5. does an inclusion proof verify?
//
// A disagreement here is worth more than agreement. Agreement is one more green check; a
// disagreement is a place where two honest readings of the same specification produce different
// bytes, which is the defect a content-addressed format cannot carry.
//
// Usage:  node cross_verify.js <log directory>

'use strict';

const fs = require('fs');
const path = require('path');
const V = require(path.join(__dirname, '..', '..', '..', 'styxx', '_data', 'v8_verify.js'));

const CERT_TAG = 'styxx.v8/cert/1';
const STH_TAG = 'styxx.v8/sth/1';

const logDir = process.argv[2];
if (!logDir) {
  console.error('usage: node cross_verify.js <log directory>');
  process.exit(2);
}

const findings = [];
const note = (ok, what, detail) => {
  findings.push({ ok, what, detail });
  console.log(`${ok ? 'OK  ' : 'FAIL'}  ${what}${detail ? '  ::  ' + detail : ''}`);
};

// ---------------------------------------------------------------- entries

function entryFiles(dir) {
  const root = path.join(dir, 'entries');
  const out = [];
  for (const sub of fs.readdirSync(root)) {
    const p = path.join(root, sub);
    if (!fs.statSync(p).isDirectory()) continue;
    for (const f of fs.readdirSync(p)) {
      if (/^\d+\.json$/.test(f)) out.push(path.join(p, f));
    }
  }
  return out.sort();
}

const files = entryFiles(logDir);
console.log(`\n[entries] ${files.length} found in ${logDir}\n`);

const leaves = [];
for (const f of files) {
  const bytes = fs.readFileSync(f);            // the entry bytes, exactly as stored
  const name = path.basename(f);
  let cert;
  try {
    cert = JSON.parse(bytes.toString('utf8'));
  } catch (e) {
    note(false, `${name} parses`, e.message);
    continue;
  }

  // 1. the id must recompute from the cert's own bytes, with id and sig removed
  const { id, sig, ...core } = cert;
  let recomputed;
  try {
    recomputed = 'sha256:' + V.digest(core);
  } catch (e) {
    note(false, `${name} canonicalizes`, e.message);
    continue;
  }
  note(recomputed === id, `${name} id recomputes`,
       recomputed === id ? id.slice(0, 26) + '...' : `stored ${id} != computed ${recomputed}`);

  // 2. the signature, over the domain-separated digest
  try {
    const pub = V.decodePublic(cert.issuer.key);
    const sigBytes = V.decodeSignature(sig);
    const digest32 = Buffer.from(recomputed.slice('sha256:'.length), 'hex');
    const preimage = V.tagged(CERT_TAG, digest32);
    note(V.verifySignature(pub, preimage, sigBytes), `${name} signature verifies`,
         `${cert.type} by ${cert.issuer.name}`);
  } catch (e) {
    note(false, `${name} signature verifies`, e.message);
  }

  // 3. the entry bytes must BE their own canonical form; anything else is a second
  //    representation of the same cert and would give the tree two leaves for one entry
  try {
    const canon = V.canonicalBytes(cert);
    note(Buffer.compare(canon, bytes) === 0, `${name} bytes are canonical`,
         Buffer.compare(canon, bytes) === 0 ? `${bytes.length} bytes` :
         `stored ${bytes.length} vs canonical ${canon.length}`);
  } catch (e) {
    note(false, `${name} bytes are canonical`, e.message);
  }

  leaves.push(V.leafHash(bytes));
}

// ---------------------------------------------------------------- the tree head

const sthDir = path.join(logDir, 'sth');
const sths = fs.existsSync(sthDir) ? fs.readdirSync(sthDir).filter(f => f.endsWith('.json')).sort() : [];
if (sths.length === 0) note(false, 'a signed tree head exists', 'none found');

const pubPath = path.join(logDir, 'keys', 'log.pub');
const logPubText = fs.existsSync(pubPath) ? fs.readFileSync(pubPath, 'utf8').trim() : null;

for (const s of sths) {
  const sth = JSON.parse(fs.readFileSync(path.join(sthDir, s), 'utf8'));
  const size = sth.tree_size;

  // 4. the entries must reproduce the root the head commits to
  let computedRoot;
  try {
    computedRoot = 'sha256:' + V.root(leaves.slice(0, size)).toString('hex');
  } catch (e) {
    note(false, `${s} root recomputes`, e.message);
    continue;
  }
  note(computedRoot === sth.root_hash, `${s} entries reproduce the signed root`,
       computedRoot === sth.root_hash ? `tree_size=${size} ${computedRoot.slice(0, 26)}...`
                                      : `signed ${sth.root_hash} != computed ${computedRoot}`);

  // 5. the head's own signature, under the key pinned out of band
  if (logPubText) {
    try {
      const pub = V.decodePublic(logPubText);
      const { sig, ...signed } = sth;
      const inner = Buffer.from(V.sha256Hex(V.canonicalBytes(signed)), 'hex');
      const preimage = V.tagged(STH_TAG, inner);
      note(V.verifySignature(pub, preimage, V.decodeSignature(sig)),
           `${s} tree head signature verifies`, `log_id ${String(sth.log_id).slice(0, 20)}...`);
    } catch (e) {
      note(false, `${s} tree head signature verifies`, e.message);
    }
  }

  // 6. an inclusion proof, generated and checked entirely in JavaScript
  if (size > 1) {
    try {
      const idx = size - 1;
      const proof = V.inclusionProof(leaves.slice(0, size), idx, size);
      const ok = V.verifyInclusion(leaves[idx], idx, size,
                                   proof, Buffer.from(computedRoot.slice(7), 'hex'));
      note(ok, `${s} inclusion proof for leaf ${idx} verifies`, `path length ${proof.length}`);
      // and it must NOT verify against a different leaf: a check that accepts everything
      // checks nothing
      const bad = V.verifyInclusion(leaves[0], idx, size,
                                    proof, Buffer.from(computedRoot.slice(7), 'hex'));
      note(bad === false, `${s} the same proof is refused for the wrong leaf`, 'negative control');
    } catch (e) {
      note(false, `${s} inclusion proof`, e.message);
    }
  }
}

// ---------------------------------------------------------------- verdict

const failed = findings.filter(f => !f.ok);
console.log(`\n${findings.length - failed.length} of ${findings.length} checks agree`);
if (failed.length) {
  console.log('\nDISAGREEMENTS — each is a place two readings of the spec diverge:');
  for (const f of failed) console.log(`  - ${f.what}: ${f.detail}`);
}
fs.writeFileSync(path.join(__dirname, 'cross_verify_report.json'),
                 JSON.stringify({ log: logDir, entries: files.length, findings }, null, 2) + '\n');
process.exit(failed.length ? 1 : 0);
