// Run the cross-certificate floor predicate of the INDEPENDENT JavaScript verifier
// against a real styxx log, and against the same forgery the Python receipt builds.
//
// Why this is a separate script and not more checks inside cross_verify.js:
// cross_verify.js writes a committed report file that is a receipt of one run of one
// check set against one log. Adding checks to it changes the count that receipt names,
// and a receipt regenerated in place stops being evidence about the run it names. A
// second question gets a second script and its own report file.
//
// THE QUESTION. papers/v8/THE_BOUNDARY_2026_09_09.md, roster member 1: five copies of
// one forward pass carrying five batch labels give a floor of 0.0, and five real runs
// that happened to agree exactly give the same bytes -- so no predicate can tell them
// apart. That is true of ONE certificate and false of a certificate in a log that
// already holds a measurement of the same subject on the same battery.
//
// THREE CONTROLS, following papers/v8/class_two_empty_2026_09_09/member1_demo.py:
//   A  the honest published floor against itself           -> must be 'agrees'
//   B  the roster's forgery against the prior logged floor -> must be 'contradicts'
//   C  the same forgery with no prior on the subject       -> must be 'unconstrained'
// and one this JavaScript adds:
//   D  the forgery against a prior on a DIFFERENT subject  -> must be 'unconstrained'
//      (the Python computes a subject key and then never consults it, so a prior from
//      any subject in the log would have been allowed to accuse; see the report)
//
// Usage:  node cross_verify_floor.js <log directory> [report path]

'use strict';

const fs = require('fs');
const path = require('path');
const V = require(path.join(__dirname, '..', '..', '..', 'styxx', '_data', 'v8_verify.js'));

const logDir = process.argv[2];
if (!logDir) {
  console.error('usage: node cross_verify_floor.js <log directory> [report path]');
  process.exit(2);
}
const reportPath = process.argv[3]
  ? path.resolve(process.argv[3])
  : path.join(__dirname, 'cross_verify_floor_report.json');

const findings = [];
const note = (ok, what, detail) => {
  findings.push({ ok, what, detail });
  console.log(`${ok ? 'OK  ' : 'FAIL'}  ${what}${detail ? '  ::  ' + detail : ''}`);
};

// ---------------------------------------------------------------- the log's own bytes

function entryFiles(dir) {
  const rootDir = path.join(dir, 'entries');
  const out = [];
  for (const sub of fs.readdirSync(rootDir)) {
    const p = path.join(rootDir, sub);
    if (!fs.statSync(p).isDirectory()) continue;
    for (const f of fs.readdirSync(p)) {
      if (/^\d+\.json$/.test(f)) out.push(path.join(p, f));
    }
  }
  return out.sort();
}

const certs = [];
for (const f of entryFiles(logDir)) {
  const cert = JSON.parse(fs.readFileSync(f, 'utf8'));
  cert.__entry = path.basename(f, '.json');
  certs.push(cert);
}
const floors = certs.filter((c) => c.type === 'fingerprint' && V.noiseFloorOf(c) !== null);
console.log(`\n[log] ${certs.length} entries in ${logDir}; ${floors.length} carry a noise floor\n`);
if (floors.length === 0) {
  note(false, 'the log holds a floor cert to check', 'none found');
  fs.writeFileSync(reportPath, JSON.stringify({ log: logDir, findings }, null, 2) + '\n');
  process.exit(1);
}

const published = floors[0];
const resolved = V.resolveFloorRuns(published, certs);
note(resolved.missing.length === 0,
     `entry ${published.__entry} names ${resolved.named} run certs and the log holds them all`,
     resolved.missing.length ? 'missing ' + resolved.missing.join(', ')
                             : `${resolved.runs.length} runs including the floor cert itself`);

const runs = resolved.runs;
const obs = V.floorObservations(published, runs);
note(obs.defects.length === 0, `entry ${published.__entry} floor is consistent with its own runs`,
     obs.defects.length ? obs.defects.join('; ')
                        : `${obs.runs} runs, ${obs.pairs} pairs, covers [${obs.covers.join(', ')}]`);

// The pair-ordering convention is not written in the cert: the distances are a bare
// list. This script reads them as lexicographic pairs over runs sorted by run_index.
// That reading is checkable against the bytes rather than assumed: under it, the zero
// distances must be exactly the pairs whose covered factors would predict a zero.
const byIndex = runs.slice().sort((a, b) => a.body.run_index - b.body.run_index);
const pairs = [];
for (let i = 0; i < byIndex.length; i += 1) {
  for (let j = i + 1; j < byIndex.length; j += 1) pairs.push([i, j]);
}
const floorBody = V.noiseFloorOf(published);
const channels = Object.keys(floorBody.per_channel).sort();
let sameBatchAllZero = true;
let diffBatchAllPositive = true;
let sameBatchCount = 0;
let diffBatchCount = 0;
for (let k = 0; k < pairs.length; k += 1) {
  const a = byIndex[pairs[k][0]].body.nuisance.batch_size;
  const b = byIndex[pairs[k][1]].body.nuisance.batch_size;
  const values = channels.map((ch) => floorBody.per_channel[ch].distances[k]);
  if (a === b) {
    sameBatchCount += 1;
    if (!values.every((v) => v === 0)) sameBatchAllZero = false;
  } else {
    diffBatchCount += 1;
    if (!values.every((v) => v > 0)) diffBatchAllPositive = false;
  }
}
note(sameBatchAllZero && diffBatchAllPositive,
     'the distance list re-attributes to run pairs under the sorted-run_index convention',
     `${sameBatchCount} same-batch pairs all exactly 0, ${diffBatchCount} different-batch ` +
     `pairs all strictly positive, across ${channels.length} channels`);

// ---------------------------------------------------------------- control A

let r = V.floorAgreement(published, runs, published, runs);
note(r.verdict === V.FLOOR_AGREES, 'A: the honest published floor does not accuse itself',
     `${r.verdict} -- ${r.reasons[0]}`);

// ---------------------------------------------------------------- the forgery

// Exactly the roster's construction: give every run cert run 0's channel values and
// items, keep every run's real nuisance labels, and set every distance to 0. No metric
// is recomputed and none needs to be: d(x, x) = 0 for any metric.
const clone = (o) => JSON.parse(JSON.stringify(o));
const zeroRun = byIndex[0];
const forgedRuns = byIndex.map((c) => {
  const f = clone(c);
  f.body.channels = clone(zeroRun.body.channels);
  f.body.items = clone(zeroRun.body.items);
  return f;
});
const forged = clone(published);
forged.body.channels = clone(zeroRun.body.channels);
forged.body.items = clone(zeroRun.body.items);
for (const ch of Object.keys(forged.body.noise_floor.per_channel)) {
  forged.body.noise_floor.per_channel[ch].distances = pairs.map(() => 0);
  forged.body.noise_floor.per_channel[ch].floor = 0;
}
// The forged floor cert is one of its own runs, so it must be substituted into the run
// set too, or the set would carry the honest run 0 beside the forged floor.
const forgedSet = forgedRuns.map((c) => (c.id === forged.id ? forged : c));

// Within one certificate the forgery is the honest artifact's shape. Recorded, not
// asserted: these are the within-cert checks the roster says cannot separate them.
const labels = forgedSet.map((c) => JSON.stringify(c.body.nuisance, Object.keys(c.body.nuisance).sort()));
console.log(`\n[forgery] ${new Set(labels).size} distinct nuisance labels over ${labels.length} runs; ` +
            `${new Set(forgedSet.map((c) => c.body.nuisance.batch_size)).size} distinct batch sizes; ` +
            `every declared factor still varies\n`);

r = V.floorAgreement(forged, forgedSet, published, runs);
note(r.verdict === V.FLOOR_CONTRADICTS, 'B: the forgery is refused against the prior logged floor',
     `${r.verdict} -- ${r.contradictions.length} of ${r.compared.total} cells contradict ` +
     `(${r.compared.factor} factor-level pairs, ${r.compared.assignment} assignment pairs compared)`);
const attack = r;
for (const c of r.contradictions.slice(0, 3)) console.log('        ' + c.why);
if (r.contradictions.length > 3) console.log(`        ... and ${r.contradictions.length - 3} more`);

// ---------------------------------------------------------------- control C

r = V.floorAgreement(forged, forgedSet, null, []);
note(r.verdict === V.FLOOR_UNCONSTRAINED,
     'C: with no prior measurement the predicate declines rather than accusing',
     `${r.verdict} -- ${r.reasons[0]}`);

// ---------------------------------------------------------------- control D

// A splice of the published cert, at fp16 rather than bf16 -- the second subject this
// verdict run actually names (papers/v8/first_verdict_2026_09_09/subject_fp16.json).
// Its floor is a measurement of a different subject, so it is not evidence about this
// one, and a predicate that let it accuse would be answering without evidence.
const otherSubject = clone(published);
otherSubject.subject.precision = 'fp16';
const otherRuns = runs.map((c) => {
  const f = clone(c);
  f.subject.precision = 'fp16';
  return f;
});
r = V.floorAgreement(forged, forgedSet, otherSubject, otherRuns);
note(r.verdict === V.FLOOR_UNCONSTRAINED,
     'D: a floor on a different subject is not allowed to accuse this one',
     `${r.verdict} -- ${r.reasons[0]}`);

// ---------------------------------------------------------------- verdict

const failed = findings.filter((f) => !f.ok);
console.log(`\n${findings.length - failed.length} of ${findings.length} checks agree`);
if (failed.length) {
  console.log('\nDISAGREEMENTS:');
  for (const f of failed) console.log(`  - ${f.what}: ${f.detail}`);
}
fs.writeFileSync(reportPath, JSON.stringify({
  log: logDir,
  floor_entry: published.__entry,
  runs: obs.runs,
  pairs: obs.pairs,
  covers: obs.covers,
  channels,
  cells_compared: attack.compared,
  contradictions: attack.contradictions.length,
  contradictions_clean: attack.contradictions.filter((c) => c.clean).length,
  contradictions_confounded: attack.contradictions.filter((c) => !c.clean).length,
  findings
}, null, 2) + '\n');
console.log(`\nreport written to ${reportPath}`);
process.exit(failed.length ? 1 : 0);
