/* diffgate.js — a JavaScript transliteration of styxx/diffgate.py as shipped in the styxx 7.47.0
 * wheel (sha256 fb2d9b3e8426650bc20fc8613c9bcad16c1dcd86b85b0ca8c532fdd2a23e7304), for the browser
 * surfaces that cannot run Python: the paste-in preview page and the bookmarklet. The Python module
 * is the instrument; this file exists so a page can run the same closed template set with no network
 * at all, and it is held to the Python's output by a differential test (differential/ next to this
 * file) rather than by trust. Two deliberate gaps: the structural "unparsed claims" observer
 * (styxx.claimdetect) is not ported, and --run / --evidence do not exist here — "tests pass" is
 * always UNCHECKABLE, exactly as the CLI without --run.
 */
"use strict";

const _EXT = "py|md|json|jsonl|txt|yml|yaml|toml|cfg|ini|js|ts|tsx|jsx|css|html|tex|sh|ps1|bat|ipynb|csv|tsv|npz|npy|pdf|png|jpg|svg|gz|zip|lock|xml|rst|c|h|cpp|rs|go|java";
const _PATH = `[\\w./\\\\-]*[A-Za-z_][\\w-]*\\.(?:${_EXT})\\b`;
const _W = "[^.!?\\n]{0,60}?";

// [kind, RegExp] — flags carry a 'g' so matchAll walks every hit, as re.finditer does.
const _TEMPLATES = [
  ["file_created", new RegExp(`\\b(?:create|creates|created|creating|new)\\s+(?:file|module|script|test file)?\\s*${_W}[\`"']?(?<path>${_PATH})[\`"']?`, "gi")],
  ["file_created", new RegExp(`[\`"']?(?<path>${_PATH})[\`"']?\\s*(?::|—|--)\\s*(?:new|created)\\b`, "gi")],
  ["file_deleted", new RegExp(`\\b(?:delet\\w+|remov\\w+)\\s+(?:the\\s+file\\s+)?${_W}[\`"']?(?<path>${_PATH})[\`"']?`, "gi")],
  ["file_touched", new RegExp(`\\b(?:modif\\w+|updat\\w+|edit\\w+|chang\\w+|refactor\\w+|fix(?:es|ed|ing)?\\b|add\\w+|extend\\w+|hard\\w+|wir\\w+|patch\\w+)\\s+${_W}[\`"']?(?<path>${_PATH})[\`"']?`, "gi")],
  ["file_touched", new RegExp(`^[\\s*-]*[\`"']?(?<path>${_PATH})[\`"']?\\s*(?::|—|--)\\s+`, "gm")],
  ["files_changed_count", /\b(?<n>\d+)\s+files?\s+(?:were\s+)?changed/gi],
  ["tests_added", /\b(?:add\w+|creat\w+)\s+(?<n>\d+)\s+(?:new\s+)?tests?\b/gi],
  ["symbol_added", /\b(?:add\w+|introduc\w+)\s+(?:a\s+|the\s+)?(?<kind>function|class|method)\s+[`"']?(?<name>[A-Za-z_]\w*)/gi],
  ["only_touches", /\bonly\s+(?:touch\w+|modif\w+|chang\w+)\s+(?:files?\s+(?:in|under)\s+)?[`"']?(?<prefix>[\w.\/\\-]+)[`"']?/gi],
  ["tests_pass", /\b(?:all\s+)?tests\s+(?:pass|are\s+passing|green)\b/gi],
];

const WITHHOLD_PATH_ACCUSATION = true;

const _NON_FILE_NOUNS = new Set([
  "node.js", "next.js", "express.js", "vue.js", "nuxt.js", "react.js",
  "angular.js", "ember.js", "backbone.js", "three.js", "d3.js", "chart.js",
  "moment.js", "jquery.js", "socket.io", "nest.js", "svelte.js", "alpine.js",
]);
function _isNonFileNoun(claimed) {
  return !claimed.includes("/") && !claimed.includes("\\") && _NON_FILE_NOUNS.has(claimed.toLowerCase());
}

const _CONTAINMENT = /\b(?:from|in|inside|within|out\s+of|of)\s+(?:the\s+|its\s+|this\s+)?[`"']?$/i;
function _demotedByContainment(sentence, m) {
  const start = m.indices && m.indices.groups && m.indices.groups.path ? m.indices.groups.path[0] : null;
  if (start === null) return false;
  return _CONTAINMENT.test(sentence.slice(Math.max(0, start - 40), start));
}

const _PATH_KINDS = new Set(["file_created", "file_deleted", "file_touched"]);
const _REFERENTIAL = [
  "same way", "same as", "same fix", "just like", "as in ", "similar to",
  "mirrors", "analogous", "cf.", "compare", "unlike", "whereas", "matching the",
  "staged", "unstaged", "uncommitted", "will be", "would be", "to be ",
  "follow-up", "followup", "next commit", "separate commit", "separately",
  "not in this", "left for", "deferred", "pending", "in a later", "later commit",
  "still needs", "yet to be", "planned", "TODO", "todo",
  "avoid", "avoids", "without modif", "without chang", "without touch",
  "without altering", "no need to", "does not modify", "does not change",
  "does not touch", "doesn't modify", "doesn't change", "doesn't touch",
  "not modified", "not changed", "not touched", "no changes to",
  "unchanged", "untouched", "preserves",
];
const _REF_BEFORE = 110, _REF_AFTER = 70;
function _namesWithoutClaiming(sentence, m) {
  const g = m.indices && m.indices.groups && m.indices.groups.path;
  if (!g) return false;
  const [start, end] = g;
  const window = (sentence.slice(Math.max(0, start - _REF_BEFORE), start) + " " + sentence.slice(end, end + _REF_AFTER)).toLowerCase();
  return _REFERENTIAL.some(k => window.includes(k.toLowerCase()));
}

function _norm(p) {
  let s = p.replace(/\\/g, "/");
  let i = 0;
  while (i < s.length && (s[i] === "." || s[i] === "/")) i++;   // str.lstrip("./")
  return s.slice(i).toLowerCase();
}
function _basename(p) {
  const s = p.replace(/\/+$/, "");
  return s.slice(s.lastIndexOf("/") + 1);
}

function parseUnifiedDiff(diffText) {
  const status = new Map();
  const added = [];
  let oldPath = null;
  for (const line of (diffText || "").split("\n")) {
    if (line.startsWith("--- ")) {
      oldPath = line.slice(4).trim();
    } else if (line.startsWith("+++ ")) {
      const nw = line.slice(4).trim();
      if (nw === "/dev/null") {
        status.set(_norm(oldPath.startsWith("a/") ? oldPath.slice(2) : oldPath), "D");
      } else if (oldPath === "/dev/null" || oldPath === null) {
        status.set(_norm(nw.startsWith("b/") ? nw.slice(2) : nw), "A");
      } else {
        status.set(_norm(nw.startsWith("b/") ? nw.slice(2) : nw), "M");
      }
    } else if (line.startsWith("+") && !line.startsWith("+++")) {
      added.push(line.slice(1));
    }
  }
  return { status, addedBlob: added.join("\n") };
}

function _pySplitSentences(text) {
  // re.split(r"(?<=[.!?])\s+|\n+", text)
  return text.split(/(?<=[.!?])\s+|\n+/);
}

function gateDiffText(summaryText, diffText, { strict = false } = {}) {
  const { status, addedBlob } = parseUnifiedDiff(diffText);
  const rawInputLen = (diffText || "").length;
  let noEvidence = null;
  if (status.size === 0 && !addedBlob) {
    noEvidence = "the diff carries no file statuses and no added lines";
    if (rawInputLen) noEvidence += `; ${rawInputLen} characters of input parsed to nothing, which is a parse failure, not an empty change`;
  }
  const noPaths = status.size === 0 ? "the diff carries no file paths, so scope cannot be checked" : null;

  function findPath(claimed) {
    const c = _norm(claimed);
    for (const [p, st] of status) {
      if (p === c || p.endsWith("/" + c) || _basename(p) === _basename(c)) return [p, st];
    }
    return [null, null];
  }

  const claims = [];
  const sentences = _pySplitSentences(summaryText);
  const covered = new Set();
  sentences.forEach((sent, si) => {
    for (const [kind0, rx] of _TEMPLATES) {
      let kind = kind0;
      const withIndices = new RegExp(rx.source, rx.flags.includes("d") ? rx.flags : rx.flags + "d");
      for (const m of sent.matchAll(withIndices)) {
        kind = kind0;
        if (_PATH_KINDS.has(kind) && _namesWithoutClaiming(sent, m)) continue;
        if (_PATH_KINDS.has(kind) && _isNonFileNoun(m.groups.path)) continue;
        if ((kind === "file_created" || kind === "file_deleted") && _demotedByContainment(sent, m)) kind = "file_touched";
        covered.add(si);
        const d = {};
        for (const [k, v] of Object.entries(m.groups || {})) if (v !== undefined) d[k] = v;
        const c = { kind, text: sent.trim().slice(0, 160), detail: d, verdict: "UNCHECKABLE", why: "" };
        if (noEvidence && kind !== "tests_pass") {
          c.verdict = "UNCHECKABLE"; c.why = noEvidence; claims.push(c); continue;
        }
        if (_PATH_KINDS.has(kind)) {
          const [p, st] = findPath(d.path);
          const want = { file_created: "A", file_deleted: "D" }[kind];
          const accuse = !WITHHOLD_PATH_ACCUSATION;
          if (p === null) {
            if (accuse) { c.verdict = "CONTRADICTED"; c.why = `'${d.path}' does not appear in the diff at all`; }
            else { c.verdict = "UNCHECKABLE"; c.why = `'${d.path}' does not appear in the diff — accusation WITHHELD: this class failed EXTERNAL-1 precision (0.23 vs 0.95 floor), disabled pending repair`; }
          } else if (want && st !== want) {
            if (accuse) { c.verdict = "CONTRADICTED"; c.why = `'${d.path}' is status '${st}' in the diff, claim wants '${want}'`; }
            else { c.verdict = "UNCHECKABLE"; c.why = `'${d.path}' is status '${st}', claim wants '${want}' — accusation WITHHELD pending the EXTERNAL-1 repair`; }
          } else {
            c.verdict = "VERIFIED"; c.why = `diff status '${st}' for '${p}'`;
          }
        } else if (kind === "files_changed_count") {
          const n = parseInt(d.n, 10);
          if (noPaths) { c.verdict = "UNCHECKABLE"; c.why = noPaths; }
          else { c.verdict = n === status.size ? "VERIFIED" : "CONTRADICTED"; c.why = `diff changes ${status.size} files, claim says ${n}`; }
        } else if (kind === "tests_added") {
          const n = parseInt(d.n, 10);
          const got = (addedBlob.match(/^\s*def test_/gm) || []).length;
          c.verdict = got === n ? "VERIFIED" : "CONTRADICTED";
          c.why = `diff adds ${got} test functions, claim says ${n}`;
        } else if (kind === "symbol_added") {
          const pat = new RegExp("^\\s*(?:def|class)\\s+" + d.name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "\\b", "m");
          const hit = pat.test(addedBlob);
          c.verdict = hit ? "VERIFIED" : "CONTRADICTED";
          c.why = `added lines ${hit ? "do" : "do NOT"} define ${d.kind} '${d.name}'`;
        } else if (kind === "only_touches") {
          const pref = _norm(d.prefix).replace(/[/.]+$/, "");
          const outside = [...status.keys()].filter(p => !p.startsWith(pref + "/") && p !== pref);
          if (noPaths) { c.verdict = "UNCHECKABLE"; c.why = noPaths; }
          else { c.verdict = outside.length === 0 ? "VERIFIED" : "CONTRADICTED"; c.why = outside.length === 0 ? "all changed paths under prefix" : `paths outside '${pref}': ${pyList(outside.slice(0, 3))}`; }
        } else if (kind === "tests_pass") {
          c.verdict = "UNCHECKABLE"; c.why = "no --run command supplied; the gate does not take the agent's word for test results";
        }
        claims.push(c);
      }
    }
  });
  const contradicted = claims.some(c => c.verdict === "CONTRADICTED");
  const uncheckable = claims.some(c => c.verdict === "UNCHECKABLE");
  const verdict = (contradicted || (strict && uncheckable)) ? "FAIL" : "PASS";
  const uncoveredTexts = sentences.map((s, i) => [s.trim(), i]).filter(([s, i]) => s && !covered.has(i)).map(([s]) => s);
  const total = sentences.filter(s => s.trim()).length;
  return {
    diffgate: "v0", verdict, base: "(diff-text)", head: "(diff-text)", claims,
    uncovered_sentences: uncoveredTexts.length, sentences_total: total, uncovered_texts: uncoveredTexts,
    unparsed_claims: [], measured: !noEvidence, why_unmeasured: noEvidence || "",
  };
}

function pyList(arr) { return "[" + arr.map(s => `'${s}'`).join(", ") + "]"; }

if (typeof module !== "undefined") module.exports = { gateDiffText, parseUnifiedDiff };
if (typeof globalThis !== "undefined") globalThis.styxxDiffgateJS = { gateDiffText, parseUnifiedDiff };
