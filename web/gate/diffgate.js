/* diffgate.js — a JavaScript transliteration of styxx/diffgate.py for the browser surfaces that
 * cannot run Python: the paste-in preview page and the bookmarklet. The Python module is the
 * instrument; this file exists so a page can run the same closed template set with no network at
 * all, and it is held to the Python's output by a differential test (differential/ next to this
 * file) rather than by trust.
 *
 * Which Python: the file on the BC-2 + COMPAT-1 + BIN-1 checkout (pull requests #113, #115 and the
 * #118 repair on fathom-lab/styxx), sha256 397624d583edc3a147c74bf8791e5356f26a946c7f905d851e453b5297dc40a1, re-cut for the PATH-2
 * repairs (PREREG_path2_resolution_2026_09_17: #97, #121, #101, as amended by AMENDMENT_path2_resolution_2026_09_17)
 * on the file that carries them, sha256
 * d9f8ddd58841d875287dd636f722965cc66424885bdcd0e07f171a6fb6e16dbf. That file also carries COMPAT-2's sharpened compatibility
 * reading (surface vs scaffolding, signature changes, the candidate flag), which this port does NOT
 * carry: its `compat_claim` reasons and detail are COMPAT-1's, and the differential reports those
 * records as disagreements. Relative to the 7.47.0 wheel the port
 * was first cut from, that file carries: the V14 repairs (containment demotes "touched" claims too;
 * a bare basename absent from the diff abstains), the BC-2 repairs for issue #110 (the def-counting
 * templates abstain when the diff has no Python; "added 3 test cases" is not a count of functions;
 * "adds a method to reload" is not a symbol; "only modifies the footer" is not a path), and the
 * COMPAT-1 reading of "no breaking changes" (one verdict, UNCHECKABLE, the public definitions the
 * diff removed named in the reason). PATH-2 adds: a path claim resolves exact, then suffix, then
 * basename, over every entry (#97); the path key keeps a dotfile's dots, and "only touches" does not
 * accuse on a dot alone and lists only the paths outside by more than a dot (#121); a `def` the same
 * file's removed lines also define is changed, not added, paired one to one per name (#101). Two
 * deliberate gaps remain: the structural "unparsed claims"
 * observer (styxx.claimdetect) is not ported, and --run / --evidence do not exist here — "tests
 * pass" is always UNCHECKABLE, exactly as the CLI without --run.
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
  // BC-1/BC-2: the counted noun is captured so "added 3 test cases" is not read as a count of
  // `def test_` functions, and "added a function named foo" reads foo, not `named`.
  ["tests_added", /\b(?:add\w+|creat\w+)\s+(?<n>\d+)\s+(?:new\s+)?tests?\b(?:\s+(?<noun>cases?|files?|scenarios?|suites?|class(?:es)?|functions?|methods?)\b)?/gi],
  ["symbol_added", /\b(?:add\w+|introduc\w+)\s+(?:(?:a|an|the|new)\s+){0,2}(?<kind>function|class|method)\s+(?:(?:named|called)\s+)?[`"']?(?<name>[A-Za-z_]\w*)/gi],
  ["only_touches", /\bonly\s+(?:touch\w+|modif\w+|chang\w+)\s+(?:files?\s+(?:in|under)\s+)?[`"']?(?<prefix>[\w.\/\\-]+)[`"']?(?:,?\s+and\s+(?:files?\s+(?:in|under)\s+)?[`"']?(?<prefix2>[\w-]*[.\/\\][\w.\/\\-]*)[`"']?)?/gi],
  ["tests_pass", /\b(?:all\s+)?tests\s+(?:pass|are\s+passing|green)\b/gi],
  // COMPAT-1: the compatibility claim. Read, never judged.
  ["compat_claim", /\b(?:no\s+breaking\s+changes?|non[- ]breaking|backwards?[- ]compatib(?:le|ility)|(?:zero|no)\s+(?:behaviou?r(?:al)?|functional)\s+changes?|fully\s+compatible|does\s+not\s+(?:break|change)\s+(?:any\s+|the\s+)?(?:existing\s+)?(?:behaviou?r|api|public\s+api))\b/gi],
];

const WITHHOLD_PATH_ACCUSATION = true;
const V14_CONTAINMENT_TOUCH = true;
const V14_BARE_NAME_ABSTAIN = true;
const BC1_BY_CONSTRUCTION = true;

const _PY_SUFFIXES = [".py", ".pyi"];
const _TEST_NOUNS_NOT_FUNCTIONS = new Set(["case", "cases", "file", "files", "scenario",
  "scenarios", "suite", "suites", "class", "classes"]);
const _SYMBOL_WORDS = new Set([
  "to", "with", "that", "for", "in", "on", "of", "by", "and", "or", "as", "the", "a",
  "an", "this", "which", "it", "its", "is", "declaration", "implementation",
  "definition", "signature", "body", "stub", "call", "wrapper", "override",
  "overload", "level", "support", "named", "called",
]);

function _diffTouchesPython(status) {
  // AMENDMENT_path2 C-2: read on the undotted key, as before #121, so a file named `.py` is not Python.
  for (const p of status.keys()) {
    const low = _undotted(p).toLowerCase();
    if (_PY_SUFFIXES.some(s => low.endsWith(s))) return true;
  }
  return false;
}

// str.strip(chars) / str.rstrip(chars) with an explicit character set, as Python does them.
function _stripChars(s, chars, left = true, right = true) {
  let i = 0, j = s.length;
  if (left) while (i < j && chars.includes(s[i])) i++;
  if (right) while (j > i && chars.includes(s[j - 1])) j--;
  return s.slice(i, j);
}
const _rstrip = (s, chars) => _stripChars(s, chars, false, true);

function _prefixIsPathShaped(prefix, status) {
  // A scope prefix is a path when it looks like one or names a segment of a changed path.
  // Judged on the prefix as written, minus a sentence-final period.
  const raw = _rstrip(_stripChars(prefix, "`\"'"), ".");
  if (!raw) return false;
  if (/[/\\.]/.test(raw)) return true;
  const low = _rstrip(_norm(raw), "/").toLowerCase();
  for (const changed of status.keys()) {
    // PATH-2 (#121): segments read with the key's leading dots dropped, as before the key kept them.
    if (_undotted(changed).split("/").some(seg => seg.toLowerCase() === low)) return true;
  }
  return false;
}

// PATH-2 (#101): a definition the removed lines of the SAME FILE also define is changed, not added.
// AMENDMENT_path2 C-1: paired one to one, per file and per name; a file whose status is `A` pairs
// nothing. The patterns carry no \s, \w or \b (JavaScript's \s matches U+FEFF and its \b is ASCII),
// so this file and styxx/diffgate.py read a BOM strip and a non-ASCII name the same way.
const _DEF_TEST_LINE = /^\uFEFF?[ \t]*def (test_[^ \t(:]*)/;
const _symbolDefLine = name => new RegExp("^\\uFEFF?[ \\t]*(?:async[ \\t]+)?(?:def|class)[ \\t]+" + _reEscape(name) + "(?=[ \\t(:]|$)");

function _changedTestDefs(sides, status) {
  // Per file whose status is not `A`, per test name, min(added lines defining it, removed lines
  // defining it), summed. The caller clamps to `got`.
  let n = 0;
  if (!sides) return 0;
  for (const [path, [added, removed]] of sides) {
    if (status && status.get(path) === "A") continue;
    const gone = new Map();
    for (const line of removed) { const m = _DEF_TEST_LINE.exec(line); if (m) gone.set(m[1], (gone.get(m[1]) || 0) + 1); }
    if (!gone.size) continue;
    const fresh = new Map();
    for (const line of added) { const m = _DEF_TEST_LINE.exec(line); if (m) fresh.set(m[1], (fresh.get(m[1]) || 0) + 1); }
    for (const [name, k] of fresh) n += Math.min(k, gone.get(name) || 0);
  }
  return n;
}

function _definitionOnlyChanged(name, sides, status) {
  // Some file both adds and removes a definition of `name`, and no file adds more definitions of it
  // than it removes (a file whose status is `A` removes none). Counted per file, one to one.
  const rx = _symbolDefLine(name);
  let paired = false;
  for (const [path, [added, removed]] of (sides || new Map())) {
    const a = added.filter(line => rx.test(line)).length;
    const r = (status && status.get(path) === "A") ? 0 : removed.filter(line => rx.test(line)).length;
    if (a > r) return false;
    if (a && r) paired = true;
  }
  return paired;
}

// COMPAT-1: which public top-level definitions the diff removed without re-defining, per language.
const _COMPAT_VERDICTS = ["UNCHECKABLE"];
const _COMPAT_LANGS = [
  // [language, suffixes, regex over ONE removed line with a `name` group]
  ["python", [".py"], /^(?:async\s+)?(?:def|class)\s+(?<name>[A-Za-z]\w*)/],
  ["js/ts", [".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".mts", ".cts"],
    /^export\s+(?:default\s+)?(?:async\s+)?(?:function\*?|class|const|let|var|interface|type|enum)\s+(?<name>[A-Za-z_$]\w*)/],
  ["go", [".go"], /^(?:func\s+(?:\([^)]*\)\s*)?|type\s+)(?<name>[A-Z]\w*)\b/],
  ["rust", [".rs"], /^\s*pub\s+(?:async\s+)?(?:fn|struct|enum|trait|type|const|static)\s+(?<name>[A-Za-z_]\w*)/],
  ["java", [".java", ".kt"], /^\s*public\s+(?:static\s+|final\s+|abstract\s+)*[\w<>\[\],\s]+?\s+(?<name>[a-zA-Z_]\w*)\s*\(/],
];
const _COMPAT_MAX_NAMED = 5;

const _reEscape = s => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

// BIN-1 (issue #118): a `diff --git` header with no `---`/`+++` pair — a binary change, a mode-only
// change, a pure rename — registers its file: A on `new file mode` / `Binary files /dev/null and …`,
// D on `deleted file mode` / `… and /dev/null differ`, else M. Files with hunks read as before.
const _DIFF_GIT = /^diff --git (?:"a\/(?<qa>(?:[^"\\]|\\.)*)"|a\/(?<a>.*?)) (?:"b\/(?<qb>(?:[^"\\]|\\.)*)"|b\/(?<b>.*))$/;
const _BINARY_LINE = /^Binary files (?<a>.+?) and (?<b>.+?) differ$/;

function _headerPaths(line) {
  const body = line.slice("diff --git ".length);
  if (body.length % 2 === 1) {
    const mid = Math.floor(body.length / 2);
    if (body[mid] === " " && body.slice(0, mid).startsWith("a/") && body.slice(mid + 1).startsWith("b/")
        && body.slice(2, mid) === body.slice(mid + 3)) return [body.slice(2, mid), body.slice(mid + 3)];
  }
  const m = _DIFF_GIT.exec(line);
  if (!m) return ["", ""];
  const a = m.groups.qa !== undefined ? m.groups.qa : (m.groups.a || "");
  const b = m.groups.qb !== undefined ? m.groups.qb : (m.groups.b || "");
  return [a, b];
}

class _Pending {
  constructor(line) { [this.a, this.b] = _headerPaths(line); this.status = "M"; }
  note(line) {
    if (line.startsWith("new file mode")) this.status = "A";
    else if (line.startsWith("deleted file mode")) this.status = "D";
    else if (line.startsWith("rename from ")) this.a = line.slice("rename from ".length);
    else if (line.startsWith("rename to ")) this.b = line.slice("rename to ".length);
    else {
      const m = _BINARY_LINE.exec(line);
      if (m) { if (m.groups.a === "/dev/null") this.status = "A"; else if (m.groups.b === "/dev/null") this.status = "D"; }
    }
  }
  path() { const raw = this.status === "D" ? this.a : this.b; return raw ? _norm(raw) : ""; }
}

function parseUnifiedDiffSides(diffText) {
  // Unified diff text -> Map(normalized new-or-old path -> [added_lines, removed_lines]).
  const sides = new Map();
  let oldPath = null;
  let cur = null;
  let pending = null;
  const flush = () => { if (pending !== null && pending.path() && !sides.has(pending.path())) sides.set(pending.path(), [[], []]); };
  for (const line of _splitlines(diffText || "")) {
    if (line.startsWith("diff --git ")) {
      flush();
      pending = new _Pending(line);
      cur = null;
    } else if (line.startsWith("--- ")) {
      oldPath = line.slice(4).trim();
      cur = null;
    } else if (line.startsWith("+++ ")) {
      const nw = line.slice(4).trim();
      let raw;
      if (nw === "/dev/null") raw = (oldPath && oldPath.startsWith("a/")) ? oldPath.slice(2) : (oldPath || "");
      else raw = nw.startsWith("b/") ? nw.slice(2) : nw;
      cur = _norm(raw);
      if (!sides.has(cur)) sides.set(cur, [[], []]);
      pending = null;
    } else if (cur !== null && line.startsWith("+") && !line.startsWith("+++")) {
      sides.get(cur)[0].push(line.slice(1));
    } else if (cur !== null && line.startsWith("-") && !line.startsWith("---")) {
      sides.get(cur)[1].push(line.slice(1));
    } else if (pending !== null) {
      pending.note(line);
    }
  }
  flush();
  return sides;
}

function _compatRemovedPublicNames(sides) {
  const byLangAdded = new Map();
  const langsPresent = [];
  // AMENDMENT_path2 C-2: the suffix tests read the undotted key -- the key before #121; paths print dotted.
  for (const [path, [added]] of sides) {
    for (const [lang, sufs] of _COMPAT_LANGS) {
      if (sufs.some(s => _undotted(path).endsWith(s))) {
        if (!byLangAdded.has(lang)) byLangAdded.set(lang, []);
        byLangAdded.get(lang).push(...added);
        if (!langsPresent.includes(lang)) langsPresent.push(lang);
      }
    }
  }
  const dropped = [];
  const seen = new Set();
  for (const [path, [, removed]] of sides) {
    for (const [lang, sufs, rx] of _COMPAT_LANGS) {
      if (!sufs.some(s => _undotted(path).endsWith(s))) continue;
      const ablob = (byLangAdded.get(lang) || []).join("\n");
      for (const line of removed) {
        const m = rx.exec(line);
        if (!m) continue;
        const name = m.groups.name;
        if (name.startsWith("_")) continue;                              // private by convention
        if (new RegExp("\\b" + _reEscape(name) + "\\b").test(ablob)) continue;  // re-defined or still referenced
        const key = path + " " + lang + " " + name;
        if (!seen.has(key)) { seen.add(key); dropped.push([path, lang, name]); }
      }
    }
  }
  return [dropped, langsPresent];
}

function _compatReading(sides) {
  if (!sides || sides.size === 0) {
    return ["UNCHECKABLE", "compatibility claimed; no per-file diff available to read (behaviour beyond names not checked)", { removed: [] }];
  }
  const [dropped, langs] = _compatRemovedPublicNames(sides);
  if (!langs.length) {
    return ["UNCHECKABLE", "compatibility claimed; no language this reading covers in the diff (python, js/ts, go, rust, java)", { removed: [], languages: [] }];
  }
  if (!dropped.length) {
    return ["UNCHECKABLE", `compatibility claimed; no public top-level definition removed (${langs.join(", ")} read; behaviour beyond names not checked)`, { removed: [], languages: langs }];
  }
  const shown = dropped.slice(0, _COMPAT_MAX_NAMED).map(([p, , n]) => `${p}: ${n}`).join(", ");
  const more = dropped.length > _COMPAT_MAX_NAMED ? ` (+${dropped.length - _COMPAT_MAX_NAMED} more)` : "";
  const why = `compatibility claimed; the diff removes ${dropped.length} public definition(s) not re-defined in the added lines: ${shown}${more}`;
  return ["UNCHECKABLE", why, { removed: dropped.map(([p, l, n]) => ({ path: p, language: l, name: n })), languages: langs }];
}

// The `tests_pass` reason with no --evidence and no --run, verbatim from the Python: the
// styxx.evidence leg reading zero files with no commit, then diffgate's own note. This port has
// neither channel, so this is the only reason it can give and the verdict is always UNCHECKABLE.
const _TESTS_PASS_NO_EVIDENCE_WHY = "styxx.evidence (styxx-evidence/v0.2) read 0 supplied file(s), with no commit supplied, so nothing ties a report to this change — no evidence was supplied. Absence of a report is not a failing report; an unattested commit is unattested.  ||  No test REPORT was handed to the gate. It does not take the agent's word for test results, so with nothing to read it declines — absence of evidence is not a contradiction. The channel that makes this readable is a report passed with --evidence: a JUnit XML, or better a test-result attestation whose subject names the head commit. Even then no signature is checked, and the answer is VERIFIED or UNCHECKABLE — there is no accusing verdict for this claim kind.";

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

// PATH-2 (#121): only a leading run of "/" and "./" segments is removed; a dotfile keeps its dots.
function _norm(p) {
  return p.replace(/\\/g, "/").replace(/^(?:\.?\/)+/, "").toLowerCase();
}
// A key with its leading dots and slashes dropped: the key _norm made before #121 (str.lstrip("./")).
function _undotted(key) {
  return _stripChars(key, "./", true, false);
}
// AMENDMENT_path2 C-3: `path` lies outside every prefix only by a dot the prose left off -- some prefix
// key has no leading dot, the path's leading segment starts with exactly one dot (not `..`), and the
// path without that dot is the prefix or lies under it.
function _dotMiss(path, prefs) {
  if (!path.startsWith(".") || path.startsWith("..")) return false;
  const rest = path.slice(1);
  return prefs.some(x => !x.startsWith(".") && (rest === x || rest.startsWith(x + "/")));
}
// PATH-2 (#97): resolve in tiers over every entry — exact, then suffix, then basename.
function _findPath(status, claimed) {
  const c = _norm(claimed);
  const tiers = [p => p === c, p => p.endsWith("/" + c), p => _basename(p) === _basename(c)];
  for (const tier of tiers) {
    for (const [p, st] of status) if (tier(p)) return [p, st];
  }
  return [null, null];
}
function _basename(p) {
  const s = p.replace(/\/+$/, "");
  return s.slice(s.lastIndexOf("/") + 1);
}
function _splitlines(text) {
  // str.splitlines() for the line endings a diff can carry
  const lines = text.split(/\r\n|\r|\n/);
  if (lines.length && lines[lines.length - 1] === "") lines.pop();
  return lines;
}

// repr() of a str, for the `why` strings the Python builds with !r.
function pyRepr(s) {
  const q = (s.includes("'") && !s.includes('"')) ? '"' : "'";
  let out = q;
  for (const ch of s) {
    const c = ch.codePointAt(0);
    if (ch === "\\") out += "\\\\";
    else if (ch === q) out += "\\" + q;
    else if (ch === "\n") out += "\\n";
    else if (ch === "\r") out += "\\r";
    else if (ch === "\t") out += "\\t";
    else if (c < 0x20 || c === 0x7f) out += "\\x" + c.toString(16).padStart(2, "0");
    else out += ch;
  }
  return out + q;
}
function pyList(arr) { return "[" + arr.map(pyRepr).join(", ") + "]"; }

function parseUnifiedDiff(diffText) {
  const status = new Map();
  const added = [];
  let oldPath = null;
  let pending = null;                       // BIN-1: a header still waiting for its pair
  const flush = () => { if (pending !== null && pending.path() && !status.has(pending.path())) status.set(pending.path(), pending.status); };
  for (const line of _splitlines(diffText || "")) {
    if (line.startsWith("diff --git ")) {
      flush();
      pending = new _Pending(line);
    } else if (line.startsWith("--- ")) {
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
      pending = null;
    } else if (line.startsWith("+") && !line.startsWith("+++")) {
      added.push(line.slice(1));
    } else if (pending !== null) {
      pending.note(line);
    }
  }
  flush();
  return { status, addedBlob: added.join("\n") };
}

function _pySplitSentences(text) {
  // re.split(r"(?<=[.!?])\s+|\n+", text)
  return text.split(/(?<=[.!?])\s+|\n+/);
}

function _pathClaimVerdict(kind, claimed, findPath) {
  const [p, st] = findPath(claimed);
  const want = { file_created: "A", file_deleted: "D" }[kind];
  const accuse = !WITHHOLD_PATH_ACCUSATION;
  const bare = V14_BARE_NAME_ABSTAIN && !claimed.includes("/") && !claimed.includes("\\");
  if (p === null && bare) {
    return ["UNCHECKABLE", `${pyRepr(claimed)} is a bare name absent from the diff — ambiguous between a file and a library, so no accusation is made (V14 repair 2, a deliberate recall sacrifice)`];
  }
  if (p === null) {
    return accuse
      ? ["CONTRADICTED", `${pyRepr(claimed)} does not appear in the diff at all`]
      : ["UNCHECKABLE", `${pyRepr(claimed)} does not appear in the diff — accusation WITHHELD: this class failed EXTERNAL-1 precision (0.23 vs 0.95 floor), disabled pending repair`];
  }
  if (want && st !== want) {
    return accuse
      ? ["CONTRADICTED", `${pyRepr(claimed)} is status ${pyRepr(st)} in the diff, claim wants ${pyRepr(want)}`]
      : ["UNCHECKABLE", `${pyRepr(claimed)} is status ${pyRepr(st)}, claim wants ${pyRepr(want)} — accusation WITHHELD pending the EXTERNAL-1 repair`];
  }
  return ["VERIFIED", `diff status ${pyRepr(st)} for ${pyRepr(p)}`];
}

function gateDiffText(summaryText, diffText, { strict = false } = {}) {
  const { status, addedBlob } = parseUnifiedDiff(diffText);
  const sides = parseUnifiedDiffSides(diffText);
  const rawInputLen = (diffText || "").length;
  let noEvidence = null;
  if (status.size === 0 && !addedBlob) {
    noEvidence = "the diff carries no file statuses and no added lines";
    if (rawInputLen) noEvidence += `; ${rawInputLen} characters of input parsed to nothing, which is a parse failure, not an empty change`;
  }
  const noPaths = status.size === 0 ? "the diff carries no file paths, so scope cannot be checked" : null;

  const findPath = claimed => _findPath(status, claimed);

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
        if (V14_CONTAINMENT_TOUCH && kind === "file_touched" && _demotedByContainment(sent, m)) continue;
        if (BC1_BY_CONSTRUCTION && kind === "symbol_added" && _SYMBOL_WORDS.has(m.groups.name.toLowerCase())) continue;
        covered.add(si);
        const d = {};
        for (const [k, v] of Object.entries(m.groups || {})) if (v !== undefined) d[k] = v;
        const c = { kind, text: sent.trim().slice(0, 160), detail: d, verdict: "UNCHECKABLE", why: "" };
        if (noEvidence) {
          c.verdict = "UNCHECKABLE"; c.why = noEvidence; claims.push(c); continue;
        }
        if (_PATH_KINDS.has(kind)) {
          [c.verdict, c.why] = _pathClaimVerdict(kind, d.path, findPath);
        } else if (kind === "files_changed_count") {
          const n = parseInt(d.n, 10);
          if (noPaths) { c.verdict = "UNCHECKABLE"; c.why = noPaths; }
          else { c.verdict = n === status.size ? "VERIFIED" : "CONTRADICTED"; c.why = `diff changes ${status.size} files, claim says ${n}`; }
        } else if (kind === "tests_added") {
          const n = parseInt(d.n, 10);
          const noun = (d.noun || "").toLowerCase();
          if (BC1_BY_CONSTRUCTION && !_diffTouchesPython(status)) {
            c.verdict = "UNCHECKABLE";
            c.why = "no Python file in the diff; this template counts `def` lines (#110)";
          } else {
            const got = (addedBlob.match(/^\s*def test_/gm) || []).length;
            // PATH-2 (#101): verify net, abstain inside [net, got], accuse only outside it.
            // AMENDMENT_path2 C-1: pairs one to one, clamped to got.
            const chg = Math.min(_changedTestDefs(sides, status), got);
            const net = got - chg;
            const note = chg ? ` (${chg} changed, not added: #101)` : "";
            if (net === n) {
              c.verdict = "VERIFIED"; c.why = `diff adds ${net} test functions, claim says ${n}${note}`;
            } else if (BC1_BY_CONSTRUCTION && _TEST_NOUNS_NOT_FUNCTIONS.has(noun)) {
              c.verdict = "UNCHECKABLE";
              const one = { classes: "class", cases: "case", files: "file", scenarios: "scenario", suites: "suite" }[noun] || noun;
              c.why = `counts test ${noun}, diff adds ${net} test functions; a ${one} is not a function (#110)${note}`;
            } else if (chg && net < n && n <= got) {
              c.verdict = "UNCHECKABLE";
              c.why = `diff adds ${net} test functions and changes ${chg}, claim says ${n}; a changed test is not an added one (#101)`;
            } else {
              c.verdict = "CONTRADICTED"; c.why = `diff adds ${net} test functions, claim says ${n}${note}`;
            }
          }
        } else if (kind === "symbol_added") {
          if (BC1_BY_CONSTRUCTION && !_diffTouchesPython(status)) {
            c.verdict = "UNCHECKABLE";
            c.why = "no Python file in the diff; this template counts `def` lines (#110)";
          } else {
            const src = "^\\s*(?:def|class)\\s+" + _reEscape(d.name) + "\\b";
            const hit = new RegExp(src, "m").test(addedBlob);
            if (hit && _definitionOnlyChanged(d.name, sides, status)) {
              c.verdict = "UNCHECKABLE";                                   // PATH-2 (#101)
              c.why = `added lines define ${d.kind} ${pyRepr(d.name)} only where the removed lines of the same file define it too; a changed definition is not an added one (#101)`;
            } else {
              c.verdict = hit ? "VERIFIED" : "CONTRADICTED";
              c.why = `added lines ${hit ? "do" : "do NOT"} define ${d.kind} ${pyRepr(d.name)}`;
            }
          }
        } else if (kind === "only_touches") {
          let prefs = [_rstrip(_norm(d.prefix), "/.")];      // sentence-final periods are not path
          if (d.prefix2) prefs.push(_rstrip(_norm(d.prefix2), "/."));
          if (d.prefix2 && !_prefixIsPathShaped(d.prefix2, status)) prefs = prefs.slice(0, 1);
          const rawPrefs = [d.prefix].concat(prefs.length === 2 ? [d.prefix2] : []);
          const notPaths = BC1_BY_CONSTRUCTION
            ? rawPrefs.filter(x => !_prefixIsPathShaped(x, status)).map(x => _rstrip(_norm(x), "/."))
            : [];
          const outside = [...status.keys()].filter(p => !prefs.some(x => p.startsWith(x + "/") || p === x));
          // PATH-2 (#121), AMENDMENT_path2 C-3: dot misses alone abstain; any real outside path accuses,
          // and only real paths are listed.
          const dotMiss = outside.filter(p => _dotMiss(p, prefs));
          const real = outside.filter(p => !dotMiss.includes(p));
          if (noPaths) { c.verdict = "UNCHECKABLE"; c.why = noPaths; }
          else if (notPaths.length) { c.verdict = "UNCHECKABLE"; c.why = `prefix ${pyRepr(notPaths[0])} is not a path (#110)`; }
          else if (dotMiss.length && !real.length) {
            c.verdict = "UNCHECKABLE";
            c.why = prefs.length === 1
              ? `paths outside ${pyRepr(prefs[0])} differ from it only by a leading dot: ${pyList(dotMiss.slice(0, 3))} (#121)`
              : `paths outside ${prefs.map(pyRepr).join(" and ")} differ from them only by a leading dot: ${pyList(dotMiss.slice(0, 3))} (#121)`;
          }
          else {
            c.verdict = real.length === 0 ? "VERIFIED" : "CONTRADICTED";
            const shown = prefs.length === 1 ? prefs[0] : prefs.map(pyRepr).join(" and ");
            c.why = real.length === 0 ? "all changed paths under prefix"
              : (prefs.length === 1 ? `paths outside ${pyRepr(shown)}: ${pyList(real.slice(0, 3))}`
                                    : `paths outside ${shown}: ${pyList(real.slice(0, 3))}`);
          }
        } else if (kind === "compat_claim") {
          let extra;
          [c.verdict, c.why, extra] = _compatReading(sides);
          Object.assign(c.detail, extra);
          if (!_COMPAT_VERDICTS.includes(c.verdict)) c.verdict = "UNCHECKABLE";   // unreachable clamp, kept anyway
        } else if (kind === "tests_pass") {
          c.verdict = "UNCHECKABLE"; c.why = _TESTS_PASS_NO_EVIDENCE_WHY;
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

if (typeof module !== "undefined") module.exports = { gateDiffText, parseUnifiedDiff, parseUnifiedDiffSides };
if (typeof globalThis !== "undefined") globalThis.styxxDiffgateJS = { gateDiffText, parseUnifiedDiff, parseUnifiedDiffSides };
