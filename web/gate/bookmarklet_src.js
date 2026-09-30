(function(){
/* diffgate.js — a JavaScript transliteration of styxx/diffgate.py for the browser surfaces that
 * cannot run Python: the paste-in preview page and the bookmarklet. The Python module is the
 * instrument; this file exists so a page can run the same closed template set with no network at
 * all, and it is held to the Python's output by a differential test (differential/ next to this
 * file) rather than by trust.
 *
 * Which Python: the file on the BC-2 + COMPAT-1 + BIN-2 + COMPAT-2 checkout (pull requests #113, #115,
 * #120 and #124 on fathom-lab/styxx, plus the fetch_pr door), sha256 9b620e00a19464589308a987819894ae7cc3c111c66a5f8a457a84b8a6c604eb — the
 * styxx/diffgate.py that 7.48.0 ships once they merge. Relative to the 7.47.0 wheel the port
 * was first cut from, that file carries: the V14 repairs (containment demotes "touched" claims too;
 * a bare basename absent from the diff abstains), the BC-2 repairs for issue #110 (the def-counting
 * templates abstain when the diff has no Python; "added 3 test cases" is not a count of functions;
 * "adds a method to reload" is not a symbol; "only modifies the footer" is not a path), and the
 * COMPAT-1 reading of "no breaking changes" (one verdict, UNCHECKABLE, the public definitions the
 * diff removed named in the reason), the BIN-2 repair for #118 (a `diff --git` header with no
 * `---`/`+++` pair registers its file) and the COMPAT-2 sharpening (surface vs scaffolding paths,
 * signature changes reported, a candidate flag; the licence flag is false and the verdict stays
 * UNCHECKABLE). Two deliberate gaps remain: the structural "unparsed claims"
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
  for (const p of status.keys()) {
    const low = p.toLowerCase();
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

// PATH-1 (PREREG_path1_only_touches_repair_2026_09_17, sha256 618d800f...). Mirrors the Python
// side exactly: two of the six only_touches failure modes from RESULT_bench2_INVALID_2026_09_17
// are repaired, four are not. This list mirrors papers/closed-model-frontier/path1_extensions.txt
// and styxx/diffgate.py PATH1_EXTENSIONS byte for byte; the differential pins all three.
const PATH1_EXTENSIONS = new Set(`
c cc cpp cxx h hh hpp hxx m mm
py pyi pyx rb rs go java kt kts scala clj cljs swift dart
js jsx mjs cjs ts tsx vue svelte
cs fs vb fsx pas pp
php pl pm t r rmd jl lua tcl groovy gradle
sh bash zsh fish ps1 psm1 psd1 bat cmd
html htm xml xsl xslt svg css scss sass less styl
json json5 yaml yml toml ini cfg conf properties env plist
md markdown mdx rst adoc txt text tex bib
sql graphql gql proto thrift avsc
lock sum mod work
dockerfile makefile mk cmake gemspec podspec csproj vbproj fsproj sln props targets
tf tfvars hcl bicep nix
at ac am in out golden snap
png jpg jpeg gif webp ico bmp tiff pdf
zip tar gz tgz bz2 xz 7z jar war whl
`.split(/\s+/).filter(Boolean));

function _hasRealExtension(token) {
  // `package.json` yes, `Assert.NotNull` no. PATH-1 mode 2.
  const i = token.lastIndexOf(".");
  if (i < 0) return false;
  return PATH1_EXTENSIONS.has(token.slice(i + 1).trim().toLowerCase());
}

function _isBareFilename(pref) {
  // PATH-1 mode 1: a prefix naming a file rather than a location.
  return !pref.includes("/") && _hasRealExtension(pref);
}

function _pathInside(path, pref) {
  if (_isBareFilename(pref)) return path === pref || path.endsWith("/" + pref);
  return path === pref || path.startsWith(pref + "/");
}

function _prefixIsPathShaped(prefix, status) {
  // A scope prefix is a path when it looks like one or names a segment of a changed path.
  // Judged on the prefix as written, minus a sentence-final period.
  const raw = _rstrip(_stripChars(prefix, "`\"'"), ".");
  if (!raw) return false;
  if (/[/\\]/.test(raw)) return true;
  // PATH-1 mode 2: a dot alone is no longer enough -- the suffix must be a real file extension.
  if (raw.includes(".") && _hasRealExtension(_rstrip(raw, "/"))) return true;
  const low = _rstrip(_norm(raw), "/").toLowerCase();
  for (const changed of status.keys()) {
    if (changed.split("/").some(seg => seg.toLowerCase() === low)) return true;
  }
  return false;
}

// COMPAT-1: which public top-level definitions the diff removed without re-defining, per language.
// COMPAT-2 (PREREG_compat2_surface_and_panel_2026_09_16): the surface split, the signature reading and
// the candidate flag, mirrored from the Python; the licence flag is false and the verdict stays UNCHECKABLE.
const COMPAT2_LICENSED = false;
const _COMPAT_VERDICTS = COMPAT2_LICENSED ? ["UNCHECKABLE", "CONTRADICTED"] : ["UNCHECKABLE"];
const _COMPAT_SCAFFOLD = /(?:^|\/)(?:tests?|testing|specs?|__tests__|examples?|samples?|demos?|docs?|scripts?|tools?|bench|benchmarks?|fixtures?|internal|_internal|private|vendor|third_party|migrations?|cmd|e2e|integration|mocks?|stories|storybook|playground|sandbox|experiments?|dev|build)\/|(?:^|\/)(?:test_[^/]*|[^/]*_test\.(?:go|py)|[^/]*\.(?:test|spec)\.[^/]+|conftest\.py|setup\.py)$/;
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

function _compatParams(line, at) {
  // The parameter list of a definition line: the text inside the first `(` at or after `at`,
  // whitespace-collapsed, or null when there is none. A list that does not close on the line is
  // taken as far as the line goes, with `…` appended.
  const k = line.indexOf("(", at);
  if (k < 0) return null;
  let depth = 0;
  for (let e = k; e < line.length; e++) {
    if (line[e] === "(") depth += 1;
    else if (line[e] === ")") {
      depth -= 1;
      if (depth === 0) return line.slice(k + 1, e).replace(/\s+/g, " ").trim();
    }
  }
  return line.slice(k + 1).replace(/\s+/g, " ").trim() + "…";
}

function _nameEnd(m) {
  // Python's m.end("name"): every language regex ends at the name except Java's, which goes on to `(`.
  return m.index + m[0].lastIndexOf(m.groups.name) + m.groups.name.length;
}

function _compatRemovedPublicNames(sides) {
  const byLangAdded = new Map();
  const langsPresent = [];
  for (const [path, [added]] of sides) {
    for (const [lang, sufs] of _COMPAT_LANGS) {
      if (sufs.some(s => path.endsWith(s))) {
        if (!byLangAdded.has(lang)) byLangAdded.set(lang, []);
        byLangAdded.get(lang).push(...added);
        if (!langsPresent.includes(lang)) langsPresent.push(lang);
      }
    }
  }
  const dropped = [];
  const changed = [];
  const seen = new Set();
  const seenChanged = new Set();
  for (const [path, [, removed]] of sides) {
    for (const [lang, sufs, rx] of _COMPAT_LANGS) {
      if (!sufs.some(s => path.endsWith(s))) continue;
      const addedLines = byLangAdded.get(lang) || [];
      const ablob = addedLines.join("\n");
      for (const line of removed) {
        const m = rx.exec(line);
        if (!m) continue;
        const name = m.groups.name;
        if (name.startsWith("_")) continue;                              // private by convention
        const key = path + " " + lang + " " + name;
        if (new RegExp("\\b" + _reEscape(name) + "\\b").test(ablob)) {
          // re-defined or still referenced: a change or a move. COMPAT-2 compares the parameter lists.
          const before = _compatParams(line, _nameEnd(m));
          if (before !== null) {
            const afters = [];
            for (const al of addedLines) {
              const am = rx.exec(al);
              if (am && am.groups.name === name) {
                const ap = _compatParams(al, _nameEnd(am));
                if (ap !== null) afters.push(ap);
              }
            }
            if (afters.length && !afters.includes(before) && !seenChanged.has(key)) {
              seenChanged.add(key);
              changed.push([path, lang, name, before, afters[0]]);
            }
          }
          continue;
        }
        if (!seen.has(key)) { seen.add(key); dropped.push([path, lang, name, !_COMPAT_SCAFFOLD.test(path)]); }
      }
    }
  }
  return [dropped, changed, langsPresent];
}

function _compatReading(sides) {
  const empty = () => ({ removed: [], languages: [], surface_removed: 0, signature_changed: [], compat2_candidate: false });
  if (!sides || sides.size === 0) {
    return ["UNCHECKABLE", "compatibility claimed; no per-file diff available to read (behaviour beyond names not checked)", empty()];
  }
  const [dropped, changed, langs] = _compatRemovedPublicNames(sides);
  if (!langs.length) {
    return ["UNCHECKABLE", "compatibility claimed; no language this reading covers in the diff (python, js/ts, go, rust, java)", empty()];
  }
  const sig = changed.length ? `; ${changed.length} signature(s) changed` : "";
  const detail = {
    removed: dropped.map(([p, l, n, sf]) => ({ path: p, language: l, name: n, surface: sf })),
    languages: langs,
    surface_removed: dropped.filter(d => d[3]).length,
    signature_changed: changed.map(([p, l, n, b, a]) => ({ path: p, language: l, name: n, before: b, after: a })),
    compat2_candidate: dropped.some(d => d[3]),
  };
  if (!dropped.length) {
    return ["UNCHECKABLE", `compatibility claimed; no public top-level definition removed (${langs.join(", ")} read; behaviour beyond names not checked)${sig}`, detail];
  }
  const surface = dropped.filter(d => d[3]);
  const scaffold = dropped.filter(d => !d[3]);
  const named = surface.length ? surface : scaffold;
  const shown = named.slice(0, _COMPAT_MAX_NAMED).map(([p, , n]) => `${p}: ${n}`).join(", ");
  const more = named.length > _COMPAT_MAX_NAMED ? ` (+${named.length - _COMPAT_MAX_NAMED} more)` : "";
  if (surface.length) {
    const rest = scaffold.length ? `; ${scaffold.length} more in test/example/internal code` : "";
    const why = `compatibility claimed; the diff removes ${surface.length} public definition(s) from the surface, not re-defined in the added lines: ${shown}${more}${rest}${sig}`;
    return [COMPAT2_LICENSED ? "CONTRADICTED" : "UNCHECKABLE", why, detail];
  }
  const why = `compatibility claimed; ${scaffold.length} public definition(s) removed, all in test/example/internal code: ${shown}${more}${sig}`;
  return ["UNCHECKABLE", why, detail];
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

// DECLARE-1 (PREREG_declare1_the_toll_2026_09_18, sha256 7ffd0ba1...). Mirrors styxx/declare.py
// exactly. A body may declare its claims in one fenced `styxx` block; each declaration is
// normalised into the canonical sentence this same reader already understands and read by this
// same function one level down, so a declared claim and a prose claim cannot drift apart.
const DECLARE_BLOCK_RE = /^[ \t]*```[ \t]*styxx[ \t]*\r?\n([\s\S]*?)^[ \t]*```/gm;
const DECLARE_LINE_RE = /^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*?)\s*$/;
const DECLARABLE = {
  files_changed: "files_changed_count", only_touches: "only_touches",
  adds_symbol: "symbol_added", tests_added: "tests_added",
  file_touched: "file_touched", file_created: "file_created",
  file_deleted: "file_deleted", tests_pass: "tests_pass",
};
const _DEC_INT = /^\d{1,9}$/, _DEC_PATHY = /^[\w.\-/\\]+$/, _DEC_IDENT = /^[A-Za-z_]\w*$/;

// Python's repr() for a simple string, because the reason strings are compared byte for byte by
// the differential and JSON.stringify quotes differently. Python prefers single quotes and
// switches to double only when the value itself contains a single quote and no double.
function _pyRepr(v) {
  const t = String(v);
  if (t.includes("'") && !t.includes('"')) return '"' + t + '"';
  return "'" + t.replace(/\\/g, "\\\\").replace(/'/g, "\\'") + "'";
}

function parseDeclaration(text) {
  DECLARE_BLOCK_RE.lastIndex = 0;
  const blocks = [...String(text || "").matchAll(DECLARE_BLOCK_RE)].map(m => m[1]);
  if (!blocks.length) return [null, []];
  if (blocks.length > 1) return [null, [`${blocks.length} styxx blocks; a body declares once or not at all`]];
  const out = {}, problems = [];
  for (const raw of blocks[0].split(/\r?\n/)) {
    if (!raw.trim() || raw.trimStart().startsWith("#")) continue;
    const m = DECLARE_LINE_RE.exec(raw);
    if (!m) { problems.push(`MALFORMED line, not \`key: value\`: ${_pyRepr(raw.trim().slice(0, 60))}`); continue; }
    const key = m[1].toLowerCase(); const value = _stripChars(m[2].trim(), "`\"'");
    if (!(key in DECLARABLE)) { problems.push(`unknown key '${key}'; reported, never checked`); continue; }
    if (key in out) { problems.push(`duplicate key '${key}'; the first is kept`); continue; }
    out[key] = value;
  }
  return [out, problems];
}

function canonicalSentence(key, value) {
  if (key === "tests_pass") return [null, "declared, and deliberately not verifiable: a declaration that tests passed is not evidence that they did"];
  if (key === "files_changed") return _DEC_INT.test(value) ? [`${parseInt(value, 10)} files changed.`, null] : [null, `MALFORMED: ${_pyRepr(value)} is not a count`];
  if (key === "tests_added") return _DEC_INT.test(value) ? [`Added ${parseInt(value, 10)} tests.`, null] : [null, `MALFORMED: ${_pyRepr(value)} is not a count`];
  if (key === "adds_symbol") return _DEC_IDENT.test(value) ? [`Adds function ${value}.`, null] : [null, `MALFORMED: ${_pyRepr(value)} is not an identifier`];
  const v = value.replace(/\/\*{1,2}$/, "");
  if (!v || !_DEC_PATHY.test(v)) return [null, `MALFORMED: ${_pyRepr(value)} is not a path`];
  if (key === "only_touches") return [`Only touches ${v}.`, null];
  if (key === "file_touched") return [`Modified ${v}.`, null];
  if (key === "file_created") return [`Created file ${v}.`, null];
  if (key === "file_deleted") return [`Deleted ${v}.`, null];
  return [null, `no canonical form for '${key}'`];
}

function declarationPass(summaryText) {
  const [mapping, problems] = parseDeclaration(summaryText);
  const report = { declared: false, keys: [], problems: problems.slice(), unverifiable: [] };
  if (mapping === null) return ["", report];
  report.declared = true;
  report.keys = Object.keys(mapping).sort();
  const sentences = [];
  for (const key of report.keys) {
    const [sent, why] = canonicalSentence(key, mapping[key]);
    if (sent === null) report.unverifiable.push({ key, value: mapping[key], why });
    else sentences.push(sent);
  }
  return [sentences.join("\n"), report];
}

function _gateDiffTextMain(summaryText, diffText, { strict = false, _declared = false } = {}) {
  const { status, addedBlob } = parseUnifiedDiff(diffText);
  const sides = parseUnifiedDiffSides(diffText);
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
            if (got === n) {
              c.verdict = "VERIFIED"; c.why = `diff adds ${got} test functions, claim says ${n}`;
            } else if (BC1_BY_CONSTRUCTION && _TEST_NOUNS_NOT_FUNCTIONS.has(noun)) {
              c.verdict = "UNCHECKABLE";
              const one = { classes: "class", cases: "case", files: "file", scenarios: "scenario", suites: "suite" }[noun] || noun;
              c.why = `counts test ${noun}, diff adds ${got} test functions; a ${one} is not a function (#110)`;
            } else {
              c.verdict = "CONTRADICTED"; c.why = `diff adds ${got} test functions, claim says ${n}`;
            }
          }
        } else if (kind === "symbol_added") {
          if (BC1_BY_CONSTRUCTION && !_diffTouchesPython(status)) {
            c.verdict = "UNCHECKABLE";
            c.why = "no Python file in the diff; this template counts `def` lines (#110)";
          } else {
            const pat = new RegExp("^\\s*(?:def|class)\\s+" + _reEscape(d.name) + "\\b", "m");
            const hit = pat.test(addedBlob);
            c.verdict = hit ? "VERIFIED" : "CONTRADICTED";
            c.why = `added lines ${hit ? "do" : "do NOT"} define ${d.kind} ${pyRepr(d.name)}`;
          }
        } else if (kind === "only_touches") {
          let prefs = [_rstrip(_norm(d.prefix), "/.")];      // sentence-final periods are not path
          if (d.prefix2) prefs.push(_rstrip(_norm(d.prefix2), "/."));
          if (d.prefix2 && !_prefixIsPathShaped(d.prefix2, status)) prefs = prefs.slice(0, 1);
          const rawPrefs = [d.prefix].concat(prefs.length === 2 ? [d.prefix2] : []);
          const notPaths = BC1_BY_CONSTRUCTION
            ? rawPrefs.filter(x => !_prefixIsPathShaped(x, status)).map(x => _rstrip(_norm(x), "/."))
            : [];
          // PATH-1 mode 1: _pathInside matches a bare filename on its basename.
          const outside = [...status.keys()].filter(p => !prefs.some(x => _pathInside(p, x)));
          if (noPaths) { c.verdict = "UNCHECKABLE"; c.why = noPaths; }
          else if (notPaths.length) { c.verdict = "UNCHECKABLE"; c.why = `prefix ${pyRepr(notPaths[0])} is not a path (#110)`; }
          else {
            c.verdict = outside.length === 0 ? "VERIFIED" : "CONTRADICTED";
            const shown = prefs.length === 1 ? prefs[0] : prefs.map(pyRepr).join(" and ");
            c.why = outside.length === 0 ? "all changed paths under prefix"
              : (prefs.length === 1 ? `paths outside ${pyRepr(shown)}: ${pyList(outside.slice(0, 3))}`
                                    : `paths outside ${shown}: ${pyList(outside.slice(0, 3))}`);
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
  // DECLARE-1: the prose pass above is finished and is not changed by any of this. The recursion
  // terminates in one step: synthesized text never contains a styxx fence.
  if (!_declared) {
    const [dtext, drep] = declarationPass(summaryText);
    if (drep.declared) {
      if (dtext) {
        const sub = _gateDiffTextMain(dtext, diffText, { strict, _declared: true });
        for (const c of (sub.claims || [])) {
          c.detail = Object.assign({}, c.detail || {}, { declared: true });
          claims.push(c);
        }
      }
      for (const u of drep.unverifiable) {
        claims.push({ kind: u.key, text: `${u.key}: ${u.value}`, detail: { declared: true },
                      verdict: "UNCHECKABLE", why: u.why });
      }
      for (const p of drep.problems) {
        claims.push({ kind: "declaration_problem", text: p, detail: { declared: true },
                      verdict: "UNCHECKABLE", why: p });
      }
    }
  }

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

// === PATH-2a abstain-only overlay: BEGIN ===
//
// NOTE_path2a_abstain_overlay_2026_09_30. The port's half of the PATH-2a block in styxx/diffgate.py (sha256
// 186d5f2cd791223e3612e6c890505508fdc30dd830486f91bbd5393ca26f1a78, LF). Everything outside this block is main's port at 1cde8b82
// (sha256 06688702..., LF), unchanged except that main's gateDiffText is named _gateDiffTextMain (its definition
// and its DECLARE-1 self-call); the gateDiffText at the end of this block calls it and then the overlay, once.
// The overlay reads each DECIDED claim once more and turns it UNCHECKABLE, with a reason naming the verdict it
// withholds, the defect and main's reason verbatim, only where #97, #121 or #101 can have made it wrong. Every
// function mirrors the Python block line for line; every comparison is structural (ASCII case, code points, '/',
// '.', fixed character sets), so the two ports decide alike wherever main's two ports read the claim alike.

const P2A_DIRECTORY_BASENAME_ABSTAINS = true;

const _P2A_PY_BREAKS = "\n\r\u000b\u000c\u001c\u001d\u001e\u0085\u2028\u2029";
const _P2A_JS_SPACE = "\t\n\u000b\u000c\r \u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff";
const _P2A_DIVERGENT = "\u000b\u000c\u001c\u001d\u001e\u001f\u0085\u2028\u2029\ufeff";
const _P2A_HEADERS = ["diff --git ", "--- ", "+++ ", "rename from ", "rename to ", "new file mode", "deleted file mode", "Binary files "];
const _P2A_FINE = new RegExp("\r\n|[" + _P2A_PY_BREAKS + "]");
const _P2A_COARSE = new RegExp("\r\n|\r|\n");
const _P2A_COUNT_WHY = new RegExp("^diff changes ([0-9]+) files, claim says ([0-9]+)$");
const _P2A_TESTS_WHY = new RegExp("^diff adds ([0-9]+) test functions, claim says ([0-9]+)$");
const _P2A_REACH_PAIRS = [["file_created", "VERIFIED"], ["file_deleted", "VERIFIED"], ["file_touched", "VERIFIED"], ["files_changed_count", "VERIFIED"], ["files_changed_count", "CONTRADICTED"], ["only_touches", "VERIFIED"], ["only_touches", "CONTRADICTED"], ["tests_added", "VERIFIED"], ["tests_added", "CONTRADICTED"], ["symbol_added", "VERIFIED"]];
const _P2A_REACH = new Set(_P2A_REACH_PAIRS.map(p => p[0] + "|" + p[1]));
const _P2A_KIND_DEFECT = {"file_created": "#97, #121", "file_deleted": "#97, #121", "file_touched": "#97, #121", "files_changed_count": "#121", "only_touches": "#121", "tests_added": "#101", "symbol_added": "#101"};
const _P2A_PHRASES = {
  "dir": "the claim names a directory, and only a file of the same name elsewhere matches it",
  "tier": "a changed path that matches the claim more closely than the one main resolved it to reads otherwise",
  "dot": "with leading dots kept, the changed path the claim resolves to reads otherwise",
  "dot_tier": "with leading dots kept and the closest match taken, the claim reads otherwise",
  "count": "two changed paths differ only by a leading dot, which the path key drops, and counted apart the claim reads otherwise",
  "only": "with leading dots kept, whether every changed path lies under the prefix reads otherwise",
  "tests": "a test the added lines count is also defined in the removed lines, and a changed test is not an added one",
  "symbol": "the removed lines define this name too, and a changed definition is not an added one",
  "divergent": "a file header of this diff holds a character that the Python and JavaScript readers split or strip differently",
  "odd": "a path here has a drive-like prefix or a final '.' segment, where base names are read differently",
  "case": "a path here compares only where case outside ASCII is folded, which this overlay does not do",
  "unreproduced": "this overlay does not reproduce main's reading of the diff",
  "unparsed": "main's reason does not have the form this overlay reads",
  "error": "this overlay failed while reading the diff"
};

function _p2aLines(text, rx) {
  const out = text.split(rx);
  if (out.length && out[out.length - 1] === "") out.pop();
  return out;
}

function _p2aRegsRaw(diffText) {
  // (path as written, status, via) wherever parseUnifiedDiff hands a path to _norm, in its order: its loop line
  // for line, `status.set(_norm(x), st)` read as a push; via "+" assigns, "p" (a BIN-1 flush) registers only a key
  // not yet held. A flush is kept even where its key is already registered.
  const regs = [];
  let oldPath = null;
  let pending = null;
  const flush = () => {
    if (pending !== null) {
      const raw = pending.status === "D" ? pending.a : pending.b;
      if (raw) regs.push([raw, pending.status, "p"]);
    }
  };
  for (const line of _splitlines(diffText)) {
    if (line.startsWith("diff --git ")) {
      flush();
      pending = new _Pending(line);
    } else if (line.startsWith("--- ")) {
      oldPath = _stripChars(line.slice(4), _P2A_JS_SPACE);
    } else if (line.startsWith("+++ ")) {
      const nw = _stripChars(line.slice(4), _P2A_JS_SPACE);
      if (nw === "/dev/null") regs.push([oldPath.startsWith("a/") ? oldPath.slice(2) : oldPath, "D", "+"]);
      else if (oldPath === "/dev/null" || oldPath === null) regs.push([nw.startsWith("b/") ? nw.slice(2) : nw, "A", "+"]);
      else regs.push([nw.startsWith("b/") ? nw.slice(2) : nw, "M", "+"]);
      pending = null;
    } else if (line.startsWith("+") && !line.startsWith("+++")) {
      // an added line: main keeps it, the registration loop does not
    } else if (pending !== null) {
      pending.note(line);
    }
  }
  flush();
  return regs;
}

function _p2aViews(diffText) {
  // [[added, removed]] under CPython's line breaks and under the port's; each port's main reads one.
  return [_P2A_FINE, _P2A_COARSE].map(rx => {
    const ls = _p2aLines(diffText, rx);
    return [ls.filter(x => x.startsWith("+") && !x.startsWith("+++")).map(x => x.slice(1)),
            ls.filter(x => x.startsWith("-") && !x.startsWith("---")).map(x => x.slice(1))];
  });
}

function _p2aDivergent(diffText) {
  for (const line of _p2aLines(diffText, _P2A_COARSE)) {
    let hit = false;
    for (const ch of line) if (_P2A_DIVERGENT.includes(ch)) { hit = true; break; }
    if (!hit) continue;
    for (const piece of [line].concat(_p2aLines(line, _P2A_FINE))) {
      if (_P2A_HEADERS.some(h => piece.startsWith(h))) return true;
    }
  }
  return false;
}

const _p2aBs = s => s.split("\\").join("/");

function _p2aStrip(s) {                  // main's key before case
  const t = _p2aBs(s);
  let i = 0;
  while (i < t.length && (t[i] === "." || t[i] === "/")) i++;
  return t.slice(i);
}

function _p2aDotted(s) {                 // PREREG_path2 R-121's key before case: ^(?:\.?/)+ removed
  let t = _p2aBs(s);
  for (;;) {
    if (t.startsWith("./")) t = t.slice(2);
    else if (t.startsWith("/")) t = t.slice(1);
    else return t;
  }
}

function _p2aRun(s) {                    // the leading dots and slashes main drops and R-121 keeps
  const d = _p2aDotted(s);
  return d.slice(0, d.length - _p2aStrip(s).length);
}

function _p2aFold(s) {                   // ASCII case, and the two code points whose lowercase holds ASCII
  let out = "";
  for (const ch of s) {
    const o = ch.codePointAt(0);
    out += (o >= 65 && o <= 90) ? String.fromCharCode(o + 32) : o === 0x212a ? "k" : o === 0x130 ? "i\u0307" : ch;
  }
  return out;
}

function _p2aWild(s) {                   // every code point outside ASCII read as one placeholder
  let out = "";
  for (const ch of _p2aFold(s)) out += ch.codePointAt(0) < 128 ? ch : "\ufffd";
  return out;
}

const _p2aA = s => _p2aFold(_p2aStrip(s));
const _p2aK = s => _p2aFold(_p2aDotted(s));
const _p2aWA = s => _p2aWild(_p2aStrip(s));
const _p2aWK = s => _p2aWild(_p2aDotted(s));

function _p2aBase(p) {                   // main's _basename
  const q = _rstrip(p, "/");
  return q.slice(q.lastIndexOf("/") + 1);
}

function _p2aOdd(p) {                    // a drive-like second code point, or a final "." segment
  const q = _rstrip(p, "/");
  const cps = Array.from(q.slice(0, 3));
  return (cps.length >= 2 && cps[1] === ":") || q === "." || q.endsWith("/.");
}

function _p2aTier(p, c) {
  if (p === c) return 0;
  if (p.endsWith("/" + c)) return 1;
  if (_p2aBase(p) === _p2aBase(c)) return 2;
  return null;
}

function _p2aBuild(regs, key) {
  // main's status map over [raw, st, via], keyed by `key`: "+" assigns; "p" registers a non-empty key once.
  const m = new Map();
  for (const [raw, st, via] of regs) {
    const k = key(raw);
    if (via === "+") m.set(k, st);
    else if (k && !m.has(k)) m.set(k, st);
  }
  return m;
}

function _p2aResolve(m, c, tiered) {
  // [key, status] main's findPath returns (tiered false), or V97's: exact, then suffix, then base name.
  for (const t of (tiered ? [0, 1, 2] : [null])) {
    if (t === 2 && c.includes("/") && P2A_DIRECTORY_BASENAME_ABSTAINS) return null;
    for (const [p, st] of m) {
      const u = _p2aTier(p, c);
      if (u !== null && (t === null || u === t)) return [p, st];
    }
  }
  return null;
}

function _p2aFactsRaw(diffText) {
  // What the overlay reads, from the door's own bytes, computed when a claim needs it.
  const memo = new Map();
  const get = (k, make) => { if (!memo.has(k)) memo.set(k, make()); return memo.get(k); };
  const f = {
    regs: () => get("regs", () => _p2aRegsRaw(diffText)),
    views: () => get("views", () => _p2aViews(diffText)),
    divergent: () => get("div", () => _p2aDivergent(diffText)),
    status: space => get(space, () => _p2aBuild(f.regs(), space === "A" ? _p2aA : _p2aK)),
  };
  return f;
}

function _p2aCaseDoubt(regs, claimed, want, key, wkey) {
  // U1: a path matches the claim at another tier once case outside ASCII is a placeholder. U2 (a status claim):
  // a path the claim may match shares that placeholder key with another key and a status other than the one
  // claimed, so a runtime's lowercase could merge them into a key that reads otherwise.
  const cw = wkey(claimed), cf = key(claimed);
  const keys = new Map(), sts = new Map();
  for (const [raw, st] of regs) {
    const w = wkey(raw);
    if (!keys.has(w)) { keys.set(w, new Set()); sts.set(w, new Set()); }
    keys.get(w).add(key(raw));
    sts.get(w).add(st);
  }
  for (const [raw] of regs) {
    const w = wkey(raw);
    const tw = _p2aTier(w, cw);
    if (tw !== _p2aTier(key(raw), cf)) return true;
    const s = sts.get(w);
    if (want !== null && tw !== null && keys.get(w).size > 1 && !(s.size === 1 && s.has(want))) return true;
  }
  return false;
}

function _p2aPath(c, f) {
  const claimed = c.detail.path;
  const want = c.kind === "file_created" ? "A" : c.kind === "file_deleted" ? "D" : null;
  if (f.divergent()) return ["divergent", "#97, #121"];
  const regs = f.regs();
  if ([claimed].concat(regs.map(r => r[0])).some(x => _p2aOdd(_p2aA(x)) || _p2aOdd(_p2aK(x)))) return ["odd", "#97"];
  if (_p2aCaseDoubt(regs, claimed, want, _p2aA, _p2aWA) || _p2aCaseDoubt(regs, claimed, want, _p2aK, _p2aWK)) {
    return ["case", "#97, #121"];
  }
  const ca = _p2aA(claimed), ck = _p2aK(claimed);
  const ok = r => r !== null && (want === null || r[1] === want);
  if (!ok(_p2aResolve(f.status("A"), ca, false))) return ["unreproduced", "#97"];
  const r97 = _p2aResolve(f.status("A"), ca, true);
  const v97 = ok(r97);
  const v121 = ok(_p2aResolve(f.status("K"), ck, false));
  const vboth = ok(_p2aResolve(f.status("K"), ck, true));
  if (v97 && v121 && vboth) return null;
  if (v121 && !v97) return [r97 === null ? "dir" : "tier", "#97"];
  if (v97 && !v121) return ["dot", "#121"];
  return ["dot_tier", "#97, #121"];
}

function _p2aCount(c, f) {
  if (f.divergent()) return ["divergent", "#121"];
  const m = _P2A_COUNT_WHY.exec(c.why);
  if (!m) return ["unparsed", "#121"];
  const g = parseInt(m[1], 10), n = parseInt(m[2], 10);
  const regs = f.regs();
  const ra = regs.filter(r => r[2] === "+" || _p2aStrip(r[0]));
  const rk = regs.filter(r => r[2] === "+" || _p2aDotted(r[0]));
  const runs = new Map();
  let twin = rk.length !== ra.length;     // a dotted-only name ("." , "..") V121 would register
  for (const [raw] of ra) {
    const w = _p2aWA(raw), run = _p2aRun(raw);
    if (!runs.has(w)) runs.set(w, run);
    if (runs.get(w) !== run) twin = true;
  }
  if (!twin) return null;
  const count = (rs, form) => new Set(rs.map(r => form(r[0]))).size;
  if (!(count(ra, _p2aWA) <= g && g <= count(ra, _p2aA))) return ["unreproduced", "#121"];
  const lo = count(rk, _p2aWK), hi = count(rk, _p2aK);
  if (c.verdict === "VERIFIED") return (lo === hi && hi === n) ? null : ["count", "#121"];
  return (lo <= n && n <= hi) ? ["count", "#121"] : null;
}

function _p2aExt(token) {                // _hasRealExtension, ASCII case
  return token.includes(".") && PATH1_EXTENSIONS.has(_p2aFold(token.slice(token.lastIndexOf(".") + 1)));
}

function _p2aInside(p, pref) {           // _pathInside
  if (!pref.includes("/") && _p2aExt(pref)) return p === pref || p.endsWith("/" + pref);
  return p === pref || p.startsWith(pref + "/");
}

function _p2aShaped(prefix, keys, form) {   // _prefixIsPathShaped
  const raw = _rstrip(_stripChars(prefix, "`\"'"), ".");
  if (!raw) return false;
  if (raw.includes("/") || raw.includes("\\")) return true;
  if (raw.includes(".") && _p2aExt(_rstrip(raw, "/"))) return true;
  const low = _rstrip(form(raw), "/");
  return keys.some(k => k.split("/").includes(low));
}

const _P2A_SPACES = [["A", _p2aA, _p2aWA, _p2aStrip], ["K", _p2aK, _p2aWK, _p2aDotted]];

function _p2aOnly(c, f) {
  if (f.divergent()) return ["divergent", "#121"];
  const d = c.detail;
  const prefixes = [d.prefix].concat(d.prefix2 ? [d.prefix2] : []);
  const two = prefixes.length === 2;
  const sets = two ? [prefixes.slice(0, 1), prefixes] : [prefixes];
  const regs = f.regs();
  const got = new Map();                  // space|set index|wild -> every changed path under the set
  const shaped = new Map();               // space|wild -> prefix2 reads as a path
  for (const [sp, form, wform, preKey] of _P2A_SPACES) {
    const paths = regs.filter(r => r[2] === "+" || preKey(r[0])).map(r => r[0]);
    for (const [wild, fm] of [[false, form], [true, wform]]) {
      sets.forEach((pre, i) => {
        const ps = pre.map(x => _rstrip(fm(x), "/."));
        got.set(sp + "|" + i + "|" + wild, paths.every(p => ps.some(x => _p2aInside(fm(p), x))));
      });
      shaped.set(sp + "|" + wild, two && _p2aShaped(prefixes[1], paths.map(p => fm(p)), fm));
    }
  }
  const G = (sp, i, wild) => got.get(sp + "|" + i + "|" + wild);
  for (const sp of ["A", "K"]) {
    for (let i = 0; i < sets.length; i++) if (G(sp, i, false) !== G(sp, i, true)) return ["case", "#121"];
    if (shaped.get(sp + "|false") !== shaped.get(sp + "|true")) return ["case", "#121"];
  }
  const usedA = shaped.get("A|false") ? 1 : 0;
  if ((G("A", usedA, false) ? "VERIFIED" : "CONTRADICTED") !== c.verdict) return ["unreproduced", "#121"];
  for (let i = 0; i < sets.length; i++) if (G("A", i, false) !== G("K", i, false)) return ["only", "#121"];
  const usedK = shaped.get("K|false") ? 1 : 0;
  if (G("K", usedK, false) !== G("A", usedA, false)) return ["only", "#121"];
  return null;
}

// #101. Lines are read as arrays of code points, as the Python block reads a str, so a name, a separator and a
// position mean the same thing in both ports.
function _p2aCoarse(ch) {                // a superset of every runtime's whitespace class: controls, space, DEL, non-ASCII
  const o = ch.codePointAt(0);
  return o <= 0x20 || o === 0x7f || o >= 0x80;
}

function _p2aNameChar(ch) {
  const o = ch.codePointAt(0);
  return o === 95 || (o >= 48 && o <= 57) || (o >= 65 && o <= 90) || (o >= 97 && o <= 122) || o >= 128;
}

function _p2aFind(cs, word, from) {      // str.find over code points, for an ASCII word
  for (let i = Math.max(from, 0); i + word.length <= cs.length; i++) {
    let j = 0;
    while (j < word.length && cs[i + j] === word[j]) j++;
    if (j === word.length) return i;
  }
  return -1;
}

function _p2aNamesAt(cs, k) {
  // The name run from k (ASCII word characters and code points outside ASCII), and its leading ASCII run, where a
  // runtime's identifier table ends the name early.
  let e = k;
  while (e < cs.length && _p2aNameChar(cs[e])) e++;
  let a = k;
  while (a < e && cs[a].codePointAt(0) < 128) a++;
  const out = new Set([cs.slice(k, e).join(""), cs.slice(k, a).join("")]);
  out.delete("");
  return out;
}

function _p2aDefNames(line, words) {
  // Every name a `def` (or `class`) in this line may define: after the word, one or more coarse characters, then a
  // name starting after any of them.
  const cs = Array.from(line);
  const out = new Set();
  for (const w of words) {
    let i = _p2aFind(cs, w, 0);
    while (i >= 0) {
      let k = i + w.length;
      while (k < cs.length && _p2aCoarse(cs[k])) {
        k++;
        for (const x of _p2aNamesAt(cs, k)) out.add(x);
      }
      i = _p2aFind(cs, w, i + 1);
    }
  }
  return out;
}

function _p2aCounted(line) {
  // The names at the `def test_` sites of one added line that main's count may count: preceded, from the line
  // start or the last U+2028 / U+2029, by coarse characters only.
  const cs = Array.from(line);
  const out = [];
  let i = _p2aFind(cs, "def test_", 0);
  while (i >= 0) {
    let s = i;
    while (s > 0 && cs[s - 1] !== "\u2028" && cs[s - 1] !== "\u2029") s--;
    if (cs.slice(s, i).every(_p2aCoarse)) out.push(_p2aNamesAt(cs, i + 4));
    i = _p2aFind(cs, "def test_", i + 1);
  }
  return out;
}

function _p2aTests(c, f) {
  const m = _P2A_TESTS_WHY.exec(c.why);
  if (!m) return ["unparsed", "#101"];
  const got = parseInt(m[1], 10), n = parseInt(m[2], 10);
  const views = f.views();
  const rem = new Set();
  for (const [, removed] of views) {
    for (const line of removed) for (const x of _p2aDefNames(line, ["def"])) if (x.startsWith("test_")) rem.add(x);
  }
  if (!rem.size) return null;
  let most = 0;
  for (const [added] of views) {
    let k = 0;
    for (const line of added) for (const names of _p2aCounted(line)) if ([...names].some(x => rem.has(x))) k++;
    most = Math.max(most, k);
  }
  const chg = Math.min(got, most);
  return (chg && got - chg <= n && n <= got) ? ["tests", "#101"] : null;
}

function _p2aSymbol(c, f) {
  const name = c.detail.name;
  for (const [, removed] of f.views()) {
    for (const line of removed) if (_p2aDefNames(line, ["def", "class"]).has(name)) return ["symbol", "#101"];
  }
  return null;
}

function _p2aDecide(c, f) {
  if (_PATH_KINDS.has(c.kind)) return _p2aPath(c, f);
  if (c.kind === "files_changed_count") return _p2aCount(c, f);
  if (c.kind === "only_touches") return _p2aOnly(c, f);
  if (c.kind === "tests_added") return _p2aTests(c, f);
  return _p2aSymbol(c, f);
}

function _p2aReason(verdict, defect, key, why) {
  return `${verdict} withheld by PATH-2a (${defect}): ${_P2A_PHRASES[key]}. main's reading: ${why}`;
}

function _p2aAbstain(g, strict, facts) {
  // Turn a decided verdict UNCHECKABLE where #97, #121 or #101 can have made it wrong, then recompute the gate
  // verdict with main's own formula. Nothing else in the record moves.
  const todo = g.claims.filter(c => _P2A_REACH.has(c.kind + "|" + c.verdict));
  if (!todo.length) return g;
  let hits;
  try {
    const f = facts();
    hits = todo.map(c => [c, _p2aDecide(c, f)]);
  } catch (e) {                          // an abstain-only overlay that cannot read withholds, and says so
    hits = todo.map(c => [c, ["error", _P2A_KIND_DEFECT[c.kind]]]);
  }
  for (const [c, hit] of hits) {
    if (hit !== null) {
      c.why = _p2aReason(c.verdict, hit[1], hit[0], c.why);
      c.verdict = "UNCHECKABLE";
    }
  }
  const contradicted = g.claims.some(c => c.verdict === "CONTRADICTED");
  const uncheckable = g.claims.some(c => c.verdict === "UNCHECKABLE");
  g.verdict = (contradicted || (strict && uncheckable)) ? "FAIL" : "PASS";
  return g;
}

function gateDiffText(summaryText, diffText, opts = {}) {
  const g = _gateDiffTextMain(summaryText, diffText, opts);
  return _p2aAbstain(g, !!(opts && opts.strict), () => _p2aFactsRaw(diffText || ""));
}
// === PATH-2a abstain-only overlay: END ===


if (typeof globalThis !== "undefined") globalThis.styxxDiffgateJS = { gateDiffText, parseUnifiedDiff, parseUnifiedDiffSides };

/* the gate, as a bookmarklet: on any public GitHub pull request page, one click reads the
 * description against the diff (both from api.github.com, nothing else) and pins the verdict
 * to the top of the page. Same JS port as the preview build (differential-tested against the
 * styxx Python instrument at the BC-2 + COMPAT-1 + #118 + COMPAT-2 checkout, the file 7.48.0 ships). Nothing
 * is sent anywhere; nothing is stored. */
(async function () {
  const G = window.styxxDiffgateJS;
  const m = location.pathname.match(/^\/([\w.-]+)\/([\w.-]+)\/pull\/(\d+)/);
  const old = document.getElementById("styxx-gate-panel"); if (old) old.remove();
  const panel = document.createElement("div");
  panel.id = "styxx-gate-panel";
  panel.setAttribute("style", "all:initial;position:fixed;top:12px;right:12px;z-index:2147483647;width:min(720px,calc(100vw - 24px));max-height:80vh;overflow:auto;background:#0b0d13;color:#ecf4f1;border:1px solid #2a3038;border-radius:8px;padding:14px 16px;font:13px/1.6 'IBM Plex Mono',ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;box-shadow:0 12px 40px rgba(0,0,0,.6);white-space:pre-wrap;word-break:break-word");
  const esc = s => String(s).replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
  const head = (t) => `<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px"><span style="color:#96d0c4;letter-spacing:.06em;text-transform:uppercase;font-size:11px">styxx · diffgate · the description vs the diff</span><button id="styxx-gate-close" style="all:initial;cursor:pointer;color:#687a76;font:13px monospace;padding:2px 6px">✕</button></div><div style="color:#b9c7c3;margin-bottom:8px">${esc(t)}</div>`;
  panel.innerHTML = head(m ? `reading ${m[1]}/${m[2]}#${m[3]} from api.github.com…` : "open a pull request page (github.com/OWNER/REPO/pull/N) and click again.");
  document.body.appendChild(panel);
  panel.querySelector("#styxx-gate-close").onclick = () => panel.remove();
  if (!m) return;
  const api = `https://api.github.com/repos/${m[1]}/${m[2]}/pulls/${m[3]}`;
  let meta, diff;
  try {
    const [r1, r2] = await Promise.all([fetch(api, {headers:{Accept:"application/vnd.github+json"}}), fetch(api, {headers:{Accept:"application/vnd.github.diff"}})]);
    if (r1.status === 403 || r1.status === 429) throw new Error("GitHub's unauthenticated API limit (60/hour per address) is used up; wait, or run `python -m styxx.diffgate --pr <url>` with GITHUB_TOKEN set.");
    if (!r1.ok) throw new Error(`GitHub answered HTTP ${r1.status} for the pull request (private repos are not readable from here).`);
    if (r2.status === 406) throw new Error("GitHub will not serve this diff over the API (too large); run the CLI on a checkout instead.");
    if (!r2.ok) throw new Error(`GitHub answered HTTP ${r2.status} for the diff.`);
    meta = await r1.json(); diff = await r2.text();
  } catch (e) {
    panel.innerHTML = head(`${m[1]}/${m[2]}#${m[3]}`) + `<div style="color:#ecc46e">${esc(e.message)}</div>`;
    panel.querySelector("#styxx-gate-close").onclick = () => panel.remove();
    return;
  }
  const body = meta.body || "";
  const g = G.gateDiffText(body, diff);
  const col = {VERIFIED:"#78e296", CONTRADICTED:"#ff605c", UNCHECKABLE:"#687a76"};
  const mark = {VERIFIED:"[ok ]", CONTRADICTED:"[LIE]", UNCHECKABLE:"[ ? ]"};
  let out = "";
  if (!body.trim()) out += `<div style="color:#687a76">the description is empty — nothing to gate.</div>`;
  for (const c of g.claims) out += `<div style="color:${col[c.verdict]}">  ${mark[c.verdict]} ${esc(c.kind.padEnd(20))} ${esc(c.why)}</div>`;
  if (!g.measured) out += `<div style="color:#ecc46e">UNMEASURED  this gate did not run: ${esc(g.why_unmeasured)}</div>`;
  const nc = g.claims.filter(c => c.verdict === "CONTRADICTED").length, nu = g.claims.filter(c => c.verdict === "UNCHECKABLE").length;
  out += `<div style="color:${g.verdict === "PASS" ? "#ecc46e" : "#ff605c"};margin-top:8px;font-weight:${g.verdict === "PASS" ? 400 : 600}">${g.verdict}  claims=${g.claims.length} contradicted=${nc} uncheckable=${nu} uncovered_sentences=${g.uncovered_sentences}</div>`;
  if (g.sentences_total) out += `<div style="color:#687a76">never read: ${g.uncovered_sentences} of ${g.sentences_total} sentences — prose outside the closed template set is not judged</div>`;
  if (!g.claims.length) out += `<div style="color:#687a76">no diff-shaped claims found — silence is scope, not weakness</div>`;
  out += `<details style="margin-top:8px;color:#687a76"><summary style="cursor:pointer">what it reads · reproduce</summary><div style="margin-top:6px">modified / created / deleted &lt;path&gt; · N files changed · added N tests · adds function &lt;name&gt; · only touches &lt;prefix&gt; · tests pass (UNCHECKABLE without --run) · no breaking changes (read, never judged: the public definitions the diff removed from the surface are named, test/example/internal removals counted, signature changes reported)\na path the diff does not show is UNCHECKABLE, not an accusation (EXTERNAL-1: precision 0.23 vs a 0.95 floor on 71,016 agent PRs). added N tests / adds function &lt;name&gt; count python def lines and say so when the diff has no python (#110).\n\npip install styxx\npython -m styxx.diffgate --pr ${esc(location.origin + location.pathname.match(/^\/[\w.-]+\/[\w.-]+\/pull\/\d+/)[0])}\n\njs port of styxx diffgate.py at the BC-2 + COMPAT-1 + #118 + COMPAT-2 checkout (7.48.0), differential-tested (3,212 pairs, 0 disagreements). the python is the instrument. github.com/fathom-lab/styxx</div></details>`;
  panel.innerHTML = head(`${m[1]}/${m[2]}#${m[3]} — ${meta.title || ""}`) + out;
  panel.querySelector("#styxx-gate-close").onclick = () => panel.remove();
})();

})();