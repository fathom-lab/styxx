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
// NOTE_path2a_abstain_overlay_2026_09_30, NOTE_path2a_second_pass_2026_09_30, NOTE_path2a_third_pass_2026_09_30,
// NOTE_path2a_fourth_pass_2026_09_30, NOTE_path2a_fifth_pass_2026_09_30 and NOTE_path2a_sixth_pass_2026_09_30;
// NOTE_path2a_ninth_pass_2026_10_04 removes the switch of passes six to eight that kept a CONTRADICTED where the
// two ports' mains might read the claims apart: a CONTRADICTED in reach is decided by its kind's rule.
// NOTE_path2a_tenth_pass_2026_10_05 splits the block in two, as the Python's: DECIDE (_p2aDecisions, every rule and
// reader below) is given a copy of main's claims and returns plain data; APPLY (_p2aApply) is the only code here that
// touches main's record, and the record can only gain abstentions, whatever the rest of the block does.
// The port's half of the PATH-2a block in styxx/diffgate.py (sha256 011538d50a5a4ed393fdaf6ef2470a7f26575568687cda9847fe9ceaf542ca76, LF). Everything outside this block is
// main's port at 1cde8b82 (sha256 06688702..., LF), unchanged except that main's gateDiffText is named
// _gateDiffTextMain (its definition and its DECLARE-1 self-call); the gateDiffText at the end of this block calls it
// and then the overlay, once. The overlay reads each DECIDED claim once more and turns it UNCHECKABLE, with a reason
// naming the verdict it withholds, the defect and main's reason verbatim, only where #97, #121 or #101 can have made
// it wrong. Every function mirrors the Python block; every comparison is structural (ASCII case, code points, '/',
// '.', fixed character sets), and a decision reads the claim's kind, verdict and detail, main's counts in its reason
// and the door's bytes, never the claim's text, so the two ports decide alike wherever main's two ports give a claim
// the same kind, verdict and detail. Where the Python reads a str by code point, this block reads UTF-16 units only
// where the two give the same answer: a unit from 0x80 up is a code point from 0x80 up, and no boundary it computes
// falls inside a surrogate pair.

const P2A_DIRECTORY_BASENAME_ABSTAINS = true;

const _P2A_PY_BREAKS = "\n\r\u000b\u000c\u001c\u001d\u001e\u0085\u2028\u2029";
const _P2A_JS_SPACE = "\t\n\u000b\u000c\r \u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff";
const _P2A_PY_SPACE = "\t\n\u000b\u000c\r\u001c\u001d\u001e\u001f \u0085\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000";
const _P2A_DIVERGENT = "\u000b\u000c\u001c\u001d\u001e\u001f\u0085\u2028\u2029\ufeff";
// NOTE_path2a_fifth_pass_2026_09_30, C-1, and its corrections: the count seam, as the Python's _P2A_ONE_SPACE,
// _P2A_ANY_SPACE and _P2A_FILE_FOLD_RX.
const _P2A_ONE_SPACE = "\u001c\u001d\u001e\u001f\u0085\ufeff";
// The neutral code points (NOTE_path2a_fourth_pass_2026_09_30, B-2), as the Python's _P2A_NEUTRAL class: all in the
// Basic Multilingual Plane, so each is one UTF-16 unit.
const _P2A_NEUTRAL_RANGES = [[0xa0, 0xa9], [0xab, 0xac], [0xae, 0xb1], [0xb4, 0xb4], [0xb6, 0xb8], [0xbb, 0xbb], [0xbf, 0xbf], [0xd7, 0xd7], [0xf7, 0xf7], [0x1680, 0x1680], [0x2000, 0x200a], [0x2010, 0x2027], [0x202f, 0x203e], [0x2041, 0x2053], [0x2055, 0x205f], [0x20a0, 0x20bf], [0x2190, 0x23ff], [0x2500, 0x2775], [0x2794, 0x27bf], [0x2b00, 0x2b73], [0x2b76, 0x2b95], [0x2b97, 0x2bff], [0x3000, 0x3004], [0x3008, 0x3020], [0x3030, 0x3030], [0x3036, 0x3037], [0x303d, 0x303f], [0xfe0e, 0xfe19], [0xfe30, 0xfe32], [0xfe35, 0xfe4c], [0xfe50, 0xfe52], [0xfe54, 0xfe66], [0xfe68, 0xfe6b], [0xff01, 0xff0f], [0xff1a, 0xff20], [0xff3b, 0xff3e], [0xff40, 0xff40], [0xff5b, 0xff65]];
const _P2A_OWN = 1;             // this port's main reads line view 1 (its own breaks) with its own white space
const _P2A_HEADERS = ["diff --git ", "--- ", "+++ ", "rename from ", "rename to ", "new file mode", "deleted file mode", "Binary files "];
const _P2A_FINE = new RegExp("\r\n|[" + _P2A_PY_BREAKS + "]");
const _P2A_COARSE = new RegExp("\r\n|\r|\n");
const _P2A_CR = new RegExp("\r");
const _P2A_FINE_ONLY = new RegExp("[\u000b\u000c\u001c\u001d\u001e\u0085\u2028\u2029]");   // where the two splits can part
const _P2A_LEAD_APART = new RegExp("[\u001c\u001d\u001e\u001f\u0085\ufeff]");   // white space of one main only
const _P2A_SEP = "\u0000";      // joins the summary's runs and zones; never in a claimed path, name, prefix or number
const _P2A_COUNT_HEAD = new RegExp("^diff changes ([0-9]+) files, claim says ");
const _P2A_TESTS_HEAD = new RegExp("^diff adds ([0-9]+) test functions, claim says ");
const _P2A_DIGITS = new RegExp("^[0-9]+$");
const _P2A_DIV_RX = new RegExp("[" + _P2A_DIVERGENT + "]");
const _P2A_FILE_FOLD_RX = new RegExp("[Ff][\u0130\u0131][Ll][Ee]|[Ff][Ii\u0130\u0131][Ll][Ee]\u017f");
const _P2A_WIDE = new RegExp("[\u0080-\uffff]");
const _P2A_REACH_PAIRS = [["file_created", "VERIFIED"], ["file_deleted", "VERIFIED"], ["file_touched", "VERIFIED"], ["files_changed_count", "VERIFIED"], ["files_changed_count", "CONTRADICTED"], ["only_touches", "VERIFIED"], ["only_touches", "CONTRADICTED"], ["tests_added", "VERIFIED"], ["tests_added", "CONTRADICTED"], ["symbol_added", "VERIFIED"]];
const _P2A_REACH = new Set(_P2A_REACH_PAIRS.map(p => p[0] + "|" + p[1]));
const _P2A_KIND_DEFECT = {"file_created": "#97, #121", "file_deleted": "#97, #121", "file_touched": "#97, #121", "files_changed_count": "#121", "only_touches": "#121", "tests_added": "#101", "symbol_added": "#101"};
// The defect tags a decision may carry, per kind, and the fields of a claim's detail that a rule reads: DECIDE's copy
// of a claim carries no other field, and not its text.
const _P2A_PATH_TAGS = ["#97", "#121", "#97, #121"];
const _P2A_TAGS = {"file_created": _P2A_PATH_TAGS, "file_deleted": _P2A_PATH_TAGS, "file_touched": _P2A_PATH_TAGS, "files_changed_count": ["#121"], "only_touches": ["#121"], "tests_added": ["#101"], "symbol_added": ["#101"]};
const _P2A_FIELDS = ["path", "n", "name", "prefix", "prefix2", "declared"];
const _P2A_PHRASES = {
  "dir": "the claim's path has a directory part, and only a changed file with the same base name in another directory matches it",
  "tier": "a changed path that matches the claim more closely than the one main resolved it to reads otherwise",
  "dot": "with leading dots kept, the changed path the claim resolves to reads otherwise",
  "dot_earliest": "with leading dots kept, the earliest changed path matching the claim reads otherwise, though the closest one does not",
  "dot_tier": "with leading dots kept and the closest match taken, the claim reads otherwise",
  "count": "two changed paths differ only by a leading dot, which the path key drops, and counted apart the claim reads otherwise",
  "only": "with leading dots kept, whether every changed path lies under the prefix reads otherwise",
  "shape": "with leading dots kept, no changed path has the prefix as a segment, so it is not read as a path",
  "tests": "a test the added lines count is also defined in the removed lines, and a changed test is not an added one",
  "split": "a test the added lines count is also defined in the removed lines, and the Python and JavaScript readers split or space these lines differently",
  "redefined": "a test the added lines count is also defined in an unchanged line of the diff, and a test defined again is not an added one",
  "symbol": "the removed lines define this name too, and a changed definition is not an added one",
  "again": "an unchanged line of the diff defines this name too, and a name defined again is not an added one",
  "extract": "the summary holds a character the Python and JavaScript readers may read differently where the claim's path, name, prefix or number is read, so the two may extract the claim differently",
  "seam": "the summary holds a character that only one of the Python and JavaScript readers reads as a space, or as a letter, where a count is read, so the two may read different counts",
  "divergent": "a file header of this diff holds a character that the Python and JavaScript readers split or strip differently",
  "odd": "a path here has a drive-like prefix or a final '.' segment, where base names are read differently",
  "case": "a path here compares only where case outside ASCII is folded, which this overlay does not do",
  "case_count": "two changed paths differ only in case outside ASCII, which the Python and JavaScript readers' case tables may merge or keep apart, so the two may count the files differently",
  "unreproduced": "this overlay does not reproduce main's reading of the diff",
  "unparsed": "main's reason does not have the form this overlay reads",
  "error": "this overlay failed while reading the diff",
  "malformed": "this overlay's decisions did not come back as a list, so none of them was applied"
};

function _p2aLines(text, rx) {
  const out = text.split(rx);
  return (out.length && out[out.length - 1] === "") ? out.slice(0, out.length - 1) : out;
}

function _p2aRegsRaw(lines) {
  // (path as written, status, via) wherever parseUnifiedDiff hands a path to _norm, in its order: its loop line
  // for line (`lines`, the diff as _splitlines splits it), `status.set(_norm(x), st)` read as a push; via "+" assigns,
  // "p" (a BIN-1 flush) registers only a key not yet held. A flush is kept even where its key is already registered.
  const regs = [];
  let oldPath = null;
  let pending = null;
  const flush = () => {
    if (pending !== null) {
      const raw = pending.status === "D" ? pending.a : pending.b;
      if (raw) regs.push([raw, pending.status, "p"]);
    }
  };
  for (const line of lines) {
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

function _p2aView(ls) {
  return [ls.filter(x => x.startsWith("+") && !x.startsWith("+++")).map(x => x.slice(1)),
          ls.filter(x => x.startsWith("-") && !x.startsWith("---")).map(x => x.slice(1))];
}

function _p2aViews(diffText, coarse) {
  // [[added, removed]] under CPython's line breaks and under the port's (`coarse`, the diff as _splitlines splits it);
  // each port's main reads one. Where the text holds no break only CPython's split reads, the two are one object.
  const v1 = _p2aView(coarse);
  if (!_P2A_FINE_ONLY.test(diffText)) return [v1, v1];
  return [_p2aView(_p2aLines(diffText, _P2A_FINE)), v1];
}

function _p2aDivergent(diffText) {
  if (!_P2A_DIV_RX.test(diffText)) return false;
  for (const line of _p2aLines(diffText, _P2A_COARSE)) {
    if (!_P2A_DIV_RX.test(line)) continue;
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
  const t = _p2aBs(s);
  let i = 0;                             // one scan, one slice (NOTE_path2a_fifth_pass_2026_09_30, A-3)
  for (;;) {
    if (t.startsWith("./", i)) i += 2;
    else if (t.startsWith("/", i)) i += 1;
    else return t.slice(i);
  }
}

function _p2aRun(s) {                    // the leading dots and slashes main drops and R-121 keeps
  const d = _p2aDotted(s);
  return d.slice(0, d.length - _p2aStrip(s).length);
}

function _p2aFold(s) {                   // ASCII case, and the two code points whose lowercase holds ASCII
  // A string with nothing to fold is returned as it is; any other is joined once, not appended to code point by code
  // point: an appended string is a chain of pieces this engine keeps until it is read whole, about 2 KB for a
  // 70-character path, and the forms are kept as keys (A-1 of the seventh pass). U+212A and U+0130 are one unit each.
  let k = 0;
  while (k < s.length) {
    const u = s.charCodeAt(k);
    if ((u >= 65 && u <= 90) || u === 0x212a || u === 0x130) break;
    k++;
  }
  if (k === s.length) return s;
  const out = [s.slice(0, k)];
  for (const ch of s.slice(k)) {
    const o = ch.codePointAt(0);
    out.push((o >= 65 && o <= 90) ? String.fromCharCode(o + 32) : o === 0x212a ? "k" : o === 0x130 ? "i\u0307" : ch);
  }
  return out.join("");
}

function _p2aWild(s) {                   // every code point outside ASCII read as one placeholder
  const f = _p2aFold(s);
  let k = 0;
  while (k < f.length && f.charCodeAt(k) < 128) k++;
  if (k === f.length) return f;
  const out = [f.slice(0, k)];
  for (const ch of f.slice(k)) out.push(ch.codePointAt(0) < 128 ? ch : "\ufffd");
  return out.join("");
}

const _p2aA = s => _p2aFold(_p2aStrip(s));
const _p2aK = s => _p2aFold(_p2aDotted(s));
const _p2aWA = s => _p2aWild(_p2aStrip(s));
const _p2aWK = s => _p2aWild(_p2aDotted(s));
const _P2A_FORMS = { A: [_p2aA, _p2aWA], K: [_p2aK, _p2aWK] };

// B-1 (NOTE_path2a_eighth_pass_2026_10_01): which code points from 0x80 up some runtime's lowercase may merge, as the
// Python's _P2A_LOWER_RUNS (start, end, step, delta: every code point whose lowercase is another single code point in
// Unicode 16.0, U+0130 and U+212A aside) and _P2A_NEVER_RX (scripts and symbols with no case, and the neutral ranges).
const _P2A_LOWER_RUNS = [[0xc0, 0xd6, 1, 32], [0xd8, 0xde, 1, 32], [0x100, 0x12e, 2, 1], [0x132, 0x136, 2, 1], [0x139, 0x147, 2, 1], [0x14a, 0x176, 2, 1], [0x178, 0x178, 1, -121], [0x179, 0x17d, 2, 1], [0x181, 0x181, 1, 210], [0x182, 0x184, 2, 1], [0x186, 0x186, 1, 206], [0x187, 0x187, 1, 1], [0x189, 0x18a, 1, 205], [0x18b, 0x18b, 1, 1], [0x18e, 0x18e, 1, 79], [0x18f, 0x18f, 1, 202], [0x190, 0x190, 1, 203], [0x191, 0x191, 1, 1], [0x193, 0x193, 1, 205], [0x194, 0x194, 1, 207], [0x196, 0x196, 1, 211], [0x197, 0x197, 1, 209], [0x198, 0x198, 1, 1], [0x19c, 0x19c, 1, 211], [0x19d, 0x19d, 1, 213], [0x19f, 0x19f, 1, 214], [0x1a0, 0x1a4, 2, 1], [0x1a6, 0x1a6, 1, 218], [0x1a7, 0x1a7, 1, 1], [0x1a9, 0x1a9, 1, 218], [0x1ac, 0x1ac, 1, 1], [0x1ae, 0x1ae, 1, 218], [0x1af, 0x1af, 1, 1], [0x1b1, 0x1b2, 1, 217], [0x1b3, 0x1b5, 2, 1], [0x1b7, 0x1b7, 1, 219], [0x1b8, 0x1b8, 1, 1], [0x1bc, 0x1bc, 1, 1], [0x1c4, 0x1c4, 1, 2], [0x1c5, 0x1c5, 1, 1], [0x1c7, 0x1c7, 1, 2], [0x1c8, 0x1c8, 1, 1], [0x1ca, 0x1ca, 1, 2], [0x1cb, 0x1db, 2, 1], [0x1de, 0x1ee, 2, 1], [0x1f1, 0x1f1, 1, 2], [0x1f2, 0x1f4, 2, 1], [0x1f6, 0x1f6, 1, -97], [0x1f7, 0x1f7, 1, -56], [0x1f8, 0x21e, 2, 1], [0x220, 0x220, 1, -130], [0x222, 0x232, 2, 1], [0x23a, 0x23a, 1, 10795], [0x23b, 0x23b, 1, 1], [0x23d, 0x23d, 1, -163], [0x23e, 0x23e, 1, 10792], [0x241, 0x241, 1, 1], [0x243, 0x243, 1, -195], [0x244, 0x244, 1, 69], [0x245, 0x245, 1, 71], [0x246, 0x24e, 2, 1], [0x370, 0x372, 2, 1], [0x376, 0x376, 1, 1], [0x37f, 0x37f, 1, 116], [0x386, 0x386, 1, 38], [0x388, 0x38a, 1, 37], [0x38c, 0x38c, 1, 64], [0x38e, 0x38f, 1, 63], [0x391, 0x3a1, 1, 32], [0x3a3, 0x3ab, 1, 32], [0x3cf, 0x3cf, 1, 8], [0x3d8, 0x3ee, 2, 1], [0x3f4, 0x3f4, 1, -60], [0x3f7, 0x3f7, 1, 1], [0x3f9, 0x3f9, 1, -7], [0x3fa, 0x3fa, 1, 1], [0x3fd, 0x3ff, 1, -130], [0x400, 0x40f, 1, 80], [0x410, 0x42f, 1, 32], [0x460, 0x480, 2, 1], [0x48a, 0x4be, 2, 1], [0x4c0, 0x4c0, 1, 15], [0x4c1, 0x4cd, 2, 1], [0x4d0, 0x52e, 2, 1], [0x531, 0x556, 1, 48], [0x10a0, 0x10c5, 1, 7264], [0x10c7, 0x10c7, 1, 7264], [0x10cd, 0x10cd, 1, 7264], [0x13a0, 0x13ef, 1, 38864], [0x13f0, 0x13f5, 1, 8], [0x1c89, 0x1c89, 1, 1], [0x1c90, 0x1cba, 1, -3008], [0x1cbd, 0x1cbf, 1, -3008], [0x1e00, 0x1e94, 2, 1], [0x1e9e, 0x1e9e, 1, -7615], [0x1ea0, 0x1efe, 2, 1], [0x1f08, 0x1f0f, 1, -8], [0x1f18, 0x1f1d, 1, -8], [0x1f28, 0x1f2f, 1, -8], [0x1f38, 0x1f3f, 1, -8], [0x1f48, 0x1f4d, 1, -8], [0x1f59, 0x1f5f, 2, -8], [0x1f68, 0x1f6f, 1, -8], [0x1f88, 0x1f8f, 1, -8], [0x1f98, 0x1f9f, 1, -8], [0x1fa8, 0x1faf, 1, -8], [0x1fb8, 0x1fb9, 1, -8], [0x1fba, 0x1fbb, 1, -74], [0x1fbc, 0x1fbc, 1, -9], [0x1fc8, 0x1fcb, 1, -86], [0x1fcc, 0x1fcc, 1, -9], [0x1fd8, 0x1fd9, 1, -8], [0x1fda, 0x1fdb, 1, -100], [0x1fe8, 0x1fe9, 1, -8], [0x1fea, 0x1feb, 1, -112], [0x1fec, 0x1fec, 1, -7], [0x1ff8, 0x1ff9, 1, -128], [0x1ffa, 0x1ffb, 1, -126], [0x1ffc, 0x1ffc, 1, -9], [0x2126, 0x2126, 1, -7517], [0x212b, 0x212b, 1, -8262], [0x2132, 0x2132, 1, 28], [0x2160, 0x216f, 1, 16], [0x2183, 0x2183, 1, 1], [0x24b6, 0x24cf, 1, 26], [0x2c00, 0x2c2f, 1, 48], [0x2c60, 0x2c60, 1, 1], [0x2c62, 0x2c62, 1, -10743], [0x2c63, 0x2c63, 1, -3814], [0x2c64, 0x2c64, 1, -10727], [0x2c67, 0x2c6b, 2, 1], [0x2c6d, 0x2c6d, 1, -10780], [0x2c6e, 0x2c6e, 1, -10749], [0x2c6f, 0x2c6f, 1, -10783], [0x2c70, 0x2c70, 1, -10782], [0x2c72, 0x2c72, 1, 1], [0x2c75, 0x2c75, 1, 1], [0x2c7e, 0x2c7f, 1, -10815], [0x2c80, 0x2ce2, 2, 1], [0x2ceb, 0x2ced, 2, 1], [0x2cf2, 0x2cf2, 1, 1], [0xa640, 0xa66c, 2, 1], [0xa680, 0xa69a, 2, 1], [0xa722, 0xa72e, 2, 1], [0xa732, 0xa76e, 2, 1], [0xa779, 0xa77b, 2, 1], [0xa77d, 0xa77d, 1, -35332], [0xa77e, 0xa786, 2, 1], [0xa78b, 0xa78b, 1, 1], [0xa78d, 0xa78d, 1, -42280], [0xa790, 0xa792, 2, 1], [0xa796, 0xa7a8, 2, 1], [0xa7aa, 0xa7aa, 1, -42308], [0xa7ab, 0xa7ab, 1, -42319], [0xa7ac, 0xa7ac, 1, -42315], [0xa7ad, 0xa7ad, 1, -42305], [0xa7ae, 0xa7ae, 1, -42308], [0xa7b0, 0xa7b0, 1, -42258], [0xa7b1, 0xa7b1, 1, -42282], [0xa7b2, 0xa7b2, 1, -42261], [0xa7b3, 0xa7b3, 1, 928], [0xa7b4, 0xa7c2, 2, 1], [0xa7c4, 0xa7c4, 1, -48], [0xa7c5, 0xa7c5, 1, -42307], [0xa7c6, 0xa7c6, 1, -35384], [0xa7c7, 0xa7c9, 2, 1], [0xa7cb, 0xa7cb, 1, -42343], [0xa7cc, 0xa7cc, 1, 1], [0xa7d0, 0xa7d0, 1, 1], [0xa7d6, 0xa7da, 2, 1], [0xa7dc, 0xa7dc, 1, -42561], [0xa7f5, 0xa7f5, 1, 1], [0xff21, 0xff3a, 1, 32], [0x10400, 0x10427, 1, 40], [0x104b0, 0x104d3, 1, 40], [0x10570, 0x1057a, 1, 39], [0x1057c, 0x1058a, 1, 39], [0x1058c, 0x10592, 1, 39], [0x10594, 0x10595, 1, 39], [0x10c80, 0x10cb2, 1, 64], [0x10d50, 0x10d65, 1, 32], [0x118a0, 0x118bf, 1, 32], [0x16e40, 0x16e5f, 1, 32], [0x1e900, 0x1e921, 1, 34]];
const _P2A_NEVER_RANGES = [[0x590, 0x8ff], [0x900, 0x109f], [0x1100, 0x139f], [0x1400, 0x1c7f], [0x2e80, 0x2fff], [0x3000, 0x9fff], [0xa000, 0xa63f], [0xa6a0, 0xa6ff], [0xa800, 0xab2f], [0xabc0, 0xd7ff], [0xf900, 0xfaff], [0xfb1d, 0xfdff], [0xfe70, 0xfefe], [0xff66, 0xffdc], [0x20000, 0x3ffff]].concat(_P2A_NEUTRAL_RANGES);
const _P2A_CASE = new Map();
for (const [lo, hi, step, delta] of _P2A_LOWER_RUNS) {
  for (let cp = lo; cp <= hi; cp += step) {
    _P2A_CASE.set(String.fromCodePoint(cp), String.fromCodePoint(cp + delta));
    _P2A_CASE.set(String.fromCodePoint(cp + delta), String.fromCodePoint(cp + delta));
  }
}
const _p2aNever = ch => { const u = ch.codePointAt(0); return _P2A_NEVER_RANGES.some(([a, b]) => a <= u && u <= b); };

function _p2aCaseCount(forms) {
  // #CA, as the Python's _p2a_case_count (B-1 of NOTE_path2a_eighth_pass_2026_10_01): how many keys main's count reads
  // over the fold forms when every two code points some runtime's lowercase may merge are merged, read by code point:
  // at each position from 0x80 up of a group of forms with one wild form, a code point with no case is read as itself,
  // and the others as their case class, or all as one placeholder where some form holds a code point of neither table.
  const groups = new Map();
  for (const x of forms) {
    const w = _p2aWild(x);
    if (!groups.has(w)) groups.set(w, []);
    groups.get(w).push(x);
  }
  let n = 0;
  for (const [wild, xs] of groups) {
    if (xs.length < 2) { n += 1; continue; }
    const keys = xs.map(x => [...x]);
    const w = [...wild];
    const open = _p2aFlags(w.length);    // the positions where some form holds a code point of neither table
    w.forEach((ch, i) => {
      if (ch === "\ufffd" && keys.some(k => !_p2aNever(k[i]) && !_P2A_CASE.has(k[i]))) open[i] = true;
    });
    const read = (ch, i) => (w[i] !== "\ufffd" || _p2aNever(ch)) ? ch : open[i] ? "\ufffd" : _P2A_CASE.get(ch);
    n += new Set(keys.map(k => k.map(read).join(""))).size;
  }
  return n;
}

function _p2aBase(p) {                   // main's _basename
  const q = _rstrip(p, "/");
  return q.slice(q.lastIndexOf("/") + 1);
}

function _p2aOdd(p) {                    // a final "." segment, or a drive-like second code point not before '/'
  const q = _rstrip(p, "/");
  const cps = Array.from(q.slice(0, 6)).slice(0, 3);
  return q === "." || q.endsWith("/.") || (cps.length >= 2 && cps[1] === ":" && (cps.length === 2 || cps[2] !== "/"));
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

function _p2aScan(m, c, tiered) {
  // _p2aResolve read key by key, for a claim whose base name is empty.
  for (const t of (tiered ? [0, 1, 2] : [null])) {
    if (t === 2 && c.includes("/") && P2A_DIRECTORY_BASENAME_ABSTAINS) return null;
    for (const p of m.keys()) {
      const u = _p2aTier(p, c);
      if (u !== null && (t === null || u === t)) return [p, m.get(p)];
    }
  }
  return null;
}

function _p2aResolve(f, space, c, tiered) {
  // [key, status] main's findPath returns (tiered false), or V97's: exact, then suffix, then base name, over the status
  // map of `space`. With a non-empty base name, the keys that match the claim at some tier are exactly the keys of its
  // base name, so main takes the earliest of those; V97 takes the claim itself, else the earliest key ending in "/" +
  // the claim (read through the claims' tree, _p2aEnds), else (only for a bare claim) the earliest key of its base name.
  const m = f.status(space);
  const b = _p2aBase(c);
  if (!b) return _p2aScan(m, c, tiered);
  const cand = f.groups(space).get(b);
  if (!cand) return null;
  if (!tiered) return [cand[0], m.get(cand[0])];
  if (m.has(c)) return [c, m.get(c)];
  const i = f.ends(space, "key", c)[0];
  if (i >= 0) {
    const p = f.order(space)[i];
    return [p, m.get(p)];
  }
  if (c.includes("/") && P2A_DIRECTORY_BASENAME_ABSTAINS) return null;
  return [cand[0], m.get(cand[0])];
}

function _p2aTree(claims) {
  // The claims' "/"-segments, read from their ends, as nested Maps; under the key null, the claim ending there.
  const root = new Map();
  for (const c of claims) {
    let node = root;
    const segs = c.split("/");
    for (let k = segs.length - 1; k >= 0; k--) {
      if (!node.has(segs[k])) node.set(segs[k], new Map());
      node = node.get(segs[k]);
    }
    node.set(null, c);
  }
  return root;
}

function _p2aEnds(texts, claims) {
  // Map c -> [the index of the earliest text that ends in "/" + c, or -1; how many texts do], as the Python's
  // _p2a_ends: each text read from its end one segment at a time through the claims' tree, only while the segments read
  // so far end some claim's (NOTE_path2a_fifth_pass_2026_09_30, A-1).
  // The two figures are kept in two Maps of numbers and joined at the end, so no array read out of a Map is stored into
  // (the store scan, A-2 of NOTE_path2a_eighth_pass_2026_10_01).
  const root = _p2aTree(claims);
  const earliest = new Map(), many = new Map();
  for (const c of claims) { earliest.set(c, -1); many.set(c, 0); }
  texts.forEach((p, i) => {
    let node = root, j = p.length;
    for (let k = p.lastIndexOf("/"); k >= 0; k = k > 0 ? p.lastIndexOf("/", k - 1) : -1) {
      node = node.get(p.slice(k + 1, j));
      if (node === undefined) break;
      const c = node.get(null);
      if (c !== undefined) {
        if (earliest.get(c) < 0) earliest.set(c, i);
        many.set(c, many.get(c) + 1);
      }
      j = k;
    }
  });
  const out = new Map();
  for (const c of claims) out.set(c, [earliest.get(c), many.get(c)]);
  return out;
}

function _p2aAutomaton(words) {
  // An Aho-Corasick automaton over `words` (non-empty strings), as the Python's _p2a_automaton: per state, its moves,
  // its fallback, and whether a word ends there or at a fallback of it (NOTE_path2a_fifth_pass_2026_09_30, A-2).
  const moves = [new Map()], hit = [false];
  for (const w of words) {
    let k = 0;
    for (const ch of w) {
      let i = moves[k].get(ch);
      if (i === undefined) {
        i = moves.length;
        moves.push(new Map());
        hit.push(false);
        moves[k].set(ch, i);
      }
      k = i;
    }
    hit[k] = true;
  }
  const back = [];                       // a fresh array, so the store scan reads back[i] as a store into the block's own
  for (let z = 0; z < moves.length; z++) back.push(0);
  const queue = [...moves[0].values()];
  for (const k of queue) {               // level by level; the queue grows as it is read
    for (const [ch, i] of moves[k]) {
      let u = back[k];
      while (u && !moves[u].has(ch)) u = back[u];
      const v = moves[u].has(ch) ? moves[u].get(ch) : 0;
      back[i] = v;
      hit[i] = hit[i] || hit[v];
      queue.push(i);
    }
  }
  return [moves, back, hit];
}

function _p2aHolds(auto, text) {
  // Whether some word of the automaton occurs in `text`, read once, code point by code point.
  const [moves, back, hit] = auto;
  let u = 0;
  for (const ch of text) {
    while (u && !moves[u].has(ch)) u = back[u];
    u = moves[u].has(ch) ? moves[u].get(ch) : 0;
    if (hit[u]) return true;
  }
  return false;
}

// #101. The Python reads code points; these read UTF-16 units, which give the same runs and the same boundaries.
const _p2aCoarseUnit = u => u <= 0x20 || u === 0x7f || u >= 0x80;
const _p2aWordUnit = u => u === 95 || (u >= 48 && u <= 57) || (u >= 65 && u <= 90) || (u >= 97 && u <= 122);
const _p2aNameUnit = u => _p2aWordUnit(u) || u >= 0x80;

function _p2aRunEnd(s, j, test) {        // where the run of units passing `test` from j ends
  while (j < s.length && test(s.charCodeAt(j))) j++;
  return j;
}

function _p2aSites(line, word) {
  // [j, r] for each `word` in the line followed by one or more coarse characters, which run from j to r (the
  // character that ends them, or the end of the line). The runs of distinct sites never overlap.
  const out = [];
  let i = line.indexOf(word);
  while (i >= 0) {
    const j = i + word.length;
    const r = _p2aRunEnd(line, j, _p2aCoarseUnit);
    if (r > j) out.push([j, r]);
    i = line.indexOf(word, i + 1 > r ? i + 1 : r);
  }
  return out;
}

const _P2A_LEADS = [_P2A_PY_SPACE, _P2A_JS_SPACE].map(s => {
  const set = new Set([...s].map(ch => ch.charCodeAt(0)));
  return u => set.has(u);
});

function _p2aCounted(line, lead) {
  // The ASCII name runs at the `def test_` sites of one added line that a main's count of `def test_` after white
  // space reads, where `lead` tests a unit of that main's white space: a site counts when, from the line start or the
  // last U+2028 / U+2029 before it, only that white space precedes it. A site counts exactly where a segment's leading
  // run ends, and a start inside a run already read reaches the same end, so each segment is read once.
  const out = [];
  if (!line.includes("def test_")) return out;
  const starts = [0];
  if (line.includes("\u2028") || line.includes("\u2029")) {
    for (let p = 0; p < line.length; p++) {
      const u = line.charCodeAt(p);
      if (u === 0x2028 || u === 0x2029) starts.push(p + 1);
    }
  }
  let last = -1;
  for (const b of starts) {
    if (b <= last) continue;
    const r = _p2aRunEnd(line, b, lead);
    last = r;
    if (line.startsWith("def test_", r)) {
      const e = _p2aRunEnd(line, r + 4, _p2aWordUnit);
      out.push(e < line.length && line.charCodeAt(e) >= 0x80 ? null : line.slice(r + 4, e));
    }
  }
  return out;
}

function _p2aWideName(line, j, r, e) {
  // Whether the coarse run [j, r) after a `def` or `class`, or the unit that ends the ASCII name run [r, e), is from
  // 0x80 up: CPython reads such a name through NFKC, so it may be any name (NOTE_path2a_fifth_pass_2026_09_30, B-2).
  for (let k = j; k < r; k++) if (line.charCodeAt(k) >= 0x80) return true;
  return e < line.length && line.charCodeAt(e) >= 0x80;
}

function _p2aRemoved(views, extra) {
  // The removed lines of each distinct line view, and the removed lines no view reads (_p2aJoined).
  return _p2aDistinct(views).map(([, removed]) => removed).concat([extra]);
}

function _p2aTestDefs(groups, nfkc = true) {
  // [the ASCII name runs of the tests the lines may define after `def`, whether some `def` there has a name read
  // through NFKC and so may define any test], as the Python's _p2a_test_defs (B-2); with `nfkc` false, such a `def` is
  // passed over (the unchanged lines, NOTE_path2a_sixth_pass_2026_09_30, B-1).
  const names = new Set();
  for (const lines of groups) {
    for (const line of lines) {
      for (const [j, r] of _p2aSites(line, "def")) {
        const e = _p2aRunEnd(line, r, _p2aWordUnit);
        if (_p2aWideName(line, j, r, e)) {
          if (nfkc) return [names, true];   // every counted site pairs now
          continue;
        }
        if (line.startsWith("test_", r)) names.add(line.slice(r, e));
      }
    }
  }
  return [names, false];
}

function _p2aPairing(views, alike = false, extra = [], unchanged = []) {
  // Per line view (0: CPython's breaks and white space; 1: this port's), [the number of `def test_` sites that view's
  // main counts, how many of them name a test a removed line may define after `def`, how many name a test a removed or
  // an unchanged line may define], as the Python's _p2a_pairing: a name read through NFKC pairs with every name (B-2);
  // `extra`, the removed lines no view reads (B-1); `unchanged`, the unchanged lines of each view and those no view
  // reads (NOTE_path2a_sixth_pass_2026_09_30, B-1).
  const [rem, wild] = _p2aTestDefs(_p2aRemoved(views, extra));
  const [ctx] = _p2aTestDefs([unchanged], false);
  const out = (alike ? _p2aDistinct(views) : views).map(([added], v) => {
    let got = 0, paired = 0, based = 0;
    for (const line of added) {
      for (const x of _p2aCounted(line, _P2A_LEADS[v])) {
        got++;
        const p = wild || (x === null ? rem.size > 0 : rem.has(x));
        if (p) paired++;
        if (p || (x === null ? ctx.size > 0 : ctx.has(x))) based++;
      }
    }
    return [got, paired, based];
  });
  return out.length === 1 ? [out[0], out[0]] : out;
}

function _p2aDistinct(views) {
  return views[0] === views[1] ? views.slice(0, 1) : views;
}

function _p2aContext(diffText, coarse) {
  // The unchanged lines (git lines starting with " ") of each distinct line view, without the " ", as the Python's
  // _p2a_context (NOTE_path2a_sixth_pass_2026_09_30, B-1).
  const unchanged = ls => ls.filter(x => x.startsWith(" ")).map(x => x.slice(1));
  if (!_P2A_FINE_ONLY.test(diffText)) return unchanged(coarse);
  return unchanged(_p2aLines(diffText, _P2A_FINE)).concat(unchanged(coarse));
}

function _p2aJoined(diffText, side = "-") {
  // The removed text no line view reads as a line, as more removed lines, as the Python's _p2a_joined
  // (NOTE_path2a_fifth_pass_2026_09_30, B-1): each piece after a lone CR inside a git line that starts with "-", and
  // each run of the base side's pieces joined where a piece ends in a backslash, the backslash read as a space, when a
  // piece of the run is removed. With `side` " ", the unchanged text no view reads the same way: the pieces after a
  // lone CR in lines starting with " ", and the joined runs no piece of which is removed (B-1 of the sixth pass).
  const out = [];
  let acc = [], hit = false;
  const flush = () => {
    if (acc.length > 1 && hit === (side === "-")) out.push(acc.map((x, k) => k < acc.length - 1 ? x.slice(0, -1) + " " : x).join(""));
  };
  for (const line of diffText.split("\n")) {
    const head = line.slice(0, 1);
    if (head === "+" || head === "\\") continue;
    if (head !== "-" && head !== " ") {
      flush();
      acc = [];
      hit = false;
      continue;
    }
    const pieces = line.includes("\r") ? _p2aLines(line.slice(1), _P2A_CR) : line.length > 1 ? [line.slice(1)] : [];
    if (head === side) for (const piece of pieces.slice(1)) out.push(piece);
    for (const piece of pieces) {
      if (!acc.length) hit = false;
      acc.push(piece);
      hit = hit || head === "-";
      if (!piece.endsWith("\\")) {
        flush();
        acc = [];
      }
    }
  }
  flush();
  return out;
}

function _p2aAnchored(line) {
  // [j, r] for each `def` or `class` of one removed line that V101's symbol check can read a definition after, as the
  // Python's _p2a_anchored: from the line start (or a U+2028 or U+2029), a run of coarse units, optionally `async` and
  // one or more coarse units, then the word, then one or more coarse units, which run from j to r.
  const starts = [0];
  if (line.includes("\u2028") || line.includes("\u2029")) {
    for (let p = 0; p < line.length; p++) {
      const u = line.charCodeAt(p);
      if (u === 0x2028 || u === 0x2029) starts.push(p + 1);
    }
  }
  const out = [];
  let last = -1;
  for (const b of starts) {
    if (b <= last) continue;
    const p = _p2aRunEnd(line, b, _p2aCoarseUnit);
    last = p;
    const at = [p];
    if (line.startsWith("async", p)) {
      const q = _p2aRunEnd(line, p + 5, _p2aCoarseUnit);
      if (q > p + 5) at.push(q);
    }
    for (const x of at) {
      for (const w of ["def", "class"]) {
        if (!line.startsWith(w, x)) continue;
        const j = x + w.length, r = _p2aRunEnd(line, j, _p2aCoarseUnit);
        if (r > j) out.push([j, r]);
      }
    }
  }
  return out;
}

function _p2aNameDefs(groups) {
  // The ASCII name run at every anchored `def` or `class` site of the lines whose name does not read through NFKC, as
  // the Python's _p2a_name_defs: the names the unchanged lines define (NOTE_path2a_sixth_pass_2026_09_30, B-1).
  const out = new Set(), seen = new Set();
  for (const lines of groups) {
    for (const line of lines) {
      if (seen.has(line)) continue;
      seen.add(line);
      for (const [j, r] of _p2aAnchored(line)) {
        const e = _p2aRunEnd(line, r, _p2aWordUnit);
        if (!_p2aWideName(line, j, r, e)) out.add(line.slice(r, e));
      }
    }
  }
  return out;
}

function _p2aDefRuns(views, extra = []) {
  // [the ASCII name run at every anchored `def` or `class` site of a removed line, in either view or in `extra`;
  // whether some such site's name is read through NFKC (_p2aWideName)]: a name of ASCII word characters can start
  // only where a site's coarse run ends, and ends where its ASCII run does.
  const out = new Set(), seen = new Set();
  for (const removed of _p2aRemoved(views, extra)) {
    for (const line of removed) {
      if (seen.has(line)) continue;
      seen.add(line);
      for (const [j, r] of _p2aAnchored(line)) {
        const e = _p2aRunEnd(line, r, _p2aWordUnit);
        if (_p2aWideName(line, j, r, e)) return [out, true];   // every claimed name is defined now
        out.add(line.slice(r, e));
      }
    }
  }
  return [out, false];
}

function _p2aDefines(views, name, extra = []) {
  // Whether a removed line, in either view or in `extra`, may define `name` after an anchored `def` or `class`: the
  // name starts where the coarse run after the word does or anywhere in it (never inside a surrogate pair), and ends
  // where its full name run or its ASCII run ends. Read this way only for a name outside ASCII word characters.
  if (!name) return false;
  const full = _p2aRunEnd(name, 0, _p2aNameUnit) === name.length;
  const word = _p2aRunEnd(name, 0, _p2aWordUnit) === name.length;
  const midPair = (s, k) => k > 0 && k < s.length && s.charCodeAt(k - 1) >= 0xd800 && s.charCodeAt(k - 1) <= 0xdbff
    && s.charCodeAt(k) >= 0xdc00 && s.charCodeAt(k) <= 0xdfff;
  for (const removed of _p2aRemoved(views, extra)) {
    for (const line of removed) {
      for (const [j, r] of _p2aAnchored(line)) {
        for (let k = j + 1; k <= r; k++) {
          if (midPair(line, k) || !line.startsWith(name, k)) continue;
          const e = k + name.length;
          if ((full && _p2aRunEnd(line, e, _p2aNameUnit) === e) || (word && _p2aRunEnd(line, e, _p2aWordUnit) === e)) return true;
        }
      }
    }
  }
  return false;
}

// The summary's checks. A unit from 0x80 up is "wordish" unless it is neutral or U+0085, U+2028, U+2029 or U+FEFF:
// CPython's templates may read it as a word character, or fold it to an ASCII letter, where the port's read neither.
// A plain array of booleans, not a typed array: a store into a typed array runs ToNumber (NOTE_path2a_sixth_pass_2026_09_30,
// C-2). Every index and every stored value here is a number or a boolean the block computes.
const _p2aFlags = n => { const out = []; for (let u = 0; u < n; u++) out.push(false); return out; };   // packed
const _P2A_NEUTRAL_UNITS = _p2aFlags(0x10000);
for (const [a, b] of _P2A_NEUTRAL_RANGES) for (let u = a; u <= b; u++) _P2A_NEUTRAL_UNITS[u] = true;
const _P2A_DIV_UNITS = new Set([..._P2A_DIVERGENT].map(ch => ch.charCodeAt(0)));
const _p2aWordishUnit = u => u >= 0x80 && _P2A_NEUTRAL_UNITS[u] === false && u !== 0x85 && u !== 0x2028 && u !== 0x2029 && u !== 0xfeff;
const _p2aBadUnit = u => _P2A_DIV_UNITS.has(u) || _p2aWordishUnit(u);
const _P2A_PATH_ASCII = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_./-\\";
const _P2A_PATH_UNITS = _p2aFlags(0x80);
for (let u = 0; u < 0x80; u++) _P2A_PATH_UNITS[u] = _P2A_PATH_ASCII.includes(String.fromCharCode(u));
const _p2aPathUnit = u => (u < 0x80 && _P2A_PATH_UNITS[u] === true) || _p2aWordishUnit(u);
const _p2aNameAnyUnit = u => _p2aWordUnit(u) || _p2aWordishUnit(u);
const _p2aCountUnit = u => (u >= 48 && u <= 57) || _p2aWordishUnit(u);
const _p2aRunUnit = kind => kind === "path" ? _p2aPathUnit : kind === "name" ? _p2aNameAnyUnit : _p2aCountUnit;
const _p2aLowUnit = u => (u >= 65 && u <= 90) ? u + 32 : u;

function _p2aRuns(s, inRun) {
  // Each maximal run of units passing inRun that holds a wordish unit, joined by _P2A_SEP, which no run holds: a string
  // that holds x, where x does not hold _P2A_SEP, exactly when some run does (NOTE_path2a_fourth_pass_2026_09_30, A-1).
  const out = [];
  let k = 0;
  while (k < s.length) {
    if (!inRun(s.charCodeAt(k))) { k++; continue; }
    const a = k;
    let wide = false;
    while (k < s.length && inRun(s.charCodeAt(k))) { if (_p2aWordishUnit(s.charCodeAt(k))) wide = true; k++; }
    if (wide) out.push(s.slice(a, k));
  }
  return out.join(_P2A_SEP);
}

function _p2aSeam(s) {
  // Whether the summary holds a count seam, as the Python's _p2a_seam: one pass over the maximal runs of either port's
  // white space, and only where a white space of one port alone occurs at all.
  if (_P2A_FILE_FOLD_RX.test(s)) return true;
  const one = new Set([..._P2A_ONE_SPACE].map(ch => ch.charCodeAt(0)));
  const any = new Set([..._P2A_PY_SPACE + "\ufeff"].map(ch => ch.charCodeAt(0)));
  let k = 0;
  while (k < s.length) {
    if (!any.has(s.charCodeAt(k))) { k++; continue; }
    const a = k;
    let hit = false;
    while (k < s.length && any.has(s.charCodeAt(k))) { if (one.has(s.charCodeAt(k))) hit = true; k++; }
    if (hit && a > 0 && k < s.length && "0123456789EeSs\u017f".includes(s.charAt(a - 1)) && "CcFfWw".includes(s.charAt(k))) return true;
  }
  return false;
}

const _P2A_MANY = 32;           // more distinct words than this are read through one automaton, not one scan each
// A-1 (NOTE_path2a_seventh_pass_2026_09_30): one automaton holds words of at most this many units, or a sixteenth of the
// text it reads if that is more, as the Python's _P2A_BUDGET
const _P2A_BUDGET = 65536;
// O-11 (NOTE_path2a_seventh_pass_2026_09_30, B-1): the emoji of five pictograph blocks (U+1F300 to U+1F64F, U+1F680 to
// U+1F6FF, U+1F900 to U+1F9FF, U+1FA70 to U+1FAFF), the two surrogates this engine's string holds for each, are read in
// the summary as one neutral unit (U+2190), as the Python's _P2A_EMOJI_RX reads them as one code point or two surrogates
const _P2A_EMOJI_RX = new RegExp("\ud83c[\udf00-\udfff]|\ud83d[\udc00-\ude4f\ude80-\udeff]|\ud83e[\udd00-\uddff\ude70-\udeff]");
const _P2A_EMOJI_AS = "\u2190";

function _p2aMarked(words, text) {
  // The words (distinct, non-empty) that occur in `text`, as the Python's _p2a_marked: one automaton over them, the text
  // read once, each state's words marked once (NOTE_path2a_sixth_pass_2026_09_30, A-1).
  const [moves, back, hit] = _p2aAutomaton(words);
  const ends = new Map();
  for (const w of words) {
    let k = 0;
    for (const ch of w) k = moves[k].get(ch);
    ends.set(k, w);
  }
  const seen = new Set(), got = new Set();
  const starts = _p2aFlags(0x10000);   // the units a word starts with: the root's only moves
  for (const w of words) {
    const k = w.charCodeAt(0);
    starts[k] = true;
  }
  let u = 0;
  for (let i = 0; i < text.length; i++) {
    const k = text.charCodeAt(i);
    if (u === 0 && starts[k] === false) continue;
    const ch = text.charAt(i);
    while (u && !moves[u].has(ch)) u = back[u];
    u = moves[u].has(ch) ? moves[u].get(ch) : 0;
    for (let v = u; v && hit[v] && !seen.has(v); v = back[v]) {
      seen.add(v);
      if (ends.has(v)) got.add(ends.get(v));
    }
  }
  return got;
}

function _p2aFound(words, text) {
  // The words (named by `tokens`: non-empty, none holding _P2A_SEP or a unit _p2aBadUnit reads) that occur in `text`, as
  // the Python's _p2a_found (NOTE_path2a_seventh_pass_2026_09_30, A-1): such a word lies inside one piece of the text
  // between such units, so the text is read as its distinct pieces; an empty text, or one with no piece, reads no word;
  // a word that is a piece is found by lookup; a word no shorter than the longest piece, and not one, is not there; the
  // rest one scan each up to _P2A_MANY, else through automata of at most _P2A_BUDGET units of words, or a sixteenth of
  // the pieces' length if that is more, each reading the pieces once. Each word is BMP text outside the surrogates, so
  // reading by UTF-16 unit finds each where the Python's code points do. Every way gives the same set.
  if (!text || !words.size) return new Set();
  const pieces = new Set();
  let a = 0;
  for (let k = 0; k <= text.length; k++) {
    if (k === text.length || text.charCodeAt(k) === 0 || _p2aBadUnit(text.charCodeAt(k))) {
      if (k > a) pieces.add(text.slice(a, k));
      a = k + 1;
    }
  }
  let longest = 0;
  for (const p of pieces) if (p.length > longest) longest = p.length;
  if (!longest) return new Set();
  const got = new Set([...words].filter(w => pieces.has(w)));
  const rest = [...words].filter(w => !pieces.has(w) && w.length < longest);
  if (!rest.length) return got;
  const joined = [...pieces].join(_P2A_SEP);
  if (rest.length <= _P2A_MANY) {
    for (const w of rest) if (joined.includes(w)) got.add(w);
    return got;
  }
  const limit = _P2A_BUDGET > (joined.length >> 4) ? _P2A_BUDGET : (joined.length >> 4);
  let batch = [], size = 0;
  for (const w of rest) {
    batch.push(w);
    size += w.length;
    if (size >= limit) {
      for (const x of _p2aMarked(batch, joined)) got.add(x);
      batch = [];
      size = 0;
    }
  }
  if (batch.length) for (const x of _p2aMarked(batch, joined)) got.add(x);
  return got;
}

function _p2aFindOnly(s, a, e) {         // the earliest "only" (ASCII case) wholly inside [a, e), or -1
  for (let k = a; k + 4 <= e; k++) {
    if (_p2aLowUnit(s.charCodeAt(k)) === 111 && _p2aLowUnit(s.charCodeAt(k + 1)) === 110
        && _p2aLowUnit(s.charCodeAt(k + 2)) === 108 && _p2aLowUnit(s.charCodeAt(k + 3)) === 121) return k;
  }
  return -1;
}

function _p2aZones(s) {
  // [where its earliest "only" ends, where it ends] for each sentence, as both ports end one (a line break, or '.', '!'
  // or '?' followed by a space, a tab or a carriage return), in which a unit the two ports may read apart follows the
  // earliest "only" (ASCII case).
  const out = [];
  const zone = (a, e) => {
    const o = _p2aFindOnly(s, a, e);
    if (o < 0) return;
    for (let k = o; k < e; k++) if (_p2aBadUnit(s.charCodeAt(k))) { out.push([o + 4, e]); return; }
  };
  let a = 0;
  for (let k = 0; k < s.length; k++) {
    const ch = s[k];
    if (ch === "\n") { zone(a, k); a = k + 1; }
    else if ((ch === "." || ch === "!" || ch === "?") && (s[k + 1] === " " || s[k + 1] === "\t" || s[k + 1] === "\r")) { zone(a, k + 1); a = k + 1; }
  }
  zone(a, s.length);
  return out;
}

function _p2aFactsRaw(diffText, summaryText) {
  // What the overlay reads, from the door's own bytes, computed once per diff when a claim needs it.
  const memo = new Map();
  const get = (k, make) => { if (!memo.has(k)) memo.set(k, make()); return memo.get(k); };
  const f = {
    // O-11 (NOTE_path2a_seventh_pass_2026_09_30, B-1): the summary is read with its pictograph emoji neutral
    summary: String(summaryText).split(_P2A_EMOJI_RX).join(_P2A_EMOJI_AS),
    diffText,
    coarse: () => get("coarse", () => _splitlines(diffText)),
    regs: () => get("regs", () => _p2aRegsRaw(f.coarse())),
    views: () => get("views", () => _p2aViews(diffText, f.coarse())),
    divergent: () => get("div", () => _p2aDivergent(diffText)),
    status: space => get(space, () => _p2aBuild(f.regs(), _P2A_FORMS[space][0])),
    order: space => get("order" + space, () => [...f.status(space).keys()]),
    groups: space => get("groups" + space, () => {
      const groups = new Map();
      for (const p of f.status(space).keys()) {
        const b = _p2aBase(p);
        if (!groups.has(b)) groups.set(b, []);
        groups.get(b).push(p);
      }
      return groups;
    }),
    // NOTE_path2a_fifth_pass_2026_09_30, A-1: the claimed paths of the claims in reach, named before any claim is read
    prime: paths => get("claimed", () => paths),
    ends: (space, which, c) => {
      const k = which === "wild" ? 1 : 0;
      const form = _P2A_FORMS[space][k];
      const texts = which === "key" ? f.order(space) : f.space(space)[k];
      const got = get("ends:" + space + ":" + which, () => _p2aEnds(texts, new Set((memo.get("claimed") || []).map(x => form(x)))));
      if (got.has(c)) return got.get(c);
      return get("ends:" + space + ":" + which + ":" + c, () => _p2aEnds(texts, new Set([c])).get(c));
    },
    automaton: want => get("automaton:" + want, () => {
      const regs = f.regs();
      const words = new Set();
      f.space("A")[1].forEach((w, i) => {
        const b = _p2aBase(w);
        if (b !== "" && !_P2A_WIDE.test(b) && (want === null || regs[i][1] === want)) words.add(b);
      });
      return _p2aAutomaton(words);
    }),
    joined: () => get("joined", () => _p2aJoined(diffText)),
    seam: () => get("seam", () => _p2aSeam(f.summary)),
    space: space => get("space" + space, () => {
      const [form, wform] = _P2A_FORMS[space];
      const regs = f.regs();
      const fs = regs.map(r => form(r[0])), ws = regs.map(r => wform(r[0]));
      const keys = new Map(), sts = new Map(), wideByBase = new Map(), wide = [];
      regs.forEach((r, i) => {
        if (!keys.has(ws[i])) { keys.set(ws[i], new Set()); sts.set(ws[i], new Set()); }
        keys.get(ws[i]).add(fs[i]);
        sts.get(ws[i]).add(r[1]);
        if (_P2A_WIDE.test(fs[i])) {
          const b = _p2aBase(ws[i]);
          if (!wideByBase.has(b)) wideByBase.set(b, []);
          wideByBase.get(b).push(i);
          wide.push(i);
        }
      });
      return [fs, ws, keys, sts, wideByBase, wide];
    }),
    odd: () => get("odd", () => ["A", "K"].some(sp => f.space(sp)[0].some(_p2aOdd))),
    count: () => get("count", () => {
      const regs = f.regs();
      const ra = regs.filter(r => r[2] === "+" || _p2aStrip(r[0])).map(r => r[0]);
      const rk = regs.filter(r => r[2] === "+" || _p2aDotted(r[0])).map(r => r[0]);
      let twin = rk.length !== ra.length;  // a dotted-only name ("." , "..") V121 would register
      const runs = new Map();
      for (const raw of ra) {
        const w = _p2aWA(raw), run = _p2aRun(raw);
        if (!runs.has(w)) runs.set(w, run);
        if (runs.get(w) !== run) twin = true;
      }
      const size = (xs, form) => new Set(xs.map(x => form(x))).size;
      // B-1 (NOTE_path2a_eighth_pass_2026_10_01): #CA, read only where #WA is below #A, else #A
      const wa = size(ra, _p2aWA), forms = new Set(ra.map(x => _p2aA(x)));
      return [twin, wa, forms.size, size(rk, _p2aWK), size(rk, _p2aK), wa < forms.size ? _p2aCaseCount(forms) : forms.size];
    }),
    // NOTE_path2a_sixth_pass_2026_09_30, B-1: the unchanged lines of each view, and those no view reads
    unchanged: () => get("unchanged", () => _p2aContext(diffText, f.coarse()).concat(_p2aJoined(diffText, " "))),
    pairing: () => get("pairing", () => _p2aPairing(f.views(), f.views()[0] === f.views()[1] && !_P2A_LEAD_APART.test(diffText), f.joined(), f.unchanged())),
    // NOTE_path2a_sixth_pass_2026_09_30, A-1: the words the claims in reach may look up in the summary's runs or zones
    tokens: (kind, named) => get("tokens:" + kind, () => new Set(named.filter(w => {
      if (!w || w.includes(_P2A_SEP)) return false;
      for (let k = 0; k < w.length; k++) if (_p2aBadUnit(w.charCodeAt(k))) return false;
      return true;
    }))),
    occurs: (kind, s) => {
      const text = kind === "zone" ? f.zoneText() : f.runs(kind);
      const named = memo.get("tokens:" + kind) || new Set();
      if (named.has(s)) return get("found:" + kind, () => _p2aFound(named, text)).has(s);
      return get("in\u0000" + kind + "\u0000" + s, () => text.includes(s));
    },
    defines: name => {
      const [defRuns, wild] = get("defs", () => _p2aDefRuns(f.views(), f.joined()));
      if (wild) return true;
      if (name && _p2aRunEnd(name, 0, _p2aWordUnit) === name.length) return defRuns.has(name);
      return get("defines:" + name, () => _p2aDefines(f.views(), name, f.joined()));
    },
    // NOTE_path2a_sixth_pass_2026_09_30, B-1: whether an unchanged line may define `name`, an ASCII name
    redefines: name => !!name && _p2aRunEnd(name, 0, _p2aWordUnit) === name.length && get("udefs", () => _p2aNameDefs([f.unchanged()])).has(name),
    runs: kind => get("runs" + kind, () => _p2aRuns(f.summary, _p2aRunUnit(kind))),
    zones: () => get("zones", () => _p2aZones(f.summary)),
    zoneText: () => get("zoneText", () => f.zones().map(([lo, hi]) => f.summary.slice(lo, hi)).join(_P2A_SEP)),
    memo: get,
    scope: prefixes => get("scope:" + prefixes.length + ":" + prefixes.join("\n"), () => _p2aScope(f, prefixes)),
  };
  return f;
}

function _p2aInRuns(f, s, kind) {
  // Whether some occurrence of s, which is ASCII, lies in the summary inside a run of path (name, number) characters
  // holding a character CPython's template may read as a word character and the port's does not. No run holds
  // _P2A_SEP, so a string holding it lies in none.
  if (!s || s.includes(_P2A_SEP)) return false;
  return f.occurs(kind, s);
}

function _p2aPortMayVerify(f, claimed, want) {
  // For a claimed path holding a code point from 0x80 up: whether the port's reading of it could be VERIFIED. The
  // port's template reads only ASCII, so its path is a part of this one, and a path main verifies ends in the base name
  // of some registration with the claimed status: some such registration's ASCII base name is held by the claimed path,
  // read once through the automaton of those base names, memoised per claimed path (NOTE_path2a_fifth_pass_2026_09_30,
  // A-2).
  const held = _p2aFold(_p2aBs(claimed));
  return f.memo("port\u0000" + want + "\u0000" + held, () => _p2aHolds(f.automaton(want), held));
}

function _p2aExtract(f, c, claimed, want) {
  // Where the two ports' templates may extract a claimed path apart, and both mains could verify it. A declared
  // claim's sentence is written by DECLARE-1 from a value both ports read only when it is ASCII.
  if (_P2A_WIDE.test(claimed)) return _p2aPortMayVerify(f, claimed, want);
  return !c.detail.declared && _p2aInRuns(f, claimed, "path");
}

function _p2aCaseDoubt(f, space, claimed, want) {
  // U1: a path matches the claim at another tier once case outside ASCII is a placeholder. U2 (a status claim):
  // a path the claim may match shares that placeholder key with another key and a status other than the one
  // claimed. Only the claim's base-name group is read, and in it only the registrations whose fold form holds a
  // code point from 0x80 up: for the others neither U1 nor U2 can hold. Memoised per fold form and status (A-2).
  const [form] = _P2A_FORMS[space];
  const cf = form(claimed);
  return f.memo("case\u0000" + space + "\u0000" + want + "\u0000" + cf, () => _p2aCase(f, space, claimed, want));
}

function _p2aCase(f, space, claimed, want) {
  // _p2aCaseDoubt without a scan of the group per claim, as the Python's _p2a_case (NOTE_path2a_fifth_pass_2026_09_30,
  // A-2): U1 holds exactly when some fold base name in the group is not the claim's, a registration with the claim's
  // wild form has another fold form, or more registrations end in "/" + the wild form than in "/" + the fold form.
  const [form, wform] = _P2A_FORMS[space];
  const [fs, ws, keys, sts, wideByBase, wide] = f.space(space);
  const cw = wform(claimed), cf = form(claimed);
  const b = _p2aBase(cw);
  const merged = i => { const s = sts.get(ws[i]); return keys.get(ws[i]).size > 1 && !(s.size === 1 && s.has(want)); };
  if (!b) {
    for (const i of wide) {
      const tw = _p2aTier(ws[i], cw);
      if (tw !== _p2aTier(fs[i], cf)) return true;
      if (want !== null && tw !== null && merged(i)) return true;
    }
    return false;
  }
  const group = wideByBase.get(b) || [];
  if (cw !== cf) {                       // a claim in ASCII compares alike under both forms with every registration
    const bases = f.memo("bases\u0000" + space + "\u0000" + b, () => new Set(group.map(i => _p2aBase(fs[i]))));
    if (bases.size > 1 || (bases.size === 1 && !bases.has(_p2aBase(cf)))) return true;
    if (keys.has(cw)) {
      const ks = keys.get(cw);
      if (!(ks.size === 1 && ks.has(cf))) return true;
    }
    if (f.ends(space, "wild", cw)[1] !== f.ends(space, "fold", cf)[1]) return true;
  }
  return want !== null && f.memo("merged\u0000" + space + "\u0000" + want + "\u0000" + b, () => group.some(merged));
}

function _p2aPath(c, f) {
  const claimed = c.detail.path;
  const want = c.kind === "file_created" ? "A" : c.kind === "file_deleted" ? "D" : null;
  if (f.divergent()) return ["divergent", "#97, #121"];
  if (_p2aExtract(f, c, claimed, want)) return ["extract", "#97, #121"];
  if (f.odd() || _p2aOdd(_p2aA(claimed)) || _p2aOdd(_p2aK(claimed))) return ["odd", "#97"];
  if (_p2aCaseDoubt(f, "A", claimed, want) || _p2aCaseDoubt(f, "K", claimed, want)) return ["case", "#97, #121"];
  const ca = _p2aA(claimed), ck = _p2aK(claimed);
  const ok = r => r !== null && (want === null || r[1] === want);
  if (!ok(_p2aResolve(f, "A", ca, false))) return ["unreproduced", "#97"];
  const r97 = _p2aResolve(f, "A", ca, true);
  const v97 = ok(r97);
  const v121 = ok(_p2aResolve(f, "K", ck, false));
  const vboth = ok(_p2aResolve(f, "K", ck, true));
  if (v97 && v121 && vboth) return null;
  if (v121 && !v97) return [r97 === null ? "dir" : "tier", "#97"];
  if (v97 && !v121) return [vboth ? "dot_earliest" : "dot", "#121"];
  return ["dot_tier", "#97, #121"];
}

function _p2aNumbers(head, why, detail) {
  // [main's own count, the claimed number], or null. The count is read from main's reason; the number from the
  // claim's detail where that is ASCII digits (this port's main prints a number of 10**21 or more in exponent form),
  // else from main's reason, each read through _p2aInt's fixed table of the ten digits (C-2).
  const m = head.exec(why);
  if (!m) return null;
  let k = _P2A_DIGITS.exec(detail.n || "");
  if (!k) k = _P2A_DIGITS.exec(why.slice(m[0].length));
  if (!k) return null;
  return [_p2aInt(m[1]), _p2aInt(k[0])];
}

const _P2A_DIGIT = new Map([["0", 0], ["1", 1], ["2", 2], ["3", 3], ["4", 4], ["5", 5], ["6", 6], ["7", 7], ["8", 8], ["9", 9]]);

function _p2aInt(digits) {
  // The value of a string of ASCII digits, read digit by digit from a fixed table, as the Python's _p2a_int: no
  // parseInt or Number, which skip the engine's white space before the digits (NOTE_path2a_fifth_pass_2026_09_30, C-2). Any other
  // character throws, and the overlay's error fallback withholds.
  let n = 0;
  for (const ch of digits) {
    const d = _P2A_DIGIT.get(ch);
    if (d === undefined) throw new Error("not an ASCII digit");
    n = n * 10 + d;
  }
  return n;
}

function _p2aCount(c, f) {
  if (f.divergent()) return ["divergent", "#121"];
  const [twin, wa, a, lo, hi, ca] = f.count();
  // B-1 (NOTE_path2a_eighth_pass_2026_10_01): two changed paths that differ only in case outside ASCII, which each
  // runtime's case tables merge or not: a claimed number in [#CA, #A] may be decided apart by the two ports' mains, so it
  // is withheld in both, whatever its verdict, as the Python's _p2a_count
  const claimed = c.detail.n || "";
  if (ca < a && (!_P2A_DIGITS.test(claimed) || (ca <= _p2aInt(claimed) && _p2aInt(claimed) <= a))) return ["case_count", "#121"];
  if (!twin) return null;
  // C-1 (NOTE_path2a_fifth_pass_2026_09_30): a count one port reads cleanly and the other does not read, from a white
  // space only one of them reads or a letter only CPython folds; withheld in both, ahead of the guard below, wherever the summary holds one.
  if (!c.detail.declared && f.seam()) return ["seam", "#121"];
  // C-1 (NOTE_path2a_fourth_pass_2026_09_30): where a digit or letter outside ASCII sits beside the number, the two
  // ports' count templates read different numbers from one match; withheld in both, as the Python's _p2a_count.
  if (!_P2A_DIGITS.test(claimed) || (!c.detail.declared && _p2aInRuns(f, claimed, "count"))) return ["extract", "#121"];
  const nums = _p2aNumbers(_P2A_COUNT_HEAD, c.why, c.detail);
  if (!nums) return ["unparsed", "#121"];
  const [g, n] = nums;
  if (!(wa <= g && g <= a)) return ["unreproduced", "#121"];
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

function _p2aScope(f, prefixes) {
  // {space|wild: [whether `prefix` reads as a path, whether prefix2 does, whether every changed path lies under
  // the prefixes that reading uses]}, in main's key space A and V121's K, with case folded (wild false) and outside
  // ASCII a placeholder (true).
  const two = prefixes.length === 2;
  const regs = f.regs();
  const out = new Map();
  for (const [sp, form, wform, preKey] of _P2A_SPACES) {
    const [fs, ws] = f.space(sp);
    const keep = [];
    regs.forEach((r, i) => { if (r[2] === "+" || preKey(r[0])) keep.push(i); });
    for (const [wild, fm, forms] of [[false, form, fs], [true, wform, ws]]) {
      const paths = keep.map(i => forms[i]);
      const lead = _p2aShaped(prefixes[0], paths, fm);
      const shaped = two && _p2aShaped(prefixes[1], paths, fm);
      const ps = (shaped ? prefixes : prefixes.slice(0, 1)).map(x => _rstrip(fm(x), "/."));
      out.set(sp + "|" + wild, [lead, shaped, paths.every(p => ps.some(x => _p2aInside(p, x)))]);
    }
  }
  return out;
}

function _p2aScopeDoubt(f, d) {
  // Whether the two ports' templates may read this scope claim's prefixes apart: a prefix holds a character they read
  // apart, or an occurrence of the prefix after the earliest "only" of a sentence (as both ports end one) shares that
  // sentence with such a character after that "only". A declared claim's sentence is written by DECLARE-1.
  const bad = s => { for (let k = 0; k < s.length; k++) if (_p2aBadUnit(s.charCodeAt(k))) return true; return false; };
  if (bad(d.prefix) || bad(d.prefix2 || "")) return true;
  if (d.declared) return false;
  if (d.prefix.includes(_P2A_SEP)) return f.zones().some(([lo, hi]) => f.summary.slice(lo, hi).includes(d.prefix));
  return f.occurs("zone", d.prefix);
}

function _p2aOnly(c, f) {
  // Keep the claim when V121 (keys with their leading dots) reads the claim as main does: `prefix` as a path
  // where main does, when a second prefix is claimed; and every changed path under the prefix set it uses exactly as
  // main does under the set main uses.
  if (f.divergent()) return ["divergent", "#121"];
  const d = c.detail;
  if (_p2aScopeDoubt(f, d)) return ["extract", "#121"];
  const got = f.scope([d.prefix].concat(d.prefix2 ? [d.prefix2] : []));
  const same = (x, y) => x[0] === y[0] && x[1] === y[1] && x[2] === y[2];
  if (!same(got.get("A|false"), got.get("A|true")) || !same(got.get("K|false"), got.get("K|true"))) return ["case", "#121"];
  const [lead, , under] = got.get("A|false");
  if (!lead || (under ? "VERIFIED" : "CONTRADICTED") !== c.verdict) return ["unreproduced", "#121"];
  // B-1 (NOTE_path2a_fourth_pass_2026_09_30): V121 does not read a `prefix` only the dropped dots make a path.
  if (d.prefix2 && !got.get("K|false")[0]) return ["shape", "#121"];
  if (got.get("K|false")[2] !== under) return ["only", "#121"];
  return null;
}

function _p2aTests(c, f) {
  // PREREG R-101's interval [got - changed, got], read for each port's main whose count gives this claim the verdict it
  // has here: "tests" when every such reading holds the claim, "split" when only one does.
  const counts = f.pairing();
  if (!counts.some(([, , b]) => b)) return null;
  const nums = _p2aNumbers(_P2A_TESTS_HEAD, c.why, c.detail);
  if (!nums) return ["unparsed", "#101"];
  const [got, n] = nums;
  if (counts[_P2A_OWN][0] !== got) return ["unreproduced", "#101"];
  const holds = k => counts.filter(([g]) => (g === n) === (c.verdict === "VERIFIED"))
    .map(x => { const g = x[0], chg = g < x[k] ? g : x[k]; return chg > 0 && g - chg <= n && n <= g; });
  const fires = holds(1);
  if (fires.length && fires.every(x => x)) return ["tests", "#101"];
  if (fires.some(x => x)) return ["split", "#101"];
  // B-1 (NOTE_path2a_sixth_pass_2026_09_30): the same interval with the tests the unchanged lines define paired too
  return holds(2).some(x => x) ? ["redefined", "#101"] : null;
}

function _p2aSymbol(c, f) {
  const name = c.detail.name;
  if (_P2A_WIDE.test(name) || (!c.detail.declared && _p2aInRuns(f, name, "name"))) return ["extract", "#101"];
  if (f.defines(name)) return ["symbol", "#101"];
  if (f.redefines(name)) return ["again", "#101"];   // B-1 (NOTE_path2a_sixth_pass_2026_09_30)
  return null;
}

function _p2aDecide(c, f) {
  if (_PATH_KINDS.has(c.kind)) return _p2aPath(c, f);
  if (c.kind === "files_changed_count") return _p2aCount(c, f);
  if (c.kind === "only_touches") return _p2aOnly(c, f);
  if (c.kind === "tests_added") return _p2aTests(c, f);
  return _p2aSymbol(c, f);
}

function _p2aDecisions(seen, facts) {
  // DECIDE (NOTE_path2a_tenth_pass_2026_10_05): every rule and reader of this block, run on a copy. Returns plain
  // data, [[claim index, phrase key, defect tag], ...], for the claims in reach it would withhold. It is never given
  // main's record.
  const todo = [];
  seen.claims.forEach((c, i) => { if (_P2A_REACH.has(c.kind + "|" + c.verdict)) todo.push([i, c]); });
  const f = facts();
  const cs = todo.map(x => x[1]);
  f.prime(cs.filter(c => _PATH_KINDS.has(c.kind) && typeof c.detail.path === "string").map(c => c.detail.path));
  const str = xs => xs.filter(v => typeof v === "string");
  f.tokens("path", str(cs.map(c => c.detail.path)));
  f.tokens("name", str(cs.map(c => c.detail.name)));
  f.tokens("count", str(cs.map(c => c.detail.n)));
  f.tokens("zone", str(cs.map(c => c.detail.prefix)));
  const out = [];
  for (const [i, c] of todo) {
    const hit = _p2aDecide(c, f);
    if (hit !== null) out.push([i, hit[0], hit[1]]);
  }
  return out;
}

function _p2aApply(g, strict, decide) {
  // APPLY (NOTE_path2a_tenth_pass_2026_10_05): the only code of this block that touches main's record `g`. `decide`
  // is called on a copy and may do anything to it; whatever it returns or throws, the record leaves here as main's
  // but for claims in reach turned UNCHECKABLE with a reason of the fixed form, and the gate verdict is main's formula
  // over the final claims. A decision is taken only as a 3-element array of a number that is the index of a claim in
  // reach, a key of _P2A_PHRASES and a tag of that claim's kind; what is kept of it is the record's own claim, the
  // table's own phrase and the tag, a string. Anything else `decide` returned is ignored. If it throws, every claim in
  // reach is withheld with the phrase `error`; if it returns something that is not an array, with `malformed`.
  const claims = g.claims;
  const pending = new Map();
  claims.forEach((c, i) => { if (_P2A_REACH.has(c.kind + "|" + c.verdict)) pending.set(i, c); });
  if (!pending.size) return g;           // nothing in reach: no copy, no call, and the record is main's object
  let plan = [], fallback = null;
  try {                                  // everything `decide` made is read here, before anything is written
    const got = decide({ claims: claims.map(c => {
      const detail = {};
      for (const k of _P2A_FIELDS) {
        const v = c.detail[k];
        if (typeof v === "string" || typeof v === "boolean") detail[k] = v;
      }
      return { kind: c.kind, verdict: c.verdict, why: c.why, detail };
    }) });
    if (!Array.isArray(got)) fallback = "malformed";
    else {
      for (let k = 0; k < got.length; k++) {
        const d = got[k];
        if (!Array.isArray(d) || d.length !== 3) continue;
        const i = d[0], key = d[1], tag = d[2];
        if (typeof i !== "number" || typeof key !== "string" || typeof tag !== "string") continue;
        const c = pending.get(i);
        if (c === undefined) continue;
        const phrase = _P2A_PHRASES[key];
        if (typeof phrase === "string" && _P2A_TAGS[c.kind].includes(tag)) plan.push([c, phrase, tag]);
      }
    }
  } catch (e) {                          // an abstain-only overlay that cannot read withholds, and says so
    fallback = "error";
  }
  if (fallback !== null) plan = [...pending.values()].map(c => [c, _P2A_PHRASES[fallback], _P2A_KIND_DEFECT[c.kind]]);
  let moved = false;
  for (const [c, phrase, tag] of plan) {
    if (_P2A_REACH.has(c.kind + "|" + c.verdict)) {   // still decided: a second decision for one claim is ignored
      c.why = `${c.verdict} withheld by PATH-2a (${tag}): ${phrase}. main's reading: ${c.why}`;
      c.verdict = "UNCHECKABLE";
      moved = true;
    }
  }
  // A-2 (NOTE_path2a_sixth_pass_2026_09_30): where no claim moved, the record is main's object, untouched
  if (moved) {
    const contradicted = claims.some(c => c.verdict === "CONTRADICTED");
    const uncheckable = claims.some(c => c.verdict === "UNCHECKABLE");
    g.verdict = (contradicted || (strict && uncheckable)) ? "FAIL" : "PASS";
  }
  return g;
}

function _p2aAbstain(g, strict, facts) {
  // What the door calls on main's result: APPLY, over DECIDE reading the door's bytes through `facts`.
  return _p2aApply(g, strict, seen => _p2aDecisions(seen, facts));
}

function gateDiffText(summaryText, diffText, opts = {}) {
  // A-2 (NOTE_path2a_sixth_pass_2026_09_30): the options are read once, in main's order (strict, then _declared), and
  // main reads them from that snapshot, so main and the overlay read one strict. A null opts reaches main as it came,
  // and main answers it as main does.
  if (opts === null) return _gateDiffTextMain(summaryText, diffText, opts);
  const strict = opts.strict, declared = opts._declared;
  const g = _gateDiffTextMain(summaryText, diffText, { strict, _declared: declared });
  return _p2aAbstain(g, !!strict, () => _p2aFactsRaw(diffText || "", summaryText));
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