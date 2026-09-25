(function(){
/* diffgate.js — a JavaScript transliteration of styxx/diffgate.py for the browser surfaces that
 * cannot run Python: the paste-in preview page and the bookmarklet. The Python module is the
 * instrument; this file exists so a page can run the same closed template set with no network at
 * all, and it is held to the Python's output by a differential test (differential/ next to this
 * file) rather than by trust.
 *
 * Which Python: the file on the BC-2 + COMPAT-1 + BIN-2 + COMPAT-2 checkout (pull requests #113, #115,
 * #120 and #124 on fathom-lab/styxx, plus the fetch_pr door, with PATH-1 (#127) and DECLARE-1 (#129)
 * on top), sha256 9b620e00a19464589308a987819894ae7cc3c111c66a5f8a457a84b8a6c604eb, re-cut for the
 * PATH-2 repairs (PREREG_path2_resolution_2026_09_17: #97, #121, #101, as amended by
 * AMENDMENT_path2_resolution_2026_09_17, NOTE_path2_third_pass_2026_09_25,
 * NOTE_path2_fourth_pass_2026_09_25 and NOTE_path2_fifth_pass_2026_09_25) on the file that
 * carries them, sha256 0a5522ebdec4c16890070f79920f04f48f4b6a439c0086a01bbf85c1b9c2ec2b — the
 * styxx/diffgate.py that 7.48.0 ships once they merge. Relative to the 7.47.0 wheel the port
 * was first cut from, that file carries: the V14 repairs (containment demotes "touched" claims too;
 * a bare basename absent from the diff abstains), the BC-2 repairs for issue #110 (the def-counting
 * templates abstain when the diff has no Python; "added 3 test cases" is not a count of functions;
 * "adds a method to reload" is not a symbol; "only modifies the footer" is not a path), and the
 * COMPAT-1 reading of "no breaking changes" (one verdict, UNCHECKABLE, the public definitions the
 * diff removed named in the reason), the BIN-2 repair for #118 (a `diff --git` header with no
 * `---`/`+++` pair registers its file) and the COMPAT-2 sharpening (surface vs scaffolding paths,
 * signature changes reported, a candidate flag; the licence flag is false and the verdict stays
 * UNCHECKABLE). PATH-2 adds: a path claim resolves exact, then suffix, then
 * basename, over every entry (#97); the path key keeps a dotfile's dots, and "only touches" does not
 * accuse on a dot alone and lists only the paths outside by more than a dot (#121); a `def` the same
 * file's removed lines also define is changed, not added, paired one to one per name (#101). The
 * fourth pass adds: a diff splits into lines on \r\n, \r and \n only, in both implementations, and
 * the added lines are read with Python's `^` (the test count and the symbol test) and, in the symbol
 * test, Python's `\s` rather than JavaScript's. The fifth pass adds: one reading of a Python
 * definition line (CPython's indentation, space, tab and form feed, after one optional U+FEFF; the same
 * keyword separator; a name that ends at an ASCII non-name character) for the test count, the symbol
 * test and both pairings, with the symbol test now the pairing's own pattern; every regex over a diff
 * line spells Python's `\s`, `\w`, `\b` and `.`, a header path is stripped as str.strip() strips it, and
 * a reason prints a path as Python's repr() does; and a prefix written to end in `..`, or holding a `..`
 * after a named segment, could hold any path. Two
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
  // NOTE_path2_fourth_pass F-1: the prefix is undotted too, as the changed paths are below.
  const low = _rstrip(_undotted(_norm(raw)), "/").toLowerCase();
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
// NOTE_path2_third_pass_2026_09_25 (R-1, R-2): the added-side pattern and the added-blob count
// `got` read the same set of lines, so `chg <= got` line by line and not only in total; the
// removed side alone accepts `async`, so `async def test_x` rewritten as `def test_x` is a changed
// test rather than an added one. `got` carries the same optional U+FEFF as the pattern.
// NOTE_path2_fourth_pass_2026_09_25 (F-3): `got` reads the indent as [ \t]*, as the pairing does, so
// the two count exactly the same lines. (F-2): Python's re.M `^` matches at the start and after "\n"
// only; the `m` flag here also matched after "\r", U+2028 and U+2029, which a git diff does not treat
// as line breaks. `(?<![^\n])` is Python's `^` under re.M, so the blob is read line by line as git
// writes it.
// NOTE_path2_fifth_pass_2026_09_25 (V-1): ONE reading of a Python definition line, the Python's own:
// the indent is `_DEF_INDENT` (space, tab and form feed, CPython's tokenizer, after one optional
// U+FEFF), keywords are separated by `_DEF_SEP`, and a symbol name ends at `_NAME_END` (an ASCII
// character that cannot continue a name, or the end of the line). `got` counts the lines the
// added-side test pairing reads, and `hit` is the added-side symbol pairing pattern tested line by
// line; neither uses \s or \b any more, so there is no JavaScript/Python class left to differ.
const _DEF_INDENT = "^\\uFEFF?[ \\t\\f]*";
const _DEF_SEP = "[ \\t\\f]+";
const _NAME_END = "(?=[\\x00-\\x2f\\x3a-\\x40\\x5b-\\x5e\\x60\\x7b-\\x7f]|$)";
const _DEF_TEST_LINE = new RegExp(_DEF_INDENT + "def (test_[^ \\t(:]*)");
const _DEF_TEST_LINE_REMOVED = new RegExp(_DEF_INDENT + "(?:async" + _DEF_SEP + ")?def (test_[^ \\t(:]*)");
const _GOT_TEST_LINE = new RegExp("(?<![^\\n])" + _DEF_INDENT.slice(1) + "def test_", "g");
// Python's `\s` for a str pattern, written out: JavaScript's `\s` also matches U+FEFF and lacks
// U+001C-U+001F and U+0085. NOTE_path2_fifth_pass (V-2): used wherever the Python writes `\s` over a
// diff line -- the COMPAT patterns and their parameter lists -- and, as `_pyStrip`, wherever the
// Python calls str.strip() on one (a `---`/`+++` header, a parameter list).
const _PY_WS_CHARS = "\\t\\n\\x0b\\x0c\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const _PY_WS = "[" + _PY_WS_CHARS + "]";
// Python's `\w` for a str pattern: str.isalnum() or "_", i.e. Unicode letters and numbers. Needs the
// `u` flag. Equal to the Python's set on every code point Unicode 15.0 (Python 3.12) assigns; an
// engine on a later Unicode also reads the letters added since (NOTE_path2_fifth_pass, V-2).
const _PY_W_CHARS = "\\p{L}\\p{N}_";
const _PY_W = "[" + _PY_W_CHARS + "]";
// Python's Unicode `\b`, from `_PY_W`: a word character on exactly one side.
const _PY_B = "(?:(?<=" + _PY_W + ")(?!" + _PY_W + ")|(?<!" + _PY_W + ")(?=" + _PY_W + "))";
const _PY_STRIP = new RegExp("^" + _PY_WS + "+|" + _PY_WS + "+$", "g");
const _pyStrip = s => s.replace(_PY_STRIP, "");
// The removed-side symbol line (`async` allowed) and the added-side one, read by `hit` and by the
// pairing's added count alike.
const _symbolDefLine = name => new RegExp(_DEF_INDENT + "(?:async" + _DEF_SEP + ")?(?:def|class)" + _DEF_SEP + _reEscape(name) + _NAME_END);
const _symbolDefLineAdded = name => new RegExp(_DEF_INDENT + "(?:def|class)" + _DEF_SEP + _reEscape(name) + _NAME_END);
function _symbolHit(name, addedBlob) {
  const rx = _symbolDefLineAdded(name);
  return addedBlob.split("\n").some(line => rx.test(line));
}

function _changedTestDefs(sides, status) {
  // Per file whose status is not `A`, per test name, min(added lines defining it, removed lines
  // defining it), summed. The caller clamps to `got`.
  let n = 0;
  if (!sides) return 0;
  for (const [path, [added, removed]] of sides) {
    if (status && status.get(path) === "A") continue;
    const gone = new Map();
    for (const line of removed) { const m = _DEF_TEST_LINE_REMOVED.exec(line); if (m) gone.set(m[1], (gone.get(m[1]) || 0) + 1); }
    if (!gone.size) continue;
    const fresh = new Map();
    for (const line of added) { const m = _DEF_TEST_LINE.exec(line); if (m) fresh.set(m[1], (fresh.get(m[1]) || 0) + 1); }
    for (const [name, k] of fresh) n += Math.min(k, gone.get(name) || 0);
  }
  return n;
}

function _definitionOnlyChanged(name, sides, status) {
  // Some file both adds and removes a definition of `name`, and no file adds more definitions of it
  // than it removes (a file whose status is `A` removes none). Counted per file, one to one. The
  // added side reads the pattern `hit` reads (NOTE_path2_fifth_pass, V-1).
  const rxAdded = _symbolDefLineAdded(name), rxRemoved = _symbolDefLine(name);
  let paired = false;
  for (const [path, [added, removed]] of (sides || new Map())) {
    const a = added.filter(line => rxAdded.test(line)).length;
    const r = (status && status.get(path) === "A") ? 0 : removed.filter(line => rxRemoved.test(line)).length;
    if (a > r) return false;
    if (a && r) paired = true;
  }
  return paired;
}

// COMPAT-1: which public top-level definitions the diff removed without re-defining, per language.
// COMPAT-2 (PREREG_compat2_surface_and_panel_2026_09_16): the surface split, the signature reading and
// the candidate flag, mirrored from the Python; the licence flag is false and the verdict stays UNCHECKABLE.
const COMPAT2_LICENSED = false;
const _COMPAT_VERDICTS = COMPAT2_LICENSED ? ["UNCHECKABLE", "CONTRADICTED"] : ["UNCHECKABLE"];
const _COMPAT_SCAFFOLD = /(?:^|\/)(?:tests?|testing|specs?|__tests__|examples?|samples?|demos?|docs?|scripts?|tools?|bench|benchmarks?|fixtures?|internal|_internal|private|vendor|third_party|migrations?|cmd|e2e|integration|mocks?|stories|storybook|playground|sandbox|experiments?|dev|build)\/|(?:^|\/)(?:test_[^/]*|[^/]*_test\.(?:go|py)|[^/]*\.(?:test|spec)\.[^/]+|conftest\.py|setup\.py)$/;
// NOTE_path2_fifth_pass_2026_09_25 (V-2): the Python's patterns, with its `\s`, `\w` and `\b` spelled
// out (`_PY_WS`, `_PY_W`, `_PY_B`, `u` flag). JavaScript's own classes read U+001C-U+001F and U+0085 as
// non-space, U+FEFF as space, and every non-ASCII letter as a name's end, and F-2 hands both ports whole
// lines that can hold any of them.
const _PS = _PY_WS, _PW = _PY_W;
const _COMPAT_LANGS = [
  // [language, suffixes, regex over ONE removed line with a `name` group]
  ["python", [".py"], new RegExp(`^(?:async${_PS}+)?(?:def|class)${_PS}+(?<name>[A-Za-z]${_PW}*)`, "u")],
  ["js/ts", [".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".mts", ".cts"],
    new RegExp(`^export${_PS}+(?:default${_PS}+)?(?:async${_PS}+)?(?:function\\*?|class|const|let|var|interface|type|enum)${_PS}+(?<name>[A-Za-z_$]${_PW}*)`, "u")],
  ["go", [".go"], new RegExp(`^(?:func${_PS}+(?:\\([^)]*\\)${_PS}*)?|type${_PS}+)(?<name>[A-Z]${_PW}*)${_PY_B}`, "u")],
  ["rust", [".rs"], new RegExp(`^${_PS}*pub${_PS}+(?:async${_PS}+)?(?:fn|struct|enum|trait|type|const|static)${_PS}+(?<name>[A-Za-z_]${_PW}*)`, "u")],
  ["java", [".java", ".kt"], new RegExp(`^${_PS}*public${_PS}+(?:static${_PS}+|final${_PS}+|abstract${_PS}+)*[${_PY_W_CHARS}<>\\[\\],${_PY_WS_CHARS}]+?${_PS}+(?<name>[a-zA-Z_]${_PW}*)${_PS}*\\(`, "u")],
];
const _COMPAT_MAX_NAMED = 5;

const _reEscape = s => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

// BIN-1 (issue #118): a `diff --git` header with no `---`/`+++` pair — a binary change, a mode-only
// change, a pure rename — registers its file: A on `new file mode` / `Binary files /dev/null and …`,
// D on `deleted file mode` / `… and /dev/null differ`, else M. Files with hunks read as before.
// NOTE_path2_fifth_pass (V-2): `[^\n]` is Python's `.`; JavaScript's `.` also stops at \r, U+2028 and U+2029,
// which F-2 leaves inside a header line.
const _DIFF_GIT = /^diff --git (?:"a\/(?<qa>(?:[^"\\]|\\[^\n])*)"|a\/(?<a>[^\n]*?)) (?:"b\/(?<qb>(?:[^"\\]|\\[^\n])*)"|b\/(?<b>[^\n]*))$/;
const _BINARY_LINE = /^Binary files (?<a>[^\n]+?) and (?<b>[^\n]+?) differ$/;

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
      oldPath = _pyStrip(line.slice(4));          // str.strip(), not trim() (V-2)
      cur = null;
    } else if (line.startsWith("+++ ")) {
      const nw = _pyStrip(line.slice(4));
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

// NOTE_path2_fifth_pass (V-2): re.sub(r"\s+", " ", ...).strip(), with Python's \s.
const _PY_WS_RUN = new RegExp(_PY_WS + "+", "g");
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
      if (depth === 0) return _pyStrip(line.slice(k + 1, e).replace(_PY_WS_RUN, " "));
    }
  }
  return _pyStrip(line.slice(k + 1).replace(_PY_WS_RUN, " ")) + "…";
}

function _nameEnd(m) {
  // Python's m.end("name"): every language regex ends at the name except Java's, which goes on to `(`.
  return m.index + m[0].lastIndexOf(m.groups.name) + m.groups.name.length;
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
  const changed = [];
  const seen = new Set();
  const seenChanged = new Set();
  for (const [path, [, removed]] of sides) {
    for (const [lang, sufs, rx] of _COMPAT_LANGS) {
      if (!sufs.some(s => _undotted(path).endsWith(s))) continue;
      const addedLines = byLangAdded.get(lang) || [];
      const ablob = addedLines.join("\n");
      for (const line of removed) {
        const m = rx.exec(line);
        if (!m) continue;
        const name = m.groups.name;
        if (name.startsWith("_")) continue;                              // private by convention
        const key = path + " " + lang + " " + name;
        if (new RegExp(_PY_B + _reEscape(name) + _PY_B, "u").test(ablob)) {      // Python's \b (V-2)
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
        // AMENDMENT_path2 C-2: the scaffold test reads the undotted key, as the Python does, so
        // `.storybook/` stays scaffolding after #121 gave the key its dots back.
        if (!seen.has(key)) { seen.add(key); dropped.push([path, lang, name, !_COMPAT_SCAFFOLD.test(_undotted(path))]); }
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
// NOTE_path2_third_pass_2026_09_25 (R-3): a prefix key opening with two dots (`..`, `...`,
// `../docs`) is relative or elided notation, not a repo path. Git never emits a changed path that
// starts with `../`, so every changed path is outside it; the gate abstains instead of accusing.
const _DOTFILE_PREFIX = /^\.[^./\\]/;
function _prefixOffTree(pref) {
  return pref.startsWith(".") && !_DOTFILE_PREFIX.test(pref);
}
// NOTE_path2_fourth_pass F-4: whether SOME reading of an off-tree prefix could hold `path` -- its named
// segments (undotted, dots-only segments dropped) occur contiguously among the path's undotted segments.
// NOTE_path2_fifth_pass V-4: the prefix as written, before rstrip("/."), when it ends in a `..` segment
// (trailing "/" and "." segments dropped), else "". Such a prefix is off-tree and could hold anything.
function _parentPrefix(raw) {
  const segs = raw.split("/");
  while (segs.length && (segs[segs.length - 1] === "" || segs[segs.length - 1] === ".")) segs.pop();
  return segs.length && segs[segs.length - 1] === ".." ? segs.join("/") : "";
}
// V-4 as well: a `..` or `...` segment AFTER a named one (`../src/../docs`) could hold anything.
function _couldLieUnder(path, pref, raw = "") {
  if (_parentPrefix(raw)) return true;
  let named = false;
  for (const seg of pref.split("/")) {
    if (_stripChars(seg, ".")) named = true;
    else if (named && seg !== "" && seg !== ".") return true;
  }
  const want = pref.split("/").filter(seg => _stripChars(seg, ".")).map(seg => _stripChars(seg, ".", true, false));
  const have = path.split("/").map(seg => _stripChars(seg, ".", true, false));
  if (!want.length) return true;
  for (let i = 0; i + want.length <= have.length; i++) {
    if (want.every((w, k) => have[i + k] === w)) return true;
  }
  return false;
}

// AMENDMENT_path2 C-3: `path` lies outside every prefix only by a dot the prose left off. The
// containment test is PATH-1's _pathInside, the same one that decided `outside`.
function _dotMiss(path, prefs) {
  if (!path.startsWith(".") || path.startsWith("..")) return false;
  const rest = path.slice(1);
  return prefs.some(x => !x.startsWith(".") && _pathInside(rest, x));
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
  // The three line endings a diff can carry, and nothing else. NOTE_path2_fourth_pass F-2: the Python
  // (`_diff_lines`) now splits exactly this way; str.splitlines() also broke on U+000B, U+000C,
  // U+001C-U+001E, U+0085, U+2028 and U+2029, and the two ports disagreed on `tests_added`.
  const lines = text.split(/\r\n|\r|\n/);
  if (lines.length && lines[lines.length - 1] === "") lines.pop();
  return lines;
}

// repr() of a str, for the `why` strings the Python builds with !r.
// NOTE_path2_fifth_pass (V-2): a non-ASCII character str.isprintable() refuses (categories C and Z,
// other than the space) is escaped as \xhh, \uhhhh or \Uhhhhhhhh, as Python's repr() does; a reason
// that prints a path or a name holding U+2028, U+0085 or U+00A0 then reads the same in both ports.
const _PY_UNPRINTABLE = /[\p{C}\p{Z}]/u;
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
    else if (c > 0x7f && _PY_UNPRINTABLE.test(ch)) {
      out += c <= 0xff ? "\\x" + c.toString(16).padStart(2, "0")
        : c <= 0xffff ? "\\u" + c.toString(16).padStart(4, "0") : "\\U" + c.toString(16).padStart(8, "0");
    }
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
      oldPath = _pyStrip(line.slice(4));          // str.strip(), not trim() (V-2)
    } else if (line.startsWith("+++ ")) {
      const nw = _pyStrip(line.slice(4));
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

function gateDiffText(summaryText, diffText, { strict = false, _declared = false } = {}) {
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
            const got = (addedBlob.match(_GOT_TEST_LINE) || []).length;
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
            // NOTE_path2_fifth_pass V-1: the added-side pairing pattern, one line at a time.
            const hit = _symbolHit(d.name, addedBlob);
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
          // PATH-1 mode 1: _pathInside matches a bare filename on its basename.
          const outside = [...status.keys()].filter(p => !prefs.some(x => _pathInside(p, x)));
          // PATH-2 (#121), AMENDMENT_path2 C-3: dot misses alone abstain; any real outside path accuses,
          // and only real paths are listed. _dotMiss uses the same _pathInside containment.
          const dotMiss = outside.filter(p => _dotMiss(p, prefs));
          let real = outside.filter(p => !dotMiss.includes(p));
          // NOTE_path2_fifth_pass V-4: off-tree-ness is also read on the prefix as written, before the
          // rstrip("/."), so a prefix ending in `..` is off-tree (_parentPrefix).
          const written = rawPrefs.map(x => _norm(x));
          const offPairs = prefs.map((x, i) => [x, written[i]]).filter(([x, r]) => _prefixOffTree(x) || _parentPrefix(r));
          const offTree = offPairs.map(([x, r]) => _parentPrefix(r) || x);
          // NOTE_path2_fourth_pass F-4: beside an on-tree prefix, a real path no reading of the off-tree
          // prefix could hold still accuses, and only such paths are listed; otherwise R-3 abstains.
          const besideOnTree = offTree.length > 0 && offTree.length < prefs.length;
          if (besideOnTree) real = real.filter(p => !offPairs.some(([x, r]) => _couldLieUnder(p, x, r)));
          if (noPaths) { c.verdict = "UNCHECKABLE"; c.why = noPaths; }
          else if (notPaths.length) { c.verdict = "UNCHECKABLE"; c.why = `prefix ${pyRepr(notPaths[0])} is not a path (#110)`; }
          else if (offTree.length && !(besideOnTree && real.length)) {   // R-3, narrowed by F-4
            c.verdict = "UNCHECKABLE";
            c.why = `prefix ${pyRepr(offTree[0])} is relative to a directory the diff does not name (#121)`;
          }
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
  // DECLARE-1: the prose pass above is finished and is not changed by any of this. The recursion
  // terminates in one step: synthesized text never contains a styxx fence.
  if (!_declared) {
    const [dtext, drep] = declarationPass(summaryText);
    if (drep.declared) {
      if (dtext) {
        const sub = gateDiffText(dtext, diffText, { strict, _declared: true });
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