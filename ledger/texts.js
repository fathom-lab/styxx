/* texts.js — the released-texts route of the ledger page (THE_PLAN 1c, page side). Runs in a browser (the page
   inlines it when the builder runs with --public-texts) and under node (`node test_texts.cjs`). Standard library
   only; the browser half uses verify.js (window.LedgerVerify) for sha256 and the anchor check.

   Pure functions, tested under node:
     normalize(text)                 unicode NFC; markdown markers * ` # > [ ] ( ) removed, '_' removed at a token's
                                     edge only (snake_case survives), the gate trailer form removed, whitespace
                                     collapsed to one space, trimmed. Case is kept.
     similarity(a, b)                token-level Dice coefficient over the normalized tokens (multisets):
                                     2 * shared / (|a| + |b|), in [0, 1]. 0 when either side has no token.
     nearest(paste, textsMap, thr)   every published text scored against the paste; the best one is a "match" only
                                     when its score reaches thr (default 0.6). A match here is NOT a hash match.
     wordDiff(a, b)                  word-level (whitespace tokens of the raw strings) longest-common-subsequence
                                     diff: [{op: 'same'|'del'|'ins', text}], del = only in a, ins = only in b.

   Browser glue (needs a document): page(dataEl) reads the embedded index + texts; mount(ctx) adds the tile, the
   caveat, the list; afterProve(ctx) extends the prove-a-message box. Absent the data element the page renders
   nothing of this: the feature is off.

   What a released text shows: the bytes the gate hashed as sent_sha256, re-hashed in this browser. What it does
   not show: who typed them, that they are true, or that a text with no match was never sent. A copy rendered by
   telegram need not hash to those bytes; a near match proves nothing by itself. */
(function (root, factory) {
  if (typeof module === "object" && module && module.exports) module.exports = factory();
  else root.LedgerTexts = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";
  var THRESHOLD = 0.6;
  var MAX_DIFF_TOKENS = 1500;
  var TRAILER_RE = /\[gate: \d+ of \d+ numerals bound (?:·|\.|-) ledger \d+\]/g;
  var MARKDOWN_RE = /[*`#>\[\]()]/g;
  // '_' is a marker only at a token's edge (_word_), never inside an identifier (sent_sha256 keeps its underscore)
  var UNDERSCORE_LEAD_RE = /(^|[^\p{L}\p{N}])_+/gu;
  var UNDERSCORE_TAIL_RE = /_+(?=[^\p{L}\p{N}]|$)/gu;

  // ---- normalization and similarity
  function normalize(text) {
    if (text === null || text === undefined) return "";
    var s = String(text);
    if (s.normalize) s = s.normalize("NFC");
    return s.replace(/\r\n?/g, "\n").replace(TRAILER_RE, " ").replace(MARKDOWN_RE, " ")
      .replace(UNDERSCORE_LEAD_RE, "$1 ").replace(UNDERSCORE_TAIL_RE, " ").replace(/\s+/g, " ").replace(/^ | $/g, "");
  }
  function tokensOf(text) {
    var n = normalize(text);
    return n ? n.split(" ") : [];
  }
  function countsOf(ts) {
    var m = Object.create(null);
    for (var i = 0; i < ts.length; i++) m[ts[i]] = (m[ts[i]] || 0) + 1;
    return m;
  }
  function similarity(a, b) {
    var ta = tokensOf(a), tb = tokensOf(b);
    if (!ta.length || !tb.length) return 0;
    var ca = countsOf(ta), cb = countsOf(tb), shared = 0;
    for (var k in ca) if (cb[k]) shared += Math.min(ca[k], cb[k]);
    return (2 * shared) / (ta.length + tb.length);
  }
  function nearest(paste, textsMap, threshold) {
    var thr = typeof threshold === "number" ? threshold : THRESHOLD;
    var out = { threshold: thr, paste_tokens: tokensOf(paste).length, candidates: [], best: null, match: null };
    if (!out.paste_tokens || !textsMap) return out;
    var shas = Object.keys(textsMap);
    for (var i = 0; i < shas.length; i++) {
      out.candidates.push({ sha: shas[i], score: similarity(paste, textsMap[shas[i]]) });
    }
    out.candidates.sort(function (x, y) { return y.score - x.score || (x.sha < y.sha ? -1 : x.sha > y.sha ? 1 : 0); });
    out.best = out.candidates.length ? out.candidates[0] : null;
    out.match = out.best && out.best.score >= thr ? out.best : null;
    return out;
  }

  // ---- word-level diff (LCS over whitespace tokens of the raw strings)
  function rawTokens(s) {
    var t = String(s === null || s === undefined ? "" : s).replace(/^\s+|\s+$/g, "");
    return t ? t.split(/\s+/) : [];
  }
  function wordDiff(a, b) {
    var ta = rawTokens(a), tb = rawTokens(b), truncated = false;
    if (ta.length > MAX_DIFF_TOKENS) { ta = ta.slice(0, MAX_DIFF_TOKENS); truncated = true; }
    if (tb.length > MAX_DIFF_TOKENS) { tb = tb.slice(0, MAX_DIFF_TOKENS); truncated = true; }
    var n = ta.length, m = tb.length, i, j;
    // L[i][j] = LCS length of ta[i..] and tb[j..]
    var L = new Array(n + 1);
    for (i = 0; i <= n; i++) { L[i] = new Int32Array(m + 1); }
    for (i = n - 1; i >= 0; i--) {
      for (j = m - 1; j >= 0; j--) {
        L[i][j] = ta[i] === tb[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
      }
    }
    var ops = [];
    function push(op, tok) {
      var last = ops.length ? ops[ops.length - 1] : null;
      if (last && last.op === op) last.text += " " + tok; else ops.push({ op: op, text: tok });
    }
    i = 0; j = 0;
    while (i < n && j < m) {
      if (ta[i] === tb[j]) { push("same", ta[i]); i++; j++; }
      else if (L[i + 1][j] >= L[i][j + 1]) { push("del", ta[i]); i++; }
      else { push("ins", tb[j]); j++; }
    }
    while (i < n) { push("del", ta[i]); i++; }
    while (j < m) { push("ins", tb[j]); j++; }
    if (truncated) ops.push({ op: "same", text: "(diff cut at " + MAX_DIFF_TOKENS + " words)" });
    return ops;
  }

  // ---- browser glue
  function esc(s) {
    return String(s === null || s === undefined ? "" : s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;" }[c];
    });
  }
  function fmt(x) { return (Math.round(x * 100) / 100).toFixed(2); }
  function seqList(xs) { return xs.length ? xs.join(", ") : "none"; }
  function seqWord(xs) { return (xs.length === 1 ? "seq " : "seqs ") + seqList(xs); }
  var CAVEAT = "a telegram-rendered copy need not hash to the gate's bytes; the released bytes are what the gate hashed; a near match proves nothing by itself.";
  var PROVES = "a hash match says these exact bytes left the gate at that second under that verdict; it does not say who typed them, that they are true, or that a text with no match was never sent (the line may be withheld, off the allow-list, not re-derivable, or newer than this page).";
  var FILES_URL = "https://fathom-lab.github.io/styxx/ledger/public_texts/";

  function page(dataEl) {
    if (!dataEl) return null;
    var doc;
    try { doc = JSON.parse(dataEl.textContent); } catch (e) { return null; }
    if (!doc || doc.enabled !== true || !doc.index) return null;
    var idx = doc.index, texts = doc.texts || {}, counts = idx.counts || {}, off = idx.off_allow_list || {};
    var bySha = Object.create(null), bySeq = Object.create(null);
    (idx.published || []).forEach(function (p) { bySha[p.sent_sha256] = p; bySeq[p.seq] = p; });
    var withheldBySeq = Object.create(null); (idx.withheld || []).forEach(function (w) { withheldBySeq[w.seq] = w; });
    var unrecBySeq = Object.create(null); (idx.unreconstructable || []).forEach(function (u) { unrecBySeq[u.seq] = u; });
    var unlisted = off.unlisted_dest || [], noSent = off.no_sent_bytes || [], canary = off.canary_chat || 0;
    var offCount = canary + unlisted.length + noSent.length;
    var offDenom = counts.public_lines || 0;   // off the list, of all public send lines
    var allowDenom = counts.allow_listed_lines || 0;
    var withheldSeqs = (idx.withheld || []).map(function (w) { return w.seq; });
    var unrecSeqs = (idx.unreconstructable || []).map(function (u) { return u.seq; });
    var rehashHex = Object.create(null), rehashError = null, ctx = null, outEl = null, tileEl = null, rehashEl = null, noteEl = null;

    function byId(id) { return (typeof document !== "undefined" && document.getElementById) ? document.getElementById(id) : null; }
    function rehashState(s) { var h = rehashHex[s]; return h === undefined ? null : h === s; }
    function rehashNote() {
      var shas = Object.keys(texts), ok = 0, bad = [], pending = 0;
      shas.forEach(function (s) { var st = rehashState(s); if (st === true) ok++; else if (st === false) bad.push(s); else pending++; });
      if (rehashError) return "could not re-hash here: " + rehashError;
      if (pending) return "re-hashing in this browser…";
      return ok + " of " + shas.length + " published texts re-hash here to their sent_sha256" + (bad.length ? "; " + bad.length + " DO NOT (" + bad.map(function (s) { return s.slice(0, 16); }).join(", ") + ")" : "");
    }
    // the tile: one number with its denominator; the sentence lives in the note under the prove box
    function tileHtml() {
      return "<b>" + esc(counts.published || 0) + " of " + esc(allowDenom) + "</b><span>released texts published (allow-listed public lines) · <a href=\"#texts-list\">the list</a></span>";
    }
    // the full sentence, every number with its denominator
    function summaryText() {
      return "released texts: " + (counts.published || 0) + " of " + allowDenom + " allow-listed public lines published; " +
        (counts.withheld || 0) + " of " + allowDenom + " withheld by the text guard" + (withheldSeqs.length ? " (" + seqWord(withheldSeqs) + ")" : "") + "; " +
        (counts.unreconstructable || 0) + " of " + allowDenom + " not re-derivable" + (unrecSeqs.length ? " (" + seqWord(unrecSeqs) + ")" : "") + "; " +
        offCount + " of " + offDenom + " public send lines off the allow-list (" + canary + " to the canary chat, " + unlisted.length + " to a chat not in the gate's config" +
        (noSent.length ? ", " + noSent.length + " without released bytes" : "") + ").";
    }
    function renderTile() {
      var tiles = byId("tiles");
      if (tiles) {
        if (!tileEl) { tileEl = document.createElement("div"); tileEl.className = "tile"; tileEl.id = "texts-tile"; tiles.appendChild(tileEl); }
        tileEl.innerHTML = tileHtml();
      }
      if (rehashEl) rehashEl.textContent = rehashNote();
      else if (noteEl) noteEl.innerHTML = noteHtml();
    }
    function anchorLinesFor(p, prefix) {
      var pre = prefix || "", out = [];
      if (!ctx || !ctx.V) return out;
      var anc = (ctx.anchorLines || []).filter(function (a) { return a.date === p.day; })[0], stamp = ctx.stampByDate ? ctx.stampByDate[p.day] : null;
      if (anc) {
        var c = ctx.V.checkAnchor(ctx.R.entries, anc, ctx.withheld);
        out.push(pre + "day " + p.day + " is fixed by the posted head: gate anchor " + anc.date + " head " + anc.head.slice(0, 16) + "… seq " + anc.seq + " — " + (c.ok ? "in chain" : "NOT in chain") + (p.seq <= anc.seq ? "" : " (this line sits after that head: not covered by it)"));
        out.push(pre + "bitcoin: " + (stamp && stamp.ots ? (stamp.status === "bitcoin" ? "the post's bytes are attested in block " + stamp.block : "proof " + stamp.status + " at the calendars") : "not stamped"));
      } else out.push(pre + "day " + p.day + " has no posted head yet in this snapshot (it posts at 00:00 new york)");
      return out;
    }
    // mode "hit": the paste's digest IS this line's; the re-hash verdict is the page's own fact (class ok).
    // mode "nearest": the paste matched nothing; the released text is shown as a search aid, and every verdict line
    // names its subject (the released text, not the paste) and never wears the ok class.
    function textBlock(p, title, mode, pasteHash) {
      var s = p.sent_sha256, st = rehashState(s), nearest = mode === "nearest";
      var h = ["<b" + (nearest ? " class=warn" : "") + ">" + esc(title) + "</b>",
        esc("seq " + p.seq + " · " + p.day + " · " + (p.dest_role || "") + " · decision " + p.decision + " · " + p.bytes + " bytes · form " + p.form + (p.form === "carried_from_previous_build" ? "" : " (" + formGloss(p.form) + ")")),
        "sent_sha256 of the line       <code>" + esc(s) + "</code>",
        "sha256 computed here of the " + (nearest ? "released text below" : "text below") + " <code>" + esc(rehashHex[s] || (rehashError ? "(failed: " + rehashError + ")" : "(pending)")) + "</code>"];
      if (nearest) {
        h.push("<span class=warn>" + esc("the released text below re-hashes here to its own sent_sha256" + (st === true ? "" : st === false ? " — it DOES NOT: the embedded copy is not the released bytes" : " (pending)") + "; your paste hashes to " + (pasteHash ? pasteHash.slice(0, 16) + "…" : "another digest") + ", which no send line carries.") + "</span>");
        h.push("<a href=\"public_texts/" + esc(s) + ".txt\">public_texts/" + esc(s.slice(0, 16)) + "….txt</a> — the file, named by its own sha256");
        anchorLinesFor(p, "for the nearest line (seq " + p.seq + "), not for your paste: ").forEach(function (l) { h.push(esc(l)); });
        h.push("the released bytes of seq " + esc(p.seq) + ", exactly (not your paste):");
      } else {
        h.push(st === true ? "<span class=ok>re-hashed in this browser: the two digests above are equal — these are the released bytes</span>"
          : st === false ? "<span class=bad>re-hashed in this browser: the two digests above DIFFER — the embedded copy is not the released bytes</span>"
          : "<span class=dim>re-hash " + esc(rehashError ? "failed: " + rehashError : "pending") + "</span>");
        h.push("<a href=\"public_texts/" + esc(s) + ".txt\">public_texts/" + esc(s.slice(0, 16)) + "….txt</a> — the file, named by its own sha256");
        anchorLinesFor(p).forEach(function (l) { h.push(esc(l)); });
        h.push("the released bytes, exactly:");
      }
      h.push("<pre style=\"color:var(--fg);max-height:240px\">" + esc(texts[s]) + "</pre>");
      return h.join("\n");
    }
    function formGloss(form) {
      var base = String(form || "").replace(/\+trailer$/, ""), tr = /\+trailer$/.test(String(form || ""));
      var what = base === "checked" ? "the text the gate checked" : base === "draft" ? "the draft as handed to the gate" : base === "draft_file" ? "the draft the gate kept as .draft.md" : base === "carried_from_previous_build" ? "an earlier build's copy, re-checked" : base;
      return what + (tr ? ", with the gate trailer appended" : ", as written");
    }
    function whyNotPublished(e) {
      if (withheldBySeq[e.seq]) return "the released bytes of seq " + e.seq + " are withheld by the text guard (" + (withheldBySeq[e.seq].reasons || []).join("; ") + "); a person must read them before they are published.";
      if (unrecBySeq[e.seq]) return "the released bytes of seq " + e.seq + " do not re-derive from the gate's copies (" + unrecBySeq[e.seq].reason + "): not published.";
      if (unlisted.indexOf(e.seq) >= 0) return "seq " + e.seq + " went to a chat that is not in the gate's config: off the allow-list, hash only.";
      if (noSent.indexOf(e.seq) >= 0) return "seq " + e.seq + " has no released bytes (nothing was sent).";
      if (e.dest_class !== "public") return "seq " + e.seq + " is not a public line: hash only, by the ledger's promise.";
      if (e.dest_role === "canary_chat") return "seq " + e.seq + " went to the canary chat: off the allow-list, hash only.";
      return "no released text is published for seq " + e.seq + " in this snapshot.";
    }
    function diffHtml(ops) {
      return ops.map(function (o) {
        if (o.op === "same") return esc(o.text);
        if (o.op === "del") return "<span class=bad>[-" + esc(o.text) + "-]</span>";
        return "<span style=\"color:var(--ac)\">{+" + esc(o.text) + "+}</span>";
      }).join(" ");
    }
    function noteHtml() {
      return "<span id=\"texts-summary\">" + esc(summaryText()) + "</span> <span id=\"texts-rehash\">" + esc(rehashNote()) + "</span><br>" +
        "the allow-list is the gate's own config, which names the public group and the anchor channel as the destinations whose texts may be published. the canary chat is where the lab's canary probes go (planted test sentences that measure the gate): a test fixture, not a public audience, so its texts stay hash-only. a chat the operator has not named in the config is not presumed public, so its texts stay hash-only until it is named. " +
        "for each published line the exact bytes the gate hashed are under <a href=\"public_texts/index.json\">public_texts/</a>, each file named by its own sha256 = the line's sent_sha256; \"form\" names which copy re-derived: the text the gate checked, as written, or with the gate trailer appended. " +
        "on a hash match the page shows the released bytes, re-hashed in your browser, both digests printed. when no digest matches, the page looks for the nearest published text by token similarity (markdown markers and the gate trailer removed, whitespace collapsed; threshold " + esc(fmt(THRESHOLD)) + ") and shows a word-level diff: a search aid, not a match; a paste holding only part of a text can reach the threshold. " +
        esc(CAVEAT) + " " + esc(PROVES) + " the released texts are the agent's bytes as they left the gate, not statements of the lab; the publisher changes none of them.<br>" +
        "check a file against its name without this page:<br><code>curl -sO " + esc(FILES_URL) + "&lt;sent_sha256&gt;.txt; sha256sum &lt;sent_sha256&gt;.txt</code> — the name is the digest; the index is <a href=\"public_texts/index.json\">public_texts/index.json</a>; this page's text code is <a href=\"texts.js\">texts.js</a>, beside <a href=\"verify.js\">verify.js</a>.";
    }
    function notListItemText() {
      return "not a transcript either where texts are released: the lines still carry no text; beside them this page publishes the released bytes of " + (counts.published || 0) + " of " + allowDenom + " allow-listed public sends (the public group and the anchor channel) and of nothing else — no operator dm, no canary chat, no chat absent from the gate's config.";
    }
    function addNotListItem() {
      if (typeof document === "undefined" || !document.getElementsByTagName) return false;
      var h2s = document.getElementsByTagName("h2"), ul = null;
      for (var i = 0; i < h2s.length; i++) {
        if (String(h2s[i].textContent || "").trim() === "what this is not") { ul = h2s[i].nextElementSibling; while (ul && String(ul.tagName || "").toUpperCase() !== "UL") ul = ul.nextElementSibling; break; }
      }
      if (!ul) return false;
      var li = document.createElement("li"); li.id = "texts-not"; li.textContent = notListItemText(); ul.appendChild(li);
      return true;
    }

    function mount(c) {
      ctx = c;
      var pre = byId("prove-out");
      if (pre && pre.parentNode) {
        outEl = document.createElement("div"); outEl.id = "texts-out"; outEl.hidden = true;
        outEl.style.border = "1px solid var(--line)"; outEl.style.padding = "8px"; outEl.style.margin = "6px 0"; outEl.style.whiteSpace = "pre-wrap"; outEl.style.wordBreak = "break-word";
        pre.parentNode.insertBefore(outEl, pre.nextSibling);
        var note = document.createElement("p"); note.id = "texts-note"; note.className = "dim"; noteEl = note;
        note.innerHTML = noteHtml();
        outEl.parentNode.insertBefore(note, outEl.nextSibling);
        rehashEl = byId("texts-rehash");
        var det = document.createElement("details"); det.id = "texts-list";
        det.innerHTML = "<summary>" + esc(summaryText()) + "</summary>" + listHtml();
        note.parentNode.insertBefore(det, note.nextSibling);
      }
      addNotListItem();
      var btn = byId("prove-btn");
      if (btn) btn.addEventListener("click", function () { if (outEl) { outEl.hidden = true; outEl.innerHTML = ""; } });
      renderTile();
      var shas = Object.keys(texts);
      return Promise.all(shas.map(function (s) {
        return c.V.sha256Hex(texts[s]).then(function (h) { rehashHex[s] = h; });
      })).then(renderTile, function (err) { rehashError = err && err.message ? err.message : String(err); renderTile(); });
    }
    function listHtml() {
      var rows = ["<div class=wrap><table><tr><th>seq</th><th>day</th><th>role</th><th>decision</th><th>bytes</th><th>form</th><th>sent_sha256</th></tr>"];
      (idx.published || []).slice().sort(function (a, b) { return a.seq - b.seq; }).forEach(function (p) {
        rows.push("<tr><td>" + esc(p.seq) + "</td><td>" + esc(p.day) + "</td><td>" + esc(p.dest_role) + "</td><td>" + esc(p.decision) + "</td><td>" + esc(p.bytes) + "</td><td>" + esc(p.form) + "</td><td><a href=\"public_texts/" + esc(p.sent_sha256) + ".txt\"><code>" + esc(p.sent_sha256.slice(0, 16)) + "…</code></a></td></tr>");
      });
      rows.push("</table></div>");
      var w = idx.withheld || [], u = idx.unreconstructable || [];
      rows.push("<p class=dim>form: which of the gate's copies re-derived — checked (the text the gate checked, as written), checked+trailer (the same with the gate trailer appended), draft / draft_file (the draft the gate kept), carried_from_previous_build (an earlier build's copy, re-checked against sent_sha256).</p>");
      rows.push("<p class=dim>withheld by the text guard, " + esc(w.length) + " of " + esc(allowDenom) + ": " + (w.length ? esc(w.map(function (x) { return "seq " + x.seq + " (" + x.dest_role + ", " + x.day + ": " + (x.reasons || []).join("; ") + ")"; }).join("; ")) : "none") + ". a person reads a withheld text before it is published; the operator's readings so far: " + esc((idx.allowed_by_operator || []).length) + ".</p>");
      rows.push("<p class=dim>not re-derivable from the gate's copies, " + esc(u.length) + " of " + esc(allowDenom) + ": " + (u.length ? esc(u.map(function (x) { return "seq " + x.seq + " (" + x.reason + ")"; }).join("; ")) : "none") + ".</p>");
      rows.push("<p class=dim>off the allow-list, " + esc(offCount) + " of " + esc(offDenom) + " public send lines: " + esc(canary) + " to the canary chat (a test fixture, never published), " + esc(unlisted.length) + " to a chat not in the gate's config" + (unlisted.length ? " (" + esc(seqWord(unlisted)) + ")" : "") + (noSent.length ? ", " + esc(noSent.length) + " without released bytes (" + esc(seqWord(noSent)) + ")" : "") + ". these stay hash-only.</p>");
      rows.push("<p class=dim>texts are append-only: a text once published is not changed or removed by a later build unless the build records why" + (idx.rewrite_reason ? " — this build did: " + esc(idx.rewrite_reason) : "") + ". built " + esc(idx.built_at) + ".</p>");
      return rows.join("");
    }

    // the prove box: hits grouped by digest, so two lines sharing one sent_sha256 render the text once and name both
    function hitBlocks(hits) {
      var groups = [], byKey = Object.create(null);
      hits.forEach(function (h) {
        var e = h.e, key = typeof e.sent_sha256 === "string" ? e.sent_sha256 : "seq:" + e.seq;
        if (!byKey[key]) { byKey[key] = { key: key, seqs: [], entries: [] }; groups.push(byKey[key]); }
        if (byKey[key].seqs.indexOf(e.seq) < 0) { byKey[key].seqs.push(e.seq); byKey[key].entries.push(e); }
      });
      return groups.map(function (g) {
        var p = bySha[g.key];
        if (!p) return g.entries.map(function (e) { return "<b>" + esc("matched line seq " + e.seq) + "</b>\n" + esc(whyNotPublished(e)); }).join("\n\n");
        var others = g.entries.filter(function (e) { return e.seq !== p.seq; });
        var title = "released bytes of the matched line (seq " + p.seq + ")";
        var block = textBlock(p, title, "hit");
        if (others.length) {
          block = "<span class=warn>" + esc("the matched digest also sits on " + seqWord(others.map(function (e) { return e.seq; })) + ": the same bytes, a different line — " +
            others.map(whyNotPublished).join(" ") + " the text below is published for seq " + p.seq + " (an allow-listed line), not for " + seqWord(others.map(function (e) { return e.seq; })) + ".") + "</span>\n" + block;
        }
        return block;
      });
    }
    function nearestBlocks(text, hashes) {
      var html = [], nr = nearest(text, texts, THRESHOLD);
      var pasteHash = hashes && hashes["as pasted"] ? hashes["as pasted"] : null;
      if (nr.match) {
        var p2 = bySha[nr.match.sha];
        html.push("<b class=warn>nearest published released text — NOT a hash match</b>\n" + esc("token similarity " + fmt(nr.match.score) + " (threshold " + fmt(THRESHOLD) + "; markdown markers and the gate trailer removed, whitespace collapsed)" + (nr.candidates.length > 1 ? "; next nearest " + fmt(nr.candidates[1].score) : "") + ". a paste holding part of a text can reach the threshold; a near match proves nothing by itself."));
        html.push("word-level diff, your paste against the released bytes: <span class=bad>[-only in your paste-]</span> <span style=\"color:var(--ac)\">{+only in the released bytes+}</span>");
        html.push("<div style=\"border:1px solid var(--line);padding:6px;max-height:240px;overflow:auto\">" + diffHtml(wordDiff(text, texts[nr.match.sha])) + "</div>");
        html.push(textBlock(p2, "the nearest published text — NOT a hash match", "nearest", pasteHash));
        html.push("<span class=warn>" + esc(CAVEAT) + "</span>");
      } else if (nr.best) {
        html.push(esc("no published released text is within token similarity " + fmt(THRESHOLD) + " of this paste (nearest: seq " + (bySha[nr.best.sha] || {}).seq + " at " + fmt(nr.best.score) + ", of " + nr.candidates.length + " published texts). " + CAVEAT));
      } else if (nr.paste_tokens === 0) {
        html.push(esc("nothing to compare after normalization."));
      }
      return html;
    }
    function afterProve(c) {
      if (!outEl) return Promise.resolve();
      var html = (c.hits && c.hits.length) ? hitBlocks(c.hits) : nearestBlocks(c.text, c.hashes);
      if (html.length) { outEl.innerHTML = html.join("\n\n"); outEl.hidden = false; }
      return Promise.resolve();
    }

    return { index: idx, texts: texts, mount: mount, afterProve: afterProve, tileHtml: tileHtml, summaryText: summaryText,
             noteHtml: noteHtml, listHtml: listHtml, textBlock: textBlock, hitBlocks: hitBlocks, nearestBlocks: nearestBlocks,
             notListItemText: notListItemText, whyNotPublished: whyNotPublished, formGloss: formGloss,
             setRehash: function (s, hex) { rehashHex[s] = hex; } };
  }

  return { THRESHOLD: THRESHOLD, normalize: normalize, tokens: tokensOf, similarity: similarity, nearest: nearest, wordDiff: wordDiff, page: page };
});
