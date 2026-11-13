"use strict";
/*
 * Loads the BUILT page and hands back what a browser would run: the embedded checker's text
 * (the raw text of <script type="text/plain" id="checker-src">) and the page's own script (the raw
 * text of <script id="page-main">), cut out by the HTML rule for script elements (raw text up to
 * the first "</script", in any case). The page script is evaluated in this realm with no
 * `document`, so only its core (ReceiptV0Page) is defined; the browser-only UI part is skipped.
 *
 * nodeEnv() is the stand-in for the page's browserEnv(): it starts a Node worker_threads Worker
 * per run from the source the page's own buildWorkerSource() made (see worker_bridge.js).
 * Lab-written test code; no network, no npm.
 */
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const crypto = require("node:crypto");
const { Worker } = require("node:worker_threads");

const BRIDGE = path.join(__dirname, "worker_bridge.js");

function rawScript(html, openTag) {
  const i = html.indexOf(openTag);
  if (i < 0) throw new Error("no " + openTag + " in the page");
  if (html.indexOf(openTag, i + 1) >= 0) throw new Error("two " + openTag + " in the page");
  const start = i + openTag.length;
  const end = html.toLowerCase().indexOf("</script", start);
  if (end < 0) throw new Error("unterminated " + openTag);
  return html.slice(start, end);
}

const sha256hex = (b) => crypto.createHash("sha256").update(b).digest("hex");

function loadPage(pagePath) {
  const bytes = fs.readFileSync(pagePath);
  const html = bytes.toString("latin1");          // byte-preserving; the page is checked to be ASCII
  for (const x of bytes) if (x >= 0x80) throw new Error("the page is not ASCII");
  const checkerText = rawScript(html, '<script type="text/plain" id="checker-src">');
  const mainText = rawScript(html, '<script id="page-main">');
  const styleStart = html.indexOf("<style>") + 7;
  const styleText = html.slice(styleStart, html.indexOf("</style>", styleStart));
  const meta = /<meta http-equiv="Content-Security-Policy" content="([^"]*)">/.exec(html);
  const csp = meta ? meta[1].replace(/&#x27;/g, "'").replace(/&quot;/g, '"').replace(/&amp;/g, "&") : null;
  const recorded = /<meta name="receipt-v0-checker-sha256" content="([0-9a-f]{64})">/.exec(html);
  // evaluate the page's own script as a classic script in this realm, with no document
  if (typeof globalThis.document !== "undefined") throw new Error("a document exists in this realm");
  vm.runInThisContext(mainText, { filename: "page-main.js" });
  const P = globalThis.ReceiptV0Page;
  return { bytes, html, checkerText, mainText, styleText, csp, recordedMeta: recorded ? recorded[1] : null, P,
    pageSha256: sha256hex(bytes) };
}

class NodeWorkerShim {
  constructor(source, log) {
    this.onmessage = null; this.onerror = null; this.onmessageerror = null;
    this.exited = false;
    this.t = new Worker(BRIDGE, { workerData: { source } });
    if (log) log.started += 1;
    this.t.on("message", (m) => { if (this.onmessage) this.onmessage({ data: m }); });
    this.t.on("error", (e) => { if (this.onerror) this.onerror({ message: String((e && e.message) || e) }); });
    this.t.on("messageerror", () => { if (this.onmessageerror) this.onmessageerror({}); });
    this.t.on("exit", () => { this.exited = true; if (log) log.exited += 1; });
  }
  postMessage(m) { this.t.postMessage(m); }
  terminate() { return this.t.terminate(); }
}

function nodeEnv(log, makeSource) {
  return {
    last: null,
    makeWorker(source) {
      this.last = new NodeWorkerShim(makeSource ? makeSource(source) : source, log);
      return this.last;
    },
    release() {},
    setTimer: (f, ms) => setTimeout(f, ms),
    clearTimer: (h) => clearTimeout(h),
    now: () => performance.now(),
  };
}

/* A stand-in document for renderOutcome: it allows only what the page's rule allows (create
 * div/p/span/code/pre/button, set className and textContent, append, a fixed attribute set with
 * fixed values, click listeners) and throws on anything else, innerHTML included. */
function fakeDocument() {
  const TAGS = ["div", "p", "span", "code", "pre", "button"];
  const ATTRS = { type: ["button"] };
  function make(tag) {
    const node = { tag, className: "", _text: "", children: [], listeners: {} };
    const api = {
      appendChild(c) { node.children.push(c); return c; },
      insertBefore(c) { node.children.unshift(c); return c; },
      setAttribute(n, v) {
        if (!Object.prototype.hasOwnProperty.call(ATTRS, n) || ATTRS[n].indexOf(v) < 0) throw new Error("attribute " + n + "=" + v);
      },
      addEventListener(t, f) { if (t !== "click") throw new Error("listener " + t); node.listeners[t] = f; },
    };
    const p = new Proxy(node, {
      get(t, k) {
        if (k === "textContent") return t._text + t.children.map((c) => c.textContent).join("");
        if (k === "__node") return t;
        if (Object.prototype.hasOwnProperty.call(api, k)) return api[k];
        if (k === "className" || k === "children" || k === "tag") return t[k];
        throw new Error("read of " + String(k));
      },
      set(t, k, v) {
        if (k === "textContent") { t._text = String(v); t.children = []; return true; }
        if (k === "className") { t.className = String(v); return true; }
        throw new Error("set of " + String(k));
      },
    });
    return p;
  }
  return {
    createElement(tag) { if (TAGS.indexOf(tag) < 0) throw new Error("element " + tag); return make(tag); },
    root() { return make("div"); },
  };
}

function walk(n, f) { f(n); for (const c of n.children) walk(c, f); }

module.exports = { loadPage, nodeEnv, NodeWorkerShim, fakeDocument, walk, sha256hex, rawScript };
