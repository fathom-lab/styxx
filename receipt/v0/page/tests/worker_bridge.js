"use strict";
/*
 * Runs inside a Node worker_threads Worker and makes its global scope look, for the page's
 * worker source, like a browser's dedicated worker: `self` is the global object, `postMessage`
 * sends to the page side, and an assigned `onmessage` receives each message as {data}.
 * The source (built by the page's own buildWorkerSource) runs as a classic script in this realm
 * through vm.runInThisContext, so `module` and `require` are not visible to it, as in a browser.
 * Differences from a browser worker that remain: Node globals such as process and Buffer exist;
 * the page's Content-Security-Policy does not apply here.
 */
const { parentPort, workerData } = require("node:worker_threads");
const vm = require("node:vm");

globalThis.self = globalThis;
globalThis.postMessage = (m) => parentPort.postMessage(m);
if (typeof globalThis.module !== "undefined" || typeof globalThis.require !== "undefined") {
  throw new Error("module or require is a global in this worker; the source would take the Node branch");
}
vm.runInThisContext(workerData.source, { filename: "receipt-v0-worker.js" });
parentPort.on("message", (d) => {
  if (typeof globalThis.onmessage === "function") globalThis.onmessage({ data: d });
});
