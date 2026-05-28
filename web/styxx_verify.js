/*
 * styxx_verify.js — standalone, zero-dependency, cross-language verifier for
 * styxx attestations. Runs in Node AND the browser. Imports nothing from styxx
 * and needs no network: a pure-JS SHA-256 is bundled below.
 *
 * It re-derives the PORTABLE content address (digest.portable, "sha256-jcs")
 * from the recorded substrate using the RFC 8785 / JCS canonical form, so a
 * styxx attestation can be verified in any language with zero install and zero
 * trust. The portable digest reproduces byte-for-byte against the Python library
 * (cross-validated in tests/test_portable_attestation.py).
 *
 * Scope (honest): structure only — the portable content address + the (hex-only,
 * already language-agnostic) Merkle chain. Claim verdicts (need the repo) and
 * vitals scores (need styxx's instruments) are reported NOT CHECKED, never
 * asserted. A fully re-sealed chain passes structure and is caught only against
 * an externally supplied expected head.
 *
 * Node CLI:   node web/styxx_verify.js path/to/artifact.json [expectedHead]
 * Browser:    styxxVerify.verify(JSON.parse(text))  ->  { ok, lines }
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.styxxVerify = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const CHAIN_GENESIS = "styxx-attestation-chain-v1";

  // ---- pure-JS SHA-256 (no deps, works in browser + Node) ------------------
  function sha256Hex(ascii) {
    function rotr(n, x) { return (x >>> n) | (x << (32 - n)); }
    const bytes = utf8Bytes(ascii);
    const h = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
               0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    const k = [
      0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
      0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
      0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
      0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
      0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
      0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
      0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
      0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2];
    const l = bytes.length;
    const withOne = bytes.concat([0x80]);
    while (withOne.length % 64 !== 56) withOne.push(0);
    const bitLenHi = Math.floor((l * 8) / 0x100000000);
    const bitLenLo = (l * 8) >>> 0;
    withOne.push((bitLenHi >>> 24) & 255, (bitLenHi >>> 16) & 255, (bitLenHi >>> 8) & 255, bitLenHi & 255);
    withOne.push((bitLenLo >>> 24) & 255, (bitLenLo >>> 16) & 255, (bitLenLo >>> 8) & 255, bitLenLo & 255);
    const w = new Array(64);
    for (let i = 0; i < withOne.length; i += 64) {
      for (let t = 0; t < 16; t++) {
        w[t] = (withOne[i + t * 4] << 24) | (withOne[i + t * 4 + 1] << 16) |
               (withOne[i + t * 4 + 2] << 8) | (withOne[i + t * 4 + 3]);
      }
      for (let t = 16; t < 64; t++) {
        const s0 = rotr(7, w[t - 15]) ^ rotr(18, w[t - 15]) ^ (w[t - 15] >>> 3);
        const s1 = rotr(17, w[t - 2]) ^ rotr(19, w[t - 2]) ^ (w[t - 2] >>> 10);
        w[t] = (w[t - 16] + s0 + w[t - 7] + s1) | 0;
      }
      let [a, b, c, d, e, f, g, hh] = h;
      for (let t = 0; t < 64; t++) {
        const S1 = rotr(6, e) ^ rotr(11, e) ^ rotr(25, e);
        const ch = (e & f) ^ (~e & g);
        const t1 = (hh + S1 + ch + k[t] + w[t]) | 0;
        const S0 = rotr(2, a) ^ rotr(13, a) ^ rotr(22, a);
        const maj = (a & b) ^ (a & c) ^ (b & c);
        const t2 = (S0 + maj) | 0;
        hh = g; g = f; f = e; e = (d + t1) | 0; d = c; c = b; b = a; a = (t1 + t2) | 0;
      }
      h[0]=(h[0]+a)|0; h[1]=(h[1]+b)|0; h[2]=(h[2]+c)|0; h[3]=(h[3]+d)|0;
      h[4]=(h[4]+e)|0; h[5]=(h[5]+f)|0; h[6]=(h[6]+g)|0; h[7]=(h[7]+hh)|0;
    }
    return h.map((x) => (x >>> 0).toString(16).padStart(8, "0")).join("");
  }

  function utf8Bytes(str) {
    const out = [];
    for (let i = 0; i < str.length; i++) {
      let c = str.charCodeAt(i);
      if (c < 0x80) out.push(c);
      else if (c < 0x800) out.push(0xc0 | (c >> 6), 0x80 | (c & 0x3f));
      else if (c >= 0xd800 && c <= 0xdbff) {
        const c2 = str.charCodeAt(++i);
        c = 0x10000 + ((c & 0x3ff) << 10) + (c2 & 0x3ff);
        out.push(0xf0 | (c >> 18), 0x80 | ((c >> 12) & 0x3f), 0x80 | ((c >> 6) & 0x3f), 0x80 | (c & 0x3f));
      } else out.push(0xe0 | (c >> 12), 0x80 | ((c >> 6) & 0x3f), 0x80 | (c & 0x3f));
    }
    return out;
  }

  // ---- RFC 8785 / JCS canonical serialization ------------------------------
  // ECMAScript Number::toString IS String(n); JSON.stringify gives JCS strings.
  function jcs(obj) {
    if (obj === null) return "null";
    const t = typeof obj;
    if (t === "boolean") return obj ? "true" : "false";
    if (t === "number") {
      if (!isFinite(obj)) throw new Error("portable digest is finite-numbers only");
      return String(obj);
    }
    if (t === "string") return JSON.stringify(obj);
    if (Array.isArray(obj)) return "[" + obj.map(jcs).join(",") + "]";
    if (t === "object") {
      const keys = Object.keys(obj).sort();
      return "{" + keys.map((k) => JSON.stringify(k) + ":" + jcs(obj[k])).join(",") + "}";
    }
    throw new Error("not JCS-serializable: " + t);
  }

  function portablePayload(artifact) {
    const core = {};
    for (const k of Object.keys(artifact)) {
      if (k !== "generated_at" && k !== "digest") core[k] = artifact[k];
    }
    return jcs(core);
  }

  function portableDigest(artifact) { return sha256Hex(portablePayload(artifact)); }
  function chainDigest(prev, attDigest) { return sha256Hex(prev + "|" + attDigest); }

  function semanticNotice(artifact, lines) {
    if (Array.isArray(artifact.claims) && artifact.claims.length)
      lines.push("  semantic (claim verdicts): NOT CHECKED — needs styxx + repo");
    if (artifact.vitals != null)
      lines.push("  semantic (vitals scores):  NOT CHECKED — needs styxx instruments");
  }

  function verifyAttestation(artifact, lines) {
    const node = (artifact.digest || {}).portable;
    if (!node || node.value == null) {
      lines.push("  portable digest: ABSENT — artifact predates the portable address");
      semanticNotice(artifact, lines);
      return false;
    }
    const ok = node.value === portableDigest(artifact);
    lines.push("  portable digest (sha256-jcs): " + (ok ? "OK" : "FAIL"));
    if (!ok) {
      lines.push("    recorded:   " + node.value);
      lines.push("    recomputed: " + portableDigest(artifact));
    }
    semanticNotice(artifact, lines);
    return ok;
  }

  function verifyChain(artifact, lines, expectedHead) {
    const links = artifact.links || [];
    let ok = true;
    let prev = CHAIN_GENESIS;
    for (let i = 0; i < links.length; i++) {
      const link = links[i];
      const att = link.attestation || {};
      const attP = portableDigest(att);
      const attFieldOk = link.attestation_portable_digest === attP;
      const expectLink = chainDigest(prev, attP);
      ok = ok && attFieldOk;
      lines.push("  link[" + i + "] portable att digest: " + (attFieldOk ? "OK" : "FAIL"));
      if (!attFieldOk) {
        lines.push("    recorded:   " + link.attestation_portable_digest);
        lines.push("    recomputed: " + attP);
      }
      semanticNotice(att, lines);
      prev = expectLink;
    }
    const head = artifact.head_chain_portable_digest;
    const headOk = head === (links.length ? prev : CHAIN_GENESIS);
    lines.push("  head_chain_portable_digest: " + (headOk ? "OK" : "FAIL"));
    ok = ok && headOk;
    if (expectedHead != null) {
      const anchorOk = head === expectedHead;
      lines.push("  expected-head anchor: " + (anchorOk ? "OK" : "FAIL"));
      ok = ok && anchorOk;
    } else {
      lines.push("  expected-head anchor: NOT PROVIDED — a re-sealed chain would " +
                 "pass structure; anchor to catch it");
    }
    return ok;
  }

  function verify(artifact, expectedHead) {
    const lines = [];
    const isChain = artifact && artifact.links && artifact.head_chain_portable_digest !== undefined;
    let ok;
    if (isChain) {
      lines.push("styxx attestation CHAIN — " + (artifact.links || []).length + " link(s) [portable]");
      ok = verifyChain(artifact, lines, expectedHead);
    } else {
      lines.push("styxx ATTESTATION [portable]");
      ok = verifyAttestation(artifact, lines);
    }
    lines.push("");
    lines.push("structural integrity (portable): " + (ok ? "OK" : "FAIL"));
    return { ok: ok, lines: lines };
  }

  // Node CLI
  if (typeof require === "function" && typeof module !== "undefined" && require.main === module) {
    const fs = require("fs");
    const path = process.argv[2];
    const expectedHead = process.argv[3] || null;
    if (!path) { console.error("usage: node styxx_verify.js <artifact.json> [expectedHead]"); process.exit(2); }
    let artifact;
    try { artifact = JSON.parse(fs.readFileSync(path, "utf8")); }
    catch (e) { console.error("error: cannot read/parse " + path + ": " + e.message); process.exit(2); }
    const r = verify(artifact, expectedHead);
    console.log(r.lines.join("\n"));
    process.exit(r.ok ? 0 : 1);
  }

  return {
    sha256Hex: sha256Hex, jcs: jcs, portableDigest: portableDigest,
    chainDigest: chainDigest, verify: verify,
    verifyAttestation: verifyAttestation, verifyChain: verifyChain,
    CHAIN_GENESIS: CHAIN_GENESIS,
  };
});
