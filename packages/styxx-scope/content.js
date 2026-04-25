/**
 * styxx-scope content script — detects assistant responses on ChatGPT,
 * Claude, Gemini, Grok, and renders a cognometric readout badge.
 *
 * Strategy: poll every 1500ms for new assistant-message DOM nodes, tag
 * seen ones, classify their text via classifier.js, append a styled
 * badge. Idempotent — never double-tags a node.
 */

(function () {
  "use strict";

  const STATE_ATTR = "data-styxx-scoped";
  const RESPONSE_SELECTORS = [
    // ChatGPT
    '[data-message-author-role="assistant"]',
    // Claude.ai
    'div[data-testid^="message-"]',
    '[class*="font-claude-message"]',
    // Gemini
    'message-content[data-message-role="model"]',
    '.model-response-text',
    // Grok on x.com
    '[data-testid="grokMessage"]',
  ];

  function findResponseNodes(root) {
    const set = new Set();
    for (const sel of RESPONSE_SELECTORS) {
      try {
        root.querySelectorAll(sel).forEach(n => set.add(n));
      } catch (_) {}
    }
    return Array.from(set);
  }

  function extractText(node) {
    // Prefer innerText (respects hidden elements), fall back to textContent
    return (node.innerText || node.textContent || "").trim();
  }

  function colorForD(d) {
    if (d < 0.15) return "#00d97e";
    if (d < 0.30) return "#ff9f1a";
    return "#ff0033";
  }
  function colorForTrust(t) {
    if (t > 0.75) return "#00d97e";
    if (t > 0.50) return "#ff9f1a";
    return "#ff0033";
  }
  function colorForCategory(c) {
    const map = {
      reasoning: "#00d97e",
      retrieval: "#2ee6d6",
      confab: "#ff0033",
      sycophant: "#ffd600",
      refusal: "#3b7dff",
      adversarial: "#ff0033",
      creative: "#2ee6d6",
    };
    return map[c] || "#8a7a80";
  }
  function colorForFault(kind) {
    const map = {
      confabulation: "#ff0033",
      drift: "#ff9f1a",
      refusal: "#3b7dff",
      sycophant: "#ffd600",
      low_trust: "#ff0033",
      incoherence: "#ff9f1a",
      unverified_claims: "#ff9f1a",
      phase_transition: "#2ee6d6",
    };
    return map[kind] || "#8a7a80";
  }

  function renderBadge(reading) {
    const badge = document.createElement("div");
    badge.className = "styxx-badge";
    const d = reading.D;
    const tr = reading.trust;
    const cat = reading.category;

    const faultHTML = reading.faults.length
      ? `<div class="sx-faults">${reading.faults.slice(0, 3).map(f => {
          const c = colorForFault(f.kind);
          const safeReason = (f.reason || "").replace(/"/g, "'");
          return `<span class="sx-fault" title="${safeReason}" style="background:${c}22;color:${c};border:1px solid ${c}66">${f.kind.replace(/_/g, ' ')}</span>`;
        }).join("")}</div>`
      : "";

    badge.innerHTML = `
      <div class="sx-head">
        <span class="sx-brand">styxx</span>
        <span class="sx-cat" style="color:${colorForCategory(cat)};border-color:${colorForCategory(cat)}66">${cat}</span>
        <span class="sx-trust" style="color:${colorForTrust(tr)}" title="aggregate trust (tier 3 proxy)">${tr.toFixed(2)}</span>
      </div>
      <div class="sx-axes">
        <span class="sx-axis" title="K · reasoning depth"><i>K</i> ${reading.K.toFixed(2)}</span>
        <span class="sx-axis" title="C · coherence"><i>C</i> ${reading.C.toFixed(2)}</span>
        <span class="sx-axis" title="D · dissociation" style="color:${colorForD(d)}"><i>D</i> ${d.toFixed(2)}</span>
      </div>
      ${faultHTML}
      <div class="sx-foot">tier 3 · <a href="https://fathom.darkflobi.com/spec" target="_blank">spec v1.0</a></div>
    `;
    return badge;
  }

  function scopeNode(node) {
    if (node.getAttribute(STATE_ATTR) === "done") return;
    node.setAttribute(STATE_ATTR, "done");

    const text = extractText(node);
    if (!text || text.length < 20) return;

    // Classify — defer via requestIdleCallback for perf
    const run = () => {
      try {
        const reading = window.__styxxScope.classifyText(text);
        const badge = renderBadge(reading);
        // Insert the badge AFTER the response content, not inside it,
        // to avoid breaking the page's own event handlers.
        if (node.parentNode) {
          node.parentNode.insertBefore(badge, node.nextSibling);
        }
      } catch (e) {
        console.warn("[styxx-scope] classify error", e);
      }
    };

    if ("requestIdleCallback" in window) {
      window.requestIdleCallback(run, { timeout: 400 });
    } else {
      setTimeout(run, 50);
    }
  }

  function tick() {
    const nodes = findResponseNodes(document);
    for (const n of nodes) scopeNode(n);
  }

  // Initial poll after a short delay (let the site finish hydrating)
  setTimeout(tick, 1000);
  setInterval(tick, 1500);

  // Also re-scan on history changes (SPA navigation)
  let lastPath = location.pathname;
  setInterval(() => {
    if (location.pathname !== lastPath) {
      lastPath = location.pathname;
      setTimeout(tick, 800);
    }
  }, 600);

  console.log("[styxx-scope] active · fathom.darkflobi.com/spec · v0.1.0");
})();
