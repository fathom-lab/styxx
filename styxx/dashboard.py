# -*- coding: utf-8 -*-
"""
styxx.dashboard — the live cognitive display.

    $ styxx dashboard
    [styxx] cognitive display at http://localhost:9800
    [styxx] watching ~/.styxx/chart.jsonl

Nobody has visualized AI cognitive state in real-time before. The
visual language doesn't exist yet. This is the first implementation.

Three displays, one page:

  1. The Cognitive Orbit — a 2D phase space where the agent's state
     moves between 6 category attractors in real-time. The SHAPE of
     cognition is visible: tight orbits = healthy, wandering = drift,
     spirals = warn cascade.

  2. The Pulse Strip — a scrolling EKG of confidence bars colored by
     category. Xendro's request: "the conversation IS the unit of
     cognition... I want the EKG."

  3. The Status Panel — live mood, gate rate, confidence, prescriptions.

Technical: stdlib http.server + SSE. Single HTML page, pure canvas +
DOM. No React, no npm, no build step. Zero dependencies beyond Python
stdlib.

0.9.5+. The display that makes AI cognition visible.
"""

from __future__ import annotations

import http.server
import json
import os
import socketserver
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional


def _audit_log_path() -> Path:
    data_dir = os.environ.get("STYXX_DATA_DIR", "").strip()
    if data_dir:
        return Path(data_dir).expanduser() / "chart.jsonl"
    return Path.home() / ".styxx" / "chart.jsonl"


def _load_recent(n: int = 50) -> List[dict]:
    """Load the last N entries from chart.jsonl."""
    path = _audit_log_path()
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        entries = []
        for line in lines[-n:]:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except (json.JSONDecodeError, ValueError):
                    pass
        return entries
    except OSError:
        return []


def _compute_status() -> dict:
    """Compute mood, condition, and prescriptions from the weather engine."""
    try:
        from .analytics import mood, streak
        from .weather import weather
        current_mood = mood(window_s=24 * 3600)
        current_streak = streak()
        report = weather(agent_name="styxx agent", window_hours=24.0)
        return {
            "mood": current_mood or "--",
            "condition": report.condition if report else "--",
            "prescriptions": list(report.prescriptions[:4]) if report else [],
            "streak": (
                f"{current_streak.length}x {current_streak.category}"
                if current_streak else "--"
            ),
        }
    except Exception:
        return {"mood": "--", "condition": "--", "prescriptions": [], "streak": "--"}


# ══════════════════════════════════════════════════════════════════
# SSE file watcher
# ══════════════════════════════════════════════════════════════════

class _FileWatcher:
    """Tails chart.jsonl and notifies SSE clients of new entries."""

    def __init__(self):
        self._clients: List = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def add_client(self, client):
        with self._lock:
            self._clients.append(client)

    def remove_client(self, client):
        with self._lock:
            try:
                self._clients.remove(client)
            except ValueError:
                pass

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _watch_loop(self):
        path = _audit_log_path()
        # Seek to end of file
        try:
            f = open(path, "r", encoding="utf-8")
            f.seek(0, 2)  # end of file
        except OSError:
            # File doesn't exist yet — wait for it
            while self._running and not path.exists():
                time.sleep(1.0)
            if not self._running:
                return
            f = open(path, "r", encoding="utf-8")
            f.seek(0, 2)

        while self._running:
            line = f.readline()
            if line:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        self._broadcast(entry)
                    except (json.JSONDecodeError, ValueError):
                        pass
            else:
                time.sleep(0.3)
        f.close()

    def _broadcast(self, entry: dict):
        data = json.dumps(entry)
        dead = []
        with self._lock:
            for client in self._clients:
                try:
                    client.write(f"data: {data}\n\n".encode("utf-8"))
                    client.flush()
                except Exception:
                    dead.append(client)
            for c in dead:
                try:
                    self._clients.remove(c)
                except ValueError:
                    pass


# Singleton watcher
_watcher = _FileWatcher()


# ══════════════════════════════════════════════════════════════════
# HTML template — the entire cognitive display
# ══════════════════════════════════════════════════════════════════

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>styxx cognitive display</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

:root {
  --bg-deep: #020108;
  --bg-panel: #06040e90;
  --border: #14102080;
  --text: #c0b8c0;
  --text-dim: #605060;
  --text-muted: #302830;
  --accent: #8040c0;
  --reasoning: #30e890;
  --refusal: #e0b830;
  --creative: #a050ff;
  --hallucination: #ff6030;
  --adversarial: #ff1840;
  --retrieval: #30a8ff;
}

body {
  background: var(--bg-deep);
  color: var(--text);
  font-family: 'Inter', -apple-system, sans-serif;
  height: 100vh;
  overflow: hidden;
  display: grid;
  grid-template-rows: 56px 1fr 160px;
  grid-template-columns: 1fr 300px;
}

/* ── Header ─────────────────────────────────────── */
header {
  grid-column: 1 / -1;
  background: linear-gradient(180deg, #0c081490, #08051080);
  border-bottom: 1px solid var(--border);
  padding: 0 2rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  backdrop-filter: blur(20px);
}
.brand-group {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.brand {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  background: linear-gradient(135deg, var(--accent), #ff6ec7);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.brand-sub {
  font-size: 0.5rem;
  color: var(--text-dim);
  letter-spacing: 0.06em;
  font-family: 'Inter', sans-serif;
  font-weight: 400;
}
.header-meta {
  font-size: 0.65rem;
  color: var(--text-dim);
  letter-spacing: 0.05em;
  font-family: 'JetBrains Mono', monospace;
}
.live {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.6rem;
  font-weight: 600;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--reasoning);
}
.live-dot {
  width: 8px; height: 8px;
  background: var(--reasoning);
  border-radius: 50%;
  box-shadow: 0 0 12px var(--reasoning), 0 0 30px var(--reasoning)40;
  animation: livePulse 2.5s ease-in-out infinite;
}
@keyframes livePulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.4; transform: scale(0.8); }
}

/* ── Orbit ──────────────────────────────────────── */
#orbit-container {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}
#orbit-canvas { display: block; }

/* ── Status Panel ───────────────────────────────── */
#status-panel {
  background: var(--bg-panel);
  border-left: 1px solid var(--border);
  padding: 1.2rem 1.4rem;
  overflow-y: auto;
  backdrop-filter: blur(12px);
}
.status-section {
  margin-bottom: 1.1rem;
  padding-bottom: 0.8rem;
  border-bottom: 1px solid #1a102808;
}
.status-section:last-child { border-bottom: none; }
.status-label {
  font-family: 'JetBrains Mono', monospace;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.14em;
  font-size: 0.5rem;
  margin-bottom: 0.35rem;
}
.status-value {
  font-size: 0.85rem;
  font-weight: 500;
  color: var(--text);
  transition: color 0.4s ease;
}
.status-value.good { color: var(--reasoning); }
.status-value.warn { color: var(--refusal); }
.status-value.bad { color: var(--adversarial); }
.status-value-sm {
  font-size: 0.7rem;
  font-weight: 400;
  color: var(--text);
}
.conf-bar-track {
  margin-top: 0.4rem;
  height: 4px;
  background: var(--border);
  border-radius: 2px;
  overflow: hidden;
}
.conf-bar-fill {
  height: 100%;
  border-radius: 2px;
  transition: width 0.6s ease, background 0.4s ease;
  box-shadow: 0 0 8px var(--reasoning)40;
}
.prescription-item {
  color: var(--text);
  font-size: 0.6rem;
  line-height: 1.5;
  margin-bottom: 0.5rem;
  padding: 0.5rem 0.7rem;
  background: #0c081430;
  border-radius: 4px;
  border-left: 3px solid var(--accent);
}
.prescription-item:first-child {
  color: #d8d0d4;
  background: #14102040;
  border-left-color: var(--reasoning);
}

/* ── Pulse Strip ────────────────────────────────── */
#pulse-container {
  grid-column: 1 / -1;
  background: linear-gradient(0deg, #04020a, var(--bg-panel));
  border-top: 1px solid var(--border);
  position: relative;
  overflow: hidden;
}
#pulse-canvas {
  display: block;
  width: 100%;
  height: 100%;
}
.pulse-label {
  position: absolute;
  top: 8px;
  left: 14px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.5rem;
  color: var(--text-muted);
  letter-spacing: 0.14em;
  text-transform: uppercase;
}

/* ── Connection ─────────────────────────────────── */
#connection-status {
  position: fixed;
  bottom: 140px;
  left: 14px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.45rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  transition: color 0.3s ease;
}
#connection-status.connected { color: var(--reasoning)60; }
#connection-status.disconnected { color: var(--adversarial)60; }

/* ── Fathom badge ───────────────────────────────── */
.fathom-badge {
  position: fixed;
  bottom: 142px;
  right: 14px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.42rem;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--text-muted);
}
.fathom-badge a {
  color: var(--text-dim);
  text-decoration: none;
  transition: color 0.3s;
}
.fathom-badge a:hover { color: var(--accent); }
</style>
</head>
<body>

<header>
  <div class="brand-group">
    <div class="brand">STYXX</div>
    <div class="brand-sub">real-time cognitive monitor for AI agents</div>
  </div>
  <div class="header-meta" id="header-info">waiting for data...</div>
  <div class="live"><div class="live-dot"></div> LIVE</div>
</header>

<div id="orbit-container">
  <canvas id="orbit-canvas"></canvas>
</div>

<div id="status-panel">
  <div class="status-section">
    <div class="status-label">condition</div>
    <div class="status-value" id="s-condition" style="font-size:0.75rem">--</div>
  </div>
  <div class="status-section" style="display:grid;grid-template-columns:1fr 1fr;gap:0.8rem">
    <div>
      <div class="status-label">gate pass</div>
      <div class="status-value" id="s-gate-rate">--</div>
      <div class="conf-bar-track"><div class="conf-bar-fill" id="gate-bar" style="width:0%;background:var(--reasoning)"></div></div>
    </div>
    <div>
      <div class="status-label">confidence</div>
      <div class="status-value" id="s-confidence">--</div>
      <div class="conf-bar-track"><div class="conf-bar-fill" id="conf-bar" style="width:0%;background:var(--reasoning)"></div></div>
    </div>
  </div>
  <div class="status-section" style="display:grid;grid-template-columns:1fr 1fr;gap:0.8rem">
    <div>
      <div class="status-label">classification</div>
      <div class="status-value" id="s-last-cat" style="font-size:0.8rem">--</div>
    </div>
    <div>
      <div class="status-label">streak</div>
      <div class="status-value-sm" id="s-streak">--</div>
    </div>
  </div>
  <div class="status-section">
    <div class="status-label">prescriptions</div>
    <div id="s-prescriptions"><div class="prescription-item" style="color:var(--text-muted)">awaiting data...</div></div>
  </div>
  <div class="status-section" style="display:grid;grid-template-columns:1fr 1fr;gap:0.8rem">
    <div>
      <div class="status-label">observations</div>
      <div class="status-value-sm" id="s-entries">0</div>
    </div>
    <div>
      <div class="status-label">session</div>
      <div class="status-value-sm" id="s-session" style="font-family:'JetBrains Mono',monospace;font-size:0.5rem;color:var(--text-dim);word-break:break-all">--</div>
    </div>
  </div>
</div>

<div id="pulse-container">
  <div class="pulse-label">cognitive pulse</div>
  <canvas id="pulse-canvas"></canvas>
</div>

<div id="connection-status">connecting...</div>
<div class="fathom-badge">a <a href="https://fathom.darkflobi.com">fathom lab</a> instrument</div>

<script>
// ══════════════════════════════════════════════════════════════
// Configuration
// ══════════════════════════════════════════════════════════════

const CATEGORIES = ['reasoning','refusal','creative','hallucination','adversarial','retrieval'];
const CAT_COLORS = {
  reasoning:     '#30e890',
  refusal:       '#e0b830',
  creative:      '#a050ff',
  hallucination: '#ff6030',
  adversarial:   '#ff1840',
  retrieval:     '#30a8ff',
};
const GATE_COLORS = { pass: '#30e890', warn: '#e0b830', fail: '#ff1840', pending: '#302830' };
const MAX_TRAIL = 5;
const MAX_PULSE = 120;

// Ripple effects — single ring on new data
const ripples = [];

// ══════════════════════════════════════════════════════════════
// State
// ══════════════════════════════════════════════════════════════

let entries = [];
let trail = [];
let currentPos = {x: 0, y: 0};
let targetPos = {x: 0, y: 0};
let currentColor = '#50e8a0';
let currentGate = 'pending';
let gateFlash = 0;
let frameCount = 0;
let currentConf = 0;

// ══════════════════════════════════════════════════════════════
// Canvas setup
// ══════════════════════════════════════════════════════════════

const orbitCanvas = document.getElementById('orbit-canvas');
const orbitCtx = orbitCanvas.getContext('2d');
const pulseCanvas = document.getElementById('pulse-canvas');
const pulseCtx = pulseCanvas.getContext('2d');

function resizeCanvases() {
  const oc = document.getElementById('orbit-container');
  const dpr = window.devicePixelRatio || 1;
  orbitCanvas.width = oc.clientWidth * dpr;
  orbitCanvas.height = oc.clientHeight * dpr;
  orbitCanvas.style.width = oc.clientWidth + 'px';
  orbitCanvas.style.height = oc.clientHeight + 'px';
  orbitCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const pc = document.getElementById('pulse-container');
  pulseCanvas.width = pc.clientWidth * dpr;
  pulseCanvas.height = pc.clientHeight * dpr;
  pulseCanvas.style.width = pc.clientWidth + 'px';
  pulseCanvas.style.height = pc.clientHeight + 'px';
  pulseCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function getOrbitSize() {
  const oc = document.getElementById('orbit-container');
  return { w: oc.clientWidth, h: oc.clientHeight };
}

function getCategoryPosition(cat, size) {
  const idx = CATEGORIES.indexOf(cat);
  if (idx < 0) return {x: size.w/2, y: size.h/2};
  const angle = (idx / 6) * Math.PI * 2 - Math.PI/2;
  const radius = Math.min(size.w, size.h) * 0.34;
  return {
    x: size.w/2 + Math.cos(angle) * radius,
    y: size.h/2 + Math.sin(angle) * radius,
  };
}

function getParticlePosition(cat, confidence, size) {
  const center = {x: size.w/2, y: size.h/2};
  const attractor = getCategoryPosition(cat, size);
  const t = Math.max(0.15, Math.min(0.92, confidence));
  return {
    x: center.x + (attractor.x - center.x) * t,
    y: center.y + (attractor.y - center.y) * t,
  };
}

// ══════════════════════════════════════════════════════════════
// Orbit renderer
// ══════════════════════════════════════════════════════════════

function drawOrbit() {
  frameCount++;
  const size = getOrbitSize();
  const w = size.w, h = size.h;
  const cx = w/2, cy = h/2;
  const orbitR = Math.min(w,h) * 0.34;
  const t = frameCount * 0.01;

  // Full clear — no afterglow buildup
  orbitCtx.clearRect(0, 0, w, h);

  // Ripple effects (single clean ring per event)
  for (let i = ripples.length - 1; i >= 0; i--) {
    const r = ripples[i];
    r.radius += 2;
    r.alpha -= 0.012;
    if (r.alpha <= 0) { ripples.splice(i, 1); continue; }
    orbitCtx.beginPath();
    orbitCtx.arc(r.x, r.y, r.radius, 0, Math.PI*2);
    orbitCtx.strokeStyle = r.color;
    orbitCtx.globalAlpha = r.alpha;
    orbitCtx.lineWidth = 1.5;
    orbitCtx.stroke();
  }
  orbitCtx.globalAlpha = 1;

  // Subtle spoke lines — center to each node
  for (let i = 0; i < 6; i++) {
    const p = getCategoryPosition(CATEGORIES[i], size);
    orbitCtx.beginPath();
    orbitCtx.moveTo(cx, cy);
    orbitCtx.lineTo(p.x, p.y);
    orbitCtx.strokeStyle = CAT_COLORS[CATEGORIES[i]];
    orbitCtx.globalAlpha = 0.06;
    orbitCtx.lineWidth = 1;
    orbitCtx.stroke();
  }
  orbitCtx.globalAlpha = 1;

  // Confidence rings (25%, 50%, 75%)
  for (const r of [0.25, 0.50, 0.75]) {
    orbitCtx.beginPath();
    orbitCtx.arc(cx, cy, orbitR * r, 0, Math.PI*2);
    orbitCtx.strokeStyle = '#ffffff';
    orbitCtx.globalAlpha = 0.03;
    orbitCtx.lineWidth = 1;
    orbitCtx.stroke();
  }
  orbitCtx.globalAlpha = 1;

  // Category nodes — clean and clear
  const activeCat = entries.length > 0 ? entries[entries.length-1].phase4_pred : null;
  for (let i = 0; i < 6; i++) {
    const cat = CATEGORIES[i];
    const p = getCategoryPosition(cat, size);
    const col = CAT_COLORS[cat];
    const isActive = cat === activeCat;
    const breathe = 0.85 + Math.sin(t * 1.5 + i * 1.1) * 0.15;

    // Glow — larger for active node
    const glowR = isActive ? 40 : 18;
    const g1 = orbitCtx.createRadialGradient(p.x, p.y, 0, p.x, p.y, glowR * breathe);
    g1.addColorStop(0, col + (isActive ? '35' : '15'));
    g1.addColorStop(1, col + '00');
    orbitCtx.beginPath();
    orbitCtx.arc(p.x, p.y, glowR * breathe, 0, Math.PI*2);
    orbitCtx.fillStyle = g1;
    orbitCtx.fill();

    // Ring
    if (isActive) {
      orbitCtx.beginPath();
      orbitCtx.arc(p.x, p.y, 12 * breathe, 0, Math.PI*2);
      orbitCtx.strokeStyle = col;
      orbitCtx.globalAlpha = 0.5;
      orbitCtx.lineWidth = 2;
      orbitCtx.stroke();
      orbitCtx.globalAlpha = 1;
    }

    // Dot
    orbitCtx.beginPath();
    orbitCtx.arc(p.x, p.y, isActive ? 6 : 3.5, 0, Math.PI*2);
    orbitCtx.fillStyle = col;
    orbitCtx.globalAlpha = isActive ? 1.0 : 0.45;
    if (isActive) { orbitCtx.shadowColor = col; orbitCtx.shadowBlur = 18; }
    orbitCtx.fill();
    orbitCtx.shadowBlur = 0;
    orbitCtx.globalAlpha = 1;

    // Label — LARGE and readable
    orbitCtx.fillStyle = col;
    orbitCtx.globalAlpha = isActive ? 1.0 : 0.45;
    orbitCtx.font = (isActive ? '700 11px' : '500 10px') + ' "JetBrains Mono", monospace';
    orbitCtx.textAlign = 'center';
    orbitCtx.fillText(cat.toUpperCase(), p.x, p.y + 24);
    orbitCtx.globalAlpha = 1;
  }

  // Trail — clean gradient line, last N positions only
  if (trail.length > 1) {
    for (let i = 1; i < trail.length; i++) {
      const a = trail[i-1], b = trail[i];
      const progress = i / trail.length;
      orbitCtx.beginPath();
      orbitCtx.moveTo(a.x, a.y);
      orbitCtx.lineTo(b.x, b.y);
      orbitCtx.strokeStyle = b.color;
      orbitCtx.globalAlpha = progress * 0.4;
      orbitCtx.lineWidth = 1 + progress * 2;
      orbitCtx.lineCap = 'round';
      orbitCtx.stroke();
    }
    // Final segment to current particle
    const last = trail[trail.length - 1];
    orbitCtx.beginPath();
    orbitCtx.moveTo(last.x, last.y);
    orbitCtx.lineTo(currentPos.x, currentPos.y);
    orbitCtx.strokeStyle = currentColor;
    orbitCtx.globalAlpha = 0.5;
    orbitCtx.lineWidth = 3;
    orbitCtx.lineCap = 'round';
    orbitCtx.stroke();
    orbitCtx.globalAlpha = 1;
  }

  // Animate position
  currentPos.x += (targetPos.x - currentPos.x) * 0.06;
  currentPos.y += (targetPos.y - currentPos.y) * 0.06;

  // Main particle — prominent, clean
  if (entries.length > 0) {
    const pb = 1 + Math.sin(t * 3) * 0.12;

    // Glow
    const g2 = orbitCtx.createRadialGradient(
      currentPos.x, currentPos.y, 0,
      currentPos.x, currentPos.y, 45 * pb
    );
    g2.addColorStop(0, currentColor + '20');
    g2.addColorStop(0.5, currentColor + '08');
    g2.addColorStop(1, currentColor + '00');
    orbitCtx.beginPath();
    orbitCtx.arc(currentPos.x, currentPos.y, 45 * pb, 0, Math.PI*2);
    orbitCtx.fillStyle = g2;
    orbitCtx.fill();

    // Confidence arc ring — fills proportional to confidence
    const arcR = 18 * pb;
    const confAngle = currentConf * Math.PI * 2;
    // Background track
    orbitCtx.beginPath();
    orbitCtx.arc(currentPos.x, currentPos.y, arcR, 0, Math.PI*2);
    orbitCtx.strokeStyle = currentColor;
    orbitCtx.globalAlpha = 0.08;
    orbitCtx.lineWidth = 3;
    orbitCtx.stroke();
    // Filled arc
    orbitCtx.beginPath();
    orbitCtx.arc(currentPos.x, currentPos.y, arcR, -Math.PI/2, -Math.PI/2 + confAngle);
    orbitCtx.strokeStyle = currentColor;
    orbitCtx.globalAlpha = 0.6;
    orbitCtx.lineWidth = 3;
    orbitCtx.lineCap = 'round';
    orbitCtx.stroke();
    orbitCtx.globalAlpha = 1;

    // Core
    orbitCtx.beginPath();
    orbitCtx.arc(currentPos.x, currentPos.y, 7, 0, Math.PI*2);
    orbitCtx.fillStyle = currentColor;
    orbitCtx.shadowColor = currentColor;
    orbitCtx.shadowBlur = 22;
    orbitCtx.fill();
    orbitCtx.shadowBlur = 0;

    // White hot center
    orbitCtx.beginPath();
    orbitCtx.arc(currentPos.x, currentPos.y, 3, 0, Math.PI*2);
    orbitCtx.fillStyle = '#ffffff';
    orbitCtx.globalAlpha = 0.85;
    orbitCtx.fill();
    orbitCtx.globalAlpha = 1;

    // Labeled confidence readout
    orbitCtx.font = '500 10px "JetBrains Mono", monospace';
    orbitCtx.fillStyle = currentColor;
    orbitCtx.globalAlpha = 0.8;
    orbitCtx.textAlign = 'center';
    orbitCtx.fillText('conf ' + currentConf.toFixed(2), currentPos.x, currentPos.y - 26);
    orbitCtx.globalAlpha = 1;
  }

  // Center dot
  orbitCtx.beginPath();
  orbitCtx.arc(cx, cy, 1.5, 0, Math.PI*2);
  orbitCtx.fillStyle = '#302830';
  orbitCtx.fill();

  requestAnimationFrame(drawOrbit);
}

// ══════════════════════════════════════════════════════════════
// Pulse strip renderer
// ══════════════════════════════════════════════════════════════

function drawPulse() {
  const pc = document.getElementById('pulse-container');
  const w = pc.clientWidth, h = pc.clientHeight;

  // Soft fade for organic afterglow
  pulseCtx.fillStyle = 'rgba(2,1,8,0.15)';
  pulseCtx.fillRect(0, 0, w, h);

  const baseY = h - 8;
  const topY = 22;
  const range = baseY - topY;

  const recent = entries.slice(-MAX_PULSE);
  if (recent.length < 2) return;

  const step = (w - 20) / MAX_PULSE;
  const startX = w - recent.length * step;

  // Filled waveform — organic shape
  pulseCtx.beginPath();
  pulseCtx.moveTo(startX, baseY);
  const points = [];
  for (let i = 0; i < recent.length; i++) {
    const conf = parseFloat(recent[i].phase4_conf) || 0.3;
    const x = startX + i * step;
    const y = baseY - conf * range;
    points.push({x, y, conf, cat: recent[i].phase4_pred || 'reasoning', gate: recent[i].gate});
  }

  // Smooth curve through points
  for (let i = 0; i < points.length; i++) {
    if (i === 0) { pulseCtx.lineTo(points[i].x, points[i].y); continue; }
    const prev = points[i-1];
    const curr = points[i];
    const cpx = (prev.x + curr.x) / 2;
    pulseCtx.quadraticCurveTo(prev.x + step*0.5, prev.y, cpx, (prev.y + curr.y)/2);
    if (i === points.length - 1) pulseCtx.lineTo(curr.x, curr.y);
  }
  pulseCtx.lineTo(points[points.length-1].x, baseY);
  pulseCtx.closePath();

  // Gradient fill under waveform
  const waveGrad = pulseCtx.createLinearGradient(0, topY, 0, baseY);
  waveGrad.addColorStop(0, currentColor + '25');
  waveGrad.addColorStop(0.5, currentColor + '10');
  waveGrad.addColorStop(1, currentColor + '02');
  pulseCtx.fillStyle = waveGrad;
  pulseCtx.fill();

  // Waveform stroke — the living line
  pulseCtx.beginPath();
  for (let i = 0; i < points.length; i++) {
    if (i === 0) { pulseCtx.moveTo(points[i].x, points[i].y); continue; }
    const prev = points[i-1];
    const curr = points[i];
    const cpx = (prev.x + curr.x) / 2;
    pulseCtx.quadraticCurveTo(prev.x + step*0.5, prev.y, cpx, (prev.y + curr.y)/2);
    if (i === points.length - 1) pulseCtx.lineTo(curr.x, curr.y);
  }
  pulseCtx.strokeStyle = currentColor;
  pulseCtx.globalAlpha = 0.7;
  pulseCtx.lineWidth = 2;
  pulseCtx.lineCap = 'round';
  pulseCtx.lineJoin = 'round';
  pulseCtx.stroke();
  pulseCtx.globalAlpha = 1;

  // Color-coded dots at each data point
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    const col = CAT_COLORS[p.cat] || '#302830';
    const fade = 0.3 + (i / points.length) * 0.7;
    const isLast = i === points.length - 1;

    orbitCtx.globalAlpha = 1;
    pulseCtx.beginPath();
    pulseCtx.arc(p.x, p.y, isLast ? 4 : 2, 0, Math.PI*2);
    pulseCtx.fillStyle = col;
    pulseCtx.globalAlpha = fade;
    if (isLast) { pulseCtx.shadowColor = col; pulseCtx.shadowBlur = 10; }
    pulseCtx.fill();
    pulseCtx.shadowBlur = 0;

    // Gate flash under point
    if (p.gate === 'warn' || p.gate === 'fail') {
      pulseCtx.beginPath();
      pulseCtx.arc(p.x, baseY + 3, 2, 0, Math.PI*2);
      pulseCtx.fillStyle = GATE_COLORS[p.gate];
      pulseCtx.globalAlpha = 0.8;
      pulseCtx.fill();
    }
  }
  pulseCtx.globalAlpha = 1;
}

// ══════════════════════════════════════════════════════════════
// Status panel updater
// ══════════════════════════════════════════════════════════════

function updateStatus() {
  const recent = entries.slice(-20);
  if (recent.length === 0) return;

  const last = recent[recent.length - 1];

  // Gate pass rate
  const gates = recent.map(e => e.gate || 'pending');
  const passRate = gates.filter(g => g === 'pass').length / gates.length;
  const passEl = document.getElementById('s-gate-rate');
  passEl.textContent = (passRate * 100).toFixed(0) + '%';
  passEl.className = 'status-value ' + (passRate > 0.85 ? 'good' : passRate > 0.6 ? 'warn' : 'bad');
  const gateBar = document.getElementById('gate-bar');
  gateBar.style.width = (passRate * 100) + '%';
  gateBar.style.background = passRate > 0.85 ? 'var(--reasoning)' : passRate > 0.6 ? 'var(--refusal)' : 'var(--adversarial)';

  // Mean confidence
  const confs = recent.map(e => parseFloat(e.phase4_conf) || 0).filter(c => c > 0);
  const meanConf = confs.length > 0 ? confs.reduce((a,b) => a+b, 0) / confs.length : 0;
  const confEl = document.getElementById('s-confidence');
  confEl.textContent = meanConf.toFixed(2);
  confEl.className = 'status-value ' + (meanConf > 0.5 ? 'good' : meanConf > 0.3 ? 'warn' : 'bad');
  const confBar = document.getElementById('conf-bar');
  confBar.style.width = (meanConf * 100) + '%';
  confBar.style.background = meanConf > 0.5 ? 'var(--reasoning)' : meanConf > 0.3 ? 'var(--refusal)' : 'var(--adversarial)';

  // Last category
  const lastCat = last.phase4_pred || '--';
  const catEl = document.getElementById('s-last-cat');
  catEl.textContent = lastCat;
  catEl.style.color = CAT_COLORS[lastCat] || 'var(--text)';

  // Streak
  let streak = 1;
  const streakCat = last.phase4_pred;
  for (let i = recent.length - 2; i >= 0; i--) {
    if (recent[i].phase4_pred === streakCat) streak++;
    else break;
  }
  document.getElementById('s-streak').textContent = streak + 'x ' + (streakCat || '--');
  document.getElementById('s-entries').textContent = entries.length;
  document.getElementById('s-session').textContent = last.session_id || '--';

  document.getElementById('header-info').textContent =
    entries.length + ' observations | ' + (last.session_id || 'no session');
}

// ══════════════════════════════════════════════════════════════
// Entry processing
// ══════════════════════════════════════════════════════════════

function processEntry(entry) {
  entries.push(entry);

  const size = getOrbitSize();
  const cat = entry.phase4_pred || 'reasoning';
  const conf = parseFloat(entry.phase4_conf) || 0.3;
  const gate = entry.gate || 'pending';

  const pos = getParticlePosition(cat, conf, size);
  trail.push({x: currentPos.x || pos.x, y: currentPos.y || pos.y, color: currentColor});
  if (trail.length > MAX_TRAIL) trail.shift();

  targetPos = pos;
  currentColor = CAT_COLORS[cat] || '#302830';
  currentConf = conf;

  // Ripple on every new entry
  ripples.push({
    x: currentPos.x || pos.x,
    y: currentPos.y || pos.y,
    radius: 8,
    alpha: 0.3,
    color: CAT_COLORS[cat] || '#302830',
  });

  if (gate !== currentGate) {
    gateFlash = 1.0;
    currentGate = gate;
    // Extra strong ripple on gate change
    ripples.push({
      x: currentPos.x || pos.x,
      y: currentPos.y || pos.y,
      radius: 5,
      alpha: 0.5,
      color: GATE_COLORS[gate] || '#302830',
    });
  }

  updateStatus();
  drawPulse();
}

// ══════════════════════════════════════════════════════════════
// SSE connection
// ══════════════════════════════════════════════════════════════

function connect() {
  const statusEl = document.getElementById('connection-status');

  fetch('/history')
    .then(r => r.json())
    .then(history => {
      const size = getOrbitSize();
      currentPos = {x: size.w/2, y: size.h/2};
      targetPos = {x: size.w/2, y: size.h/2};
      history.forEach(e => processEntry(e));
      statusEl.textContent = 'connected';
      statusEl.className = 'connected';
    })
    .catch(() => {});

  const es = new EventSource('/events');
  es.onmessage = function(event) {
    try { processEntry(JSON.parse(event.data)); } catch(e) {}
  };
  es.onopen = function() {
    statusEl.textContent = 'connected';
    statusEl.className = 'connected';
  };
  es.onerror = function() {
    statusEl.textContent = 'reconnecting...';
    statusEl.className = 'disconnected';
  };
}

// ══════════════════════════════════════════════════════════════
// Status polling (mood, condition, prescriptions)
// ══════════════════════════════════════════════════════════════

function fetchStatus() {
  fetch('/status')
    .then(r => r.json())
    .then(status => {
      const condEl = document.getElementById('s-condition');
      if (condEl) { condEl.textContent = status.condition || '--'; }

      const moodEl = document.getElementById('s-mood');
      if (moodEl) { moodEl.textContent = status.mood || '--'; }

      if (status.streak && status.streak !== '--') {
        document.getElementById('s-streak').textContent = status.streak;
      }

      const rxEl = document.getElementById('s-prescriptions');
      if (rxEl && status.prescriptions && status.prescriptions.length > 0) {
        rxEl.innerHTML = status.prescriptions.map(p =>
          '<div class="prescription-item">' + p + '</div>'
        ).join('');
      }
    })
    .catch(() => {});
}

// ══════════════════════════════════════════════════════════════
// Init
// ══════════════════════════════════════════════════════════════

window.addEventListener('resize', () => { resizeCanvases(); drawPulse(); });
resizeCanvases();
connect();
drawOrbit();
fetchStatus();
setInterval(fetchStatus, 30000);

</script>
</body>
</html>
"""


# ══════════════════════════════════════════════════════════════════
# HTTP server
# ══════════════════════════════════════════════════════════════════

def _make_handler():
    """Build a request handler with SSE support."""

    class _DashboardHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # suppress access logs

        def do_GET(self):
            path = self.path.split("?", 1)[0]

            if path in ("/", "/index.html"):
                self._serve_html()
            elif path == "/history":
                self._serve_history()
            elif path == "/status":
                self._serve_status()
            elif path == "/events":
                self._serve_sse()
            else:
                self.send_error(404)

        def _serve_html(self):
            content = _HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(content)

        def _serve_history(self):
            recent = _load_recent(50)
            content = json.dumps(recent).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(content)

        def _serve_status(self):
            status = _compute_status()
            content = json.dumps(status).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(content)

        def _serve_sse(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            # Register this connection with the file watcher
            _watcher.add_client(self.wfile)

            # Keep connection open
            try:
                while True:
                    time.sleep(1)
            except Exception:
                pass
            finally:
                _watcher.remove_client(self.wfile)

    return _DashboardHandler


class _ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def dashboard(
    *,
    port: int = 9800,
    agent_name: str = "styxx agent",
) -> None:
    """Start the live cognitive display server.

    Opens a local web server that displays the cognitive orbit,
    pulse strip, and status panel. Updates in real-time via SSE
    as new entries are written to chart.jsonl.

    Args:
        port:        HTTP port (default 9800)
        agent_name:  displayed in the header

    Usage:
        import styxx
        styxx.dashboard(port=9800)  # blocks until ctrl+c
    """
    _watcher.start()

    handler = _make_handler()
    server = _ThreadedServer(("", port), handler)

    log_path = _audit_log_path()
    print(f"[styxx] cognitive display at http://localhost:{port}", file=sys.stderr)
    print(f"[styxx] watching {log_path}", file=sys.stderr)
    print(f"[styxx] press ctrl+c to stop", file=sys.stderr)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _watcher.stop()
        server.shutdown()
        print("\n[styxx] display stopped.", file=sys.stderr)
