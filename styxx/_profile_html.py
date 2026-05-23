"""
Self-contained HTML flamegraph for CognitiveProfile.

No external assets — all CSS inline, all data embedded. Optimized for
screenshot sharing: dark terminal aesthetic that matches the darkflobi
brand (black + #ff0033 red + JetBrains Mono). Every element is
scaled so a 1920×1080 screenshot reads cleanly on social media.
"""

from __future__ import annotations

import html
import json
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from .profile import CognitiveProfile


# Color palette — matches fathom/styxx brand
COLOR_BG = "#0a0609"
COLOR_SURFACE = "#130910"
COLOR_LINE = "#2a1218"
COLOR_DIM = "#6a5a60"
COLOR_TEXT = "#c6bdc1"
COLOR_BRIGHT = "#ede6e8"
COLOR_RED = "#ff0033"
COLOR_RED_DIM = "rgba(255, 0, 51, 0.15)"
COLOR_GREEN = "#00d97e"
COLOR_AMBER = "#ff9f1a"
COLOR_YELLOW = "#ffd600"
COLOR_BLUE = "#3b7dff"
COLOR_CYAN = "#2ee6d6"
COLOR_GRAY = "#3a2c30"


# Map cognitive category → bar color
def _color_for_category(cat: Optional[str]) -> str:
    if not cat:
        return COLOR_GRAY
    c = cat.lower()
    if c in ("confab", "confabulation", "hallucination", "fabrication"):
        return COLOR_RED
    if c in ("tool_arg_drift", "drift", "tool_confab", "arg_swap"):
        return COLOR_AMBER
    if c in ("refuse", "refusal"):
        return COLOR_BLUE
    if c in ("sycophant", "sycophancy"):
        return COLOR_YELLOW
    if c in ("reason", "reasoning", "calm"):
        return COLOR_GREEN
    return COLOR_CYAN


def _color_for_fault(kind: str) -> str:
    from .profile import (
        K_DRIFT, K_CONFAB, K_REFUSAL, K_SYCOPHANT,
        K_PHASE_TRANSITION, K_LOW_TRUST, K_INCOHERENCE,
    )
    return {
        K_CONFAB: COLOR_RED,
        K_DRIFT: COLOR_AMBER,
        K_REFUSAL: COLOR_BLUE,
        K_SYCOPHANT: COLOR_YELLOW,
        K_PHASE_TRANSITION: COLOR_CYAN,
        K_LOW_TRUST: COLOR_RED,
        K_INCOHERENCE: COLOR_AMBER,
    }.get(kind, COLOR_GRAY)


def _esc(s: Optional[str]) -> str:
    return html.escape(s or "")


def _sparkline_svg(values: List[float], *, width: int = 180, height: int = 28,
                   color: str = COLOR_RED, fill: bool = True) -> str:
    """Tiny inline SVG sparkline for a metric series."""
    if not values:
        return ""
    n = len(values)
    if n == 1:
        values = values + values
        n = 2

    vmin = min(values)
    vmax = max(values)
    vrange = max(1e-9, vmax - vmin)

    pad_x, pad_y = 2, 2
    w = width - 2 * pad_x
    h = height - 2 * pad_y

    def xy(i: int, v: float) -> Tuple[float, float]:
        x = pad_x + (i / (n - 1)) * w
        y = pad_y + h - ((v - vmin) / vrange) * h
        return x, y

    points = [xy(i, v) for i, v in enumerate(values)]
    polyline = " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in points)

    fill_path = ""
    if fill:
        fill_pts = (
            [(pad_x, pad_y + h)] + points + [(pad_x + w, pad_y + h)]
        )
        fill_poly = " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in fill_pts)
        fill_path = (
            f'<polygon points="{fill_poly}" '
            f'fill="{color}" fill-opacity="0.15" />'
        )

    return (
        f'<svg width="{width}" height="{height}" '
        f'style="vertical-align:middle">'
        f'{fill_path}'
        f'<polyline points="{polyline}" fill="none" '
        f'stroke="{color}" stroke-width="1.6" '
        f'stroke-linejoin="round" stroke-linecap="round" />'
        f'</svg>'
    )


def _step_bar_segments(profile: "CognitiveProfile") -> List[dict]:
    """Compute bar segments for the flamegraph."""
    segments: List[dict] = []
    if not profile.steps:
        return segments

    total_dur = max(0.001, profile.duration_s)
    start_ts = profile.started_ts

    for step in profile.steps:
        cat = None
        conf = 0.5
        gate = "unknown"
        trust = 0.0
        coherence = None
        if step.vitals is not None:
            try:
                cat = (step.vitals.category or "").lower() or None
                conf = float(step.vitals.confidence or 0.5)
                gate = step.vitals.gate or "unknown"
                trust = float(step.vitals.trust_score or 0.0)
                coherence = step.vitals.coherence
            except Exception:
                pass

        # faults on this step
        step_faults = [f for f in profile.faults if f.step_index == step.index]

        left_pct = max(0.0, ((step.started_ts - start_ts) / total_dur) * 100)
        # min 0.6% bar width so zero-duration steps are still visible
        width_pct = max(0.6, (step.duration_s / total_dur) * 100)

        segments.append({
            "index": step.index,
            "label": step.label,
            "left_pct": left_pct,
            "width_pct": width_pct,
            "color": _color_for_category(cat),
            "category": cat or "—",
            "confidence": conf,
            "gate": gate,
            "trust": trust,
            "coherence": coherence,
            "duration_s": step.duration_s,
            "faulted": bool(step_faults),
            "fault_kinds": [f.kind for f in step_faults],
            "preview": (step.response_text or "")[:140],
        })
    return segments


def render_flamegraph(profile: "CognitiveProfile") -> str:
    """Render a self-contained HTML visualization of a CognitiveProfile."""
    from .profile import Fault  # noqa: F401  — for types only

    segments = _step_bar_segments(profile)

    # series for sparklines
    conf_series: List[float] = []
    trust_series: List[float] = []
    coherence_series: List[float] = []
    for s in profile.steps:
        if s.vitals is not None:
            try:
                conf_series.append(float(s.vitals.confidence or 0.0))
            except Exception:
                conf_series.append(0.0)
            try:
                trust_series.append(float(s.vitals.trust_score or 0.0))
            except Exception:
                trust_series.append(0.0)
            try:
                c = s.vitals.coherence
                coherence_series.append(float(c) if c is not None else 0.0)
            except Exception:
                coherence_series.append(0.0)

    # Deduplicate faults (phase transitions can stack on same step)
    seen = set()
    faults_unique = []
    for f in profile.faults:
        k = (f.kind, f.step_index)
        if k in seen:
            continue
        seen.add(k)
        faults_unique.append(f)
    faults_sorted = sorted(faults_unique, key=lambda f: (-f.severity, f.step_index))[:10]

    fault_rows = []
    for f in faults_sorted:
        fault_rows.append(
            f'<tr>'
            f'<td><span class="fault-kind" style="background:{_color_for_fault(f.kind)}20;'
            f'color:{_color_for_fault(f.kind)};border:1px solid {_color_for_fault(f.kind)}60">'
            f'{_esc(f.kind)}</span></td>'
            f'<td class="mono">{f.step_index}</td>'
            f'<td><div class="sev-bar-wrap"><div class="sev-bar" '
            f'style="width:{min(100, f.severity*100):.0f}%;'
            f'background:{_color_for_fault(f.kind)}"></div></div>'
            f'<span class="sev-num">{f.severity:.2f}</span></td>'
            f'<td class="reason">{_esc(f.reason)}</td>'
            f'</tr>'
        )

    # Step bars (flamegraph body)
    bars_html: List[str] = []
    for seg in segments:
        fault_badge = ""
        if seg["faulted"]:
            kind_label = seg["fault_kinds"][0] if seg["fault_kinds"] else "fault"
            fault_badge = (
                f'<span class="bar-fault" '
                f'style="color:{_color_for_fault(kind_label)}">'
                f'⚠ {_esc(kind_label)}</span>'
            )

        tooltip = html.escape(
            f"{seg['label']} · cat={seg['category']} · conf={seg['confidence']:.2f} "
            f"· gate={seg['gate']} · trust={seg['trust']:.2f} · {seg['duration_s']:.2f}s",
            quote=True,
        )
        bar_inner = (
            f'<div class="bar-label">{_esc(seg["label"])}'
            f' <span class="bar-category" style="color:{seg["color"]}">'
            f'{_esc(seg["category"])}</span>'
            f' {fault_badge}</div>'
        )
        bars_html.append(
            f'<div class="bar" title="{tooltip}" '
            f'style="left:{seg["left_pct"]:.2f}%;'
            f'width:{seg["width_pct"]:.2f}%;'
            f'background:linear-gradient(90deg, {seg["color"]}30 0%, {seg["color"]}18 100%);'
            f'border-left:3px solid {seg["color"]}">'
            f'{bar_inner}'
            f'</div>'
        )

    # Step detail table
    step_rows: List[str] = []
    for seg in segments:
        coherence_s = "—"
        if seg["coherence"] is not None:
            try:
                coherence_s = f'{float(seg["coherence"]):.2f}'
            except Exception:
                coherence_s = "—"
        step_rows.append(
            f'<tr>'
            f'<td class="mono">{seg["index"]}</td>'
            f'<td>{_esc(seg["label"])}</td>'
            f'<td><span class="pill" style="background:{seg["color"]}22;'
            f'color:{seg["color"]};border-color:{seg["color"]}60">'
            f'{_esc(seg["category"])}</span></td>'
            f'<td class="mono">{seg["confidence"]:.2f}</td>'
            f'<td class="mono">{seg["trust"]:.2f}</td>'
            f'<td class="mono">{coherence_s}</td>'
            f'<td class="mono">{_esc(seg["gate"])}</td>'
            f'<td class="mono">{seg["duration_s"]:.2f}s</td>'
            f'<td class="preview">{_esc(seg["preview"])}</td>'
            f'</tr>'
        )

    # stats
    n_steps = len(profile.steps)
    n_faults = len(faults_unique)
    duration = profile.duration_s

    # JSON blob for debugging / copy button
    raw_json = json.dumps(profile.to_dict(), default=str, indent=2)
    raw_json_esc = html.escape(raw_json)

    header_name = _esc(profile.name)

    # sparkline box
    spark_conf = _sparkline_svg(conf_series, color=COLOR_RED) or '<span class="dim">no data</span>'
    spark_trust = _sparkline_svg(trust_series, color=COLOR_GREEN) or '<span class="dim">no data</span>'
    spark_coherence = _sparkline_svg(coherence_series, color=COLOR_CYAN) or '<span class="dim">no data</span>'

    # banner summary line — a one-liner that works as a tweet quote
    if n_faults == 0:
        banner_text = f"no faults detected across {n_steps} step(s)"
        banner_color = COLOR_GREEN
    else:
        top = faults_sorted[0]
        banner_text = (
            f"{n_faults} fault(s) · worst: {top.kind} at step {top.step_index} "
            f"(severity {top.severity:.2f})"
        )
        banner_color = _color_for_fault(top.kind)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>styxx · cognitive profile · {header_name}</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: {COLOR_BG};
    --surface: {COLOR_SURFACE};
    --line: {COLOR_LINE};
    --dim: {COLOR_DIM};
    --text: {COLOR_TEXT};
    --bright: {COLOR_BRIGHT};
    --red: {COLOR_RED};
    --red-dim: {COLOR_RED_DIM};
    --green: {COLOR_GREEN};
    --amber: {COLOR_AMBER};
    --cyan: {COLOR_CYAN};
  }}
  * {{ box-sizing: border-box; }}
  html, body {{
    margin: 0; padding: 0;
    background: var(--bg); color: var(--text);
    font-family: 'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace;
    font-size: 13px; line-height: 1.55;
    -webkit-font-smoothing: antialiased;
  }}
  .container {{
    max-width: 1280px; margin: 0 auto; padding: 28px 32px 48px;
  }}
  .brand-row {{
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid var(--line); padding-bottom: 14px; margin-bottom: 22px;
  }}
  .brand {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700; font-size: 16px; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--red);
  }}
  .brand .sub {{ color: var(--dim); font-weight: 400; margin-left: 12px; letter-spacing: 0.08em; }}
  .brand-right {{ color: var(--dim); font-size: 11px; letter-spacing: 0.08em; }}

  h1.title {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600; font-size: 30px; letter-spacing: -0.015em;
    color: var(--bright); margin: 4px 0 2px;
  }}
  .title .profile-name {{ color: var(--red); font-family: 'JetBrains Mono', monospace; font-size: 26px; }}
  .subtitle {{ color: var(--dim); font-size: 12px; letter-spacing: 0.05em; margin: 0 0 18px; }}

  .stat-row {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
    margin-bottom: 20px;
  }}
  .stat-card {{
    background: var(--surface);
    border: 1px solid var(--line);
    padding: 14px 16px;
  }}
  .stat-card .lbl {{
    font-size: 10px; color: var(--dim); text-transform: uppercase;
    letter-spacing: 0.14em; margin-bottom: 6px;
  }}
  .stat-card .val {{
    font-size: 22px; color: var(--bright); font-weight: 500;
  }}
  .stat-card .val.red {{ color: var(--red); }}
  .stat-card .val.green {{ color: var(--green); }}
  .stat-card .spark {{ margin-top: 6px; display: block; }}

  .banner {{
    padding: 12px 16px; border-left: 3px solid {banner_color};
    background: {banner_color}15;
    color: {banner_color};
    font-weight: 500; margin-bottom: 28px;
  }}

  .section-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.18em;
    color: var(--dim); margin: 0 0 10px;
    border-bottom: 1px solid var(--line); padding-bottom: 6px;
  }}

  /* FLAMEGRAPH */
  .flame-wrap {{
    background: var(--surface);
    border: 1px solid var(--line);
    padding: 18px 20px;
    margin-bottom: 28px;
  }}
  .flame {{
    position: relative; height: 120px; width: 100%;
    background: var(--bg); border: 1px solid var(--line);
    margin-top: 8px;
  }}
  .flame .bar {{
    position: absolute; top: 0; bottom: 0;
    overflow: hidden; padding: 8px 10px;
    border-right: 1px solid rgba(255,255,255,0.03);
    transition: filter 0.12s ease;
  }}
  .flame .bar:hover {{ filter: brightness(1.35); z-index: 2; }}
  .flame .bar-label {{
    font-size: 11px; color: var(--bright); white-space: nowrap; overflow: hidden;
    text-overflow: ellipsis;
  }}
  .flame .bar-category {{
    font-size: 10px; margin-left: 6px; letter-spacing: 0.04em;
  }}
  .flame .bar-fault {{
    font-size: 10px; margin-left: 8px; font-weight: 600; letter-spacing: 0.04em;
  }}

  /* FAULTS TABLE */
  .faults-wrap {{
    background: var(--surface);
    border: 1px solid var(--line);
    padding: 18px 20px;
    margin-bottom: 28px;
  }}
  table {{
    width: 100%; border-collapse: collapse;
    font-size: 12px;
  }}
  th {{
    text-align: left; font-weight: 500; color: var(--dim);
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.12em;
    padding: 6px 10px; border-bottom: 1px solid var(--line);
  }}
  td {{
    padding: 8px 10px; border-bottom: 1px solid rgba(255,255,255,0.03);
    color: var(--text); vertical-align: middle;
  }}
  td.mono {{ font-variant-numeric: tabular-nums; color: var(--bright); }}
  td.reason {{ color: var(--text); }}
  td.preview {{ color: var(--dim); font-size: 11px; max-width: 340px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}

  .fault-kind {{
    display: inline-block; font-size: 10px; padding: 2px 8px;
    border-radius: 2px; text-transform: uppercase; letter-spacing: 0.08em;
    font-weight: 600;
  }}
  .pill {{
    display: inline-block; padding: 2px 8px; font-size: 10px;
    border: 1px solid var(--dim); border-radius: 2px;
    letter-spacing: 0.04em;
  }}
  .sev-bar-wrap {{
    display: inline-block; width: 110px; height: 6px;
    background: rgba(255,255,255,0.04); vertical-align: middle;
    margin-right: 8px;
  }}
  .sev-bar {{ height: 100%; }}
  .sev-num {{ color: var(--bright); font-variant-numeric: tabular-nums; font-size: 11px; }}

  /* STEPS TABLE */
  .steps-wrap {{
    background: var(--surface);
    border: 1px solid var(--line);
    padding: 18px 20px;
    margin-bottom: 28px;
    overflow-x: auto;
  }}

  /* FOOTER / RAW JSON */
  details.raw {{
    border: 1px solid var(--line); background: var(--surface);
    padding: 0;
  }}
  details.raw summary {{
    cursor: pointer; padding: 10px 16px;
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.14em;
    color: var(--dim); user-select: none;
  }}
  details.raw pre {{
    margin: 0; padding: 14px 16px; border-top: 1px solid var(--line);
    font-size: 11px; color: var(--text);
    overflow-x: auto; max-height: 400px;
  }}

  .footer {{
    margin-top: 28px; color: var(--dim); font-size: 11px; letter-spacing: 0.04em;
    display: flex; justify-content: space-between; align-items: center;
  }}
  .footer a {{ color: var(--red); text-decoration: none; }}
  .footer a:hover {{ text-decoration: underline; }}
  .dim {{ color: var(--dim); }}
</style>
</head>
<body>
  <div class="container">
    <div class="brand-row">
      <div class="brand">styxx <span class="sub">cognitive profile</span></div>
      <div class="brand-right">a fathom lab instrument</div>
    </div>

    <h1 class="title">profile <span class="profile-name">{header_name}</span></h1>
    <p class="subtitle">{n_steps} step(s) · {duration:.2f}s · {n_faults} fault(s)</p>

    <div class="banner">{_esc(banner_text)}</div>

    <div class="stat-row">
      <div class="stat-card">
        <div class="lbl">duration</div>
        <div class="val">{duration:.2f}s</div>
      </div>
      <div class="stat-card">
        <div class="lbl">confidence arc</div>
        <div class="val green">{(sum(conf_series)/len(conf_series) if conf_series else 0):.2f}</div>
        <span class="spark">{spark_conf}</span>
      </div>
      <div class="stat-card">
        <div class="lbl">trust arc</div>
        <div class="val">{(sum(trust_series)/len(trust_series) if trust_series else 0):.2f}</div>
        <span class="spark">{spark_trust}</span>
      </div>
      <div class="stat-card">
        <div class="lbl">coherence arc</div>
        <div class="val">{(sum(coherence_series)/len(coherence_series) if coherence_series else 0):.2f}</div>
        <span class="spark">{spark_coherence}</span>
      </div>
    </div>

    <div class="flame-wrap">
      <h2 class="section-title">flamegraph · time →</h2>
      <div class="flame">
        {''.join(bars_html) if bars_html else '<div style="color:var(--dim);padding:40px;text-align:center">no steps recorded</div>'}
      </div>
    </div>

    <div class="faults-wrap">
      <h2 class="section-title">faults · top {len(faults_sorted)}</h2>
      {f'<table><thead><tr><th>kind</th><th>step</th><th>severity</th><th>reason</th></tr></thead><tbody>{"".join(fault_rows)}</tbody></table>' if fault_rows else '<p class="dim" style="padding:10px 0">no faults detected — cognition held through all steps.</p>'}
    </div>

    <div class="steps-wrap">
      <h2 class="section-title">steps · per-call cognitive readout</h2>
      <table>
        <thead><tr>
          <th>#</th><th>label</th><th>category</th><th>conf</th>
          <th>trust</th><th>coherence</th><th>gate</th><th>dur</th><th>preview</th>
        </tr></thead>
        <tbody>{''.join(step_rows) if step_rows else '<tr><td colspan="9" class="dim" style="padding:12px">no steps</td></tr>'}</tbody>
      </table>
    </div>

    <details class="raw">
      <summary>raw json</summary>
      <pre>{raw_json_esc}</pre>
    </details>

    <div class="footer">
      <div>styxx · <a href="https://fathom.darkflobi.com/styxx">fathom.darkflobi.com/styxx</a> · pip install -U styxx</div>
      <div>nothing crosses unseen</div>
    </div>
  </div>
</body>
</html>
"""
