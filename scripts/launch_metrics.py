"""
launch_metrics.py — one-shot funnel readout for the styxx launch.

Polls every public data source we have and prints a single
dashboard so you can see at a glance what's landing.

No dependencies beyond stdlib. Run anytime:

    python scripts/launch_metrics.py

Sources:
  · Zenodo  — download counts on each DOI (concept + 3 versions)
  · PyPI    — last-day, last-week, last-month install counts (pypistats.org)
  · GitHub  — stars, watchers, forks on the canonical repo

Twitter is not auto-readable (X is paywalled to anonymous fetchers).
Check engagement on the launch tweet manually.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

# Reconfigure stdout/stderr to utf-8 on Windows cp1252 consoles so the
# unicode box-drawing chars render. Fail-open if reconfigure isn't
# available (older Pythons or non-stream stdouts).
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    _reconfigure = getattr(_stream, "reconfigure", None) if _stream else None
    if _reconfigure is not None:
        try:
            _enc = (getattr(_stream, "encoding", "") or "").lower()
            if _enc and "utf" not in _enc:
                _reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# ── data sources ──────────────────────────────────────────────────────

# Zenodo reports stats at the concept level — all versions in the same
# concept-chain return the same numbers. We show one row per concept.
ZENODO_RECORDS = [
    ("19326174", "Fathom main chain (concept · spec · robustness)"),
    ("19758619", "Software · styxx v6.2.0 (separate concept)"),
]

PYPI_PACKAGE = "styxx"
GITHUB_REPO = "fathom-lab/styxx"
TIMEOUT_S = 10.0


# ── fetchers ──────────────────────────────────────────────────────────

def _get_json(url: str, headers: dict | None = None) -> dict | None:
    """Fetch JSON from a URL. Returns None on any failure."""
    req = urllib.request.Request(url, headers=headers or {"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return None


def fetch_zenodo(record_id: str) -> dict | None:
    j = _get_json(f"https://zenodo.org/api/records/{record_id}")
    if not j or "stats" not in j:
        return None
    s = j["stats"]
    return {
        "title": j.get("metadata", {}).get("title", "?")[:60],
        "version": j.get("metadata", {}).get("version", "?"),
        "views": s.get("unique_views", s.get("views", 0)),
        "downloads": s.get("unique_downloads", s.get("downloads", 0)),
    }


def fetch_pypi() -> dict | None:
    # primary: pypistats (richest data)
    j = _get_json(f"https://pypistats.org/api/packages/{PYPI_PACKAGE}/recent")
    if j and "data" in j:
        return j["data"]
    # fallback: pypi.org json (only gives release info, no installs)
    j = _get_json(f"https://pypi.org/pypi/{PYPI_PACKAGE}/json")
    if j and "info" in j:
        return {
            "_fallback": True,
            "version": j["info"].get("version", "?"),
            "release_count": len(j.get("releases", {})),
        }
    return None


def fetch_github() -> dict | None:
    j = _get_json(
        f"https://api.github.com/repos/{GITHUB_REPO}",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "styxx-launch-metrics"},
    )
    if not j:
        return None
    return {
        "stars": j.get("stargazers_count", 0),
        "watchers": j.get("subscribers_count", 0),
        "forks": j.get("forks_count", 0),
        "open_issues": j.get("open_issues_count", 0),
        "pushed_at": j.get("pushed_at", "?"),
    }


# ── render ────────────────────────────────────────────────────────────

def main() -> int:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print()
    print(f"  styxx launch metrics · {now}")
    print(f"  {'─' * 60}")

    # zenodo
    print("\n  zenodo (research artifacts)")
    for rid, label in ZENODO_RECORDS:
        z = fetch_zenodo(rid)
        if not z:
            print(f"    {label:48s}  unreachable")
            continue
        print(f"    {label:48s}  {z['views']:>5} views · {z['downloads']:>5} downloads")

    # pypi
    print("\n  pypi (styxx · install funnel)")
    p = fetch_pypi()
    if p and not p.get("_fallback"):
        print(f"    {'last day':48s}  {p.get('last_day', 0):>5} installs")
        print(f"    {'last week':48s}  {p.get('last_week', 0):>5} installs")
        print(f"    {'last month':48s}  {p.get('last_month', 0):>5} installs")
    elif p and p.get("_fallback"):
        print(f"    pypistats unreachable; pypi.org reports v{p['version']}")
        print(f"    ({p['release_count']} total releases · install counts unavailable)")
    else:
        print("    pypistats and pypi.org both unreachable (try again later)")

    # github
    print(f"\n  github ({GITHUB_REPO})")
    g = fetch_github()
    if g:
        print(f"    {'stars':38s}  {g['stars']:>5}")
        print(f"    {'watchers':38s}  {g['watchers']:>5}")
        print(f"    {'forks':38s}  {g['forks']:>5}")
        print(f"    {'open issues':38s}  {g['open_issues']:>5}")
        print(f"    last push                                {g['pushed_at']}")
    else:
        print("    github api unreachable")

    # x (manual)
    print("\n  x · @fathom_lab")
    print("    not auto-readable (paywall). check the launch tweet manually:")
    print("    x.com/fathom_lab")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
