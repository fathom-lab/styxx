"""Capture stunning playground screenshots for the README hero.

Runs headless Chromium via Playwright, loads each preset scenario,
waits for the verdict to render, and saves PNGs for:

  - fabricated-number (two red-underlined spans, 7-signal breakdown)
  - invented-entity (multi-span annotation, entity_novelty 1.0)
  - full-page capture showing header + scenarios + verdict

Outputs to release/playground-*.png.
"""
from __future__ import annotations
import asyncio
import os
import sys
from pathlib import Path

from playwright.async_api import async_playwright

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "release"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE = "https://fathom.darkflobi.com/cognometry/try"

SHOTS = [
    # (scenario, output_filename, clip_to_results_panel)
    ("fabricated-number", "playground-hero-fabricated-number.png", True),
    ("invented-entity",   "playground-hero-invented-entity.png",   True),
    ("fabricated-number", "playground-fullpage-fabricated.png",    False),
]


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            device_scale_factor=2,  # retina sharpness
        )
        page = await ctx.new_page()

        for scenario, fname, clip_to_results in SHOTS:
            print(f"[capture] {scenario} -> {fname}")
            await page.goto(f"{BASE}?scenario={scenario}")
            # Wait for Pyodide to load and auto-run verdict
            try:
                await page.wait_for_function(
                    """() => {
                        const action = document.getElementById('action-badge')?.textContent || '';
                        return action.length > 1 && action !== '\u2014' &&
                               document.querySelectorAll('.span-mark').length > 0;
                    }""",
                    timeout=90_000,
                )
            except Exception as e:
                print(f"  !! verdict did not render for {scenario}: {e}", file=sys.stderr)
                continue

            # Small settle delay for animations
            await asyncio.sleep(0.8)

            out = OUT_DIR / fname
            if clip_to_results:
                # Capture just the results panel (the hero shot)
                results = await page.query_selector("#results")
                if results:
                    await results.screenshot(path=str(out), omit_background=False)
                else:
                    await page.screenshot(path=str(out), full_page=False)
            else:
                await page.screenshot(path=str(out), full_page=True)

            print(f"  wrote {out}  ({out.stat().st_size // 1024} KB)")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
