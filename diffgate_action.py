"""GitHub Action entry for styxx.diffgate — checkout-free, injection-safe.

Reads the event payload from GITHUB_EVENT_PATH (the summary text never touches a shell),
fetches the diff from the GitHub API with the workflow token, gates the summary against it
with ``styxx.diffgate.gate_diff_text``, writes a job-summary table, emits ::error::
annotations for contradictions, and exits per the gate verdict and the strict/soft-fail
inputs. Supports ``pull_request`` (body vs PR diff) and ``push`` (head commit message vs
compare diff). Anything else: reports and passes.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

from styxx.diffgate import gate_diff_text


def api(url: str, accept: str) -> str:
    req = urllib.request.Request(url, headers={
        "Accept": accept, "User-Agent": "styxx-diffgate-action",
        "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
        "X-GitHub-Api-Version": "2022-11-28"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def main() -> int:
    strict = os.environ.get("STYXX_STRICT", "false").lower() == "true"
    soft = os.environ.get("STYXX_SOFT_FAIL", "false").lower() == "true"
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    event = json.loads(open(os.environ["GITHUB_EVENT_PATH"], encoding="utf-8").read())

    if event_name == "pull_request":
        summary = (event.get("pull_request") or {}).get("body") or ""
        diff_url = (event.get("pull_request") or {}).get("url", "")
        what = f"PR #{(event.get('pull_request') or {}).get('number', '?')} body"
    elif event_name == "push":
        summary = (event.get("head_commit") or {}).get("message") or ""
        diff_url = (event.get("compare", "")
                    .replace("github.com", "api.github.com/repos", 1)
                    .replace("/compare/", "/compare/", 1))
        # canonical form: repository.compare_url template
        tpl = (event.get("repository") or {}).get("compare_url", "")
        if tpl:
            diff_url = tpl.replace("{base}", event.get("before", "")) \
                          .replace("{head}", event.get("after", ""))
        what = "head commit message"
    else:
        print(f"styxx diffgate: event {event_name!r} not gated (pull_request/push only)")
        return 0

    if not summary.strip():
        print(f"styxx diffgate: empty {what} — nothing to gate")
        return 0
    try:
        diff = api(diff_url, "application/vnd.github.diff")
    except Exception as e:  # noqa: BLE001 — a broken fetch must not fake a verdict
        print(f"::warning::styxx diffgate could not fetch the diff ({e}); not gating")
        return 0

    g = gate_diff_text(summary, diff, strict=strict)

    lines = ["## styxx diffgate — the summary vs the diff", "",
             f"**{g.verdict}** · {len(g.claims)} claim(s) checked on the {what}", ""]
    if g.claims:
        lines += ["| verdict | claim | evidence |", "|---|---|---|"]
        for c in g.claims:
            mark = {"VERIFIED": "✅", "CONTRADICTED": "❌", "UNCHECKABLE": "❓"}[c.verdict]
            lines.append(f"| {mark} {c.verdict} | {c.text[:80]} | {c.why[:100]} |")
    else:
        lines += ["_No diff-shaped claims found. The gate checks a closed template set "
                  "(touched/created/deleted paths, added functions/tests, file counts, "
                  "only-touches scopes, tests-pass); prose outside it is not judged._"]
    lines += ["", "_A ❓ fails only with `strict: true`. Zero false accusations across "
              "both public validation corpora — receipts in the "
              "[styxx CHANGELOG](https://github.com/fathom-lab/styxx/blob/main/CHANGELOG.md)._"]
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    failing = False
    for c in g.claims:
        if c.verdict == "CONTRADICTED":
            failing = True
            print(f"::error title=styxx diffgate — contradicted claim::{c.text[:120]} — {c.why}")
        elif c.verdict == "UNCHECKABLE" and strict:
            failing = True
            print(f"::error title=styxx diffgate — uncheckable under strict::{c.text[:120]} — {c.why}")
        elif c.verdict == "UNCHECKABLE":
            print(f"::warning title=styxx diffgate — uncheckable::{c.text[:120]} — {c.why}")
    print(f"styxx diffgate: {g.verdict} — {len(g.claims)} claim(s), "
          f"{sum(1 for c in g.claims if c.verdict == 'CONTRADICTED')} contradicted")
    if failing and not soft:
        return 1
    if failing and soft:
        print("::warning::styxx diffgate found failures but soft-fail is on")
    return 0


if __name__ == "__main__":
    sys.exit(main())
