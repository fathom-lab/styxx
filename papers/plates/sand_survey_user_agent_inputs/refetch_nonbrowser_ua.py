"""Re-fetch, once, every URL that the sand survey's fetchers requested under a browser-form user agent.

`fetch_pass3.py`, `fetch_pass4.py` and the correction's `prepare_correction_inputs.py` sent a user agent that
began "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", the form browsers send, and then named the lab. This script
asks whether any recorded fetch depended on that token. It requests each recorded URL once with a user agent
that carries no browser token, and changes nothing else in the curl call (-sSL, --max-time 120, redirects
followed, no cookie, no other header).

Frozen before the run: this file is committed and pushed before it is executed.

- Inventory: every attempt in `sand_survey_fetch_record_pass3_2026_09_14.json`,
  `sand_survey_pass4_inputs/fetch_record_run_{A,B,B2,B3}.json`, and the entries of
  `sand_survey_pass4_correction_inputs/fetch_record_correction.json` that carry attempts. Each distinct URL is
  requested once, in inventory order (pass 3, runs A, B, B2, B3, correction; attempts in recorded order).
- Spacing: at least 4 seconds between requests. No retry. A curl error is recorded as it is.
- An attempt "answered" when its http code is 2xx and it carried at least one byte (an empty HTTP 202, as
  Semantic Scholar's pages served, did not answer). A rate limit is an http code of 429 or 503.
- Outcome per URL, against every recorded attempt of that URL, taken in this order:
  - RATE_LIMITED: now a rate limit. It says nothing about the user agent.
  - NOW_REFUSED: a recorded attempt answered, and now the request did not answer (or curl failed).
  - STILL_REFUSED: no recorded attempt answered, and now the request did not answer.
  - NOW_ANSWERS: no recorded attempt answered, and now the request answered.
  - SAME_BYTES: now answered, and the sha256 equals the sha256 of a recorded attempt that answered.
  - SAME_KIND: now answered, and a recorded attempt that answered had the same kind (pdf or not), other bytes.
  - KIND_CHANGED: now answered, and no recorded attempt that answered had the same kind.
- What it can show. NOW_REFUSED or KIND_CHANGED on a URL whose bytes were read marks a place where the
  browser-form token may have decided what the lab received. Time is confounded with the user agent, since
  pages change. No control request is sent with the browser form, because no route may present as a browser.
- No fetched bytes are kept in the tree. The record keeps http code, content type, effective URL, byte count,
  sha256 and UTC time.

usage: python refetch_nonbrowser_ua.py <repo> <scratch_dir> <out_json> [--inventory-only]
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

UA = "fathom-lab-sand-survey-ua-check/1 (research; github.com/fathom-lab)"
SPACING_S = 4.0

SOURCES = [
    ("pass3", "papers/plates/sand_survey_fetch_record_pass3_2026_09_14.json"),
    ("pass4_run_A", "papers/plates/sand_survey_pass4_inputs/fetch_record_run_A.json"),
    ("pass4_run_B", "papers/plates/sand_survey_pass4_inputs/fetch_record_run_B.json"),
    ("pass4_run_B2", "papers/plates/sand_survey_pass4_inputs/fetch_record_run_B2.json"),
    ("pass4_run_B3", "papers/plates/sand_survey_pass4_inputs/fetch_record_run_B3.json"),
    ("correction", "papers/plates/sand_survey_pass4_correction_inputs/fetch_record_correction.json"),
]


def is2xx(code):
    return isinstance(code, str) and len(code) == 3 and code.startswith("2")


def kind_of(content_type):
    return "pdf" if "pdf" in (content_type or "").lower() else "other"


def inventory(repo):
    urls = {}
    order = []
    for record, rel in SOURCES:
        with open(os.path.join(repo, rel), encoding="utf-8") as fh:
            data = json.load(fh)
        for sid, entry in data.items():
            if not isinstance(entry, dict) or not isinstance(entry.get("attempts"), list):
                continue
            for i, att in enumerate(entry["attempts"]):
                url = att.get("url")
                if not url:
                    continue
                if url not in urls:
                    urls[url] = []
                    order.append(url)
                urls[url].append({
                    "record": record, "id": sid, "attempt": i, "entry_status": entry.get("status"),
                    "accepted": att.get("accepted"), "http": att.get("http"), "bytes": att.get("bytes"),
                    "content_type": att.get("content_type"), "sha256": att.get("sha256"),
                    "fetched_at": att.get("fetched_at"),
                })
    return order, urls


def fetch(url, path):
    r = subprocess.run(["curl", "-sSL", "-A", UA, "--max-time", "120", "-o", path,
                        "-w", "%{http_code} %{url_effective} %{content_type}", url],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    parts = r.stdout.strip().split(" ", 2) + ["", ""]
    code, eff, ctype = parts[:3]
    raw = open(path, "rb").read() if os.path.exists(path) else b""
    return {
        "http": code or None, "url_effective": eff or None, "content_type": ctype or None,
        "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest() if raw else None,
        "curl_exit": r.returncode, "curl_error": r.stderr.strip() or None,
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def answered(att):
    return is2xx(att.get("http")) and (att.get("bytes") or 0) > 0 and att.get("curl_exit", 0) == 0


def outcome(recorded, now):
    ok_before = [a for a in recorded if answered(a)]
    if now["http"] in ("429", "503"):
        return "RATE_LIMITED"
    if not answered(now):
        return "NOW_REFUSED" if ok_before else "STILL_REFUSED"
    if not ok_before:
        return "NOW_ANSWERS"
    if now["sha256"] and any(a["sha256"] == now["sha256"] for a in ok_before):
        return "SAME_BYTES"
    if any(kind_of(a["content_type"]) == kind_of(now["content_type"]) for a in ok_before):
        return "SAME_KIND"
    return "KIND_CHANGED"


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    repo, scratch, out = sys.argv[1], sys.argv[2], sys.argv[3]
    order, urls = inventory(repo)
    print(f"{len(order)} distinct URLs, {sum(len(v) for v in urls.values())} recorded attempts")
    if "--inventory-only" in sys.argv[4:]:
        for u in order:
            print(" ", [f"{a['record']}:{a['id']}:{a['http']}" for a in urls[u]], u)
        return
    os.makedirs(scratch, exist_ok=True)
    rows = []
    for n, url in enumerate(order):
        if n:
            time.sleep(SPACING_S)
        now = fetch(url, os.path.join(scratch, f"u{n:03d}.bin"))
        row = {"n": n, "url": url, "recorded": urls[url], "now": now, "outcome": outcome(urls[url], now)}
        rows.append(row)
        print(n, row["outcome"], now["http"], url, flush=True)
    counts = {}
    for r in rows:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    result = {"_note": "non-browser user-agent re-fetch of every URL the sand survey fetched under a browser-form user agent; see refetch_nonbrowser_ua.py for the frozen rules",
              "user_agent": UA, "spacing_s": SPACING_S, "counts": counts, "rows": rows}
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
