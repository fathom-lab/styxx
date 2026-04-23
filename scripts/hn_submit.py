"""Submit cognometry URL to HN using Firefox session cookie.

1. GET /submit → extract fnid
2. POST /r → creates submission
3. Follow redirect → print item URL
"""
from __future__ import annotations

import re
import sys
import time

import browser_cookie3
import requests


def main():
    jar = browser_cookie3.firefox(domain_name="news.ycombinator.com")
    s = requests.Session()
    for c in jar:
        s.cookies.set(c.name, c.value, domain=c.domain)
    s.headers.update({"User-Agent": "Mozilla/5.0 (styxx-launch)"})

    # Verify login first
    r = s.get("https://news.ycombinator.com/news", timeout=15)
    m = re.search(
        r'user\?id=([a-zA-Z0-9_\-]+)"[^>]*>[^<]+</a>\s*'
        r'\(\s*<span id="karma">\s*(\d+)',
        r.text,
    )
    if not m:
        print("NOT LOGGED IN — abort")
        sys.exit(1)
    print(f"logged in as: {m.group(1)} (karma {m.group(2)})")

    # Get submit form
    r = s.get("https://news.ycombinator.com/submit", timeout=15)
    if "You need to log in" in r.text or r.status_code != 200:
        print(f"submit form unavailable (status {r.status_code})")
        sys.exit(1)
    fm = re.search(r'<input[^>]+name="fnid"[^>]+value="([^"]+)"', r.text)
    if not fm:
        print("fnid not found in submit form")
        sys.exit(1)
    fnid = fm.group(1)
    print(f"fnid: {fnid[:20]}...")

    # Submit
    data = {
        "fnid": fnid,
        "fnop": "submit-page",
        "title": "Cognometry: The measurement of machine cognition",
        "url": "https://fathom.darkflobi.com/cognometry?ref=hn",
        "text": "",
    }
    r = s.post(
        "https://news.ycombinator.com/r",
        data=data,
        allow_redirects=False,
        timeout=20,
    )
    print(f"submit status: {r.status_code}")
    loc = r.headers.get("Location")
    print(f"redirect to: {loc}")

    if r.status_code not in (200, 302):
        print("unexpected status — may have failed")
        print(r.text[:500])
        sys.exit(2)

    # Follow to figure out item id
    # HN typically redirects to /newest after submission
    time.sleep(2)
    r2 = s.get("https://news.ycombinator.com/submitted?id="
               + m.group(1), timeout=15)
    # find the most recent submission matching our title
    it = re.search(
        r'id="(\d+)"[^>]*class="athing[^>]*>[^<]*<td[^>]*>'
        r'[^<]*<span[^>]*rank[^>]*>[^<]*</span>[\s\S]{0,300}'
        r'Cognometry',
        r2.text,
    )
    if it:
        item_id = it.group(1)
        print(f"\nITEM: https://news.ycombinator.com/item?id={item_id}")
    else:
        print(
            "\n(couldn't parse item id from submitted page — check manually "
            "at https://news.ycombinator.com/submitted?id=" + m.group(1) + ")"
        )


if __name__ == "__main__":
    main()
