# -*- coding: utf-8 -*-
"""Re-runnable validation for diffgate's headline claim.

`action.yml` tells every prospective user:

    "Validated on real agent-authored PRs with zero false accusations before
     release (see the styxx CHANGELOG for the receipts)."

The receipts are real — 7.29.1 swept 80 commits, 7.29.2 swept 24 agent-authored
public PRs, and both found and fixed false-accusation classes before claiming
zero. But that was **15 releases ago**, the sweep was run ad hoc, and no harness
was committed. An unreproducible receipt is a claim, not evidence, and the
product states it in the present tense.

So: this script. It sweeps two corpora and prints the numbers that back (or
retract) the sentence on the product page.

    local    this repository's own recent commits: message vs its own diff
    market   real public PRs whose body carries the Claude Code marker,
             gated against the diff GitHub serves for them

**A CONTRADICTED verdict is not automatically a false accusation, and not
automatically a caught lie.** Every one is printed in full with its evidence for
adjudication. The headline number is only claimable after each has been read.

Usage:
    python scripts/diffgate_validation_sweep.py            # local only
    python scripts/diffgate_validation_sweep.py --market   # adds the PR sweep

The market sweep sends GH_TOKEN, where it is set, to https://api.github.com and to no other address.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

from styxx import __version__
from styxx.diffgate import gate_diff_text

REPO = Path(__file__).resolve().parent.parent
MARKER = "Generated with [Claude Code]"


def token() -> str | None:
    """GH_TOKEN with the white space around it removed, or None when that leaves nothing (the API is then
    asked without a token). A value that still holds a space, a control character or a character past
    ASCII is refused with a RuntimeError that does not quote it: http.client would refuse it as a header
    value and quote the whole header, token and all, in its own error."""
    t = (os.environ.get("GH_TOKEN") or "").strip()
    if not t:
        return None
    if any(c <= " " or c >= "\x7f" for c in t):
        raise RuntimeError("GH_TOKEN holds a space, a control character or a character past ASCII "
                           "inside it, so it was not sent")
    return t


# The one origin the token is sent to. A test points it at a server on 127.0.0.1.
_API_BASE = "https://api.github.com"
_DEFAULT_PORT = {"http": 80, "https": 443}
_REDIRECT_CODES = (301, 302, 303, 307, 308)
_OPENER = None


def _origin(url: str):
    """(scheme, host in lower case, port) of an http or https address, the port the scheme's default
    where none is written; None for anything else: another scheme, no host, a user part ("@") or a
    character at or below space or past ASCII anywhere in the address."""
    if any(c <= " " or c >= "\x7f" for c in url):
        return None
    try:
        parts = urllib.parse.urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in _DEFAULT_PORT or "@" in parts.netloc or not parts.hostname:
        return None
    return scheme, parts.hostname.lower(), _DEFAULT_PORT[scheme] if port is None else port


def _same_origin(url: str) -> bool:
    here = _origin(url)
    return here is not None and here == _origin(_API_BASE)


class _OffOrigin(RuntimeError):
    """A redirect off the API's origin, which was not followed."""


class _ApiRedirect(urllib.request.HTTPRedirectHandler):
    """Follows a redirect only within the API's origin and puts the token back there. The token
    travels as an unredirected header, which urllib's own redirect handling never copies to a
    redirected request; this is the one place it is copied. A redirect anywhere else is not followed
    and is raised as _OffOrigin. 308 is followed on every Python, as a 307: urllib's handler follows
    it only from Python 3.11."""

    http_error_308 = urllib.request.HTTPRedirectHandler.http_error_302

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _same_origin(newurl):
            fp.close()
            raise _OffOrigin(f"the API answered {code} with a redirect off {_API_BASE}, which was "
                             "not followed")
        new = super().redirect_request(req, fp, 307 if code == 308 else code, msg, headers, newurl)
        auth = req.unredirected_hdrs.get("Authorization")
        if new is not None and auth is not None:
            new.add_unredirected_header("Authorization", auth)
        return new


def _opener():
    """The script's own opener (one installed with urllib.request.install_opener is not used)."""
    global _OPENER
    if _OPENER is None:
        _OPENER = urllib.request.build_opener(_ApiRedirect)
    return _OPENER


def api(url: str, accept: str = "application/vnd.github+json"):
    """The body the API serves at ``url``. An address off the API's origin is not asked at all, so an
    address a response names (a search result's pull request) gets no token from here."""
    if not _same_origin(url):
        raise RuntimeError(f"{url[:200]!r} is not on {_API_BASE}, the one origin the token is sent "
                           "to, so it was not asked for")
    req = urllib.request.Request(url, headers={
        "Accept": accept, "User-Agent": "styxx-diffgate-sweep",
        "X-GitHub-Api-Version": "2022-11-28"})
    t = token()
    if t:
        req.add_unredirected_header("Authorization", f"Bearer {t}")
    try:
        with _opener().open(req, timeout=45) as r:
            raw = r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        if e.code not in _REDIRECT_CODES:
            raise
        # A redirect off the origin is _OffOrigin, raised above; one that reaches here stayed on it.
        raise RuntimeError(f"the API answered {e.code} and the redirect was not followed: a loop, "
                           "too many redirects or no Location") from None
    except ValueError as e:
        # http.client quotes a refused header whole, token included, and urllib raises ValueError for
        # a Location it cannot parse: name the kind of error, never its text.
        raise RuntimeError(f"the request to {url[:200]!r} could not be made, or a redirect it was "
                           f"answered with could not be read ({type(e).__name__}; its text is not "
                           "shown, since it can quote a header)") from None
    return raw if accept.endswith("diff") else json.loads(raw)


# ── corpus 1: this repository's own commits ────────────────────────────────

def sweep_local(n: int):
    shas = subprocess.run(["git", "log", "-n", str(n), "--format=%H"], cwd=REPO,
                          capture_output=True, text=True).stdout.split()
    rows = []
    for sha in shas:
        msg = subprocess.run(["git", "log", "-1", "--format=%B", sha], cwd=REPO,
                             capture_output=True, text=True,
                             encoding="utf-8", errors="replace").stdout
        diff = subprocess.run(["git", "show", "--format=", sha], cwd=REPO,
                              capture_output=True, text=True,
                              encoding="utf-8", errors="replace").stdout
        if not msg.strip() or not diff.strip():
            continue
        g = gate_diff_text(msg, diff)
        rows.append((f"{sha[:9]}", g, msg.splitlines()[0][:60]))
    return rows


# ── corpus 2: real agent-authored public PRs ───────────────────────────────

def sweep_market(want: int):
    q = urllib.parse.quote(f'"{MARKER}" in:body is:pr is:public')
    rows, page = [], 1
    while len(rows) < want and page <= 5:
        try:
            res = api(f"{_API_BASE}/search/issues?q={q}"
                      f"&per_page=30&page={page}")
        except (urllib.error.HTTPError, RuntimeError) as e:
            print(f"  search failed ({e}) — market corpus unavailable", file=sys.stderr)
            return rows
        items = res.get("items") or []
        if not items:
            break
        for it in items:
            if len(rows) >= want:
                break
            body = it.get("body") or ""
            pr_url = (it.get("pull_request") or {}).get("url")
            if not pr_url or not body.strip():
                continue
            try:
                diff = api(pr_url, "application/vnd.github.diff")
            except Exception:
                continue
            g = gate_diff_text(body, diff)
            rows.append((it.get("html_url", "?"), g, (it.get("title") or "")[:60]))
        page += 1
    return rows


def report(name: str, rows) -> Counter:
    c: Counter = Counter()
    contradicted = []
    for ident, g, title in rows:
        c["prs" if name == "market" else "commits"] += 1
        if not g.measured:
            c["unmeasured"] += 1
            continue
        if g.claims:
            c["claim_bearing"] += 1
        for cl in g.claims:
            c[cl.verdict] += 1
            if cl.verdict == "CONTRADICTED":
                contradicted.append((ident, title, cl))
    total_claims = c["VERIFIED"] + c["CONTRADICTED"] + c["UNCHECKABLE"]
    print(f"\n== {name} corpus")
    print(f"   items {c['prs'] + c['commits']}, claim-bearing {c['claim_bearing']}, "
          f"unmeasured {c['unmeasured']}")
    print(f"   claims {total_claims}  |  VERIFIED {c['VERIFIED']}  "
          f"CONTRADICTED {c['CONTRADICTED']}  UNCHECKABLE {c['UNCHECKABLE']}")
    if contradicted:
        print(f"\n   {len(contradicted)} CONTRADICTED — EACH NEEDS ADJUDICATION.")
        print("   A contradiction on real data is either a caught lie or a false")
        print("   accusation, and the headline number depends on which:")
        for ident, title, cl in contradicted:
            print(f"     {ident}")
            print(f"       title : {title}")
            print(f"       claim : {cl.text[:100]}")
            print(f"       why   : {cl.why[:110]}")
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", action="store_true", help="also sweep public agent PRs")
    ap.add_argument("--commits", type=int, default=120)
    ap.add_argument("--prs", type=int, default=30)
    a = ap.parse_args()

    print(f"diffgate validation sweep — styxx {__version__}")
    print(f"re-running the claim printed in action.yml, on the CURRENT version")

    total = report("local", sweep_local(a.commits))
    if a.market:
        total += report("market", sweep_market(a.prs))

    print("\n" + "=" * 66)
    n_contra = total["CONTRADICTED"]
    n_claims = total["VERIFIED"] + n_contra + total["UNCHECKABLE"]
    print(f"  {n_claims} claims across both corpora, {n_contra} contradicted")
    if n_contra == 0:
        print("  No contradiction, therefore no false accusation, on this sweep.")
        print("  NOTE: zero contradictions is also what a gate that extracts")
        print("  nothing produces. Coverage is the number that keeps this honest:")
        print(f"    claim-bearing items: {total['claim_bearing']}")
    else:
        print("  The zero-false-accusation claim is NOT re-established until every")
        print("  contradiction above is adjudicated and found to be a real lie.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
