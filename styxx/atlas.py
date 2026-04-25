"""
styxx.atlas — Atlas Pro fetch helper.

Free tier: the Cognitive Atlas v0.3 ships with the package at
``styxx/centroids/atlas_v0.3.json`` (CC-BY-4.0).

Pro tier: weekly-updated calibrations + daily Telescope archive + closed-API
proxy fingerprints, gated on holding 100,000 $STYXX in a connected Solana
wallet. Get a token at https://fathom.darkflobi.com/atlas-pro and pass it via
``token=`` or set ``STYXX_ATLAS_PRO_TOKEN`` in the environment.

Usage::

    from styxx.atlas import fetch_pro

    # Explicit token
    cal = fetch_pro(token="atlas-pro-...")

    # Or set env var STYXX_ATLAS_PRO_TOKEN once and call without args
    cal = fetch_pro()

    print(cal["telescope_archive"][:5])         # daily fingerprints
    print(cal["closed_model_calibrations"])     # claude/gemini proxy data

The free tier always returns a status object pointing at the public Atlas;
no token is required to call ``fetch_free()``.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

# Default endpoint. Override via STYXX_ATLAS_PRO_URL for self-hosted setups.
ATLAS_PRO_URL = os.environ.get(
    "STYXX_ATLAS_PRO_URL",
    "https://darkflobi.com/api/atlas-pro/calibrations.json",
)
ATLAS_PRO_FATHOM_URL = os.environ.get(
    "STYXX_ATLAS_PRO_URL_FATHOM",
    "https://fathom.darkflobi.com/api/atlas-pro/calibrations.json",
)
TIMEOUT_S = 15.0


class AtlasProError(RuntimeError):
    """Raised when Atlas Pro access fails (token invalid, expired, network)."""


def _resolve_token(token: Optional[str]) -> Optional[str]:
    if token:
        return token.strip()
    env = os.environ.get("STYXX_ATLAS_PRO_TOKEN")
    return env.strip() if env else None


def _fetch_url(url: str, timeout: float = TIMEOUT_S) -> dict:
    req = urllib.request.Request(url, headers={
        "Accept": "application/json",
        "User-Agent": f"styxx.atlas/python",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        return json.loads(data.decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
        except Exception:
            payload = {"error": body[:300]}
        raise AtlasProError(f"HTTP {e.code}: {payload.get('error', body[:200])}")
    except urllib.error.URLError as e:
        raise AtlasProError(f"network error: {e.reason}")


def fetch_pro(
    token: Optional[str] = None,
    *,
    url: Optional[str] = None,
    timeout: float = TIMEOUT_S,
) -> dict:
    """Fetch the Atlas Pro gated calibration payload.

    Args:
        token:   Atlas Pro access token. If None, reads from
                 ``STYXX_ATLAS_PRO_TOKEN`` env var. Get a token at
                 https://fathom.darkflobi.com/atlas-pro by signing a
                 challenge with a Solana wallet holding ≥100K $STYXX.
        url:     Override the calibrations endpoint (default: production
                 Fathom site). Useful for self-hosted setups.
        timeout: HTTP timeout in seconds.

    Returns:
        A dict containing:
          - tier: "atlas-pro"
          - telescope_archive: list of daily Spec-v1.0 fingerprints
          - closed_model_calibrations: per-substrate proxy-pipeline tuning
          - calibration_atlas: weekly-updated full atlas
          - generated_at, expires_at, spec_doi, etc.

    Raises:
        AtlasProError: token missing, invalid, or network failure.
    """
    tok = _resolve_token(token)
    if not tok:
        raise AtlasProError(
            "no Atlas Pro token. Get one at https://fathom.darkflobi.com/atlas-pro "
            "and pass via token= or set STYXX_ATLAS_PRO_TOKEN env var."
        )

    base = url or ATLAS_PRO_FATHOM_URL
    full = f"{base}?{urllib.parse.urlencode({'token': tok})}"
    try:
        return _fetch_url(full, timeout=timeout)
    except AtlasProError:
        # Fallback to alternate hostname (darkflobi.com vs fathom.darkflobi.com)
        if base == ATLAS_PRO_FATHOM_URL:
            return _fetch_url(
                f"{ATLAS_PRO_URL}?{urllib.parse.urlencode({'token': tok})}",
                timeout=timeout,
            )
        raise


def fetch_free(
    *,
    url: Optional[str] = None,
    timeout: float = TIMEOUT_S,
) -> dict:
    """Fetch the free-tier pointer (no token required).

    Returns the public Atlas v0.3 pointer + spec DOI references.
    """
    base = url or ATLAS_PRO_FATHOM_URL
    return _fetch_url(base, timeout=timeout)


def is_pro(token: Optional[str] = None) -> bool:
    """Cheap check: do we have a usable Atlas Pro token?

    Does NOT validate the token against the server (that requires a network
    call to ``fetch_pro``). Returns True if a token-shaped string is
    available, False otherwise.
    """
    tok = _resolve_token(token)
    if not tok:
        return False
    return tok.startswith("atlas-pro-") and len(tok) >= len("atlas-pro-") + 16


__all__ = [
    "fetch_pro",
    "fetch_free",
    "is_pro",
    "AtlasProError",
]
