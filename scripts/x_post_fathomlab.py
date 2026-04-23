"""Post a single tweet from @fathom_lab via cookie auth.

Verify the session is fathom_lab BEFORE posting. If verification
returns a different screen_name, abort rather than tweet from the
wrong account.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import requests

SECRETS = Path(r"C:\Users\heyzo\clawd\secrets\x-cookies.json")

# X's well-known public bearer token (used by x.com web client for all
# internal-API calls — same value for every browser session).
PUBLIC_BEARER = (
    "AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D"
    "1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA"
)

# CreateTweet GraphQL queryId — public, stable, visible in devtools
CREATE_TWEET_QUERY_ID = "VyX6OPvkQx9G3FRCh1V5cA"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

TWEET_TEXT = """every LLM hallucinates. we built the detector.

@trust
def my_rag(q, *, context): ...

one line. cross-validated on 8 public benchmarks. two failure modes published openly. open source.

pip install styxx[nli]
https://fathom.darkflobi.com/cognometry"""


def load_cookies():
    data = json.loads(SECRETS.read_text(encoding="utf-8"))
    # Extract fathom_lab auth_token from the auth_multi field.
    # Format: "USER_ID:TOKEN" (quoted)
    auth_multi = data["auth_multi"].strip('"')
    if ":" not in auth_multi:
        raise SystemExit("auth_multi missing USER:TOKEN format")
    fathom_uid, fathom_token = auth_multi.split(":", 1)
    if fathom_uid != "2017011970419879936":
        raise SystemExit(
            f"auth_multi has unexpected user id {fathom_uid}; aborting"
        )
    return {
        "ct0": data["ct0"],
        "fathom_token": fathom_token,
        "fathom_uid": fathom_uid,
    }


def make_session(auth_token: str, ct0: str, twid_uid: str):
    s = requests.Session()
    for name, val, dom in [
        ("auth_token", auth_token, ".x.com"),
        ("ct0", ct0, ".x.com"),
        ("twid", f"u%3D{twid_uid}", ".x.com"),
        ("auth_token", auth_token, ".twitter.com"),
        ("ct0", ct0, ".twitter.com"),
        ("twid", f"u%3D{twid_uid}", ".twitter.com"),
    ]:
        s.cookies.set(name, val, domain=dom)
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Authorization": f"Bearer {PUBLIC_BEARER}",
        "x-csrf-token": ct0,
        "x-twitter-auth-type": "OAuth2Session",
        "x-twitter-active-user": "yes",
        "content-type": "application/json",
        "referer": "https://x.com/",
        "origin": "https://x.com",
    })
    return s


def verify_account(session):
    """GET settings to discover which screen_name the session is for."""
    # Multiple endpoints can reveal screen_name; try the settings one.
    urls = [
        "https://api.x.com/1.1/account/settings.json",
        "https://api.twitter.com/1.1/account/settings.json",
    ]
    for u in urls:
        try:
            r = session.get(u, timeout=10)
            if r.status_code == 200:
                d = r.json()
                return d.get("screen_name") or d.get("user", {}).get("screen_name")
        except Exception as e:
            print(f"  verify via {u}: {type(e).__name__} {e}", file=sys.stderr)
    return None


def post_tweet(session, text: str):
    """POST CreateTweet via GraphQL internal API."""
    url = (
        "https://x.com/i/api/graphql/"
        f"{CREATE_TWEET_QUERY_ID}/CreateTweet"
    )
    payload = {
        "variables": {
            "tweet_text": text,
            "dark_request": False,
            "media": {"media_entities": [], "possibly_sensitive": False},
            "semantic_annotation_ids": [],
        },
        "features": {
            "communities_web_enable_tweet_community_results_fetch": True,
            "c9s_tweet_anatomy_moderator_badge_enabled": True,
            "responsive_web_edit_tweet_api_enabled": True,
            "graphql_is_translatable_rweb_tweet_is_translatable_enabled": True,
            "view_counts_everywhere_api_enabled": True,
            "longform_notetweets_consumption_enabled": True,
            "responsive_web_twitter_article_tweet_consumption_enabled": True,
            "tweet_awards_web_tipping_enabled": False,
            "creator_subscriptions_quote_tweet_preview_enabled": False,
            "longform_notetweets_rich_text_read_enabled": True,
            "longform_notetweets_inline_media_enabled": True,
            "articles_preview_enabled": True,
            "rweb_video_timestamps_enabled": True,
            "rweb_tipjar_consumption_enabled": True,
            "responsive_web_graphql_exclude_directive_enabled": True,
            "verified_phone_label_enabled": False,
            "freedom_of_speech_not_reach_fetch_enabled": True,
            "standardized_nudges_misinfo": True,
            "tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled": True,
            "responsive_web_graphql_skip_user_profile_image_extensions_enabled": False,
            "responsive_web_graphql_timeline_navigation_enabled": True,
            "responsive_web_enhance_cards_enabled": False,
        },
        "queryId": CREATE_TWEET_QUERY_ID,
    }
    r = session.post(url, json=payload, timeout=20)
    return r


def main():
    cookies = load_cookies()
    # Build session as fathom_lab (not the currently-active darkflobi)
    s = make_session(
        auth_token=cookies["fathom_token"],
        ct0=cookies["ct0"],
        twid_uid=cookies["fathom_uid"],
    )

    print("verifying account...")
    screen = verify_account(s)
    if screen is None:
        print("  FAIL: could not determine screen_name. Not posting.")
        sys.exit(1)
    print(f"  screen_name: @{screen}")
    if screen.lower() != "fathom_lab":
        print(f"  ABORT: expected @fathom_lab, got @{screen}.")
        sys.exit(2)

    print("\nposting tweet:")
    print("---")
    print(TWEET_TEXT)
    print("---")
    r = post_tweet(s, TWEET_TEXT)
    print(f"\nHTTP {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        try:
            tw = (data.get("data", {})
                       .get("create_tweet", {})
                       .get("tweet_results", {})
                       .get("result", {}))
            tweet_id = tw.get("rest_id") or tw.get("legacy", {}).get("id_str")
            if tweet_id:
                url = f"https://x.com/fathom_lab/status/{tweet_id}"
                print(f"POSTED: {url}")
                return 0
        except Exception:
            pass
        print("response:", json.dumps(data, indent=2)[:1500])
    else:
        print("body:", r.text[:1500])
    return 3


if __name__ == "__main__":
    sys.exit(main())
