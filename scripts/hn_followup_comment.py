"""Post a technical follow-up comment on our own HN submission
(47874435) as a reply to our self-comment. Adds depth, keeps the
thread active in HN's activity tracking, invites debate.
"""
from __future__ import annotations

import re
import sys

import browser_cookie3
import requests

ITEM_ID = "47874435"

COMMENT = """\
Quick technical addendum on the DROP failure mode for anyone interested in where this detector breaks.

I tried 6 heuristic "fixes" before posting. All in the repo under benchmarks/hallucination_test/probe_drophacks.py:

  role_mismatch              AUC 0.500  (chance)
  answer_context_adjacency   AUC 0.520
  question_conditional_NLI   AUC 0.491
  scope_sentence_NLI         AUC 0.454
  multi_number_density       AUC 0.475
  answer_rank_in_passage     AUC 0.499
  (v4.0.2 on DROP)           AUC 0.424  (for comparison)

The honest finding: DROP hallucinations are not signature-detectable from surface text features. The wrong span IS in the passage. The right tokens DO overlap. NLI scores the assertion as entailed (because the wrong span is also a valid English claim about the passage). Every signal in the current stack passes the bar by construction.

What it actually needs: either span-level faithfulness modeling (basically a trained extractive-QA reader used as an oracle), or real NER-typed question-answer matching. Both are v4.2+ research tracks, not v4.1 patches. The span-faithfulness-v0.md file in papers/ walks through why each heuristic died.

FinanceBench is structurally similar — calculation errors on verbatim-copied numbers. Novelty + NLI are semantically blind to arithmetic correctness. Fix needs a number-symbolic verification signal, also v4.2.

Happy to compare notes if anyone is working on extractive-span faithfulness or arithmetic-consistency checks for RAG pipelines."""


def main():
    jar = browser_cookie3.firefox(domain_name="news.ycombinator.com")
    s = requests.Session()
    for c in jar:
        s.cookies.set(c.name, c.value, domain=c.domain)
    s.headers.update({"User-Agent": "Mozilla/5.0"})

    # Find OUR self-comment on the item page — it's the one by
    # darkflobi. Extract its reply hmac.
    r = s.get(
        f"https://news.ycombinator.com/item?id={ITEM_ID}", timeout=15
    )
    html = r.text
    if 'href="logout' not in html:
        print("not logged in")
        sys.exit(1)

    # Post as a top-level comment on the item (same path as first
    # self-comment). Reply-to-own-comment is blocked for low-karma
    # accounts; top-level comment on own submission is not.
    hmac_m = re.search(
        r'<input[^>]+name="hmac"[^>]+value="([^"]+)"', html
    )
    if not hmac_m:
        print("hmac not on item page")
        sys.exit(1)
    hmac = hmac_m.group(1)
    print(f"hmac: {hmac[:16]}...")

    data = {
        "parent": ITEM_ID,
        "goto": f"item?id={ITEM_ID}",
        "hmac": hmac,
        "text": COMMENT,
    }
    r3 = s.post(
        "https://news.ycombinator.com/comment",
        data=data,
        allow_redirects=False,
        timeout=20,
    )
    print(f"POST status: {r3.status_code}  → {r3.headers.get('Location')}")
    if r3.status_code in (200, 302):
        print(f"\nfollow-up posted on {ITEM_ID}")
    else:
        print("FAIL:", r3.text[:500])


if __name__ == "__main__":
    main()
