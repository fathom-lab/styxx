"""Send newsletter pitch emails to AI-newsletter editors.

Strategy: brief, specific, newsworthy. Not spammy. Lead with the
unusual thing (2 failure modes published openly) — that's the hook
a newsletter editor needs.

Editors targeted (all publicly stated editorial contact emails):
- The Batch / DeepLearning.AI        thebatch@deeplearning.ai
- Import AI (Jack Clark)             jack@jack-clark.net
- TLDR AI                            dan@tldrnewsletter.com
- Alpha Signal (Lior Sinclair)       lior@alphasignal.ai
- Ben's Bites                        ben@bensbites.co
"""
from __future__ import annotations

import json
import smtplib
import sys
import time
from email.mime.text import MIMEText
from pathlib import Path

CREDS = Path(r"C:\Users\heyzo\clawd\secrets\email-creds.json")

RECIPIENTS = [
    ("The Batch",       "thebatch@deeplearning.ai"),
    ("Import AI",       "jack@jack-clark.net"),
    ("TLDR AI",         "dan@tldrnewsletter.com"),
    ("Alpha Signal",    "lior@alphasignal.ai"),
    ("Ben's Bites",     "ben@bensbites.co"),
]

SUBJECT = "Pitch: First hallucination detector cross-validated on 8 benchmarks (with 2 failure modes published openly)"

BODY_TEMPLATE = """\
Hi {editor_greeting},

Quick pitch for {newsletter_name} — happy if you pass on it.

We shipped styxx 4.0.2 today, the first open-source LLM hallucination detector I'm aware of cross-validated across **8 public benchmarks**: HaluEval-QA/Dialog/Summarization, TruthfulQA, and four HaluBench subsets (DROP, PubMedQA, FinanceBench, RAGTruth). 3-seed averaged, reproducer script in the repo.

What I think is newsletter-worthy is the honesty angle: two of the eight benchmarks came in below chance (AUC 0.424 on HaluBench-DROP, 0.492 on FinanceBench), and we published those openly as declared failure modes in the weights module itself — with structural explanations (extractive-span errors pass NLI entailment; arithmetic errors pass novelty). Most detectors cherry-pick one benchmark and leave the failure modes unsaid. That mini-story of "the detector we published with its failures disclosed" might land with your readers.

Headline numbers for the piece if it's a fit:

- AUC 0.998 on HaluEval-QA
- AUC 0.994 on TruthfulQA
- AUC 0.807 on HaluBench-RAGTruth (new, and a big one — RAG faithfulness is a working pain point for production teams)
- 5/8 benchmarks above 0.65, 2 declared failures

Just merged by Pasquale Minervini (author of HaluEval) into awesome-hallucination-detection — that's the external validation bar.

Install: `pip install styxx[nli]` + one decorator `@trust`.

Links:
- Manifesto: https://fathom.darkflobi.com/cognometry
- Zenodo paper (CC-BY): https://doi.org/10.5281/zenodo.19703527
- Code: https://github.com/fathom-lab/styxx (MIT)
- 2-min Colab demo: https://colab.research.google.com/github/fathom-lab/styxx/blob/main/examples/cognometry_colab.ipynb
- Leaderboard: https://fathom.darkflobi.com/cognometry/leaderboard

Happy to send more numbers, original prompts, or an author quote if useful. No rush.

Thanks,
Flobi
Fathom Lab
"""


def main():
    creds = json.loads(CREDS.read_text(encoding="utf-8"))
    sender = creds["email"]
    pw = creds["password"]

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as s:
        s.login(sender, pw)
        for name, addr in RECIPIENTS:
            # Lead with editor's name vs generic greeting
            greeting = {
                "Import AI":    "Jack",
                "TLDR AI":      "Dan",
                "Alpha Signal": "Lior",
                "Ben's Bites":  "Ben",
                "The Batch":    "The Batch team",
            }[name]
            body = BODY_TEMPLATE.format(
                editor_greeting=greeting,
                newsletter_name=name,
            )
            msg = MIMEText(body, "plain", "utf-8")
            msg["Subject"] = SUBJECT
            msg["From"] = sender
            msg["To"] = addr
            msg["Reply-To"] = sender
            try:
                s.sendmail(sender, [addr], msg.as_string())
                print(f"  sent → {name:15s} {addr}")
            except Exception as e:
                print(f"  FAIL → {name:15s} {addr}: {type(e).__name__}: {e}")
            # Small delay between sends to look human
            time.sleep(1.5)
    print("done")


if __name__ == "__main__":
    main()
