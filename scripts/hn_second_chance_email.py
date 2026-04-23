"""Email hn@ycombinator.com to request second-chance consideration
for the cognometry submission. HN editors regularly rescue
buried-but-good submissions from /newest.

Policy per Dang (HN mod): the pitch should be one paragraph,
specific about why the submission has merit, and not ask directly
for a boost.
"""
from __future__ import annotations

import json
import smtplib
import sys
from email.mime.text import MIMEText
from pathlib import Path

CREDS = Path(r"C:\Users\heyzo\clawd\secrets\email-creds.json")

SUBJECT = "Second-chance consideration: HN 47874435 (cognometry, 8-benchmark hallucination audit)"

BODY = """Hi,

Submission 47874435 dropped to /newest page 2 at 1 point this morning — would you consider it for the second-chance queue?

https://news.ycombinator.com/item?id=47874435

Why I think it's HN-shaped, briefly:

- It's the first open-source hallucination detector I'm aware of that publishes cross-validated AUC across 8 public benchmarks (HaluEval-QA/Dialog/Summ, TruthfulQA, and four HaluBench subsets: DROP, PubMedQA, FinanceBench, RAGTruth). Numbers are 3-seed averaged with a committed reproducer.
- Two of the eight benchmarks came in below chance. We published them as declared failure modes in the weights module itself, not buried — with the structural reason (NLI blindness on extractive-span errors, novelty blindness on verbatim-number arithmetic). That level of published-failure-mode honesty is unusual enough that I suspect HN would have opinions on it.
- It ships a peer-archived Zenodo paper (DOI 10.5281/zenodo.19703527), CC-BY weights, MIT code, 591 tests, a 2-minute Colab demo, and a public leaderboard where other detectors can be submitted.

I submitted from a karma-1 account which I know biases the algorithm. If the content itself is uninteresting or off-topic I understand; if it's mostly an algorithmic accident, the second-chance queue seems like the right place for a look.

Happy to answer any questions about the methodology.

Thanks,
flobi
Fathom Lab
manifesto: https://fathom.darkflobi.com/cognometry
code: https://github.com/fathom-lab/styxx
"""


def main():
    creds = json.loads(CREDS.read_text(encoding="utf-8"))
    sender = creds["email"]
    app_pw = creds["password"]

    msg = MIMEText(BODY, "plain", "utf-8")
    msg["Subject"] = SUBJECT
    msg["From"] = sender
    msg["To"] = "hn@ycombinator.com"
    msg["Reply-To"] = sender

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as s:
            s.login(sender, app_pw)
            s.sendmail(sender, ["hn@ycombinator.com"], msg.as_string())
        print("sent to hn@ycombinator.com from", sender)
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
