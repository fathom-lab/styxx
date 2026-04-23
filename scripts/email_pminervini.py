"""Follow-up email to Pasquale Minervini (HaluEval author) thanking
him for merging PR #55 into awesome-hallucination-detection, and
offering a reproducer + collaboration on the v4.2 span-faithfulness
extensions.

Sent from the user's gmail address. One email. No follow-up unless
reply arrives.
"""
from __future__ import annotations

import json
import smtplib
import sys
from email.mime.text import MIMEText
from pathlib import Path

CREDS = Path(r"C:\Users\heyzo\clawd\secrets\email-creds.json")

TO = "p.minervini@gmail.com"
SUBJECT = "Thank you for the merge — reproducer for HaluEval"

BODY = """\
Pasquale,

Thank you for the quick merge on #55 this morning. Appreciated.

Our paper leans heavily on HaluEval — the QA, Dialog, and Summarization subsets are 3/8 of the benchmarks in the cross-validation, alongside TruthfulQA and the four HaluBench subsets. AUC 0.998 on HaluEval-QA is carrying the headline number.

Two things you might find useful:

1. The 3-seed reproducer that generates our per-dataset AUCs (paired/unpaired, NLI on/off) runs from a single command and loads the HaluEval subsets directly from `pminervini/HaluEval` on the Hub. If a HaluEval-specific variant or a reproducer link would be useful on the dataset page, happy to PR it upstream:
   https://github.com/fathom-lab/styxx/blob/main/benchmarks/hallucination_test/cross_dataset_8bench_multiseed.py

2. We published two failures honestly — HaluBench-DROP (AUC 0.424) and HaluBench-FinanceBench (AUC 0.492) came in below chance. DROP is the more interesting one structurally: extractive-span hallucinations pass NLI entailment by construction, so none of our six heuristic fixes moved the number. The null probe + full analysis is in papers/span-faithfulness-v0.md. If you or a student have interest in a co-authored extension on span-level faithfulness as a v4.2 research track, the door is open.

Happy to run any benchmark at different random seeds if that would serve independent verification. No rush on either.

Thanks again,
Flobi
Fathom Lab
https://github.com/fathom-lab/styxx
https://fathom.darkflobi.com/cognometry
Paper DOI: https://doi.org/10.5281/zenodo.19703527
"""


def main():
    creds = json.loads(CREDS.read_text(encoding="utf-8"))
    sender = creds["email"]
    app_pw = creds["password"]

    msg = MIMEText(BODY, "plain", "utf-8")
    msg["Subject"] = SUBJECT
    msg["From"] = sender
    msg["To"] = TO
    msg["Reply-To"] = sender

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as s:
        s.login(sender, app_pw)
        s.sendmail(sender, [TO], msg.as_string())
    print(f"sent to {TO} from {sender}")


if __name__ == "__main__":
    main()
