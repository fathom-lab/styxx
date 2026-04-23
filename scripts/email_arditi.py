"""Warm email to Andy Arditi — his 'Refusal in Language Models Is
Mediated by a Single Direction' is the paper cognometry Law III
explicitly replicates at Llama-3.2-1B. Strongest possible citation
relationship.
"""
from __future__ import annotations

import json
import smtplib
from email.mime.text import MIMEText
from pathlib import Path

CREDS = Path(r"C:\Users\heyzo\clawd\secrets\email-creds.json")

TO = "andyrdt@gmail.com"
SUBJECT = "Replicated your refusal-direction paper at 1B — cognometry v0"

BODY = """\
Hi Andy,

Quick note because your refusal-direction paper is the single most-directly-cited prior work in the cognometry v0 paper we deposited today. "Law III" of the manifesto is essentially a reproduction of your main result at smaller open-weight scale:

- refuse@unsafe drops 97% → 17% at α=3.0 with multi-position residual patching, on meta-llama/Llama-3.2-1B-Instruct (n=60 JBB test split)
- Asymmetry holds: inducing refusal on safe prompts barely moves the rate (0.13 → 0.17 at α=3.0)
- Single-direction refusal probe trained with supervised contrast; same subtract-project protocol as your paper

I didn't want to ship a replication of your result without letting you know. If the numbers look off at a seed you'd want to check, or if there's a specific extension you'd want us to run while the infrastructure's warm, let me know.

There's also a broader frame in the manifesto — three laws of "cognometry" (measurement of cognitive states in LLMs). Law II is a cross-vendor direction-transfer study (cos +0.464 llama-1B → llama-3B; null on qwen → phi-3.5). That piece owes your work a lot of the methodological vocabulary.

Paper: https://doi.org/10.5281/zenodo.19703527
Code + reproducer: https://github.com/fathom-lab/styxx
Manifesto: https://fathom.darkflobi.com/cognometry

No ask — mostly wanted the citation + reproducer on record. Happy to run additional α-sweeps, layer ablations, or different model families if there's a variant you'd want nailed down.

Best,
Flobi
Fathom Lab
"""


def main():
    creds = json.loads(CREDS.read_text(encoding="utf-8"))
    sender = creds["email"]
    pw = creds["password"]

    msg = MIMEText(BODY, "plain", "utf-8")
    msg["Subject"] = SUBJECT
    msg["From"] = sender
    msg["To"] = TO
    msg["Reply-To"] = sender

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as s:
        s.login(sender, pw)
        s.sendmail(sender, [TO], msg.as_string())
    print(f"sent to {TO} from {sender}")


if __name__ == "__main__":
    main()
