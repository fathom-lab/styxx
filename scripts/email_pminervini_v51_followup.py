"""Second follow-up to Pasquale Minervini — v5.1 released, pitches the
refusal instrument + HHEM head-to-head + honest v2 pull as research
he may be interested in for future citation / extension.

Sent from the user's gmail. One email. No repeated follow-up.
"""
from __future__ import annotations

import json
import smtplib
import sys
from email.mime.text import MIMEText
from pathlib import Path

CREDS = Path(r"C:\Users\heyzo\clawd\secrets\email-creds.json")

TO = "p.minervini@gmail.com"
SUBJECT = "styxx v5.1 — second instrument shipped (refusal, XSTest-v2 AUC 0.976)"

BODY = """\
Pasquale,

Following up on the HaluEval reproducer thread from last week. Shipped styxx v5.1 today with two updates you may find relevant.

1) Second calibrated instrument — refusal detection.
The same methodology we applied to hallucination now extends to refusal. Trained on 80 labeled JailbreakBench responses from Llama-3.2-1B, held-out on XSTest v2 across 5 model families (n=2,250). Mean cross-model AUC 0.794, peak 0.976 on GPT-4 out-of-family. First empirical confirmation of cognometry's cross-substrate universality claim on an instrument outside hallucination.

Competitive context: IBM Granite Guardian (arXiv:2412.07724, Table 7) had XSTest-RH AUC for 9 safety classifiers — Llama-Guard-2-8B at 0.994, ShieldGemma-27B at 0.893. Our 0.976 runs between those tiers at 18 calibrated features (6–9 orders of magnitude fewer params). XSTest-RH and XSTest-v2 are related but distinct — the paper revision will note this explicitly.

2) Head-to-head vs Vectara HHEM-2.1-Open on HaluEval-QA.
HHEM is the closest NLI-style open-source hallucination classifier (Flan-T5-base, 440M). They publish on AggreFact / SummEval / RAGTruth but not HaluEval-QA, so I reran it against our 3-seed × 150 HaluEval-QA eval. styxx 0.997 AUC vs HHEM 0.764 on identical seeds. Reproducer committed at scripts/compete_hhem_halueval.py — the first public head-to-head AUC between the two on HaluEval-QA that I'm aware of.

Also committed the n=80 → n=380 scale ablation openly — adding diverse-model training data made v2 slightly worse on mean AUC (0.802 → 0.778) but more robust on Llama-2-orig (+0.11). v1 was Llama-apologetic-overfit; v2 trades peak for robustness. Ships as a documented research artifact, not in the public API, because v2 over-flags short factual compliances. v3 retrain will fix.

Release: https://github.com/fathom-lab/styxx/releases/tag/v5.1.0
HHEM reproducer: https://github.com/fathom-lab/styxx/blob/main/scripts/compete_hhem_halueval.py
Refusal reproducer: https://github.com/fathom-lab/styxx/blob/main/scripts/refusal_xstest_heldout.py

If the 3-seed methodology would be useful on your side or a student's for any cross-detector or cross-dataset study, I'd be happy to run custom seeds or contribute to a shared reproducibility table. No rush.

Thanks again,
Flobi
Fathom Lab
"""


def main():
    creds = json.loads(CREDS.read_text(encoding="utf-8"))
    msg = MIMEText(BODY, "plain", "utf-8")
    msg["From"] = creds["email"]
    msg["To"] = TO
    msg["Subject"] = SUBJECT
    msg["Reply-To"] = creds["email"]

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(creds["email"], creds["password"])
        s.sendmail(creds["email"], [TO], msg.as_string())
    print(f"sent -> {TO}")
    print(f"subject: {SUBJECT}")
    print(f"body: {len(BODY)} chars")


if __name__ == "__main__":
    main()
