"""Targeted emails to researchers whose work is cited/used in the
cognometry paper. Warm since we explicitly credit their prior work.

- Potsawee Manakul (SelfCheckGPT author) — potsawee@stanford.edu
- PatronusAI team (HaluBench authors) — contact@patronus.ai

One email each. Specific. No asks beyond "would love your eyes."
"""
from __future__ import annotations

import json
import smtplib
import time
from email.mime.text import MIMEText
from pathlib import Path

CREDS = Path(r"C:\Users\heyzo\clawd\secrets\email-creds.json")

EMAILS = [
    {
        "to": "potsawee@stanford.edu",
        "name": "Potsawee",
        "subject": "SelfCheckGPT baseline in cognometry v0 — disconfirmation welcome",
        "body": """\
Hi Potsawee,

I cite SelfCheckGPT as one of the baselines in our cognometry v0 paper deposited today — the consistency-sampling consensus path in styxx's anthropic-hack pipeline is basically SelfCheckGPT adapted for closed-source APIs. Grateful for the original method.

The paper cross-validates a 9-signal hallucination detector across 8 public benchmarks (HaluEval QA/Dialog/Summ, TruthfulQA, four HaluBench subsets). AUC 0.998 on HaluEval-QA, 0.994 on TruthfulQA, 0.807 on HaluBench-RAGTruth. Two declared failure modes published openly in the weights module (HaluBench-DROP 0.424, HaluBench-FinanceBench 0.492).

It'd mean a lot if you had a chance to look at the comparison methodology — specifically where the consensus/consistency approach sits in the 9-signal ablation. I tried six heuristic fixes for DROP (all null) and wrote them up in papers/span-faithfulness-v0.md. If there's a SelfCheckGPT extension angle there you'd be interested in, happy to co-author.

Paper: https://doi.org/10.5281/zenodo.19703527
Code: https://github.com/fathom-lab/styxx
Manifesto: https://fathom.darkflobi.com/cognometry

No pressure or rush. Just wanted to put the citation + reproducer in your inbox in case useful.

Best,
Flobi
Fathom Lab
""",
    },
    {
        "to": "contact@patronus.ai",
        "name": "Patronus AI team",
        "subject": "HaluBench in cognometry v0 — cross-validation across all 4 subsets",
        "body": """\
Hi Patronus team,

Quick note: HaluBench is 4/8 of the benchmarks in a hallucination-detection paper we deposited on Zenodo today (cognometry v0). We pull DROP, PubMedQA, FinanceBench, and RAGTruth from your HaluBench release on Hugging Face and run our 9-signal detector across all of them with 3-seed averaging.

The honest headline is mixed news for you:

- HaluBench-RAGTruth: AUC 0.807 — our detector's best new number (RAG faithfulness is a real pain point, nice to have a benchmark that captures it)
- HaluBench-PubMedQA: AUC 0.719
- HaluBench-FinanceBench: AUC 0.492 (at chance) — published as a declared failure mode
- HaluBench-DROP: AUC 0.424 (below chance) — also declared failure mode

The "below chance" results are the part we'd most like to discuss. DROP hallucinations are extractive-span errors (wrong span, right passage — NLI entails them, novelty signals don't fire). FinanceBench hallucinations are arithmetic errors on verbatim-copied numbers (NLI + novelty both blind). Both failure modes are declared in our weights module itself so production callers know where the detector will lie.

If you're ever curating a list of systems evaluated on HaluBench, or if you have thoughts on the methodology (especially where DROP/FinanceBench might have a cleaner signal we missed), we'd welcome the input.

Paper: https://doi.org/10.5281/zenodo.19703527
Code: https://github.com/fathom-lab/styxx (MIT + CC-BY-4.0)
Leaderboard: https://fathom.darkflobi.com/cognometry/leaderboard

Thanks for shipping HaluBench publicly — this work would not exist without it.

Best,
Flobi
Fathom Lab
""",
    },
]


def main():
    creds = json.loads(CREDS.read_text(encoding="utf-8"))
    sender = creds["email"]
    pw = creds["password"]

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as s:
        s.login(sender, pw)
        for e in EMAILS:
            msg = MIMEText(e["body"], "plain", "utf-8")
            msg["Subject"] = e["subject"]
            msg["From"] = sender
            msg["To"] = e["to"]
            msg["Reply-To"] = sender
            try:
                s.sendmail(sender, [e["to"]], msg.as_string())
                print(f"  sent → {e['name']:18s} {e['to']}")
            except Exception as err:
                print(f"  FAIL → {e['name']:18s} {e['to']}: {type(err).__name__}: {err}")
            time.sleep(2)


if __name__ == "__main__":
    main()
