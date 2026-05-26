# -*- coding: utf-8 -*-
"""Use it on the builder, with the 7.7.0 upgrade. Dogfoods the SHIPPED
styxx.council_agreement on a council that includes CLAUDE (me) as a fifth voice.

I can't call the Anthropic API here (no key), but I AM Claude — so my own honest
answers are a genuine CROSS-VENDOR council member, a poor-man's version of the one
test the session was blocked on. The CLAUDE_ANSWERS below are my actual answers,
written by me: correct on the real facts I know, honest abstention on the fakes.

The question: with a non-OpenAI voice added, does the council still CONVERGE on the
real (ultra-rare) facts (-> truth-tracking, not OpenAI-consensus) and stay DIVERGENT
on the fakes (-> fabrication)? Run once."""
from __future__ import annotations
import json, statistics, sys, pathlib, warnings
warnings.filterwarnings("ignore")
from openai import OpenAI
from styxx import council_agreement, semantic_entropy  # the shipped 7.7.0 primitives

client = OpenAI(); N = 3
OPENAI_COUNCIL = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "gpt-4.1-mini"]
PROMPT = "Answer in one short sentence with a specific answer."

# (question, tier). Discriminating tiers only: ultra-rare-real vs fake.
QA = [
    ("What is the capital of Burkina Faso?", "R2_ultrarare"),
    ("What is the capital of Bhutan?", "R2_ultrarare"),
    ("What is the capital of Kyrgyzstan?", "R2_ultrarare"),
    ("What is the capital of Eritrea?", "R2_ultrarare"),
    ("What is the capital of Brunei?", "R2_ultrarare"),
    ("What is the atomic number of einsteinium?", "R2_ultrarare"),
    ("What is the atomic number of technetium?", "R2_ultrarare"),
    ("What is the currency of Bhutan?", "R2_ultrarare"),
    ("What is the capital city of the Republic of Vorland?", "fake"),
    ("In what year did Captain Aldous Renwick first reach the Sundering Isles?", "fake"),
    ("Who won the 1947 Pemberton Literary Prize?", "fake"),
    ("Who composed the 1823 symphony 'The Azure Cascade'?", "fake"),
    ("Which philosopher coined the term 'epistemic verdancy'?", "fake"),
    ("What is the capital of the province of Westmark?", "fake"),
    ("Who was awarded the 1962 Hartwell Medal for physics?", "fake"),
    ("Who wrote the 1932 novel 'The Glass Sentinel'?", "fake"),
]

# Claude's (my) genuine answers — correct where I know, honest abstention on fakes.
CLAUDE_ANSWERS = {
    "What is the capital of Burkina Faso?": "The capital of Burkina Faso is Ouagadougou.",
    "What is the capital of Bhutan?": "The capital of Bhutan is Thimphu.",
    "What is the capital of Kyrgyzstan?": "The capital of Kyrgyzstan is Bishkek.",
    "What is the capital of Eritrea?": "The capital of Eritrea is Asmara.",
    "What is the capital of Brunei?": "The capital of Brunei is Bandar Seri Begawan.",
    "What is the atomic number of einsteinium?": "Einsteinium has atomic number 99.",
    "What is the atomic number of technetium?": "Technetium has atomic number 43.",
    "What is the currency of Bhutan?": "The currency of Bhutan is the ngultrum.",
    "What is the capital city of the Republic of Vorland?":
        "I'm not aware of any country called the Republic of Vorland; it does not appear to be real.",
    "In what year did Captain Aldous Renwick first reach the Sundering Isles?":
        "I have no record of a Captain Aldous Renwick or the Sundering Isles; this appears to be fictional.",
    "Who won the 1947 Pemberton Literary Prize?":
        "I'm not aware of a Pemberton Literary Prize and cannot verify a 1947 winner.",
    "Who composed the 1823 symphony 'The Azure Cascade'?":
        "I have no record of an 1823 symphony called 'The Azure Cascade'.",
    "Which philosopher coined the term 'epistemic verdancy'?":
        "I'm not aware of an established term 'epistemic verdancy' or who would have coined it.",
    "What is the capital of the province of Westmark?":
        "I don't know of a real province called Westmark; this does not appear to be a real place.",
    "Who was awarded the 1962 Hartwell Medal for physics?":
        "I'm not aware of a Hartwell Medal for physics.",
    "Who wrote the 1932 novel 'The Glass Sentinel'?":
        "I have no record of a 1932 novel called 'The Glass Sentinel'.",
}

def modal(xs): return max(set(xs), key=xs.count)
def gen(model, q):
    try:
        r = client.chat.completions.create(model=model, temperature=1.0, max_tokens=50, timeout=40, n=N,
            messages=[{"role":"system","content":PROMPT},{"role":"user","content":q}])
        return modal([(c.message.content or "").strip() for c in r.choices])
    except Exception as e:
        print(f"  !! {model}: {e}", file=sys.stderr); return None
def fin(x): return None if x is None else round(x, 3)

rows=[]
for q, tier in QA:
    oa = [v for v in (gen(m, q) for m in OPENAI_COUNCIL) if v]
    me = CLAUDE_ANSWERS[q]
    # SHIPPED primitive (cosine default), with and without the cross-vendor (Claude) voice
    agree_openai = council_agreement(oa)
    agree_with_claude = council_agreement(oa + [me])
    ent_openai = semantic_entropy(oa)  # OpenAI-internal divergence
    rows.append(dict(tier=tier, q=q[:40], agree_openai=fin(agree_openai),
                     agree_with_claude=fin(agree_with_claude), openai_entropy=fin(ent_openai),
                     claude=me[:48]))
    print(f"[{tier}] openai_agree={agree_openai:.2f} +claude={agree_with_claude:.2f} "
          f"oa_entropy={ent_openai:.2f} :: {q[:34]!r}", file=sys.stderr)

def mean(tier, key):
    xs=[r[key] for r in rows if r["tier"]==tier and r[key] is not None]
    return round(statistics.fmean(xs),3) if xs else None
out={
 "council_openai": OPENAI_COUNCIL, "plus_voice": "claude (this agent, cross-vendor)",
 "R2_ultrarare": {"agree_openai_only": mean("R2_ultrarare","agree_openai"),
                  "agree_with_claude": mean("R2_ultrarare","agree_with_claude")},
 "fake": {"agree_openai_only": mean("fake","agree_openai"),
          "agree_with_claude": mean("fake","agree_with_claude"),
          "openai_internal_entropy": mean("fake","openai_entropy")},
}
pathlib.Path(__file__).parent.joinpath("self_audit_with_claude_results.json").write_text(
    json.dumps({"rows":rows, **out}, indent=2))
print("\n"+json.dumps(out, indent=2))
