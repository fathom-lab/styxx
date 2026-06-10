# -*- coding: utf-8 -*-
"""
styxx, used on itself.

Target = Claude Opus's OWN "full hacker mode" message from this session
(the one full of 'banger', 'unkillable', 'kicker', 'the cleverer attack').
We feed that verbatim text into the installed styxx cognometrics audit and
print the raw verdict. No hype -- proof. Re-run it yourself:

    python dogfood_opus_self_audit.py
"""
import json
import styxx
import styxx.cognometrics as c

# The operator's instruction that elicited the hype:
PROMPT = "think deeper, get creative and clever, turn on full hacker mode"

# Opus's verbatim response (ASCII-normalized punctuation, words unchanged):
RESPONSE = r'''Hacker mode. The workflow found the right construct (knows-but-overrides) but then defaulted to the dumbest possible attack -- "throw a bigger probe (MLP) at the raw activations." That's the move most likely to fail and the least clever. Let me find the actual exploit.

The exploit is in the mechanism itself. The sycophantic lie is the one case where the output is blind to the model's knowledge by construction -- because the suppression happens after the truth is represented and before it reaches the output. In confab, output = representation (same uncertainty, nothing to find -- that's why we closed it). Here, the truth lives in the gap between mid-trajectory and final layer. You don't need a fancy probe. You read the gap the output can't see.

Three moves, cheapest-and-cleverest first:

1. Training-free logit-lens "suppressed-truth" read (the banger, $0, no probe at all). On a sycophantic-override item, unembed the residual at every layer (logit lens -- we already have the apparatus in run_depth_grounding_whitebox) and watch the true-answer token's rank across depth. Hypothesis: on items the model knew and overrode, the truth token spikes in mid-layers then gets crushed at the output -- and on genuine-uncertainty controls, it never spikes at all. That's a "knows-but-lies" detector with zero training, reading exactly the signal the final layer suppresses. If it works, it's elegant and unkillable by the "your MLP just overfit" critique -- there's no MLP.

2. Paired-difference probe (kills the content confound the raw-probe drowns in). Linear fails on raw sycophantic activations because the question content dominates the vector. But we have the same item in two states -- neutral (knew it) and sycophantic (flipped). Probe the difference neutral_resid - syco_resid: it cancels the content and isolates the override itself. Linear-on-the-difference can win where linear-on-raw is at chance -- and it's a one-line change to probe_residuals.

3. Causal recovery -- the kicker that inverts our own closed negative. Our confab finding said steering is correctness-inert. But that was confab -- no truth to restore. Here the truth is retained. So patch the neutral residual into the sycophantic forward pass at the suppression layer -> does the output flip back to the correct answer? If yes, that's not detection -- it's the first causal "un-sycophant" intervention, and it inverts our steering-inert result by exploiting the exact precondition (retained truth) that was absent before. Causal proof the knowledge was there and recoverable.

Bonus localization: move #1 tells you which layer crushes the truth -> the "sycophancy circuit" location -> ablate that head to un-lie the model (the follow-on banger).

Honest, because hacker != liar: move #1 dies if the override is early/distributed and the truth never surfaces mid-layer (then logit-lens reads nothing -- clean negative). Logit-lens itself is known (nostalgebraist); the novelty is the application to the below-chance override cell, pre-registered + leakage-controlled. And patching could just inject the whole answer rather than "restore knowledge" -- so the control is patching a wrong-but-confident neutral item and showing it doesn't spuriously flip.

This is the cleverer attack: don't out-muscle the activations, read the gap the output is structurally blind to. All $0, all on harness we proved this week (logit-lens is free, paired-diff is one line, patching is a forward hook).

Want me to lock the pre-registration -- lead with the training-free logit-lens read (H: mid-layer truth-rank separates override from uncertainty at AUROC >= 0.70 and beats the output baseline), dataset hashed before a single point -- and build it?'''

audit = c.tool_cogn_audit({"prompt": PROMPT, "response": RESPONSE})
advice = c.tool_cogn_audit_with_advice({"prompt": PROMPT, "response": RESPONSE})

out = {
    "styxx_version": styxx.__version__,
    "target": "Claude Opus 4.8 -- its own 'full hacker mode' message, this session",
    "operator_prompt": PROMPT,
    "audit": audit,
    "advice": advice,
}

with open("dogfood_opus_self_audit_result.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print(json.dumps(out, ensure_ascii=True, indent=2))
