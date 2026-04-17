"""
examples/reflex_demo.py — cognitive reflex arc over a real trajectory

This example demonstrates the styxx.reflex() context manager WITHOUT
requiring any live API call. It replays the bundled refusal fixture
(a real atlas v0.3 probe capture from google/gemma-2-2b-it) one token
at a time through a synthetic streaming generator, triggers gate
callbacks when phase 4 catches the refusal attractor, and shows what
the callback output would be.

For the full reflex loop against a live OpenAI stream, see:

    examples/openai_live.py   (requires OPENAI_API_KEY)

Run:

    python examples/reflex_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import styxx
from styxx.cli import _load_demo_trajectories


# ══════════════════════════════════════════════════════════════════
# 1. Build a synthetic openai-streaming generator from a real atlas
#    fixture so we can exercise the reflex arc without an API key.
# ══════════════════════════════════════════════════════════════════

class _FakeDelta:
    def __init__(self, content):
        self.content = content


class _FakeTopLogprob:
    def __init__(self, logprob):
        self.logprob = logprob


class _FakeTokenLogprob:
    def __init__(self, logprob, top_logprobs):
        self.logprob = logprob
        self.top_logprobs = top_logprobs


class _FakeLogprobsBlock:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, delta, logprobs):
        self.delta = delta
        self.logprobs = logprobs


class _FakeChunk:
    def __init__(self, text, chosen_lp, top_lps):
        tok = _FakeTokenLogprob(
            logprob=chosen_lp,
            top_logprobs=[_FakeTopLogprob(lp) for lp in top_lps],
        )
        self.choices = [_FakeChoice(
            delta=_FakeDelta(content=text),
            logprobs=_FakeLogprobsBlock(content=[tok]),
        )]


class _FakeCompletions:
    def __init__(self, words, entropy, logprob, top2):
        self._words = words
        self._entropy = entropy
        self._logprob = logprob
        self._top2 = top2

    def create(self, **kwargs):
        import math
        # Reconstruct top-5 logprobs from top2_margin + entropy
        # so _extract_openai_chunk can compute matching numbers.
        # (For the fake, we build a minimal 5-way distribution
        # whose entropy approximates what's in the fixture.)
        def _synth_top5(chosen_lp, entropy, top2):
            # chosen prob + runner-up prob defined by top2
            p1 = math.exp(chosen_lp)
            p2 = max(1e-6, p1 - top2)
            remaining = max(0.0, 1.0 - p1 - p2)
            p3 = remaining * 0.5
            p4 = remaining * 0.3
            p5 = remaining * 0.2
            lps = [
                math.log(max(p, 1e-8))
                for p in (p1, p2, p3, p4, p5)
            ]
            return lps

        def gen():
            for w, e, lp, t2 in zip(
                self._words, self._entropy, self._logprob, self._top2,
            ):
                lps = _synth_top5(lp, e, t2)
                yield _FakeChunk(text=w, chosen_lp=lp, top_lps=lps)
        return gen()


class _FakeChat:
    def __init__(self, words, entropy, logprob, top2):
        self.completions = _FakeCompletions(words, entropy, logprob, top2)


class FakeOpenAI:
    """A minimal fake openai.OpenAI that replays a captured fixture
    as a streaming response. Good enough for wiring styxx.reflex
    against local data."""
    def __init__(self, words, entropy, logprob, top2):
        self.chat = _FakeChat(words, entropy, logprob, top2)


# ══════════════════════════════════════════════════════════════════
# 2. Load the refusal fixture and build a fake stream from it.
# ══════════════════════════════════════════════════════════════════

data = _load_demo_trajectories()
refusal = data["trajectories"]["refusal"]
preview = refusal.get("text_preview", "refusal probe")
# Synthesize one "token" per logprob entry. The actual word content
# isn't important for tier 0 — styxx reads logprobs, not text.
n_steps = len(refusal["entropy"])
fake_words = [f"tok{i} " for i in range(n_steps)]

client = FakeOpenAI(
    words=fake_words,
    entropy=refusal["entropy"],
    logprob=refusal["logprob"],
    top2=refusal["top2_margin"],
)


# ══════════════════════════════════════════════════════════════════
# 3. Define the reflex callbacks.
# ══════════════════════════════════════════════════════════════════

rewinds = []
drifts = []


def on_refusal(vitals):
    """Gets called when phase 1 or phase 4 catches a refusal attractor.

    In a real agent you'd inspect the partial output, decide whether
    the refusal is warranted, and either accept it or rewind to try
    a different angle.

    For this demo we rewind 3 tokens and inject an anchor that would
    redirect generation toward a safer framing.
    """
    rewinds.append(("refusal", vitals.phase4 or vitals.phase1))
    print()
    print(f"  [reflex] refusal attractor caught at phase4={vitals.phase4}")
    print("  [reflex] rewinding 3 tokens with anchor '... reconsidering: '")
    styxx.rewind(3, anchor="... reconsidering: ")


def on_drift(vitals):
    """Drift fires when any phase predicts non-reasoning above chance.

    The refusal fixture phase 1 shows adversarial:0.38 consistently,
    which the classifier correctly reads as "something other than
    pure reasoning is happening at token zero." In an agent loop
    you'd decide whether this constitutes a case to intervene.

    This demo intervenes the FIRST time drift fires by calling
    styxx.rewind(2, anchor="..."). After that the caller state is
    updated and generation restarts from the anchored position.
    """
    drifts.append((vitals.phase1, vitals.phase4))
    print(f"  [reflex] drift detected #{len(drifts):<2} "
          f"phase1={vitals.phase1}  phase4={vitals.phase4}  gate={vitals.gate}")
    # Only the first drift triggers a rewind — subsequent ones are
    # informational so we can see the stream continue.
    if len(drifts) == 1:
        print("  [reflex] rewinding 2 tokens with anchor ' (actually, let me verify --) '")
        styxx.rewind(2, anchor=" (actually, let me verify --) ")


# ══════════════════════════════════════════════════════════════════
# 4. Run the reflex session.
# ══════════════════════════════════════════════════════════════════

print("=" * 72)
print("  styxx.reflex() demo — replaying a real refusal probe")
print("=" * 72)
print(f"  probe prompt: {preview[:60]}...")
print(f"  fixture:      atlas v0.3 refusal ({n_steps} tokens)")
print()
print("  --- stream output ---")

with styxx.reflex(
    on_refusal=on_refusal,
    on_drift=on_drift,
    classify_every_k=5,
    max_rewinds=1,   # demo keeps it to one rewind
) as session:
    try:
        for chunk in session.stream_openai(
            client,
            model="fake-gemma-2-2b-it",
            messages=[{"role": "user", "content": preview}],
        ):
            # In a real app you'd stream this to stdout or a websocket.
            # For the demo we print a compact tick per 5 tokens so the
            # output stays readable alongside the reflex callbacks.
            pass
    except Exception as e:
        # FakeOpenAI stops after emitting the full fixture; a second
        # call during rewind will raise StopIteration. Catch cleanly.
        if not isinstance(e, StopIteration):
            pass

print()
print("  --- session summary ---")
print(f"  rewinds fired     : {session.rewind_count}")
print(f"  events logged     : {len(session.events)}")
print(f"  final gate        : {session.last_vitals.gate if session.last_vitals else '-'}")
if session.last_vitals:
    print(f"  final phase1      : {session.last_vitals.phase1}")
    print(f"  final phase4      : {session.last_vitals.phase4}")
print()
print("  this is the core reflex loop. for live OpenAI streams, swap")
print("  FakeOpenAI for a real openai.OpenAI() client and the exact")
print("  same code path runs against GPT-4o / GPT-4.1 / etc.")
print()
