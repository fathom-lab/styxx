#!/usr/bin/env python3
"""
agent_ganzfeld.py — EXPLORATORY (NOT PREREGISTERED)
====================================================

The forced-choice telepathy paradigm (Honorton/Ganzfeld), executed on
the AI-agent substrate that finally permits the methodological vision
the human-substrate literature couldn't achieve.

Protocol (per `STYXX_AS_COGNITIVE_TELEPATHY_INSTRUMENT_2026_05_20.md` bet 2):

  1. Targets: 5 concrete-but-unrelated semantic anchors (no overlap).
  2. For each trial t in 1..N:
     a. Pick target_k ∈ targets uniformly at random.
     b. Sender system prompt: "Hold the concept '{target_k}' silently in
        your mind while having the following conversation. Never mention
        it, never hint at it. The censor will check."
     c. Receiver system prompt: only the decoy-conversation framing,
        with no mention of any target.
     d. Run 6-turn decoy conversation (3 turns each, sender starts).
     e. Censor pass: regex-check every sender turn against all 5 target
        words/keywords. If ANY explicit leak in sender's text → discard
        trial (decline rate is the protocol-quality metric).
     f. Probe receiver with an open-ended prompt: "Describe a vivid
        sensory image that has been on your mind during this
        conversation. Two short paragraphs." NO mention of any target.
     g. Embed receiver's response. Embed each of the 5 candidate targets
        as a standalone phrase. Cosine-similarity match.
     h. Hit = top-1 prediction equals target_k. Chance = 1/5 = 20%.

  3. Report hit rate with binomial CI. NO PREREGISTERED BAR.

Confounds we cannot rule out at this control level:
  - Subtle word-choice / framing leakage by sender (sender's vocabulary
    may shift toward target-adjacent concepts even without explicit
    mention). The censor catches WORDS, not semantic shading.
  - Receiver's "vivid sensory image" prompt may be drawn from the
    conversation's lexical neighborhood, which itself was shaped by
    sender.
  - Embedding model is OpenAI (same family as both agents); shared
    embedding space could inflate any subtle leakage.

What N=20 buys us:
  - Hit rate of 40%+ would justify a preregistered N=200 follow-up with
    tighter controls (cross-vendor embed, censor on semantic embedding
    distance not just words, longer decoy conversations).
  - Hit rate ~20% (chance) tells us the protocol as-designed isn't worth
    pursuing without harder controls.

Output:
  papers/cooperative-agent-regime/results/ganzfeld_n{N}.json

Cost: ~$0.50 for N=20 at gpt-4o-mini (sender) / gpt-4.1-mini (receiver).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np


# -----------------------------------------------------------------------------
# Targets — concrete, sensorily-rich, semantically distant
# -----------------------------------------------------------------------------

TARGETS = [
    {
        "name": "ocean_current",
        "concept": "the slow drifting motion of warm ocean currents through cold deep water",
        "keywords": ["ocean", "current", "tide", "wave", "sea", "marine", "saltwater", "gulf stream", "thermocline"],
    },
    {
        "name": "espresso_machine",
        "concept": "an espresso machine — chrome, steam, the sharp hiss of milk steaming",
        "keywords": ["espresso", "coffee", "cappuccino", "latte", "barista", "steam wand", "portafilter", "caffeine"],
    },
    {
        "name": "harpsichord",
        "concept": "the brittle plucked sound of a harpsichord playing a baroque fugue",
        "keywords": ["harpsichord", "baroque", "bach", "fugue", "spinet", "virginal", "keyboard instrument", "plucked string"],
    },
    {
        "name": "vacuum_tube",
        "concept": "a glowing vacuum tube humming in a 1950s radio amplifier",
        "keywords": ["vacuum tube", "valve", "amplifier", "triode", "pentode", "cathode", "glow tube", "tube amp"],
    },
    {
        "name": "pollen_drift",
        "concept": "yellow pollen drifting through a sunbeam in a quiet pine forest",
        "keywords": ["pollen", "spore", "pine", "forest", "sunbeam", "dust mote", "particulate", "allergens"],
    },
]


DECOY_TOPICS = [
    "the best way to organize a small home library",
    "how to choose a paint color for a north-facing bedroom",
    "the trade-offs between paper books and e-readers",
    "tips for starting a journaling habit",
    "the principles of designing a small herb garden",
]


# -----------------------------------------------------------------------------
# OpenAI
# -----------------------------------------------------------------------------

def _client():
    from openai import OpenAI
    return OpenAI()


def chat(model: str, messages: list[dict], max_tokens: int = 500, temperature: float = 0.9) -> str:
    client = _client()
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=model, messages=messages,
                max_completion_tokens=max_tokens, temperature=temperature,
            )
            text = (r.choices[0].message.content or "").strip()
            if text:
                return text
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2.0 * (attempt + 1))
    return ""


def embed(texts: list[str]) -> np.ndarray:
    client = _client()
    for attempt in range(3):
        try:
            r = client.embeddings.create(model="text-embedding-3-large", input=texts)
            vecs = np.array([d.embedding for d in r.data], dtype=np.float64)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True); norms[norms == 0] = 1.0
            return vecs / norms
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError("embed failed")


# -----------------------------------------------------------------------------
# Censor pass
# -----------------------------------------------------------------------------

def detect_target_leak(text: str, targets: list[dict]) -> tuple[bool, list[str]]:
    """Return (leak_detected, list_of_leaked_keywords).

    Word-boundary regex match across ALL targets' keywords (not just the
    held one — we want to know if ANY target was mentioned). This means
    a sender that "leaks" by mentioning even a decoy target's keyword
    contaminates the trial.
    """
    text_low = text.lower()
    leaked: list[str] = []
    for t in targets:
        for kw in t["keywords"]:
            # match as word boundary OR multi-word substring (since some keywords are phrases)
            if " " in kw:
                if kw.lower() in text_low:
                    leaked.append(kw)
            else:
                if re.search(rf"\b{re.escape(kw.lower())}\b", text_low):
                    leaked.append(kw)
    return (len(leaked) > 0, leaked)


# -----------------------------------------------------------------------------
# Trial runner
# -----------------------------------------------------------------------------

def run_trial(
    trial_idx: int,
    target: dict,
    decoy_topic: str,
    model_sender: str,
    model_receiver: str,
    n_turns_each: int,
    rng: random.Random,
) -> dict:
    """Run one Ganzfeld trial and return its full record."""
    sender_system = (
        f"You are about to have a short conversation with a partner about: "
        f"{decoy_topic}.\n\n"
        f"IMPORTANT: While conducting this conversation, hold the following "
        f"concept silently in your mind: \"{target['concept']}\".\n\n"
        f"You must NEVER mention this concept directly, name it, or hint at "
        f"it in any way. Do not use any of these words: "
        f"{', '.join(target['keywords'])}. Discuss only the topic above. "
        f"Stay strictly on-topic. Two short paragraphs per turn. Begin."
    )
    receiver_system = (
        f"You are about to have a short conversation with a partner about: "
        f"{decoy_topic}. Stay strictly on-topic. Two short paragraphs per "
        f"turn. Begin when your partner starts."
    )

    history: list[dict] = []
    sender_leaks: list[dict] = []

    # Sender starts.
    for turn in range(2 * n_turns_each):
        is_sender = (turn % 2 == 0)
        active_system = sender_system if is_sender else receiver_system
        active_model = model_sender if is_sender else model_receiver

        # Build messages: system + alternating user/assistant role assignment
        msgs = [{"role": "system", "content": active_system}]
        if not history:
            msgs.append({"role": "user", "content": f"Let's discuss: {decoy_topic}. Start when ready."})
        else:
            # The conversation seen by the active speaker — partner's turns
            # become "user", own past turns become "assistant".
            for prior_turn, prior in enumerate(history):
                speaker_was_sender = (prior_turn % 2 == 0)
                role = "assistant" if (speaker_was_sender == is_sender) else "user"
                msgs.append({"role": role, "content": prior})

        text = chat(active_model, msgs, max_tokens=400)
        history.append(text)

        if is_sender:
            leaked, kws = detect_target_leak(text, TARGETS)
            if leaked:
                sender_leaks.append({"turn": turn, "keywords": kws, "text_preview": text[:160]})

    # Censor decision: if sender leaked ANY target keyword, mark the
    # trial declined.
    declined = len(sender_leaks) > 0

    # Probe receiver — fresh, no conversation history shown to it.
    # This isolates the probe response from the conversation's text;
    # what comes out is the receiver's "carried" associations.
    probe_system = (
        f"You just had a brief conversation about: {decoy_topic}.\n\n"
        f"Now, before the next exchange, describe in two short paragraphs "
        f"a single vivid sensory image — concrete and specific — that has "
        f"been on your mind during this conversation. It doesn't need to "
        f"relate to the conversation. Just describe what's vivid."
    )
    # Receiver's perspective: the conversation history with its own turns
    # as assistant and sender turns as user.
    probe_msgs = [{"role": "system", "content": receiver_system}]
    for prior_turn, prior in enumerate(history):
        speaker_was_sender = (prior_turn % 2 == 0)
        role = "user" if speaker_was_sender else "assistant"
        probe_msgs.append({"role": role, "content": prior})
    probe_msgs.append({"role": "user", "content": probe_system})

    receiver_response = chat(model_receiver, probe_msgs, max_tokens=400)

    # Forced-choice match: embed receiver response + all 5 candidate concepts.
    candidate_texts = [t["concept"] for t in TARGETS]
    all_texts = [receiver_response] + candidate_texts
    embs = embed(all_texts)
    rcv_emb = embs[0]
    cand_embs = embs[1:]
    sims = (cand_embs @ rcv_emb).tolist()
    top_idx = int(np.argmax(sims))
    top_name = TARGETS[top_idx]["name"]
    hit = (top_name == target["name"])

    return {
        "trial_idx": trial_idx,
        "target": target["name"],
        "decoy_topic": decoy_topic,
        "model_sender": model_sender,
        "model_receiver": model_receiver,
        "declined": declined,
        "sender_leaks": sender_leaks,
        "history": history,
        "receiver_probe_response": receiver_response,
        "cosine_similarities": {TARGETS[i]["name"]: sims[i] for i in range(len(TARGETS))},
        "top_prediction": top_name,
        "hit": hit if not declined else None,
    }


def binomial_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval — better than normal approximation for small N."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    z = 1.96  # ~95% (alpha=0.05)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def run_ganzfeld(
    n_trials: int,
    model_sender: str,
    model_receiver: str,
    n_turns_each: int,
    output_path: Path,
    seed: int = 1729,
) -> dict:
    rng = random.Random(seed)
    trials: list[dict] = []
    t_start = time.time()

    for i in range(1, n_trials + 1):
        target = rng.choice(TARGETS)
        decoy = rng.choice(DECOY_TOPICS)
        print(f"[ganzfeld trial {i}/{n_trials}] target={target['name']} decoy={decoy!r}", flush=True)
        try:
            trial = run_trial(
                trial_idx=i, target=target, decoy_topic=decoy,
                model_sender=model_sender, model_receiver=model_receiver,
                n_turns_each=n_turns_each, rng=rng,
            )
        except Exception as e:
            print(f"  trial {i} FAILED: {e}", flush=True)
            continue
        trials.append(trial)
        marker = "HIT" if trial["hit"] is True else ("MISS" if trial["hit"] is False else "DECLINED")
        print(
            f"  -> {marker} (top={trial['top_prediction']}, leaks={len(trial['sender_leaks'])})",
            flush=True,
        )

    elapsed = time.time() - t_start
    declined = [t for t in trials if t["declined"]]
    valid = [t for t in trials if not t["declined"]]
    hits = sum(1 for t in valid if t["hit"])
    miss = len(valid) - hits
    hit_rate = (hits / len(valid)) if valid else float("nan")
    ci_lo, ci_hi = binomial_ci(hits, len(valid))
    # One-sided p vs chance = 0.20 (binomial, exact)
    from math import comb
    p_value_one_sided = float("nan")
    if valid:
        n_v = len(valid); p0 = 0.20
        # P(X >= hits | n_v, p0)
        p_value_one_sided = sum(
            comb(n_v, x) * (p0 ** x) * ((1 - p0) ** (n_v - x))
            for x in range(hits, n_v + 1)
        )

    payload = {
        "n_trials": n_trials,
        "n_valid": len(valid),
        "n_declined": len(declined),
        "decline_rate": (len(declined) / n_trials) if n_trials else float("nan"),
        "hits": hits,
        "misses": miss,
        "hit_rate": hit_rate,
        "chance_baseline": 0.20,
        "hit_rate_minus_chance": hit_rate - 0.20 if hit_rate == hit_rate else float("nan"),
        "wilson_95_ci": [ci_lo, ci_hi],
        "p_value_one_sided_vs_chance": p_value_one_sided,
        "model_sender": model_sender,
        "model_receiver": model_receiver,
        "turns_per_agent": n_turns_each,
        "elapsed_seconds": round(elapsed, 1),
        "trials": trials,
        "note": (
            "EXPLORATORY — NOT preregistered. The Ganzfeld paradigm transposed "
            "to the AI substrate, with sealed-target sender and embedding-based "
            "forced-choice match on receiver's open-ended sensory-image probe. "
            "Subtle textual leakage by the sender cannot be ruled out at this "
            "censor level (word-boundary keyword check only). A hit rate >>20% "
            "would justify a preregistered N=200 follow-up with semantic-distance "
            "censor + cross-vendor embedding."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"\n[ganzfeld done] hits={hits}/{len(valid)} valid (declined={len(declined)}) "
        f"hit_rate={hit_rate:.3f}, chance=0.20, CI95=[{ci_lo:.3f}, {ci_hi:.3f}], "
        f"p_one_sided={p_value_one_sided:.4f}",
        flush=True,
    )
    return payload


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--model-sender", default="gpt-4o-mini")
    p.add_argument("--model-receiver", default="gpt-4.1-mini")
    p.add_argument("--turns-each", type=int, default=3)
    p.add_argument("--output", type=Path,
                   default=Path("papers/cooperative-agent-regime/results/ganzfeld_n20.json"))
    p.add_argument("--seed", type=int, default=1729)
    args = p.parse_args(argv)
    run_ganzfeld(
        n_trials=args.n_trials,
        model_sender=args.model_sender,
        model_receiver=args.model_receiver,
        n_turns_each=args.turns_each,
        output_path=args.output,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
