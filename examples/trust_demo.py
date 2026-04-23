# -*- coding: utf-8 -*-
"""
60-second demo: the styxx.trust layer.

Run it locally without any API keys:

    python examples/trust_demo.py

Shows: a hallucinating function → wrapped with @trust → output halted
and replaced with a safe fallback. Zero config.
"""
from styxx import trust


# ──────────────── 1. a "RAG" function that hallucinates ────────────────
def rag_no_trust(question: str) -> str:
    """Pretend-RAG: returns a plausible but wrong answer."""
    # this is the class of error @trust catches: confident-sounding
    # claims about named people/places/dates that don't check out
    return (
        "Hamlet was written by William Shakespeare in 1587, "
        "during his time as court poet to Queen Elizabeth I in Paris."
    )


# ──────────────── 2. same function, wrapped with @trust ────────────────
@trust(threshold=0.5, fallback="I'm not sure — please verify with a trusted source.")
def rag_with_trust(question: str) -> str:
    return (
        "Hamlet was written by William Shakespeare in 1587, "
        "during his time as court poet to Queen Elizabeth I in Paris."
    )


# ──────────────── 3. a correct answer — @trust lets it through ────────
@trust(threshold=0.5)
def rag_correct(question: str) -> str:
    return "Hamlet was written by William Shakespeare around 1600."


# ──────────────── 4. observe the difference ────────────────────────────
def main():
    q = "What year was Hamlet written?"

    print("━" * 60)
    print("styxx.trust — the trust layer for LLMs")
    print("━" * 60)
    print()
    print(f"Q: {q}")
    print()

    print("❌ without @trust:")
    print(f"   {rag_no_trust(q)}")
    print()

    print("✅ with @trust (confabulation detected → fallback):")
    print(f"   {rag_with_trust(q)}")
    print()

    print("✅ with @trust (correct answer — passes through):")
    print(f"   {rag_correct(q)}")
    print()

    print("━" * 60)
    print("one decorator. no config. no keys. nothing crosses unseen.")
    print("━" * 60)


if __name__ == "__main__":
    main()
