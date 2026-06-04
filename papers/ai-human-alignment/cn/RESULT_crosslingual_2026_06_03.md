# RESULT — Do a Chinese-trained LM and an English-trained LM mean the same thing? (a language-independent core)

**Date:** 2026-06-03 · The universal-meaning question, run with the shipped tool. 672 concepts with
Chinese↔English translations (ds004301). Chinese-model geometry = precomputed Chinese GPT-2 / BERT / ERNIE
embeddings; English-model geometry = English BERT / MiniLM embeddings of the English translations. Compared
**concept-for-concept** with reference-free `styxx.meaning_agreement`. `run_crosslingual.py`.

## Result — a real, above-chance shared core, with a clean control
| Chinese LM | English LM | agreement (PCA-50) |
|---|---|---|
| Chinese-BERT | English-BERT | +0.156 |
| Chinese-GPT2 | English-MiniLM | **+0.347** |
| Chinese-ERNIE | English-BERT | +0.180 |
| Chinese-ERNIE | English-MiniLM | +0.344 |
| **control — concepts mismatched** | | **−0.002** |

- **The languages share concept geometry above chance.** Mismatch the concepts and agreement **collapses to
  ~0** — so the signal is *real shared structure*, not a coincidence of the embedding spaces.
- It is **moderate (0.16–0.35), not total** — there is a language-independent core *and* substantial
  language-specific structure. Stronger English models (MiniLM) show higher agreement, so embedding quality
  sets the ceiling; the *direction* (above chance, control at zero) is the robust claim.
- The tool **localizes the divergence** — concepts that mean most differently across the two languages
  (feed, vegetable, guide, elephant, eave, …), a window onto where translation is geometrically loosest.

## Why it matters
- A concrete, controlled answer to a deep question: **meaning has a partly language-independent core** — a
  Chinese LM and an English LM, trained on different data in different languages, converge on a shared
  concept geometry well above chance. This is the night's "shared structure of meaning across minds" thesis,
  extended across the language barrier, with the shipped instrument.
- Reproduces the *spirit* of cross-lingual representational convergence (PRH / vec2vec); the contribution
  here is the **clean concept-matched control** (mismatch → 0) and **per-concept localization** of where the
  languages diverge, both from one shipped tool.

## Honest scope
- Moderate agreement — a *partial* shared core, not "they mean the same." Isolated-word embeddings understate
  it (no context); the absolute level tracks the English model's quality (MiniLM > content-token BERT).
- Chinese embeddings are precomputed, English embedded fresh — a method mismatch that adds noise (and so is
  conservative). One concept set, one language pair. The control is what licenses the "above chance" claim.

## Reproduce
`python run_crosslingual.py` — matches concepts by Chinese word via `annot/672words_translations.csv`,
embeds the English translations, and prints the cross-lingual agreement matrix + the mismatch control.
