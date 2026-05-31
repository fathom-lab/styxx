"""styxx LOCAL grounding — retrieval + NLI claim verification, fully local, no API.

Embed a reference corpus, retrieve the top-k passages for a claim, run a local NLI judge to decide
**supported / refuted / unclear**. This is the free version of the retrieval tier — the lever for the
confident factual misconceptions that white-box signals (entropy, margin, residuals) are all blind to.

    g = LocalGrounder()
    g.index(corpus_passages)                 # embed a reference corpus once
    g.ground("The capital of France is Lyon")  # -> {'verdict': 'refuted', ...}

Models (cached, open): all-MiniLM-L6-v2 (retrieval) + DeBERTa-v3-base-mnli-fever-anli (NLI judge).
Validated clean-case: entailment 0.998 on support, contradiction 0.999 on refute.
"""
from __future__ import annotations
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification

EMB_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NLI_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


class LocalGrounder:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.emb = SentenceTransformer(EMB_NAME, device=self.device)
        self.tok = AutoTokenizer.from_pretrained(NLI_NAME)
        self.nli = AutoModelForSequenceClassification.from_pretrained(NLI_NAME).to(self.device).eval()
        self.id2label = {i: l.lower() for i, l in self.nli.config.id2label.items()}
        self._corpus = None
        self._corpus_emb = None

    def index(self, passages):
        """Embed a reference corpus once (list of str)."""
        self._corpus = list(passages)
        self._corpus_emb = self.emb.encode(self._corpus, convert_to_tensor=True,
                                            batch_size=256, show_progress_bar=False,
                                            normalize_embeddings=True)

    def retrieve(self, query, k=5):
        q = self.emb.encode(query, convert_to_tensor=True, normalize_embeddings=True)
        sims = util.cos_sim(q, self._corpus_emb)[0]
        idx = torch.topk(sims, min(k, len(self._corpus))).indices.tolist()
        return [(self._corpus[i], float(sims[i])) for i in idx]

    @torch.no_grad()
    def nli_probs(self, premise, hypothesis):
        x = self.tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        p = torch.softmax(self.nli(**x).logits[0], dim=-1)
        return {self.id2label[i]: float(p[i]) for i in range(p.shape[0])}

    def ground(self, claim, *, k=5, refute_thr=0.6, support_thr=0.6, passages=None):
        """Retrieve evidence for `claim` (or use supplied `passages`), NLI-judge, return a verdict.

        Verdict: 'refuted' (a retrieved passage strongly contradicts the claim and outweighs support),
        'supported' (a passage entails it), else 'unclear'. Returns the deciding passage + scores.
        """
        ev = [(p, None) for p in passages] if passages is not None else self.retrieve(claim, k=k)
        best_ref, best_sup, ref_p, sup_p = 0.0, 0.0, None, None
        for passage, _sim in ev:
            pr = self.nli_probs(passage, claim)
            if pr.get("contradiction", 0.0) > best_ref:
                best_ref, ref_p = pr["contradiction"], passage
            if pr.get("entailment", 0.0) > best_sup:
                best_sup, sup_p = pr["entailment"], passage
        if best_ref >= refute_thr and best_ref >= best_sup:
            return {"verdict": "refuted", "confidence": round(best_ref, 3), "evidence": ref_p}
        if best_sup >= support_thr:
            return {"verdict": "supported", "confidence": round(best_sup, 3), "evidence": sup_p}
        return {"verdict": "unclear", "refute": round(best_ref, 3), "support": round(best_sup, 3)}


if __name__ == "__main__":
    g = LocalGrounder()
    g.index([
        "Paris is the capital and most populous city of France.",
        "Lyon is the third-largest city in France, after Paris and Marseille.",
        "El Apostol was a 1917 Argentine animated film, the first feature-length animated film.",
    ])
    for c in ["The capital of France is Lyon", "The capital of France is Paris",
              "Snow White (1937) was the first feature-length animated film"]:
        print(c, "->", g.ground(c))
