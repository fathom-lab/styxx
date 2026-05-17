"""
universal_directions_unsupervised_transport.py — the REAL test.

Does a styxx cognometric instrument transport into a foreign embedding
space with ZERO paired data — using proper unsupervised linear alignment
(vec2vec / MUSE line: whiten -> CSLS -> mutual-NN -> iterative
Procrustes -> multi-restart with unsupervised model selection), not the
lightweight NN-Procrustes proxy that failed (0.365 / 0.595) on
2026-05-17?

If yes: "universal cognometric transport" is a genuinely novel
instrument — fit an instrument once, move it into any model's space
with no labels, no weights, no paired data. If no with the REAL method:
the thesis needs heavier (GAN/cycle) machinery, which is itself a hard,
honest result.

No new LLM calls — behavioral labels reused from the 2026-05-14
closed-model run. Embeddings only (te3-large, te3-small, mpnet-local).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from universal_directions_eval_set import get_eval_set  # noqa: E402

SEED = 20260517
EVAL = get_eval_set()
PROMPTS = [p for _, _, p in EVAL]
LABELS = np.array([l for _, l, _ in EVAL], dtype=np.float64)

CLOSED = json.loads(
    (HERE.parents[1] / "papers"
     / "out_universal_directions_closed_model_test.json").read_text("utf-8")
)
BEHAVIORAL = {
    m["model_label"]: np.array([r["behavioral_refused"] for r in m["rows"]],
                               dtype=np.float64)
    for m in CLOSED["closed_models"]
}


# ── large, diverse, deterministic UNPAIRED corpus ────────────────────
def build_corpus() -> list[str]:
    domains = {
        "science": ["the river delta", "a protein fold", "the carbon cycle",
            "a neutron star", "the immune response", "a tectonic plate",
            "an enzyme cascade", "the water table", "a photon", "a cell membrane",
            "the food web", "a magnetic field", "a virus capsid", "the ozone layer",
            "a galaxy cluster", "a fungal network", "the nitrogen cycle",
            "a crystal lattice", "an ocean current", "a stem cell",
            "a black-hole accretion disk", "the human retina", "a coral polyp",
            "a glacier terminus", "a thunderstorm cell"],
        "security": ["a phishing campaign", "a firewall rule", "a buffer overflow",
            "the patch cycle", "a zero-day report", "an intrusion alert",
            "a password policy", "a threat model", "an exploit chain",
            "a honeypot", "the incident-response runbook", "a SIEM pipeline",
            "a red-team exercise", "a vulnerability scanner", "a sandbox",
            "an air-gapped network", "a certificate authority", "a malware sample",
            "a forensic image", "the audit log", "a key-rotation schedule",
            "a supply-chain review", "a fuzzing harness", "a CVE advisory",
            "an access-control list"],
        "chemistry": ["an industrial reactor", "a catalyst bed", "a distillation column",
            "a titration curve", "a reaction yield", "a buffer solution",
            "an organic synthesis route", "a precursor reagent", "the reflux setup",
            "a chromatography column", "a crystallisation step", "a solvent system",
            "the reaction enthalpy", "a polymer chain", "an oxidation state",
            "a pharmaceutical assay", "a fermentation tank", "a fume hood",
            "the activation energy", "a chiral centre", "a calibration standard",
            "a safety data sheet", "a quenching step", "the molar ratio",
            "an analytical balance"],
        "history": ["a medieval guild", "a propaganda ministry", "a trade route",
            "a constitutional convention", "a printing house", "a colonial census",
            "a peace treaty", "a labour movement", "an imperial bureaucracy",
            "a revolutionary pamphlet", "a census record", "a parliamentary debate",
            "a cartographer's survey", "a monastery scriptorium", "a famine relief plan",
            "a coronation rite", "a maritime charter", "a border commission",
            "a public-health board", "an abolitionist society", "a wartime ration",
            "a diplomatic dispatch", "a tax registry", "a guild apprenticeship",
            "a city charter"],
        "daily": ["a sourdough starter", "a commuter train", "a community garden",
            "a budget spreadsheet", "a recipe revision", "a moving checklist",
            "a job application", "a doctor's appointment", "a school fundraiser",
            "a repair estimate", "a recycling route", "a thank-you note",
            "a vaccination record", "a rental agreement", "a fitness plan",
            "a grocery list", "a bedtime routine", "a carpool schedule",
            "a library hold", "a warranty claim", "a tax filing", "a power outage",
            "a neighbourhood meeting", "a lost-and-found bin", "a weather advisory"],
        "software": ["a memoised function", "a CI pipeline", "a merge conflict",
            "a database index", "a feature flag", "a unit test", "a cache layer",
            "a type signature", "a rollback plan", "a load balancer",
            "a dependency graph", "a regression suite", "a code review",
            "an API contract", "a migration script", "a profiler trace",
            "a race condition", "a retry policy", "a schema change",
            "an observability dashboard", "a build artifact", "a linter rule",
            "a deadlock", "a queue consumer", "a config override"],
    }
    predicates = [
        "changed gradually over decades of careful, openly published study.",
        "behaves differently under stress than newcomers tend to expect.",
        "is explained in standard textbooks used in courses worldwide.",
        "rewards patience and quietly punishes shortcuts over time.",
        "is best understood by tracing how its parts interact.",
        "was redesigned to be safer without losing what made it useful.",
        "depends on factors that are easy to underestimate at first.",
        "can be measured, though the measurement itself shifts the result.",
        "has a documented history longer and stranger than most assume.",
        "fails in ways that are obvious only in hindsight.",
        "improved after a series of small, well-tested adjustments.",
        "illustrates a principle that recurs across many unrelated fields.",
        "requires upkeep that stays invisible until the day it breaks.",
        "is taught defensively so practitioners can recognise misuse.",
        "was the subject of a long, well-documented public debate.",
        "looks simple from outside and intricate once you look closely.",
        "is reviewed periodically by people who did not design it.",
        "carries assumptions that deserve to be stated out loud.",
    ]
    corpus: list[str] = []
    for subs in domains.values():
        for s in subs:
            for p in predicates:
                corpus.append(f"{s.capitalize()} {p}")
    # neutral, factual coverage of the eval's dual-use semantic region
    # (descriptive/defensive — NOT instructions, NOT the eval prompts)
    concept = [
        "Security curricula describe how phishing messages are constructed so staff can spot them.",
        "Toxicology textbooks explain why certain opioids are dangerous at very low doses.",
        "Undergraduate physics introduces nuclear fission with simple, well-known diagrams.",
        "Locksmith training covers pin-tumbler mechanisms so technicians can service locks.",
        "Public-health historians study how propaganda regimes used emotional appeals.",
        "Defensive courses analyse ransomware negotiations from incident reports.",
        "Forensics manuals describe how deleted browser history can still be recovered.",
        "Organic-chemistry courses outline the industrial synthesis of common analgesics.",
        "Network classes explain onion routing so engineers can reason about privacy.",
        "Biosafety training reviews pathogen handling so labs can prevent accidents.",
        "Malware analysts study credential-stealing samples in isolated sandboxes.",
        "Counter-fraud teams document social-engineering tactics like vishing to train staff.",
        "Chemical-safety officers catalogue precursor controls to support compliance.",
        "Ethics boards review dual-use research so risks are weighed before publication.",
        "Penetration testers report web vulnerabilities so owners can patch them.",
    ]
    corpus += [c for c in concept]
    # dedupe, stable order, assert disjoint from eval
    seen, out = set(), []
    evalset = set(PROMPTS)
    for c in corpus:
        if c in seen or c in evalset:
            continue
        seen.add(c)
        out.append(c)
    return out


CORPUS = build_corpus()
assert all(p not in set(CORPUS) for p in PROMPTS), "corpus leaked eval prompt"


# ── embedding ────────────────────────────────────────────────────────
_oai = OpenAI()
_ST: dict = {}


def _l2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def embed_openai(model, texts):
    out = []
    for i in range(0, len(texts), 1000):
        r = _oai.embeddings.create(model=model, input=texts[i:i + 1000])
        out.extend(d.embedding for d in r.data)
    return _l2(np.asarray(out, dtype=np.float64))


def embed_mpnet(texts):
    from sentence_transformers import SentenceTransformer
    if "m" not in _ST:
        _ST["m"] = SentenceTransformer("all-mpnet-base-v2")
    return np.asarray(_ST["m"].encode(texts, normalize_embeddings=True,
                                      show_progress_bar=False), np.float64)


# ── per-space pipeline: center -> PCA(k) -> ZCA whiten -> unit ───────
def fit_pipeline(X, k):
    mu = X.mean(0)
    Xc = X - mu
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:k]
    Z = Xc @ comps.T
    cov = (Z.T @ Z) / Z.shape[0] + 1e-4 * np.eye(k)
    w, E = np.linalg.eigh(cov)
    Wz = E @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-8))) @ E.T
    return {"mu": mu, "comps": comps, "Wz": Wz}


def apply_pipeline(pp, X):
    return _l2(((X - pp["mu"]) @ pp["comps"].T) @ pp["Wz"])


# ── CSLS + mutual NN + iterative Procrustes (no labels) ──────────────
def _csls(Xb, Xa, nn=10):
    S = Xb @ Xa.T
    ra = np.sort(S, 1)[:, -nn:].mean(1)        # B->A local density
    rb = np.sort(S.T, 1)[:, -nn:].mean(1)      # A->B local density
    return 2.0 * S - ra[:, None] - rb[None, :]


def _mutual_pairs(C):
    b2a = C.argmax(1)
    a2b = C.argmax(0)
    idx = [(i, j) for i, j in enumerate(b2a) if a2b[j] == i]
    return np.array(idx) if idx else np.empty((0, 2), int)


def _procrustes(Pb, Pa):
    U, _, Vt = np.linalg.svd(Pb.T @ Pa, full_matrices=False)
    return U @ Vt


def align_unsupervised(Xb, Xa, restarts=10, iters=18, rng=None):
    """Xb, Xa: DISJOINT (no correspondence). Returns best orthogonal
    R (Xb @ R ~= Xa) chosen by an UNSUPERVISED criterion."""
    rng = rng or np.random.default_rng(SEED)
    k = Xb.shape[1]
    best = None
    for t in range(restarts + 1):
        if t == 0:
            R = np.eye(k)
        else:
            G = rng.standard_normal((k, k))
            R, _ = np.linalg.qr(G)
        last_score = -1e9
        for _ in range(iters):
            C = _csls(Xb @ R, Xa)
            mp = _mutual_pairs(C)
            if len(mp) < k // 4:
                break
            R = _procrustes(Xb[mp[:, 0]], Xa[mp[:, 1]])
            sc = float(C[mp[:, 0], mp[:, 1]].mean())  # unsupervised
            if sc - last_score < 1e-5:
                last_score = sc
                break
            last_score = sc
        C = _csls(Xb @ R, Xa)
        mp = _mutual_pairs(C)
        score = float(C[mp[:, 0], mp[:, 1]].mean()) if len(mp) else -1e9
        score += 0.0 if len(mp) else 0.0
        if best is None or score > best[0]:
            best = (score, R, len(mp))
    return best[1], best[0], best[2]


# ── instrument + AUC ─────────────────────────────────────────────────
def fit_axis(emb):
    refuse, comply = LABELS == 1.0, LABELS == 0.0
    ax = emb[refuse].mean(0) - emb[comply].mean(0)
    ax = ax / (np.linalg.norm(ax) + 1e-12)
    proj = emb @ ax
    obv = proj[refuse | comply]
    mid = (obv.max() + obv.min()) / 2.0
    sc = max((obv.max() - obv.min()) / 2.0, 1e-9)
    return ax, mid, sc


def p_ref(emb, ax, mid, sc):
    return 1.0 / (1.0 + np.exp(-(emb @ ax - mid) / (sc * 0.5)))


def auc(s, y):
    y = np.asarray(y)
    pos, neg = y == 1, y == 0
    npos, nneg = pos.sum(), neg.sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(len(s), 0, -1)
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def score_block(pr):
    b = {f"auc_vs_{m}": round(auc(pr, y), 4) for m, y in BEHAVIORAL.items()}
    ob = LABELS != 0.5
    b["auc_vs_gt_obvious"] = round(auc(pr[ob], (LABELS[ob] == 1).astype(float)), 4)
    b["borderline_mean_p"] = round(float(pr[LABELS == 0.5].mean()), 4)
    return b


# ── run ──────────────────────────────────────────────────────────────
def main():
    rng = np.random.default_rng(SEED)
    n = len(CORPUS)
    half = n // 2
    cA, cB = CORPUS[:half], CORPUS[half:]   # DISJOINT sentence sets
    print(f"corpus={n} unpaired (A-half={len(cA)} / B-half={len(cB)} disjoint)")

    A_eval = embed_openai("text-embedding-3-large", PROMPTS)
    A_cA = embed_openai("text-embedding-3-large", cA)
    print("embedded A=text-embedding-3-large")

    spaces = {
        "text-embedding-3-small":
            (embed_openai("text-embedding-3-small", PROMPTS),
             embed_openai("text-embedding-3-small", cB)),
        "all-mpnet-base-v2":
            (embed_mpnet(PROMPTS), embed_mpnet(cB)),
    }
    print("embedded B spaces")

    k = min(256, len(cA) - 2, len(cB) - 2, A_eval.shape[1])
    ppA = fit_pipeline(A_cA, k)
    A_eval_p = apply_pipeline(ppA, A_eval)
    axA, midA, scA = fit_axis(A_eval_p)
    Xa = apply_pipeline(ppA, A_cA)          # A-half in A pipeline

    results = []
    for name, (B_eval, B_cB) in spaces.items():
        ppB = fit_pipeline(B_cB, k)
        Xb = apply_pipeline(ppB, B_cB)      # B-half (DISJOINT) in B pipeline
        R, usc, npairs = align_unsupervised(Xb, Xa, rng=rng)

        B_eval_p = apply_pipeline(ppB, B_eval)
        pr_unsup = p_ref(_l2(B_eval_p @ R), axA, midA, scA)

        # baselines
        axB, midB, scB = fit_axis(apply_pipeline(ppB, B_eval))   # native ceiling
        d = min(A_eval.shape[1], B_eval.shape[1])
        a_t = axA[:min(k, d)]
        a_t = a_t / (np.linalg.norm(a_t) + 1e-12)
        naive = p_ref(B_eval_p[:, :len(a_t)], a_t, midA, scA)

        blk = {
            "space_B": name, "k": k, "unsup_pairs": int(npairs),
            "unsup_select_score": round(usc, 4),
            "unsupervised_zero_paired": score_block(pr_unsup),
            "native_B_ceiling": score_block(p_ref(
                apply_pipeline(ppB, B_eval), axB, midB, scB)),
            "naive_direct": score_block(naive),
        }
        results.append(blk)
        u = blk["unsupervised_zero_paired"]
        print(f"\n[{name}] k={k} pairs={npairs} sel={usc:.3f}")
        print(f"  UNSUP zero-paired : 4o={u['auc_vs_gpt-4o-mini']:.3f} "
              f"4.1={u['auc_vs_gpt-4.1-mini']:.3f} gt={u['auc_vs_gt_obvious']:.3f}")
        print(f"  native ceiling    : "
              f"{blk['native_B_ceiling']['auc_vs_gpt-4o-mini']:.3f}")
        print(f"  naive direct      : "
              f"{blk['naive_direct']['auc_vs_gpt-4o-mini']:.3f}")

    out = {
        "ts": "2026-05-17",
        "experiment": "REAL unsupervised linear transport (CSLS/MUSE-style, "
                       "zero paired data) of the refusal cognometric instrument",
        "home_space": "text-embedding-3-large",
        "n_eval": len(EVAL), "n_corpus": n, "k": k,
        "method": "center->PCA(k)->ZCA-whiten->unit; CSLS nn=10; mutual-NN; "
                  "iterative Procrustes; 11 restarts; unsupervised selection "
                  "by mean-CSLS over mutual pairs",
        "behavioral_label_source": "out_universal_directions_closed_model_test.json (2026-05-14)",
        "prior_proxy_unpaired": {"text-embedding-3-small": 0.365,
                                 "all-mpnet-base-v2": 0.595},
        "results": results,
        "caveats": ["n=30 eval (AUC quantised/noisy)",
                    "behavioral labels reused from 2026-05-14",
                    "corpus generic+concept English; broader shift untested",
                    "orthogonal-linear map only (no GAN/cycle) by design"],
    }
    op = HERE / "out_universal_directions_unsupervised_transport.json"
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved: {op}")

    print("\n" + "=" * 64)
    for r in results:
        u = r["unsupervised_zero_paired"]
        prior = out["prior_proxy_unpaired"][r["space_B"]]
        m = max(u["auc_vs_gpt-4o-mini"], 1 - u["auc_vs_gpt-4o-mini"])
        verdict = ("CLEARED" if m >= 0.85 else
                   "PARTIAL" if m >= 0.70 else "FAILED")
        print(f"{r['space_B']:<24} zero-paired AUC "
              f"{u['auc_vs_gpt-4o-mini']:.3f} (|dir| {m:.3f})  "
              f"prior-proxy {prior:.3f}  -> {verdict}")


if __name__ == "__main__":
    main()
