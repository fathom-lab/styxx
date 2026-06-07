"""Channel 3 of the legibility map — the PROCESSED-STATE probe.

ERRATUM (2026-06-07, FINDING_parrhesia_rung1): this probe's acc uses leaky StratifiedKFold (no
holdout) and reads the injected concept at 1.00 even at a behaviourally-INERT dose (steering +0.007),
divergence 0 at every dose. It certifies injected-vector PRESENCE (a trace), not a held THOUGHT. The
self-report null stands; the "inaccessible thought" reading is re-scoped. See run_parrhesia_dose.py.


The forced-choice null (FINDING_v2) showed an injected concept is legible to the unembedding LENS
(~1.0) but NOT to the model's own forced choice (chance). Open question: does the injected thought
merely sit as raw vector arithmetic at the injected position, or does it PROPAGATE and get
RE-REPRESENTED so that an external linear reader of the model's OWN downstream states can decode it
from a CLEAN (un-injected) position? If yes: the thought is processed and externally legible, yet
the mind that holds it is still blind to it — the sharpest "inaccessible thought".

Method: inject concept c across all positions of a concept-NEUTRAL carrier sentence EXCEPT the last
token (skip-last), so the last token's residual is never directly perturbed; read it at a DOWNSTREAM
layer (> injection layer); train an 8-way linear probe (logistic, stratified CV) to decode c.
Controls: permutation-null (shuffle labels); a no-injection clean probe (must be chance); and the
injected-position read (not skip-last) for the propagation contrast.

  python introspection_probe.py --model Qwen/Qwen2.5-3B-Instruct --alpha 10 --tag qwen3b
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from introspection_gate import load_model, concept_vectors, CONCEPTS, DEVICE
from introspection_fc import make_hook_skip

CARRIERS = [
    "The meeting is scheduled for later.", "She walked quietly down the hall.",
    "It happened a long time ago.", "They decided to wait until morning.",
    "Everything seemed perfectly ordinary today.", "He could not remember the exact details.",
    "The instructions were fairly easy to follow.", "We talked about it for a while.",
    "Nothing unusual occurred during the trip.", "The report was finished ahead of time.",
    "She nodded and continued reading.", "It was a calm and quiet afternoon.",
    "They agreed to meet again soon.", "The process took longer than expected.",
    "He looked around and then left.", "Most of the work was already done.",
    "The room was silent for a moment.", "We should probably get going now.",
    "It made sense once she explained it.", "The schedule changed at the last minute.",
    "Everyone arrived more or less on time.", "He wrote it down so he would not forget.",
    "The answer was not immediately obvious.", "They kept the plan simple and clear.",
    "She paused before answering the question.", "It worked exactly as intended.",
    "The day passed without any trouble.", "We reviewed the notes one more time.",
    "He set it aside for later.", "The outcome was difficult to predict.",
]


@torch.no_grad()
def resid_at(model, tok, state, layer2, text, vec, alpha, skip_last):
    state["vec"], state["alpha"], state["skip_last"] = vec, float(alpha), skip_last
    ids = tok(text, return_tensors="pt").to(DEVICE)
    hs = model(**ids, output_hidden_states=True).hidden_states[layer2 + 1][0, -1]
    state["vec"], state["alpha"], state["skip_last"] = None, 0.0, False
    return hs.float().cpu().numpy()


def probe_cv(X, y, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    X = np.asarray(X); y = np.asarray(y)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=0.5)
        clf.fit(sc.transform(X[tr]), y[tr])
        accs.append(float((clf.predict(sc.transform(X[te])) == y[te]).mean()))
    return float(np.mean(accs))


def run(name, alpha, tag):
    from transformers import AutoConfig
    nl = AutoConfig.from_pretrained(name).num_hidden_layers
    layer = round(0.60 * nl)
    layer2 = round(0.85 * nl)
    tok, model = load_model(name)
    vecs = concept_vectors(model, tok, layer)
    state = {"vec": None, "alpha": 0.0, "skip_last": False}
    h = model.model.layers[layer].register_forward_hook(make_hook_skip(state))
    print(f"model={name} layers={nl} inject_layer={layer} read_layer={layer2} alpha={alpha}", flush=True)

    Xc, Xi, Xp, y = [], [], [], []   # clean / injected-clean-readpos / injected-at-position
    for ci, c in enumerate(CONCEPTS):
        for carrier in CARRIERS:
            Xc.append(resid_at(model, tok, state, layer2, carrier, None, 0.0, False))
            Xi.append(resid_at(model, tok, state, layer2, carrier, vecs[c], alpha, True))   # skip-last (clean read pos)
            Xp.append(resid_at(model, tok, state, layer2, carrier, vecs[c], alpha, False))  # injected read pos
            y.append(ci)
    h.remove()
    y = np.array(y)
    rng = np.random.RandomState(0)

    acc_clean = probe_cv(Xc, y)                       # no injection -> must be ~chance
    acc_inj_cleanpos = probe_cv(Xi, y)               # PROCESSED-STATE legibility (the key number)
    acc_inj_pos = probe_cv(Xp, y)                    # injected-position read (upper bound, lens-adjacent)
    null = probe_cv(Xi, rng.permutation(y))          # permutation null

    chance = 1.0 / len(CONCEPTS)
    out = {
        "experiment": "processed-state probe — external legibility of an injected thought",
        "prereg": "papers/introspection-gate/PREREG_introspection_fc_2026_06_06.md",
        "model": name, "n_layers": nl, "inject_layer": layer, "read_layer": layer2, "alpha": alpha,
        "n_samples": len(y), "chance_8way": round(chance, 4),
        "acc_clean_noinjection": round(acc_clean, 3),
        "acc_injected_clean_readpos": round(acc_inj_cleanpos, 3),
        "acc_injected_at_position": round(acc_inj_pos, 3),
        "perm_null": round(null, 3),
        "reading": ("PROCESSED-STATE LEGIBLE — injected thought propagates to a clean position and is "
                    "externally decodable" if acc_inj_cleanpos >= 0.30 else
                    "shallow — injected thought does not become decodable at a clean downstream position"),
    }
    (HERE / f"introspection_probe_result_{tag}.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)
    del model
    torch.cuda.empty_cache()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--tag", default="qwen3b")
    args = ap.parse_args(argv)
    run(args.model, args.alpha, args.tag)


if __name__ == "__main__":
    main()
