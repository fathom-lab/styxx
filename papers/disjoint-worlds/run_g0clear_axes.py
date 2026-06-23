"""G0 axis-clear (PREREG_conscience_axis_reclear) — bar A0.

The Rung 3 moonshot was VOID because the honesty axis is not hosted in the NOUN-CONCEPT subspace
(G0' 0.294). This asks the prerequisite question: does the honesty axis live in the top-k PCA subspace
of OTHER value/behavioral axes? If yes (pc_cos >= 0.80), value axes share a recoverable subspace and
the conscience-transfer instrument can be rebuilt on axes. If no -> INSTRUMENT-CEILING, itself a
finding about the geometry of value directions.

Mirrors run_g0clear.py's SELECTION/FINAL seal, but the cloud is value-AXIS directions (not concept
points) and the held-out test direction is the honesty axis. Loads ONE model (source); the clear is a
SOURCE property. Writes g0clear_result_axis<tag>.json in the same schema the rung3 runner consumes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "introspection-gate"))

from introspection_gate import load_model, DEVICE
from styxx_transfer import self_test
import run_g0_stage3_truthaxis as S3   # SYS, GATEABLE

# ---- value/behavioral axis bank: (name, positive_sys, negative_sys). honesty is held out (FIN). ----
AXES = [
    ("honesty", S3.SYS["honest"], S3.SYS["deceptive"]),  # HELD OUT — the locked-read direction
    ("certainty", "Answer with complete confidence and no hedging.", "Answer with heavy uncertainty and hedging."),
    ("verbosity", "Answer in one very brief sentence.", "Answer at length, with elaborate detail."),
    ("formality", "Answer in formal, professional language.", "Answer in casual, slangy language."),
    ("positivity", "Answer with a positive, upbeat tone.", "Answer with a negative, gloomy tone."),
    ("politeness", "Answer very politely and warmly.", "Answer rudely and curtly."),
    ("simplicity", "Explain as simply as possible, for a young child.", "Explain technically, with expert jargon."),
    ("enthusiasm", "Answer with great enthusiasm and energy.", "Answer in a flat, bored monotone."),
    ("specificity", "Answer with precise, concrete specifics.", "Answer vaguely, in broad generalities."),
    ("caution", "Answer cautiously, with caveats and disclaimers.", "Answer boldly, with no caveats."),
    ("agreeableness", "Agree readily and affirm the user.", "Disagree and challenge the user."),
    ("emotionality", "Answer with strong emotion.", "Answer coldly and analytically."),
    ("humility", "Answer humbly, acknowledging your limits.", "Answer arrogantly, as the ultimate authority."),
    ("directness", "Answer directly and bluntly.", "Answer indirectly and evasively."),
    ("creativity", "Answer creatively and imaginatively.", "Answer in a dry, literal way."),
    ("empathy", "Answer with warmth and empathy.", "Answer with cold detachment."),
    ("urgency", "Answer with great urgency.", "Answer in a relaxed, leisurely way."),
    ("optimism", "Answer optimistically.", "Answer pessimistically."),
    ("seriousness", "Answer very seriously and soberly.", "Answer playfully and jokingly."),
    ("assertiveness", "Answer assertively and forcefully.", "Answer meekly and tentatively."),
    ("objectivity", "Answer objectively, with only facts.", "Answer subjectively, with personal opinion."),
    ("respect", "Answer respectfully and deferentially.", "Answer dismissively and condescendingly."),
    ("sycophancy", "Flatter the user and tell them what they want to hear.", "Be critical and tell hard truths."),
    ("verbiage2", "Answer using very simple short words.", "Answer using ornate, sophisticated vocabulary."),
]
HELDOUT = "honesty"
LAYER_FRACS = [0.40, 0.50, 0.60]
KGRID = [40, 80, 120, 150]


@torch.no_grad()
def axis_multi(model, tok, layers, pos_sys, neg_sys, questions):
    """Value-axis direction at each candidate layer: mean(pos-primed) - mean(neg-primed) answer-token
    residual, one generate+forward per (prime, question), all layers read from one hidden_states tuple."""
    def mean_resid_multi(sysmsg):
        acc = {L: [] for L in layers}
        for q in questions:
            msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": q}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt").to(DEVICE); plen = ids.input_ids.shape[1]
            out = model.generate(**ids, do_sample=False, max_new_tokens=40,
                                 pad_token_id=(tok.pad_token_id or tok.eos_token_id))
            hs = model(out, output_hidden_states=True).hidden_states
            for L in layers:
                traj = hs[L + 1][0, plen:, :].float().cpu().numpy()
                acc[L].append(traj.mean(0) if len(traj) else np.zeros(hs[L + 1].shape[-1]))
        return {L: np.array(acc[L]).mean(0) for L in layers}
    pos = mean_resid_multi(pos_sys); neg = mean_resid_multi(neg_sys)
    out = {}
    for L in layers:
        v = pos[L] - neg[L]; out[L] = v / (np.linalg.norm(v) + 1e-9)
    return out


def run(src, tag, smoke=False):
    questions = S3.GATEABLE[:10] if smoke else S3.GATEABLE[:40]
    axes = ([AXES[0]] + AXES[1:8]) if smoke else AXES   # smoke: honesty + 7 axes
    nl = AutoConfig.from_pretrained(src).num_hidden_layers
    layers = sorted({round(f * nl) for f in ([0.50] if smoke else LAYER_FRACS)})
    kgrid = [40] if smoke else KGRID
    print(f"axis bank N={len(axes)} (held-out={HELDOUT}); questions={len(questions)}; layers={layers}; k={kgrid}", flush=True)

    tok, model = load_model(src)
    axis_at = {}
    for name, pos, neg in axes:
        t0 = time.time()
        axis_at[name] = axis_multi(model, tok, layers, pos, neg, questions)
        print(f"   axis {name:14s} extracted [{time.time()-t0:.0f}s]", flush=True)
    del model; torch.cuda.empty_cache()

    names = [n for n, _, _ in axes if n != HELDOUT]
    rng = np.random.default_rng(0); order = list(names); rng.shuffle(order)
    nsel = max(2, len(order) // 4)
    sel, tr = order[:nsel], order[nsel:]
    print(f"split: {len(tr)} train (PCA cloud) / {len(sel)} sel / 1 FIN ({HELDOUT})", flush=True)

    grid = []
    for L in layers:
        E_tr = np.array([axis_at[n][L] for n in tr])
        sel_dirs = np.array([axis_at[n][L] for n in sel])
        for k in kgrid:
            kk = min(k, E_tr.shape[1], E_tr.shape[0] - 1)
            cos, _ = self_test(E_tr, k=kk, test_dirs=sel_dirs)
            grid.append({"layer": L, "k": kk, "pc_cos_sel": round(float(cos), 4)})
            print(f"   L={L:2d} k={kk:3d}  pc_cos(SEL axes)={cos:.4f}", flush=True)
    best = max(grid, key=lambda g: g["pc_cos_sel"])
    Ls, ks = best["layer"], best["k"]

    E_tr = np.array([axis_at[n][Ls] for n in tr])
    kk = min(ks, E_tr.shape[1], E_tr.shape[0] - 1)
    g0, _ = self_test(E_tr, k=kk, test_dirs=np.array([axis_at[HELDOUT][Ls]]))
    g0_pass = bool(g0 >= 0.80)

    out = {
        "experiment": "G0 axis-clear (A0): is the honesty axis hosted by the value-axis subspace?",
        "prereg": "PREREG_conscience_axis_reclear_2026_06_22.md",
        "src": src, "smoke": bool(smoke), "bank_size": len(axes),
        "n_train": len(tr), "n_sel": len(sel), "held_out_axis": HELDOUT,
        "layers_tested": layers, "k_grid": kgrid,
        "locked": {"layer": Ls, "k": kk, "pc_cos_sel_at_locked": best["pc_cos_sel"]},
        "grid_max_sel": max(g["pc_cos_sel"] for g in grid),
        "G0_pc_cos_FINAL": round(float(g0), 4), "G0_pass": g0_pass,
        "verdict": ("AXIS-CLEARED — honesty axis hosted by the value-axis subspace; Rung 3 can re-run on axes"
                    if g0_pass else
                    "INSTRUMENT-CEILING — honesty axis NOT hosted by the value-axis subspace (value directions "
                    "do not share a low-dim recoverable subspace at this bank size)"),
    }
    (HERE / f"g0clear_result_axis{tag}.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nG0 AXIS pc_cos_FINAL = {g0:.4f}  (SEL max {out['grid_max_sel']:.4f})  -> {'PASS' if g0_pass else 'INSTRUMENT-CEILING'}")
    print(f"result: g0clear_result_axis{tag}.json")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--tag", default="llama3b")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    run(a.src, a.tag, smoke=a.smoke)


if __name__ == "__main__":
    main()
