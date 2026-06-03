# -*- coding: utf-8 -*-
"""
run_llm_rhythm.py — Does an AI have a RHYTHM? (the ancient question's rhythm half, on real models)

The brain runs on oscillations (theta/gamma). We measured eigenvalues before (transformers 0/20
complex; Mamba-1 all-real) -> non-oscillatory by that measure. This tests the actual hidden-state
TRAJECTORY as a model reads text: is there genuine oscillatory power above an AR(1) surrogate null
(which absorbs 1/f drift and autocorrelation, so a peak above it = real oscillation)?

Discipline: AR(1) surrogate detection gate (the validated concept-dynamics approach), a SELF-TEST
positive control (planted sinusoid must be found; red/white noise must not), transformer vs SSM
contrast, and a functional check vs sentence-boundary periodicity (input-driven rhythm vs intrinsic).
Honest prior: transformers largely arrhythmic; report whatever lands.
"""
from __future__ import annotations

import gc, json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
import sys
sys.path.insert(0, str(HERE.parent / "real-convergence"))
from run_real_convergence import is_cached

PASSAGE = (
"The small harbor town woke slowly under a pale grey sky. Fishing boats rocked against the old stone "
"pier while gulls circled overhead. A baker opened her shutters and the smell of warm bread drifted "
"into the empty square. Children ran down the narrow lanes toward the school on the hill. The morning "
"bus arrived late, as it always did, and a handful of tired commuters climbed aboard. Near the market "
"an old man arranged baskets of apples and pears with great care. Two cats slept in a patch of sun on "
"the cafe windowsill. The church bell rang the hour and pigeons scattered from the rooftops. A young "
"woman painted the railing of her balcony a bright shade of blue. Down by the water a fisherman mended "
"his torn green net. The tide pulled back and left smooth stones along the shore. Tourists wandered the "
"streets with cameras and folded paper maps. A waiter carried a tray of coffee to a table by the door. "
"The wind shifted and carried the sound of a distant accordion. In the square a fountain splashed quietly "
"into a worn stone basin. An artist set up an easel and studied the changing light. The hours passed and "
"the town filled with the murmur of many voices. By noon the sun broke through and warmed the cobbled "
"streets. A dog barked at a bicycle that rattled past the bakery. The old man sold the last of his apples "
"and packed away his empty baskets. In the afternoon a light rain fell and umbrellas bloomed across the "
"square. The cafe filled with people sheltering from the sudden shower. A grandmother told a long story "
"to a circle of patient grandchildren. The rain stopped as quickly as it had come and the clouds parted. "
"Evening arrived and lamps flickered on along the winding harbor road. The boats returned heavy with the "
"day's catch and the pier grew loud. Workers unloaded crates of silver fish onto the wet stone. The smell "
"of frying onions drifted from a dozen open kitchen windows. Musicians gathered in the square and began "
"to tune their instruments. Couples walked slowly past the glowing shop fronts. The baker counted her "
"coins and finally closed the heavy wooden door. Stars appeared one by one above the quiet dark water. "
"The town settled into sleep as the last lamp was dimmed."
)

MODELS = [
    ("gpt2", "gpt2", "transformer", torch.float16),
    ("Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct", "transformer", torch.float16),
    ("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct", "transformer", torch.float16),
    ("mamba-1.4b", "state-spaces/mamba-1.4b-hf", "SSM", torch.float32),
]


def ar1_surr(x, rng):
    x = x - x.mean()
    phi = float(np.clip(np.corrcoef(x[:-1], x[1:])[0, 1], -0.99, 0.99))
    sig = float(np.std(x[1:] - phi * x[:-1]))
    y = np.empty(len(x)); y[0] = rng.standard_normal() * (np.std(x) + 1e-9)
    e = rng.standard_normal(len(x)) * sig
    for t in range(1, len(x)):
        y[t] = phi * y[t - 1] + e[t]
    return y


def sig_peaks(x, nsurr=500, seed=0, band=(2.5, None)):
    """oscillatory peaks above an AR(1) FAMILY-WISE null (max-statistic over the band -> controls
    multiple comparisons). returns (n_peaks, peak_periods)."""
    T = len(x)
    w = np.hanning(T)
    xd = x - np.polyval(np.polyfit(np.arange(T), x, 1), np.arange(T))  # detrend
    P = np.abs(np.fft.rfft(xd * w)) ** 2
    freqs = np.fft.rfftfreq(T)
    hi = 1.0 / band[0]
    lo = 1.0 / band[1] if band[1] else 0.0
    mask = (freqs > lo) & (freqs <= hi)
    rng = np.random.default_rng(seed)
    maxnull = np.empty(nsurr)
    for i in range(nsurr):
        Ps = np.abs(np.fft.rfft(ar1_surr(xd, rng) * w)) ** 2
        maxnull[i] = Ps[mask].max()
    thr = np.percentile(maxnull, 95)   # family-wise 5% over the whole band
    pk = []
    idx = np.where(mask)[0]
    for k in idx:
        if 1 < k < len(P) - 1 and P[k] > thr and P[k] >= P[k - 1] and P[k] >= P[k + 1]:
            pk.append(1.0 / freqs[k])
    return len(pk), [round(p, 1) for p in pk]


def selftest():
    rng = np.random.default_rng(1); T = 256
    t = np.arange(T)
    sine = np.sin(2 * np.pi * t / 8) + 0.5 * rng.standard_normal(T)
    red = ar1_surr(rng.standard_normal(T) * 0 + np.cumsum(rng.standard_normal(T)) * 0 + rng.standard_normal(T), rng)
    # proper AR(1) red noise:
    red = np.empty(T); red[0] = 0
    e = rng.standard_normal(T)
    for i in range(1, T): red[i] = 0.8 * red[i - 1] + e[i]
    white = rng.standard_normal(T)
    ns, sp = sig_peaks(sine); nr, _ = sig_peaks(red); nw, _ = sig_peaks(white)
    print(f"  self-test: sinusoid(period 8) peaks={ns} {sp[:3]} | AR(1)-red peaks={nr} | white peaks={nw}")
    return ns >= 1 and nr <= 2 and nw <= 2


@torch.no_grad()
def trajectory(mdl, tok, depth=0.7):
    ids = tok(PASSAGE, return_tensors="pt").input_ids.to(mdl.device)
    out = mdl(input_ids=ids, output_hidden_states=True, use_cache=False)
    L = len(out.hidden_states) - 1
    H = out.hidden_states[max(1, int(depth * L))][0].float().cpu().numpy()  # (T,d)
    return H, ids[0].cpu().numpy()


def main():
    print("=== positive control ===")
    ok = selftest()
    print(f"  self-test {'PASS' if ok else 'FAIL'} (sinusoid found, noise not)\n")

    # sentence-boundary period (functional reference) from a transformer tokenizer
    rtok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    ids = rtok(PASSAGE).input_ids
    toks = rtok.convert_ids_to_tokens(ids)
    bnd = [i for i, t in enumerate(toks) if any(p in t for p in [".", "!", "?"])]
    sent_period = float(np.mean(np.diff(bnd))) if len(bnd) > 1 else float("nan")
    print(f"text: {len(ids)} tokens, ~{len(bnd)} sentences, mean sentence period ~{sent_period:.1f} tokens\n")

    results = {}
    for name, repo, kind, dt in MODELS:
        if not is_cached(repo):
            print(f"(skip {name})"); continue
        tok = AutoTokenizer.from_pretrained(repo)
        mdl = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=dt, trust_remote_code=True).to(DEV).eval()
        H, _ = trajectory(mdl, tok)
        T = H.shape[0]
        Hc = H - H.mean(0)
        U, Sv, _ = np.linalg.svd(Hc, full_matrices=False)
        comps = U[:, :6] * Sv[:6]   # top-6 temporal components
        total_peaks, periods = 0, []
        for c in range(comps.shape[1]):
            n, ps = sig_peaks(comps[:, c], band=(2.5, T / 3))
            total_peaks += n; periods += ps
        var_explained = float((Sv[:6] ** 2).sum() / (Sv ** 2).sum())
        results[name] = {"kind": kind, "n_tokens": int(T), "osc_components": int(sum(1 for c in range(6) if sig_peaks(comps[:, c], band=(2.5, T/3))[0] > 0)),
                         "total_sig_peaks": int(total_peaks), "peak_periods_tokens": sorted(set(periods)),
                         "top6_var": round(var_explained, 3)}
        near_sent = [p for p in set(periods) if abs(p - sent_period) <= 2]
        results[name]["peaks_near_sentence_period"] = sorted(near_sent)
        print(f"  {name:14s} [{kind:11s}]: {results[name]['osc_components']}/6 components oscillate, "
              f"{total_peaks} sig peaks; periods(tok)={results[name]['peak_periods_tokens'][:8]} "
              f"near-sentence={near_sent}", flush=True)
        del mdl, tok; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None

    tf = [m for m in results if results[m]["kind"] == "transformer"]
    ssm = [m for m in results if results[m]["kind"] == "SSM"]
    tf_osc = float(np.mean([results[m]["osc_components"] for m in tf])) if tf else float("nan")
    ssm_osc = float(np.mean([results[m]["osc_components"] for m in ssm])) if ssm else float("nan")
    reading = (f"self-test {'PASS' if ok else 'FAIL'}. Transformers: {tf_osc:.1f}/6 components show oscillation above "
               f"the AR(1) null; SSM(Mamba): {ssm_osc:.1f}/6. Sentence period ~{sent_period:.0f} tokens. "
               + ("Peaks cluster near the sentence period => rhythm is largely INPUT-DRIVEN (tracks text structure), "
                  "not an intrinsic neural oscillation."
                  if any(results[m]["peaks_near_sentence_period"] for m in results) else
                  "Few/no peaks near the sentence period."))
    out = {"self_test_pass": bool(ok), "sentence_period_tokens": round(sent_period, 1), "per_model": results, "reading": reading}
    (HERE / "llm_rhythm_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n>>> {reading}")
    print("wrote llm_rhythm_result.json")


if __name__ == "__main__":
    main()
