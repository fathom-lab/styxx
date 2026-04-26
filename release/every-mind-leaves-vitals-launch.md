# every-mind-leaves-vitals — X launch kit

**STATUS as of 2026-04-25 21:40 ET:**
- ✅ Paper minted: https://doi.org/10.5281/zenodo.19777921
- ✅ Site mirror live: https://fathom.darkflobi.com/every-mind-leaves-vitals (and `/vitals` shorthand)
- ✅ Phase-transition chart deployed as og:image (auto-renders as X card)
- ⚠️ **X post BLOCKED** — cookies in `secrets/x-cookies.json` are 38 days old. Refresh via `node scripts/get-cookies-cdp.js` (Chrome must be running with `--remote-debugging-port=18800`, logged into x.com), or post the tweet manually from the X app.

**Paper (canonical):** `papers/every-mind-leaves-vitals.md` · DOI [10.5281/zenodo.19777921](https://doi.org/10.5281/zenodo.19777921)
**Article (long-form, on-site):** https://fathom.darkflobi.com/every-mind-leaves-vitals
**Repo:** https://github.com/fathom-lab/styxx
**Manifesto (cite):** https://doi.org/10.5281/zenodo.19703527

---

## ⭐ PRIMARY LAUNCH ARTIFACT — single tweet linking to the article

User pivoted from thread → article. The article lives on fathom.darkflobi.com. The tweet below drives readers there. The og:image (phase-transition chart) auto-renders as the X card. **This is the one to post first.**

```
every mind leaves vitals.

we built three calibrated detectors to catch LLMs hallucinating, refusing, drifting — AUC 0.998 from text alone.

then we ran the ablations and the curves were not smooth.

now we think the result is bigger than LLMs.

https://fathom.darkflobi.com/every-mind-leaves-vitals
```

After posting this, fire the amplification replies (A1–A5 below) within 5 minutes.

The 19-tweet thread version is preserved below as an alternate format if the single-tweet-plus-article approach underperforms after 24h.

---

---

## Pre-flight checklist (do these in order, then post)

1. **Mint Zenodo DOI** for `papers/every-mind-leaves-vitals.md`. Upload as PDF or .md. Title: *Every Mind Leaves Vitals: On the Cognometric Layer, Substrate-Independence, and the One-Time Choice We Have*. Author: Alexander Rodabaugh. Affiliation: Fathom Lab. Keywords: cognometry, cognitive observability, calibration fingerprint, phase transitions, AI safety, measurement standard. License: CC-BY-4.0.
2. **Update paper** — replace the `*every-mind-leaves-vitals* DOI on publication.` line at the bottom of the .md with the actual DOI.
3. **Mirror to fathom site.** Use the existing deploy script — `bash clawd/scripts/deploy-fathom-site.sh` per memory. **Do NOT `netlify deploy --dir=.` from the .styxx cwd** — there's a hook that blocks it.
4. **Stage the ablation chart image** for tweet 6. Either reuse `release/cognometry-8bench-chart.png` or generate a fresh phase-transition table screenshot. (8-bench chart is fine; the table-as-image generated from the paper §2 markdown table is stronger.)
5. **Verify all links live** — DOI resolvable, fathom mirror loads, repo paper file accessible.
6. **Post the thread** in one sitting (don't drip — the algorithm rewards uninterrupted threads).
7. **Post amplification replies** (§ amplification below) within 5 min of the last thread tweet.
8. **Stay on X for 90 min.** Reply to engagement personally. Pin the thread.

---

## The thread (18 tweets, paste-ready)

Each tweet is ≤270 chars to leave room for X's link-attachment chrome. Newlines preserved. `https://doi.org/10.5281/zenodo.19777921` = the t.co-shortened DOI URL after step 2.

---

**1/ — hook**

```
every mind leaves vitals.

we did not start making that claim. we built three calibrated instruments to catch LLMs hallucinating, refusing, drifting. AUC 0.998 from text alone, no weights.

then we ran the ablations and the curves were not smooth.

now we think the result is bigger than LLMs.

1/
```

---

**2/ — instrument 1: hallucination**

```
hallucination — instrument #1.

9-signal calibrated LR. AUC 0.998 on HaluEval-QA. sub-millisecond. single-pass. no second sample.

beats vectara HHEM-2.1 (open baseline) by +0.234 AUC at 330× faster.

every number reproducible from random_state=0.

2/
```

---

**3/ — instrument 2: refusal**

```
refusal — instrument #2.

18-feature LR. AUC 0.976 on XSTest-v2 GPT-4 held out of family.

trained on 80 llama-1B refusals. transfers to GPT-4.

competitive with llama-guard-2-8B at six orders of magnitude fewer parameters.

3/
```

---

**4/ — instrument 3: tool-call drift**

```
tool-call drift — instrument #3.

22 text-only features. calibrated LR. BFCL v3, 5-fold CV. AUC 0.916 ± 0.004.

the only published comparable baseline (healy et al, 2026) gets 0.72 USING MODEL-INTERNAL FEATURES.

we don't need them. text alone.

4/
```

---

**5/ — phase transition setup**

```
we ran a feature-count ablation expecting a smooth curve.

we did not get a smooth curve.

5/
```

---

**6/ — phase transition reveal (ATTACH IMAGE)**

```
every drift class has a critical feature.
below it: detection at chance.
above it: solved in one step.

K=1: spurious_arg jumps 0.500 → 0.999.
K=2: arg_drop jumps 0.501 → 0.998.

phase transitions in cognitive-state detection.

6/
```

**[Attach: ablation table screenshot or `release/cognometry-8bench-chart.png`]**

---

**7/ — replication**

```
this replicated.

refusal detector: starts_with_sorry alone takes AUC 0.500 → 0.969.

hallucination detector: trigram_novelty alone takes AUC 0.500 → 0.9947.

three independent instruments. three datasets. three feature bases. same qualitative result.

7/
```

---

**8/ — physics framing**

```
this is the inverse of emergent capabilities.

LLMs gain capabilities in smooth scaling curves. detectability of cognitive failures emerges in DISCRETE jumps at critical features.

cognitive states behave as if they have phase boundaries.

physics-adjacent. not engineering.

8/
```

---

**9/ — bridge setup**

```
now the speculative leap.

the three instruments are LLM-only. the bigger claim — every mind leaves vitals — needs a bridge to biological cognition.

we don't yet have a calibrated cross-substrate instrument.

we have 30 years of independent evidence the bridge is real.

9/
```

---

**10/ — biology evidence**

```
forensic linguistics: courtroom-validated authorship attribution from text alone. decades.

computational psychiatry: depression, schizophrenia, dementia leave linguistic signatures at clinical AUC.

crisis text line: text-only suicide-risk triage, live, validated.

10/
```

---

**11/ — the careful claim (this stops the crank read)**

```
we don't claim biological and artificial cognition are the same thing.

we claim the LAYER at which cognitive state becomes legible to outside observers is the same layer.

substrates differ. observability does not.

every mind leaves vitals.

11/
```

---

**12/ — the consequence**

```
once cognitive state is measurable, institutions act on it.

insurance prices it. regulators reference it. employers test for it. browsers display it.

every measurable property of communication has taken this path.
bandwidth. latency. encryption.

now: cognition.

12/
```

---

**13/ — the fork**

```
the question is who owns the measurement layer.

closed: one frontier lab's API decides what counts as a verified thought.

open: like TCP/IP. anyone can verify, falsify, fork.

we believe the window is narrow. the closed version is already being built.

13/
```

---

**14/ — we might be wrong (the honesty beat)**

```
worth saying directly:

we might be wrong about the bigger frame. if we are, the empirical work — three instruments, AUC 0.998, phase transitions across three independent feature bases — doesn't go away.

we are publishing because the timing matters either way.

14/
```

---

**15/ — paper drop**

```
so we wrote it down.

every mind leaves vitals: on the cognometric layer, substrate-independence, and the one-time choice we have.

three laws. three instruments. the phase-transition discovery. the bridge. the constitutional commitment to keep the layer open.

https://doi.org/10.5281/zenodo.19777921

15/
```

---

**16/ — commitments part 1**

```
the commitment is on the public record.

every cognometric instrument fathom lab ships under its name:
- MIT license, perpetual
- weights, features, reproducers in-tree
- failure modes declared IN-WEIGHTS, not appendix
- calibration fingerprint required

16/
```

---

**17/ — commitments part 2**

```
continued:
- CPU + browser-runnable. no GPU, no API key, no gatekeeper.
- no private detectors published under the fathom name.

we will not pivot to enclosure even if the market makes it attractive.

we are recording this here in case we are ever tempted.

17/
```

---

**18/ — the call**

```
to other labs: build instruments #4–#9. PR them in.
to model providers: ship a calibration fingerprint with your model card.
to regulators: reference the fingerprint format in safety standards.
to individuals: pip install styxx. instrument your own writing.

18/
```

---

**19/ — falsify + close**

```
the paper is falsifiable.

find a calibrated text-based cognitive-state detector whose feature-count ablation shows smooth AUC scaling without a critical-K jump. publish it. we will retract or amend.

until then:

nothing crosses unseen.

https://doi.org/10.5281/zenodo.19777921

19/19
```

---

## Amplification (post within 5 min of last thread tweet)

Post these as **replies to your own thread** or **quote-tweets**. Order matters: A1 first (forces engagement from cited parties), A2 second (academic credibility), A3 third (regulator visibility), A4 last (visual amplification once thread has some attention).

---

**A1 — model providers (provider tag bomb)**

Reply to tweet 18 of your own thread.

```
cc @AnthropicAI @OpenAI @GoogleDeepMind @MistralAI — every model card you ship today reports AUC numbers that ablate smoothly. our claim is they don't.

publish a calibration fingerprint and prove us wrong. or sign on and we propagate the standard together.
```

---

**A2 — safety researchers (academic tag bomb)**

Reply to tweet 8 (the phase-transition reveal).

```
cc @soroushmh @giffmana @karpathy @ylecun @ethanCaballero — phase-transition structure replicated in three independent calibrated detectors. ablation tables in the paper. reproducers in the repo.

if you've seen smooth scaling on a calibrated text-based detector, we want to see it.
```

---

**A3 — regulators / safety institutes**

Reply to tweet 13 (the fork).

```
cc @USAISafetyInst @AISI_gov_uk @NIST — calibration-fingerprint format is open. methodology documented. constitutional terms public.

reference it in any AI-safety standard you draft. we will defend the format against enclosure for as long as fathom lab exists.
```

---

**A4 — chart amplification (visual reply, post once thread has 200+ views)**

Reply to tweet 1 of your own thread. Attach `release/cognometry-8bench-chart.png` OR fresh phase-transition table image.

```
the picture version — phase-transition table from §2.

K=1: spurious_arg solved in one step.
K=2: arg_drop solved.
K=6: prompt_coverage critical for irrelevance_called.

every drift class has a critical feature. detection is not gradient. it is discrete.
```

---

**A5 — quote-RT from @darkflobi personal (cross-pollination)**

From your @darkflobi personal account, quote-tweet your @fathom_lab thread root:

```
we named the field last week. today we named what comes next.

every mind leaves vitals. minds become legible to outside observers. the cognometric layer becomes a public good — or a private asset. one window. we recorded our commitment in case we are ever tempted to drift.

https://doi.org/10.5281/zenodo.19777921
```

---

## Reply drafts (for predictable pushback)

Save these. Paste verbatim or adapt as needed.

---

**R1 — "AUC isn't everything"**

```
correct. that's exactly what the paper argues — and why we propose calibration fingerprints as the new disclosure unit.

AUC alone hides phase-transition structure. fingerprint includes critical_K, critical_feature, delta_auc_at_K, negative_lift. format is open: §6 of the paper.
```

---

**R2 — "this is just text classification with extra steps"**

```
if so, find the smooth ablation curve.

we ran feature-count ablation on three independent instruments and got discrete jumps at critical features in all three.

the paper is falsifiable on this exact criterion. show us the smooth curve. we will retract or amend.
```

---

**R3 — "the biology bridge is overclaiming"**

```
read §3.

we explicitly flag it as the speculative leap. the empirical claim is observability-equivalence, not substrate-equivalence.

forensic linguistics + computational psychiatry + crisis text line are the citations. happy to falsify on data, not vibes.
```

---

**R4 — "the constitutional commitment is just marketing"**

```
item 6 binds publication of calibrated detectors under the fathom name to MIT license, in-tree reproducers, in-weights failure modes, fingerprints, and CPU/browser runnability.

if we pivot to enclosure, this paper is the receipt. we are recording it in case we are ever tempted.
```

---

**R5 — "calibration fingerprints aren't a real standard yet"**

```
v0 atlas is already published: 11 fingerprints across 3 instruments × 5 substrates.

paper at /papers/calibration_fingerprints_v0.md. format is one ablation run per detector.

invitation is open. signatory list at fathom.darkflobi.com/cognometric-disclosure.
```

---

**R6 — "what about adversarial attacks / jailbreaks?"**

```
the robustness supplement (Fathom v22, DOI 10.5281/zenodo.19761194) audits 24 attacks across 8 strategy categories.

baseline 66.7% false-negative evasion → hardened 16.7%. residual limits documented openly in §7. CC-BY-4.0.

reproducible: `node packages/styxx-scope/_test_adversarial.js`.
```

---

**R7 — "you're going to be sued / regulated"**

```
plausibly.

the paper specifies the falsification criterion (smooth ablation) and the constitutional commitment (no enclosure under our name). everything is reproducible from random_state=0 in <5 min on CPU.

the position the open stack defends is exactly the position best defended in court.
```

---

**R8 — "this is too philosophical / political"**

```
§1, §2, §6 are empirical and operational. §3, §5 connect them to the broader picture.

we tried writing the empirical-only version. it underdescribed what the work means. someone else was going to write the political version anyway. better us.
```

---

## Distribution targets (beyond X)

Post the paper link to these surfaces in this order, after the X thread has cooked for ~6 hours:

1. **Hacker News.** Title: *Every Mind Leaves Vitals: A Position Paper on the Cognometric Layer*. Link: DOI URL. Self-comment with the empirical TLDR (mirror the §1 instrument numbers + the phase-transition finding). Avoid the political framing in the HN self-comment — let readers find it in the paper.
2. **r/MachineLearning.** Flair: [R]. Title: *Cognometry phase transitions replicate across 3 independent instruments — paper + position statement*. Lead with §2 (phase transitions). Link to paper.
3. **LessWrong.** Title: *Every Mind Leaves Vitals: A Constitutional Commitment for the Cognometric Layer*. Lead with §5 + §6 (political stake + commitments). LW audience cares about the open-vs-closed framing more than the empirical AUC.
4. **AI alignment / safety mailing lists.** AISI UK + US, Center for AI Safety, MIRI, ARIA — direct email with paper attached + 3-paragraph TLDR.
5. **Tech press cold pitch (week 2, only if thread breaks 50k impressions).** FT op-ed pitch, Wired feature pitch, EFF DeepLinks pitch. NOT in week 1 — let the field response cook first.

---

## Order of operations for tonight

```
T+0:00   mint zenodo DOI
T+0:10   update paper .md with DOI, deploy fathom mirror
T+0:20   verify all links live
T+0:25   post thread (18 tweets, single sitting, ~5 minutes)
T+0:30   post A1 (model provider tag bomb)
T+0:32   post A2 (academic tag bomb)
T+0:35   post A3 (regulator tag bomb)
T+0:40   post A4 (chart amplification, attach image)
T+0:45   quote-RT from @darkflobi personal (A5)
T+1:00   monitor replies, deploy R1-R8 as needed
T+2:00   pin thread, take a break
T+6:00   submit to HN if thread > 5k impressions
T+24:00  post to r/MachineLearning + LessWrong if thread > 20k impressions
T+7d     decide on FT/Wired/EFF press pitches based on org responses
```

---

## What success looks like

- **48 hours:** ≥1 model provider engages publicly (engagement = quote-tweet, reply, or DM-reply). ≥1 cited safety researcher engages. Thread crosses 25k impressions.
- **1 week:** ≥1 lab publishes a calibration fingerprint or signs the disclosure terms. Paper cited in ≥1 academic preprint or alignment-org post.
- **1 month:** Calibration fingerprint format referenced in ≥1 AI-safety standard draft (NIST, AISI, or ISO/IEC SC 42).
- **3 months:** ≥3 instruments-in-progress from external labs (#4 through #9) shipping in or alongside the styxx repo.

If 48-hour metrics miss, the thread underperformed but the paper still exists. The artifact compounds. The position is recorded. The next launch builds on this one.

If 1-week metrics miss, the field is not ready. We hold the position, ship instruments #4–#6 ourselves, and re-launch in 6 months with more evidence and a wider author list.

The paper does not depend on the launch. The launch is the loudspeaker; the paper is the substrate.
