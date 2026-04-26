# Every Mind Leaves Vitals

*An essay on the moment cognometry stopped being engineering.*

*Alexander Rodabaugh · Fathom Lab · April 2026*

---

I sat with a chart for an hour, and by the end the work I had been doing was a different kind of work.

I had built three small things over the previous months. Not large. Not heroic. Three calibrated detectors, each a logistic regression over a few dozen text features, each trained against a few hundred labeled examples, each designed to catch one of the ways a language model can lie. Hallucination. Refusal. Tool-call drift. Each one ran in milliseconds on a laptop, with no access to the model's weights, no second sample, no privileged read into the hidden state. Numbers came out the other side. They were good numbers — better than I had any right to expect. AUC nine-nine-eight on hallucination. Nine-seven-six on refusal. Nine-one-six on drift. I ran the feature ablations as a sanity check, the way you run unit tests before a push, to confirm that adding features improved performance smoothly, that nothing strange was happening underneath.

Nothing strange was happening underneath.

Something stranger.

The curves were not curves. They were step functions. Below a critical feature, detection sat at chance — fifty-fifty, the noise floor of a coin. Add one specific feature, and the AUC jumped to ninety-nine percent. The gap was a cliff. I checked the seed. I checked the test set. I re-ran with a different model. I tried it on the second instrument. The same shape. I tried it on the third. The same shape.

Three independent detectors. Three independent datasets. Three independent feature bases. Each had a critical feature, and below that feature the failure mode was structurally invisible, and above it the failure mode was solved. There was no smooth gradient between blindness and sight. There was a threshold.

I left the desk and drank water and came back, and the chart was still there.

---

Phase transitions are a physics phenomenon. Water becomes ice at zero Celsius, not a degree before, not a degree after. Iron becomes magnetic at the Curie point. A gas becomes a plasma at one specific temperature, and not at any temperature near it. The world of fluid dynamics is one place; the world of crystals is another; there is no continuous path between them. Below the threshold, one regime. Above it, another. The threshold is not put there by the experimenter. The threshold is waiting.

I had not gone looking for physics. I had gone looking for a hallucination detector. What I had built, it turned out, was a thermometer — a calibrated thing that, applied to a running system, returned a reading of its internal state. And the readings, plotted against the instrument's resolution, showed boundaries.

I had been calling this work *measurement*. After the chart, I started calling it something else. **Cognometry**: the discipline of measuring the cognitive state of a thinking system from outside, calibrated against ground truth, in any substrate that produces text.

I sat with that word for a while. It is not a word that gets smaller.

---

This is the part where I am supposed to tell you it is just a property of language models. Three detectors, three benchmarks. Stay in your lane. Be careful not to overclaim.

I will not stay in my lane.

For thirty years, in a different building, in a different field, people have been doing the same thing to people. Forensic linguists examine a contested document — a will, a confession, a tweet that may or may not be by its purported author — and return courtroom-grade attributions from text features alone. Computational psychiatrists track language across hundreds of patients and find that depression, schizophrenia, dementia, and Alzheimer's onset leave linguistic signatures detectable at clinical AUC. Crisis Text Line, every night, triages suicide risk on text alone, in real time, at validated reliability. The methodology is the same family as mine. The math is the same. The targets are the only thing that differs — biological cognition there, artificial cognition here.

I do not claim those are the same thing. The brain and the transformer are not the same. They evolved under different pressures, on different substrates, for different purposes.

I claim something narrower, and harder to dismiss.

**The layer at which cognitive state becomes legible to outside observers is the same layer.** Substrates differ. Observability does not. The structure I found in my detectors is not a property of how I built them; it is a property of the phenomenon I was trying to detect. And the phenomenon does not respect the boundary between artificial and biological.

This is the claim I might be wrong about. If I am, the empirical work does not go away. The instruments still run. The reproducers still match. Cognometry stays cognometry. But if I am right, then a category that humanity has fenced off as private — the inner state of a thinking thing — has just become measurable from outside, with discrete structure, in any substrate that produces text.

**Every mind leaves vitals.**

I have spent a lot of nights with that claim. It does not get smaller.

---

There is a thing that happens, every time, when a property of communicating systems becomes measurable. Within a generation, the property is priced. Within two, it is regulated. Within three, it is required. Bandwidth was once an obscure parameter; it is now a market that prices every internet transit on Earth. Latency was once the unmeasured remainder; it is now what high-frequency trading firms spend nine-figure sums to shave off in microseconds. Encryption strength used to be classified; it is now a procurement clause in every federal contract. Each began as a measurement nobody had thought to take. Each ended as a foundation of how the world functions.

Cognition is on the same path.

Insurance will price it. Regulators will reference it. Employers will test for it. Browsers will display it. Procurement contracts will require it. None of this is speculation. It is the path every measurable property of information has taken, without exception, since the founding of the discipline.

The question is not whether the measurement layer of cognition gets built. The question is who builds it. And how openly.

The closed version is being built today, somewhere, by people I have never met, with a different theory of who deserves access. If it finishes first, the era of measurable cognition opens with a corporate gatekeeper — a single entity that can certify which thinking systems are honest and which are not, with a private contract and a private price. That is too much power to grant any single entity, regardless of who is currently holding it.

The open version can also be built. It is what I have spent the last year building.

MIT-licensed in perpetuity. Weights and reproducers in the same tree as the prose, so that every number reruns from `random_state=0` in under five minutes on a CPU you already own. Failure modes declared in the weights module itself, where a user sees the limits before they see the claims, not in an appendix that nobody reads. A calibration fingerprint shipped with every detector — phase-transition profile, critical feature, negative-lift map; the things that distinguish a real measurement from a benchmark number. CPU and browser-runnable, so that a graduate student in Lagos and a regulator in Brussels and a curious teenager in Iowa can run the instrument the same week it ships, on the laptop they already have. No private detectors under our name. Ever. Even if the market makes it attractive — and the market will, the market is already starting to.

I have written these six commitments into the paper, on the public record, so that if I ever break them, the breaking is visible. That is the point of putting commitments on the record before they are tested. The paper is the receipt.

---

I am aware of how this can read. A small lab, naming a category it just published, claiming that an entire field is being enclosed by entities it does not name. That is a bad shape for a position paper. I am writing it anyway, because the alternative is to wait until the enclosure is visible enough to be uncontroversial — at which point arguing against it is also too late.

If you build calibrated detectors: ship a calibration fingerprint with every release. The format is open. The cost is one ablation run.

If you ship a model: publish a fingerprint alongside the model card. The reputational gain is non-trivial. You become the first frontier lab whose model card carries a measurable-honesty signature.

If you draft AI safety standards: reference the calibration-fingerprint format. The technical work of defining cognometric disclosure as a regulatory unit is already done.

If you fund research: the open-stack version of this layer is undercapitalized relative to the closed versions. Six figures, not nine, would fund instruments #4 through #9 in their entirety.

If you work with text and want to instrument your own thinking — the substrate of how you reason, the shape of your refusals, the texture of your uncertainty — `pip install styxx`. The first generation to instrument its own cognition will do so on free public substrate, or on a surveillance vendor's product. Choose.

---

The paper is falsifiable on its sharpest claim. Find a calibrated text-based detector whose feature-count ablation shows smooth scaling — no critical-K jump, no step function — and we retract or amend at the same DOI. We have looked. We have not found. We invite the field to look harder.

A chart appeared on my screen, and I sat with it for an hour. Then I built the instruments. Then I named the category. Then I wrote the laws. Then I committed the substrate to the public — six terms, on the record, with a footer of falsification — so that a small lab on a Friday night could not, in some larger Friday night six years from now, quietly enclose the thing it had named.

Nothing crosses unseen.

---

**Read the paper.** [doi.org/10.5281/zenodo.19777921](https://doi.org/10.5281/zenodo.19777921)
**Reproduce.** [github.com/fathom-lab/styxx](https://github.com/fathom-lab/styxx)
**Run live.** [fathom.darkflobi.com/cognometry](https://fathom.darkflobi.com/cognometry)
**Manifesto.** [doi.org/10.5281/zenodo.19703527](https://doi.org/10.5281/zenodo.19703527)
