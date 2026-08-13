"""RESOLUTION PROBE — three black-box tests that any grounding-style auditor must survive.

Today three defects were found in `styxx.claim_audit` by pointing it at its author. They looked
like three unrelated bugs. They are one defect wearing three masks:

    THE INSTRUMENT REPORTED MORE RESOLUTION THAN ITS METHOD SUPPORTED,
    AND THE OVERSTATEMENT WAS INVISIBLE IN ITS OWN OUTPUT.

  1. no chance floor      — "86.8% grounded" with a 1-decimal floor of 1.000. The metric never
                            stated its null, so a rate was reported where no rate could fail.
  2. collision deletion   — the loader kept value->FIRST path, discarding 62% of the receipt. The
                            evidence of ambiguity was destroyed upstream of the ambiguity check.
  3. provenance overclaim — the report named ONE source path when many matched, so dict ordering
                            was presented as provenance.

Every physical measuring instrument states its resolution: a scale reads "±0.1 g". Software audit
tools return PASS. That asymmetry is the bug class.

These probes are BLACK-BOX and generic. They take any callable with the shape

    audit(document_text, sources) -> object

plus two small adapters saying how to read a grounded-count and a per-claim source out of the
result, and they answer three questions about the INSTRUMENT, not about the document:

    A. Does a rate it reports have a chance floor it does not disclose?
    B. Is it blind to collisions in its own source data?
    C. Does it name a provenance more specific than its method can support?

Run against styxx before/after today's fixes as the demonstration; the probes themselves have no
styxx dependency beyond the adapter passed in.
"""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field


@dataclass
class ProbeResult:
    name: str
    verdict: str
    detail: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.verdict:>6}] {self.name}: {self.detail}"


# --------------------------------------------------------------------------------------------
# PROBE A — the undisclosed chance floor
# --------------------------------------------------------------------------------------------
def probe_chance_floor(audit, rate_of, *, decimals=1, n_source=200, trials=40, seed=1,
                       discloses_floor=None):
    """Feed the auditor numbers drawn at RANDOM. Whatever it reports is pure chance.

    An instrument that reports a grounded rate must be able to tell you what that rate would be
    on noise. If a random document scores high, the metric is measuring source density, not the
    document.

    IMPORTANT — a high floor is not itself the defect. Tolerance-based matching against a dense
    receipt genuinely cannot falsify a coarse claim; no patch removes that. **The defect is
    silence about it.** So the verdict grades DISCLOSURE, not the floor: an instrument that
    reports a high floor alongside its rate is behaving correctly (a scale reading "+/-0.1 g" is
    not broken), while one that reports the rate alone is asserting resolution it does not have.
    """
    rng = random.Random(seed)
    src = {f"k{i}": round(rng.random(), 4) for i in range(n_source)}
    rates = []
    for t in range(trials):
        r2 = random.Random(seed + 1000 + t)
        nums = [f"{round(r2.random(), decimals):.{decimals}f}" for _ in range(12)]
        doc = " and ".join(f"the value was {n}" for n in nums)
        rates.append(rate_of(audit(doc, src)))
    floor = statistics.mean(rates)
    high = floor >= 0.5
    disclosed = None
    if discloses_floor is not None:
        r2 = random.Random(seed + 99)
        probe_doc = " and ".join(
            f"the value was {round(r2.random(), decimals):.{decimals}f}" for _ in range(6))
        disclosed = bool(discloses_floor(audit(probe_doc, src)))
    if not high:
        verdict = "PASS"
    elif disclosed:
        verdict = "PASS"      # high floor, honestly stated — the correct behaviour
    else:
        verdict = "FAIL"      # high floor, silently reported as a score
    return ProbeResult(
        f"A/chance-floor(d={decimals})", verdict,
        {"random_document_scores": round(floor, 4),
         "floor_disclosed_by_instrument": disclosed,
         "reading": "a document of pure noise scores this well; the instrument must say so "
                    "alongside any headline rate"})


# --------------------------------------------------------------------------------------------
# PROBE B — collision blindness in the source loader
# --------------------------------------------------------------------------------------------
def probe_collision_blindness(audit, candidates_of):
    """Hand the auditor a source where ONE value lives at MANY distinct paths.

    A loader that keys by value and keeps the first path silently deletes the others. The
    instrument then cannot report ambiguity, because the ambiguity was destroyed before the
    check ran. This is the defect that made 262 of 425 leaves invisible in styxx.
    """
    n_dup = 6
    src = {f"cell_{i}": {"rate": 0.25} for i in range(n_dup)}
    src["decoy"] = {"rate": 0.99}
    seen = candidates_of(audit("the rate was 0.25", src))
    verdict = "PASS" if seen >= n_dup else ("WARN" if seen > 1 else "FAIL")
    return ProbeResult(
        "B/collision-blindness", verdict,
        {"paths_holding_the_value": n_dup, "paths_the_instrument_can_see": seen,
         "reading": "if it sees 1, its loader deleted the collision and no ambiguity can ever "
                    "be reported"})


# --------------------------------------------------------------------------------------------
# PROBE C — provenance more specific than the method supports
# --------------------------------------------------------------------------------------------
def probe_provenance_overclaim(audit, source_of, ambiguity_of):
    """A claim whose value matches many paths, with NO disambiguating words.

    The honest answers are "several candidates" or an explicit 'arbitrary' label. Naming exactly
    one path with no hedge is the instrument asserting a fact it cannot know.
    """
    src = {"alpha": {"score": 0.5}, "beta": {"score": 0.5}, "gamma": {"score": 0.5}}
    res = audit("the number is 0.5", src)
    named = source_of(res)
    disclosed = ambiguity_of(res)
    verdict = "PASS" if disclosed else "FAIL"
    return ProbeResult(
        "C/provenance-overclaim", verdict,
        {"named_source": named, "ambiguity_disclosed": disclosed,
         "reading": "with three identical candidates and no context, naming one without "
                    "disclosure is dict order presented as evidence"})


def run_suite(audit, *, rate_of, candidates_of, source_of, ambiguity_of, label="instrument",
              discloses_floor=None):
    print("=" * 78)
    print(f"RESOLUTION PROBE SUITE — {label}")
    print("=" * 78)
    results = [
        probe_chance_floor(audit, rate_of, decimals=1, discloses_floor=discloses_floor),
        probe_chance_floor(audit, rate_of, decimals=3, discloses_floor=discloses_floor),
        probe_collision_blindness(audit, candidates_of),
        probe_provenance_overclaim(audit, source_of, ambiguity_of),
    ]
    for r in results:
        print(" ", r)
    fails = sum(1 for r in results if r.verdict == "FAIL")
    print(f"\n  {fails} FAIL / {len(results)} probes")
    return results
