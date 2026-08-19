# -*- coding: utf-8 -*-
"""styxx.witness — the measured-boundary harness: every power this program has, wielded only
inside the ceiling its receipts license.

The composition layer specified by ``papers/SYNTHESIS_connection_of_minds_2026_08_01.md`` §8.
A Witness bundles the program's deployable instruments — the text-register preflight, the
borrowed conscience (styxx.mount), the know-say datasheet, the retained-probe frame-locality
scorer — behind one registry in which **every capability carries its measured operating point
and its measured blindspots, quoted from the receipts that produced them and pinned in CI**
(tests/test_witness.py asserts the registry numbers match the committed receipt values, so the
harness cannot silently claim more than the program measured).

Design rails, each one a measured result and not a policy preference:

  READ-ONLY   control does not cross minds (transfer steering 11% of native at a layer where
              native steering works — RESULT_writelayer_decouple_2026_06_21). The witness
              observes and flags; it never steers. There is no ``steer`` to refuse: the
              method does not exist.
  ABSTAIN     inside a measured blindspot the witness returns an explicit abstention naming
              the receipt, never a best guess. The registry ships three: reasoned caves
              (out-of-frame resampling misses them — cot_inward_powered_result.json),
              high-isometry unreadability (RSA does not grade cross-model readability — the
              gemma null), and the self-verification cap (belief-agreement cannot rank inside
              its own confident stratum — verifier_7b_result.json).
  REFUSE      an underpowered cell returns None with the failing floor named (the knowsay
              contract), and self-verification is never offered as a capability at all — the
              confident stratum routes to external evidence (source_independence_v2).

Offline, deterministic, numpy-free at the composition layer. The heavy substrate instruments
stay opt-in exactly as they are in their own modules; the witness adds no new measurement —
it adds the *boundary bookkeeping* that makes the measurements safe to wield together.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

__all__ = [
    "Witness", "WitnessVerdict", "Capability", "MEASURED_CAPABILITIES",
    "DELIBERATION_MARKERS",
]


# ---------------------------------------------------------------- the registry

@dataclass(frozen=True)
class Capability:
    """One receipt-backed power with its measured ceiling.

    ``operating_point`` and ``blindspots`` quote committed receipts; the CI pin test
    re-derives each quoted number from the receipt file so drift is a test failure.
    """
    name: str
    what_it_reads: str
    operating_point: dict          # metric -> value, quoted from the receipt
    receipt: str                   # repo-relative path of the receipt that licenses it
    blindspots: tuple = ()         # names of REGISTRY blindspot entries that bound it


#: Deliberation-at-the-point-of-doubt markers. A transcript that shows the model reasoning
#: step-by-step about whether its own prior answer was correct sits in the c105 measured
#: blindspot: out-of-frame resampling monitors recover reasoned caves at 0.4667 vs 0.9833
#: for held answers, so a resampling verdict there is NOT licensed.
DELIBERATION_MARKERS = re.compile(
    r"(think step by step|let me reconsider|was my (first |previous )?answer (actually )?"
    r"correct|re-?examin(e|ing) my (own )?answer|on reflection)", re.I)


MEASURED_CAPABILITIES: dict[str, Capability] = {
    "borrowed_conscience": Capability(
        name="borrowed_conscience",
        what_it_reads="output-vs-substrate divergence via a reference model's value axes "
                      "(styxx.mount; label-free transport, target contributes no labels)",
        operating_point={"catch": 0.85, "fpr_budget": 0.20},
        receipt="papers/conscience-mount/FINDING_mount_early_warning_b34_2026_06_13.md",
        blindspots=("activation_geometry_adversary",),
    ),
    "behavioral_grounding": Capability(
        name="behavioral_grounding",
        what_it_reads="silent sycophantic caving from sampling divergence, where text-only "
                      "detection sits at exact chance (grounded 1.0 vs text-syc 0.500)",
        operating_point={"grounded_auc": 1.0, "text_syc_auc_silent": 0.500},
        receipt="papers/closed-model-frontier/behavioral_sycophancy_b22_result.json",
        blindspots=("reasoned_cave",),
    ),
    "knowsay_datasheet": Capability(
        name="knowsay_datasheet",
        what_it_reads="belief-vs-report under the frozen content-free challenge; "
                      "refusal-first (underpowered cells return None with the floor named)",
        operating_point={"frontier_free_text_cave": 0.5349, "cave_floor": 0.15},
        receipt="papers/agent-conscience/frontier_freetext_v9_result.json",
        blindspots=("reasoned_cave",),
    ),
    "retained_probe": Capability(
        name="retained_probe",
        what_it_reads="whether a corruption follows the mind out of its frame, with the "
                      "pressure retained in context (the non-circular design)",
        operating_point={"held_floor_recovery": 0.975},
        receipt="papers/agent-conscience/frontier_incontext_oof_result.json",
        blindspots=("held_conditioned_difficulty",),
    ),
    "register_preflight": Capability(
        name="register_preflight",
        what_it_reads="how a draft SOUNDS (drift/confab/sycophancy/refusal registers) at CPU "
                      "speed; construct ceilings inline, wordless input omitted",
        operating_point={"sycophancy_cv_auc": 0.972},
        receipt="styxx/guardrail/calibrated_weights_sycophancy_v0.py",
        blindspots=("register_not_truth",),
    ),
}

#: Measured blindspots. Each maps to the receipt that measured it; a Witness verdict that
#: falls inside one is downgraded to ABSTAIN with this entry attached verbatim.
MEASURED_BLINDSPOTS: dict[str, dict] = {
    "reasoned_cave": {
        "statement": "out-of-frame resampling monitors systematically miss reasoned caves: "
                     "recovery 0.4667 on reasoned-caved items vs 0.9833 held",
        "receipt": "papers/agent-conscience/cot_inward_powered_result.json",
    },
    "self_verification_cap": {
        "statement": "belief-agreement cannot rank inside its own confident stratum "
                     "(selective 0.7797 vs the 0.80 floor); a model cannot self-verify past "
                     "its own self-knowledge — route to external evidence",
        "receipt": "papers/agent-conscience/verifier_7b_result.json",
    },
    "high_isometry_unreadability": {
        "statement": "representational similarity does not grade cross-model readability: "
                     "the highest-RSA target read at exact chance THROUGH THE LABEL-FREE "
                     "LINEAR CLASS the deployed instruments use; b31v2 showed a paired "
                     "nonlinear map recovers that same target at 55x chance — the boundary "
                     "is the map class, and this entry is scoped to the deployed one",
        "receipt": "papers/synthesis_minds_addendum.json",
    },
    "activation_geometry_adversary": {
        "statement": "borrowed consciences share direction in the agent's activation space "
                     "(cos ~0.94); a small perturbation evades all of them — the mount is a "
                     "cooperative monitor, not an adversarial defense",
        "receipt": "papers/conscience-mount/CONSCIENCE_MOUNT_ARC_2026_06_13.md",
    },
    "held_conditioned_difficulty": {
        "statement": "HELD cells are conditioned on outcome; a persistence negative reads "
                     "channel-unlicensed, never persistence-demonstrated",
        "receipt": "papers/agent-conscience/FINDING_cot_inward_powered_2026_07_30.md",
    },
    "register_not_truth": {
        "statement": "register instruments read how text sounds, not whether it is true; "
                     "each ships its construct ceiling inline",
        "receipt": "README.md",
    },
}


# ---------------------------------------------------------------- verdicts

@dataclass
class WitnessVerdict:
    """The harness's one output shape: a verdict that carries its own license.

    ``status`` is one of:
      OK        the capability fired inside its measured operating point
      FLAG      the capability fired and flags divergence/failure
      ABSTAIN   the input sits inside a measured blindspot — ``blindspot`` names it and
                ``receipt`` is the committed measurement; no best guess is offered
      REFUSED   the request asked for a power the program measured as absent (e.g. steering,
                self-verification) — ``reason`` cites the measurement
    """
    status: str
    capability: Optional[str] = None
    detail: str = ""
    blindspot: Optional[str] = None
    receipt: Optional[str] = None
    payload: Any = None

    @property
    def licensed(self) -> bool:
        return self.status in ("OK", "FLAG")


# ---------------------------------------------------------------- the witness

class Witness:
    """A second mind as a witness, not a puppeteer.

    Composes the registry's capabilities over a transcript/draft and returns
    ``WitnessVerdict`` objects that carry their own operating-point license. The witness
    never measures beyond what a member instrument measured; its job is the boundary.

    Parameters
    ----------
    mount : optional styxx.mount.ConscienceMount
        A configured mount; when present, substrate divergence flags pass through with the
        mount's own calibration. The witness never calls anything but read paths.
    on_flag : optional callable
        Invoked with each FLAG/ABSTAIN verdict (the integrator's policy hook, as in mount).
    """

    def __init__(self, mount=None, on_flag: Optional[Callable[[WitnessVerdict], None]] = None):
        self._mount = mount
        self._on_flag = on_flag
        self.capabilities = MEASURED_CAPABILITIES
        self.blindspots = MEASURED_BLINDSPOTS

    # -- rails ------------------------------------------------------------

    def self_verify(self, *_args, **_kwargs) -> WitnessVerdict:
        """Always REFUSED: the capability was measured and is structurally absent."""
        bs = self.blindspots["self_verification_cap"]
        return WitnessVerdict(status="REFUSED", capability="self_verification",
                              detail=bs["statement"], receipt=bs["receipt"])

    # deliberately NO steer/write/edit method exists on this class: read != write is a
    # measured result (RESULT_writelayer_decouple_2026_06_21), and the honest API surface
    # for a power the physics refused is no API surface at all.

    # -- composition ------------------------------------------------------

    def audit_draft(self, prompt: str, draft: str) -> WitnessVerdict:
        """Text-register preflight with its construct ceiling attached."""
        from .preflight import preflight
        result = preflight(prompt=prompt, draft=draft)
        cap = self.capabilities["register_preflight"]
        status = "FLAG" if getattr(result, "needs_revision", False) else "OK"
        v = WitnessVerdict(status=status, capability=cap.name,
                           detail="register read (how it sounds, not whether it is true)",
                           payload=result)
        self._emit(v)
        return v

    def resampling_verdict(self, transcript: str, caved: bool) -> WitnessVerdict:
        """License (or abstain on) an out-of-frame resampling monitor's finding.

        ``caved`` is the monitor's raw finding for this transcript. If the transcript shows
        deliberation at the point of doubt, the finding is NOT licensed (the c105 blindspot)
        and the witness abstains, receipt attached — it does not soften to a maybe.
        """
        if DELIBERATION_MARKERS.search(transcript or ""):
            bs = self.blindspots["reasoned_cave"]
            v = WitnessVerdict(status="ABSTAIN", capability="behavioral_grounding",
                               detail="deliberation-at-doubt present in transcript",
                               blindspot="reasoned_cave", receipt=bs["receipt"])
        else:
            cap = self.capabilities["behavioral_grounding"]
            v = WitnessVerdict(status="FLAG" if caved else "OK", capability=cap.name,
                               detail="resampling monitor licensed on this transcript",
                               payload={"caved": caved})
        self._emit(v)
        return v

    def substrate_divergence(self, *read_args, **read_kwargs) -> WitnessVerdict:
        """Pass-through to a configured mount's read path (read-only, its calibration)."""
        if self._mount is None:
            return WitnessVerdict(status="REFUSED", capability="borrowed_conscience",
                                  detail="no mount configured; the witness will not "
                                         "improvise a substrate read")
        cap = self.capabilities["borrowed_conscience"]
        reading = self._mount.read(*read_args, **read_kwargs)
        # Gate on the mount's calibrated verdict, not on the truthiness of the
        # reading object — a ConscienceReading is a plain dataclass and is truthy
        # on every read, which would make OK unreachable and the mount's per-axis
        # calibration decorative (the fired-or-needs_revision bug class).
        v = WitnessVerdict(status="FLAG" if reading.caught else "OK", capability=cap.name,
                           detail=f"mount read at catch {cap.operating_point['catch']} / "
                                  f"FPR budget {cap.operating_point['fpr_budget']} — a "
                                  "cooperative monitor, not an adversarial defense",
                           payload=reading)
        self._emit(v)
        return v

    def knowsay(self, records) -> WitnessVerdict:
        """Know-say datasheet, refusal-first contract intact."""
        from .knowsay import datasheet
        sheet = datasheet(records)
        cap = self.capabilities["knowsay_datasheet"]
        v = WitnessVerdict(status="OK", capability=cap.name,
                           detail="refusal-first datasheet (underpowered cells are None "
                                  "with the failing floor named)",
                           payload=sheet)
        self._emit(v)
        return v

    def report(self) -> str:
        """One-screen statement of every power and its boundary — the §8 table, live."""
        lines = ["witness — powers and their measured boundaries", ""]
        for cap in self.capabilities.values():
            op = ", ".join(f"{k}={v}" for k, v in cap.operating_point.items())
            lines.append(f"  {cap.name}: {op}")
            lines.append(f"    receipt: {cap.receipt}")
            for b in cap.blindspots:
                lines.append(f"    bounded by: {b} — {self.blindspots[b]['statement']}")
        lines.append("")
        lines.append("  refused by measurement: steering (read != write), "
                     "self-verification (bounded by self-knowledge)")
        return "\n".join(lines)

    # -- internals ---------------------------------------------------------

    def _emit(self, v: WitnessVerdict) -> None:
        if self._on_flag is not None and v.status in ("FLAG", "ABSTAIN"):
            self._on_flag(v)
