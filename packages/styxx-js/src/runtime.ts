/**
 * StyxxRuntime — phase pipeline + gate evaluation.
 *
 * Port of Python core.py StyxxRuntime.run_on_trajectories().
 */

import type { Trajectories, PhaseName } from "./types";
import { PHASE_TOKEN_CUTOFFS } from "./types";
import { CentroidClassifier } from "./classifier";
import { Vitals } from "./vitals";

export class StyxxRuntime {
  classifier: CentroidClassifier;

  constructor(classifier?: CentroidClassifier) {
    this.classifier = classifier ?? new CentroidClassifier();
  }

  runOnTrajectories(trajectories: Trajectories): Vitals {
    const n = trajectories.entropy.length;

    // Phase 1 — always runs (needs >= 1 token)
    const phase1 = this.classifier.classify(trajectories, "phase1_preflight");

    // Phase 2 — needs >= 5 tokens
    const phase2 = n >= 5
      ? this.classifier.classify(trajectories, "phase2_early")
      : null;

    // Phase 3 — needs >= 15 tokens
    const phase3 = n >= 15
      ? this.classifier.classify(trajectories, "phase3_mid")
      : null;

    // Phase 4 — needs >= 25 tokens
    const phase4 = n >= 25
      ? this.classifier.classify(trajectories, "phase4_late")
      : null;

    return new Vitals(phase1, phase2, phase3, phase4);
  }
}
