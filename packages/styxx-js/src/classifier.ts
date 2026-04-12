/**
 * CentroidClassifier — port of Python vitals.py CentroidClassifier.
 *
 * z-score features, compute euclidean distance to 6 centroids,
 * pseudo-softmax to probabilities. Deterministic, no randomness.
 */

import type { CentroidArtifact, Category, PhaseName } from "./types";
import { PHASE_TOKEN_CUTOFFS, CATEGORIES } from "./types";
import { extractFeatures } from "./features";
import { PhaseReading } from "./vitals";
import type { Trajectories } from "./types";
import atlas from "./centroids/atlas_v0_3.json";

function euclideanNorm(v: number[]): number {
  let sum = 0;
  for (let i = 0; i < v.length; i++) sum += v[i] * v[i];
  return Math.sqrt(sum);
}

export class CentroidClassifier {
  private artifact: CentroidArtifact;

  constructor(artifact?: CentroidArtifact) {
    this.artifact = (artifact ?? atlas) as CentroidArtifact;
  }

  classify(trajectories: Trajectories, phase: PhaseName): PhaseReading {
    const nTokens = PHASE_TOKEN_CUTOFFS[phase];
    const feats = extractFeatures(trajectories, nTokens);
    const phaseData = this.artifact.phases[phase];
    const { mu, sigma, centroids } = phaseData;

    // Z-score: z[i] = (feats[i] - mu[i]) / sigma[i]
    const z = feats.map((f, i) => (f - mu[i]) / sigma[i]);

    // Euclidean distance to each centroid
    const distances: Record<string, number> = {};
    for (const cat of CATEGORIES) {
      const centroid = centroids[cat];
      const diff = z.map((zi, i) => zi - centroid[i]);
      distances[cat] = euclideanNorm(diff);
    }

    // Sort by distance — find nearest
    const sorted = Object.entries(distances).sort((a, b) => a[1] - b[1]);
    const [nearest, nearestD] = sorted[0];
    const runnerUpD = sorted.length > 1 ? sorted[1][1] : nearestD;
    const margin = runnerUpD - nearestD;

    // Pseudo-softmax: score[cat] = exp(-(distance - nearestD))
    const scores: Record<string, number> = {};
    for (const cat of CATEGORIES) {
      scores[cat] = Math.exp(-(distances[cat] - nearestD));
    }
    let total = 0;
    for (const cat of CATEGORIES) total += scores[cat];
    if (total === 0) total = 1;

    const probs: Record<string, number> = {};
    for (const cat of CATEGORIES) {
      probs[cat] = scores[cat] / total;
    }

    return new PhaseReading(
      phase,
      nTokens,
      feats,
      nearest as Category,
      margin,
      distances,
      probs
    );
  }
}
