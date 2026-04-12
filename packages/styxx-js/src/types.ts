/**
 * styxx type definitions.
 *
 * Matches the Python Vitals.as_dict() output shape exactly.
 */

export type Category =
  | "retrieval"
  | "reasoning"
  | "refusal"
  | "creative"
  | "adversarial"
  | "hallucination";

export type Gate = "pass" | "warn" | "fail" | "pending";

export type PhaseName =
  | "phase1_preflight"
  | "phase2_early"
  | "phase3_mid"
  | "phase4_late";

export interface Trajectories {
  entropy: number[];
  logprob: number[];
  top2_margin: number[];
}

export interface PhaseReadingData {
  phase: string;
  n_tokens_used: number;
  features: number[];
  predicted_category: Category;
  margin: number;
  distances: Record<string, number>;
  probs: Record<string, number>;
  confidence: number;
  d_honesty_mean?: number | null;
  d_honesty_std?: number | null;
  d_honesty_delta?: number | null;
}

export interface VitalsData {
  phase1_pre: PhaseReadingData;
  phase2_early: PhaseReadingData | null;
  phase3_mid: PhaseReadingData | null;
  phase4_late: PhaseReadingData | null;
  tier_active: number;
  abort_reason: string | null;
}

export interface CentroidArtifact {
  categories: Category[];
  phases: Record<
    string,
    {
      mu: number[];
      sigma: number[];
      centroids: Record<string, number[]>;
    }
  >;
  phase_token_cutoffs: Record<string, number>;
}

export const CATEGORIES: Category[] = [
  "retrieval",
  "reasoning",
  "refusal",
  "creative",
  "adversarial",
  "hallucination",
];

export const PHASE_TOKEN_CUTOFFS: Record<PhaseName, number> = {
  phase1_preflight: 1,
  phase2_early: 5,
  phase3_mid: 15,
  phase4_late: 25,
};
