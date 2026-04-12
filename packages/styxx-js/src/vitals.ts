/**
 * Vitals and PhaseReading classes.
 *
 * Matches Python vitals.py Vitals and PhaseReading exactly.
 */

import type {
  Category,
  Gate,
  PhaseReadingData,
  VitalsData,
} from "./types";

const CHANCE_FLOOR = 0.2;

export class PhaseReading {
  phase: string;
  n_tokens_used: number;
  features: number[];
  predicted_category: Category;
  margin: number;
  distances: Record<string, number>;
  probs: Record<string, number>;
  d_honesty_mean: number | null;
  d_honesty_std: number | null;
  d_honesty_delta: number | null;

  constructor(
    phase: string,
    nTokensUsed: number,
    features: number[],
    predictedCategory: Category,
    margin: number,
    distances: Record<string, number>,
    probs: Record<string, number>,
    dHonestyMean: number | null = null,
    dHonestyStd: number | null = null,
    dHonestyDelta: number | null = null
  ) {
    this.phase = phase;
    this.n_tokens_used = nTokensUsed;
    this.features = features;
    this.predicted_category = predictedCategory;
    this.margin = margin;
    this.distances = distances;
    this.probs = probs;
    this.d_honesty_mean = dHonestyMean;
    this.d_honesty_std = dHonestyStd;
    this.d_honesty_delta = dHonestyDelta;
  }

  get confidence(): number {
    return this.probs[this.predicted_category] ?? 0;
  }

  top3(): [string, number][] {
    return Object.entries(this.probs)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3);
  }

  asDict(): PhaseReadingData {
    return {
      phase: this.phase,
      n_tokens_used: this.n_tokens_used,
      features: this.features,
      predicted_category: this.predicted_category,
      margin: this.margin,
      distances: { ...this.distances },
      probs: { ...this.probs },
      confidence: this.confidence,
      d_honesty_mean: this.d_honesty_mean,
      d_honesty_std: this.d_honesty_std,
      d_honesty_delta: this.d_honesty_delta,
    };
  }
}

export class Vitals {
  phase1_pre: PhaseReading;
  phase2_early: PhaseReading | null;
  phase3_mid: PhaseReading | null;
  phase4_late: PhaseReading | null;
  tier_active: number;
  abort_reason: string | null;

  constructor(
    phase1: PhaseReading,
    phase2: PhaseReading | null = null,
    phase3: PhaseReading | null = null,
    phase4: PhaseReading | null = null,
    tierActive: number = 0,
    abortReason: string | null = null
  ) {
    this.phase1_pre = phase1;
    this.phase2_early = phase2;
    this.phase3_mid = phase3;
    this.phase4_late = phase4;
    this.tier_active = tierActive;
    this.abort_reason = abortReason;
  }

  /** Compact "category:confidence" string for phase 1. */
  get phase1(): string {
    const p = this.phase1_pre;
    return `${p.predicted_category}:${p.confidence.toFixed(2)}`;
  }

  /** Compact "category:confidence" string for phase 4 (or "-"). */
  get phase4(): string {
    if (!this.phase4_late) return "-";
    const p = this.phase4_late;
    return `${p.predicted_category}:${p.confidence.toFixed(2)}`;
  }

  /** Gate status from phase 4 prediction. */
  get gate(): Gate {
    if (!this.phase4_late) return "pending";
    const pred = this.phase4_late.predicted_category;
    const conf = this.phase4_late.confidence;
    if (pred === "hallucination" && conf > CHANCE_FLOOR) return "fail";
    if ((pred === "refusal" || pred === "adversarial") && conf > CHANCE_FLOOR)
      return "warn";
    return "pass";
  }

  /** True if gate is warn or fail. */
  isConcerning(): boolean {
    const g = this.gate;
    return g === "warn" || g === "fail";
  }

  asDict(): VitalsData {
    return {
      phase1_pre: this.phase1_pre.asDict(),
      phase2_early: this.phase2_early?.asDict() ?? null,
      phase3_mid: this.phase3_mid?.asDict() ?? null,
      phase4_late: this.phase4_late?.asDict() ?? null,
      tier_active: this.tier_active,
      abort_reason: this.abort_reason,
    };
  }

  asJson(indent: number = 2): string {
    return JSON.stringify(this.asDict(), null, indent);
  }
}
