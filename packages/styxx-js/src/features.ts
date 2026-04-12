/**
 * Feature extraction — 12-dim vector from logprob trajectories.
 *
 * Computes (mean, std, min, max) for each of (entropy, logprob, top2_margin)
 * over tokens [0, nTokens). Matches Python extract_features() exactly.
 */

import type { Trajectories } from "./types";

const SIGNALS = ["entropy", "logprob", "top2_margin"] as const;

function mean(arr: number[]): number {
  if (arr.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < arr.length; i++) sum += arr[i];
  return sum / arr.length;
}

function std(arr: number[]): number {
  // ddof=1 (Bessel's correction) to match numpy
  if (arr.length <= 1) return 0;
  const m = mean(arr);
  let sumSq = 0;
  for (let i = 0; i < arr.length; i++) {
    const d = arr[i] - m;
    sumSq += d * d;
  }
  return Math.sqrt(sumSq / (arr.length - 1));
}

export function extractFeatures(
  trajectories: Trajectories,
  nTokens: number
): number[] {
  const feats: number[] = [];

  for (const signal of SIGNALS) {
    const raw = trajectories[signal];
    if (!raw || raw.length === 0) {
      feats.push(0, 0, 0, 0);
      continue;
    }
    const window = raw.slice(0, nTokens);
    if (window.length === 0) {
      feats.push(0, 0, 0, 0);
      continue;
    }
    feats.push(mean(window));
    feats.push(std(window));
    feats.push(Math.min(...window));
    feats.push(Math.max(...window));
  }

  return feats;
}
