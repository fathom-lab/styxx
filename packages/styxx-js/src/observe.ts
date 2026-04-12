/**
 * observe() — extract logprobs from OpenAI responses and compute vitals.
 *
 * Port of Python watch.py observe() and observe_raw().
 */

import type { Trajectories } from "./types";
import { StyxxRuntime } from "./runtime";
import { Vitals } from "./vitals";

const runtime = new StyxxRuntime();

/**
 * Observe an OpenAI ChatCompletion response and compute cognitive vitals.
 *
 * Extracts logprobs from response.choices[0].logprobs.content,
 * computes entropy/logprob/top2_margin trajectories, and runs the
 * classifier. Returns null if logprobs are not available.
 *
 * @example
 * ```ts
 * const r = await client.chat.completions.create({
 *   model: "gpt-4o",
 *   messages: [{ role: "user", content: "hello" }],
 *   logprobs: true,
 *   top_logprobs: 5,
 * });
 * const vitals = observe(r);
 * console.log(vitals?.phase4);  // "reasoning:0.45"
 * ```
 */
export function observe(response: any): Vitals | null {
  try {
    const trajectories = extractOpenAILogprobs(response);
    if (!trajectories) return null;
    return runtime.runOnTrajectories(trajectories);
  } catch {
    return null;
  }
}

/**
 * Compute vitals from raw trajectory arrays (no response parsing).
 *
 * @example
 * ```ts
 * const vitals = observeRaw({
 *   entropy: [2.1, 1.8, 1.5, ...],
 *   logprob: [-0.5, -0.3, -0.2, ...],
 *   top2_margin: [0.4, 0.5, 0.6, ...],
 * });
 * ```
 */
export function observeRaw(trajectories: Trajectories): Vitals {
  return runtime.runOnTrajectories(trajectories);
}

/**
 * Extract entropy/logprob/top2_margin from an OpenAI ChatCompletion.
 *
 * Uses the top-5 logprob bridge (same as Python):
 *   entropy = -sum(p * log(p)) over top-5 tokens
 *   logprob = chosen token's logprob
 *   top2_margin = top1_logprob - top2_logprob
 */
function extractOpenAILogprobs(response: any): Trajectories | null {
  const choice = response?.choices?.[0];
  if (!choice) return null;

  const content = choice?.logprobs?.content;
  if (!content || !Array.isArray(content) || content.length === 0)
    return null;

  const entropy: number[] = [];
  const logprob: number[] = [];
  const top2_margin: number[] = [];

  for (const tokenData of content) {
    // Chosen token logprob
    const chosenLp = tokenData?.logprob ?? 0;
    logprob.push(chosenLp);

    // Top logprobs for entropy + margin
    const topLps: { logprob: number }[] = tokenData?.top_logprobs ?? [];

    if (topLps.length === 0) {
      entropy.push(0);
      top2_margin.push(0);
      continue;
    }

    // Entropy: -sum(p * log(p)) over top-k
    const lps = topLps.map((t) => t.logprob);
    const probs = lps.map((lp) => Math.exp(lp));
    const probSum = probs.reduce((a, b) => a + b, 0) || 1;
    const normalized = probs.map((p) => p / probSum);
    let ent = 0;
    for (const p of normalized) {
      if (p > 0) ent -= p * Math.log(p);
    }
    entropy.push(ent);

    // Top-2 margin
    const sortedLps = [...lps].sort((a, b) => b - a);
    const margin =
      sortedLps.length >= 2 ? sortedLps[0] - sortedLps[1] : 0;
    top2_margin.push(margin);
  }

  return { entropy, logprob, top2_margin };
}
