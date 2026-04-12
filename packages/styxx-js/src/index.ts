/**
 * @fathom-lab/styxx — cognitive vitals monitor for LLM agents.
 *
 * nothing crosses unseen.
 *
 * @example
 * ```ts
 * import { observe, withVitals } from "@fathom-lab/styxx";
 * import OpenAI from "openai";
 *
 * // Option 1: wrap the client
 * const client = withVitals(new OpenAI());
 * const r = await client.chat.completions.create({ model: "gpt-4o", messages: [...] });
 * console.log(r.vitals?.gate);  // "pass"
 *
 * // Option 2: observe a response manually
 * const vitals = observe(response);
 * console.log(vitals?.phase4);  // "reasoning:0.45"
 *
 * // Option 3: raw trajectories
 * const vitals2 = observeRaw({ entropy: [...], logprob: [...], top2_margin: [...] });
 * ```
 */

// Core
export { Vitals, PhaseReading } from "./vitals";
export { CentroidClassifier } from "./classifier";
export { StyxxRuntime } from "./runtime";

// Observation
export { observe, observeRaw } from "./observe";

// OpenAI adapter
export { withVitals, createWithVitals } from "./adapters/openai";

// Types
export type {
  VitalsData,
  PhaseReadingData,
  Category,
  Gate,
  Trajectories,
  PhaseName,
} from "./types";
export { CATEGORIES, PHASE_TOKEN_CUTOFFS } from "./types";

// Version
export const VERSION = "0.1.0";
