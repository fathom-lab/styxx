/**
 * OpenAI adapter — wraps the OpenAI JS SDK with cognitive vitals.
 *
 * @example
 * ```ts
 * import { withVitals } from "@fathom-lab/styxx";
 * import OpenAI from "openai";
 *
 * const client = withVitals(new OpenAI());
 * const r = await client.chat.completions.create({
 *   model: "gpt-4o",
 *   messages: [{ role: "user", content: "why is the sky blue?" }],
 * });
 *
 * console.log(r.choices[0].message.content); // text, unchanged
 * console.log((r as any).vitals?.phase4);    // "reasoning:0.45"
 * console.log((r as any).vitals?.gate);      // "pass"
 * ```
 */

import { observe } from "../observe";
import type { Vitals } from "../vitals";

/**
 * Wrap an OpenAI client so every chat completion gets .vitals attached.
 *
 * Injects `logprobs: true, top_logprobs: 5` into create params if not
 * already set. After the response, runs observe() and attaches .vitals.
 * Fail-open: if anything goes wrong, the original response is returned
 * unmodified.
 */
export function withVitals<T extends object>(client: T): T {
  const chat = (client as any)?.chat;
  if (!chat?.completions?.create) return client;

  const originalCreate = chat.completions.create.bind(chat.completions);

  chat.completions.create = async function (
    params: any,
    ...rest: any[]
  ): Promise<any> {
    // Inject logprobs if not already set
    const enrichedParams = { ...params };
    if (enrichedParams.logprobs === undefined) {
      enrichedParams.logprobs = true;
    }
    if (enrichedParams.top_logprobs === undefined && enrichedParams.logprobs) {
      enrichedParams.top_logprobs = 5;
    }

    // Call the original
    const response = await originalCreate(enrichedParams, ...rest);

    // Attach vitals (fail-open)
    try {
      const vitals = observe(response);
      (response as any).vitals = vitals;
    } catch {
      (response as any).vitals = null;
    }

    return response;
  };

  return client;
}

/**
 * Explicit function-style wrapper (no Proxy magic).
 *
 * For users who prefer calling a function over wrapping the client.
 */
export async function createWithVitals(
  client: any,
  params: any
): Promise<any & { vitals: Vitals | null }> {
  const enrichedParams = { ...params };
  if (enrichedParams.logprobs === undefined) {
    enrichedParams.logprobs = true;
  }
  if (enrichedParams.top_logprobs === undefined && enrichedParams.logprobs) {
    enrichedParams.top_logprobs = 5;
  }

  const response = await client.chat.completions.create(enrichedParams);

  try {
    (response as any).vitals = observe(response);
  } catch {
    (response as any).vitals = null;
  }

  return response;
}
