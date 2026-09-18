/**
 * TRIAGE — Jev decides what gets *read*. It never decides what gets *said*.
 *
 * DECIDE-1 hand-adjudicated 100 agent-PR claims and found 71% of them settleable
 * from the diff alone (52% for `only_touches`, 13 of 25). Against the current
 * instrument the gate returns a verdict on 16 of 299 `only_touches` claims --
 * 5.4% [3.3%, 8.5%] -- counted in `scope1_footprint.json`: 8 VERIFIED plus 8
 * CONTRADICTED, 281 UNCHECKABLE, 2 with no verdict. (DECIDE-1's own headline says
 * 17 of 299; it was measured a day earlier against instrument `4ba947a8`, three
 * changes ago. The two papers do not cross-reference each other, and until one of
 * them does, the receipted count is the one to quote.)
 *
 * The gap is not judgement, it is extraction: the reader cannot tell a sentence
 * that claims a file scope from one that merely contains the word "only".
 *
 * PATH-1 wrote that down and gave up on it:
 *
 *   > Modes 3-6 have the same surface shape as claims that are genuinely
 *   > checkable. DECIDE-1 adjudicated the difference by reading the sentences;
 *   > no test available to the instrument separates them.
 *
 * Jev is a test that reads the sentence. So this module asks it one question --
 * *is this sentence a claim about which files this pull request changes?* -- and
 * nothing else.
 *
 * ## The rule this file exists to enforce
 *
 * **A triage answer is never a verdict.** It can only move a sentence from
 * "ignored" to "handed to the deterministic reader". Every VERIFIED and every
 * CONTRADICTED styxx emits is still computed from the diff bytes by code you can
 * run offline, and `bench_reproduce.py` still re-derives all of it without a
 * network. A model in this loop widens the input; it cannot change an answer.
 *
 * That is not a stylistic preference. A hosted model is not reproducible, cannot
 * be sealed into a capsule, and can change under us without notice. The moment a
 * verdict depends on one, "check it yourself without trusting anyone in this
 * repository" stops being true, and that sentence is the entire product.
 *
 * ## Refusal
 *
 * Jev always returns a number. An instrument that always returns a number cannot
 * refuse, and `styxx.diffgate`'s whole thesis is that an instrument that cannot
 * refuse cannot be trusted. So the band around 0.5 is spent: inside it this
 * module returns `UNDECIDED` and the sentence is dropped exactly as it is today.
 * The thresholds are arguments rather than constants because the right values are
 * a measurement, not a taste -- see CALIB-1, and do not ship a default until it
 * has run.
 *
 * ## The key
 *
 * The SDK reads `TYPESAFE_API_KEY` from the environment itself. Nothing in this
 * file reads, stores, logs or forwards it, and `TriageReport` carries no
 * credential. The client is injected so that the deterministic package never has
 * to import a network library, and so the tests run without a key.
 */

/** What the deterministic reader would do with a sentence, once triage has spoken. */
export type TriageVerdict =
  /** Hand it to the reader. Jev says this is a claim about file scope. */
  | "READ"
  /** Drop it, as today. Jev says this is not a claim about file scope. */
  | "SKIP"
  /** Inside the refusal band, or Jev could not be reached. Drop it, as today. */
  | "UNDECIDED";

export interface TriageThresholds {
  /**
   * `noul` at or above this reads the sentence. Preregister it; do not guess it.
   */
  readonly read: number;
  /**
   * `noul` at or below this skips the sentence. Between the two is the refusal
   * band and the answer is UNDECIDED.
   */
  readonly skip: number;
}

export interface TriageReport {
  readonly verdict: TriageVerdict;
  /** Jev's probability that the sentence claims a file scope, 0..1. */
  readonly noul: number | null;
  /** Distance from maximum uncertainty, `|noul - 0.5| * 2`. Null when unreached. */
  readonly decisiveness: number | null;
  /** Plain-words reason, in the style of the Python gate's `why`. */
  readonly why: string;
  /** The exact question asked, so a reader can see what the model was shown. */
  readonly question: string;
  readonly thresholds: TriageThresholds;
  /**
   * Which model answered, and what the call cost in tokens.
   *
   * A verdict in this repository names the sha256 of the instrument that produced
   * it. A routing decision cannot do that, so it names the next best thing: the
   * model string the service reported and the tokens it billed. Null when Jev was
   * not reached.
   */
  readonly provenance: { readonly model: string; readonly usage: JevUsage } | null;
}

export interface JevUsage {
  readonly input_tokens: number;
  readonly output_tokens: number;
}

/**
 * The shape this module needs from `@typesafe-ai/sdk`, and no more.
 *
 * Declared structurally rather than imported so that the deterministic package
 * takes no hard dependency on a network client, and so the tests run with a fake
 * and no key. The real client is `new TypeSafeClient()` from `@typesafe-ai/sdk`
 * v0.6.0 (Node 20+); these types are transcribed from its `index.d.mts` rather
 * than from the prose documentation, which describes `noul` criteria as a string
 * and is wrong — it is an object keyed `true` / `false`.
 */
export interface JevClient {
  systemOne(request: {
    state: string;
    questions: Record<string, JevNoulQuestion>;
  }): Promise<{
    model: string;
    answers: Record<string, { type?: string; noul?: number }>;
    usage: JevUsage;
  }>;
}

export interface JevNoulQuestion {
  readonly type: "noul";
  readonly instructions?: string;
  readonly criteria?: { readonly true?: string; readonly false?: string } | null;
}

/** The `noul(...)` question constructor, injected for the same reason. */
export type NoulFactory = (
  instructions: string,
  criteria?: { true?: string; false?: string } | null,
) => JevNoulQuestion;

const QUESTION =
  "Does this sentence state which files this pull request changes? " +
  "Answer about the sentence's subject, not whether it is true.";

const CRITERIA = {
  true:
    "The sentence makes a claim about the set of files, paths or directories this " +
    "pull request touches — a scope it stayed inside, or files it did not go near.",
  false:
    "The sentence describes what the code does at runtime, quotes documentation or " +
    "an instruction to someone, or uses a word like 'only' about something that is " +
    "not a path.",
} as const;

/**
 * Ask Jev whether a sentence is a file-scope claim.
 *
 * `changedPaths` is passed as state so the model is judging the sentence against
 * this diff rather than in the abstract -- the cert-controller case turns on
 * `.githiub` being a path nothing in the diff resembles. Paths are truncated:
 * the question is about the sentence, and a 600-file diff would drown it.
 *
 * `totalPaths` exists because a caller's list is sometimes already short of the
 * diff. `decide1_adjudication.json` caps `paths` at 25 while recording the real
 * `n_files`, and four of the 25 `only_touches` items are capped that way -- one
 * of them a 175-file pull request. Deriving the total from the array would tell
 * the model that a 175-file diff has 25 files in it, which is the opposite of
 * the reason the paths are shown at all. Callers that hold the true count pass
 * it; the default is the array's own length, so nothing changes for callers that
 * hand over everything. A total below the number of paths supplied is a caller
 * bug and throws rather than being quietly clamped.
 */
export async function triageSentence(
  sentence: string,
  changedPaths: readonly string[],
  thresholds: TriageThresholds,
  client: JevClient,
  noul: NoulFactory,
  maxPaths = 40,
  totalPaths = changedPaths.length,
): Promise<TriageReport> {
  assertThresholds(thresholds);
  assertTotal(totalPaths, changedPaths.length);

  const shown = changedPaths.slice(0, maxPaths);
  const omitted = totalPaths - shown.length;
  const state =
    `Sentence from a pull request description:\n${sentence}\n\n` +
    `Files this pull request changes (${totalPaths} total` +
    `${omitted > 0 ? `, ${omitted} not shown` : ""}):\n` +
    (shown.length ? shown.map((p) => `- ${p}`).join("\n") : "- (none)");

  let answer: number;
  let provenance: TriageReport["provenance"] = null;
  try {
    const res = await client.systemOne({
      state,
      questions: { file_scope_claim: noul(QUESTION, CRITERIA) },
    });
    const raw = res?.answers?.file_scope_claim?.noul;
    if (typeof raw !== "number" || !Number.isFinite(raw) || raw < 0 || raw > 1) {
      return unreached(thresholds, `jev returned no usable noul (${String(raw)})`);
    }
    answer = raw;
    if (typeof res.model === "string" && res.usage) {
      provenance = { model: res.model, usage: res.usage };
    }
  } catch (err) {
    // Unreachable, unauthorised, rate-limited: all the same answer, UNDECIDED.
    // Triage can only ever ADD sentences to what the reader sees, so a failure
    // adds none and the gate behaves exactly as it does today. An outage must
    // never be able to change a verdict, in either direction.
    return unreached(thresholds, `jev unreachable: ${errText(err)}`);
  }

  const decisiveness = Math.abs(answer - 0.5) * 2;
  if (answer >= thresholds.read) {
    return {
      verdict: "READ",
      noul: answer,
      decisiveness,
      why: `jev reads this as a file-scope claim (noul ${answer.toFixed(3)} >= ${thresholds.read})`,
      question: QUESTION,
      thresholds,
      provenance,
    };
  }
  if (answer <= thresholds.skip) {
    return {
      verdict: "SKIP",
      noul: answer,
      decisiveness,
      why: `jev reads this as not a file-scope claim (noul ${answer.toFixed(3)} <= ${thresholds.skip})`,
      question: QUESTION,
      thresholds,
      provenance,
    };
  }
  return {
    verdict: "UNDECIDED",
    noul: answer,
    decisiveness,
    why:
      `noul ${answer.toFixed(3)} is inside the refusal band ` +
      `(${thresholds.skip}, ${thresholds.read}) — dropped, as without triage`,
    question: QUESTION,
    thresholds,
    provenance,
  };
}

function unreached(thresholds: TriageThresholds, why: string): TriageReport {
  return {
    verdict: "UNDECIDED",
    noul: null,
    decisiveness: null,
    why,
    question: QUESTION,
    thresholds,
    provenance: null,
  };
}

function errText(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function assertTotal(total: number, supplied: number): void {
  if (!Number.isInteger(total) || total < supplied) {
    throw new RangeError(
      `totalPaths must be an integer >= the number of paths supplied; ` +
        `got ${total} with ${supplied} paths`,
    );
  }
}

function assertThresholds(t: TriageThresholds): void {
  const ok =
    Number.isFinite(t.read) &&
    Number.isFinite(t.skip) &&
    t.skip >= 0 &&
    t.read <= 1 &&
    t.skip < t.read;
  if (!ok) {
    throw new RangeError(
      `triage thresholds must satisfy 0 <= skip < read <= 1; got skip=${t.skip} read=${t.read}`,
    );
  }
}

/**
 * There is deliberately no exported default threshold pair.
 *
 * TypeSafe's own documentation declines to say how `confidence` is computed, says
 * nothing about whether the probabilities are calibrated, and states that the
 * right thresholds "depend on your domain and the performance of the model for
 * your use case". This repository does not take a vendor's word for a number it
 * has not measured -- that is what EXTERNAL-1 and BENCH-2 are about, and both of
 * them were measurements against us.
 *
 * CALIB-1 measures calibration on the 100 claims DECIDE-1 adjudicated by hand,
 * which were labelled before this model was in the picture. Until it has run and
 * published a number, every caller states its own thresholds and owns them.
 */
export const NO_DEFAULT_THRESHOLDS = true as const;
