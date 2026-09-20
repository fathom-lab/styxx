/**
 * CALIB-1 asker — puts the 25 `only_touches` sentences to Jev and writes down
 * what came back. It does not score anything.
 *
 *   node --experimental-strip-types calib1_ask.ts --dry-run
 *   TYPESAFE_API_KEY=... node --experimental-strip-types calib1_ask.ts
 *
 * Node 22.6+ (type stripping) and Node 20+ (the SDK). No build step, no new
 * dependency: `@typesafe-ai/sdk` is imported only on the path that actually
 * calls it, so `--dry-run` runs in a checkout that has never installed it.
 *
 * ## Why this is two programs
 *
 * This one spends money and talks to a server. `calib1_score.py` reads its
 * output and applies the gates. Splitting them means the thresholds can be
 * chosen and the gates re-run over the same recorded answers as many times as a
 * reader likes, without another call, and a reader who does not trust the
 * scoring can take `calib1_raw.json` and write their own. It also means a
 * failed scoring run never costs anything to repeat.
 *
 * ## What this program is structurally unable to do
 *
 * It passes `{ skip: 0, read: 1 }` as thresholds, which makes the entire range
 * the refusal band, and it writes no verdict field at any point. CALIB-1's whole
 * design is that thresholds are chosen on a development split *after* the
 * answers exist (G-C1-1). A recorder that could emit a verdict is a recorder
 * that could quietly pick a threshold, so this one cannot.
 *
 * ## The key
 *
 * Read by the SDK from `TYPESAFE_API_KEY` in the environment. This file never
 * reads it, never logs it, never writes it, and refuses to run rather than
 * accept one on the command line.
 */

import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import {
  triageSentence,
  type JevClient,
  type NoulFactory,
  type JevNoulQuestion,
} from "../../packages/styxx-js/src/triage.ts";

const HERE = dirname(fileURLToPath(import.meta.url));
const ADJUDICATION = join(HERE, "decide1_adjudication.json");
const PREREG = join(HERE, "PREREG_calib1_jev_2026_09_18.md");
const DEFAULT_OUT = join(HERE, "calib1_raw.json");

/**
 * Both hashes, because the document has one of each.
 *
 * `AT_FREEZE` is what it hashed to when it was frozen, before any call existed.
 * `PREREG_SHA256` is what it hashes to now: Amendments A and B were appended --
 * never edited in -- before the first call, as A6 and B1 record. The runner
 * checks the file against the amended hash and writes both down, so a reader can
 * verify the appends without taking anyone's word for what was in the frozen half.
 */
const PREREG_SHA256_AT_FREEZE = "7d550cd50473c6642770662149553eaf3db1e1a52308e8d80e1d518b3ad94aaa";
const PREREG_SHA256 = "52d3a5cc4c15f6f9ad69dd3fe7a445292a0d7e1c546545354f20fbf0053d61ea";

/**
 * The model this calibration is of. Amendment B1.
 *
 * `@typesafe-ai/sdk` v0.6.0 resolves an omitted `model` to the client's
 * `defaultModel`, which falls back to `TYPESAFE_DEFAULT_MODEL` and then to
 * `jev-latest` -- an alias TypeSafe documents as free to move, and which an
 * operator's environment can redirect without appearing in the receipt. A
 * calibration of an alias is a calibration of nothing in particular, so this is
 * set on the client AND on every request, and every answer is checked against it.
 */
const PINNED_MODEL = "jev-1.13.0";

/**
 * Characters of `state` allowed, standing in for Jev's 32k `state` + longest
 * question budget (B2).
 *
 * This counts characters, not tokens, on purpose: there is no Jev tokenizer here
 * and inventing a chars-per-token ratio would be a guess wearing a bound's
 * clothes. Every BPE token is at least one character, so a `state` of N
 * characters is at most N tokens -- the bound below is therefore SOUND rather
 * than estimated, at the cost of being loose. 2000 characters are reserved for
 * the question and criteria, which are fixed strings in triage.ts.
 */
const STATE_CHAR_BUDGET = 30_000;

/** Recording only; see the header. Never used to decide anything. */
const RECORD_ONLY_THRESHOLDS = { skip: 0, read: 1 } as const;

interface AdjudicationItem {
  readonly id: number;
  readonly kind: string;
  readonly url: string;
  readonly claim: string;
  readonly paths: readonly string[];
  readonly paths_truncated: boolean;
  readonly n_files: number;
  readonly decidable: boolean;
  readonly reason_code: string | null;
}

interface Call {
  readonly id: number;
  readonly repeat: number;
  readonly noul: number | null;
  readonly ms: number;
  readonly model: string | null;
  readonly input_tokens: number | null;
  readonly output_tokens: number | null;
  readonly error: string | null;
}

function sha256(path: string): string {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

const noul: NoulFactory = (instructions, criteria) =>
  ({ type: "noul", instructions, criteria: criteria ?? null }) as JevNoulQuestion;

/** True when an answer did not come from the pinned model. Null means the call never reached it. */
function modelMismatch(seen: string | null): boolean {
  return typeof seen === "string" && seen !== PINNED_MODEL;
}

/**
 * A stub that answers from a hash of the sentence.
 *
 * Deterministic, free, and reachable with no key, so the whole pipeline —
 * selection, state construction, repeat loop, output schema, scorer — can be
 * proved to work before a single real call is paid for. Its answers are
 * arbitrary by construction and the output is stamped `dry_run: true` so the
 * scorer refuses to treat them as evidence.
 */
function stubClient(): JevClient {
  return {
    async systemOne(request) {
      const h = createHash("sha256").update(request.state).digest();
      return {
        model: "dry-run-stub",
        answers: {
          file_scope_claim: { type: "noul", noul: h.readUInt16BE(0) / 65535 },
        },
        usage: { input_tokens: Math.ceil(request.state.length / 4), output_tokens: 1 },
      };
    },
  };
}

async function realClient(): Promise<JevClient> {
  if (!process.env.TYPESAFE_API_KEY) {
    throw new Error(
      "TYPESAFE_API_KEY is not in the environment. Put it there or in a repository " +
        "secret; this program will not accept it as an argument. Use --dry-run to " +
        "exercise everything except the calls.",
    );
  }
  const sdk = await import("@typesafe-ai/sdk");
  const client = new sdk.TypeSafeClient({ defaultModel: PINNED_MODEL });
  return pinned(client as unknown as JevClient);
}

/** The largest `state` any client has been handed, for the budget check (B2). */
const widest = { chars: 0 };

/**
 * Pin the model on every request and remember how wide the state was.
 *
 * This wrapper does NOT throw on a wrong answer, and that is deliberate.
 * `triageSentence` catches everything a client throws and returns UNDECIDED, so a
 * mismatch raised here would be recorded as an unreachable call and scored as a
 * null -- an absent measurement reported as an ordinary one. The comparison is
 * made in main(), on `report.provenance.model`, outside that catch. See B1.
 */
function pinned(client: JevClient): JevClient {
  // `JevClient` is the subset triage.ts needs and deliberately carries no `model`:
  // the shipped module takes no dependency on a network client and has no business
  // holding a vendor's version string. The SDK's own SystemOneRequest does have
  // one -- "Model override; omitted values inherit defaultModel" -- and the SDK
  // forwards additional properties, so the field is declared here, where the
  // experiment lives, rather than cast away.
  type Pinnable = Parameters<JevClient["systemOne"]>[0] & { model: string };
  return {
    async systemOne(request) {
      widest.chars = Math.max(widest.chars, request.state.length);
      const withModel: Pinnable = { ...request, model: PINNED_MODEL };
      return client.systemOne(withModel);
    },
  };
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const dryRun = argv.includes("--dry-run");
  const outIdx = argv.indexOf("--out");
  const out = outIdx >= 0 ? argv[outIdx + 1] : DEFAULT_OUT;
  const repIdx = argv.indexOf("--repeats");
  const repeats = repIdx >= 0 ? Number(argv[repIdx + 1]) : 5;

  if (!Number.isInteger(repeats) || repeats < 1) {
    throw new Error(`--repeats must be a positive integer; got ${String(repeats)}`);
  }
  for (const a of argv) {
    if (/^--(key|api-key|typesafe-api-key)/.test(a)) {
      throw new Error("Refusing a key on the command line. Put it in the environment.");
    }
  }

  const prereg = sha256(PREREG);
  if (prereg !== PREREG_SHA256) {
    // Not fatal here — the scorer is where this is a gate — but say it loudly.
    console.error(
      `WARNING: prereg sha256 is ${prereg}, frozen at ${PREREG_SHA256}. ` +
        `The scorer will refuse this run.`,
    );
  }

  const all = JSON.parse(readFileSync(ADJUDICATION, "utf8")) as AdjudicationItem[];
  const items = all
    .filter((x) => x.kind === "only_touches")
    .sort((a, b) => a.id - b.id);
  if (items.length !== 25) {
    throw new Error(`expected 25 only_touches items, found ${items.length}`);
  }

  const client = dryRun ? pinned(stubClient()) : await realClient();
  const calls: Call[] = [];

  for (let repeat = 0; repeat < repeats; repeat++) {
    for (const item of items) {
      const t0 = performance.now();
      let report;
      try {
        report = await triageSentence(
          item.claim,
          item.paths,
          RECORD_ONLY_THRESHOLDS,
          client,
          noul,
          40,
          // The real file count, not the length of a list this corpus capped at
          // 25. Four of these items are capped; one is a 175-file pull request.
          item.n_files,
        );
      } catch (err) {
        calls.push({
          id: item.id,
          repeat,
          noul: null,
          ms: Math.round(performance.now() - t0),
          model: null,
          input_tokens: null,
          output_tokens: null,
          error: err instanceof Error ? err.message : String(err),
        });
        continue;
      }
      calls.push({
        id: item.id,
        repeat,
        noul: report.noul,
        ms: Math.round(performance.now() - t0),
        model: report.provenance?.model ?? null,
        input_tokens: report.provenance?.usage.input_tokens ?? null,
        output_tokens: report.provenance?.usage.output_tokens ?? null,
        // `report.why` carries the reason when noul is null; keep it, it is the
        // difference between "the model refused" and "the network did".
        error: report.noul === null ? report.why : null,
      });
      // B1. Outside triageSentence's catch, on purpose: a throw inside the client
      // becomes UNDECIDED, and a version change would have been filed as an
      // outage. A dry run is exempt because the stub answers as itself and the
      // scorer refuses dry runs anyway; the control below proves the check bites.
      const seen = report.provenance?.model ?? null;
      if (!dryRun && modelMismatch(seen)) {
        process.stderr.write("\n");
        throw new Error(
          `Jev answered as ${String(seen)}, not ${PINNED_MODEL}. Stopping after ` +
            `${calls.length} call(s) with nothing written. CALIB-1 is a calibration OF ` +
            `${PINNED_MODEL}; answers from another version are not a noisier version of ` +
            `the same measurement, they are a different one. See Amendment B1.`,
        );
      }
      // B2. The bound is sound rather than estimated: a token is at least one
      // character, so a state of N characters is at most N tokens.
      if (widest.chars > STATE_CHAR_BUDGET) {
        process.stderr.write("\n");
        throw new Error(
          `A state of ${widest.chars} characters exceeds the ${STATE_CHAR_BUDGET}-character ` +
            `budget standing in for Jev's 32k state-plus-question window. Stopping. ` +
            `See Amendment B2.`,
        );
      }
      process.stderr.write(
        `\r${dryRun ? "dry-run " : ""}${calls.length}/${items.length * repeats}`,
      );
    }
  }
  process.stderr.write("\n");

  if (dryRun) {
    // The control for B1. The dry run cannot exercise the mismatch abort without
    // writing a false model into its own receipt, so it exercises the predicate
    // instead, against the value the stub actually reports.
    if (!modelMismatch("dry-run-stub") || modelMismatch(PINNED_MODEL)) {
      throw new Error("the model-pin check does not discriminate; B1 is decorative");
    }
    process.stderr.write(
      `dry-run: model-pin check discriminates (rejects "dry-run-stub", accepts "${PINNED_MODEL}")\n`,
    );
  }
  process.stderr.write(`widest state: ${widest.chars} of ${STATE_CHAR_BUDGET} characters\n`);

  const payload = {
    prereg: "PREREG_calib1_jev_2026_09_18.md",
    prereg_sha256: prereg,
    prereg_sha256_expected: PREREG_SHA256,
    prereg_sha256_at_freeze: PREREG_SHA256_AT_FREEZE,
    adjudication: "decide1_adjudication.json",
    adjudication_sha256: sha256(ADJUDICATION),
    triage_module_sha256: sha256(
      join(HERE, "..", "..", "packages", "styxx-js", "src", "triage.ts"),
    ),
    dry_run: dryRun,
    model_pinned: PINNED_MODEL,
    models_seen: [...new Set(calls.map((c) => c.model).filter((m): m is string => m !== null))]
      .sort(),
    widest_state_chars: widest.chars,
    state_char_budget: STATE_CHAR_BUDGET,
    node: process.version,
    repeats,
    asked_at: new Date().toISOString(),
    question: null as string | null,
    items: items.map((x) => ({
      id: x.id,
      url: x.url,
      claim: x.claim,
      n_files: x.n_files,
      paths_shown: x.paths.length,
      paths_truncated: x.paths_truncated,
      decidable: x.decidable,
      reason_code: x.reason_code,
    })),
    calls,
  };

  // The question verbatim, so a reader can see what was asked without running
  // anything. Taken from the module rather than retyped.
  const probe = await triageSentence(
    "probe",
    [],
    RECORD_ONLY_THRESHOLDS,
    stubClient(),
    noul,
  );
  payload.question = probe.question;

  writeFileSync(out, JSON.stringify(payload, null, 2) + "\n", "utf8");
  const ok = calls.filter((c) => c.noul !== null).length;
  console.log(
    `wrote ${out}\n  ${calls.length} calls, ${ok} with a usable noul, ` +
      `${calls.length - ok} without${dryRun ? "  [DRY RUN — not evidence]" : ""}`,
  );
}

await main();
