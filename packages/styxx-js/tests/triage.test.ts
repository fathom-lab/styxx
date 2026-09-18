/**
 * Triage tests — including the ones whose job is to stop this module growing.
 *
 * Everything here runs with a fake client. No API key, no network, no cost. That
 * is deliberate: a test suite that needs a credential is a test suite that gets
 * skipped in CI and then stops being true.
 *
 * The last block is the important one. Jev decides what gets *read*; the
 * deterministic reader decides what gets *said*. Those tests fail if anyone ever
 * teaches this module to emit a verdict, which is exactly the change that would
 * make a styxx answer unreproducible without anybody noticing in review.
 */

import { describe, it, expect, vi } from "vitest";
import {
  triageSentence,
  type JevClient,
  type NoulFactory,
  type TriageThresholds,
  type TriageVerdict,
} from "../src/triage";

/** Preregister your own; these exist to exercise the code. See CALIB-1. */
const T: TriageThresholds = { read: 0.8, skip: 0.2 };

const noul: NoulFactory = (instructions, criteria) => ({ type: "noul", instructions, criteria: criteria ?? null });

/** A client that answers with a fixed probability and records what it was asked. */
function fake(answer: number | undefined | unknown) {
  const seen: { state?: string; questions?: Record<string, unknown> } = {};
  const client: JevClient = {
    async systemOne(req) {
      seen.state = req.state;
      seen.questions = req.questions;
      return {
        model: "jev-1",
        answers: { file_scope_claim: { type: "noul", noul: answer as number } },
        usage: { input_tokens: 210, output_tokens: 1 },
      };
    },
  };
  return { client, seen };
}

const PATHS = ["github/workflows/dependabot.yml"];

describe("triage — the three answers", () => {
  it("reads a sentence Jev is confident is a file-scope claim", async () => {
    const { client } = fake(0.94);
    const r = await triageSentence("Only modifies CHANGELOG.md", PATHS, T, client, noul);
    expect(r.verdict).toBe("READ");
    expect(r.noul).toBe(0.94);
    expect(r.decisiveness).toBeCloseTo(0.88, 6);
    expect(r.why).toContain("0.940");
  });

  it("skips a sentence Jev is confident is not one", async () => {
    const { client } = fake(0.04);
    // The Memoria case: runtime behaviour, which the gate accused three times.
    const r = await triageSentence(
      "Only changed mods/submods are serialized, all others preserved bytewise",
      PATHS,
      T,
      client,
      noul,
    );
    expect(r.verdict).toBe("SKIP");
    expect(r.decisiveness).toBeCloseTo(0.92, 6);
  });

  it("refuses inside the band rather than guessing", async () => {
    for (const p of [0.21, 0.5, 0.79]) {
      const { client } = fake(p);
      const r = await triageSentence("only touches the footer", PATHS, T, client, noul);
      expect(r.verdict).toBe("UNDECIDED");
      expect(r.why).toContain("refusal band");
    }
  });

  it("puts the boundaries inside the decided regions, not the band", async () => {
    const at = async (p: number) =>
      (await triageSentence("s", PATHS, T, fake(p).client, noul)).verdict;
    expect(await at(0.8)).toBe("READ");
    expect(await at(0.2)).toBe("SKIP");
  });
});

describe("triage — failure lowers nothing", () => {
  it("is UNDECIDED when Jev cannot be reached", async () => {
    const client: JevClient = {
      async systemOne() {
        throw new Error("ECONNREFUSED");
      },
    };
    const r = await triageSentence("Only modifies CHANGELOG.md", PATHS, T, client, noul);
    expect(r.verdict).toBe("UNDECIDED");
    expect(r.noul).toBeNull();
    expect(r.why).toContain("jev unreachable");
  });

  it("is UNDECIDED on a missing, non-numeric or out-of-range answer", async () => {
    for (const bad of [undefined, null, "0.9", NaN, Infinity, -0.1, 1.1]) {
      const { client } = fake(bad);
      const r = await triageSentence("s", PATHS, T, client, noul);
      expect(r.verdict).toBe("UNDECIDED");
      expect(r.noul).toBeNull();
    }
  });

  it("never returns READ on any failure path", async () => {
    const failures: JevClient[] = [
      { async systemOne() { throw new Error("401 unauthorized"); } },
      { async systemOne() { throw new Error("429 rate limited"); } },
      { async systemOne() { return { model: "jev-1", answers: {}, usage: { input_tokens: 1, output_tokens: 1 } }; } },
      { async systemOne() { return { model: "jev-1", answers: { other: { type: "noul", noul: 0.99 } }, usage: { input_tokens: 1, output_tokens: 1 } }; } },
    ];
    for (const client of failures) {
      const r = await triageSentence("s", PATHS, T, client, noul);
      expect(r.verdict).not.toBe("READ");
    }
  });

  it("rejects thresholds that do not leave a refusal band", async () => {
    const bad = [
      { read: 0.5, skip: 0.5 },
      { read: 0.2, skip: 0.8 },
      { read: 1.1, skip: 0.2 },
      { read: 0.8, skip: -0.1 },
      { read: NaN, skip: 0.2 },
    ];
    for (const t of bad) {
      await expect(triageSentence("s", PATHS, t, fake(0.9).client, noul)).rejects.toThrow(RangeError);
    }
  });
});

describe("triage — what the model is shown", () => {
  it("shows the sentence and this diff's paths, so it judges against the diff", async () => {
    const { client, seen } = fake(0.9);
    await triageSentence(
      "Only change .githiub/workflows/dependabot.yml",
      ["github/workflows/dependabot.yml"],
      T,
      client,
      noul,
    );
    expect(seen.state).toContain("Only change .githiub/workflows/dependabot.yml");
    expect(seen.state).toContain("- github/workflows/dependabot.yml");
    expect(seen.state).toContain("1 total");
  });

  it("truncates a large diff and says how many it withheld", async () => {
    const many = Array.from({ length: 645 }, (_, i) => `src/file${i}.ts`);
    const { client, seen } = fake(0.9);
    await triageSentence("s", many, T, client, noul, 40);
    expect(seen.state).toContain("645 total, 605 not shown");
    expect(seen.state).toContain("- src/file39.ts");
    expect(seen.state).not.toContain("- src/file40.ts");
  });

  it("reports the real file count when the caller's path list is already capped", async () => {
    // decide1_adjudication.json item 45: 175 files, `paths` recorded as 25.
    const capped = Array.from({ length: 25 }, (_, i) => `src/file${i}.ts`);
    const { client, seen } = fake(0.9);
    await triageSentence("s", capped, T, client, noul, 40, 175);
    expect(seen.state).toContain("175 total, 150 not shown");
    expect(seen.state).toContain("- src/file24.ts");
  });

  it("refuses a total smaller than the paths it was handed", async () => {
    await expect(
      triageSentence("s", ["a.ts", "b.ts"], T, fake(0.9).client, noul, 40, 1),
    ).rejects.toThrow(RangeError);
    await expect(
      triageSentence("s", ["a.ts"], T, fake(0.9).client, noul, 40, 1.5),
    ).rejects.toThrow(RangeError);
  });

  it("says (none) rather than nothing on an empty diff", async () => {
    const { client, seen } = fake(0.9);
    await triageSentence("s", [], T, client, noul);
    expect(seen.state).toContain("- (none)");
  });

  it("asks exactly one question and reports it verbatim", async () => {
    const { client, seen } = fake(0.9);
    const r = await triageSentence("s", PATHS, T, client, noul);
    expect(Object.keys(seen.questions ?? {})).toEqual(["file_scope_claim"]);
    expect(r.question).toBe(seen.questions!.file_scope_claim.instructions);
  });

  it("sends a noul question whose criteria are an object, not a string", async () => {
    // The published prose documents `criteria` as a string. The SDK's own types
    // say it is `{ true?, false? }`. This pins the types, which are the truth.
    const { client, seen } = fake(0.9);
    await triageSentence("s", PATHS, T, client, noul);
    const q = seen.questions!.file_scope_claim;
    expect(q.type).toBe("noul");
    expect(typeof q.criteria).toBe("object");
    expect(q.criteria).toHaveProperty("true");
    expect(q.criteria).toHaveProperty("false");
  });

  it("records which model answered and what it billed", async () => {
    const { client } = fake(0.9);
    const r = await triageSentence("s", PATHS, T, client, noul);
    expect(r.provenance).toEqual({ model: "jev-1", usage: { input_tokens: 210, output_tokens: 1 } });
  });

  it("has null provenance when Jev was never reached", async () => {
    const client: JevClient = { async systemOne() { throw new Error("down"); } };
    const r = await triageSentence("s", PATHS, T, client, noul);
    expect(r.provenance).toBeNull();
  });
});

describe("triage — the boundary this module must never cross", () => {
  /**
   * If this fails, someone has taught triage to emit a verdict. That is the
   * change that makes a styxx answer depend on a hosted model, and it does not
   * happen by editing a type — it happens by preregistering it first.
   */
  it("cannot express any verdict the diff gate emits", async () => {
    const forbidden = ["VERIFIED", "CONTRADICTED", "UNCHECKABLE", "SUPPORTED"];
    const seen: TriageVerdict[] = [];
    for (const p of [0.99, 0.5, 0.01]) {
      seen.push((await triageSentence("s", PATHS, T, fake(p).client, noul)).verdict);
    }
    const client: JevClient = { async systemOne() { throw new Error("down"); } };
    seen.push((await triageSentence("s", PATHS, T, client, noul)).verdict);

    expect(new Set(seen)).toEqual(new Set(["READ", "SKIP", "UNDECIDED"]));
    for (const v of seen) expect(forbidden).not.toContain(v as string);
  });

  it("carries no credential in anything it returns", async () => {
    const KEY = "sk-typesafe-do-not-leak-me";
    vi.stubEnv("TYPESAFE_API_KEY", KEY);
    const { client } = fake(0.9);
    const r = await triageSentence("s", PATHS, T, client, noul);
    expect(JSON.stringify(r)).not.toContain(KEY);
    expect(JSON.stringify(r)).not.toContain("TYPESAFE_API_KEY");
    vi.unstubAllEnvs();
  });

  it("ships no default thresholds, because no calibration has been measured", async () => {
    const mod = await import("../src/triage");
    const exported = Object.keys(mod);
    expect(exported).not.toContain("DEFAULT_THRESHOLDS");
    expect(exported).toContain("NO_DEFAULT_THRESHOLDS");
  });
});
