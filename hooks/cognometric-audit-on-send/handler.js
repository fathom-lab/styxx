/**
 * cognometric-audit-on-send (skeleton)
 *
 * Hook for clawdbot's planned `message:outbound:pre-send` event.
 *
 * Status: SKELETON. The event does not yet exist in clawdbot core. This
 * file is committed against the planned event contract documented in
 * HOOK.md. It is a safe no-op until clawdbot core starts emitting
 * `message:outbound:pre-send`.
 *
 * Wiring: shells out to a Python bridge that wraps
 * `styxx.cogn_audit_on_send`. All cognometric decision logic
 * (cleanest-of-trajectory selection, climbing-composite degradation
 * guard, persistence to chart.jsonl) lives inside the styxx primitive.
 * The handler's only job is plumbing.
 */
import { spawnSync } from "node:child_process";

const BRIDGE_TIMEOUT_MS = 10_000;

const cognometricAuditOnSend = {
  name: "cognometric-audit-on-send",
  events: ["message:outbound:pre-send"],

  /**
   * @param {{
   *   prompt: string,
   *   draft: string,
   *   channel?: string,
   *   sessionKey?: string,
   *   msg_id?: string,
   * }} event
   * @returns {Promise<undefined | {draft: string} | {abort: true, reason?: string}>}
   */
  handler: async (event) => {
    // Fail-safe: if anything is missing, do nothing and let the original
    // draft ship. This handler must NEVER block the send path.
    if (
      !event ||
      typeof event.prompt !== "string" ||
      typeof event.draft !== "string"
    ) {
      return undefined;
    }

    const payload = JSON.stringify({
      prompt: event.prompt,
      draft: event.draft,
      channel: event.channel ?? null,
      sessionKey: event.sessionKey ?? null,
      msg_id: event.msg_id ?? null,
    });

    let res;
    try {
      res = spawnSync(
        "python",
        ["-m", "clawdbot_bridge.audit_on_send"],
        {
          input: payload,
          encoding: "utf8",
          timeout: BRIDGE_TIMEOUT_MS,
          maxBuffer: 1024 * 1024,
        },
      );
    } catch (err) {
      // subprocess spawn failed entirely — fall through, ship draft as-is
      console.error(
        "[cognometric-audit-on-send] bridge spawn failed:",
        err && err.message ? err.message : String(err),
      );
      return undefined;
    }

    if (res.error) {
      console.error(
        "[cognometric-audit-on-send] bridge error:",
        res.error.message,
      );
      return undefined;
    }
    if (res.status !== 0) {
      console.error(
        "[cognometric-audit-on-send] bridge exited",
        res.status,
        "stderr:",
        (res.stderr || "").slice(0, 500),
      );
      return undefined;
    }

    let out;
    try {
      out = JSON.parse(res.stdout);
    } catch (err) {
      console.error(
        "[cognometric-audit-on-send] bridge produced invalid JSON:",
        (res.stdout || "").slice(0, 200),
      );
      return undefined;
    }

    if (out && typeof out.shipped_draft === "string") {
      // Identity short-circuit: no need to round-trip an unchanged string
      if (out.shipped_draft === event.draft) {
        return undefined;
      }
      return { draft: out.shipped_draft };
    }

    return undefined;
  },
};

export default cognometricAuditOnSend;
