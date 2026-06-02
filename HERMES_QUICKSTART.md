# Mount styxx as the integrity layer for Hermes Agent

styxx is the cognitive-vitals monitor for LLM agents. Hermes Agent (Nous Research)
is model-agnostic, runs reduced-refusal models, exposes a 40+ tool surface (a large
prompt-injection surface), and carries memory across sessions — which is exactly the
population that needs a monitor most. Both speak MCP, so mounting styxx is config,
not code.

## 1. Install

```bash
pip install "styxx[mcp]"
```

This installs the `styxx-mcp` console script — an MCP server exposing styxx's
cognitive-vitals tools.

## 2. Register the MCP server in Hermes

Hermes mounts MCP servers (MCP server-management CLI since v0.4.0). Add styxx to
your Hermes MCP config the same way you add any stdio MCP server:

```json
{
  "mcpServers": {
    "styxx": { "command": "styxx-mcp", "args": [] }
  }
}
```

(The exact config key may vary by Hermes version — use your version's "add MCP
server" command; the server is the `styxx-mcp` stdio executable.)

## 3. What your agent gains

Once mounted, the Hermes agent can call styxx tools mid-loop — to read its own
state before it acts, not after:

- `cogn_audit` / `cogn_audit_with_advice` — cognitive vitals on a response
- `weather_report` — a quick honesty/confab read
- `cogn_recover_posture` — restore cognitive-integrity state across a memory/
  compaction boundary (matters for Hermes' persistent memory)
- `cogn_multiturn_audit`, `cogn_red_team`, `cogn_self_heal_protocol`,
  `observe_response`, `verify_response`, `classify_trajectory`, ...

## 4. Cost-routing (optional, in-process)

To run a cheap local model by default and escalate only the calls a behavioral
honesty signal flags as low-validity:

```python
from styxx import EpistemicSpeculativeRouter, calibrate_threshold

# drafter/verifier are callables (prompt, temperature, n) -> list[styxx.Draft];
# wrap your local Hermes backend and a stronger escalation model.
router = EpistemicSpeculativeRouter(drafter=local, verifier=frontier, tau=tau_star)
result = router.run(prompt)           # result.escalated tells you which model answered
```

Calibrate `tau` on held-out data with `calibrate_threshold` — do not guess it.

## Honest scope

- The **vitals tools** (`cogn_*`) are calibrated cognometric instruments with
  published construct ceilings — an instrument panel, not a fortune teller.
- The **router** is validated held-out on small open models and narrow tasks
  (arithmetic, sorting), not yet at frontier scale or across arbitrary tasks.
- Behavioral signals catch *uncertainty* errors; they are blind to confident
  *shared-belief* errors (use external grounding / `styxx.retrieval_check` there).

styxx fails open: if it can't read vitals, your agent's normal behavior is
unchanged. Nothing crosses unseen — but nothing breaks, either.
