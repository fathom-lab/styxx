# styxx-mcp

Model Context Protocol (MCP) server that exposes [styxx](https://fathom.darkflobi.com/styxx)
cognitive-vitals tools to MCP-compatible clients (Claude Desktop, Cursor, Cline,
and any autonomous agent runtime that speaks MCP).

## Tools exposed

| Tool | Purpose |
| --- | --- |
| `observe_response` | Observe an LLM response, return `{classification, confidence, gate}`. |
| `verify_response` | Verify an LLM response, return a `VerificationResult` with trajectory features + anomalies. |
| `classify_trajectory` | Classify a raw logprob sequence into one of six cognitive classes. |
| `weather_report` | Fleet-level cognitive weather summary over the last N observations. |

Six cognitive classes: **reasoning · retrieval · refusal · creative · adversarial · hallucination**.

## Install

```bash
pip install styxx>=3.3.1 mcp
pip install -e ./mcp     # from the styxx repo root
# or, once published:
# pip install styxx-mcp
```

Verify it starts cleanly:

```bash
python -m styxx_mcp.server
```

The process reads MCP messages on stdin and writes on stdout; you can Ctrl-C to quit.

## Claude Desktop config

Add this block to `claude_desktop_config.json`
(macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`,
Windows: `%APPDATA%\Claude\claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "styxx": {
      "command": "python",
      "args": ["-m", "styxx_mcp.server"]
    }
  }
}
```

Restart Claude Desktop. The four tools above will appear in the tool picker.

## Cursor / Cline

Any MCP-aware client can point at the same command:

```
command: python
args:    ["-m", "styxx_mcp.server"]
transport: stdio
```

## Offline behaviour

The server works fully offline. If `styxx.observe(response)` cannot reach an
upstream inference API, the tools fall back to `styxx.observe_raw` and classify
directly from the logprob arrays in the request — no network calls required.

## License

MIT © Fathom Lab
