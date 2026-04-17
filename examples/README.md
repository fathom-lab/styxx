# styxx examples

Start here. Every example is runnable from the repo root.

## Start here

- **[`quickstart.py`](quickstart.py)** — the hello-world. One-line drop-in for
  `openai.OpenAI`, plus a vitals card on every response. Falls back to a
  bundled trajectory demo if `OPENAI_API_KEY` is not set.

  ```bash
  python examples/quickstart.py
  ```

## Advanced

Deeper features, each focused on one capability. Run any of them with
`python examples/advanced/<file>.py`.

- **`basic.py`** — three-in-one tour: `Raw()` adapter, `OpenAI()` adapter, and
  `vitals.as_dict()` for agent-side routing.
- **`openai_live.py`** — end-to-end live OpenAI call with full vitals inspection.
- **`reflex_demo.py`** — reflex loop: detect degraded generations and retry.
- **`reflex_visual_demo.py`** — same as above, with a live terminal UI.
- **`watch_demo.py`** — long-running `watch()` mode that streams vitals over time.
- **`gates_demo.py`** — policy gates that abort generation on failed vitals.
- **`thought_demo.py`** — chain-of-thought instrumentation and phase forecasts.
- **`crewai_self_correcting.py`** — CrewAI integration for self-correcting agents.
- **`styxx_demo.ipynb`** — Jupyter walkthrough of the full API.
