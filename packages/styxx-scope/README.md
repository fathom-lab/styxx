# styxx scope — browser extension for AI chatbot transparency

**Cognitive vitals next to every response from ChatGPT, Claude, Gemini, and Grok.**

`styxx scope` is a Chrome/Firefox extension that watches the pages of the
major chatbot products and, next to every assistant response, renders a
small cognometric readout showing:

- **category** · reasoning / retrieval / confab / refusal / sycophant / …
- **K / C / D** · the three axes of the Cognometric Fingerprint
  Specification v1.0
- **trust** · aggregate, Tier-3 proxy-pipeline penalty-adjusted
- **faults** · confabulation, drift, refusal, sycophant, low-trust flags

## Install (unpacked, while extension is under review)

1. Download this folder (`packages/styxx-scope/`) or clone the styxx repo.
2. Chrome/Edge: open `chrome://extensions`, toggle **Developer mode** on,
   click **Load unpacked**, select the `styxx-scope` folder.
3. Firefox: open `about:debugging#/runtime/this-firefox`, click
   **Load Temporary Add-on**, select `manifest.json` inside this folder.
4. Open chat.openai.com / claude.ai / gemini.google.com — badges appear
   next to assistant responses automatically.

## Supported sites

- `chat.openai.com`, `chatgpt.com`
- `claude.ai`
- `gemini.google.com`, `bard.google.com`
- `grok.com`, `x.com/i/grok*`

## What it does NOT do

- **Does not send your chat to any server.** All classification runs
  locally in the browser, powered by `classifier.js` (~300 lines of
  pure-JS port of the styxx text-heuristic pipeline).
- **Does not store chat content.** The extension has `storage` permission
  only for user preferences.
- **Does not replace the full styxx Profiler.** Tier 3 text-only readings
  are substantially less accurate than the logprob-based Tier 1/2
  pipelines. Treat the badge as a hint, not a verdict.

## Accuracy caveat

The browser extension runs the Tier-3 proxy-signal pipeline (Spec v1.0
§7.3) — text-only features, no logprobs. The aggregate trust value
includes a documented `confidence_penalty` of 0.25.

For production use, run the full `styxx.profile` Python pipeline via
`pip install -U styxx`.

## License

- Code: MIT.
- Methodology documented in the Cognometric Fingerprint Specification
  v1.0 (CC-BY-4.0, DOI [10.5281/zenodo.19746215](https://doi.org/10.5281/zenodo.19746215)).
- Measurement architecture protected by US Provisionals 64/020,489 ·
  64/021,113 · 64/026,964.

## Credits

Built by [Fathom Lab](https://fathom.darkflobi.com).
Spec: <https://fathom.darkflobi.com/spec>.
Reference Python implementation: <https://pypi.org/project/styxx/>.

*Nothing crosses unseen.*
