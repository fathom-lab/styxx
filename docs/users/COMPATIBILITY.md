# styxx provider compatibility matrix

**Closes #3.**

styxx tier-0 cognitive vitals require the LLM provider to expose the
`top_logprobs` field on chat-completion responses (typically activated
by passing `logprobs=True` and `top_logprobs=5` on the create call).
Without that field, `observe()` falls back to the text classifier
(lower accuracy) or returns `None`.

This matrix lists which providers and models actually expose the
required surface, and how to invoke them. **Verified entries only**
— anything marked "not yet verified" is a TODO for a contributor with
access to that platform.

If you verify a row, please open a PR updating this file and
referencing the test you ran. See the *Definition of done* at the
bottom for the format.

---

## At a glance

| provider | tier-0 supported | how to invoke | notes |
|---|---|---|---|
| OpenAI | ✅ verified | `styxx.OpenAI()` | wrapper auto-injects `logprobs=True, top_logprobs=5` |
| OpenRouter | ✅ verified (model-dependent) | `styxx.OpenAI(base_url=...)` | passthrough; works when the underlying model supports it |
| Anthropic Claude | ❌ not supported | `styxx.Anthropic()` | Messages API has no `logprobs` parameter; passthrough wrapper warns once and falls back to text classifier |
| Google Gemini | ⚠️ not yet verified | (use openai-compatible endpoint if available) | Gemini's native API has logprobs support but the surface differs from OpenAI's; needs verification |
| Azure OpenAI | ⚠️ not yet verified | `styxx.OpenAI(base_url=AZURE_OPENAI_ENDPOINT, api_key=...)` | should be identical to OpenAI; needs end-to-end test |
| AWS Bedrock | ⚠️ not yet verified | (use openai-compatible gateway) | varies by underlying model; needs per-model verification |
| Groq | ⚠️ not yet verified | `styxx.OpenAI(base_url="https://api.groq.com/openai/v1")` | API is OpenAI-compatible but `top_logprobs` support varies by model |
| vLLM (self-hosted) | ⚠️ not yet verified | `styxx.OpenAI(base_url="http://localhost:8000/v1")` | OpenAI-compatible server; `logprobs` is supported in recent vLLM versions but verify per-version |
| llama.cpp server | ⚠️ not yet verified | `styxx.OpenAI(base_url="http://localhost:8080/v1")` | OpenAI-compatible mode supports logprobs in recent builds |
| Ollama | ⚠️ not yet verified | `styxx.OpenAI(base_url="http://localhost:11434/v1")` | OpenAI-compatible mode; `top_logprobs` field support has been intermittent across releases |
| LiteLLM gateway | ⚠️ not yet verified | `styxx.OpenAI(base_url="http://localhost:4000")` | should pass through to whatever the underlying provider supports |

---

## Verified entries — usage snippets

### OpenAI

```python
from styxx import OpenAI

client = OpenAI()
r = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "why is the sky blue?"}],
)

print(r.choices[0].message.content)   # the response text, unchanged
print(r.vitals.phase4)                  # "reasoning:0.69"
print(r.vitals.gate)                    # "pass"
```

`styxx.OpenAI()` is a drop-in replacement for `openai.OpenAI()` that
auto-injects `logprobs=True, top_logprobs=5` on every chat-completion
call. You don't need to remember the flag.

**Models tested:** `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo`.

### OpenRouter

```python
from styxx import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="<your openrouter key>",
)
r = client.chat.completions.create(
    model="anthropic/claude-3.5-sonnet",   # OR any model OpenRouter routes to
    messages=[{"role": "user", "content": "..."}],
)

# Will work if the underlying model supports top_logprobs.
# Some models (e.g. anthropic via OpenRouter) do not — same as Anthropic direct.
print(r.vitals)
```

**Caveat:** OpenRouter is a passthrough. Whether a given model
supports `top_logprobs` depends on the underlying provider. OpenAI and
OpenAI-compatible models on OpenRouter generally work. Anthropic
models on OpenRouter do not, because the upstream Anthropic API
doesn't expose logprobs.

### Anthropic Claude (NOT supported at tier 0)

```python
from styxx import Anthropic

client = Anthropic()
r = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "..."}],
)

print(r.content[0].text)   # normal Anthropic response
print(r.vitals)             # always None — see below
```

The Anthropic Messages API does not have a `logprobs` or
`top_logprobs` parameter. There is no way to obtain per-token
probabilities from the public API as of 2026-04. `styxx.Anthropic()`
exists as a passthrough wrapper that emits a one-time stderr warning
the first time you call it and otherwise behaves identically to
`anthropic.Anthropic()`.

**Workarounds for Anthropic users:**

- Route through OpenRouter to an OpenAI-compatible model and use
  `styxx.OpenAI(base_url=...)`.
- Use `styxx.Raw` with a pre-captured logprob trajectory if you have
  one from another source.
- Wait for tier 1 (D-axis honesty from residual stream — does not
  need logprobs, but requires a local open-weight model).

---

## Unverified entries — contributor TODOs

For each row in the matrix marked ⚠️ *not yet verified*, the verification
procedure is identical:

1. Set up an account / install the provider's library.
2. Make one chat-completion call with `logprobs=True, top_logprobs=5`.
3. Check whether the response object has a `.choices[0].logprobs`
   field with a non-empty `content` list, and whether each entry has
   a `top_logprobs` list of length 5.
4. If yes, wrap the call with `styxx.observe(response)` and confirm
   that `vitals.phase4` is not `None`.
5. If steps 3 and 4 pass, open a PR moving the row from ⚠️ to ✅,
   adding the model list, and adding a usage snippet to this file.

Contributors: `# TODO(contributor): verify Gemini`,
`# TODO(contributor): verify Azure OpenAI`,
`# TODO(contributor): verify AWS Bedrock`,
`# TODO(contributor): verify Groq`,
`# TODO(contributor): verify vLLM`,
`# TODO(contributor): verify llama.cpp server`,
`# TODO(contributor): verify Ollama`,
`# TODO(contributor): verify LiteLLM`.

---

## Definition of done for a verified entry

A row is "verified" when all four conditions hold:

1. The styxx maintainer or a community contributor has personally run
   the snippet and observed non-`None` `vitals.phase4`.
2. The PR adding the verification cites the model name, the styxx
   version it was tested against, and the date of the test.
3. The matrix row has the ✅ marker, the model list, and a runnable
   usage snippet in this file.
4. The PR description references this issue / file so the next
   reviewer can find it.

---

*If your provider isn't on this list and you'd like it added, open
an issue or a PR. Cognitive metrology benefits from broader provider
coverage — every new verified row makes styxx work for one more
slice of the field.*
