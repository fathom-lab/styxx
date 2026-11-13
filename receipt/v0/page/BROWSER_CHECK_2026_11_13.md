# The committed page, opened in a real browser

Status: a record of one check by the lab's lead on one machine. It is the lab watching its own page. It is not a test anyone can rerun from this repository, and no test reads this file.

## What was opened

- File: `receipt_check_v0.html` of this folder, sha256 `577bfeb10865dce7afb9c261683b0cab967c7c1bb6ca52e3f6b996023018b507`, 163,704 bytes.
- Served by `python -m http.server` bound to 127.0.0.1, from this folder. A download of the served file had the same sha256.
- Browser: Chrome 156.0.8078.75 inside the Claude desktop app's browser pane, on Windows.
- When: 2026-11-13, between 22:07 and 22:10 UTC.

## What was seen

| What was done | What the page did |
|---|---|
| Page loaded | The page computed the sha256 of its embedded checker, `cbdae74e95cbfb0280d38b0d38380dcbf2f32caf12d12c860c483aa2ab9e70a3`, showed the same value as recorded at build, and said the checker may run. |
| Example 1, CHECK | SWORN-HELD; 4 spans, 4 HELD. The page said the answer is the same as the command line checker's recorded answer. |
| Example 2, CHECK | SWORN-FAILED; 2 spans, 1 HELD and 1 FAILED. Same as the recorded answer. |
| Example 3, CHECK | SWORN-UNCHECKED; 1 span, UNRESOLVED. Same as the recorded answer. |
| Example 1, RECHECK | REDERIVES. Same as the recorded answer. |
| The same receipt with its verdict word edited, RECHECK | REFUSED `receipt_digest`. |
| A document whose claim, name and manifest hold `img`, `script`, `iframe` and `b` markup | The markup was drawn as text. A flag the markup would have set stayed unset. Under the result there were only `button`, `code`, `div`, `p`, `pre` and `span` elements, and the page held no `img`, `iframe`, `object`, `embed`, `link`, `form` or `a` element. |
| A document of 1,000 spans | SWORN-FAILED, 1,000 MALFORMED. The page drew 200 and said 800 more are not drawn. The run button stayed usable. |
| Width of 375 pixels | No sideways scroll; no element past the right edge. |

Each run said it ran in a fresh worker made from the embedded checker. The longest took 103 ms against the limit of 10,000 ms.

## Nine probes of the policy

Each was run from the browser's console against this page. The addresses were on the same 127.0.0.1 server, so nothing could leave the machine either way.

| Probe | Result |
|---|---|
| `fetch` of a file on the same server | refused; the browser reported `connect-src` |
| an image from the same server | refused; `img-src` |
| `navigator.sendBeacon` | refused; `connect-src` |
| a WebSocket | refused; `connect-src` |
| `innerHTML` with a plain string | refused; Trusted Types |
| an inline script element | refused; Trusted Types |
| `eval` of a string | refused; the policy has no `unsafe-eval` |
| a worker from a `data:` address | refused; Trusted Types |
| a second Trusted Types policy under another name | refused; `trusted-types` |

Every report was in enforce mode. The server's log for the whole check holds only requests for the folder listing, for the page (one by the browser, one by the download named above) and for a favicon the browser asked for by itself (answered 404). It holds no request from a probe and none from a checker run.

## What was not done

- The page was not opened from disk (`file://`). The browser pane shows a file from disk as a still picture, so it could not run one.
- No file was chosen with the file buttons; inputs were typed or loaded by the example buttons.
- No other browser, no phone, no other operating system.
- Nothing was checked from inside the worker: whether the worker inherits the page's policy is still unknown.
- The time limit was not reached by any input, so the page's behaviour at the limit was not seen.
- The inputs were written into the text boxes by script with an input event, which is close to a paste and is not one.
