# SPEC — the Action discovers documents with the lexer's own test (v0.1)

**Frozen 2026-09-06, before the code.** One rule, C1. Found by the eight-dimension adversarial
audit (`wf_9466dcba-f49`, dimension `output`). Last of the eight.

## The defect

The Action decides which documents to verify with a **case-sensitive byte test**:

    if isinstance(body, str) and "<sworn" in body:          # sworn/sworn_action.py:460
    elif b"<sworn" not in data:                             # :478
        skipped.append({"path": path, "why": "carries no <sworn tag"})

The verifier's candidate lexer is **case-insensitive by construction**:

    _CANDIDATE = re.compile(rb"<(/?)[sS][wW][oO][rR][nN](?![A-Za-z0-9_\-])")

and `DECISIONS["tag_grammar"]` says what a non-lowercase candidate is: *"exactly `<sworn r="…"
k="…">` with single spaces, double quotes, lowercase, and `</sworn>`; **any other tag-shaped
candidate … is MALFORMED, never narrative**"*.

So the two disagree about what a document even *is*. Measured at `65c98012`:

```
document: <SWORN r="r1" k="numeric">the rate was 0.42 on the panel</SWORN>

the Action's test  b'<sworn' in data : False      -> skipped
the lexer's test   _CANDIDATE.search : True
the verifier's verdict               : SWORN-FAILED
                                       [('MALFORMED','tag_syntax'), ('MALFORMED','tag_syntax')]
```

**CI reports "carries no `<sworn` tag" for a document the verifier calls SWORN-FAILED.** The
skip is silent by design — a skipped document is not a failure — so an uppercase tag is a way to
put a tag-shaped candidate in a pull request and have the gate say nothing about it.

This is not the same as a false HELD: the Action never claims the document held, it claims the
document has no tags. But the claim is false, and it is the claim a reviewer reads.

### It is wrong in the other direction too, found by the guard before the code

Writing C-G3 turned up a second half the audit had not named. `b"<sworn" in data` is a **substring**
test, so it matches `<swornish>` — while the lexer's candidate pattern ends in a negative lookahead,
`(?![A-Za-z0-9_\-])`, and correctly does not. The Action would therefore *verify* a document whose
only tag-shaped text is a longer name, where the verifier sees no candidate at all.

That direction is harmless in outcome — such a document simply comes back UNSWORN with zero spans —
but it is the same defect: a second spelling of "what a tag looks like" that agrees with the first
only by accident. C1 closes both directions at once, because it stops spelling it twice.

## C1 — the Action asks the lexer

Both discovery tests use `styxx.sworn._CANDIDATE`, imported rather than restated:

    if isinstance(body, str) and _CANDIDATE.search(body.encode("utf-8")):
    elif _CANDIDATE.search(data) is None:

The Action already imports `_headline` and `_write_json_lf` from `styxx.sworn`, so reaching for a
private name here is the file's existing practice rather than a new liberty. **Importing rather than
copying is the point**: a second spelling of "what a tag looks like" is exactly the drift that the
`U+0085` path-segment defect was, one repair earlier in this same audit.

### The skip messages do not change

`"carries no <sworn tag"` and `"the body carries no <sworn tag"` stay byte-identical. They appear in
the committed Action samples, which `sworn_action_sample.py --check` compares **byte for byte**, and
a sample is history: changing the wording would require a new prefix at a new commit under that
script's own rule. The messages also remain accurate, because after C1 they are only ever emitted
for a document that carries no candidate in any case.

## What moves

- **Nothing committed.** No committed `.md` carries a non-lowercase tag-shaped candidate, and the
  Action samples' documents are all lowercase, so `sworn_action_sample.py --check` must still
  reproduce. That is a guard here, not an assumption.
- No verifier code changes. `styxx/sworn.py` is untouched, so the conformance set does not move —
  not even by the build pin.
- The Action will now verify (and fail) a document it used to skip. That is the repair: the gate
  stops being evadable by holding down shift.

## Guards, watched to fail before the code

| # | guard | before | after |
|---|---|---|---|
| C-G1 | a changed `.md` whose only tag is `<SWORN …>` is verified, not skipped | red: skipped, "carries no <sworn tag" | green: verified, SWORN-FAILED |
| C-G2 | the same for a pull-request body | red | green |
| C-G3 | a `.md` with genuinely no candidate in any case is still skipped, with the message unchanged | green throughout |
| C-G4 | the Action's discovery test and the lexer's agree on a spread of tag spellings | red | green |
| C-G5 | `sworn_action_sample.py --check` still reproduces the committed sample | green throughout — the sample is history |

C-G1 is the guard that must be seen red. C-G5 is the one that must never go red.

## What this does not claim

That the Action's discovery is otherwise right. It looks only at changed `.md` files and the pull
request body; a tag in a `.txt`, in a commit message, or in an unchanged file is out of scope and
stays out of scope. This closes the case where the Action and the verifier disagree about the same
bytes.
