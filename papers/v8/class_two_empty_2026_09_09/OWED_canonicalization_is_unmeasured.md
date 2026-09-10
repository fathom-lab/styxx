# OWED — the canonicalization layer is not in the mutation measurement, and here is what to propose

Fathom Lab · 2026-09-09 · **A statement of work, not a result.** Nothing here was measured. It
records a real gap, its actual cause, an attempted fix that the codebase correctly refused, and the
five mutations somebody should measure once the mechanism allows it. Not sworn.

## The gap

`conformance/v8/mutation_coverage.json` is this project's answer to its own rule that an agreement
number without detection power is not a number. It mutates the implementation and reports which
mutations the committed vector set would catch. Re-measured against the tree today, the committed
receipt reproduces exactly, so the instrument is honest about what it did.

What it did not do is measure `styxx/v8/jcs.py`. That module produces the bytes behind every
certificate id, every signature preimage and every Merkle leaf in the design. A silent defect there
changes every artifact this system has ever produced, and the published detection rate says nothing
about it. `styxx/v8/fingerprint.py`, which builds the objects being certified, is also absent.

## The cause, which is not neglect

The first version of `tests/test_v8_mutation_surface.py` framed this as an oversight. That was
wrong and the reason string has been corrected.

The coverage tool applies a mutation by rebinding, in every loaded module, each attribute holding
the same object under the same name. An import that **renames** what it binds does not follow the
rebind, and jcs is imported that way, for example as `from styxx.v8.jcs import digest as
_jcs_digest`. A mutation of jcs would therefore measure the patcher rather than the vector set. The
tool states this in its module docstring and lists jcs under `not_mutable` deliberately.

## The fix that was tried, and correctly refused

Five mutations were added to the catalogue anyway, on the theory that a recorded-but-unmeasurable
defect tells a reader more than a bare exclusion note. The catalogue's own contract refused them:
an anchor must name a module in `MUTABLE`, and four tests failed by design. The entries were
reverted.

That refusal is right, and it is worth naming as a small instance of the day's larger lesson. A gap
was assumed to be neglect, turned out to be a documented decision, and the workaround was blocked by
a contract written before anyone thought of it.

**The real repair is a source-level patch replayed in a subprocess**, so a mutated `jcs.py` is
imported fresh rather than rebound into an already-loaded process. That is a change to a tool whose
output is a committed receipt, and it should be done deliberately, with the receipt regenerated and
the miss list compared before and after.

## The five to measure

Each is a defect a second implementation could plausibly ship rather than a random edit. Each was
checked to anchor on exactly one place in `jcs.py` and to leave the module parseable.

| name | change | why it matters |
|---|---|---|
| `jcs-key-order-utf16-to-codepoint` | the ordering guard encodes keys as UTF-8 instead of UTF-16-BE | RFC 8785 §3.2.3 orders keys by UTF-16 code unit, and the two orders disagree above U+FFFF. This module cannot sort that way on the fallback backend, so it refuses objects whose orders differ; the mutation makes the guard compare code-point order with itself, so it never refuses and such an object serializes wrong. This is the divergence a second implementation is most likely to have, and this lab measured 531,162 objects without finding one that would trip it. |
| `jcs-duplicate-key-accepted` | the duplicate-key refusal becomes `pass` | two keys normalizing to one string collapse silently, so an object has two canonical forms depending on iteration order and a content address stops being a function of the content |
| `jcs-nan-and-infinity-accepted` | the non-finite refusal returns `0.0` | a distance or floor that overflowed is recorded as a perfect score, which is the worst direction for this failure to take |
| `jcs-unsafe-integer-accepted` | the 2**53 bound refusal becomes `pass` | an integer beyond the exactly-representable double range reads back differently in a language whose only number type is a double, which is precisely the cross-language divergence the bound exists to prevent |
| `jcs-lone-surrogate-accepted` | the lone-surrogate refusal returns | an object containing one has no canonical byte sequence at all, so the failure is deferred to whatever encodes next, which may substitute a replacement character and digest bytes nobody chose |

## The expectation, recorded before the measurement

All five are expected **missed**, for one reason: every string, number and key in the committed
vector set comes from a certificate, and certificates do not carry keys above U+FFFF, duplicate
keys, non-finite numbers, integers near 2**53, or lone surrogates. If that expectation holds, the
finding is that jcs's guards are exactly the part of the design the conformance set cannot see, and
the guards are what a second implementation would have to get right.

Recording the expectation first is the point. A prediction written after the numbers is not a
prediction.
