# The implementer registry

Implementations of the styxx specification, listed by submission rather than by curation. Anyone may
submit; see [`SUBMIT.md`](SUBMIT.md).

## What is here today

| | |
|---|---|
| reference implementations | **2** — `styxx` (Python, published) and `@fathom_lab/styxx` (TypeScript, 0.1.0, in development) |
| third-party implementations | **0** |
| adopters listed | **0** |
| implementations carrying the certified mark | **0**, including our own |

That last row is the one that matters and it is not a formality.

## The certification gate does not exist yet

The governance policy says an implementation carries the mark when it passes the conformance suite
for a named spec version, and that the mark is revocable on conformance regression. **There is no
conformance suite for spec v1.0.0.** `conformance/` in this repository holds the sworn-measurement
vector set, which is a different thing and does not test an implementation against the spec.

So, concretely, until that harness ships:

- a submission can be **listed**, and nothing more;
- no `registry_token` in any manifest means anything, and none will be issued;
- nobody may describe an implementation as certified, **and that includes Fathom Lab**.

This is written down here rather than left implied because the alternative — a registry that quietly
accepts certification claims it cannot check — is precisely the failure this project exists to
measure in other people's tools.

## Reference implementations

| name | language | status | source |
|---|---|---|---|
| `styxx` | Python 3.10+ | published — `pip install styxx` | this repository |
| `@fathom_lab/styxx` | Node 20+ | 0.1.0, in development, unpublished | [`packages/styxx-js`](../packages/styxx-js) |

They are the canonical interpretation in one direction only: when the spec and a reference
implementation disagree, **the spec wins and the implementation is patched.**

## Third-party implementations

None yet. `manifests/` is empty for that reason; the first entry will be a real one.

## Adopters

None listed. An adopter listing is a claim that someone runs this in production, so it goes in only
when they say so themselves, in their own pull request.

---

*Nothing crosses unseen — including the part where our own implementation is not certified either.*
