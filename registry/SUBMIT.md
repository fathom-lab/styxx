# Submitting an implementation

One pull request adding one file. No account, no form, no approval queue you cannot see.

## Before you start: what a listing is and is not

A listing says **an implementation of this specification exists, here is where, here is who
maintains it.** That is all it says today.

It does **not** say the implementation is conformant, and it does not carry the certified mark,
because [the conformance suite for spec v1.0.0 does not exist yet](README.md#the-certification-gate-does-not-exist-yet).
When it ships, every listing already here will be run against it and the results published — passes
and failures alike, with no quiet removals.

If you would rather wait for a gate that means something, that is a reasonable thing to do, and
this file will still be here.

## The steps

1. Fork this repository.
2. Add **one** file: `registry/manifests/<your-impl>.toml`, using the schema below.
3. Open a pull request titled `registry: <your-impl>`.
4. CI checks the file parses and the required fields are present. A maintainer checks that the URL
   resolves and that the manifest describes what is at the other end.
5. It is merged, or you get a written reason. There is no third outcome.

## The schema

```toml
# registry/manifests/<your-impl>.toml

[impl]
name        = "your-impl-name"       # required · lowercase, matches the filename
language    = "rust"                 # required
maintainer  = "Your Org"             # required · who to contact about a regression
url         = "https://github.com/your-org/your-impl"   # required · must resolve
contact     = "spec@your-org.example"                   # required

[conformance]
spec_version = "1.0.0"               # required · the spec version you implement against
# registry_token — do NOT set this field. No tokens are issued and none are recognised;
# see registry/README.md. A manifest that sets it will be asked to remove it.

[scope]
modes      = ["direct", "proxy"]     # required · which modes you implement
models     = ["gemma-2-9b"]          # optional · what you have actually run it on
extensions = []                      # optional
```

`models` is the field worth being careful with. List what you have run, not what you expect to
work — the point of a registry is that a reader can check it.

## What gets you rejected

- A URL that does not resolve, or resolves to something other than an implementation.
- A manifest asserting certification, conformance or a token.
- A listing for something that does not exist yet. Come back when it does.

## Adopters

Running styxx in production and willing to say so? Same process, `registry/manifests/` with
`[impl] name` set to your organisation and a note in the PR that it is an adopter listing rather
than an implementation. We do not add adopters on their behalf, ever, including from a conversation
or a public post — a listing has to come from the party it names.
