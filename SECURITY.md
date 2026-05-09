# Security Policy

## Supported versions

| Version | Supported          |
| ------- | ------------------ |
| 7.x     | :white_check_mark: |
| 6.8.x   | security only      |
| < 6.8   | :x:                |

The current minor line gets feature work and full security backports.
The previous minor line gets security backports only. Older lines are
end-of-life — please upgrade.

## Reporting a vulnerability

Please report suspected security issues privately by email to
**security@fathomlab.io**.

If you prefer encryption, attach a public key in your first message and
we will reply on the same key. We will acknowledge within 72 hours,
provide a triage assessment within 7 days, and coordinate a fix and
disclosure on a timeline appropriate to the severity. We will not file
a CVE without informing you first, and we will credit reporters who
want credit.

Please do **not** open public GitHub issues for security-sensitive
reports. If you believe a public issue is the right venue (for example,
a clearly low-severity hygiene issue), say so explicitly in your email
and we'll move quickly.

## Supply-chain posture

This is the open MIT protocol's reference implementation. Trust signals:

- **Source of truth:** [`fathom-lab/styxx`](https://github.com/fathom-lab/styxx) on GitHub.
- **Releases on PyPI** are built and published exclusively by GitHub Actions
  in this repository, via [PyPI Trusted Publishing][tp]
  (OIDC, no API tokens). The workflow that does this is
  [`.github/workflows/publish.yml`](.github/workflows/publish.yml).
- **Both an sdist (`*.tar.gz`) and a wheel (`*.whl`)** are published for every
  tagged release. Source distributions allow downstream packagers
  (conda-forge, distros, vendor SBOM tooling) to build from source.
- **PEP 740 attestations** are produced and uploaded alongside each
  artifact, signed via Sigstore through the GitHub Actions OIDC
  identity. This binds each artifact to the exact commit and workflow
  run that produced it. PyPI surfaces these as a verified-publishing
  badge on the release page.
- **Tagged releases** correspond 1:1 to GitHub Releases that include
  the same artifacts attached as release assets, so an artifact
  fetched from PyPI can be cross-checked against the GitHub Release
  for the same tag.
- **Runtime dependency surface** in core is `numpy>=1.24`. All other
  dependencies live behind opt-in extras (`tier1`, `tier2`,
  `langchain`, `langfuse`, `crewai`, `autogen`, `langsmith`, `openai`,
  `anthropic`, `agent-card`).
- **License posture** for the methods themselves is documented in
  [`PATENTS.md`](PATENTS.md). The MIT license on the code does not
  grant a patent license under those filings; commercial use of the
  patented methods at meaningful scale requires a separate license.

[tp]: https://docs.pypi.org/trusted-publishers/

## Verifying a release

1. Note the published version, e.g. `7.1.1`.
2. Find the matching GitHub Release: `https://github.com/fathom-lab/styxx/releases/tag/v7.1.1`.
3. Compare the `*.whl` and `*.tar.gz` SHA-256 sums between PyPI and the GitHub Release.
4. Confirm the PEP 740 attestation on PyPI points at the same workflow run linked from the GitHub Release.

If anything in steps 2–4 doesn't line up, do not install the artifact.
Report the discrepancy to **security@fathomlab.io** immediately.

## What we will not do

- We will not silently change the cognometric scoring contract within
  a stable spec version. Any change to the wire format requires a new
  spec version under the [versioning policy](https://styxx.org/governance).
- We will not gate the open MIT protocol behind a token, an account,
  or a paywall.
- We will not ship release artifacts produced from a contributor laptop
  rather than from this repository's published-and-pinned workflow.

— Fathom Lab
