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

Please report suspected security issues privately through GitHub's
private vulnerability reporting for this repository:

**<https://github.com/fathom-lab/styxx/security/advisories/new>**

(the same form is behind **Report a vulnerability** on the repository's
**Security** tab). A report filed there is private: GitHub shows it to
you and the repository's maintainers, not to the public.

**If you emailed a report to the address this file used to name, we
never received it.** That address is on a domain with no mail (MX)
record, so nothing sent to it reached anyone. Please send the report
again through the link above.

We will acknowledge within 72 hours, provide a triage assessment within
7 days, and coordinate a fix and disclosure on a timeline appropriate to
the severity. We will not file a CVE before informing you, and we
will credit reporters who want credit.

Please do **not** open public GitHub issues for security-sensitive
reports. If you believe a public issue is the right venue (for example,
a clearly low-severity hygiene issue), say so explicitly in your private
report and we'll move quickly.

## Supply-chain posture

This is the open MIT protocol's reference implementation. Trust signals:

- **Source of truth:** [`fathom-lab/styxx`](https://github.com/fathom-lab/styxx) on GitHub.
- **Releases on PyPI** are built and published by GitHub Actions in this
  repository when a version tag is pushed. The workflow that does this is
  [`.github/workflows/publish.yml`](.github/workflows/publish.yml). It uploads
  with a PyPI API token held as a repository secret; [PyPI Trusted
  Publishing][tp] (OIDC, no long-lived token) is not set up yet.
- **Both an sdist (`*.tar.gz`) and a wheel (`*.whl`)** are published for every
  tagged release. Source distributions allow downstream packagers
  (conda-forge, distros, vendor SBOM tooling) to build from source.
- **PEP 740 attestations are not produced yet** (the publish step sets
  `attestations: false`; they need Trusted Publishing). Until they are,
  check an artifact by its SHA-256, as below.
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
3. Compare the `*.whl` and `*.tar.gz` SHA-256 sums between PyPI, the
   assets attached to the GitHub Release, and the hashes its notes give.

If anything in steps 2–3 doesn't line up, do not install the artifact.
Report the discrepancy immediately through
[private vulnerability reporting](https://github.com/fathom-lab/styxx/security/advisories/new).

## What we will not do

- We will not silently change the cognometric scoring contract within
  a stable spec version. Any change to the wire format requires a new
  spec version under the [versioning policy](https://styxx.org/governance).
- We will not gate the open MIT protocol behind a token, an account,
  or a paywall.
- We will not ship release artifacts produced from a contributor laptop
  rather than from this repository's published-and-pinned workflow.

— Fathom Lab
