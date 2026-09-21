# -*- coding: utf-8 -*-
"""The declared list of checking actions -- the living copy of SWALLOW-3's catalogue
(`benchmarks/harness_mutation/action_checks.py`, frozen at the sha256 the SWALLOW-3 receipt names).
`tests/test_ciaudit.py` holds this copy equal to the frozen one, entry for entry.

The rule (the preregistration states it): an action step (`uses:`) is a check when the action's
documented purpose is to produce a verdict on the repository's code -- run its tests, lint or
format-check it, type-check it, or scan it -- and either fails the step on findings by default or
publishes the findings to a status the repository can require. Not a check: setup, cache, checkout,
build, publish, deploy, uploads of results already produced, notify, comment, label, a judgement of
the pull request's metadata, a reporter that does not fail the step by default. Inputs decide when
the action documents them; a carried command (`nick-fields/retry`'s `command`) is judged by the
run-step rule. A local action, a docker image, a reusable workflow and `actions/github-script`
cannot be read.
"""
from __future__ import annotations

KINDS = ("test", "lint", "typecheck", "security", "status", "carried")

# ----------------------------------------------------------------------------- the catalogue
#
# entry keys: kind, why, verified (default True);
#   off      -- [(input, values)]: the check is removed when the input has one of the values
#   on       -- [(input, values)]: the check exists only when some listed input has a listed value
#                (values None = any non-empty value)
#   format_input -- an input that, when present without the word "check", turns the tool into a formatter that writes
#   carry    -- the input whose text is judged by SWALLOW-2's verification() (prefix prepended)
#   carry_as_name -- the input whose text is judged as a step NAME (NAME_RX / EXCL_RX)
#   nested   -- the input naming another action, judged in its place (Wandalen/wretry.action)
_T, _F = ("true", "1", "yes", "on"), ("false", "0", "no", "off")
CATALOGUE: dict[str, dict] = {
    # --- test
    "cypress-io/github-action": {"kind": "test", "why": "runs the Cypress suite; fails on failing tests", "off": [("runTests", _F)]},
    "dorny/test-reporter": {"kind": "test", "why": "reporter that fails the step on recorded test failures by default (fail-on-error: true); after `npm test || true` it is the gate", "off": [("fail-on-error", _F)]},
    "EnricoMi/publish-unit-test-result-action": {"kind": "test", "why": "reporter; fails the step only with action_fail: true", "on": [("action_fail", _T)]},
    "mikepenz/action-junit-report": {"kind": "test", "why": "reporter; fails the step only with fail_on_failure: true", "on": [("fail_on_failure", _T)]},
    "ScaCap/action-surefire-report": {"kind": "test", "why": "reporter; fails the step only with fail_on_test_failures: true", "on": [("fail_on_test_failures", _T)]},
    "ArtiomTr/jest-coverage-report-action": {"kind": "test", "why": "runs jest with coverage; fails on failing tests or a threshold"},
    "mattallty/jest-github-action": {"kind": "test", "why": "runs jest; fails on failing tests"},
    "pavelzw/pytest-action": {"kind": "test", "why": "runs pytest; fails on failing tests"},
    "Use-Tusk/test-runner": {"kind": "test", "why": "runs the repository's testScript / lintScript under Tusk's runner; fails on failures", "verified": False},
    "grafana/run-k6-action": {"kind": "test", "why": "runs k6 scripts; thresholds fail the step"},
    "chromaui/action": {"kind": "test", "why": "Chromatic visual tests; exits non-zero on changes unless exitZeroOnChanges / exitOnceUploaded", "off": [("exitZeroOnChanges", _T), ("exitOnceUploaded", _T)]},
    "r-lib/actions/check-r-package": {"kind": "test", "why": "R CMD check; fails on errors"},
    "anuraag016/Jest-Coverage-Diff": {"kind": "carried", "why": "runs runCommand (the test command) on both branches; fails on failures or a coverage drop", "carry": "runCommand"},
    # --- lint / format-check / docs / links / workflows / api
    "pre-commit/action": {"kind": "lint", "why": "runs the repository's pre-commit hooks; fails on findings"},
    "golangci/golangci-lint-action": {"kind": "lint", "why": "runs golangci-lint; fails on findings"},
    "dominikh/staticcheck-action": {"kind": "lint", "why": "runs staticcheck; fails on findings"},
    "super-linter/super-linter": {"kind": "lint", "why": "runs the linters it detects; fails on findings"},
    "github/super-linter": {"kind": "lint", "why": "runs the linters it detects; fails on findings"},
    "oxsecurity/megalinter": {"kind": "lint", "why": "runs the linters it detects; fails on errors by default"},
    "astral-sh/ruff-action": {"kind": "lint", "why": "ruff check by default; `args: format` without --check rewrites files", "format_input": "args"},
    "chartboost/ruff-action": {"kind": "lint", "why": "ruff check by default; args without check rewrite files", "format_input": "args"},
    "psf/black": {"kind": "lint", "why": "black --check --diff by default; options without --check rewrite files", "format_input": "options"},
    "actions-rs/clippy-check": {"kind": "lint", "why": "runs clippy; fails on warnings"},
    "actions-rs/cargo": {"kind": "carried", "why": "runs `cargo <command>`; a check when the command is test / clippy / fmt --check", "carry": "command", "carry_prefix": "cargo "},
    "hadolint/hadolint-action": {"kind": "lint", "why": "lints Dockerfiles; fails on findings", "off": [("failure-threshold", ("ignore",))]},
    "ludeeus/action-shellcheck": {"kind": "lint", "why": "runs shellcheck; fails on findings"},
    "luizm/action-sh-checker": {"kind": "lint", "why": "shellcheck and shfmt; fails on findings"},
    "raven-actions/actionlint": {"kind": "lint", "why": "lints workflows; fails on findings by default", "off": [("fail-on-error", _F)]},
    "DavidAnson/markdownlint-cli2-action": {"kind": "lint", "why": "lints markdown; fails on findings; fix: true rewrites", "off": [("fix", _T)]},
    "crate-ci/typos": {"kind": "lint", "why": "spell-checks source; fails on findings; write_changes rewrites", "off": [("write_changes", _T)]},
    "codespell-project/actions-codespell": {"kind": "lint", "why": "codespell; fails on findings unless only_warn", "off": [("only_warn", None)]},
    "lycheeverse/lychee-action": {"kind": "lint", "why": "link checker; fails on broken links by default (v2: fail: true); fail: false removes it", "off": [("fail", _F)]},
    "JustinBeckwith/linkinator-action": {"kind": "lint", "why": "link checker; fails on broken links"},
    "gaurav-nelson/github-action-markdown-link-check": {"kind": "lint", "why": "link checker; fails on broken links"},
    "becheran/mlc": {"kind": "lint", "why": "markup link checker; fails on broken links"},
    "katexochen/go-tidy-check": {"kind": "lint", "why": "fails when `go mod tidy` would change go.mod / go.sum"},
    "bnjbvr/cargo-machete": {"kind": "lint", "why": "fails on unused dependencies"},
    "arduino/arduino-lint-action": {"kind": "lint", "why": "lints Arduino projects; fails on errors"},
    "oasdiff/oasdiff-action/breaking": {"kind": "lint", "why": "fails on breaking OpenAPI changes when fail-on is set", "on": [("fail-on", None)]},
    "sonarsource/sonarqube-quality-gate-action": {"kind": "lint", "why": "waits for the quality gate; fails the step when it fails"},
    "SonarSource/sonarqube-quality-gate-action": {"kind": "lint", "why": "waits for the quality gate; fails the step when it fails"},
    "wearerequired/lint-action": {"kind": "lint", "why": "runs linters; fails the step only with continue_on_error: false (default true)", "on": [("continue_on_error", _F)]},
    "ataylorme/eslint-annotate-action": {"kind": "lint", "why": "reads an eslint report; fails the step on errors by default", "off": [("fail-on-error", _F)]},
    "creyD/prettier_action": {"kind": "lint", "why": "commits prettier's changes by default; a check only with dry: true", "on": [("dry", _T)]},
    "errata-ai/vale-action": {"kind": "lint", "why": "reviewdog-based prose linter; fails only with fail_on_error: true", "on": [("fail_on_error", _T)]},
    "aaronsteers/tk-todo-check-action": {"kind": "lint", "why": "checks for TODO markers; assumed to fail on findings", "verified": False},
    "dotnet/docs-actions/docs-verifier": {"kind": "lint", "why": "verifies docs; assumed to fail on findings", "verified": False},
    "dotnet/docs-actions/status-checker": {"kind": "lint", "why": "checks links in docs; assumed to fail on findings", "verified": False},
    "TrueBrain/actions-flake8": {"kind": "lint", "why": "runs flake8; fails on findings"},
    "py-actions/flake8": {"kind": "lint", "why": "runs flake8; fails on findings"},
    "tj-actions/eslint-changed-files": {"kind": "lint", "why": "runs eslint on changed files; fails on errors"},
    "mansagroup/nrwl-nx-action": {"kind": "carried", "why": "runs nx targets; a check when a target is test / lint / typecheck", "carry_as_name": "targets"},
    "magefile/mage-action": {"kind": "carried", "why": "runs `mage <args>`", "carry": "args", "carry_prefix": "mage "},
    "gradle/gradle-build-action": {"kind": "carried", "why": "runs `gradle <arguments>` (deprecated input); a check when the arguments are test / check / lint", "carry": "arguments", "carry_prefix": "gradle "},
    # --- typecheck
    "jakebailey/pyright-action": {"kind": "typecheck", "why": "runs pyright; fails on errors"},
    "jpetrucciani/mypy-check": {"kind": "typecheck", "why": "runs mypy; fails on errors"},
    # --- security: code, dependencies, secrets, licenses, containers
    "actions/dependency-review-action": {"kind": "security", "why": "fails on vulnerable or disallowed dependency changes by default"},
    "gitleaks/gitleaks-action": {"kind": "security", "why": "fails on leaked secrets"},
    "trufflesecurity/trufflehog": {"kind": "security", "why": "fails on verified secrets"},
    "returntocorp/semgrep-action": {"kind": "security", "why": "semgrep ci; fails on blocking findings"},
    "semgrep/semgrep-action": {"kind": "security", "why": "semgrep ci; fails on blocking findings"},
    "bridgecrewio/checkov-action": {"kind": "security", "why": "fails on findings unless soft_fail", "off": [("soft_fail", _T)]},
    "securego/gosec": {"kind": "security", "why": "fails on findings"},
    "anchore/scan-action": {"kind": "security", "why": "grype scan; fail-build: true by default", "off": [("fail-build", _F)]},
    "aquasecurity/trivy-action": {"kind": "security", "why": "exit-code: 0 by default (report only); a check with exit-code: 1", "on": [("exit-code", ("1",))]},
    "docker/scout-action": {"kind": "security", "why": "report only by default; a check with exit-code: true", "on": [("exit-code", _T)]},
    "EmbarkStudios/cargo-deny-action": {"kind": "security", "why": "cargo deny check: advisories, licenses, bans; fails on findings"},
    "pilosus/action-pip-license-checker": {"kind": "security", "why": "fails on the license types named in `fail`", "on": [("fail", None)]},
    "getsentry/action-enforce-license-compliance": {"kind": "security", "why": "FOSSA license test; fails on violations", "verified": False},
    "oxc-project/security-action": {"kind": "security", "why": "the oxc project's own scanner; assumed to fail on findings", "verified": False},
    "promptfoo/code-scan-action": {"kind": "security", "why": "LLM code scan with min-severity; assumed to fail on findings at or above it", "verified": False},
    # --- status-checks: findings go to a status the repository can require; the step does not fail
    "github/codeql-action/analyze": {"kind": "status", "why": "CodeQL analysis; alerts go to code scanning, whose status can be required"},
    "zizmorcore/zizmor-action": {"kind": "status", "why": "workflow audit; findings go to code scanning by default, or fail the step with advanced-security: false"},
    "sonarsource/sonarqube-scan-action": {"kind": "status", "why": "the scan feeds the quality gate status"},
    "SonarSource/sonarqube-scan-action": {"kind": "status", "why": "the scan feeds the quality gate status"},
    "SonarSource/sonarcloud-github-action": {"kind": "status", "why": "the scan feeds the quality gate status"},
    # --- carried commands: the run-step rule applied to the command the action runs
    "nick-fields/retry": {"kind": "carried", "why": "retries `command`", "carry": "command"},
    "nick-invision/retry": {"kind": "carried", "why": "retries `command`", "carry": "command"},
    "Wandalen/wretry.action": {"kind": "carried", "why": "retries `command`, or the nested `action`", "carry": "command", "nested": "action"},
    "reactivecircus/android-emulator-runner": {"kind": "carried", "why": "runs `script` on the emulator", "carry": "script"},
    "vmactions/freebsd-vm": {"kind": "carried", "why": "runs `run` in the VM", "carry": "run"},
    "CodSpeedHQ/action": {"kind": "carried", "why": "runs `run` (the benchmark command, usually pytest --codspeed)", "carry": "run"},
}
# prefix families: (prefix, entry); a name matching a prefix and not an exact entry uses it
FAMILIES: list[tuple[str, dict]] = [
    ("reviewdog/action-", {"kind": "lint", "why": "reviewdog: comments by default; fails the step only with fail_on_error: true or fail_level: any|warning|error",
                           "on": [("fail_on_error", _T), ("fail_level", ("any", "warning", "error"))]}),
    ("tsuyoshicho/action-", {"kind": "lint", "why": "reviewdog-based; fails the step only with fail_on_error / fail_level",
                             "on": [("fail_on_error", _T), ("fail_level", ("any", "warning", "error"))]}),
    ("snyk/actions/", {"kind": "security", "why": "snyk test; fails on vulnerabilities", "not": ("snyk/actions/setup",)}),
]

# every other action the population's hand-written workflows use, by the category that excludes it
NOT_CHECKS: dict[str, str] = {
    # checkout
    "actions/checkout": "checkout", "taiki-e/checkout-action": "checkout",
    # setup: toolchains, tools, environments
    "actions/setup-node": "setup", "actions/setup-python": "setup", "actions/setup-java": "setup", "actions/setup-go": "setup",
    "actions/setup-dotnet": "setup", "astral-sh/setup-uv": "setup", "pnpm/action-setup": "setup", "pnpm/setup": "setup",
    "oven-sh/setup-bun": "setup", "dtolnay/rust-toolchain": "setup", "oxc-project/setup-node": "setup", "oxc-project/setup-rust": "setup",
    "actions-rust-lang/setup-rust-toolchain": "setup", "taiki-e/install-action": "setup", "github/gh-aw-actions/setup": "setup",
    "github/gh-aw-actions/setup-cli": "setup", "github/gh-aw/actions/setup-cli": "setup", "gradle/actions/setup-gradle": "setup",
    "mlugg/setup-zig": "setup", "goto-bus-stop/setup-zig": "setup", "ruby/setup-ruby": "setup", "erlef/setup-beam": "setup",
    "shivammathur/setup-php": "setup", "snok/install-poetry": "setup", "extractions/setup-just": "setup", "jaxxstorm/action-install-gh-release": "setup",
    "cargo-bins/cargo-binstall": "setup", "foundry-rs/foundry-toolchain": "setup", "foundry-rs/setup-snfoundry": "setup", "software-mansion/setup-scarb": "setup",
    "aiken-lang/setup-aiken": "setup", "acifani/setup-tinygo": "setup", "bytecodealliance/actions/wasm-tools/setup": "setup", "arduino/setup-arduino-cli": "setup",
    "arduino/setup-protoc": "setup", "android-actions/setup-android": "setup", "subosito/flutter-action": "setup", "maxim-lobanov/setup-xcode": "setup",
    "swift-actions/setup-swift": "setup", "graalvm/setup-graalvm": "setup", "r-lib/actions/setup-r": "setup", "r-lib/actions/setup-r-dependencies": "setup",
    "r-lib/actions/setup-pandoc": "setup", "ilammy/msvc-dev-cmd": "setup", "microsoft/setup-msbuild": "setup", "azure/setup-helm": "setup",
    "azure/setup-kubectl": "setup", "google-github-actions/setup-gcloud": "setup", "helm/kind-action": "setup", "engineerd/setup-kind": "setup",
    "bazel-contrib/setup-bazel": "setup", "samypr100/setup-dev-drive": "setup", "browser-actions/setup-chrome": "setup", "grafana/setup-k6-action": "setup",
    "supabase/setup-cli": "setup", "superfly/flyctl-actions/setup-flyctl": "setup", "jfrog/setup-jfrog-cli": "setup", "getsentry/action-setup-cli": "setup",
    "getsentry/action-setup-venv": "setup", "biomejs/setup-biome": "setup", "Boshen/setup-ohos-sdk": "setup", "useblacksmith/setup-node": "setup",
    "useblacksmith/setup-docker-builder": "setup", "useblacksmith/begin-testbox": "setup", "crazy-max/ghaction-setup-docker": "setup",
    "crazy-max/ghaction-github-runtime": "setup", "step-security/harden-runner": "setup", "tailscale/github-action": "setup", "webfactory/ssh-agent": "setup",
    "browserstack/github-actions/setup-local": "setup", "browserstack/github-actions/setup-env": "setup", "gittools/actions/gitversion/setup": "setup",
    "docker/setup-buildx-action": "setup", "docker/setup-qemu-action": "setup", "cachix/install-nix-action": "setup", "FastLED/fbuild/.github/actions/setup": "setup",
    "snyk/actions/setup": "setup", "github/codeql-action/init": "setup", "github/codeql-action/autobuild": "build",
    # cache
    "actions/cache": "cache", "actions/cache/restore": "cache", "actions/cache/save": "cache", "Swatinem/rust-cache": "cache",
    "namespacelabs/nscloud-cache-action": "cache", "burrunan/gradle-cache-action": "cache", "rharkor/caching-for-turbo": "cache",
    "hendrikmuhs/ccache-action": "cache", "mozilla-actions/sccache-action": "cache", "Mozilla-Actions/sccache-action": "cache",
    "cachix/cachix-action": "cache", "useblacksmith/cache": "cache", "jlumbroso/free-disk-space": "housekeeping",
    # auth, tokens, credentials
    "actions/create-github-app-token": "auth", "tibdex/github-app-token": "auth", "getsentry/action-github-app-token": "auth",
    "docker/login-action": "auth", "aws-actions/configure-aws-credentials": "auth", "aws-actions/amazon-ecr-login": "auth", "azure/login": "auth",
    "google-github-actions/auth": "auth", "NuGet/login": "auth", "rust-lang/crates-io-auth-action": "auth",
    "dotnet/docs-tools/.github/actions/azure-oidc-auth": "auth",
    # artifacts: upload, download, pages artifacts, coverage and result uploads
    "actions/upload-artifact": "artifact", "actions/download-artifact": "artifact", "dawidd6/action-download-artifact": "artifact",
    "actions/upload-pages-artifact": "artifact", "actions/upload-code-coverage": "coverage-upload", "codecov/codecov-action": "coverage-upload",
    "codecov/test-results-action": "coverage-upload", "coverallsapp/github-action": "coverage-upload", "codacy/codacy-coverage-reporter-action": "coverage-upload",
    "github/codeql-action/upload-sarif": "results-upload", "primer/.github/.github/actions/upload-versions": "artifact",
    # build, package, sign
    "docker/build-push-action": "build", "useblacksmith/build-push-action": "build", "docker/bake-action": "build", "docker/metadata-action": "build",
    "PyO3/maturin-action": "build", "getsentry/action-build-and-push-images": "build", "flatpak/flatpak-github-actions/flatpak-builder": "build",
    "gittools/actions/gitversion/execute": "build", "actions/attest-build-provenance": "sign", "actions/attest": "sign", "sigstore/cosign-installer": "sign",
    "danielroe/provenance-action": "sign", "azure/trusted-signing-action": "sign", "sslcom/esigner-codesign": "sign", "crazy-max/ghaction-import-gpg": "sign",
    "kolaente/action-gpg": "sign",
    # publish, release, deploy, push
    "softprops/action-gh-release": "publish", "ncipollo/release-action": "publish", "googleapis/release-please-action": "publish",
    "release-drafter/release-drafter": "publish", "changesets/action": "publish", "pypa/gh-action-pypi-publish": "publish",
    "facebook/dotslash-publish-release": "publish", "goreleaser/goreleaser-action": "publish", "getsentry/action-release": "publish",
    "getsentry/craft": "publish", "linear/linear-release-action": "publish", "Comfy-Org/publish-node-action": "publish",
    "openjournals/openjournals-draft-action": "publish", "KSXGitHub/github-actions-deploy-aur": "publish", "expo/expo-github-action": "publish",
    "actions/deploy-pages": "deploy", "actions/configure-pages": "deploy", "peaceiris/actions-gh-pages": "deploy", "peaceiris/actions-hugo": "setup",
    "JamesIves/github-pages-deploy-action": "deploy", "cloudflare/wrangler-action": "deploy", "amondnet/vercel-action": "deploy",
    "nwtgck/actions-netlify": "deploy", "appleboy/ssh-action": "deploy", "marcodallasanta/ssh-scp-deploy": "deploy",
    "aws-actions/amazon-ecs-deploy-task-definition": "deploy", "aws-actions/amazon-ecs-render-task-definition": "deploy",
    "shallwefootball/s3-upload-action": "deploy", "kolaente/s3-action": "deploy", "chrnorm/deployment-action": "deploy", "chrnorm/deployment-status": "deploy",
    "newrelic/deployment-marker-action": "notify", "ad-m/github-push-action": "push", "stefanzweifel/git-auto-commit-action": "fixer",
    "getsentry/action-github-commit": "push", "getsentry/action-fast-revert": "push", "getsentry/action-migrations": "comment",
    "novuhq/actions-novu-sync": "deploy", "novuhq/clickhouse-cloud-whitelist-ip-action": "deploy", "lingodotdev/lingo.dev": "fixer",
    "crowdin/github-action": "publish", "peter-evans/dockerhub-description": "publish", "bots-house/ghcr-delete-image-action": "housekeeping",
    "snok/container-retention-policy": "housekeeping",
    # notify
    "slackapi/slack-github-action": "notify", "tokorom/action-slack-incoming-webhook": "notify", "tsickert/discord-webhook": "notify",
    "masci/datadog": "notify", "jgehrcke/github-repo-stats": "notify",
    # comments, pull requests, labels, issues, metadata
    "peter-evans/create-or-update-comment": "comment", "peter-evans/find-comment": "comment", "marocchino/sticky-pull-request-comment": "comment",
    "peter-evans/create-pull-request": "pr", "dotnet/actions-create-pull-request": "pr", "actions/labeler": "pr", "actions-ecosystem/action-add-labels": "pr",
    "actions/stale": "housekeeping", "dessant/lock-threads": "housekeeping", "getsentry/forked-action-lock-threads": "housekeeping",
    "kentaro-m/auto-assign-action": "pr", "eps1lon/actions-label-merge-conflict": "pr", "dependabot/fetch-metadata": "pr",
    "amannn/action-semantic-pull-request": "metadata", "steebchen/action-auto-semantic-pull-request": "metadata", "codelytv/pr-size-labeler": "metadata",
    "bcgov/action-pr-description-add": "metadata", "microsoft/PR-Metrics": "metadata", "AhmedBaset/checklist": "metadata",
    "contributor-assistant/github-action": "metadata", "manaflow-ai/cla-github-action": "metadata", "xt0rted/pull-request-comment-branch": "utility",
    "tj-actions/branch-names": "utility", "launchdarkly/find-code-references-in-pull-request": "comment", "e18e/action-dependency-diff": "comment",
    "be-hase/gradle-dependency-diff-action": "comment", "barecheck/code-coverage-action": "report", "LouisBrunner/checks-action": "report",
    "dotnet/docs-actions/ops-build-reporter": "report", "github/accessibility-alt-text-bot": "comment", "benchmark-action/github-action-benchmark": "report",
    "codespell-project/codespell-problem-matcher": "report", "ossf/scorecard-action": "report",
    # fixers and formatters that write
    "autofix-ci/action": "fixer", "steebchen/actions/autofix": "fixer", "rolfbjarne/autoformat": "fixer", "rolfbjarne/autoformat-push": "fixer",
    "calibreapp/image-actions": "fixer", "steebchen/image-actions": "fixer", "dotnet/docs-tools/cleanrepo": "fixer",
    "martincostello/update-dotnet-sdk": "fixer", "dotnet/docs-actions/dotnet-version-updater": "fixer", "dotnet/docs-actions/dependabot-bot": "fixer",
    "jacobtomlinson/gha-find-replace": "utility",
    # agents
    "anthropics/claude-code-action": "agent", "anthropics/claude-code-action/base-action": "agent", "openai/codex-action": "agent",
    "aaronsteers/devin-action": "agent", "appleboy/LLM-action": "agent", "actions/ai-inference": "agent", "openai/fence": "agent",
    "aaronsteers/poe-command-processor": "agent",
    # dispatch, chat-ops, flow control
    "benc-uk/workflow-dispatch": "dispatch", "peter-evans/repository-dispatch": "dispatch", "peter-evans/slash-command-dispatch": "dispatch",
    "github/command": "dispatch", "fkirc/skip-duplicate-actions": "utility",
    # utilities: strings, env files, versions, paths
    "dorny/paths-filter": "utility", "tj-actions/changed-files": "utility", "nrwl/nx-set-shas": "utility", "SpicyPizza/create-envfile": "utility",
    "kanga333/variable-mapper": "utility", "winterjung/split": "utility", "ASzc/change-string-case-action": "utility", "chuhlomin/render-template": "utility",
    "Saionaro/extract-package-version": "utility", "martinbeentjes/npm-get-version-action": "utility", "proudust/gh-describe": "utility",
    "actions-ecosystem/action-regex-match": "utility", "azure/azure-sdk-actions": "utility", "dotnet/docs-tools/.github/actions/sequester": "utility",
    "primer/.github/.github/actions/report-versions": "report",
    # issue-labeler models (verdicts on issues, not on the code)
    "dotnet/issue-labeler/restore": "utility", "dotnet/issue-labeler/download": "utility", "dotnet/issue-labeler/predict": "utility",
    "dotnet/issue-labeler/promote": "utility", "dotnet/issue-labeler/test": "utility", "dotnet/issue-labeler/train": "utility",
    # cannot be read
    "actions/github-script": "unreadable", "nick-fields/private-action-loader": "unreadable", "prebid/code-scanner": "unreadable",
}
UNREADABLE = ("actions/github-script", "nick-fields/private-action-loader")


def action_name(uses) -> tuple[str, str] | None:
    """('owner/repo[/path]', 'action') for a `uses:`; ('./x', 'local'); ('docker://…', 'docker');
    ('owner/repo/.github/workflows/x.yml', 'workflow'). The @ref is dropped. None when not a string."""
    if not isinstance(uses, str) or not uses.strip():
        return None
    u = uses.strip()
    if u.startswith("./") or u == ".":
        return (u, "local")
    if u.startswith("docker://"):
        return (u.split("@")[0], "docker")
    name = u.split("@")[0]
    if name.endswith((".yml", ".yaml")):
        return (name, "workflow")
    return (name, "action")


def _entry(name: str) -> dict | None:
    e = CATALOGUE.get(name)
    if e is not None:
        return e
    for prefix, fe in FAMILIES:
        if name.startswith(prefix) and name not in fe.get("not", ()):
            return fe
    return None


def _norm(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v).strip().lower() if v is not None else ""


def _inputs(step: dict) -> dict:
    w = step.get("with")
    return {str(k): v for k, v in w.items()} if isinstance(w, dict) else {}


def _has(w: dict, inp: str, values) -> bool:
    """The input is present with one of `values` (None: any non-empty value)."""
    if inp not in w:
        return False
    v = _norm(w[inp])
    return v != "" if values is None else v in values


def action_check(step: dict, _depth: int = 0, *, verification=None, fixer_rx=None) -> tuple[str, str] | None:
    """(kind, check_by) when a `uses:` step is a check under the catalogue, else None. check_by is
    'action', 'action-carried:tool' / 'action-carried:name' for a carried command, or
    'action-nested' for a retried action."""
    if verification is None or fixer_rx is None:
        from .engine import FIXER_RX as _f, verification as _v   # the run-step rule, applied to carried commands and fixer flags
        verification, fixer_rx = verification or _v, fixer_rx or _f
    an = action_name(step.get("uses"))
    if not an or an[1] != "action":
        return None
    e = _entry(an[0])
    if e is None:
        return None
    w = _inputs(step)
    if any(_has(w, inp, values) for inp, values in e.get("off", [])):
        return None
    if e.get("on") and not any(_has(w, inp, values) for inp, values in e["on"]):
        return None
    fi = e.get("format_input")
    if fi and fi in w and "check" not in _norm(w[fi]):
        return None                                   # `ruff format`, `black .`: the formatter writes
    if any(fixer_rx.search(_norm(v)) for v in w.values() if isinstance(v, (str, int, float, bool))):
        return None
    if e["kind"] == "carried":
        if e.get("nested") and isinstance(w.get(e["nested"]), str) and _depth < 2:
            nested_with = w.get("with")
            if isinstance(nested_with, str):
                try:
                    import yaml
                    nested_with = yaml.safe_load(nested_with)
                except Exception:  # noqa: BLE001
                    nested_with = None
            r = action_check({"uses": w[e["nested"]], "with": nested_with if isinstance(nested_with, dict) else {}, "name": step.get("name")}, _depth + 1,
                             verification=verification, fixer_rx=fixer_rx)
            return (r[0], "action-nested") if r else None
        if e.get("carry_as_name"):
            text = w.get(e["carry_as_name"])
            why = verification(str(text) if text is not None else "", "")
            return ("carried", "action-carried:name") if why else None
        text = w.get(e.get("carry", ""))
        if not isinstance(text, (str, int, float)) or not str(text).strip():
            return None
        why = verification(step.get("name"), e.get("carry_prefix", "") + str(text))
        return ("carried", "action-carried:" + why) if why else None
    return (e["kind"], "action")


def action_class(step: dict, *, verification=None, fixer_rx=None) -> str:
    """For the census line: 'check:<kind>', 'conditional-off' (a listed action whose inputs turn the
    check off), 'not:<category>', 'unlisted', 'local', 'docker', 'workflow', or 'run'."""
    if isinstance(step.get("run"), str):
        return "run"
    an = action_name(step.get("uses"))
    if not an:
        return "other"
    if an[1] != "action":
        return an[1]
    r = action_check(step, verification=verification, fixer_rx=fixer_rx)
    if r:
        return "check:" + r[0]
    if _entry(an[0]) is not None:
        return "conditional-off"
    return "not:" + NOT_CHECKS[an[0]] if an[0] in NOT_CHECKS else "unlisted"
