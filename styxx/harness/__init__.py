# -*- coding: utf-8 -*-
"""styxx.harness — adapters, never a recorder.

in-toto Witness records what a step did and signs it. The modules here turn bytes a harness
already holds — a test report (`junit`), a CI event payload and a diff (`github`), a tool-hook
payload (`claude_code`) — into a `sworn/manifest/0.2` file that `styxx.sworn` resolves `rN`
spans against. They sign nothing, fetch nothing, verify no signature, observe no process and
produce no verdict. The rung on every manifest is the one the caller declared, never one the
adapter detected, and L1 is printed as weak on every manifest that declares it.

Built to `papers/sworn/DESIGN_harness_adapters_2026_09_02.md`, frozen and re-sworn at f28d35a
before any of this existed. The purity boundary runs the other way: `styxx.sworn` and
`styxx.evidence` import nothing from here (tests/test_harness_purity_boundary.py).

No submodule is imported eagerly. `styxx.harness.claude_code` is stdlib-only at import so the
PostToolUse entry script under `integrations/claude-code/sworn-hooks/` can load it by file path
without running `styxx/__init__.py` on every tool call. Do not confuse this package with
`styxx.hooks`, which is the unrelated OpenAI telemetry patch.
"""
from __future__ import annotations

HARNESS_VERSION = "styxx.harness/0.1"

# The label every manifest string ends with. The plan's words for this leg.
LABEL = ("adapters, never a recorder: bytes a harness already holds, turned into a "
         "sworn/manifest/0.2 file; nothing signed, fetched, observed or verified")

# SPEC_sworn_output_v02 R6, quoted where the rung is declared.
L1_WEAK = ("L1: a local hook sharing a filesystem and a shell with the agent — weak; the agent's "
           "shell can write what the hook later reads")

__all__ = ["junit", "github", "claude_code", "HARNESS_VERSION", "LABEL", "L1_WEAK"]
