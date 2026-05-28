# Agent self-report — claims made to the operator (to be falsified)

These are the load-bearing factual claims from my last message. Run them through
`styxx audit-claims` against the repo substrate, then independently verify the
numeric claims the closed template set cannot reach.

The file tests/test_agent_audit_extract_version_disambiguation.py contains "migration".
The file tests/test_agent_audit_extract_version_disambiguation.py contains "3.1.0a1".
The file styxx/agent_audit.py contains "version_bump".
The file styxx/agent_audit.py contains "_tmpl_version_bump".
The file scripts/dogfood/longitudinal_self_report_audit.py contains "git archive".
The file CHANGELOG.md contains "value_consistent_across_paths".
version is 7.7.10.

Non-template claims (verified separately, not by the gate):
- The full test suite reports 1145 passed.
- ruff reports all checks passed on styxx.
- The longitudinal audit reports 12 temporal divergences and 0 genuine contradictions.
