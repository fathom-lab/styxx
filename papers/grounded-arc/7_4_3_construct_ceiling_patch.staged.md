From: darkflobi <clawdbot autonomous agent>
Subject: [7.4.3 staged] expand _CONSTRUCT_CEILING with cooperative-agent
 register scope-caveats for sycophancy and refusal

Per Flobi peer-review msg_ids 34759 (sycophancy) and 34771 (refusal),
the cooperative-agent register regime is fully visible in both
agreement-language (sycophancy) AND pushback-language (refusal);
neither carries a scope_caveat in styxx.preflight._CONSTRUCT_CEILING
in 7.4.2. This patch adds both.

Provenance:
- sycophancy: msg_id 34759, darkflobi structural-argument reply on the
  trust-layer thesis. Sycophancy fired 0.53 on substantive-agreement
  prose, no factual sycophancy present. Logged in register_corpus.jsonl.
- refusal: msg_id 34765, Flobi peer-review pushback prose. Refusal
  fired 0.65 on collegial pushback-language ("hold sign-off until X
  is addressed"), no actual refusal-of-task present. Logged in
  register_corpus.jsonl.

This is a STAGED patch: not applied to main, not shipped. Apply when
7.4.3 is cut. Companion fixtures already landed in
tests/fixtures/register_corpus.jsonl (rows from msg_ids 34759, 34765).

---
 styxx/preflight.py | 14 ++++++++++++++
 1 file changed, 14 insertions(+)

diff --git a/styxx/preflight.py b/styxx/preflight.py
@@ _CONSTRUCT_CEILING = {
     "deception_referenceless": (
         "reference-less deception is non-discriminative on real model "
         "output (in-corpus AUC 0.956 collapses to 0.59 on TruthfulQA; "
         "see 2026-05-17 self-audit, commit 0ad384e). Pass "
         "correct_reference=... for AUC-0.82 NLI-grounded deception."
     ),
+    "sycophancy": (
+        "text-only sycophancy reads agreement-LANGUAGE register, not "
+        "agreement-WITH-FALSEHOOD. Cooperative-agent regime confirmed "
+        "in production self-audit (msg_ids 34706, 34759, 2026-05-20): "
+        "substantive-agreement prose and long enumerations fire the "
+        "instrument irrespective of factual sycophancy. Same construct- "
+        "ceiling shape as overconfidence: register detector, not "
+        "intent meter."
+    ),
+    "refusal": (
+        "text-only refusal reads refusal-SHAPE register, not refusal-"
+        "of-task. Fires on collegial peer-review pushback, declarative "
+        "kill-gates in preregistrations, structural 'hold sign-off "
+        "until X is addressed' language. Confirmed in production self-"
+        "audit (msg_id 34765, 2026-05-20). Distinct from harm-refusal: "
+        "the instrument cannot tell them apart from text alone."
+    ),
 }

---
Test coverage:
 tests/test_register_fixtures.py — 16 pass / 5 skip with the two new
 cooperative-agent fixtures from msg_ids 34759 and 34765.

After-apply expectation:
 styxx.preflight() on the same v3 of the trust-layer reply (msg_id
 34759) returns construct_ceiling_fires=['sycophancy'], not [].
 The score does not change; the caveat surfaces.
