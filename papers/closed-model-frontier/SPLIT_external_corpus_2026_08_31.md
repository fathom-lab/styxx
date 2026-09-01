# SPLIT COMMITMENT — the external corpus is divided before its numbers are known

Fathom Lab · 2026-08-31 · Frozen while the EXTERNAL-1 harness is still streaming, before
any aggregate result over the corpus has been computed or seen. Governs every present and
future styxx measurement over AIDev.

## Why this document exists now

EXTERNAL-1 measures the instrument **exactly as it ships**. Its most likely honest outcome
is that the closed template set is narrow on real-world agent prose: a body that says
*"adding an extensive list … to the `README.md` file"* names a real path and makes a real,
checkable claim, but not in a form the shipped templates parse. If that is what EXTERNAL-1
finds, the obvious successor is a template that checks **path mentions** rather than only
path-shaped verb phrases.

That successor is exactly where a lab fools itself: design a rule by reading the corpus,
then report the rule's score on the same corpus. This lab has already published what that
looks like when it is caught — OBLIGATE-1's in-sample precision collapsed on held-out
adjudication, and the collapse was published. The split below is committed **before** the
temptation exists.

## The split, frozen

Every PR is assigned by its **repository**, never individually, so that no repository's
prose appears on both sides:

```
bucket(pr) = int(sha256(repo_url_normalized).hexdigest()[:8], 16) % 10
DEVELOPMENT  if bucket < 3      (~30% of repositories)
HELD-OUT     otherwise          (~70% of repositories)
```

`repo_url_normalized` is the corpus's `repo_url` field, lowercased and stripped of any
trailing slash. The rule is deterministic, needs no stored state, and can be recomputed by
anyone from the published corpus.

## The rules this binds us to

1. **Any new template, threshold, or heuristic is designed on DEVELOPMENT only.** Reading
   held-out prose to inform a rule voids the held-out set, and the void must be disclosed.
2. **Every headline number for a new instrument is reported on HELD-OUT**, with the
   development number shown beside it. Where they diverge, the divergence is the finding —
   as it was for OBLIGATE-1.
3. **EXTERNAL-1 is exempt and reports over both**, because it measures an instrument that
   was frozen and shipped long before this corpus was chosen. Its per-bucket numbers are
   published anyway, so a reader can verify the buckets are not pathological.
4. **The split is never re-drawn.** Not for a better result, not for a fuller development
   set. A re-drawn split is a new corpus and must be declared as one.

---

*The rule that lets us fool ourselves is written before we have anything to gain by
breaking it. That is the only time such a rule is cheap.*
