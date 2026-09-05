"""styxx.corpus_audit — re-certify a whole corpus of OATH certificates under the CURRENT verifier.

The mutant battery and the cycle-18/26 sweeps, productized into a standing, anyone-can-run check:
for every ``*.certificate.json`` under a root, resolve the receipts it recorded (next to the doc),
SHA-verify them (flag drift), re-run :func:`styxx.certify.certify_doc` under the *current* verifier,
and report each document's live verdict. Answers, on demand: *is every number we ever shipped still
grounded at the receipts it cited?*

Open by design (see ``OPEN_CORE.md``): this is a measurement primitive, never gated.

Two modes:
  * default — re-certification only (fast, deterministic): HELD / FAILED / receipt-drift / verdict-drift.
  * ``--tamper`` — additionally mutate every VERIFIED token once (single significant digit, seeded)
    and report the corpus tamper-catch rate (caught / false-verify / abstain-degrade). This is the
    ``papers/autopilot/mutant_battery.py`` scheme lifted into the package.

CLI::

    python -m styxx.corpus_audit [ROOT] [--tamper] [--seed N] [--json OUT.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path

from styxx.certify import certify_doc
from styxx import receipt_binding as _rb

# restore styxx.certify (the provenance function) over the submodule the line
# above just setattr'd onto the package — see the twin note in styxx/seal.py.
import styxx as _styxx_pkg
from styxx.provenance import certify as _provenance_certify
_styxx_pkg.certify = _provenance_certify
del _styxx_pkg, _provenance_certify

__all__ = ["discover_certificates", "audit_document", "audit_corpus", "mutate_token",
           "verdict_class"]


def verdict_class(verdict) -> str:
    """The HELD/FAILED dichotomy of a verdict string, with the v0.13 coverage suffix stripped.

    Since the UNCOVERED band (`styxx.certify.uncovered_spans`) a verdict reads
    `OATH-HELD, 3 uncovered` when the document carries numeric spans the extractor never
    examined. The suffix is a COVERAGE report travelling in the headline; it is not a verdict
    change — `counts["UNGROUNDED"]` is untouched and no token's status moved. An auditor that
    bucketed on the whole string put 131 of 208 certificates in neither HELD nor FAILED and
    reported every one of them as verdict drift the day the band shipped, which is false and
    would have buried the one real drift the corpus is holding. Bucket and drift-detect on the
    class; the suffix is folded separately as `uncovered_*` corpus totals.
    """
    return str(verdict or "").split(",")[0].strip()


def discover_certificates(root: Path) -> list[Path]:
    """Every ``*.certificate.json`` under *root*, sorted.

    Paths with an ``anc`` segment are skipped: arXiv submission staging mirrors the source
    certificate into ``submission/anc/`` next to a renamed ``source.md`` that does not exist
    on disk, so auditing those copies reports phantom MISSING_DOC entries for documents whose
    canonical certificates are audited at their real location.
    """
    return sorted(p for p in _search(root, "*.certificate.json") if "anc" not in p.parts)


# Directory names that are never part of the corpus: build outputs, virtualenvs, caches — and
# `.claude`, which is the one that actually bit.
#
# `.claude/worktrees/` holds agent scratch clones of this repository. A worktree is a full copy,
# so EVERY receipt in the tree has a byte-identical phantom twin inside each one: 1,611 JSON files
# at the time this was written, against 0 tracked JSON under `.claude` (its only tracked file is a
# skill markdown). The receipt search had no exclusions at all and walked straight into them.
#
# The cost was paid twice, and the larger half was the DENOMINATOR. `discover_certificates` had the
# same unscoped rglob, so the audit enumerated 365 certificates of which 178 — 49% — were phantoms
# from one stale worktree. Every corpus-wide number this module printed was computed over a
# population that was half ephemeral clone, and every finding was reported twice. Scoping the run
# to `papers/` (what REPLICATIONS.md documents, and what CI does on a fresh checkout) dodged it
# entirely, which is exactly why it survived: the published numbers were right and the numbers on
# a working copy were not.
#
# The other half was a HIDDEN FINDING. CAPSTONE_universal_mind's `mind_v0_validation.json`
# is present-and-CHANGED — one real file whose content is not what was certified. The phantom twin
# made two candidates out of one, so `classify_missing` reported `ambiguous` ("several candidates,
# no non-arbitrary choice") instead of `changed`, and the audit printed `receipt-changed 0` over a
# receipt it could see had drifted. The comment in `_resolve_receipts` had said "present and
# changed" about this exact file since it was written; the classifier disagreed with it because
# its search population included files that are not in the corpus.
#
# Same defect as everything else found on 2026-08-27: the measurement's population was defined by
# what a glob matched rather than by what the thing is. Exclusion is by directory NAME anywhere in
# the path, which is deliberately blunt — a receipt that exists ONLY inside one of these stays
# unresolved and is reported `absent`, which is the honest answer, not resolved from scratch space.
_EXCLUDED_DIRS = frozenset({
    ".git", ".claude", ".venv", "venv", "node_modules", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", "build", "dist",
})


def _search(root: Path, name: str):
    """`root.rglob(name)`, minus paths through a directory that is not part of the corpus."""
    for p in root.rglob(name):
        if _EXCLUDED_DIRS.isdisjoint(p.parts):
            yield p


def _receipt_sha_matches(raw: bytes, recorded: str) -> bool:
    """Does *raw* hash to *recorded*, on ANY platform's line endings?

    THE DEFECT THIS CLOSES. Receipt JSONs are stored in git as LF and checked out as CRLF on
    Windows, and every `receipts_sha256` in this corpus was recorded from a Windows working
    tree — so the pinned hashes are CRLF hashes. On Linux the same committed bytes hash
    differently, the cross-directory branch below finds no match, the receipt reports as
    `missing`, and the document is silently DROPPED from the drift guard.

    That is not hypothetical and it is not new. `.gitattributes` carries a note about the
    identical bug hitting `styxx/centroids/*.json` — "the pin was a CRLF-rendered hash, so the
    LF Linux CI checkout failed to verify" — fixed there with `-text` and nowhere else. It
    recurred here, and it was found when CI went red on a document that passes on Windows.

    A receipt is a JSON document. Its meaning does not depend on its line endings, so a hash
    that does is pinning the wrong thing: it makes a certificate platform-dependent, which
    defeats the whole promise that anyone can re-run it. This compares the raw bytes first and
    falls back to both newline normalisations.

    DISCLOSED WEAKENING: two files differing ONLY in line endings now resolve as the same
    receipt. That is intended — they carry identical JSON — but it does mean the sha no longer
    certifies byte-identity, only content-identity-modulo-newlines. The stricter alternative is
    to re-record every certificate's hashes from normalised bytes, which is a corpus migration
    and needs its own preregistration.
    """
    if hashlib.sha256(raw).hexdigest() == recorded:
        return True
    lf = raw.replace(b"\r\n", b"\n")
    if hashlib.sha256(lf).hexdigest() == recorded:
        return True
    return hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest() == recorded


def _resolve_receipts(cert_path: Path, cert: dict,
                      search_root: Path | None = None) -> tuple[list[Path], list[str], list[str]]:
    """Receipt filenames recorded in the cert, resolved to real files.

    Resolution order: (1) next to the doc — the common case; (2) if absent there, search
    ``search_root`` for a file of that name whose sha256 matches the recorded one, on any
    platform's line endings (see ``_receipt_sha_matches``). Step 2 is stricter than
    location-trust: a same-named file only resolves if its CONTENT is what was certified.
    Cross-directory receipts (a synthesis citing arcs from several folders) previously reported
    as ``missing``, which re-certified the document against a crippled receipt set and produced
    spurious OATH-FAILED verdicts on documents whose committed certificates are HELD.

    Returns (existing_paths, missing_names, sha_drifted_names).
    """
    paths, missing, drift = [], [], []
    for name, sha in cert.get("receipts_sha256", {}).items():
        rp = cert_path.parent / name
        if rp.exists():
            if not _receipt_sha_matches(rp.read_bytes(), sha):
                drift.append(name)
            paths.append(rp)
            continue
        found, candidates = None, []
        if search_root is not None:
            for cand in _search(search_root, name):
                try:
                    if _receipt_sha_matches(cand.read_bytes(), sha):
                        found = cand
                        break
                    candidates.append(cand)
                except OSError:
                    continue
        if found is not None:
            paths.append(found)
        else:
            # PRESENT-BUT-CHANGED still reports as `missing`, and that is DELIBERATE.
            #
            # It is tempting to resolve a lone same-named candidate and flag it as drift, so the
            # document stays examinable instead of vanishing from the guard — a version of this
            # function did exactly that for about ten minutes. It is wrong. `tests/test_corpus_audit.py`
            # pins the reason: "a same-named file with DIFFERENT content must NOT satisfy the
            # receipt — the search is stricter than location-trust, not looser." This repository
            # is full of files called `*_result.json`, and resolving one whose content does not
            # match would certify a document against ANOTHER EXPERIMENT'S data while reporting
            # success. That is a worse failure than invisibility, and it is the failure this
            # whole programme exists to prevent.
            #
            # The visibility problem it created was real, and it was fixed WITHOUT touching this
            # strictness: `classify_missing` below says whether an unresolved receipt is genuinely
            # absent or present-and-changed, and `audit_document` reports `incomplete_receipts`,
            # so a verdict computed from a partial receipt set can no longer print like a clean
            # one. A reporting channel, not a resolution change. The live instance that motivated
            # it: CAPSTONE_universal_mind's `mind_v0_validation.json`, present and changed.
            # Catalogued as VP-C in RECON_vacuous_pass_2026_08_27.md.
            missing.append(name)
    return paths, missing, drift


def classify_missing(cert_path: Path, cert: dict, missing: list,
                     search_root: Path | None = None) -> dict:
    """Why is each unresolved receipt unresolved — genuinely absent, or present and CHANGED?

    `_resolve_receipts` deliberately refuses to resolve a same-named file whose content differs
    from what was certified (see the comment there, and
    `tests/test_corpus_audit.py::test_cross_directory_wrong_sha_does_not_resolve`). That
    strictness is correct and stays. What was missing is the ability to SAY WHY, so a changed
    receipt and an absent one were indistinguishable to every caller — which let the audit print
    `receipt-drift 0` over a receipt sitting in the tree with different content.

    This is the reporting channel, not a resolution change. Nothing here decides what gets
    certified; it decides what a reader is told.

      absent      no file of that name under the search root
      changed     exactly one candidate exists and its content is not what was certified
      ambiguous   several candidates exist, none matching — no non-arbitrary choice
    """
    out = {}
    for name in missing:
        cands = []
        if search_root is not None:
            cands = [p for p in _search(search_root, name) if p.is_file()]
        local = cert_path.parent / name
        if local.exists() and local not in cands:
            cands.append(local)
        if not cands:
            status = "absent"
        elif len(cands) == 1:
            status = "changed"
        else:
            status = "ambiguous"
        out[name] = {"status": status,
                     "candidates": [str(p) for p in cands[:4]]}
    return out


def mutate_token(tok: str, rng: random.Random) -> str:
    """Perturb one significant digit, keeping format (the frozen validate_oath_v0 scheme)."""
    digits = [i for i, ch in enumerate(tok) if ch.isdigit()]
    sig = [i for i in digits if not (tok[i] == "0" and (i == 0 or not tok[:i].strip("+-0.")))]
    pos = rng.choice(sig or digits)
    old = int(tok[pos])
    new = rng.choice([d for d in range(10) if d != old])
    return tok[:pos] + str(new) + tok[pos + 1:]


def _doc_for(cert_path: Path) -> Path:
    return cert_path.with_name(cert_path.name.replace(".certificate.json", ".md"))


def _binding_for(repo, cert_path: Path, cert: dict, resolved: dict, doc: Path) -> dict:
    """SPEC_oath_receipt_binding R3 + R4: every citation's cell, and whether the certificate
    stands over the bytes it swore to. Reporting only — no live verdict reads this."""
    try:
        cl = _rb.classify_certificate(repo, cert_path, cert, resolved)
    except _rb.RepoUnavailable as e:
        return {"available": False, "reason": str(e)}
    cl["available"] = True
    cl["stands_over_sworn_bytes"] = None
    try:
        sworn = _rb.sworn_bytes_at_issue(repo, cert_path, cl, resolved)
    except _rb.RepoUnavailable as e:
        sworn = None
        cl["stands_error"] = str(e)[:200]
    if sworn is not None:
        # The bytes at the issuing commit, re-certified under the CURRENT verifier in a scratch
        # directory outside the repository, so the mint-time binding block finds no repo and
        # the verdict is a pure function of these bytes. Class comparison, as everywhere else.
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            d = tdp / doc.name
            d.write_bytes(sworn["document"])
            rps = []
            for name, raw in sworn["receipts"].items():
                (tdp / name).write_bytes(raw)
                rps.append(tdp / name)
            try:
                at_issue = certify_doc(d, rps)
                cl["verdict_over_sworn_bytes"] = at_issue["verdict"]
                cl["stands_over_sworn_bytes"] = (verdict_class(at_issue["verdict"])
                                                 == verdict_class(cert.get("verdict")))
            except Exception as e:   # noqa: BLE001 — a re-derivation failure is a null, not a verdict
                cl["stands_error"] = str(e)[:200]
    return cl


def open_history(root: Path, mode: str = "auto"):
    """SPEC R5. ``(repo, None)`` when history can answer; ``(None, reason)`` when it cannot —
    disabled, no git, not a repository, or a shallow clone (CI checks out at depth 1)."""
    if mode == "off":
        return None, "disabled (--history off)"
    try:
        repo = _rb.Repo(Path(root))
        if repo.shallow:
            repo.close()
            return None, "shallow clone"
        return repo, None
    except _rb.RepoUnavailable as e:
        return None, str(e)


def _fold_binding(docs: list, repo, why) -> dict:
    """Corpus totals for the five cells and the stands-over-sworn-bytes split; the
    `unrecoverable` issuing commit is its own cell, as the plan asked."""
    if repo is None:
        return {"available": False, "reason": why}
    cells = {c: 0 for c in _rb.CELLS}
    stands = {"true": 0, "false": 0, "null": 0}
    n_unrec = n_regen = n_unbacked = n_not_standing = n_with = 0
    for d in docs:
        b = d.get("receipt_binding")
        if not b or not b.get("available"):
            continue
        n_with += 1
        for c, n in b["cells"].items():
            cells[c] += n
        if b.get("issuing_commit") is None:
            n_unrec += 1
        s = b.get("stands_over_sworn_bytes")
        stands["true" if s is True else "false" if s is False else "null"] += 1
        moved = b["cells"]["at_issue"] + b["cells"]["elsewhere"]
        if moved and s is True:
            n_regen += 1
        if b["cells"]["unbacked"]:
            n_unbacked += 1
        if s is False:
            n_not_standing += 1
    return {"available": True, "certificates_classified": n_with, "citations": cells,
            "certificates_issuing_commit_unrecoverable": n_unrec,
            "certificates_receipt_regenerated_and_standing": n_regen,
            "certificates_with_unbacked_citation": n_unbacked,
            "certificates_not_standing_over_sworn_bytes": n_not_standing,
            "stands_over_sworn_bytes": stands,
            "reading": ("a citation's cell says where the bytes the certificate swore to are; "
                        "stands_over_sworn_bytes says whether the current verifier reproduces "
                        "the recorded verdict class over those bytes. Neither is a verdict about "
                        "the document, and false on a same-only certificate is the verifier "
                        "having moved (SKEW), not a binding defect.")}


def audit_document(cert_path: Path, tamper: bool = False, seed: int = 1,
                   search_root: Path | None = None, history=None) -> dict:
    """Re-certify one document under the current verifier. Optionally run the tamper battery.

    *history* is an open ``receipt_binding.Repo`` (or None): with it, the record carries the
    SPEC_oath_receipt_binding cells for every citation; without it the field is absent.
    """
    cert = json.loads(cert_path.read_text(encoding="utf-8"))
    doc = _doc_for(cert_path)
    rec = {"certificate": cert_path.name, "document": doc.name,
           "recorded_verdict": cert.get("verdict")}
    if not doc.exists():
        rec.update(status="MISSING_DOC", live_verdict=None)
        return rec
    receipts, missing, drift = _resolve_receipts(cert_path, cert, search_root)
    rec["receipt_drift"] = drift
    rec["missing_receipts"] = missing
    # WHY each one is missing, and whether this verdict is being computed from partial evidence.
    # A document certified against 11 of its 12 receipts used to print exactly like one certified
    # against all 12; `incomplete_receipts` is what stops that.
    rec["missing_detail"] = classify_missing(cert_path, cert, missing, search_root)
    rec["incomplete_receipts"] = bool(missing) and bool(receipts)
    rec["receipt_changed"] = sorted(n for n, d in rec["missing_detail"].items()
                                    if d["status"] == "changed")
    # SPEC_oath_receipt_binding: computed BEFORE the no-receipts return, because a certificate
    # whose receipts have all gone is exactly the one whose binding history matters most.
    if history is not None:
        rec["receipt_binding"] = _binding_for(history, cert_path, cert,
                                              {p.name: p for p in receipts}, doc)
    if not receipts:
        rec.update(status="NO_RECEIPTS", live_verdict=None)
        return rec
    live = certify_doc(doc, receipts)
    rec["live_verdict"] = live["verdict"]
    rec["live_verdict_class"] = verdict_class(live["verdict"])
    rec["counts"] = live["counts"]
    # v0.13: the count of numeric spans the extractor never examined. Reporting only — it is
    # carried as its own field so a corpus total can be printed WITHOUT reading it as a verdict.
    rec["uncovered"] = int(live.get("uncovered", 0) or 0)
    # The auditor reads the verifier's own epistemics_summary rather than re-deriving anything;
    # it may only ADD corpus totals, never touch a verdict or an existing count.
    rec["epistemics_summary"] = live.get("epistemics_summary")
    # Drift is a change of CLASS. A committed `OATH-HELD` re-issued today as `OATH-HELD, 2
    # uncovered` has not stopped holding; the suffix is new information, not a moved verdict.
    rec["verdict_changed"] = (verdict_class(live["verdict"]) != verdict_class(cert.get("verdict")))
    rec["status"] = "OK"
    if tamper:
        rng = random.Random(seed)
        text = doc.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        verified = [e for e in live["ledger"] if e["status"] == "VERIFIED"]
        caught = fv = ad = dropped = 0
        for e in verified:
            li = e["line"] - 1
            if li >= len(lines) or e["token"] not in lines[li]:
                dropped += 1
                continue
            mut = mutate_token(e["token"], rng)
            ml = list(lines)
            ml[li] = lines[li].replace(e["token"], mut, 1)
            with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False,
                                             encoding="utf-8") as tf:
                tf.write("".join(ml))
                tmp = Path(tf.name)
            try:
                mc = certify_doc(tmp, receipts)
            finally:
                tmp.unlink(missing_ok=True)
            st = next((x["status"] for x in mc["ledger"]
                       if x["line"] == e["line"] and x["token"] == mut), None)
            if st == "UNGROUNDED":
                caught += 1
            elif st == "VERIFIED":
                fv += 1
            elif st == "ABSTAIN":
                ad += 1
            else:
                dropped += 1
        rec["tamper"] = {"n_mutants": len(verified) - dropped, "caught": caught,
                         "false_verify": fv, "abstain_degrade": ad, "dropped": dropped}
    return rec


def _fold_epistemics(docs: list) -> dict:
    """Corpus-wide composition, summed from every certificate's own epistemics_summary.

    Answers, over a whole corpus, the question a single certificate now answers per token: of what
    this corpus swears to, how much was the verifier ever OBLIGATED to examine, and how much is the
    weakest form -- a volunteered value coincidence no binding filter touched. Additive: reads the
    summary, sums it, invents nothing. A certificate predating epistemics_summary v1 contributes
    nothing (older certificates carry no block) and is counted in `certificates_without_summary`.
    """
    ver = 0
    obligated_ver = 0
    weakest = 0            # unobligated value-match, no integer filter
    accusations = 0
    without = 0
    for d in docs:
        s = d.get("epistemics_summary")
        if not s or s.get("schema") != "styxx-oath/epistemics-summary/v1":
            without += 1
            continue
        vm = s["verified"]["value_match"]
        der = s["verified"]["derived"]
        ver += s["verified"]["total"]
        obligated_ver += (vm["obligated_integer_filter_ran"] + vm["obligated_integer_filter_na"]
                          + der["obligated"])
        weakest += vm["unobligated_integer_filter_na"]
        accusations += s["by_branch"]["obligated-accusation"]
    unobligated = ver - obligated_ver
    return {
        "certificates_with_summary": len(docs) - without,
        "certificates_without_summary": without,
        "verified_total": ver,
        "verified_obligated": obligated_ver,
        "verified_unobligated": unobligated,
        "unobligated_oath_rate": round(unobligated / ver, 4) if ver else None,
        "weakest_attestations": weakest,
        "weakest_share": round(weakest / ver, 4) if ver else None,
        "accusations": accusations,
        "reading": ("unobligated_oath_rate is the share of this corpus's VERIFIED tokens that "
                    "nothing obligated the verifier to examine; weakest_share is the subset that "
                    "is value-match alone with no binding filter. Both are composition, not "
                    "quality -- claim-share is the panel's job, not the auditor's."),
    }


def audit_corpus(root: Path, tamper: bool = False, seed: int = 1,
                 history: str = "auto") -> dict:
    """Audit every certificate under *root*; return per-doc records + a corpus summary.

    *history* is ``auto`` (binding cells when the root sits in a full git clone), ``on`` (the
    same, but the reason is reported if it cannot) or ``off``. The summary's verdict counters
    are identical in every mode — SPEC_oath_receipt_binding R5.
    """
    repo, why = open_history(root, history)
    try:
        docs = [audit_document(cp, tamper, seed, search_root=root, history=repo)
                for cp in discover_certificates(root)]
    finally:
        if repo is not None:
            repo.close()
    held = sum(1 for d in docs if verdict_class(d.get("live_verdict")) == "OATH-HELD")
    failed = sum(1 for d in docs if verdict_class(d.get("live_verdict")) == "OATH-FAILED")
    unresolved = sum(1 for d in docs if d.get("status") in ("MISSING_DOC", "NO_RECEIPTS"))
    changed = sum(1 for d in docs if d.get("verdict_changed"))
    drifted = sum(1 for d in docs if d.get("receipt_drift"))
    # A verdict computed from a partial receipt set is not a clean verdict, and until these two
    # counters existed it printed like one.
    incomplete = sum(1 for d in docs if d.get("incomplete_receipts"))
    changed_receipts = sum(1 for d in docs if d.get("receipt_changed"))
    # v0.13 UNCOVERED, folded as corpus totals beside the verdict counts and never into them.
    uncovered_docs = sum(1 for d in docs if d.get("uncovered"))
    uncovered_spans = sum(int(d.get("uncovered") or 0) for d in docs)
    summary = {"root": str(root), "n_certificates": len(docs), "held": held, "failed": failed,
               "unresolved": unresolved, "verdict_changed": changed, "receipt_drift": drifted,
               "incomplete_receipts": incomplete, "receipt_changed": changed_receipts,
               "uncovered_documents": uncovered_docs, "uncovered_spans": uncovered_spans,
               "epistemics": _fold_epistemics(docs),
               "binding": _fold_binding(docs, repo, why)}
    if tamper:
        tot = {"n_mutants": 0, "caught": 0, "false_verify": 0, "abstain_degrade": 0}
        for d in docs:
            for k in tot:
                tot[k] += d.get("tamper", {}).get(k, 0)
        n = max(tot["n_mutants"], 1)
        summary["tamper"] = {**tot, "catch_rate": round(tot["caught"] / n, 3),
                             "false_verify_rate": round(tot["false_verify"] / n, 3)}
    return {"summary": summary, "documents": docs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="styxx.corpus_audit",
                                 description="Re-certify a corpus of OATH certificates under the current verifier.")
    ap.add_argument("root", nargs="?", default=".", help="directory to scan (default: cwd)")
    ap.add_argument("--tamper", action="store_true", help="also run the single-digit tamper battery")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json", default=None, help="write the full report to this path")
    ap.add_argument("--history", choices=("auto", "on", "off"), default="auto",
                    help="receipt-binding cells from git history (SPEC_oath_receipt_binding); "
                         "auto = when the root is in a full clone; the verdict line never changes")
    a = ap.parse_args(argv)
    report = audit_corpus(Path(a.root), tamper=a.tamper, seed=a.seed, history=a.history)
    s = report["summary"]
    print(f"corpus {s['root']}: {s['n_certificates']} certificates | "
          f"HELD {s['held']}  FAILED {s['failed']}  unresolved {s['unresolved']}  "
          f"verdict-drift {s['verdict_changed']}  receipt-drift {s['receipt_drift']}  "
          f"incomplete {s['incomplete_receipts']}  receipt-changed {s['receipt_changed']}")
    # v0.13: the corpus-wide UNCOVERED total on its own line, beside the verdict line and never
    # inside it. The verdict line is what REPLICATIONS.md pins; this line is what it cannot see.
    print(f"  uncovered: {s.get('uncovered_spans', 0)} numeric spans across "
          f"{s.get('uncovered_documents', 0)} documents that the extractor never examined "
          f"(reporting only; no verdict moved)")
    # SPEC_oath_receipt_binding R5: the binding line sits beneath the uncovered line and beside
    # the verdict line, never inside it; when history cannot answer, the reason is printed.
    b = s.get("binding") or {}
    if not b.get("available"):
        print(f"  binding: history unavailable ({b.get('reason', '?')})")
    else:
        c = b["citations"]
        st = b["stands_over_sworn_bytes"]
        print(f"  binding: citations same {c['same']}  at-issue {c['at_issue']}  "
              f"elsewhere {c['elsewhere']}  unbacked {c['unbacked']}  "
              f"unrecoverable {c['unrecoverable']} | certificates: "
              f"issuing-commit-unrecoverable {b['certificates_issuing_commit_unrecoverable']}  "
              f"regenerated-and-standing {b['certificates_receipt_regenerated_and_standing']}  "
              f"unbacked {b['certificates_with_unbacked_citation']}  "
              f"stands-over-sworn-bytes {st['true']}/{st['true'] + st['false']} "
              f"(null {st['null']})")
    ep = s.get("epistemics", {})
    if ep.get("verified_total"):
        print(f"  epistemics: {ep['verified_total']} verified | "
              f"obligated {ep['verified_obligated']} unobligated {ep['verified_unobligated']} "
              f"(rate {ep['unobligated_oath_rate']}) | weakest {ep['weakest_attestations']} "
              f"({ep['weakest_share']}) | {ep['certificates_without_summary']} pre-v1")
    for d in report["documents"]:
        if (verdict_class(d.get("live_verdict")) == "OATH-FAILED" or d.get("verdict_changed")
                or d.get("receipt_drift") or d.get("incomplete_receipts")):
            # the CLASS is the tag: the exception list is pinned by class in REPLICATIONS.md, and
            # the coverage suffix is reported on its own line above, not smuggled into the tag.
            tag = verdict_class(d.get("live_verdict")) or d.get("status")
            extra = " receipt-drift" if d.get("receipt_drift") else ""
            extra += " verdict-CHANGED" if d.get("verdict_changed") else ""
            if d.get("incomplete_receipts"):
                det = d.get("missing_detail", {})
                why = ",".join(sorted({v["status"] for v in det.values()})) or "?"
                extra += f" INCOMPLETE-RECEIPTS({why})"
            print(f"  [{tag}]{extra}  {d['document']}")
    if a.tamper and "tamper" in s:
        t = s["tamper"]
        print(f"tamper-catch: {t['caught']}/{t['n_mutants']} = {t['catch_rate']}  "
              f"(false-verify {t['false_verify']} = {t['false_verify_rate']})")
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"-> {a.json}")
    return 1 if s["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
