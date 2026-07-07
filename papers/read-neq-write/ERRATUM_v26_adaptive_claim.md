# ERRATUM — Fathom v26 (DOI 10.5281/zenodo.21241185): the "adaptive attacker" claim is demoted

**Filed 2026-07-07, same day as publication, by the program's own pre-committed verification.**

## What the published paper says

The v26 abstract and §5.4 state that the private-calibration defense "survives an *adaptive* attacker
that re-fits a moving probe on its own clean split and scrubs along it (clean-calibrated read
0.733–0.782, two model families)." §7.1 flagged, as a load-bearing caveat, that the adaptive
attacker's extra bite over naive was *unmeasured*.

## What E3′ found

`PREREG_E3PRIME_bite_verification_2026_07_07.md` (frozen before its run) measured the bite.
Verdict: **NO_BITE** (`RESULT_E3PRIME_bite_verification_2026_07_07.md`, OATH-HELD). The adaptive
scheme did not out-attack naive — on its own target read it left 0.195 *more* AUROC legible than
naive did. The §7.1 caveat was not just prudent; it was correct.

## The correction (binding on the next version)

- **Stands, unchanged:** the private-calibration defense beats the *naive*, knowledge-preserving
  weight attacker (E1 clean-calibrated read 0.711–0.838; E3′ naive control 0.775–0.786). The paper's
  central result — apparent evasion was calibration poisoning, not erasure — is intact.
- **Demoted:** "survives an adaptive attacker" is not earned. The tested adaptive scheme was not a
  stronger attack than naive. The honest statement is: "survives the naive knowledge-preserving
  attacker; the adaptive scheme tested (refit-every-25, deploy-layer scrub) did not out-attack naive
  (E3′), so robustness to an adaptive escalation is open."
- **Owed:** E3″, a genuinely stronger adaptive scheme, frozen before its run.

## Disposition

The next paper version (v0.2 / Fathom v27) will fold in this correction, the E2 INCONCLUSIVE result,
and the E2′ verdict, and be deposited as a new Zenodo version under concept DOI
10.5281/zenodo.19326174. Until then, cite v26 *with this erratum*. The published record is not
silently edited; it is versioned with the correction attached, which is the point.
