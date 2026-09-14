# CORRECTION — RESULT_portability_smollm_quant_2026_09_13: what a two-machine spread is, and what the instrument's first version got wrong

**status: a correction beside a sworn document, written the night the document's instrument was
red-teamed. The RESULT is not edited and its receipt stands over the bytes it swore to; every number
it binds still resolves. What follows corrects its prose and names the instrument's repairs. Not
sworn: it corrects sentences, not numbers.**

## the prose

1. The RESULT calls the per-number spread "the tolerance a RESULT may state" and then reads a re-run
   inside it as a replication and outside it as a challenge. Both cannot hold, and the second is
   withdrawn. One pairwise difference from two machines is a single sample of the spread — a floor on
   the tolerance, never a ceiling — and a third machine that lands outside it has widened the floor,
   not failed. A challenge under `BOUNTY.md` is decided at the **verdict** level: a stranger whose
   re-run flips a verdict holds one; a stranger whose magnitudes land anywhere does not, on that
   ground alone. The instrument's own reading string now says so.
2. "No magnitude survives the move" is true of the quantized and random arms and false of the null
   pair, whose mean distance moved by 0 and whose geometry leaves moved by at most 2.2e-16 — within
   the floor. The sworn span "the null pair's distance moved by 0" binds `mean_abs_nats`, which is
   what the RESULT's sentence names; it is not a statement about every leaf, and the title's "no
   magnitude does" should be read as "no magnitude of the arms that changed weights does".

## the instrument, v0 to v1 (`styxx.portability`)

The red team of 2026-09-13 (findings portability-1/2/3/6/7/8, none refuted) found, on v0:

- an absent or non-finite number on one machine was read as a number that survived — `_spread`
  dropped it and the MOVE test never saw it; v1 reads it as UNCOMPARABLE and says so;
- the base arm chose the verdict: `--base-arm Q` on the lab's own pair turned MOVE into
  WITHIN-FLOOR; v1 refuses a base arm that is not in the fingerprints or is not the `a` side of every
  cert that names its side, and says when it cannot verify that (v0 certs carry no `a.rdm_sha256`);
- the cross-machine floor was the per-item **max** while `checksum.null_floor` is the per-item
  **mean**; v1 grades with the mean and reports the max beside it — on the lab's pair the mean is
  8.2e-6, still below the resolution, so the grading floor and the reading are unchanged;
- the digest covered the labels and not the inputs; v1 digests the input certs' digests and the
  fingerprints' written hashes and leaves labels and paths outside;
- fingerprints were not bound to the certs (a wrong canary set or a wrong item count passed silently);
  v1 refuses both;
- interval endpoints were graded against a per-item floor; v1 reports their spreads and grades none
  of them, because a bootstrap percentile is not a point;
- a missing verdict compared as `None == None`; v1 refuses it. Arms present on one machine only are
  listed, not hidden. The CLI's usage line named options that did not exist; it now matches.

The v0 record the RESULT swears to (`portability_smollm_quant_two_machines_2026_09_13.json`,
schema `styxx.portability/v0`) stays as committed. A v1 record over the same pair is committed
beside it and reads the same: verdicts AGREE, magnitudes MOVE, base-arm binding unverified because
the certs are v0.
