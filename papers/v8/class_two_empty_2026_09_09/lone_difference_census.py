"""Which fields can differ ALONE? A mechanical way to build the roster that was built by hand.

THE CRITICISM THIS ANSWERS. `THE_BOUNDARY_2026_09_09.md` kept a roster of defects it called
unreachable, all four entries were wrong, and then a seventh pass found a FIFTH entry the roster
never listed at all: `precision`. That is a worse failure than four wrong entries, because the
method used -- an author deciding by inspection -- can only interrogate defects someone already
thought to write down. Nothing about testing the entries finds a missing one.

THE TEST, and it needs no judgement. A field has a corroborating byte when forging it would force
some OTHER byte in the same certificate to change. So: find two real certificates that differ in
exactly one field. If such a pair exists, that field moved and nothing else did, which is a
demonstration on real bytes that the field carries no internal corroboration. No inspection, no
opinion, no list of defects anyone thought of first.

`precision` is the worked case. The published bf16 and fp16 subjects share `weights_sha256`,
`config_sha256`, `tokenizer_sha256`, `generation_config_sha256`, `revision`, `hf_repo`,
`model_family`, `kind` and `environment`. They differ in one string. Excluding it from the A.2
snapshot-agreement predicate is CORRECT -- a rule that hashed it would refuse that honest pair --
and the consequence is that nothing anywhere can contradict a forged value.

WHAT THIS DOES NOT ESTABLISH, stated first because the same document has been wrong six times in
the other direction. A field with no witnessing pair in this corpus is not thereby corroborated: it
may simply never have varied here. This census reports **demonstrated** lone differences and a
separate list of fields that never varied at all, and it calls the second list unknown rather than
safe. It is a lower bound on the roster, never an upper one.

Corpus: every certificate in the published verdict log, plus the standalone fingerprints beside it.
"""

import glob
import itertools
import json
import pathlib

BASE = pathlib.Path(r"C:\Users\heyzo\clawd\wt\v8\papers\v8\first_verdict_2026_09_09")
LOG = BASE / "log" / "entries"


def load_certs():
    """Every cert on disk in this arc, from the log and from the loose fingerprint files."""
    out = []
    for f in sorted(glob.glob(str(LOG / "*" / "*[0-9].json"))):
        out.append((f"log:{pathlib.Path(f).stem}", json.loads(pathlib.Path(f).read_text("utf-8"))))
    for sub in ("fp_bf16", "fp_fp16"):
        for f in sorted(glob.glob(str(BASE / sub / "*.json"))):
            out.append((f"{sub}:{pathlib.Path(f).stem[:28]}",
                        json.loads(pathlib.Path(f).read_text("utf-8"))))
    return out


def flatten(obj, prefix=""):
    """Leaf paths to canonical values. Lists are compared whole: an ordering is one fact."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else k))
    else:
        out[prefix] = json.dumps(obj, sort_keys=True)
    return out


# Blocks that are supposed to differ between any two certificates, so a difference there says
# nothing about corroboration. `id` and `sig` are functions of everything else by construction.
IGNORED_PREFIXES = ("id", "sig", "created", "body", "refs", "issuer", "styxx")

# Which object a path belongs to. Only identity-bearing objects are interesting: these are the
# fields that say WHAT WAS MEASURED, and a forged one misdescribes the measurement.
INTERESTING = ("subject", "recipe")


def interesting(path):
    if path.split(".")[0] in IGNORED_PREFIXES:
        return False
    return path.split(".")[0] in INTERESTING


certs = load_certs()
print(f"{len(certs)} certificates on disk in this arc\n")

flat = {name: {k: v for k, v in flatten(c).items() if interesting(k)} for name, c in certs}

lone = {}          # field -> list of (a, b) pairs differing in that field alone
varied = set()     # every field seen to differ in any pair at all
seen = set()       # every field present anywhere

for name, f in flat.items():
    seen.update(f)

for (na, fa), (nb, fb) in itertools.combinations(flat.items(), 2):
    keys = set(fa) | set(fb)
    diff = [k for k in keys if fa.get(k) != fb.get(k)]
    varied.update(diff)
    if len(diff) == 1:
        lone.setdefault(diff[0], []).append((na, nb))

print("=" * 78)
print("DEMONSTRATED LONE DIFFERENCES -- a field that moved while nothing else did.")
print("Each is a field with no corroborating byte inside the certificate, shown on real bytes.")
print("=" * 78)
if not lone:
    print("  none in this corpus")
for field, pairs in sorted(lone.items()):
    print(f"\n  {field}")
    print(f"    witnessed by {len(pairs)} pair(s), e.g. {pairs[0][0]}  vs  {pairs[0][1]}")
    a, b = pairs[0]
    print(f"      {flat[a].get(field)}  vs  {flat[b].get(field)}")

print("\n" + "=" * 78)
print("VARIED, BUT NEVER ALONE -- moved only alongside other fields in this corpus.")
print("Not evidence of corroboration; evidence that this corpus never separated them.")
print("=" * 78)
rest = sorted(varied - set(lone))
print("  " + (", ".join(rest) if rest else "none"))

print("\n" + "=" * 78)
print("NEVER VARIED -- constant across every certificate here. UNKNOWN, not safe.")
print("=" * 78)
never = sorted(seen - varied)
print(f"  {len(never)} field(s): " + ", ".join(never[:14]) + (" ..." if len(never) > 14 else ""))

print(f"""
READING
  {len(lone)} field(s) are demonstrated to carry no internal corroboration, by a test that needed
  no roster and no author's judgement: two real certificates differ there and nowhere else.

  THE_BOUNDARY's four-member roster was assembled by inspection and missed `precision`. This census
  is how the roster should have been built. It is a LOWER bound -- {len(never)} fields never varied
  in this corpus and are therefore unknown, not cleared -- and the honest use of it is as a
  generator of candidates for the next pass, never as a completeness claim. This document has made
  six completeness claims today and withdrawn all six.

  Note what the census does NOT say about the fields it names. A field with no internal
  corroboration may still be constrained from outside: `precision` is checkable by anyone who
  re-runs the battery and compares outputs, which is the challenge, which is the only part of this
  design that introduces a byte the issuer did not write.""")
