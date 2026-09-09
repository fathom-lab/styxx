"""Demonstrate the cross-certificate predicate that reaches class-two member 1.

THE CLAIM UNDER ATTACK (THE_BOUNDARY_2026_09_09.md, roster member 1):

    five copies of one forward pass with five batch labels give executions = 5,
    pairs_same_execution = 0 and a floor of 0.0 -- and five real runs that happened to agree
    exactly give the same bytes. The two artifacts are identical. No predicate over them can
    differ.

Both halves of that sentence are true OF ONE CERTIFICATE. Neither is true of a certificate IN A
LOG that already contains a measurement of the same subject on the same battery, which is the
situation the published verdict log is actually in.

WHAT THIS SCRIPT DOES, on the lab's own published bytes and nothing else:

  control A   the honest published floor, checked against itself. Must pass.
  attack      the exact forgery the roster describes, built by collapsing the five real run
              bodies to one and keeping the five real labels. Must be refused.
  control B   an honest floor from a subject with NO logged history. Must be neither passed nor
              refused -- it must be reported as unconstrained, because there is nothing to
              contradict, and a check that answers when it has no evidence is the defect this
              lab already has a receipt for.

The predicate needs no styxx import. It reads stored entry bytes.
"""
import glob
import itertools
import json
import pathlib

LOGDIR = pathlib.Path(
    r"C:\Users\heyzo\clawd\wt\v8\papers\v8\first_verdict_2026_09_09\log\entries"
)


def load_fingerprints(logdir):
    out = []
    for f in sorted(glob.glob(str(logdir / "*" / "*[0-9].json"))):
        c = json.loads(pathlib.Path(f).read_text(encoding="utf-8"))
        if c.get("type") == "fingerprint":
            c["_entry"] = pathlib.Path(f).stem
            out.append(c)
    return out


SUBJECT_FIELDS = ("kind", "model_family", "hf_repo", "revision", "precision",
                  "weights_sha256", "config_sha256", "generation_config_sha256",
                  "tokenizer_sha256")
RECIPE_FIELDS = ("battery", "chat_template_sha256", "system_prompt_sha256")
DECODING_FIELDS = ("max_new_tokens", "temperature", "top_p", "seed", "stop")


def subject_key(cert):
    """What makes two floors comparable: the same thing measured the same way.

    This approximates section 2.3. It is NOT the authority -- the independent JavaScript verifier
    implements 2.3 properly, through the comparability predicate it already had for challenges,
    and this list exists so the receipt does not have to import styxx.

    It was narrower on the first two attempts. Attempt one never called this function at all.
    Attempt two called it but keyed on four fields not including `precision`, so a floor for the
    same weights at a different precision compared as the same subject -- which is exactly the
    comparison the published verdict treats as a DRIFT CANDIDATE. Widening it is not a refinement;
    the narrow key was wrong.
    """
    s = cert.get("subject", {})
    r = cert.get("recipe", {})
    d = r.get("decoding", {}) or {}
    return (
        tuple(json.dumps(s.get(f), sort_keys=True) for f in SUBJECT_FIELDS)
        + tuple(json.dumps(r.get(f), sort_keys=True) for f in RECIPE_FIELDS)
        + tuple(json.dumps(d.get(f), sort_keys=True) for f in DECODING_FIELDS)
    )


def observed_pairs(cert, runs_by_index):
    """(unordered batch pair) -> distance, read off a floor cert and the runs it names.

    The floor stores distances in run_index pair order. This reconstructs which declared
    assignment each distance belongs to, which is the whole point: the number is only evidence
    about a factor if you know which factor level it separated.
    """
    floor = cert["body"]["noise_floor"]
    order = sorted(runs_by_index)
    pairs = list(itertools.combinations(order, 2))
    out = {}
    for ch, d in floor["per_channel"].items():
        dist = d["distances"]
        if len(dist) != len(pairs):
            continue
        for (i, j), v in zip(pairs, dist):
            bi = runs_by_index[i]["body"]["nuisance"]["batch_size"]
            bj = runs_by_index[j]["body"]["nuisance"]["batch_size"]
            out.setdefault(ch, {}).setdefault(tuple(sorted((bi, bj))), set()).add(v)
    return out


def comparable(candidate, prior):
    """May this prior floor speak about this candidate at all?

    THE BUG THIS FIXES, and it was found by a second implementation rather than by its author.
    The first version of this script defined `subject_key` and then never called it, so the
    predicate would accept a prior floor from ANY subject in the log and accuse on it -- a floor
    for a different model, or the same weights at a different precision, would have contradicted
    a candidate that had done nothing wrong. That is a false accusation across subjects, which is
    the defect class this lab already has a receipt for at 0.23 precision.

    The independent JavaScript verifier gated comparability first, through the section 2.3
    predicate it already had, and its control D -- a prior floor on a different subject -- returns
    `unconstrained`. Under the unfixed Python that same prior contradicted. Two implementations
    disagreed and the other one was right.
    """
    a, b = subject_key(candidate), subject_key(prior)
    if a == b:
        return True, []
    names = (tuple(f"subject.{f}" for f in SUBJECT_FIELDS)
             + tuple(f"recipe.{f}" for f in RECIPE_FIELDS)
             + tuple(f"recipe.decoding.{f}" for f in DECODING_FIELDS))
    assert len(names) == len(a), "the name list and the key have drifted apart"
    return False, [n for n, x, y in zip(names, a, b) if x != y]


def predicate(candidate_pairs, prior_pairs):
    """Does a candidate floor contradict a prior floor on the same subject and battery?

    Contradiction, not disagreement in magnitude: the prior says a declared factor level pair
    SEPARATES this subject (distance > 0), and the candidate says it does not (distance == 0).
    A different positive value is not a contradiction -- machines drift, and a rule that demanded
    equality would refuse honest re-measurement. A zero where the log holds a positive is a claim
    that the factor does nothing, against a logged measurement that it does.

    Returns (verdict, reasons). verdict is one of: agrees, contradicts, unconstrained.
    """
    reasons = []
    compared = 0
    for ch, bp in candidate_pairs.items():
        if ch not in prior_pairs:
            continue
        for key, vals in bp.items():
            if key not in prior_pairs[ch]:
                continue
            compared += 1
            prior = prior_pairs[ch][key]
            if all(v == 0 for v in vals) and any(p > 0 for p in prior):
                reasons.append(
                    f"channel {ch}: batch pair {key[0]} vs {key[1]} measured "
                    f"{sorted(prior)} in the prior floor, {sorted(vals)} here"
                )
    if compared == 0:
        return "unconstrained", ["no prior floor on this subject exercised these factor levels"]
    return ("contradicts" if reasons else "agrees"), reasons or [
        f"{compared} factor-level pairs compared, none contradicted"
    ]


# ----------------------------------------------------------------------------- the real log

certs = load_fingerprints(LOGDIR)
runs = {c["body"]["run_index"]: c for c in certs if "run_index" in c["body"]}
canonical = next(c for c in certs if "noise_floor" in c["body"])
prior = observed_pairs(canonical, runs)

print("PUBLISHED FLOOR  entry", canonical["_entry"])
print("  subject key:", subject_key(canonical)[0][:24], "...")
for ch, bp in prior.items():
    print(f"  {ch}: " + "  ".join(f"{k[0]}v{k[1]}={sorted(v)}" for k, v in sorted(bp.items())))

print("\n--- control A: the honest floor against itself")
v, why = predicate(prior, prior)
print(f"  verdict: {v}")
for r in why:
    print("   ", r)
assert v == "agrees", "the honest floor must not accuse itself"

# ------------------------------------------------------------------- the roster's own forgery

print("\n--- attack: the forgery the roster calls byte-indistinguishable")
print("  built by giving all five runs run 0's body and keeping their five real labels.")
print("  every pairwise distance is then 0 for any metric, since d(x, x) = 0 -- no")
print("  recomputation needed and none is done here.")

forged_runs = {}
for i, r in runs.items():
    c = json.loads(json.dumps(r))
    c["body"]["channels"] = json.loads(json.dumps(runs[0]["body"]["channels"]))
    c["body"]["items"] = json.loads(json.dumps(runs[0]["body"]["items"]))
    forged_runs[i] = c   # labels in body['nuisance'] left exactly as published

forged = json.loads(json.dumps(canonical))
order = sorted(forged_runs)
npairs = len(list(itertools.combinations(order, 2)))
for ch, d in forged["body"]["noise_floor"]["per_channel"].items():
    d["distances"] = [0] * npairs
    d["floor"] = 0

# The within-certificate checks the roster says cannot tell these apart.
labels = [json.dumps(forged_runs[i]["body"]["nuisance"], sort_keys=True) for i in order]
print(f"  distinct nuisance labels : {len(set(labels))} of {len(labels)}")
print(f"  plan declared runs       : 5   cert carries: {len(order)}")
print(f"  every declared factor varies across the labels: "
      f"{len({json.loads(l)['batch_size'] for l in labels}) > 1}")
print("  -> within the certificate, this is the honest artifact's shape")

forged_pairs = observed_pairs(forged, forged_runs)
v, why = predicate(forged_pairs, prior)
print(f"  verdict against the prior logged floor: {v}")
for r in why:
    print("   ", r)
assert v == "contradicts", "the forgery must be refused"

# --------------------------------------------------------- control B: nothing to contradict

print("\n--- control B: the same forgery, on a subject the log has never measured")
fresh = {ch: {} for ch in prior}
v, why = predicate(forged_pairs, fresh)
print(f"  verdict: {v}")
for r in why:
    print("   ", r)
assert v == "unconstrained", "with no prior measurement the predicate must decline, not accuse"

# --------------------------------------------------- control D: a prior about a DIFFERENT subject

print("\n--- control D: an HONEST floor, judged against a prior about a different subject")
print("  this control did not exist until the independent JavaScript implementation added it and")
print("  the omission turned out to be a bug here, not a difference of taste.")
other = json.loads(json.dumps(canonical))
other["subject"] = dict(other["subject"], precision="fp16")
ok, differing = comparable(other, canonical)
print(f"  comparable: {ok}" + ("" if ok else f"   differs on: {', '.join(differing)}"))
if not ok:
    v, why = "unconstrained", [f"the prior is not comparable: differs on {', '.join(differing)}"]
else:
    v, why = predicate(observed_pairs(other, runs), prior)
print(f"  verdict: {v}")
for r in why:
    print("   ", r)
assert v == "unconstrained", (
    "a prior about another subject must not be allowed to accuse this one. Before the gate was "
    "wired in, this returned a contradiction."
)

print("\n" + "=" * 78)
print("The roster's member 1 is reachable BY A PREDICATE OVER LOGGED BYTES.")
print("What it does not reach is control B: the first claim about a subject, where no prior")
print("measurement exists to contradict. That residue is not a field an issuer writes. It is")
print("the act of claiming something for the first time.")
