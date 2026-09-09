"""Is a FIRST claim really unconstrained? Test the residue against the published bytes.

THE_BOUNDARY's third correction empties class two and says the residue is a single unreachable
act: the first claim about a subject, because nothing logged can contradict it. That is an
impossibility claim, and this document's author has made four of those today and been wrong four
times. This script tries to be wrong a fifth time on purpose.

The attack on the residue: a floor's distance matrix is not free. On the `exact` channel the
distance between two runs is the fraction of the battery's items whose outputs differ. Any such
matrix must be REALIZABLE -- there must exist an actual assignment of outputs to items that
produces it. That is a constraint on one certificate alone, needing no prior log at all.

Three nested conditions, weakest first:

  1. METRIC        symmetry, zero diagonal, triangle inequality.
  2. GRANULARITY   every exact distance is k/64 for an integer k, because the battery has 64
                   items and a per-item disagreement count cannot be fractional.
  3. REALIZABLE    the matrix must be a non-negative integer combination of CUT semimetrics
                   totalling at most 64. Each item of the battery partitions the five runs into
                   groups that agreed on it; a partition contributes its cut to the count matrix.
                   This is membership in the cut cone -- equivalently, exact embeddability in
                   l1 -- and for five points it is a small exact feasibility problem, solved here
                   by integer search over the 15 non-trivial cuts rather than numerically.

If the published matrix satisfies all three and a plausible forgery can fail them, then a first
claim IS constrained by an internal predicate, the residue as written is too large, and the
correction is wrong in the same direction as the four errors before it.
"""
import glob
import itertools
import json
import pathlib

LOG = pathlib.Path(
    r"C:\Users\heyzo\clawd\wt\v8\papers\v8\first_verdict_2026_09_09\log\entries"
)
ITEMS = 64   # the battery size, stated in the published RESULT and checked below


def load():
    runs, canonical = {}, None
    for f in sorted(glob.glob(str(LOG / "*" / "*[0-9].json"))):
        c = json.loads(pathlib.Path(f).read_text(encoding="utf-8"))
        if c.get("type") != "fingerprint":
            continue
        if "run_index" in c["body"]:
            runs[c["body"]["run_index"]] = c
        if "noise_floor" in c["body"]:
            canonical = c
    return runs, canonical


def matrix(distances, n):
    pairs = list(itertools.combinations(range(n), 2))
    assert len(distances) == len(pairs), f"{len(distances)} distances, {len(pairs)} pairs"
    m = [[0] * n for _ in range(n)]
    for (i, j), v in zip(pairs, distances):
        m[i][j] = m[j][i] = v
    return m


def is_metric(m):
    n = len(m)
    bad = []
    for i in range(n):
        if m[i][i] != 0:
            bad.append(f"d({i},{i}) = {m[i][i]}")
    for i, j, k in itertools.permutations(range(n), 3):
        if m[i][j] > m[i][k] + m[k][j] + 1e-12:
            bad.append(f"triangle: d({i},{j})={m[i][j]} > d({i},{k})+d({k},{j})"
                       f"={m[i][k] + m[k][j]}")
    return (not bad), bad


def tight_triangles(m):
    """Distinct triangle identities that hold exactly.

    THE BUG THIS FIXES, found by an adversary and not by its author. The first version walked the
    orientations ``((i,j,k), (i,k,j), (j,i,k))``. On a symmetric matrix ``(j,i,k)`` tests the same
    condition as ``(i,j,k)`` -- both ask whether k lies between the other two -- so every apex-k
    identity was counted twice, while the apex-i identity (the one about side (j,k)) was never
    tested at all. It printed 21, 17, 17 for the three channels. The distinct identities that
    actually hold are 18, 15, 15 of 30, with nine double-counted and ten never checked.

    The axioms held then and hold now; the COUNT was wrong, and a count nobody could reproduce is
    the kind of number this lab has a standing rule against publishing.

    A triple has exactly three identities, one per apex. That is what is enumerated here.
    """
    n = len(m)
    out = []
    for tri in itertools.combinations(range(n), 3):
        for apex in tri:
            a, b = [x for x in tri if x != apex]
            if abs(m[a][b] - (m[a][apex] + m[apex][b])) < 1e-15:
                out.append(f"d({a},{b}) = d({a},{apex}) + d({apex},{b}) exactly")
    return out


def triangle_identity_count(n):
    """How many distinct identities exist to hold: one per apex per triple."""
    return len(list(itertools.combinations(range(n), 3))) * 3


def cuts(n):
    """The 2^(n-1) - 1 non-trivial cut semimetrics on n points, as pair-indexed vectors."""
    pairs = list(itertools.combinations(range(n), 2))
    out = []
    for r in range(1, n):
        for S in itertools.combinations(range(1, n), r):
            s = set(S)
            out.append((frozenset(s), tuple(1 if ((i in s) != (j in s)) else 0
                                            for i, j in pairs)))
    return pairs, out


def realizable(counts, n, budget):
    """Is the integer count matrix a non-negative integer sum of cuts with total <= budget?

    Exact search, no floating point and no LP: the counts are small integers here.
    """
    pairs, cs = cuts(n)
    target = tuple(counts[i][j] for i, j in pairs)
    best = None

    def rec(k, remaining, used, total):
        nonlocal best
        if best is not None:
            return
        if all(v == 0 for v in remaining):
            best = list(used)
            return
        if k == len(cs) or total > budget:
            return
        _, vec = cs[k]
        cap = budget - total
        for m in range(min([remaining[p] for p in range(len(target)) if vec[p]] + [cap]), -1, -1):
            nxt = tuple(remaining[p] - m * vec[p] for p in range(len(target)))
            if any(v < 0 for v in nxt):
                continue
            rec(k + 1, nxt, used + [(cs[k][0], m)] if m else used, total + m)
            if best is not None:
                return

    rec(0, target, [], 0)
    return best


runs, canonical = load()
n = len(runs)
floor = canonical["body"]["noise_floor"]
print(f"published floor: {n} runs, channels {sorted(floor['per_channel'])}")

battery_items = len(canonical["body"].get("items") or [])
print(f"battery items recorded on the canonical cert: {battery_items} (assuming {ITEMS})")

verdicts = {}
for ch, d in floor["per_channel"].items():
    m = matrix(d["distances"], n)
    ok, bad = is_metric(m)
    tight = tight_triangles(m)
    print(f"\n--- {ch}")
    print(f"  metric axioms hold: {ok}" + ("" if ok else f"  violations: {bad[:3]}"))
    print(f"  distinct triangle identities holding exactly: {len(tight)} of "
          f"{triangle_identity_count(n)}")
    for t in tight[:4]:
        print(f"    {t}")
    verdicts[ch] = {"metric": ok, "tight": len(tight)}

    if ch == "exact":
        scaled = [v * ITEMS for v in d["distances"]]
        integral = all(abs(v - round(v)) < 1e-9 for v in scaled)
        print(f"  every distance is k/{ITEMS}: {integral}   counts k = "
              f"{[round(v) for v in scaled]}")
        verdicts[ch]["granular"] = integral
        if integral:
            counts = matrix([round(v) for v in scaled], n)
            sol = realizable(counts, n, ITEMS)
            print(f"  realizable as item disagreements over {ITEMS} items: {sol is not None}")
            if sol:
                total = sum(m_ for _, m_ in sol)
                print(f"    a witness uses {total} of {ITEMS} items; the other "
                      f"{ITEMS - total} are items all five runs agreed on:")
                for S, m_ in sol:
                    print(f"      {m_} item(s) where runs {sorted(S)} differed from the rest")
            verdicts[ch]["realizable"] = sol is not None

# ---------------------------------------------------------------- does it have teeth?

print("\n--- controls: can a plausible forged matrix fail these?")
trials = [
    ("a forger who inflates one pair to widen the floor",
     [2, 1, 0, 0, 9, 2, 2, 1, 1, 0]),
    ("a forger who picks round numbers",
     [2, 2, 0, 0, 5, 2, 2, 2, 2, 0]),
    ("the published counts",
     [2, 1, 0, 0, 3, 2, 2, 1, 1, 0]),
]
for name, counts_flat in trials:
    cm = matrix(counts_flat, n)
    ok, bad = is_metric(cm)
    sol = realizable(cm, n, ITEMS) if ok else None
    status = "metric FAILS" if not ok else ("realizable" if sol else "metric holds, NOT realizable")
    print(f"  {status:32s}  {name}")
    if not ok:
        print(f"      {bad[0]}")

print(f"""
FINDING
  The published exact-channel matrix satisfies all three conditions, and its triangle inequalities
  are {verdicts['exact']['tight']} times EXACTLY tight -- the disagreeing item sets are disjoint and
  their counts add, 2 + 1 = 3 items out of 64. None of that needed a prior certificate. It is a
  predicate over ONE certificate, available to the very first claim about a subject.

  So the residue as written in THE_BOUNDARY's third correction is TOO LARGE. A first claim is not
  unconstrained. It is constrained by having to be realizable.

  BUT THE CONTROLS BELOW ARE WEAKER THAN THEY LOOK, and an adversary showed why. They inflate a
  distance by editing a CELL, which breaks the triangle inequality against the rest of the row.
  A forger who instead substitutes a whole RUN moves a row and a column together, and the matrix
  stays a metric by construction. Measured against the published bytes: a later floor declaring a
  batch level the log has never seen, with one run spliced from genuine bytes at another precision,
  keeps symmetry, the triangle inequality, k/64 granularity and realizability on all three channels
  while inflating the floor 1.67x on exact and 2.12x on seqlp -- enough to flip this lab's own
  published verdict from exceeds_floor to same. So metric structure refuses careless inflation and
  does not refuse the profitable kind.

AND NOW THE DEFLATION, which belongs in the same breath and is stated here rather than left for
someone else to find.

  The two control forgeries above are refused by the metric axioms, but they were ALREADY refused.
  Round 4 of the sequence made the log recompute the floor from the run certificates it names, so a
  floor's distances are derived and not writable. A forger who hand-edits a distance matrix is
  caught by arithmetic before these conditions are consulted, and a forger who instead fabricates
  five plausible OUTPUT sets gets a metric, granular, realizable matrix for free, because a real
  computation over real sets cannot produce anything else.

  What survives that deflation is the category, not this instance. Internal realizability
  constraints on a single certificate are a NON-EMPTY class, and others in it are not subsumed by
  recomputation: outputs must tokenize under the named tokenizer, must respect the declared 16-token
  limit, and a topk channel's values must be a valid log-softmax -- monotone, normalized, and
  agreeing with the token the run says it emitted. Fabricating those consistently is work, and it is
  work a predicate over one certificate can demand.

  The corrected residue is therefore NOT "the first claim about anything". It is a first claim
  fabricated carefully enough to be internally realizable. That is a smaller residue than the third
  correction wrote, and the difference is the difference between forbidding carelessness and
  forbidding fabrication. Only the second would matter, and no internal predicate reaches it.

  NARROWED AGAIN, by a sixth pass, and this is the one that matters. A forger does not have to
  satisfy a prior at all: they declare a factor level the log has never held, and the comparison
  has nothing to say. Demonstrated -- a floor declaring batch 64, four of five runs the lab's own
  published bytes and the fifth spliced from genuine runs at another precision, inflates the floor
  1.67x on exact and flips the published verdict from exceeds_floor to same, with every structural
  property here intact. So the unreachable case is a first claim in any (subject, factor level)
  cell, and the issuer picks the cell.

  This is the sixth impossibility claim by this author today and the sixth to need narrowing. The
  pattern is the most reliable finding in the document.""")
