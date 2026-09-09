"""Is the published floor's distance matrix a function of the declared factors?

THE_BOUNDARY lists as class two ("reachable by nothing") the batch labels written beside the
recipe meant to corroborate them, on the grounds that five relabelled copies of one forward pass
are byte-indistinguishable from five real runs that agreed exactly.

That argument examines ONE cert. The test that reclassified the other two members looks OUTSIDE
the cert: does another logged cert already carry information a forged value would contradict?

This script asks that question of the lab's own published verdict log. It reads only the stored
entry bytes. It does not import styxx.
"""
import glob
import itertools
import json
import pathlib

LOG = pathlib.Path(
    r"C:\Users\heyzo\clawd\wt\v8\papers\v8\first_verdict_2026_09_09\log\entries"
)

runs = {}
for f in sorted(glob.glob(str(LOG / "*" / "*[0-9].json"))):
    c = json.loads(pathlib.Path(f).read_text(encoding="utf-8"))
    if c.get("type") != "fingerprint":
        continue
    b = c["body"]
    idx = b.get("run_index")
    runs[idx] = {
        "entry": pathlib.Path(f).stem,
        "batch": c["recipe"]["decoding"]["batch_size"],
        "nuisance": b.get("nuisance"),
        "channels": b.get("channels"),
        "items": b.get("items"),
        "floor": b.get("noise_floor"),
    }

order = sorted(runs)
print("runs, by run_index:")
for i in order:
    r = runs[i]
    print(f"  {i}  entry {r['entry']}  batch={r['batch']:>2}  nuisance={json.dumps(r['nuisance'])}")

floor = next(r["floor"] for r in runs.values() if r["floor"])
pairs = list(itertools.combinations(order, 2))
print(f"\npairs in run_index order ({len(pairs)}): {pairs}")

for ch, d in floor["per_channel"].items():
    dist = d["distances"]
    if len(dist) != len(pairs):
        print(f"\n{ch}: {len(dist)} distances for {len(pairs)} pairs -- cannot align, skipping")
        continue
    print(f"\n--- channel {ch}  floor={d['floor']}")
    by_batchpair = {}
    same_batch, diff_batch = [], []
    for (i, j), v in zip(pairs, dist):
        bi, bj = runs[i]["batch"], runs[j]["batch"]
        key = tuple(sorted((bi, bj)))
        by_batchpair.setdefault(key, []).append(((i, j), v))
        (same_batch if bi == bj else diff_batch).append(((i, j), bi, bj, v))
    print(f"  same-batch pairs   : {[(p, v) for p, _, _, v in same_batch]}")
    print(f"  all exactly zero   : {all(v == 0 for *_, v in same_batch)}")
    print(f"  diff-batch pairs   : {[(p, f'{bi}v{bj}', v) for p, bi, bj, v in diff_batch]}")
    print(f"  all strictly > 0   : {all(v > 0 for *_, v in diff_batch)}")
    functional = all(len({v for _, v in vs}) == 1 for vs in by_batchpair.values())
    print(f"  d depends ONLY on the unordered batch pair: {functional}")
    for key, vs in sorted(by_batchpair.items()):
        vals = sorted({v for _, v in vs})
        print(f"    batches {key}: {len(vs)} pair(s) -> {vals}")

# Does item_order move anything at all, holding batch fixed?
print("\n--- item_order, held against batch")
for i, j in pairs:
    if runs[i]["batch"] != runs[j]["batch"]:
        continue
    ni, nj = runs[i]["nuisance"], runs[j]["nuisance"]
    print(f"  runs {i},{j}: batch {runs[i]['batch']} both; nuisance {json.dumps(ni)} vs {json.dumps(nj)}")

# Are the per-run recorded outputs identical for same-batch runs? If they are, determinism is a
# logged fact and a later cert's run at a batch size already logged must reproduce it.
print("\n--- per-run recorded channel values, same-batch pairs")
for i, j in pairs:
    if runs[i]["batch"] != runs[j]["batch"]:
        continue
    a, b = runs[i]["channels"], runs[j]["channels"]
    same = json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    ia, ib = runs[i]["items"], runs[j]["items"]
    isame = json.dumps(ia, sort_keys=True) == json.dumps(ib, sort_keys=True)
    print(f"  runs {i},{j} (batch {runs[i]['batch']}): channels identical={same}  items identical={isame}")
