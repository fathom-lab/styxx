# Diagnostic, read-only: how each trivial twin's canonical text differs from its document's sidecar text.
# Reads population.json, the committed sidecars (git, at the pinned commit) and the twin files. No sealed file, no model.
import json, sys, difflib
from pathlib import Path
M = Path(r"C:/Users/heyzo/clawd/wt/leg2-run/papers/sworn/measurement")
sys.path.insert(0, str(M))
import common as C, population as P
from styxx import sworn
pop = json.loads((M / "population.json").read_text(encoding="utf-8"))
out = {}
for e in P.iter_documents(pop):
    side, tree, rec = C.open_document(e)
    side = sworn.load_sidecar(side)
    a = side["text"]
    tw = json.loads((M / "seat_outputs/claude/trivial" / (e["doc_id"] + ".trivial-twin.json")).read_text(encoding="utf-8"))
    b = tw.get("text")
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    ops = [op for op in sm.get_opcodes() if op[0] != "equal"]
    at = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    out[e["doc_id"]] = {"orig_chars": len(a), "twin_chars": len(b), "ratio": round(sm.ratio(), 6),
                        "non_equal_opcodes": len(ops),
                        "diff_at_offset": at,
                        "ops_head": [(t, i1, i2, repr(a[i1:i2][:60]), repr(b[j1:j2][:60])) for t, i1, i2, j1, j2 in ops[:4]]}
for k, v in out.items():
    print(k, json.dumps(v, ensure_ascii=False))
