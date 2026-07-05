"""M1 rung 3, step 1 (PREREG_rebinding §1): re-fetch the Llama-3.2-1B-Instruct card, verify sha256,
mechanically parse its benchmark tables into claims. Deterministic — no model, no judgment calls."""
import hashlib
import json
import re

from huggingface_hub import hf_hub_download

CARD_ID = "meta-llama/Llama-3.2-1B-Instruct"
p = hf_hub_download(CARD_ID, "README.md")
raw = open(p, "rb").read()
sha = hashlib.sha256(raw).hexdigest()
manifest = json.load(open("CARD_MANIFEST.json"))
pinned = manifest["files"]["card_meta-llama_Llama-3.2-1B-Instruct.md"]["sha256"]
print(f"sha256: {sha[:16]}…  pinned: {pinned[:16]}…  match: {sha == pinned}")
if sha != pinned:
    raise SystemExit("HASH MISMATCH — card changed since rung 1; stop and report (prereg §1)")

text = raw.decode("utf-8")
# mechanical parse: markdown tables whose header row mentions shots/metric — the eval tables.
claims = []
for tbl in re.finditer(r"((?:^\|.*\|\s*$\n?){3,})", text, re.M):
    rows = [r.strip() for r in tbl.group(1).strip().splitlines()]
    header = [c.strip() for c in rows[0].strip("|").split("|")]
    if not any(re.search(r"# ?shots|shots", h, re.I) for h in header):
        continue
    # locate the special columns
    def col(pat):
        for i, h in enumerate(header):
            if re.search(pat, h, re.I): return i
        return None
    i_bench = col(r"benchmark") if col(r"benchmark") is not None else 1 if len(header) > 1 else 0
    i_shots, i_metric = col(r"shots"), col(r"metric")
    value_cols = [i for i, h in enumerate(header)
                  if i not in (0, i_bench, i_shots, i_metric) and h and not re.search(r"capability|category", h, re.I)]
    for row in rows[2:]:
        cells = [c.strip() for c in row.strip("|").split("|")]
        if len(cells) != len(header): continue
        bench = re.sub(r"\s+", " ", cells[i_bench]).strip()
        if not bench or set(bench) <= set("-: "): continue
        shots = cells[i_shots] if i_shots is not None and i_shots < len(cells) else ""
        metric = cells[i_metric] if i_metric is not None and i_metric < len(cells) else ""
        for vc in value_cols:
            v = cells[vc] if vc < len(cells) else ""
            m = re.fullmatch(r"-?\d+(?:\.\d+)?", v.replace("%", "").strip())
            if m:
                claims.append({"benchmark": bench, "shots": shots, "metric": metric,
                               "column": header[vc], "value": float(m.group())})
print(f"mechanical claim count: {len(claims)}  (rung-1 single-scout said 201)")
onecol = [c for c in claims if re.search(r"1B", c["column"], re.I)]
print(f"in-scope for the 1B-Instruct receipts dataset (1B columns): {len(onecol)}")
json.dump(claims, open("rebind_card_claims.json", "w"), indent=1)
print("-> rebind_card_claims.json")
