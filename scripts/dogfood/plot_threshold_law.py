"""Build the threshold-law figure for papers/threshold-law-2026-05-18.md.

Reads:
  out_corpus_coverage_law.json       (original 5-level)
  out_corpus_coverage_law_fine.json  (12-level replication; FAILS strict criterion)

Plots:
  - cross-family AUC (mpnet) vs overlap, original + fine
  - same-family AUC (te3-small) vs overlap, original + fine
  - vertical threshold line at overlap=0.31
  - horizontal floor at AUC=0.80
  - mpnet x corpus_2 cross-vendor exemplar failure flagged
Saves to ../../papers/figures/threshold-law-curve.png
"""
import json, os
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
out_orig = json.load(open(os.path.join(here, "out_corpus_coverage_law.json")))
out_fine = json.load(open(os.path.join(here, "out_corpus_coverage_law_fine.json")))

# Original: rows have foreign + mean_auc
orig_cf = sorted([(r["overlap"], r["mean_auc"]) for r in out_orig["rows"] if r["foreign"] == "all-mpnet-base-v2"])
orig_sf = sorted([(r["overlap"], r["mean_auc"]) for r in out_orig["rows"] if r["foreign"] == "text-embedding-3-small"])

# Fine: rows have both foreign keys flat
fine_cf = sorted([(r["overlap"], r["all-mpnet-base-v2"]) for r in out_fine["rows"]])
fine_sf = sorted([(r["overlap"], r["text-embedding-3-small"]) for r in out_fine["rows"]])

fig, ax = plt.subplots(figsize=(8, 5.2))

# Threshold + floor
ax.axvspan(0.0, 0.31, color="#fde2e2", alpha=0.5, zorder=0, label="below threshold (transport fails)")
ax.axvspan(0.31, 0.5, color="#e2f5e2", alpha=0.5, zorder=0, label="above threshold (transport holds, same-family regime)")
ax.axhline(0.80, color="#888", linestyle=":", linewidth=1, zorder=1)
ax.axvline(0.31, color="#444", linestyle="--", linewidth=1.2, zorder=1)
ax.text(0.312, 0.595, "threshold ≈ 0.31", fontsize=9, color="#333")
ax.text(0.49, 0.805, "AUC floor 0.80", fontsize=9, color="#555", ha="right")

# Curves
ax.plot([o for o, _ in fine_cf], [a for _, a in fine_cf], "o-", color="#c0392b",
        label="cross-family (mpnet) — fine, n=12", linewidth=1.4, markersize=5)
ax.plot([o for o, _ in orig_cf], [a for _, a in orig_cf], "s--", color="#e67e22",
        label="cross-family (mpnet) — orig, n=5", linewidth=1.2, markersize=6, alpha=0.85)
ax.plot([o for o, _ in fine_sf], [a for _, a in fine_sf], "o-", color="#2c7fb8",
        label="same-family (te3-small) — fine, n=12", linewidth=1.4, markersize=5)
ax.plot([o for o, _ in orig_sf], [a for _, a in orig_sf], "s--", color="#5fa9d3",
        label="same-family (te3-small) — orig, n=5", linewidth=1.2, markersize=6, alpha=0.85)

# Flag the cross-vendor exemplar failure cell: mpnet x corpus_2, min Anthropic 0.617
# (overlap for corpus_2 in the original stress run is the low-overlap leg of the cross-family curve)
ax.annotate(
    "exemplar failure:\nmpnet × corpus_2\nAnthropic min AUC 0.617\n(cross-vendor confirm)",
    xy=(0.21, 0.69), xytext=(0.21, 0.62),
    fontsize=8.5, ha="center", color="#7b0000",
    arrowprops=dict(arrowstyle="->", color="#7b0000", lw=1),
)

ax.set_xlabel("corpus ↔ domain overlap (mean-max cosine, te3-large home)")
ax.set_ylabel("transported refusal AUC")
ax.set_title("Corpus ↔ Domain-Overlap Threshold for Label-Free Cognometric Transport\n"
             "Same-family flat; cross-family step at overlap ≈ 0.31  (replication failed strict control criterion)")
ax.set_xlim(0.15, 0.40)
ax.set_ylim(0.58, 0.92)
ax.legend(loc="lower right", fontsize=8.5, framealpha=0.93)
ax.grid(True, linestyle=":", alpha=0.4)

out = os.path.abspath(os.path.join(here, "..", "..", "papers", "figures", "threshold-law-curve.png"))
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.tight_layout()
plt.savefig(out, dpi=160)
print("wrote", out)
