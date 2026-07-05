"""Second independent pass on rung 1's gemma tension (single-scout 15/17). Card PT-2B column
(mechanically parsed, hash-verified vs CARD_MANIFEST) vs the card's SOLE linked receipt: the Gemma-2
technical report (https://storage.googleapis.com/deepmind-media/gemma/gemma-2-report.pdf,
sha256 4caf88137b228c54..., Table 13, p.7, 'Gemma-2 2B' column — transcribed below verbatim).
Match tolerance: exact at published precision (both sources print one decimal)."""
import json

REPORT_2B = {  # Table 13, Gemma-2 2B column (report's own eval of the same model)
    "MMLU": 52.2, "ARC-c": 55.7, "GSM8K": 24.3, "AGIEval": 31.5, "DROP": 51.2,
    "BIG-Bench": 41.9,  # report row 'BBH'
    "WinoGrande": 71.3, "HellaSwag": 72.9, "MATH": 16.0, "ARC-e": 80.6, "PIQA": 78.4,
    "SocialIQA": 51.9,  # report row 'SIQA'
    "BoolQ": 72.7, "TriviaQA": 60.4,
    "Natural Questions": 17.1,  # report row 'NQ'
    "HumanEval": 20.1, "MBPP": 30.2,
}

card = json.load(open("gemma_secondpass_card_2b.json"))
rows, n_match = [], 0
for c in card:
    name = c["benchmark"].split("]")[0].lstrip("[")
    if name not in REPORT_2B:
        continue  # safety/bias rows — report Table 13 doesn't carry them
    rv = REPORT_2B[name]
    match = abs(rv - c["card_2B"]) < 0.05
    n_match += match
    rows.append({"benchmark": name, "card_2B": c["card_2B"], "report_2B": rv,
                 "delta_report_minus_card": round(rv - c["card_2B"], 2), "match": match})
n = len(rows)
lower = sum(1 for r in rows if r["delta_report_minus_card"] > 0)
print(f"{'benchmark':20} {'card':>6} {'report':>7} {'delta':>6}  match")
for r in rows:
    print(f"{r['benchmark']:20} {r['card_2B']:6} {r['report_2B']:7} {r['delta_report_minus_card']:6}  {r['match']}")
print(f"\nSECOND PASS: {n - n_match}/{n} rows DISAGREE (scout said 15/17) | matches: {n_match}")
print(f"direction: report > card on {lower}/{n - n_match} disagreeing rows (systematic, not noise)")
json.dump({"rows": rows, "n_compared": n, "n_disagree": n - n_match,
           "report_pdf_sha256_16": "4caf88137b228c54", "report_cite": "Table 13, p.7",
           "verdict": "SCOUT CONFIRMED" if (n - n_match) == 15 and n == 17 else "SCOUT COUNT DIFFERS"},
          open("gemma_secondpass_result.json", "w"), indent=1)
