# observatory: HuggingFaceTB/SmolLM2-135M

A demo, not an observation of anything: the three `when` dates are labels handed to observe() by
`run_observatory_demo.py`; `taken` is the clock, and all three lines were taken within a minute of
each other. Day 3 is the same weights int8-quantized. The head hash below is what a ledger anchor
would pin: `57b74ba49972d6196f816a817fdc778391029196ec7364897eac383d1a43e9f2` over 3 lines.

`when` is a label the caller supplied; `taken` is the clock at observation.

| # | when | taken | kind | floor measured | floor applied | vs baseline | vs previous | note |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-09-13 | 2026-09-14T00:53:32Z | baseline | 0.00e+00 | 1.00e-04 | — | — | day 1 (the same weights) |
| 2 | 2026-09-14 | 2026-09-14T00:53:45Z | observation | 0.00e+00 | 1.00e-04 | SAME 0.0000 (r 1.000) | SAME 0.0000 (r 1.000) | day 2 (the same weights) |
| 3 | 2026-09-15 | 2026-09-14T00:53:59Z | observation | 0.00e+00 | 1.00e-04 | DRIFT 1.4134 (r 0.829) | DRIFT 1.4134 (r 0.829) | day 3 (the same weights, int8 dynamic quantization) |
