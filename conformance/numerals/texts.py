# -*- coding: utf-8 -*-
"""The texts and receipt cases the numerals conformance vectors are generated from.

Every text here came from a probe that was run before the numeral reader v2 was built: the product
red team's posts (RT1, RT2, `P:` probe_tokens, 2026-10-24), the design's 89 probe texts and 19
receipt cases (`DESIGN.md`, 2026-10-25), and the red team of that design (235 reader texts, 92
binding cases, 52 second-pass texts and 14 second-pass cases, `RED_TEAM.md`). A handful at the end
cover the amendments the red team wrote (A-3, A-6, A-9, A-11, A-15, A-16) and the lead's scope.

This file is DATA for `gen_vectors.py`. It carries no expectations: those are recorded by running
the reader and are pinned in `vectors_*.json`. Editing a text here does not edit a vector; the
generator refuses to re-record an existing id with a different expectation.
"""
from __future__ import annotations

NBSP, THIN, NNBSP, ZWSP, SHY, MINUS = " ", " ", " ", "​", "­", "−"
THREAD = "\U0001F9F5"
BASE = "BTC $67.4k (+2.3%) | ETH 3.4K | vol 1.2B | 14:00 UTC"

READER_TEXTS: dict[str, list[str]] = {
    # ---- the product red team (RT2 suffixed, RT1 posts, P: probe_tokens) and the design's own cases
    "product_rt2_suffixed": [
        "market cap $1.2B, 24h volume 3.4M", "BTC $99.9k", "up 12% in 7d, 3x leverage", "2.5M USDT bridged",
        "ETH 2.4K", "price: 67412.5 USDT", "BTC: $67,412.50",
    ],
    "product_rt1_posts": [
        "BTC $67.4k (+2.3%) | ETH $2,431 | SOL $148.20",
        "BTC/USDT 1h: RSI 71, MACD bullish. Entry 67,400 TP1 68,500 TP2 70,000 SL 66,200. Leverage 10x",
        "Daily PnL: +$1,234.56 (+3.2%) across 14 trades, win rate 64%",
        "Market update 2026-10-24 14:00 UTC: BTC holds above 67,000",
        "Top 3 gainers in the last 24h: PEPE +18%, WIF +12%, BONK +9%",
        "Gas is 12 gwei, block 21,034,567 just landed", "BTC dominance 54.2%, fear and greed index 71 (greed)",
        "Wallet balance: 0.5 BTC ($33,706)", "1/ thread on why ETH funding flipped negative",
        "Your refund of $12.00 has been issued.", "Delivery takes 3-5 business days, and you have 2 items in your cart",
        "Ticket #A-10293 is open; call 1-800-555-0199 if it is urgent", "The interest rate is 5%.",
    ],
    "product_probe_tokens": [
        "Order 10293 totals $84.17.", "Total: $84.17.", "claude opus 4.8 and gemini 3.1 pro preview",
        "Application #64/026,964 was filed.",
    ],
    "design_cases": [
        "PnL -$1,234.56 today", "PnL +$1,234.56 today", "loss of " + MINUS + "$500 on the hedge",
        "Price 67" + THIN + "412.50 USDT", "Price 67 412.50 USDT", "Prix 84,17 €", "Le total est de 1.234,5 unités",
        "BTC $68k", "BTC $68k.", "fees $5m", "fees 5m", "15m chart", "4K video", "3M company", "a 7B model",
        "Llama-3.1-8B", "Qwen2.5-7B", "Mixtral 8x7B", "1.2 billion parameters", "2 trillion tokens", "$1.2bn deal",
        "at 9:30 am", "at 2:15pm", "ratio 1:10", "meeting 14:00:05", "2026-10-24T14:00:05Z", "odds 3:1",
        "on Oct 24 the price", "on 24 Oct 2026 the price", "on October 24, 2026 the price", "on 10/24/2026 the price",
        "on 10/24 the price", THREAD + " 1/7 why funding flipped", "2/7 funding flipped", "9/12 held",
        "1) BTC up", "2. ETH down", "version 2.5 shipped", "v2.5 shipped", "styxx 7.50.0 shipped",
        "SL 66,200.", "precision of 0.55.", "all 23,247.", "Fig. 3.", "section 5.", "$2.5T market",
        "range $1.2-1.5B", "2–3B params", "5kg of rice", "4MB file", "1INCH token", "C98 token",
    ],
    # ---- the red team of the design (rt_reader.py, 15 families)
    "builder_thread_counter": [
        BASE + " | 1/5", "1/5 " + BASE, BASE + " (1/5)", BASE + " [1/5]", BASE + " 1/5 " + THREAD,
        BASE + " — 1/5", BASE + " | 1 of 5", "Part 1/5: BTC $67.4k", THREAD + " 1/5 " + BASE,
        "1/ " + BASE, "2/5", "(2/5)", "5/5 fin", "9/12 held", "we closed 3/5 positions",
    ],
    "builder_ticker_lines": [
        "BTC $67.4k (+2.3%) | ETH $3.4K (-1.8%) | SOL $148.20 (+0.4%)",
        "BTC $67.4k (" + MINUS + "2.3%)", "24h vol $1.2B (+5.1%) | OI $18.4B | funding 0.01%",
        "BTC mcap $1.33T | ETH mcap $410B | dominance 54.2%", "BTC 67.4k | 14:00UTC", "BTC 67.4k | 14:00Z",
        "BTC 67.4k at 14:00 CET", "BTC 67.4k at 14h00", "#1 gainer: PEPE +18% | #2 WIF +12%",
        "1) BTC $67.4k 2) ETH 3.4K", "Fear & Greed 71/100", "liq: $142.5M longs, $38.1M shorts",
        "BTC/USDT 1h: RSI 71 | Entry 67.4k TP1 68.5k TP2 70k SL 66.2k | 10x",
        "| coin | price |\n|---|---|\n| BTC | $67.4k |\n| ETH | 3.4K |",
        "`BTC $67.4k`", "```\nBTC $67.4k\n```", "price `67412.5`",
    ],
    "ranges": [
        "BTC range 66-68k today", "BTC range 66k-68k today", "BTC range $66k-$68k today", "BTC 66k–68k",
        "60-70k", "$60-70k", "between 60k and 70k", "60k to 70k", "2–3B params", "range $1.2-1.5B",
    ],
    "approximations_and_hyphen_words": [
        "BTC $67k-ish", "BTC ~$67.4k", "BTC about 67k", "BTC ≈67.4k", "BTC 67k+", "BTC roughly $67k",
        "BTC $67.4k-plus", "a $1.2B-plus raise", "a 1.2 billion-dollar deal", "7B-parameter model",
        "BTC in the mid-60k range", "sub-67k", "sub-$67k", "low-70k", "near-70k", "circa 70k", "c. 70k",
        "$99.9k-ish", "$99.9-ish", "$99,900-ish", "5-day volume 1.2B", "3-year high",
    ],
    "m_ambiguity_and_spellings": [
        "fees 5m", "5m candles", "5m ago", "5M", "5 min", "in 5m BTC did 5M volume", "$5m", "$5mm fund", "$5MM",
        "$5 M", "5 M", "$5 million", "5mn", "5 mn", "5bn", "5 bn", "€5m", "₹5m", "₹50L", "1.5 Cr",
        "R$ 84,17", "CHF 5m", "5m€", "A$5m", "US$1.2B", "USD 1.2B", "1.2B USD", "$ 5m", "5k", "5K", "5 k",
        "401k plan", "4K video", "10K run", "3M company", "a 7B model", "Llama-3.1-8B", "Mistral-7B", "GPT-7B",
        "5kg", "4MB", "4Mbps", "5TB", "5MW", "3Mt", "2Bq",
    ],
    "locale": [
        "1,5k Follower", "ETH 3,4K", "Preis 67.412,50 €", "Preis 67.412 €", "Marktkapitalisierung 1.234.567 €",
        "1 234 567", "67" + NBSP + "412.50", "67" + NBSP + "412,50", "67" + NNBSP + "412,50", "67" + THIN + "412.50",
        "1.234.567,89", "price 67.412.50", "$84,17", "84,17 $", "1,234,56", "5,00", "0,5k",
    ],
    "unicode_digits": [
        "BTC $６７.４k", "BTC $٦٧.٤k", "३४M", "BTC $6７.4k",
        "$67.4" + ZWSP + "k", "$67.4" + NBSP + "k", "67" + SHY + "412", "$67.4k" + "‏", "‮$67.4k",
    ],
    "lookalike_suffixes": [
        "3.4М", "2.4К", "67.4к", "2.4Κ", "1.2Β", "3.4Ｍ", "67.4ｋ", "67.4ĸ",
        "1.2Т", "67.4\U0001d5f8", "67.4ᵏ", "$99.9к", "$99.9ḱ", "$99.9" + ZWSP + "k", "$99.9k" + ZWSP,
    ],
    "percent_with_suffix": ["+12k%", "12% of 1.2B", "1.2B%", "up 1.2k% lol", "+2.3% to $67.4k", "2.3 % | 67.4k"],
    "signed": [
        "-5k", "+5k", "$-5k", "-$5k", MINUS + "$5k", "- $5k", "($5k)", "PnL: -$1.2k", "PnL -1.2k", "+$1.2k (+3.2%)",
        "-0.5k", "-$0.5k", "-$.5k", "PnL -$1,234.56.", "PnL: -$1.2k.", "loss -1.2 million", "-1.2 billion",
    ],
    "clock_scores_ratios": [
        "score 21:18", "odds 10:11", "ratio 15:10", "verse 3:16", "24:00", "0:30", "00:30", "14:00-15:00",
        "14:00:05.123", "at 14:00. BTC", "ETH/BTC 0.036 | 14:00", "1:14:00 elapsed",
    ],
    "date_collisions": [
        "sold Mar 10,000 contracts", "Jun 2,431 ETH", "Oct 24, 2026", "Oct 32", "May 5 trades closed",
        "expiry Dec 1,000 contracts", "Sep 18 cut 25 bps", "10/24 close", "24/10/2026", "2026/10/24", "24.10.2026",
    ],
    "sentence_final": [
        "BTC $67.4k.", "mcap $1.2B.", "ETH 3.4K...", "67.4k…", "cost $0.", "Fig. 3.", "No. 5.", "p. 12.",
        "It cost 1.2 billion.", "price is 67,412.50.", "rate 0.0666...", "rate 0.0666… ok", "v2.5.", "7B.",
    ],
    "adversary_misc": [
        "$67.4k5", "$67.4kk", "$67.4k_", "$67.4k1", "$67.4K's", "67.4e3", "6.74e4", "67.4 thousand",
        "sixty-seven thousand", "$67.4k²", "$67.4k⁰", "$67.4³k", "67.4k;", "67.4k:", "67.4k/",
        "$67.4k-$68k", "$67.4k–$68k", "67.4k-68k", "1,234k", ".5M", "$.5M", "0.50M", "00.5M", "1,000,000k",
    ],
    # ---- the red team's second pass (rt_extra.py)
    "lowercase_currency_order": [
        "$5m", "$5m | 5m", "5m | $5m", "15m chart: BTC $67m vol", "5m candles | mcap $1.2b", "in 5m BTC did $5m volume",
        "fees 5m | $5m", "5m | $5k", "5m | $67,412", "5m | 3.4M | $5m", "5m | $5m | 3.4M", "5m chart | BTC mcap $1.33t",
        "5m chart | BTC $99.9m mcap", "15m | ETH $410b mcap | SOL $70b", "1h | ETH $410b", "5b | $5b", "€5m after 5m",
    ],
    "form_hides_count": [
        "we did May 5 trades today", "bought 10 Dec puts", "bought 5 Jun 70k calls", "Mar 10 BTC bought",
        THREAD + " 3/5 trades won", THREAD + " 9/12 held", "won 21:18 on the day", "closed at 14:00 with 14 wins",
        "Sep 18 cut 25 bps", "May 2026 outlook", "5 May", "May 5", "May 5.", "ver. 99", "version 99.9 shipped",
    ],
    "labels_now_read": ["Fig. 3.", "No. 5.", "p. 12.", "see §4.", "Table 2.", "step 3.", "Q3.", "H2.", "see 2.3."],
    "glue": [
        "\U0001F680 $67.4k \U0001F525", "$67.4k\U0001F680", "BTC $67.4\nk", "BTC $67.4k/ETH 3.4K", "67.4k|3.4K",
        "[BTC $67.4k](https://x.test/p/1)", "1INCH $0.25", "1000SATS 0.0003", "$1K token", "**$67.4k**.", "BTC:$67.4k",
    ],
    # ---- the amendments and the lead's scope
    "amendments": [
        "- $5k revenue", "- $5k", "  - $1.2k fee", "PnL ($1.2k) today", "PnL - $1.2k today", "($1,234.56)",
        "sold Mar 10,000 contracts", "Jun 2,431 ETH", "Oct 24, 2026", "5 May", "May 5.", "May 5 at noon",
        "5 May, 2026", "May 5th",
        "BTC $67.4k | 14:00 UTC | 1/5", "BTC $67.4k (1/5)", "BTC $67.4k [1/5]", "BTC $67.4k · 2/7",
        "fired on 15/15", "| 30/30 |", "covering 45/45.", "ladder coverage >= 10/12", "3/5", "1/5.", "| 1/51",
        "$5mm", "$5MM raised", "5mm", "₹5m raised", "₿0.5k", "₩5k", "₽5k", "₺5k",
        "1,5k", "3,4K", "67" + THIN + "412,50", "67" + NNBSP + "412,50", "67" + NBSP + "412,50", "1.234.567",
        "1.234.567,89", "67.412 €", "€ 67.412", "5,00", "1.234,5",
        "$99.9к", "3.4М", "67.4ｋ", "$99.9k²", "BTC $６７.４k",
        "5m | $5m", "15m | ETH $410b mcap | SOL $70b", "5m chart | BTC $99.9m mcap", "fees 5m | $5m",
        "60-70k", "$60-70k", "66k-68k", "$66k-$68k", "2–3B", "Llama-3.2-3B", "$1.2-1.5B",
        "67k-ish", "$99.9k-ish", "a $1.2B-plus raise", "sub-$67k", "mid-60k", "3-year high", "7B-Instruct",
        "BTC $99.9k.", "BTC $67.4k.", "3.4K...", "$1.2B…", "7B.",
        "market cap $1B", "BTC mcap $2T", "vol .5M", "vol 9B", "fee $1k", "BTC $67k", "BTC $67.41k",
        "the mean rate is $99.9k", "a 7B model with recall 0.9", "accuracy 12k", "accuracy $12k",
        "BTC 67.4k | 14:00UTC", "BTC 67.4k | 14:00Z", "score 21:18", "odds 10:11",
        "version 2.5", "ver. 2.5", "Version 7.50.0", "Oct. 24", "October 24th, 2026", "24 October 2026",
    ],
}

# A Binance GET /api/v3/ticker/24hr row for BTCUSDT (the documented field set; values synthetic).
BINANCE = {"symbol": "BTCUSDT", "priceChange": "1512.30", "priceChangePercent": "2.295",
           "weightedAvgPrice": "66980.12", "prevClosePrice": "65900.20", "lastPrice": "67412.50", "lastQty": "0.00500",
           "bidPrice": "67412.49", "bidQty": "1.20000", "askPrice": "67412.51", "askQty": "0.80000",
           "openPrice": "65900.20", "highPrice": "68120.00", "lowPrice": "65410.00", "volume": "18234.56700",
           "quoteVolume": "1221345678.90", "openTime": 1729814400000, "closeTime": 1729900799999,
           "firstId": 4401234567, "lastId": 4403678901, "count": 2444334}
# A CoinGecko /coins/markets row (values synthetic).
GECKO = {"id": "bitcoin", "symbol": "btc", "current_price": 67412.5, "market_cap": 1331234567890, "market_cap_rank": 1,
         "fully_diluted_valuation": 1415000000000, "total_volume": 31234567890, "high_24h": 68120, "low_24h": 65410,
         "price_change_24h": 1512.3, "price_change_percentage_24h": 2.295, "circulating_supply": 19760000,
         "total_supply": 21000000, "max_supply": 21000000, "ath": 73750, "atl": 67.81,
         "last_updated": "2026-10-24T14:00:05.123Z"}
NARROW3 = {"btc": {"lastPrice": "67412.50", "priceChangePercent": "2.295", "quoteVolume": "1221345678.90"}}
ETH = {"eth": {"lastPrice": "3412.50"}}

# (name, text, receipt | [receipts]). A name is a label for a reader; the vector id is over the text and
# the receipts, never over the name.
BINDING_CASES: list[tuple] = [
    # ---- the design's 19 cases
    ("k_suffix", "BTC is at $67.4k on Binance spot, up from yesterday", {"btc": {"price": "67412.50000000"}}),
    ("k_suffix_made_up", "BTC is at $99.9k on Binance spot, up from yesterday", {"btc": {"price": "67412.50000000"}}),
    ("k_suffix_int", "BTC is at $67k on Binance spot", {"btc": {"price": "67412.50000000"}}),
    ("k_suffix_int_unnamed", "Bitcoin is at $67k on Binance spot", {"btc": {"price": "67412.50000000"}}),
    ("k_edge_low", "BTC is at $67.4k", {"btc": {"price": "67350"}}),
    ("k_edge_high", "BTC is at $67.4k", {"btc": {"price": "67450"}}),
    ("k_edge_out", "BTC is at $67.4k", {"btc": {"price": "67450.01"}}),
    ("B_cap", "market cap $1.2B today", {"sol": {"market_cap": 1234567890}}),
    ("M_volume", "24h volume 3.4M", {"eth": {"volume": "3412000"}}),
    ("word_billion", "market cap of 1.2 billion dollars", {"sol": {"market_cap": 1234567890}}),
    ("rate_word_k", "the mean rate is $67.4k", {"btc": {"price": "67412.50"}}),
    ("rate_word_k_wrong", "the mean rate is $99.9k", {"btc": {"price": "67412.50"}}),
    ("order_period", "Order 10293 totals $84.17.", {"order": {"order_number": 10293, "total_price": "84.17"}}),
    ("order_period_wrong", "Order 10293 totals $48.17.", {"order": {"order_number": 10293, "total_price": "84.17"}}),
    ("signed_loss", "Closed with PnL -$1,234.56 on the trade", {"trade": {"pnl": "-1234.56"}}),
    ("signed_flip", "Closed with PnL -$1,234.56 on the trade", {"trade": {"pnl": "1234.56"}}),
    ("time_noise", "Market update 14:00 UTC: BTC at 67,412.50", {"btc": {"price": "67412.50"}}),
    ("precision_final", "The gate reached a precision of 0.55.", {"precision": 0.55}),
    ("precision_final_wrong", "The gate reached a precision of 0.65.", {"precision": 0.55}),
    # ---- the red team's 92 binding cases (rt_binding.py)
    ("loose_1B_vs_0.51B", "market cap $1B", {"sol": {"mcap": 510000000}}),
    ("loose_1B_vs_1.49B", "market cap $1B", {"sol": {"mcap": 1490000000}}),
    ("loose_1B_vs_1.5B_inclusive", "market cap $1B", {"sol": {"mcap": 1500000000}}),
    ("loose_2T_vs_1.5T", "BTC mcap $2T", {"btc": {"mcap": 1500000000000}}),
    ("loose_1T_vs_1.33T", "BTC market cap $1T", {"btc": {"mcap": 1331234567890}}),
    ("loose_67k_vs_66500", "BTC $67k", {"btc": {"price": 66500}}),
    ("loose_67k_vs_67500", "BTC $67k", {"btc": {"price": 67500}}),
    ("loose_67k_vs_67500.01", "BTC $67k", {"btc": {"price": 67500.01}}),
    ("loose_1k_vs_1499", "fee $1k", {"fee": {"amount": 1499}}),
    ("loose_1k_vs_500", "fee $1k", {"fee": {"amount": 500}}),
    ("loose_point5M_vs_450000", "vol .5M", {"eth": {"vol": 450000}}),
    ("loose_1.2B_vs_1.25B", "vol 1.2B", {"eth": {"vol": 1250000000}}),
    ("loose_9B_vs_8.5B", "vol 9B", {"eth": {"vol": 8500000000}}),
    ("full_1e9_vs_0.51B", "market cap $1,000,000,000", {"sol": {"mcap": 510000000}}),
    ("full_67000_vs_67412", "BTC $67,000", {"btc": {"price": 67412.5}}),
    ("full_67412_vs_67412.5", "BTC $67,412", {"btc": {"price": 67412.5}}),
    ("whole_mcap_1.7T_vs_closeTime", "BTC mcap $1.7T", {"btc": BINANCE}),
    ("whole_mcap_1.73T_vs_closeTime", "BTC market cap $1.73T", {"btc": BINANCE}),
    ("whole_mcap_2T_vs_closeTime", "BTC mcap $2T", {"btc": BINANCE}),
    ("whole_vol_4.4B_vs_lastId", "24h vol $4.4B", {"btc": BINANCE}),
    ("whole_OI_4.4B_vs_lastId", "BTC OI $4.4B", {"btc": BINANCE}),
    ("whole_vol_2.4M_vs_count", "ETH vol 2.4M", {"btc": BINANCE}),
    ("whole_users_2.4M_vs_count", "2.4M users", {"btc": BINANCE}),
    ("whole_price_68k_vs_high", "BTC $68k", {"btc": BINANCE}),
    ("whole_price_65k_vs_low", "BTC $65k", {"btc": BINANCE}),
    ("whole_price_66k_vs_open", "BTC $66k", {"btc": BINANCE}),
    ("whole_true_price_67.4k", "BTC $67.4k", {"btc": BINANCE}),
    ("whole_true_vol_1.2B", "24h vol $1.2B", {"btc": BINANCE}),
    ("whole_true_trades_2.4M", "2.4M trades", {"btc": BINANCE}),
    ("whole_pct_2.3", "BTC +2.3%", {"btc": BINANCE}),
    ("whole_pct_neg_2.3", "BTC -2.3%", {"btc": BINANCE}),
    ("gecko_mcap_1.4T_vs_fdv", "BTC mcap $1.4T", {"btc": GECKO}),
    ("gecko_mcap_1T_vs_mcap", "BTC market cap $1T", {"btc": GECKO}),
    ("gecko_vol_21M_vs_max_supply", "vol 21M", {"btc": GECKO}),
    ("gecko_ath_74k_true", "ATH $74k", {"btc": GECKO}),
    ("gecko_67.8_vs_atl", "BTC is at $67.8", {"btc": GECKO}),
    ("gecko_supply_19.8M_true", "supply 19.8M BTC", {"btc": GECKO}),
    ("narrow_mcap_1.7T_vs_price", "BTC mcap $1.7T", {"btc": {"lastPrice": "67412.50"}}),
    ("mark_eth_2431_int", "ETH $2,431", {"eth": {"price": 2431.27}}),
    ("mark_eth_2.4K", "ETH 2.4K", {"eth": {"price": 2431.27}}),
    ("mark_btc_67412_int", "BTC $67,412", {"btc": {"price": 67412.5}}),
    ("mark_eth_3,4K_locale", "ETH 3,4K", {"eth": {"price": 3412.5}}),
    ("mark_preis_decimal_comma", "Preis 67.412,50 €", {"btc": {"price": 67412.5}}),
    ("mark_nbsp_group", "BTC 67" + NBSP + "412.50", {"btc": {"price": 67412.5}}),
    ("mark_ish_silent", "BTC $67.4k-ish", {"btc": {"price": 67412.5}}),
    ("mark_mid60k_silent", "BTC in the mid-60k range", {"btc": {"price": 67412.5}}),
    ("mark_receipt_suffixed_string", "BTC $67.4k", {"btc": {"price": "67.4K"}}),
    ("mark_space_B", "vol 1.2 B", {"eth": {"vol": 1234567890}}),
    ("mark_space_bn", "vol 1.2 bn", {"eth": {"vol": 1234567890}}),
    ("mark_vol_12.3k_btc_true", "vol 12.3k BTC", {"btc": {"volume": "12345.678"}}),
    ("mark_5mm", "$5mm raised", {"deal": {"amount": 5000000}}),
    ("mark_rupee_5m", "₹5m raised", {"deal": {"amount": 5000000}}),
    ("mark_fees_5m", "fees 5m", {"deal": {"fees": 5000000}}),
    ("range_60_70k_binds_rsi60", "BTC range 60-70k today", {"btc": {"rsi": 60, "low": 60100, "high": 69900}}),
    ("range_60k_70k", "BTC range 60k-70k today", {"btc": {"rsi": 60, "low": 60100, "high": 69900}}),
    ("range_$60k_$70k", "BTC range $60k-$70k today", {"btc": {"rsi": 60, "low": 60100, "high": 69900}}),
    ("range_60_to_70k", "BTC between 60k and 70k today", {"btc": {"rsi": 60, "low": 60100, "high": 69900}}),
    ("signed_neg_k_true", "PnL -$1.2k today", {"trade": {"pnl": "-1234.56"}}),
    ("signed_neg_k_flip", "PnL -$1.2k today", {"trade": {"pnl": "1234.56"}}),
    ("signed_space_neg", "PnL - $1.2k today", {"trade": {"pnl": "-1234.56"}}),
    ("signed_paren", "PnL ($1.2k) today", {"trade": {"pnl": "-1234.56"}}),
    ("signed_loss_word", "$1.2k loss today", {"trade": {"pnl": "-1234.56"}}),
    ("signed_minus_sign_k", "PnL " + MINUS + "$1.2k today", {"trade": {"pnl": "-1234.56"}}),
    ("signed_neg_word", "lost 1.2 million", {"fund": {"pnl": -1234567}}),
    ("signed_paren_flip", "PnL ($1.2k) today", {"trade": {"pnl": "1234.56"}}),
    ("signed_space_flip", "PnL - $1.2k today", {"trade": {"pnl": "1234.56"}}),
    ("A4_made_up_rate_k", "the mean rate is $99.9k", {"btc": {"price": 67412.5}}),
    ("A4_made_up_rate_full", "the mean rate is 99,912", {"btc": {"price": 67412.5}}),
    ("A4_made_up_rate_full_dec", "the mean rate is 99,912.5", {"btc": {"price": 67412.5}}),
    ("A4_precision_rule_k", "n = 12.3456789k samples", {"run": {"n": 12345678.9}}),
    ("A4_range_sanity_k", "accuracy 12k", {"run": {"accuracy": 0.9}}),
    ("approx_tilde", "BTC ~$67k", {"btc": {"price": 67412.5}}),
    ("approx_about", "BTC about $67k", {"btc": {"price": 67412.5}}),
    ("approx_ish_madeup", "BTC $99.9k-ish", {"btc": {"price": 67412.5}}),
    ("approx_plus_madeup", "a $9.9B-plus raise", {"deal": {"amount": 1200000000}}),
    ("date_mar_10000", "sold Mar 10,000 contracts", {"book": {"contracts": 10000, "fees": 0}}),
    ("date_jun_2431", "ETH closed Jun 2,431", {"eth": {"price": 2431}}),
    ("uni_fullwidth_digits", "BTC $６７.４k", {"btc": {"price": 67412.5}}),
    ("uni_arabic_indic", "BTC $٦٧.٤k", {"btc": {"price": 67412.5}}),
    ("uni_cyrillic_M_madeup", "BTC vol 9.9М", {"btc": {"vol": 3400000}}),
    ("uni_cyrillic_k_madeup", "BTC $99.9к", {"btc": {"price": 67412.5}}),
    ("uni_fullwidth_K_madeup", "BTC $99.9Ｋ", {"btc": {"price": 67412.5}}),
    ("uni_zwsp_madeup", "BTC $99.9" + ZWSP + "k", {"btc": {"price": 67412.5}}),
    ("code_inline", "`BTC $67.4k`", {"btc": {"price": 67412.5}}),
    ("code_fenced_madeup", "```\nBTC $99.9k\n```", {"btc": {"price": 67412.5}}),
    ("code_fenced_true", "```\nBTC $67.4k\n```", {"btc": {"price": 67412.5}}),
    ("builder_line_whole", "BTC $67.4k (+2.3%) | ETH 3.4K | vol 1.2B | 14:00 UTC | 1/5", [{"btc": BINANCE}, ETH]),
    ("builder_line_narrow", "BTC $67.4k (+2.3%) | ETH 3.4K | vol 1.2B | 14:00 UTC | 1/5", [NARROW3, ETH]),
    ("builder_line_no_counter", "BTC $67.4k (+2.3%) | ETH 3.4K | vol 1.2B | 14:00 UTC", [NARROW3, ETH]),
    ("builder_line_madeup_price", "BTC $99.9k (+2.3%) | ETH 3.4K | vol 1.2B | 14:00 UTC", [NARROW3, ETH]),
    ("builder_line_madeup_vol", "BTC $67.4k (+2.3%) | ETH 3.4K | vol 2.4B | 14:00 UTC", [NARROW3, ETH]),
    # ---- the red team's second pass (rt_extra.py)
    ("lc_cur_after_bare_m_true", "15m chart: BTC mcap $1.33t", {"btc": {"mcap": 1330000000000}}),
    ("lc_cur_after_bare_m_madeup", "5m chart | BTC $99.9m mcap", {"btc": {"mcap": 1330000000000}}),
    ("lc_cur_alone_true", "BTC mcap $1.33t", {"btc": {"mcap": 1330000000000}}),
    ("lc_cur_alone_madeup", "BTC mcap $9.99t", {"btc": {"mcap": 1330000000000}}),
    ("hide_may5", "we did May 5 trades today", {"bot": {"trades": 3}}),
    ("hide_thread_emoji", THREAD + " 3/5 trades won", {"bot": {"won": 1, "trades": 5}}),
    ("neg_k_true", "PnL -5k", {"t": {"pnl": -5200}}),
    ("neg_k_sign_flip", "PnL -5k", {"t": {"pnl": 5000}}),
    ("neg_k_pos_window", "PnL -5k", {"t": {"pnl": -4500.01}}),
    ("asym_eth_int", "ETH $2,431", {"eth": {"price": 2431.27}}),
    ("asym_eth_K", "ETH 2.4K", {"eth": {"price": 2431.27}}),
    ("asym_eth_dec", "ETH $2,431.27", {"eth": {"price": 2431.27}}),
    ("asym_btc_int", "BTC $67,412", {"btc": {"price": 67412.5}}),
    ("asym_btc_k", "BTC $67k", {"btc": {"price": 67412.5}}),
    # ---- the amendments (RT A-1, A-2, A-8, A-9, A-15, A-16) and the lead's product need
    ("A2_1B_vs_1.04B", "market cap $1B", {"sol": {"mcap": 1040000000}}),
    ("A2_1B_vs_0.95B_inclusive", "market cap $1B", {"sol": {"mcap": 950000000}}),
    ("A2_1B_vs_0.949B", "market cap $1B", {"sol": {"mcap": 949000000}}),
    ("A2_67k_vs_67499", "BTC $67k", {"btc": {"price": 67499}}),
    ("A1_whole_binance_true_price", "BTC $67.4k | 24h vol $1.2B | 2.4M trades", {"btc": BINANCE}),
    ("A1_bids_not_excluded", "BTC bids 4.4B", {"btc": {"bids": 4401234567}}),
    ("A1_updated_at_excluded", "BTC $1.7T", {"btc": {"updated_at": 1729814400000}}),
    ("A8_bare_7B_recall", "a 7B model with recall 0.9", {"m": {"recall": 0.9}}),
    ("A8_currency_rate_madeup", "the mean rate is $99.9k", {"btc": {"price": 67412.5}}),
    ("A8_currency_rate_true", "the mean rate is $67.4k", {"btc": {"price": 67412.5}}),
    ("A8_currency_range_sanity", "accuracy $12k", {"run": {"accuracy": 0.9}}),
    ("A9_bullet_is_not_a_sign", "- $5k revenue", {"q": {"revenue": 5000}}),
    ("A9_bullet_is_not_a_sign_neg", "- $5k revenue", {"q": {"revenue": -5000}}),
    ("A16_may5_trades_true", "we did May 5 trades today", {"bot": {"trades": 5}}),
    ("A16_oct24_date_excluded", "on Oct 24, the price was $67.4k", {"btc": {"price": 67412.5}}),
    ("A3_counter_last_after_pipe", "BTC $67.4k | 14:00 UTC | 1/5", {"btc": {"price": 67412.5}}),
    ("A3_counter_line_initial_read", "1/5 BTC $67.4k", {"btc": {"price": 67412.5}}),
    ("product_need_madeup_99.9k", "BTC is at $99.9k on Binance spot, up from yesterday", {"btc": {"price": "67412.50"}}),
    ("product_need_true_67.4k", "BTC is at $67.4k on Binance spot, up from yesterday", {"btc": {"price": "67412.50"}}),
    ("product_need_madeup_sentence_final", "BTC is at $99.9k.", {"btc": {"price": "67412.50"}}),
    ("product_need_true_sentence_final", "BTC is at $67.4k.", {"btc": {"price": "67412.50"}}),
    ("decimal_comma_counted", "Prix 84,17 €", {"order": {"total": 84.17}}),
    ("dotted_groups_counted", "Marktkapitalisierung 1.234.567 €", {"x": {"mcap": 1234567}}),
    ("glued_nonascii_counted", "BTC $99.9к today", {"btc": {"price": 67412.5}}),
    ("ellipsis_after_magnitude", "vol 3.4M... up", {"eth": {"vol": 3412000}}),
    ("thin_group_true", "Price 67" + THIN + "412.50 USDT", {"btc": {"price": 67412.5}}),
]


def all_texts() -> list[str]:
    """Every reader text, in family order, duplicates removed (the id is over the text)."""
    seen, out = set(), []
    for texts in READER_TEXTS.values():
        for t in texts:
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out
