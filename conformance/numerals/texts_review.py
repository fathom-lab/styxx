# -*- coding: utf-8 -*-
"""The texts and receipt cases the two reviews of the numeral reader v2 ran (2026-10-25), as data.

`texts.py` is not edited: these are new texts, appended by `gen_vectors.py` to that file's texts for a
set written after the review fixes. Sources: the correctness review's adversarial cases
(`review_correctness/adversarial.py` in the product kit, FB01-FB22, FM01-FM10, OK01-OK07, with its
receipts) and the claims review's probes (F1 timestamp names, F2 lakh grouping, F3 spaced and trailing
currency, F7 word magnitudes, F8 sentence-final exceptions, F9 word-led versions, A-18). Like `texts.py`,
this file carries no expectations; the generator records them by running the reader.
"""
from __future__ import annotations

ROCKET = "\U0001F680"
NOW_MS, NOW_S = 1792929600000, 1792929600
TICK = {"btc": {"lastPrice": "67412.50", "priceChangePercent": "2.31", "highPrice": "68120.00", "lowPrice": "66010.10",
                "openPrice": "66800.00", "weightedAvgPrice": "67102.33", "volume": "18234.551",
                "quoteVolume": "1229876543.21", "count": "1834221", "openTime": NOW_MS - 86400000, "closeTime": NOW_MS,
                "firstId": 3812345678, "lastId": 3814179899}}
OKX = {"btc": {"instId": "BTC-USDT", "last": "67412.5", "high24h": "68120", "low24h": "66010.1", "vol24h": "18234.5",
               "ts": str(NOW_MS)}}
WS = {"btc": {"e": "24hrTicker", "E": NOW_MS + 123, "s": "BTCUSDT", "c": "67412.50", "O": NOW_MS - 86400000,
              "C": NOW_MS, "n": 1834221}}
KLINES = {"btc": {"klines": [[NOW_MS - 3600000, "67000.0", "68120.0", "66010.1", "67412.5"]]}}
NARROW = {"btc": {"lastPrice": "67412.50"}}

READER_TEXTS: list[str] = [
    # R2: conversions, dashes and bullets before a price
    "Holding 1 ETH ($2,431) in the vault", "BTC ($67.4k) leads the majors", "**BTC** - $67,412.50",
    "Bitcoin (BTC) - $67,412.50", "- prices\n    - $67.4k BTC spot", "BTC " + ROCKET + " - $67.4k",
    "| BTC | - $67.4k |", "BTC PnL ($5k) today", "PnL ($1,234.56) on the hedge",
    # F2: lakh and crore grouping
    "the mean rate is 12,34,567 today", "1,00,000 rows", "a 1,23,45,678 rupee budget",
    # F3: a symbol not glued to the digits, a currency word, a space before the suffix
    "BTC $ 67.4k", "€ 5m raised", "US$ 1.2B", "BTC 67.4k$", "5m€ raised", "BTC USD 67.4k", "BTC 67.4k USD",
    "BTC $ 99.9k", "BTC 99.9k USD", "BTC $67.4 k", "2.5M USDT bridged today",
    # m1: an ellipsis after a right number; m3: a three-decimal euro price
    "the drift rate 0.0666…", "BTC at $67.4k… waiting for the close", "Super E10 costs €1.789 per litre",
    # F8: before a sentence period (`SL 66,200.` is in texts.py); F9: word-led versions and counts the form rule does not name
    "BTC is at $84.17k.", "opus 4.8 and gemini 3.1", "Top 3 gainers today",
    # F7: a bare word magnitude reads its mantissa
    "a 1.2 billion users", "BTC 24h volume 3.4 million coins",
]

# (name, text, receipt | [receipts])
BINDING_CASES: list[tuple] = [
    # R1: the correctness review's false binds, and what binds right on the same responses
    ("FB01_mcap_vs_okx_ts", "BTC market cap is $1.79T today", OKX),
    ("FB02_mcap_vs_ws_event_time", "Total crypto market cap $1.8T", WS),
    ("FB03_volume_vs_seconds_ts", "ETH 24h volume $1.79B", {"eth": {"price": "2431.20", "ts": NOW_S}}),
    ("FB04_volume_vs_updated", "SOL volume hit $1.79B", {"sol": {"price": "148.20", "updated": NOW_S}}),
    ("FB05_mcap_vs_kline_open_time", "BTC market cap $1.8T", KLINES),
    ("FB06_liquidations_vs_trade_count", "BTC liquidations were $1.83M in 24h", TICK),
    ("FB07_price_vs_base_volume", "BTC is at $18.2k on Binance spot", TICK),
    ("FB08_wrong_price_vs_low", "BTC is at $66k on Binance spot", TICK),
    ("FB09_wrong_price_vs_high", "BTC is at $68.1k on Binance spot", TICK),
    ("FB10_wrong_price_vs_open", "BTC is at $66.8k on Binance spot", TICK),
    ("FB06_narrow", "BTC liquidations were $1.83M in 24h", NARROW),
    ("FB07_narrow", "BTC is at $18.2k on Binance spot", NARROW),
    ("FB08_narrow", "BTC is at $66k on Binance spot", NARROW),
    ("FB11_true_quote_volume", "BTC 24h volume $1.23B on Binance", TICK),
    ("FB12_no_percent_scaling", "BTC fees today $2.31k", TICK),
    ("FB13_tradeId", "BTC market cap $1.8T", {"btc": {"tradeId": 1800000000000}}),
    ("FB14_rounds_wrong", "BTC is at $67.5k on Binance", {"btc": {"price": "67412.50"}}),
    ("FB15_rounds_wrong_grouped", "BTC treasury holds $1,234.5k", {"btc": {"treasury": "1234567"}}),
    ("FB16_neg_vs_pos", "BTC PnL today -$5k", {"btc": {"pnl": "5000"}}),
    ("FB17_pos_vs_neg", "BTC PnL today $5k", {"btc": {"pnl": "-5000"}}),
    ("FB18_bare_word_vs_percent", "BTC 24h volume 3.4 million coins", {"btc": {"priceChangePercent": "3.4"}}),
    ("FB19_word_billion_vs_ratio", "BTC ETF inflows $1.2 billion", {"btc": {"ratio": "1.2"}}),
    ("FB20_range_lower_half_vs_rsi", "BTC range $60-70k this month, RSI near", {"btc": {"rsi": "60"}}),
    ("FB21_sub_67k_vs_low", "BTC slipped sub-$67k briefly", {"btc": {"low": "66980"}}),
    ("FB22_paren_vs_pos", "BTC PnL ($5k) today", {"btc": {"pnl": "5000"}}),
    ("FB22_paren_vs_neg", "BTC PnL ($5k) today", {"btc": {"pnl": "-5000"}}),
    ("F1_created", "BTC market cap is $1.8T today", {"data": [{"created": 1792345678901}]}),
    ("F1_last_updated", "BTC market cap $1.7T", {"k": {"last_updated": 1700000000000}}),
    ("F1_true_okx_last", "BTC is at $67.4k on OKX", OKX),
    ("F1_true_ws_c", "BTC is at $67.4k on Binance", WS),
    ("F1_kline_close_cell", "BTC is at $67.4k", KLINES),
    # R2: the regressions and the new wrong-sign readings, each with a right receipt
    ("FM01_t8_conversion", "Wallet balance: 0.5 BTC ($33,706)", {"wallet": {"btc": "0.5", "balance_usd": "33706"}}),
    ("FM02_conversion", "Holding 1 ETH ($2,431) in the vault", {"eth": {"price": "2431"}}),
    ("FM03_paren_price", "BTC ($67.4k) leads the majors", {"btc": {"price": "67412.50"}}),
    ("FM04_bold_dash", "**BTC** - $67,412.50", {"btc": {"price": "67412.50"}}),
    ("FM05_ticker_dash", "Bitcoin (BTC) - $67,412.50", {"btc": {"price": "67412.50"}}),
    ("FM06_nested_bullet", "- prices\n    - $67.4k BTC spot", {"btc": {"price": "67412.50"}}),
    ("FM07_euro_fuel", "Super E10 costs €1.789 per litre", {"fuel": {"e10_eur": "1.789"}}),
    ("FM08_ellipsis_right", "BTC at $67.4k… waiting for the close", {"btc": {"price": "67412.50"}}),
    ("FM09_emoji_dash", "BTC " + ROCKET + " - $67.4k", {"btc": {"price": "67412.50"}}),
    ("FM10_table_dash", "| BTC | - $67.4k |", {"btc": {"price": "67412.50"}}),
    ("m1_plain_ellipsis_right", "the drift rate 0.0666…", {"drift": {"rate": 0.0666}}),
    # what binds right (OK01-OK07)
    ("OK01", "BTC is at $67.4k on Binance spot", {"btc": {"price": "67412.50"}}),
    ("OK02", "BTC ETF inflows US$1.2B this week", {"btc": {"etf_inflow": "1210000000"}}),
    ("OK03", "ETH treasury €5m", {"eth": {"treasury_eur": "5000000"}}),
    ("OK04", "BTC PnL -$1,234.56 today", {"btc": {"pnl": "-1234.56"}}),
    ("OK05", "BTC 67 412.50 USDT", {"btc": {"price": "67412.50"}}),
    ("OK06", "SOL 15m candle | ETH $410b mcap", {"eth": {"mcap": "410000000000"}}),
    ("OK07", "BTC is at $99.9k.", {"btc": {"price": "99912.10"}}),
    # F2, F3, A-18 (`ETH $2,431` against 2431.27 is texts.py's `mark_eth_2431_int`)
    ("F2_lakh_rate", "the mean rate is 12,34,567 today", {"rate": 34567}),
    ("F3_spaced_madeup", "BTC $ 99.9k", NARROW),
    ("F3_trailing_word_madeup", "BTC 99.9k USD", NARROW),
    ("F3_space_before_suffix", "BTC $67.4 k", NARROW),
    ("A18_short_form", "ETH at $2.4k", {"eth": {"price": 2431.27}}),
]
