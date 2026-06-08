# Bitcoin Market Sentiment vs. Trader Performance

An analysis of how Bitcoin Fear & Greed sentiment impacts Hyperliquid trader profitability, position sizing, and market direction.

## Objective

This project answers a core question for any crypto trading firm: **Does market sentiment predict trader behavior and profitability, and can that signal be systematically exploited?** By merging on-chain Bitcoin Fear & Greed Index data with Hyperliquid trader-level historical data, we surface actionable patterns in how traders position, size, and perform across fear and greed regimes. For a prop desk or algorithmic trading operation, understanding these behavioral asymmetries means knowing when to lean in, when to hedge, and when to sit out entirely.

## Key Findings

- **Both sentiment extremes outperform the middle.** Extreme Greed has the highest win rate (89%, mean PnL $130/trade) and Fear follows closely (76%, $113/trade) — both significantly above Neutral ($71/trade). The edge doesn't come from one direction; it comes from having a clear market mood to trade against. **Strategy implication:** Allocate more capital and widen targets during sentiment extremes. Tighten risk and reduce size when the market is Neutral.
- **Traders are contrarian by instinct.** During Fear, 62% of trades are Long. During Greed, the mix flips to 58% Short. The crowd consistently buys the dip and fades the top. **Strategy implication:** The contrarian trade — long during fear, short during greed — has historically been the winning side. Systematic strategies aligned with this bias would have captured consistent alpha across both regimes.
- **Sizing matches conviction.** Average trade size peaks at $7.8K during Fear (traders "double down" on dips) and shrinks to $3.1K during Extreme Greed (cautious profit‑taking). **Strategy implication:** Follow the smart money's sizing pattern — scale into fear, scale out of euphoria.
- **Top traders exploit every sentiment regime.** The #1 trader earned $2.1M with a 79% win rate. The cross‑sentiment heatmap shows that elite traders maintain high win rates consistently across Fear, Greed, and everything in between — they don't rely on a single market condition. **Strategy implication:** Model your entries and risk management after top‑tier traders rather than the aggregate crowd.
- **PnL can spike when sentiment is muted.** Dec 2024 – Mar 2025 produced the largest PnL spikes of the dataset despite Fear & Greed scores trending lower. On‑chain trading opportunities can decouple from mainstream sentiment indexes. **Strategy implication:** Supplement Fear & Greed with on‑chain volume and volatility metrics to avoid missing high‑alpha periods.
- **Neutral sentiment is a dead zone.** Neutral classification has the lowest mean PnL ($71) of any regime. Traders struggle to find edges when sentiment lacks conviction. **Strategy implication:** Reduce capital allocation and tighten stop levels during Neutral periods; focus on regimes with stronger directional bias.

## Dataset

- **Fear & Greed Index:** 2,644 rows, 2018-02-01 to 2025-05-02, columns: `date`, `value`, `classification`
- **Historical Trader Data:** 211,224 rows, late 2024 to early 2025, columns: `Account`, `Coin`, `Execution Price`, `Size USD`, `Side`, `Direction`, `Closed PnL`, `Timestamp IST`

## Analysis Breakdown

1. **PnL by Sentiment** (`pnl_by_sentiment.png`): Shows mean and median PnL across sentiment classifications, confirming that Extreme Greed outperforms all other regimes.
2. **Long/Short Ratio** (`long_short_ratio_by_sentiment.png`): Reveals the contrarian positioning behavior — traders go long into Fear and short into Greed.
3. **Trade Size by Sentiment** (`trade_size_by_sentiment.png`): Demonstrates how average position sizing changes with market sentiment, spiking during Fear.
4. **Monthly PnL vs FG Score** (`monthly_pnl_vs_fg_score.png`): Dual‑axis chart correlating aggregate trader PnL with sentiment over time, highlighting the Dec 2024 – Mar 2025 decoupling.
5. **Top 10 Traders** (`top_10_traders_pnl.png`): Identifies the most profitable traders and their win rates — the #1 trader earned $2.1M at 79%.
6. **Win Rate Heatmap** (`top_traders_winrate_heatmap.png`): Shows how top traders perform across all sentiment regimes, with NaN cells (grey) representing no‑trade zones.

## How to Run

```bash
pip install -r requirements.txt
python src/data_preprocessing.py
python src/analysis.py
jupyter notebook notebooks/analysis.ipynb
```

## Repo Structure

```
.
├── data/
│   ├── fear_greed_index.csv
│   ├── historical_data.csv
│   └── merged_data.csv
├── notebooks/
│   └── analysis.ipynb
├── src/
│   ├── data_preprocessing.py
│   └── analysis.py
├── outputs/
│   └── charts/         # 6 analysis PNGs + supporting visuals
├── requirements.txt
└── README.md
```

## Tech Stack

- **pandas** — data loading, merging (inner join on date), filtering, and group‑by aggregations throughout the pipeline
- **numpy** — vectorized PnL classification and descriptive statistics across trader cohorts
- **matplotlib** — base plotting framework for all 6 analysis charts and custom figure styling
- **seaborn** — heatmap and statistical plots (boxen, countplot) with built‑in color palettes
- **jupyter** — interactive exploration notebook (`notebooks/analysis.ipynb`) for ad‑hoc slicing
