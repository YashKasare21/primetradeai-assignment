import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

ROOT = "/home/yash/Desktop/Projects/primetrade-ai"
df = pd.read_csv(f"{ROOT}/data/merged_data.csv")

sentiment_order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
closed = df[df["closed_pnl"] != 0].copy()
open_trades = df[df["direction"].isin(["Open Long", "Open Short"])].copy()
strategy_trades = df[~df["direction"].isin([
    "Spot Dust Conversion", "Settlement", "Buy", "Sell"
])].copy()

print(f"Filter sizes:  closed={len(closed)},  open_trades={len(open_trades)},  strategy_trades={len(strategy_trades)}")

# ===========================================================================
# ANALYSIS 1: Win rate, mean PnL, median PnL by sentiment
# ===========================================================================
print("\n" + "=" * 65)
print("ANALYSIS 1: Win Rate, Mean PnL, Median PnL by Sentiment")
print("=" * 65)

a1 = closed.copy()
a1["is_win"] = a1["closed_pnl"] > 0

stats = a1.groupby("classification").agg(
    win_rate=("is_win", "mean"),
    mean_pnl=("closed_pnl", "mean"),
    median_pnl=("closed_pnl", "median"),
    trade_count=("closed_pnl", "count"),
).reindex(sentiment_order)

print(stats.to_string(float_format=lambda x: f"{x:,.2f}"))

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(sentiment_order))
w = 0.35
ax.bar(x - w / 2, stats["mean_pnl"].values, w, label="Mean PnL", color="steelblue")
ax.bar(x + w / 2, stats["median_pnl"].values, w, label="Median PnL", color="coral")
ax.set_xticks(x)
ax.set_xticklabels(sentiment_order)
ax.set_xlabel("Sentiment")
ax.set_ylabel("Closed PnL")
ax.set_title("Mean and Median PnL by Market Sentiment")
ax.legend()
plt.tight_layout()
plt.savefig(f"{ROOT}/outputs/charts/pnl_by_sentiment.png", dpi=150)
plt.close()

# ===========================================================================
# ANALYSIS 2: Long/Short ratio by sentiment
# ===========================================================================
print("\n" + "=" * 65)
print("ANALYSIS 2: Long / Short Ratio by Sentiment")
print("=" * 65)

a2 = open_trades.copy()
a2["position_type"] = np.where(a2["direction"] == "Open Long", "Long", "Short")
ct = pd.crosstab(a2["classification"], a2["position_type"]).reindex(sentiment_order, fill_value=0)
pct = ct.div(ct.sum(axis=1), axis=0).fillna(0)

merge_table = ct.copy()
merge_table["Long %"] = (pct["Long"] * 100).round(1)
merge_table["Short %"] = (pct["Short"] * 100).round(1)
merge_table["Total"] = ct.sum(axis=1)
print(merge_table.to_string())

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(sentiment_order, pct["Long"].values, label="Long", color="seagreen")
ax.bar(sentiment_order, pct["Short"].values, bottom=pct["Long"].values, label="Short", color="tomato")
ax.set_ylabel("Proportion")
ax.set_xlabel("Sentiment")
ax.set_title("Long / Short Ratio by Market Sentiment")
ax.legend()
plt.tight_layout()
plt.savefig(f"{ROOT}/outputs/charts/long_short_ratio_by_sentiment.png", dpi=150)
plt.close()

# ===========================================================================
# ANALYSIS 3: Average trade size (size_usd) by sentiment
# ===========================================================================
print("\n" + "=" * 65)
print("ANALYSIS 3: Average Trade Size (USD) by Sentiment")
print("=" * 65)

a3 = df.groupby("classification")["size_usd"].mean().reindex(sentiment_order)
print(a3.to_string(float_format=lambda x: f"${x:,.2f}"))

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(sentiment_order, a3.values, color="mediumpurple")
ax.set_xlabel("Sentiment")
ax.set_ylabel("Average Size (USD)")
ax.set_title("Average Trade Size by Market Sentiment")
for i, v in enumerate(a3.values):
    ax.text(i, v, f"${v:,.0f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(f"{ROOT}/outputs/charts/trade_size_by_sentiment.png", dpi=150)
plt.close()

# ===========================================================================
# ANALYSIS 4: Monthly total PnL vs average Fear/Greed score
# ===========================================================================
print("\n" + "=" * 65)
print("ANALYSIS 4: Monthly Total PnL vs Avg Fear/Greed Score")
print("=" * 65)

df["trade_date"] = pd.to_datetime(df["trade_date"])
df["year_month"] = df["trade_date"].dt.strftime("%Y-%m")
monthly = df.groupby("year_month").agg(
    total_pnl=("closed_pnl", "sum"),
    avg_fg=("value", "mean"),
).sort_index()
print(monthly.to_string(float_format=lambda x: f"{x:,.2f}"))

fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.bar(range(len(monthly)), monthly["total_pnl"].values, color="steelblue", alpha=0.7, label="Total PnL")
ax1.set_ylabel("Total Closed PnL", color="steelblue")
ax1.tick_params(axis="y", labelcolor="steelblue")

ax2 = ax1.twinx()
ax2.plot(range(len(monthly)), monthly["avg_fg"].values, color="darkorange", marker="o", linewidth=1.5, label="Avg FG Score")
ax2.set_ylabel("Avg Fear/Greed Score", color="darkorange")
ax2.tick_params(axis="y", labelcolor="darkorange")
ax2.set_ylim(0, 100)

ax1.set_xticks(range(len(monthly)))
ax1.set_xticklabels(monthly.index, rotation=45, ha="right", fontsize=7)
ax1.set_xlabel("Year-Month")
ax1.set_title("Monthly Total PnL vs Average Fear/Greed Score")
fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
plt.tight_layout()
plt.savefig(f"{ROOT}/outputs/charts/monthly_pnl_vs_fg_score.png", dpi=150)
plt.close()

# ===========================================================================
# ANALYSIS 5: Top 10 traders by total PnL
# ===========================================================================
print("\n" + "=" * 65)
print("ANALYSIS 5: Top 10 Traders by Total PnL")
print("=" * 65)

a5 = closed.groupby("account")["closed_pnl"].agg(["sum", "count"]).rename(columns={"sum": "total_pnl", "count": "trades"})
a5["win_rate"] = closed.groupby("account")["closed_pnl"].apply(lambda x: (x > 0).mean())
top10 = a5.nlargest(10, "total_pnl")

print(f"{'Account':<50} {'Total PnL':>12} {'Trades':>8} {'Win Rate':>10}")
print("-" * 80)
for acc, row in top10.iterrows():
    print(f"{acc:<50} {row['total_pnl']:>12,.2f} {row['trades']:>8,} {row['win_rate']:>9.1%}")

fig, ax = plt.subplots(figsize=(10, 6))
y_pos = range(len(top10))
bars = ax.barh(list(top10.index.str[:50]), top10["total_pnl"].values, color="teal")
ax.set_xlabel("Total Closed PnL")
ax.set_title("Top 10 Traders by Total PnL")
ax.invert_yaxis()
for i, (bar, idx) in enumerate(zip(bars, top10.index)):
    wr = top10.loc[idx, "win_rate"]
    ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
            f"  Win: {wr:.1%}", va="center", fontsize=8)
plt.tight_layout()
plt.savefig(f"{ROOT}/outputs/charts/top_10_traders_pnl.png", dpi=150)
plt.close()

# ===========================================================================
# ANALYSIS 6: Win rate heatmap for Top 10 traders x sentiment
# ===========================================================================
print("\n" + "=" * 65)
print("ANALYSIS 6: Win Rate Heatmap — Top 10 Traders × Sentiment")
print("=" * 65)

top_accounts = top10.index.tolist()
a6 = closed[closed["account"].isin(top_accounts)].copy()
a6["is_win"] = a6["closed_pnl"] > 0

pivot = a6.groupby(["account", "classification"])["is_win"].mean().unstack()
pivot = pivot.reindex(columns=sentiment_order)

print(pivot.to_string(float_format=lambda x: f"{x:.2%}"))

fig, ax = plt.subplots(figsize=(10, max(6, len(top_accounts) * 0.5 + 2)))
sns.heatmap(pivot, annot=True, fmt=".0%", cmap="RdYlGn", mask=pivot.isnull(),
            cbar_kws={"label": "Win Rate"}, linewidths=0.5, ax=ax, vmin=0, vmax=1)
ax.set_title("Win Rate by Trader and Sentiment")
ax.set_xlabel("Sentiment")
ax.set_ylabel("Account")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{ROOT}/outputs/charts/top_traders_winrate_heatmap.png", dpi=150)
plt.close()

print("\nAll 6 analyses complete. Charts saved to outputs/charts/")
