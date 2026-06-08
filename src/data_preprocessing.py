import pandas as pd
import numpy as np

print("=" * 60)
print("  Bitcoin Sentiment vs Trader Performance")
print("  Data Preprocessing Pipeline")
print("=" * 60)

# ---------------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------------
fear_greed = pd.read_csv("data/fear_greed_index.csv")
historical = pd.read_csv("data/historical_data.csv")

print(f"\n[1] Raw shapes:")
print(f"    fear_greed_index.csv  -> {fear_greed.shape}")
print(f"    historical_data.csv   -> {historical.shape}")

# ---------------------------------------------------------------------------
# 2. Clean fear / greed index
# ---------------------------------------------------------------------------
fear_greed["date"] = pd.to_datetime(fear_greed["date"], format="%Y-%m-%d")
fg_clean = fear_greed[["date", "value", "classification"]].copy()
print(f"    Fear/Greed date range      : {fg_clean['date'].min()} to {fg_clean['date'].max()}")

# ---------------------------------------------------------------------------
# 3. Clean historical data – extract trade_date from Timestamp IST
# ---------------------------------------------------------------------------
historical["Timestamp IST"] = pd.to_datetime(
    historical["Timestamp IST"], format="%d-%m-%Y %H:%M"
)
historical["trade_date"] = historical["Timestamp IST"].dt.date.astype(str)
historical["trade_date"] = pd.to_datetime(historical["trade_date"], format="%Y-%m-%d")

# ---------------------------------------------------------------------------
# 4. Merge on trade_date = date
# ---------------------------------------------------------------------------
merged = historical.merge(
    fg_clean, how="left", left_on="trade_date", right_on="date"
)

# ---------------------------------------------------------------------------
# 5. Clean column names – lowercase, spaces → underscores
# ---------------------------------------------------------------------------
merged.columns = [col.lower().replace(" ", "_") for col in merged.columns]

# ---------------------------------------------------------------------------
# 6. Drop failed merge rows (null sentiment / null closedpnl)
# ---------------------------------------------------------------------------
before_drop = len(merged)
merged = merged.dropna(subset=["value", "classification", "closed_pnl"])
after_drop = len(merged)

print(f"\n[2] After merge + drop nulls:")
print(f"    Rows before dropping nulls  : {before_drop}")
print(f"    Rows after dropping nulls   : {after_drop}")
print(f"    Dropped                     : {before_drop - after_drop}")
print(f"    Merged shape                : {merged.shape}")

# ---------------------------------------------------------------------------
# 7. Save
# ---------------------------------------------------------------------------
merged.to_csv("data/merged_data.csv", index=False)
print(f"\n[3] Saved → data/merged_data.csv")

# ---------------------------------------------------------------------------
# 8. Preview
# ---------------------------------------------------------------------------
print(f"\n[4] Preview (head of 5 rows):")
print(merged.head().to_string(index=False))
