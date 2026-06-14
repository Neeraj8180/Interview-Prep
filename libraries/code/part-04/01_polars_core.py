# Polars 0.20+ | Part 4: Polars
# Topics: Lazy evaluation, expressions, groupby, join, streaming
# Run: python 01_polars_core.py

import polars as pl
import numpy as np
import time

print("Polars version:", pl.__version__)

# ============================================================
# 1. LAZY VS EAGER
# ============================================================
print("\n--- 1. Lazy vs Eager ---")

# Create a sample DataFrame
rng = np.random.default_rng(42)
n = 100_000
df = pl.DataFrame({
    "user_id":  rng.integers(0, 1000, n),
    "region":   rng.choice(["North", "South", "East", "West"], n),
    "amount":   rng.exponential(100, n).astype(np.float32),
    "score":    rng.uniform(0, 1, n).astype(np.float32),
})

# Eager: executes immediately
t0 = time.perf_counter()
result_eager = df.filter(pl.col("score") > 0.9).group_by("region").agg(
    pl.col("amount").sum().alias("total")
)
t1 = time.perf_counter()

# Lazy: builds a query plan, optimizes, then executes
t2 = time.perf_counter()
result_lazy = (
    df.lazy()
    .filter(pl.col("score") > 0.9)
    .group_by("region")
    .agg(pl.col("amount").sum().alias("total"))
    .collect()
)
t3 = time.perf_counter()

print(f"Eager:  {(t1-t0)*1000:.2f}ms")
print(f"Lazy:   {(t3-t2)*1000:.2f}ms")
print(f"Result:\n{result_lazy.sort('region')}")

# ============================================================
# 2. EXPRESSIONS — THE CORE API
# ============================================================
print("\n--- 2. Expressions ---")

# Expressions are composable objects describing transformations
df_small = pl.DataFrame({
    "name":   ["Alice", "Bob", "Carol"],
    "score":  [0.92, 0.67, 0.85],
    "amount": [100.0, 200.0, 150.0],
})

# select: returns only specified columns/expressions
result = df_small.select([
    pl.col("name").str.to_uppercase().alias("NAME"),
    (pl.col("score") * 100).round(1).alias("pct_score"),
    pl.col("amount").log1p().alias("log_amount"),
])
print(result)

# with_columns: adds new columns (keeps existing)
df_small = df_small.with_columns([
    pl.when(pl.col("score") > 0.8)
      .then(pl.lit("high"))
      .otherwise(pl.lit("low"))
      .alias("tier"),
    (pl.col("amount") / pl.col("amount").sum()).alias("amount_pct"),
])
print(df_small)

# ============================================================
# 3. GROUPBY AND WINDOW FUNCTIONS
# ============================================================
print("\n--- 3. GroupBy and Window Functions ---")

df_grp = pl.DataFrame({
    "region":  ["N", "N", "S", "S", "N", "S"] * 5,
    "revenue": list(range(30)),
    "cost":    list(range(1, 31)),
})

# GroupBy aggregation
summary = df_grp.group_by("region").agg([
    pl.col("revenue").sum().alias("total_rev"),
    pl.col("revenue").mean().alias("avg_rev"),
    pl.col("cost").mean().alias("avg_cost"),
    pl.len().alias("n_rows"),
])
print(f"GroupBy summary:\n{summary.sort('region')}")

# Window function with over() — equivalent to groupby + transform
df_grp = df_grp.with_columns([
    # Mean revenue per region (broadcast back to all rows)
    pl.col("revenue").mean().over("region").alias("region_avg_rev"),
    # Normalized revenue within region
    ((pl.col("revenue") - pl.col("revenue").mean().over("region"))
     / pl.col("revenue").std().over("region"))
    .alias("rev_normalized"),
    # Rank within region
    pl.col("revenue").rank(descending=True).over("region").alias("rank"),
])
print(f"With window functions:\n{df_grp.head(6)}")

# ============================================================
# 4. JOIN
# ============================================================
print("\n--- 4. Joins ---")

users = pl.DataFrame({
    "user_id": [1, 2, 3, 4],
    "tier":    ["gold", "silver", "gold", "bronze"],
})
orders = pl.DataFrame({
    "order_id": [101, 102, 103, 104, 105],
    "user_id":  [1, 2, 1, 3, 99],    # user 99 doesn't exist
    "amount":   [50.0, 30.0, 75.0, 20.0, 10.0],
})

inner = users.join(orders, on="user_id", how="inner")
left  = users.join(orders, on="user_id", how="left")
print(f"Inner: {inner.shape[0]} rows | Left: {left.shape[0]} rows")
print(f"Inner:\n{inner}")

# ============================================================
# 5. LAZY SCAN WITH PARQUET (simulate with temp file)
# ============================================================
print("\n--- 5. Lazy Parquet Scan ---")

import tempfile, os

# Write to temp parquet
tmp = tempfile.mktemp(suffix=".parquet")
df.write_parquet(tmp)

# Lazy scan — executes with predicate pushdown
result = (
    pl.scan_parquet(tmp)
    .filter(pl.col("amount") > 200.0)   # pushed to scan
    .select(["user_id", "amount"])        # projection pushdown
    .group_by("user_id")
    .agg(pl.col("amount").sum().alias("total_amount"))
    .sort("total_amount", descending=True)
    .head(5)
    .collect()
)
print(f"Top 5 users by amount:\n{result}")
os.remove(tmp)

# ============================================================
# 6. PERFORMANCE COMPARISON: Polars vs Pandas
# ============================================================
print("\n--- 6. Performance: Polars vs Pandas ---")
import pandas as pd

n = 1_000_000
data = {
    "a": rng.integers(0, 100, n),
    "b": rng.uniform(0, 1000, n).astype(np.float32),
    "cat": rng.choice(["X", "Y", "Z"], n),
}

pd_df = pd.DataFrame(data)
pl_df = pl.DataFrame(data)

# Pandas groupby
t0 = time.perf_counter()
pd_result = pd_df.groupby("cat")["b"].mean()
t1 = time.perf_counter()

# Polars groupby
t2 = time.perf_counter()
pl_result = pl_df.group_by("cat").agg(pl.col("b").mean())
t3 = time.perf_counter()

print(f"Pandas groupby: {(t1-t0)*1000:.1f}ms")
print(f"Polars groupby: {(t3-t2)*1000:.1f}ms")
print(f"Speedup: ~{(t1-t0)/(t3-t2):.1f}x")

print("\nPolars demonstrations complete.")
