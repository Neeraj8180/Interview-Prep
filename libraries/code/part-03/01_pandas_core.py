# Pandas 2.2.x | Part 3: Pandas
# Topics: DataFrame operations, groupby, merge, memory optimization
# Run: python 01_pandas_core.py

import pandas as pd
import numpy as np
import sys

print("Pandas version:", pd.__version__)

# ============================================================
# 1. CREATING AND INSPECTING DATAFRAMES
# ============================================================
print("\n--- 1. Creating DataFrames ---")

np.random.seed(42)
n = 500

df = pd.DataFrame({
    "user_id":  np.random.randint(1, 100, n),
    "region":   np.random.choice(["North", "South", "East", "West"], n),
    "revenue":  np.random.exponential(500, n).round(2),
    "units":    np.random.randint(1, 50, n),
    "score":    np.random.uniform(0, 1, n).round(3),
    "label":    np.random.choice(["A", "B", "C", None], n),
})

print(f"Shape: {df.shape}")
print(f"Dtypes:\n{df.dtypes}")
print(f"Null counts:\n{df.isna().sum()}")
print(f"Memory: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

# ============================================================
# 2. SELECTION AND FILTERING
# ============================================================
print("\n--- 2. Selection and Filtering ---")

# .loc: label-based (names or boolean)
high_rev = df.loc[df["revenue"] > 1000, ["user_id", "revenue", "region"]]
print(f"High revenue rows: {len(high_rev)}")

# Multiple conditions: use & and |, NOT 'and'/'or'
premium = df[(df["score"] > 0.9) & (df["revenue"] > 500)]
print(f"Premium users: {len(premium)}")

# .iloc: position-based
first_5 = df.iloc[0:5, 0:3]
print(f"First 5 rows:\n{first_5}")

# ============================================================
# 3. GROUPBY — MOST IMPORTANT FOR INTERVIEWS
# ============================================================
print("\n--- 3. GroupBy ---")

# agg: reduce each group to summary statistics
summary = df.groupby("region").agg(
    total_revenue=("revenue", "sum"),
    avg_score=("score", "mean"),
    count=("user_id", "size"),
).round(2)
print(f"Summary by region:\n{summary}")

# transform: keep original shape, fill with group statistics
df["region_avg_rev"] = df.groupby("region")["revenue"].transform("mean")
df["revenue_zscore"] = (
    df.groupby("region")["revenue"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
)
print(f"Added group statistics: {df[['region', 'revenue', 'region_avg_rev']].head(3)}")

# ============================================================
# 4. MERGE (SQL-STYLE JOINS)
# ============================================================
print("\n--- 4. Merge/Join ---")

users = pd.DataFrame({
    "user_id": range(1, 6),
    "name":    ["Alice", "Bob", "Carol", "Dave", "Eve"],
    "tier":    ["gold", "silver", "gold", "bronze", "silver"],
})

orders = pd.DataFrame({
    "order_id":  range(101, 109),
    "user_id":   [1, 2, 1, 3, 5, 5, 2, 9],  # user 9 doesn't exist
    "amount":    [50, 30, 75, 20, 100, 40, 60, 25],
})

inner = pd.merge(users, orders, on="user_id", how="inner")
left  = pd.merge(users, orders, on="user_id", how="left")
print(f"Inner join: {len(inner)} rows | Left join: {len(left)} rows")

# ============================================================
# 5. MEMORY OPTIMIZATION
# ============================================================
print("\n--- 5. Memory Optimization ---")

before_mb = df.memory_usage(deep=True).sum() / 1e6
print(f"Before: {before_mb:.3f} MB")

# Downcast numeric types
df["user_id"] = df["user_id"].astype("int16")   # max 32767
df["units"] = df["units"].astype("int8")         # max 127
df["revenue"] = df["revenue"].astype("float32")  # half precision
df["score"] = df["score"].astype("float32")

# Convert low-cardinality strings to category
df["region"] = df["region"].astype("category")
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype("category")

after_mb = df.memory_usage(deep=True).sum() / 1e6
print(f"After:  {after_mb:.3f} MB")
print(f"Reduction: {(1 - after_mb/before_mb)*100:.0f}%")

# ============================================================
# 6. AVOID COMMON MISTAKES
# ============================================================
print("\n--- 6. Common Patterns ---")

# WRONG: apply when vectorized op exists (slow Python loop)
# df["upper"] = df["region"].astype(str).apply(str.upper)

# RIGHT: vectorized string operations (Rust-speed in Pandas 2.x with Arrow)
df["region_upper"] = df["region"].astype(str).str.upper()

# WRONG: concat in a loop (O(n²))
# result = pd.DataFrame()
# for batch in batches: result = pd.concat([result, batch])

# RIGHT: collect then concat once (O(n))
batches = [df.iloc[i:i+50] for i in range(0, len(df), 50)]
result = pd.concat(batches, ignore_index=True)  # single O(n) operation
print(f"Concat result shape: {result.shape}")

# ============================================================
# 7. FEATURE ENGINEERING PIPELINE
# ============================================================
print("\n--- 7. Feature Engineering ---")

def compute_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-user aggregate features — common in recommendation systems."""
    return df.groupby("user_id").agg(
        total_revenue=("revenue", "sum"),
        num_orders=("revenue", "count"),
        avg_score=("score", "mean"),
        max_score=("score", "max"),
    ).reset_index()

user_features = compute_user_features(df)
print(f"User features shape: {user_features.shape}")
print(user_features.head(3).to_string())

print("\nPandas demonstrations complete.")
