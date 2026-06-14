# DuckDB analytical SQL — Part 30 code examples
# duckdb 1.2.x | Python 3.11+
# Install: pip install duckdb pandas numpy

import numpy as np
import pandas as pd
import time


def demo_zero_copy_queries():
    """Query Pandas DataFrames directly with DuckDB."""
    print("=" * 60)
    print("1. DuckDB Zero-Copy Queries over Pandas DataFrames")
    print("=" * 60)

    import duckdb

    # Generate dataset
    np.random.seed(42)
    N = 500_000
    df = pd.DataFrame({
        "user_id":   np.random.randint(1, 1001, N),
        "category":  np.random.choice(["electronics","clothing","food","travel","sports"], N),
        "amount":    np.abs(np.random.normal(75, 40, N)),
        "timestamp": pd.date_range("2024-01-01", periods=N, freq="1min")[:N],
        "is_fraud":  (np.random.rand(N) < 0.015).astype(int),
    })

    # DuckDB queries Pandas directly — no copy
    t0 = time.time()
    result = duckdb.sql("""
        SELECT
            category,
            COUNT(*)                  AS n_transactions,
            ROUND(AVG(amount), 2)     AS avg_amount,
            ROUND(SUM(amount), 2)     AS total_amount,
            ROUND(SUM(amount * is_fraud) / SUM(amount) * 100, 3) AS fraud_pct_amount,
            ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY amount), 2) AS p95_amount
        FROM df
        GROUP BY category
        ORDER BY total_amount DESC
    """).df()
    t_duckdb = time.time() - t0

    # Pandas equivalent
    t1 = time.time()
    pd_result = df.groupby("category").agg(
        n_transactions=("amount", "count"),
        avg_amount=("amount", "mean"),
        total_amount=("amount", "sum"),
    )
    t_pandas = time.time() - t1

    print(f"\nRows: {N:,}")
    print(f"DuckDB: {t_duckdb*1000:.1f}ms")
    print(f"Pandas: {t_pandas*1000:.1f}ms  ({t_pandas/t_duckdb:.1f}× slower)")
    print(f"\nResult:\n{result.to_string()}")
    return df


def demo_window_functions(df: pd.DataFrame):
    """Window functions for ML feature engineering."""
    print("\n" + "=" * 60)
    print("2. Window Functions for ML Feature Engineering")
    print("=" * 60)

    import duckdb

    features = duckdb.sql("""
        WITH ranked AS (
            SELECT
                user_id,
                amount,
                timestamp,
                ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY timestamp) AS seq_num,
                AVG(amount)  OVER (
                    PARTITION BY user_id
                    ORDER BY timestamp
                    ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
                ) AS rolling_10_avg,
                MAX(amount)  OVER (
                    PARTITION BY user_id
                    ORDER BY timestamp
                    ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
                ) AS rolling_30_max,
                LAG(amount, 1)  OVER (PARTITION BY user_id ORDER BY timestamp) AS prev_amount,
                SUM(amount)     OVER (PARTITION BY user_id) AS user_total
            FROM df
        )
        SELECT * FROM ranked
        WHERE seq_num > 30       -- only users with 30+ transactions
        ORDER BY user_id, timestamp
    """).df()

    print(f"Feature table shape: {features.shape}")
    print(features[["user_id", "amount", "rolling_10_avg", "prev_amount"]].head(5).to_string())


def demo_parquet(df: pd.DataFrame):
    """Write and read Parquet with DuckDB."""
    print("\n" + "=" * 60)
    print("3. Parquet Read/Write with DuckDB")
    print("=" * 60)

    import duckdb
    import tempfile, pathlib

    # Save subset to Parquet
    tmp = pathlib.Path(tempfile.mkdtemp())
    parquet_path = str(tmp / "transactions.parquet")
    df.head(10_000).to_parquet(parquet_path, index=False)
    print(f"Saved {parquet_path}")

    # DuckDB reads Parquet directly (column-pruning, predicate pushdown)
    result = duckdb.sql(f"""
        SELECT category, COUNT(*) AS n, ROUND(AVG(amount), 2) AS avg_amt
        FROM read_parquet('{parquet_path}')
        WHERE amount > 50
        GROUP BY category
        ORDER BY n DESC
    """).df()
    print(result.to_string())

    # Copy filtered result to new Parquet
    filtered_path = str(tmp / "filtered.parquet")
    duckdb.sql(f"""
        COPY (SELECT * FROM read_parquet('{parquet_path}') WHERE is_fraud = 1)
        TO '{filtered_path}' (FORMAT PARQUET)
    """)
    fraud_df = pd.read_parquet(filtered_path)
    print(f"\nFraud transactions saved: {len(fraud_df)}")


def demo_performance_comparison():
    """DuckDB vs Pandas for different operations."""
    print("\n" + "=" * 60)
    print("4. Performance Comparison: DuckDB vs Pandas")
    print("=" * 60)

    import duckdb

    N = 2_000_000
    df = pd.DataFrame({
        "cat":   np.random.choice(list("ABCDE"), N),
        "val":   np.random.rand(N),
        "group": np.random.randint(0, 100, N),
    })

    tests = [
        ("GROUP BY (5 groups)",
         lambda: df.groupby("cat")["val"].agg(["mean","sum","count"]),
         lambda: duckdb.sql("SELECT cat, AVG(val), SUM(val), COUNT(*) FROM df GROUP BY cat")),
        ("FILTER + GROUP BY",
         lambda: df[df["val"] > 0.5].groupby("cat")["val"].mean(),
         lambda: duckdb.sql("SELECT cat, AVG(val) FROM df WHERE val > 0.5 GROUP BY cat")),
        ("JOIN (self join on group)",
         lambda: df.merge(df.groupby("group")["val"].mean().reset_index().rename(
                          columns={"val":"group_avg"}), on="group"),
         lambda: duckdb.sql("SELECT t.*, g.group_avg FROM df t JOIN ("
                            "SELECT group, AVG(val) AS group_avg FROM df GROUP BY group) g "
                            "ON t.group = g.group")),
    ]

    print(f"\n  N={N:,} rows")
    print(f"  {'Operation':<30} {'Pandas ms':>12} {'DuckDB ms':>12} {'Speedup':>8}")
    print("  " + "-" * 65)
    for name, pandas_fn, duckdb_fn in tests:
        t0 = time.time(); pandas_fn(); t_pd = (time.time() - t0) * 1000
        t1 = time.time(); duckdb_fn(); t_db = (time.time() - t1) * 1000
        print(f"  {name:<30} {t_pd:>10.1f}  {t_db:>10.1f}  {t_pd/max(t_db,0.1):>6.1f}×")


if __name__ == "__main__":
    df = demo_zero_copy_queries()
    demo_window_functions(df)
    demo_parquet(df)
    demo_performance_comparison()
