# PyArrow 16.x | Part 5: PyArrow
# Topics: Arrow arrays, Parquet I/O, compute functions, datasets streaming
# Run: python 01_pyarrow_core.py

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow.dataset as ds
import numpy as np
import tempfile, os, shutil

print("PyArrow version:", pa.__version__)

# ============================================================
# 1. ARROW ARRAYS: typed, null-safe, zero-copy
# ============================================================
print("\n--- 1. Arrow Arrays ---")

# Create arrays
int_arr   = pa.array([1, 2, None, 4], type=pa.int32())
float_arr = pa.array([1.0, 2.5, None, 4.0], type=pa.float32())
str_arr   = pa.array(["hello", "world", None, "arrow"])

print(f"int_arr:   {int_arr}")
print(f"Null count: {int_arr.null_count}")   # 1
print(f"Type:      {int_arr.type}")           # int32

# True null (not float NaN) — works for integers too
print(f"Element 2 is null: {int_arr[2].is_valid}")  # False

# Zero-copy from NumPy (for contiguous numeric arrays)
rng = np.random.default_rng(42)
np_arr = rng.standard_normal(1_000_000).astype(np.float32)
arrow_arr = pa.array(np_arr)  # no copy for contiguous C-order float arrays
print(f"NumPy → Arrow zero-copy: {np_arr.ctypes.data == arrow_arr.buffers()[1].address}")

# ============================================================
# 2. RECORD BATCHES AND TABLES
# ============================================================
print("\n--- 2. RecordBatch vs Table ---")

# RecordBatch: single contiguous chunk (streaming unit)
schema = pa.schema([
    pa.field("id",    pa.int32(),   nullable=False),
    pa.field("score", pa.float32(), nullable=True),
    pa.field("label", pa.string(),  nullable=True),
])
batch = pa.record_batch(
    {"id": [1, 2, 3], "score": [0.9, None, 0.7], "label": ["A", "B", None]},
    schema=schema
)
print(f"RecordBatch: {batch.num_rows} rows, schema={batch.schema}")

# Table: collection of ChunkedArrays (full in-memory result)
n = 10_000
table = pa.table({
    "id":    pa.array(range(n), type=pa.int32()),
    "score": pa.array(rng.uniform(0, 1, n).astype(np.float32)),
    "label": pa.array(rng.choice(["A", "B", "C"], n)),
})
print(f"Table: {table.num_rows} rows, {table.num_columns} columns")
print(f"Schema:\n{table.schema}")
print(f"Size: {table.nbytes / 1024:.1f} KB")

# ============================================================
# 3. PARQUET: COLUMN PRUNING AND PREDICATE PUSHDOWN
# ============================================================
print("\n--- 3. Parquet I/O ---")

tmpdir = tempfile.mkdtemp()
parquet_path = os.path.join(tmpdir, "data.parquet")

# Write with metadata
pq.write_table(table, parquet_path,
               compression="snappy",
               row_group_size=1000)  # 10 row groups of 1000 rows each

# Read: full table
full = pq.read_table(parquet_path)
print(f"Full read: {full.num_rows} rows, {full.num_columns} cols")

# Read: column pruning — only reads 2 columns from disk
pruned = pq.read_table(parquet_path, columns=["id", "score"])
print(f"Pruned:    {pruned.num_rows} rows, {pruned.num_columns} cols")

# Read: predicate pushdown — filters at row group level
filtered = pq.read_table(parquet_path,
    columns=["id", "score"],
    filters=[("score", ">", 0.9)]
)
print(f"Filtered:  {filtered.num_rows} rows (score > 0.9)")

# ============================================================
# 4. COMPUTE FUNCTIONS
# ============================================================
print("\n--- 4. Compute Functions ---")

scores = table.column("score")

# Aggregation
total = pc.sum(scores)
mean  = pc.mean(scores)
print(f"Sum: {total.as_py():.2f}, Mean: {mean.as_py():.4f}")

# Filtering
mask = pc.greater(scores, 0.9)
high_scores = pc.filter(scores, mask)
print(f"High scores (>0.9): {len(high_scores)} values")

# Sorting
sort_idx = pc.sort_indices(scores, sort_keys=[("x", "descending")])

# String operations (fast Arrow string kernels)
labels = table.column("label")
lower = pc.utf8_lower(labels)
lengths = pc.utf8_length(labels)
print(f"Label types: {labels.type}, lower sample: {lower[0].as_py()}")

# ============================================================
# 5. DATASETS API: STREAMING LARGE FILES
# ============================================================
print("\n--- 5. Datasets Streaming ---")

# Write a partitioned Parquet dataset
part_dir = os.path.join(tmpdir, "partitioned")
pq.write_to_dataset(table, part_dir, partition_cols=["label"])

# Stream in batches — never loads full dataset into RAM
dataset = ds.dataset(part_dir, format="parquet")
print(f"Dataset schema: {dataset.schema}")

total_rows = 0
total_sum = 0.0
for batch in dataset.to_batches(batch_size=500, columns=["id", "score"]):
    total_rows += len(batch)
    total_sum  += pc.sum(batch.column("score")).as_py()

print(f"Streamed {total_rows} rows in batches of 500")
print(f"Streaming sum: {total_sum:.2f} vs full sum: {pc.sum(scores).as_py():.2f}")

# ============================================================
# 6. ZERO-COPY INTEROP: Arrow ↔ Pandas ↔ Polars ↔ NumPy
# ============================================================
print("\n--- 6. Zero-Copy Interop ---")
import pandas as pd
import polars as pl

# Arrow → Pandas (Arrow backend = zero-copy for numeric cols)
pd_df = table.to_pandas(types_mapper=pd.ArrowDtype)
print(f"Arrow → Pandas: {pd_df.dtypes['score']}")   # ArrowDtype(float)

# Arrow → Polars (zero-copy — Polars uses Arrow natively)
pl_df = pl.from_arrow(table)
print(f"Arrow → Polars: {pl_df['score'].dtype}")    # Float32

# Polars → Arrow
arrow_back = pl_df.to_arrow()
print(f"Polars → Arrow: {arrow_back.schema['score']}")

# NumPy (float32) → Arrow (zero-copy)
np_scores = np_arr[:100]
arrow_from_np = pa.array(np_scores)
print(f"NumPy → Arrow (float32 contiguous = zero-copy): {arrow_from_np.type}")

# Arrow numeric → NumPy (zero-copy)
np_from_arrow = table.column("score").to_numpy(zero_copy_only=True)
print(f"Arrow → NumPy zero-copy: True (no copy made)")

# Cleanup
shutil.rmtree(tmpdir)

print("\nPyArrow demonstrations complete.")
