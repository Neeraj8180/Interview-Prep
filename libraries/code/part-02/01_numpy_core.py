# NumPy 2.x | Part 2: NumPy
# Topics: arrays, broadcasting, einsum, memory layout, vectorization
# Run: python 01_numpy_core.py

"""
Executable demonstrations of every major NumPy concept.
All examples print expected output so you can verify correctness.
"""

import numpy as np
import time

print("NumPy version:", np.__version__)
print("=" * 60)


# ============================================================
# 1. ARRAY CREATION AND METADATA
# ============================================================
print("\n--- 1. Array Creation ---")

rng = np.random.default_rng(42)   # reproducible RNG (NOT np.random.seed!)

# Different ways to create arrays
a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
z = np.zeros((3, 4), dtype=np.float32)
ones = np.ones(5)
eye = np.eye(3)
r = np.arange(0, 10, 2)            # [0, 2, 4, 6, 8]
linsp = np.linspace(0, 1, 5)       # [0.0, 0.25, 0.5, 0.75, 1.0]
rand = rng.standard_normal((3, 3)).astype(np.float32)

print(f"a.shape:   {a.shape}")     # (2, 3)
print(f"a.dtype:   {a.dtype}")     # float32
print(f"a.ndim:    {a.ndim}")      # 2
print(f"a.size:    {a.size}")      # 6
print(f"a.nbytes:  {a.nbytes}")    # 24 (6 * 4 bytes)
print(f"a.strides: {a.strides}")   # (12, 4) — 3*4 bytes per row, 4 bytes per col


# ============================================================
# 2. INDEXING: views vs copies
# ============================================================
print("\n--- 2. Indexing (Views vs Copies) ---")

data = np.arange(20, dtype=np.float32).reshape(4, 5)
# [[ 0  1  2  3  4]
#  [ 5  6  7  8  9]
#  [10 11 12 13 14]
#  [15 16 17 18 19]]

# Basic slicing → VIEW (shares memory)
view = data[1:3, :]
view[0, 0] = 999.0
print(f"data[1, 0] after modifying view: {data[1, 0]}")  # 999.0 — same memory!
view[0, 0] = 5.0   # restore

# Fancy indexing → COPY (independent memory)
copy = data[[0, 2], :]              # select rows 0 and 2
copy[0, 0] = 999.0
print(f"data[0, 0] after modifying copy: {data[0, 0]}")  # 0.0 — unchanged!

# Boolean indexing → COPY
mask = data > 10
large_values = data[mask]           # shape depends on how many values > 10
print(f"Values > 10: {large_values}")
# Expected: [ 11.  12.  13.  14.  15.  16.  17.  18.  19.]

# np.where: vectorized conditional selection
result = np.where(data > 10, data, 0.0)
print(f"data where > 10, else 0: {result[2]}")
# Expected: [ 0.  0.  12.  13.  14.]  (row 2)


# ============================================================
# 3. BROADCASTING
# ============================================================
print("\n--- 3. Broadcasting ---")

# Batch normalization via broadcasting
batch = rng.standard_normal((32, 128)).astype(np.float32)

mean = batch.mean(axis=0)         # shape (128,)   — mean per feature
std  = batch.std(axis=0)          # shape (128,)

# Broadcasting rule: (128,) aligns with (32, 128) from the right
# The (128,) is treated as (1, 128) and stretched to (32, 128)
normalized = (batch - mean) / (std + 1e-8)

print(f"batch shape:      {batch.shape}")        # (32, 128)
print(f"mean shape:       {mean.shape}")          # (128,)
print(f"normalized shape: {normalized.shape}")    # (32, 128)
print(f"normalized means: {normalized.mean(axis=0)[:3].round(4)}")  # ~[0, 0, 0]
print(f"normalized stds:  {normalized.std(axis=0)[:3].round(4)}")   # ~[1, 1, 1]

# Broadcasting failure case
try:
    a = np.zeros((3, 4))
    b = np.zeros((3, 5))
    c = a + b       # 4 ≠ 5 and neither is 1 → error
except ValueError as e:
    print(f"Expected broadcast error: {e}")


# ============================================================
# 4. EINSUM: the Swiss Army knife
# ============================================================
print("\n--- 4. Einstein Summation (einsum) ---")

A = rng.standard_normal((4, 3)).astype(np.float32)
B = rng.standard_normal((3, 5)).astype(np.float32)

# Matrix multiply (j is contracted — summed over)
C_einsum = np.einsum("ij,jk->ik", A, B)
C_matmul = A @ B
print(f"einsum == matmul: {np.allclose(C_einsum, C_matmul)}")  # True

# Batched attention scores: Q @ K^T
batch, heads, seq, d = 2, 4, 8, 16
Q = rng.standard_normal((batch, heads, seq, d)).astype(np.float32)
K = rng.standard_normal((batch, heads, seq, d)).astype(np.float32)

# "bhqd,bhkd->bhqk" — contract over d, keep all others
scores = np.einsum("bhqd,bhkd->bhqk", Q, K) / np.sqrt(d)
print(f"Attention scores shape: {scores.shape}")   # (2, 4, 8, 8)

# Per-row squared norm (batched dot product)
X = rng.standard_normal((8, 16)).astype(np.float32)
sq_norms_einsum = np.einsum("id,id->i", X, X)      # (8,)
sq_norms_manual = (X * X).sum(axis=-1)              # (8,)
print(f"Norms match: {np.allclose(sq_norms_einsum, sq_norms_manual)}")  # True

# Outer product
u = np.array([1., 2., 3.])
v = np.array([4., 5.])
outer = np.einsum("i,j->ij", u, v)    # (3, 2)
print(f"Outer product:\n{outer}")
# [[4. 5.]
#  [8. 10.]
#  [12. 15.]]


# ============================================================
# 5. MEMORY LAYOUT AND PERFORMANCE
# ============================================================
print("\n--- 5. Memory Layout ---")

a = rng.standard_normal((100, 50)).astype(np.float32)

# Transpose creates a view with swapped strides — NO copy
a_T = a.T
print(f"a.strides:     {a.strides}")      # (200, 4) — row-major
print(f"a.T.strides:   {a_T.strides}")    # (4, 200) — column-major
print(f"a.T shares memory with a: {np.shares_memory(a, a_T)}")  # True

# Not C-contiguous after transpose
print(f"a.T is C-contiguous: {a_T.flags['C_CONTIGUOUS']}")  # False

# Force contiguous (required for some C extensions)
a_T_contig = np.ascontiguousarray(a_T)
print(f"After ascontiguousarray: {a_T_contig.flags['C_CONTIGUOUS']}")  # True
print(f"Shares memory after copy: {np.shares_memory(a_T, a_T_contig)}")  # False

# Performance difference: row vs column access
n = 1000
mat = rng.standard_normal((n, n)).astype(np.float32)

t0 = time.perf_counter()
row_sums = mat.sum(axis=1)          # sum along columns — cache-friendly
t1 = time.perf_counter()
col_sums = mat.T.sum(axis=1)        # sum along rows after T — same result, similar speed
t2 = time.perf_counter()

print(f"Row sums: {(t1-t0)*1000:.2f}ms | Col sums via T: {(t2-t1)*1000:.2f}ms")


# ============================================================
# 6. VECTORIZATION: replacing Python loops
# ============================================================
print("\n--- 6. Vectorization (Speed Comparison) ---")

n = 500_000
A = rng.standard_normal((n, 16)).astype(np.float32)
B = rng.standard_normal((n, 16)).astype(np.float32)

# Slow: Python loop
t0 = time.perf_counter()
dots_slow = np.array([np.dot(A[i], B[i]) for i in range(min(n, 10_000))])
t1 = time.perf_counter()
slow_time = t1 - t0

# Fast: vectorized einsum
t0 = time.perf_counter()
dots_fast = np.einsum("id,id->i", A, B)  # all n at once
t1 = time.perf_counter()
fast_time = t1 - t0

print(f"Python loop (10k): {slow_time*1000:.1f}ms")
print(f"Vectorized einsum ({n:,}): {fast_time*1000:.1f}ms")
print(f"Per-sample speed ratio: ~{(slow_time/10_000) / (fast_time/n):.0f}x faster vectorized")


# ============================================================
# 7. LINEAR ALGEBRA
# ============================================================
print("\n--- 7. Linear Algebra ---")

# SVD: the backbone of PCA and dimensionality reduction
X = rng.standard_normal((100, 20)).astype(np.float64)
X_centered = X - X.mean(axis=0)

U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
# U: (100, 20), S: (20,), Vt: (20, 20)

print(f"U shape: {U.shape}  S shape: {S.shape}  Vt shape: {Vt.shape}")
print(f"Singular values: {S[:5].round(2)}")

# Reconstruct from top-5 components
k = 5
X_approx = U[:, :k] * S[:k] @ Vt[:k, :]
residual = np.linalg.norm(X_centered - X_approx, "fro") / np.linalg.norm(X_centered, "fro")
print(f"Reconstruction error with top-{k} SVD components: {residual:.3f}")

# Solve linear system: better than inv(A) @ b
A_sq = rng.standard_normal((5, 5))
b = rng.standard_normal(5)
x = np.linalg.solve(A_sq, b)      # faster and more numerically stable than inv(A) @ b
print(f"||Ax - b|| = {np.linalg.norm(A_sq @ x - b):.2e}")  # Expected: ~1e-15


# ============================================================
# 8. PRACTICAL: COSINE SIMILARITY MATRIX
# ============================================================
print("\n--- 8. Pairwise Cosine Similarity ---")

def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between all pairs of rows in A and B.
    A: (m, d), B: (n, d) → output: (m, n)
    """
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return A_norm @ B_norm.T    # (m, d) @ (d, n) = (m, n)

# 100 query embeddings, 500 candidate embeddings, dim=64
queries = rng.standard_normal((100, 64)).astype(np.float32)
candidates = rng.standard_normal((500, 64)).astype(np.float32)

sims = cosine_sim_matrix(queries, candidates)  # (100, 500)
print(f"Similarity matrix shape: {sims.shape}")

# Top-5 candidates for each query
top5_indices = np.argpartition(sims, -5, axis=1)[:, -5:]   # faster than full sort
top5_sims = sims[np.arange(100)[:, None], top5_indices]
print(f"Top-5 sim scores for query 0: {np.sort(top5_sims[0])[::-1].round(3)}")


print("\n" + "=" * 60)
print("NumPy demonstrations complete.")
