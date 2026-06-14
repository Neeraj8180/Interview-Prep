"""
Part 1: Linear Algebra — Core Examples
Run: python part-01/linear_algebra_examples.py
"""

import numpy as np
import torch
import torch.nn.functional as F
import math

print("=" * 60)
print("LESSON 1.1: Scalars, Vectors, Matrices, Tensors")
print("=" * 60)

# Scalar
a = 3.14
print(f"Scalar: {a}, type: {type(a)}")

# Vector
v = np.array([0.2, 0.8, -0.1])
print(f"Vector shape: {v.shape}")   # (3,)

# Matrix
X = np.array([
    [0.2,  0.8, -0.1],   # "The"
    [0.9,  0.1,  0.4],   # "cat"
    [-0.3, 0.5,  0.7],   # "sat"
])
print(f"Matrix shape: {X.shape}")   # (3, 3)

# Tensor (PyTorch)
T = torch.randn(2, 3, 4)
print(f"Tensor shape: {T.shape}")   # torch.Size([2, 3, 4])

print("\n" + "=" * 60)
print("LESSON 1.3: Linear Independence")
print("=" * 60)

V = np.array([
    [1, 2, 1],
    [2, 4, 0],
    [3, 6, 1],
])
rank = np.linalg.matrix_rank(V)
print(f"Rank: {rank}")   # 2 — v2 is dependent on v1

print("\n" + "=" * 60)
print("LESSON 1.5: Norms")
print("=" * 60)

v = np.array([3.0, -4.0])
print(f"L1 norm: {np.sum(np.abs(v))}")         # 7.0
print(f"L2 norm: {np.linalg.norm(v)}")          # 5.0
print(f"L∞ norm: {np.max(np.abs(v))}")          # 4.0

u = np.array([1.0, 1.0])
v2 = np.array([2.0, 2.0])
cos_sim = np.dot(u, v2) / (np.linalg.norm(u) * np.linalg.norm(v2))
print(f"Cosine similarity: {cos_sim}")           # 1.0

print("\n" + "=" * 60)
print("LESSON 1.6: Products")
print("=" * 60)

u = np.array([2.0, 3.0, 1.0])
v = np.array([4.0, 1.0, 2.0])
print(f"Dot product: {np.dot(u, v)}")            # 13
print(f"Outer product:\n{np.outer(u, v)}")
print(f"Cross product: {np.cross(u, v)}")

print("\n" + "=" * 60)
print("LESSON 1.7: Matrix Multiplication")
print("=" * 60)

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A @ B
print(f"A @ B:\n{C}")  # [[19,22],[43,50]]

print("\n" + "=" * 60)
print("LESSON 1.8: Determinant, Inverse, Pseudo-Inverse")
print("=" * 60)

A = np.array([[3., 2.], [1., 4.]])
print(f"det(A) = {np.linalg.det(A):.2f}")        # 10.0
print(f"trace(A) = {np.trace(A):.2f}")            # 7.0
A_inv = np.linalg.inv(A)
print(f"A @ A_inv:\n{np.round(A @ A_inv, 10)}")   # Identity

print("\n" + "=" * 60)
print("LESSON 1.10: Eigenvalues and Eigenvectors")
print("=" * 60)

A = np.array([[4., 1.], [2., 3.]])
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Eigenvalues: {eigenvalues}")              # [5., 2.]
v1 = eigenvectors[:, 0]
lambda1 = eigenvalues[0]
print(f"Av1 = {A @ v1}")
print(f"λ1*v1 = {lambda1 * v1}")

print("\n" + "=" * 60)
print("LESSON 1.11: SVD")
print("=" * 60)

A = np.array([[3., 1., 1.], [1., 3., 1.]])
U, s, Vt = np.linalg.svd(A)
print(f"Singular values: {s}")
Sigma = np.zeros(A.shape)
np.fill_diagonal(Sigma, s)
A_recon = U @ Sigma @ Vt
print(f"Reconstruction error: {np.max(np.abs(A - A_recon)):.2e}")

print("\n" + "=" * 60)
print("LESSON 1.12: PCA")
print("=" * 60)

np.random.seed(42)
n = 200
X = np.random.multivariate_normal([0, 0], [[3, 2], [2, 2]], n)
X_centered = X - X.mean(axis=0)
C = (X_centered.T @ X_centered) / (n - 1)
eigenvalues, eigenvectors = np.linalg.eigh(C)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
print(f"Explained variance ratios: {eigenvalues / eigenvalues.sum()}")
Z = X_centered @ eigenvectors[:, :1]
print(f"Reduced from {X.shape} to {Z.shape}")

print("\n" + "=" * 60)
print("LESSON 1.13: Scaled Dot-Product Attention")
print("=" * 60)

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)
    return attn_weights @ V, attn_weights

B, H, T, d_k = 1, 1, 4, 8
Q = torch.randn(B, H, T, d_k)
K = torch.randn(B, H, T, d_k)
V = torch.randn(B, H, T, d_k)
output, weights = scaled_dot_product_attention(Q, K, V)
print(f"Attention output shape: {output.shape}")     # (1, 1, 4, 8)
print(f"Row sums (should be 1): {weights.sum(dim=-1)}")

print("\nAll Part 1 examples ran successfully!")
