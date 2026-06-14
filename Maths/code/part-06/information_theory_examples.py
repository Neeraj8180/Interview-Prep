"""
Part 6: Information Theory — Key Examples
Run: python part-06/information_theory_examples.py
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

print("=" * 60)
print("INFORMATION CONTENT AND ENTROPY")
print("=" * 60)

def info_content(p, base=2):
    return -np.log2(p) if base == 2 else -np.log(p)

def entropy(probs, base=2):
    probs = np.array(probs)
    return -np.sum(probs * (np.log2(probs + 1e-15) if base == 2 else np.log(probs + 1e-15)))

print(f"I(heads, p=0.5) = {info_content(0.5):.2f} bits")
print(f"I(die shows 6, p=1/6) = {info_content(1/6):.2f} bits")
print(f"I(certain event) = {info_content(1.0):.2f} bits")

fair_coin = [0.5, 0.5]
biased_coin = [0.9, 0.1]
print(f"\nH(fair coin)   = {entropy(fair_coin):.4f} bits")
print(f"H(biased coin) = {entropy(biased_coin):.4f} bits")  # less uncertainty

print("\n" + "=" * 60)
print("CROSS-ENTROPY LOSS")
print("=" * 60)

def cross_entropy(p, q):
    p, q = np.array(p), np.array(q)
    return -np.sum(p * np.log(q + 1e-10))

# True one-hot label (class 2 out of 3)
p = np.array([0.0, 0.0, 1.0])
q_good = np.array([0.05, 0.10, 0.85])
q_bad  = np.array([0.33, 0.33, 0.34])

print(f"H(P, Q_good) = {cross_entropy(p, q_good):.4f} nats")
print(f"H(P, Q_bad)  = {cross_entropy(p, q_bad):.4f} nats")

# PyTorch cross-entropy
logits = torch.tensor([[-2.0, -1.0, 2.0]])
target = torch.tensor([2])
print(f"PyTorch CE loss: {F.cross_entropy(logits, target).item():.4f}")

print("\n" + "=" * 60)
print("KL DIVERGENCE")
print("=" * 60)

def kl_divergence(p, q):
    p, q = np.array(p) + 1e-10, np.array(q) + 1e-10
    return np.sum(p * np.log(p / q))

P = [0.3, 0.5, 0.2]
Q = [0.25, 0.5, 0.25]
print(f"KL(P||Q) = {kl_divergence(P, Q):.4f} nats")
print(f"KL(Q||P) = {kl_divergence(Q, P):.4f} nats")  # different (asymmetric)
print(f"KL(P||P) = {kl_divergence(P, P):.6f} (should be 0)")

print("\n" + "=" * 60)
print("JS DIVERGENCE")
print("=" * 60)

def js_divergence(p, q):
    p, q = np.array(p) + 1e-10, np.array(q) + 1e-10
    M = (p + q) / 2
    return 0.5 * kl_divergence(p, M) + 0.5 * kl_divergence(q, M)

print(f"JS(P||Q) = {js_divergence(P, Q):.4f} (symmetric)")
print(f"JS(Q||P) = {js_divergence(Q, P):.4f} (same as above)")

print("\n" + "=" * 60)
print("MUTUAL INFORMATION")
print("=" * 60)

def mutual_information(joint_pxy):
    """Compute I(X;Y) from joint distribution."""
    joint = np.array(joint_pxy) + 1e-10
    p_x = joint.sum(axis=1, keepdims=True)
    p_y = joint.sum(axis=0, keepdims=True)
    return np.sum(joint * np.log(joint / (p_x * p_y)))

# Two independent variables: I(X;Y) should be ~0
independent = np.array([[0.25, 0.25], [0.25, 0.25]])
print(f"I(X;Y) for independent: {mutual_information(independent):.6f}")  # ~0

# Fully correlated: I(X;Y) = H(X) = H(Y)
dependent = np.array([[0.5, 0.0], [0.0, 0.5]])
print(f"I(X;Y) for fully correlated: {mutual_information(dependent):.4f}")  # = H(X) = 1 bit

print("\n" + "=" * 60)
print("PERPLEXITY")
print("=" * 60)

def compute_perplexity(log_probs):
    return np.exp(-np.mean(log_probs))

# Simulate log-probs from a language model
np.random.seed(42)
good_model_lps = np.random.normal(-1.0, 0.5, 1000)   # high probability = low perplexity
bad_model_lps  = np.random.normal(-3.0, 0.5, 1000)   # low probability = high perplexity

print(f"Good model perplexity: {compute_perplexity(good_model_lps):.2f}")
print(f"Bad model perplexity:  {compute_perplexity(bad_model_lps):.2f}")

# A uniform language model over V=10,000 words: perplexity = 10,000
log_uniform = np.full(100, -np.log(10000))
print(f"Uniform LM (V=10000) perplexity: {compute_perplexity(log_uniform):.0f}")

print("\nAll Part 6 examples ran successfully!")
