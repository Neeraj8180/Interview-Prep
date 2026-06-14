"""
Part 3: Probability — Key Examples
Run: python part-03/probability_examples.py
"""

import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture

print("=" * 60)
print("BAYES THEOREM: Medical Test Example")
print("=" * 60)

p_disease = 0.01
p_pos_given_disease = 0.90
p_pos_given_healthy = 0.10

p_pos = p_pos_given_disease * p_disease + p_pos_given_healthy * (1 - p_disease)
p_disease_given_pos = (p_pos_given_disease * p_disease) / p_pos
print(f"P(positive): {p_pos:.4f}")
print(f"P(disease | positive): {p_disease_given_pos:.4f}")  # ~0.083
print("Despite 90% accurate test, only 8.3% chance of disease if positive!")

print("\n" + "=" * 60)
print("DISTRIBUTIONS")
print("=" * 60)

np.random.seed(42)

# Binomial
n, p = 10, 0.3
k = 3
pmf = stats.binom.pmf(k, n, p)
print(f"Binomial P(X=3): {pmf:.4f}")  # ~0.267

# Normal
print(f"P(-1 ≤ Z ≤ 1) = {stats.norm.cdf(1) - stats.norm.cdf(-1):.4f}")  # 0.6827
print(f"P(-2 ≤ Z ≤ 2) = {stats.norm.cdf(2) - stats.norm.cdf(-2):.4f}")  # 0.9545

# Beta
alpha, beta = 2, 5
print(f"Beta({alpha},{beta}) mean = {alpha/(alpha+beta):.4f}")
print(f"Beta(1,1) = Uniform[0,1]: {stats.beta.mean(1, 1):.4f}")

print("\n" + "=" * 60)
print("GAUSSIAN MIXTURE MODEL (EM)")
print("=" * 60)

X1 = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 200)
X2 = np.random.multivariate_normal([5, 5], [[1, -0.3], [-0.3, 1]], 200)
X = np.vstack([X1, X2])

gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(X)
print(f"Fitted means:\n{gmm.means_}")
print(f"Mixture weights: {gmm.weights_}")
labels = gmm.predict(X)
print(f"Cluster sizes: {np.bincount(labels)}")

print("\n" + "=" * 60)
print("MONTE CARLO: Estimate π")
print("=" * 60)

N = 1_000_000
x, y = np.random.uniform(-1, 1, N), np.random.uniform(-1, 1, N)
inside = (x**2 + y**2 <= 1)
pi_estimate = 4 * inside.mean()
print(f"Estimated π = {pi_estimate:.5f} (true: 3.14159)")
assert abs(pi_estimate - np.pi) < 0.01, "Monte Carlo estimate too far off"

print("\n" + "=" * 60)
print("MCMC: Metropolis-Hastings for Normal Distribution")
print("=" * 60)

def metropolis_hastings(target_log_prob, n_samples=50_000, proposal_std=1.0):
    """Sample from target distribution using MH algorithm."""
    samples = []
    current = 0.0

    for _ in range(n_samples):
        proposal = current + np.random.normal(0, proposal_std)
        log_alpha = target_log_prob(proposal) - target_log_prob(current)
        if np.log(np.random.uniform()) < log_alpha:
            current = proposal
        samples.append(current)

    return np.array(samples[1000:])   # discard burn-in

# Sample from N(2, 0.5^2)
mu, sigma = 2.0, 0.5
log_prob = lambda x: stats.norm.logpdf(x, loc=mu, scale=sigma)
samples = metropolis_hastings(log_prob)

print(f"Target: N({mu}, {sigma}²)")
print(f"Sample mean: {samples.mean():.4f} (should be ~{mu})")
print(f"Sample std:  {samples.std():.4f} (should be ~{sigma})")

print("\nAll Part 3 examples ran successfully!")
