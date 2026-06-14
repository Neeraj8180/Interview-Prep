"""
Part 9: Classical ML Mathematics — Runnable Examples
Covers: Linear Regression (OLS, Ridge, Lasso), Logistic Regression,
        SVMs, Decision Trees, Ensemble Methods, Gradient Boosting,
        K-Means, Bias-Variance, Kernel Methods
"""

import numpy as np
from sklearn.datasets import make_regression, make_classification
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. Linear Regression: OLS, Ridge, Lasso
# ============================================================
print("=" * 60)
print("1. LINEAR REGRESSION")
print("=" * 60)

# Generate data: y = 2 + 3x + noise
np.random.seed(42)
n = 50
X_raw = np.random.randn(n)
y = 2 + 3 * X_raw + np.random.randn(n) * 0.5

# Design matrix with bias column
X = np.column_stack([np.ones(n), X_raw])

# OLS Normal Equation
w_ols = np.linalg.solve(X.T @ X, X.T @ y)
print(f"OLS: intercept={w_ols[0]:.3f}, slope={w_ols[1]:.3f}")  # ~2, ~3

# Ridge: (X^T X + lambda I)^{-1} X^T y
for lam in [0.1, 1.0, 10.0]:
    w_ridge = np.linalg.solve(X.T @ X + lam * np.eye(2), X.T @ y)
    print(f"Ridge (λ={lam:4.1f}): intercept={w_ridge[0]:.3f}, slope={w_ridge[1]:.3f}")

# Condition number shows why ridge is numerically stabilizing
print(f"\nCondition number (X^T X):          {np.linalg.cond(X.T @ X):.2f}")
print(f"Condition number (X^T X + 1.0*I):  {np.linalg.cond(X.T @ X + 1.0 * np.eye(2)):.2f}")

# Compare MSE
y_hat_ols = X @ w_ols
w_r = np.linalg.solve(X.T @ X + 1.0 * np.eye(2), X.T @ y)
y_hat_ridge = X @ w_r
print(f"\nOLS MSE:   {np.mean((y - y_hat_ols)**2):.4f}")
print(f"Ridge MSE: {np.mean((y - y_hat_ridge)**2):.4f}")


# ============================================================
# 2. Logistic Regression from Scratch
# ============================================================
print("\n" + "=" * 60)
print("2. LOGISTIC REGRESSION")
print("=" * 60)

def sigmoid(z):
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))

def logistic_train(X, y, lr=0.1, n_iters=500, lam=0.01):
    """Train logistic regression with gradient descent."""
    n, d = X.shape
    w = np.zeros(d)
    losses = []

    for _ in range(n_iters):
        p = sigmoid(X @ w)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        # Cross-entropy loss + L2 regularization
        loss = -np.mean(y * np.log(p) + (1-y) * np.log(1-p)) + 0.5 * lam * np.sum(w[1:]**2)
        losses.append(loss)
        # Gradient: X^T (p - y) / n + lambda * w (excluding bias)
        grad = X.T @ (p - y) / n
        grad[1:] += lam * w[1:]
        w -= lr * grad

    return w, losses

# Binary classification data
X_cls, y_cls = make_classification(n_samples=300, n_features=2, n_redundant=0,
                                    random_state=42)
X_cls_b = np.column_stack([np.ones(len(X_cls)), X_cls])  # Add bias

w_lr, losses = logistic_train(X_cls_b, y_cls)
p_hat = sigmoid(X_cls_b @ w_lr)
acc = np.mean((p_hat >= 0.5) == y_cls)
print(f"Logistic Regression accuracy: {acc:.1%}")
print(f"Final loss: {losses[-1]:.4f}")
print(f"Loss decreased: {losses[0]:.4f} → {losses[-1]:.4f}")

# Gradient at optimal is near zero
p_opt = sigmoid(X_cls_b @ w_lr)
grad_opt = np.abs(X_cls_b.T @ (p_opt - y_cls)).mean()
print(f"Mean absolute gradient at convergence: {grad_opt:.2e}")


# ============================================================
# 3. Decision Tree Splitting Criterion
# ============================================================
print("\n" + "=" * 60)
print("3. DECISION TREE — INFORMATION GAIN")
print("=" * 60)

def entropy(y):
    """Shannon entropy of a binary label array."""
    if len(y) == 0:
        return 0.0
    p = np.mean(y)
    if p == 0 or p == 1:
        return 0.0
    return -(p * np.log2(p) + (1-p) * np.log2(1-p))

def information_gain(y, y_left, y_right):
    """Information gain of a split."""
    n = len(y)
    ig = entropy(y) - (len(y_left)/n * entropy(y_left) + len(y_right)/n * entropy(y_right))
    return ig

def gini(y):
    """Gini impurity."""
    if len(y) == 0:
        return 0.0
    p = np.mean(y)
    return 1 - p**2 - (1-p)**2

def best_split(X, y):
    """Find best feature and threshold by information gain."""
    best_ig = -1
    best_feat, best_thresh = None, None

    for j in range(X.shape[1]):
        thresholds = np.unique(X[:, j])
        for t in thresholds:
            left = y[X[:, j] <= t]
            right = y[X[:, j] > t]
            ig = information_gain(y, left, right)
            if ig > best_ig:
                best_ig = ig
                best_feat, best_thresh = j, t

    return best_feat, best_thresh, best_ig

# Demo on simple dataset
X_tree = np.array([[1, 0], [2, 0], [3, 1], [4, 1], [5, 0], [6, 1]])
y_tree = np.array([0, 0, 1, 1, 0, 1])

feat, thresh, ig = best_split(X_tree, y_tree)
print(f"Best split: Feature {feat}, Threshold {thresh}, IG={ig:.4f}")
print(f"Entropy before split: {entropy(y_tree):.4f}")
print(f"Entropy left:  {entropy(y_tree[X_tree[:, feat] <= thresh]):.4f}")
print(f"Entropy right: {entropy(y_tree[X_tree[:, feat] > thresh]):.4f}")


# ============================================================
# 4. Bias-Variance Decomposition (Monte Carlo)
# ============================================================
print("\n" + "=" * 60)
print("4. BIAS-VARIANCE DECOMPOSITION")
print("=" * 60)

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge

def bias_variance_decomposition(ModelClass, X_train_all, y_train_all,
                                 X_test, y_test_true, n_experiments=50, **kwargs):
    """
    Monte Carlo estimation of bias^2 and variance.
    """
    predictions = []
    for _ in range(n_experiments):
        # Bootstrap sample
        idx = np.random.choice(len(X_train_all), len(X_train_all), replace=True)
        X_b, y_b = X_train_all[idx], y_train_all[idx]

        model = ModelClass(**kwargs)
        model.fit(X_b, y_b)
        predictions.append(model.predict(X_test))

    predictions = np.array(predictions)  # (n_experiments, n_test)
    mean_pred = predictions.mean(axis=0)

    bias_sq = np.mean((mean_pred - y_test_true)**2)
    variance = np.mean(predictions.var(axis=0))
    return bias_sq, variance

# Generate regression data
np.random.seed(42)
n_train = 100
X_bv = np.sort(np.random.uniform(0, 10, n_train)).reshape(-1, 1)
y_true_fn = lambda x: np.sin(x.ravel()) + 0.1 * x.ravel()
noise_std = 0.3
y_bv = y_true_fn(X_bv) + np.random.randn(n_train) * noise_std

X_test_bv = np.linspace(0, 10, 50).reshape(-1, 1)
y_test_true_bv = y_true_fn(X_test_bv)

print(f"{'Model':<30} {'Bias^2':>10} {'Variance':>10} {'Total':>10}")
print("-" * 60)

for max_depth, label in [(1, "Tree depth=1 (high bias)"),
                          (5, "Tree depth=5 (balanced)"),
                          (20, "Tree depth=20 (high var)")]:
    b2, v = bias_variance_decomposition(DecisionTreeRegressor,
                                         X_bv, y_bv, X_test_bv, y_test_true_bv,
                                         max_depth=max_depth)
    print(f"{label:<30} {b2:>10.4f} {v:>10.4f} {b2+v:>10.4f}")


# ============================================================
# 5. Ensemble Methods: Random Forest vs Gradient Boosting
# ============================================================
print("\n" + "=" * 60)
print("5. ENSEMBLE METHODS")
print("=" * 60)

X_ens, y_ens = make_regression(n_samples=500, n_features=10, noise=20, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X_ens, y_ens, test_size=0.2, random_state=42)

# Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                max_depth=3, random_state=42)
gb.fit(X_tr, y_tr)
print(f"GB RMSE:  {np.sqrt(mean_squared_error(y_te, gb.predict(X_te))):.2f}")

# Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_tr, y_tr)
print(f"RF RMSE:  {np.sqrt(mean_squared_error(y_te, rf.predict(X_te))):.2f}")

# Staged predictions (show how GB improves with more trees)
staged_rmse = [np.sqrt(mean_squared_error(y_te, pred))
               for pred in gb.staged_predict(X_te)]
best_iter = int(np.argmin(staged_rmse))
print(f"GB best iteration: {best_iter}, RMSE: {staged_rmse[best_iter]:.2f}")
print(f"GB at 200 trees:   RMSE: {staged_rmse[-1]:.2f}")
print(f"GB at 10 trees:    RMSE: {staged_rmse[9]:.2f}")

# Feature importance
feat_imp = rf.feature_importances_
print(f"\nTop RF feature importances: {np.sort(feat_imp)[::-1][:3].round(3)}")


# ============================================================
# 6. K-Means from Scratch
# ============================================================
print("\n" + "=" * 60)
print("6. K-MEANS FROM SCRATCH")
print("=" * 60)

def kmeans_pp_init(X, k, seed=42):
    """k-means++ initialization."""
    rng = np.random.default_rng(seed)
    n = len(X)
    centers = [X[rng.integers(n)]]

    for _ in range(k - 1):
        dists = np.array([min(np.linalg.norm(x - c)**2 for c in centers) for x in X])
        probs = dists / dists.sum()
        idx = rng.choice(n, p=probs)
        centers.append(X[idx])

    return np.array(centers)

def kmeans(X, k, max_iter=100, seed=42):
    centers = kmeans_pp_init(X, k, seed)
    labels = np.zeros(len(X), dtype=int)

    for iteration in range(max_iter):
        # E-step: assign to nearest center
        dists = np.array([[np.linalg.norm(x - c)**2 for c in centers] for x in X])
        new_labels = dists.argmin(axis=1)

        if np.all(new_labels == labels):
            print(f"  K-means converged at iteration {iteration}")
            break
        labels = new_labels

        # M-step: update centers
        for c in range(k):
            mask = labels == c
            if mask.sum() > 0:
                centers[c] = X[mask].mean(axis=0)

    inertia = sum(np.linalg.norm(X[i] - centers[labels[i]])**2 for i in range(len(X)))
    return labels, centers, inertia

# Generate 3-cluster data
from sklearn.datasets import make_blobs
X_km, y_km_true = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=42)
labels, centers, inertia = kmeans(X_km, k=3)

# Accuracy via Hungarian matching (check cluster purity)
from collections import Counter
purity = 0
for c in range(3):
    cluster_labels = y_km_true[labels == c]
    if len(cluster_labels) > 0:
        purity += Counter(cluster_labels).most_common(1)[0][1]
print(f"K-Means purity: {purity/len(X_km):.1%}")
print(f"K-Means inertia: {inertia:.2f}")


# ============================================================
# 7. Kernel Methods: RBF Kernel Ridge Regression
# ============================================================
print("\n" + "=" * 60)
print("7. KERNEL RIDGE REGRESSION")
print("=" * 60)

def rbf_kernel(X1, X2, gamma=1.0):
    """Radial Basis Function (Gaussian) kernel."""
    # ||x - x'||^2 = ||x||^2 + ||x'||^2 - 2 x^T x'
    sq_dists = (np.sum(X1**2, axis=1, keepdims=True)
                + np.sum(X2**2, axis=1)
                - 2 * X1 @ X2.T)
    return np.exp(-gamma * sq_dists)

def kernel_ridge_regression(X_train, y_train, X_test, gamma=1.0, lam=0.1):
    """Kernel Ridge Regression: alpha = (K + lambda*I)^{-1} y"""
    K = rbf_kernel(X_train, X_train, gamma)
    alpha = np.linalg.solve(K + lam * np.eye(len(X_train)), y_train)

    K_test = rbf_kernel(X_test, X_train, gamma)
    return K_test @ alpha

# 1D regression example: learn f(x) = sin(x)
np.random.seed(42)
X_krr_tr = np.sort(np.random.uniform(0, 2 * np.pi, 30)).reshape(-1, 1)
y_krr_tr = np.sin(X_krr_tr.ravel()) + 0.1 * np.random.randn(30)
X_krr_te = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
y_krr_true = np.sin(X_krr_te.ravel())

for gamma, lam in [(0.5, 0.01), (2.0, 0.01), (0.5, 1.0)]:
    y_pred = kernel_ridge_regression(X_krr_tr, y_krr_tr, X_krr_te, gamma, lam)
    rmse = np.sqrt(np.mean((y_pred - y_krr_true)**2))
    print(f"KRR γ={gamma:.1f}, λ={lam:.2f}: RMSE={rmse:.4f}")


# ============================================================
# 8. VC Dimension Illustration
# ============================================================
print("\n" + "=" * 60)
print("8. GENERALIZATION BOUND (PAC)")
print("=" * 60)

def pac_generalization_bound(train_error, n, vc_dim, delta=0.05):
    """
    Generalization bound: err_test <= err_train + sqrt((vc_dim * log(n/vc_dim) + log(1/delta)) / n)
    """
    import math
    complexity = (vc_dim * math.log(n / vc_dim) + math.log(1 / delta)) / n
    return train_error + math.sqrt(complexity)

print(f"{'n':>8} {'VC=10':>12} {'VC=100':>12} {'VC=1000':>12}")
print("-" * 48)
for n in [100, 1000, 10000, 100000]:
    bounds = [pac_generalization_bound(0.05, n, vc) for vc in [10, 100, 1000]]
    bound_strs = " ".join(f"{b:>12.3f}" for b in bounds)
    print(f"{n:>8} {bound_strs}")

print("\nInterpretation: bound shrinks as n increases, grows with VC dimension.")
print("Need n >> VC_dim for reliable generalization.")


# ============================================================
# 9. Causal Inference: Doubly Robust ATE
# ============================================================
print("\n" + "=" * 60)
print("9. CAUSAL INFERENCE — DOUBLY ROBUST ATE")
print("=" * 60)

from sklearn.linear_model import LogisticRegression as LRSklearn, LinearRegression

np.random.seed(42)
n_causal = 1000
X_c = np.random.randn(n_causal, 3)

# True propensity: P(T=1|X) depends on X[0]
e_true = 1 / (1 + np.exp(-0.8 * X_c[:, 0]))
T = np.random.binomial(1, e_true)

# True outcome: linear with heterogeneous treatment effect
true_ate = 2.0
mu0 = X_c[:, 0] + X_c[:, 1]
mu1 = mu0 + true_ate
Y = T * mu1 + (1-T) * mu0 + np.random.randn(n_causal) * 0.5

# Naive estimate (confounded)
naive_ate = Y[T==1].mean() - Y[T==0].mean()

# Doubly robust estimate
# Step 1: fit propensity model
prop_model = LRSklearn()
prop_model.fit(X_c, T)
e_hat = prop_model.predict_proba(X_c)[:, 1]
e_hat = np.clip(e_hat, 0.01, 0.99)

# Step 2: fit outcome models
mu1_model = LinearRegression().fit(X_c[T==1], Y[T==1])
mu0_model = LinearRegression().fit(X_c[T==0], Y[T==0])
mu1_hat = mu1_model.predict(X_c)
mu0_hat = mu0_model.predict(X_c)

# Step 3: doubly robust score
dr_scores = (mu1_hat - mu0_hat
             + T * (Y - mu1_hat) / e_hat
             - (1-T) * (Y - mu0_hat) / (1 - e_hat))
ate_dr = dr_scores.mean()
ate_se = dr_scores.std() / np.sqrt(n_causal)

print(f"True ATE:              {true_ate:.3f}")
print(f"Naive (biased) ATE:    {naive_ate:.3f}")
print(f"Doubly Robust ATE:     {ate_dr:.3f} ± {1.96*ate_se:.3f} (95% CI)")


print("\n" + "=" * 60)
print("All Part 9 examples completed successfully.")
print("=" * 60)
