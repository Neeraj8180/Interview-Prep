# Part 16: Projects

> **Prerequisites:** Parts 0–15
> **Status:** Complete
> **Projects:** 12

---

## How to Use This Part

Each project:
- Is **self-contained** — you can do it independently
- Builds on specific parts of the course
- Has a clear deliverable and success criterion
- Includes starter code and key implementation hints
- Estimates time to completion

The projects are roughly ordered by difficulty.

---

## Project 1: Build PCA from Scratch

**From:** Part 1 (Linear Algebra), Part 2 (Calculus)
**Time:** 3–4 hours
**Level:** Beginner

### Goal

Implement PCA using only NumPy (no sklearn.decomposition.PCA). Apply it to a real dataset and visualize the principal components.

### What You Will Build

A `PCA` class with methods:
- `fit(X)`: compute eigenvectors of the covariance matrix
- `transform(X)`: project data onto principal components
- `fit_transform(X)`: both steps
- `explained_variance_ratio_`: proportion of variance explained per component

### Key Math to Implement

1. Center the data: $\tilde{\mathbf{X}} = \mathbf{X} - \bar{\mathbf{x}}$
2. Compute covariance: $\mathbf{C} = \frac{1}{n-1}\tilde{\mathbf{X}}^T\tilde{\mathbf{X}}$
3. Eigendecomposition: $\mathbf{C}\mathbf{V} = \mathbf{V}\mathbf{\Lambda}$
4. Sort by descending eigenvalue
5. Project: $\mathbf{Z} = \tilde{\mathbf{X}}\mathbf{V}_r$

### Starter Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X):
        # TODO: Center the data
        # TODO: Compute covariance matrix
        # TODO: Eigendecomposition
        # TODO: Sort by descending eigenvalue
        # TODO: Store components and variance explained
        pass

    def transform(self, X):
        # TODO: Center and project
        pass

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# Test on digits dataset
digits = load_digits()
X, y = digits.data, digits.target  # (1797, 64)

pca = PCA(n_components=2)
Z = pca.fit_transform(X)

# Visualize
plt.figure(figsize=(10, 8))
scatter = plt.scatter(Z[:, 0], Z[:, 1], c=y, cmap='tab10', alpha=0.7)
plt.colorbar(scatter)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Digits Dataset — First Two Principal Components')
plt.savefig('pca_digits.png')
print(f"Variance explained by 2 PCs: {pca.explained_variance_ratio_[:2].sum():.1%}")
```

### Success Criterion

- Output matches `sklearn.decomposition.PCA` to 5 significant figures (up to sign flips — eigenvectors are unique up to sign)
- Your 2D visualization shows clear digit clusters

---

## Project 2: Backpropagation from Scratch

**From:** Part 2 (Calculus), Part 10 (Deep Learning)
**Time:** 6–8 hours
**Level:** Intermediate

### Goal

Build a full automatic differentiation system from scratch that can compute gradients for arbitrary computation graphs. Train a small neural network on MNIST without using PyTorch's autograd.

### What You Will Build

A `Value` class (like micrograd by Karpathy) that wraps a scalar and tracks its gradient, then extend it to tensors.

### Key Concepts to Implement

- Forward pass: compute value
- Backward pass: apply chain rule
- Operations: `+`, `*`, `**`, `exp`, `log`, `relu`, `sigmoid`

### Starter Code

```python
import numpy as np

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, power):
        out = Value(self.data**power, (self,), f'**{power}')
        def _backward():
            self.grad += power * (self.data**(power-1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        """Topological sort and backprop."""
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

# Test: compute gradient of f(x,y) = (x + y) * y at x=2, y=3
x = Value(2.0, label='x')
y = Value(3.0, label='y')
f = (x + y) * y
f.backward()
print(f"df/dx = {x.grad}")  # Should be 3
print(f"df/dy = {y.grad}")  # Should be 5 (= x + 2y = 2 + 6 = 8... wait: (x+y)*y, d/dy = (x+y) + y = x+2y = 8)

# TODO: Extend to matrix operations
# TODO: Build a 2-layer MLP
# TODO: Train on XOR problem first, then MNIST subset
```

### Success Criterion

- Gradients match PyTorch's autograd to 6 decimal places
- A 2-layer network converges on XOR (achieves > 99% accuracy)

---

## Project 3: Implement K-Means and GMM

**From:** Part 3 (Probability), Part 9 (Classical ML)
**Time:** 4–5 hours
**Level:** Intermediate

### Goal

Implement K-Means with k-means++ initialization and GMM with EM from scratch. Visualize the clustering process and compare the two algorithms.

### Key Math to Implement

**K-Means:**
- k-means++ initialization: probability proportional to $d^2$
- E-step: $r_{ic} = \mathbf{1}[c = \arg\min_k \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2]$
- M-step: $\boldsymbol{\mu}_k = \frac{\sum_i r_{ik}\mathbf{x}_i}{\sum_i r_{ik}}$

**GMM with EM:**
- E-step: $r_{ic} = \frac{\pi_c \mathcal{N}(\mathbf{x}_i; \boldsymbol{\mu}_c, \mathbf{\Sigma}_c)}{\sum_k \pi_k \mathcal{N}(\mathbf{x}_i; \boldsymbol{\mu}_k, \mathbf{\Sigma}_k)}$
- M-step: Update $\pi_c, \boldsymbol{\mu}_c, \mathbf{\Sigma}_c$ from weighted statistics

### Starter Code

```python
import numpy as np
from scipy.stats import multivariate_normal

def kmeans_plus_plus_init(X, k):
    """k-means++ initialization."""
    n = len(X)
    centers = [X[np.random.randint(n)]]

    for _ in range(k - 1):
        dists = np.array([min(np.linalg.norm(x - c)**2 for c in centers) for x in X])
        probs = dists / dists.sum()
        centers.append(X[np.random.choice(n, p=probs)])

    return np.array(centers)

class KMeans:
    def __init__(self, k, max_iter=100):
        self.k = k
        self.max_iter = max_iter

    def fit(self, X):
        self.centers_ = kmeans_plus_plus_init(X, self.k)
        prev_labels = None

        for i in range(self.max_iter):
            # E-step
            dists = np.array([[np.linalg.norm(x - c)**2 for c in self.centers_] for x in X])
            labels = dists.argmin(axis=1)

            if np.all(labels == prev_labels):
                print(f"Converged at iteration {i}")
                break
            prev_labels = labels.copy()

            # M-step
            for k in range(self.k):
                mask = labels == k
                if mask.sum() > 0:
                    self.centers_[k] = X[mask].mean(axis=0)

        self.labels_ = labels
        self.inertia_ = sum(np.linalg.norm(X[i] - self.centers_[labels[i]])**2 for i in range(len(X)))
        return self

# TODO: Implement GMM class
# TODO: Generate synthetic data with 3 Gaussian clusters
# TODO: Compare K-Means vs GMM on non-spherical clusters
# TODO: Plot convergence of WCSS per iteration
```

### Success Criterion

- K-Means correctly clusters 3 well-separated Gaussian clusters
- GMM achieves better NLL than K-Means on elliptical clusters
- Plot shows how cluster assignments change during EM iterations

---

## Project 4: Implement Logistic Regression with Multiple Solvers

**From:** Part 9 (Classical ML), Part 2 (Calculus)
**Time:** 4–5 hours
**Level:** Intermediate

### Goal

Implement binary logistic regression with three optimization methods: gradient descent, Newton-Raphson, and L-BFGS (using scipy). Compare convergence speed.

### Key Math

**Gradient:** $\nabla J = \frac{1}{n}\mathbf{X}^T(\boldsymbol{\sigma}(\mathbf{X}\mathbf{w}) - \mathbf{y})$

**Hessian:** $\mathbf{H} = \frac{1}{n}\mathbf{X}^T\mathbf{D}\mathbf{X}$ where $\mathbf{D} = \text{diag}(\hat{p}_i(1-\hat{p}_i))$

**Newton step:** $\mathbf{w} \leftarrow \mathbf{w} - \mathbf{H}^{-1}\nabla J$

```python
import numpy as np
from sklearn.datasets import make_classification

class LogisticRegression:
    def __init__(self, method='gd', lr=0.1, max_iter=200, tol=1e-6, C=1.0):
        self.method = method
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.C = C  # 1/lambda for L2 regularization
        self.loss_history_ = []

    def _sigmoid(self, z):
        return np.where(z >= 0,
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z)))

    def _loss(self, X, y, w):
        p = self._sigmoid(X @ w)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -np.mean(y * np.log(p) + (1-y) * np.log(1-p)) + 0.5/self.C * np.sum(w[1:]**2)

    def fit_gd(self, X, y):
        n, d = X.shape
        w = np.zeros(d)
        for i in range(self.max_iter):
            p = self._sigmoid(X @ w)
            grad = X.T @ (p - y) / n + np.r_[0, w[1:]/self.C]
            w -= self.lr * grad
            self.loss_history_.append(self._loss(X, y, w))
            if np.linalg.norm(grad) < self.tol:
                break
        self.w_ = w

    def fit_newton(self, X, y):
        n, d = X.shape
        w = np.zeros(d)
        for i in range(self.max_iter):
            p = self._sigmoid(X @ w)
            grad = X.T @ (p - y) / n + np.r_[0, w[1:]/self.C]
            D = np.diag(p * (1 - p))
            H = X.T @ D @ X / n + np.diag(np.r_[0, np.ones(d-1)/self.C])
            try:
                w -= np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                w -= self.lr * grad  # Fallback to GD
            self.loss_history_.append(self._loss(X, y, w))
            if np.linalg.norm(grad) < self.tol:
                break
        self.w_ = w

    def predict_proba(self, X):
        return self._sigmoid(X @ self.w_)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# Generate dataset
X, y = make_classification(n_samples=500, n_features=10, random_state=42)
X = np.column_stack([np.ones(len(X)), X])  # Add bias

# Compare solvers
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for method, label, ax_idx in [('gd', 'Gradient Descent', 0), ('newton', 'Newton-Raphson', 1)]:
    lr_gd = LogisticRegression(method=method, max_iter=100, lr=0.5)
    if method == 'gd':
        lr_gd.fit_gd(X, y)
    else:
        lr_gd.fit_newton(X, y)
    axes[ax_idx].plot(lr_gd.loss_history_)
    axes[ax_idx].set_title(f'{label} Convergence')
    axes[ax_idx].set_xlabel('Iteration')
    axes[ax_idx].set_ylabel('Loss')
plt.tight_layout()
plt.savefig('logistic_convergence.png')
```

---

## Project 5: Random Forest vs Gradient Boosting Comparison

**From:** Part 9 (Classical ML)
**Time:** 3–4 hours
**Level:** Beginner-Intermediate

### Goal

Build a complete comparison study of Random Forest and Gradient Boosting on multiple datasets. Analyze: accuracy, training time, feature importance, bias-variance tradeoff.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_wine, make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
import time

datasets = {
    'Breast Cancer': load_breast_cancer(return_X_y=True),
    'Wine': load_wine(return_X_y=True),
    'Synthetic (n=5000, d=20)': make_classification(5000, 20, random_state=42),
}

results = []
for name, (X, y) in datasets.items():
    for ModelClass, params, model_name in [
        (RandomForestClassifier, {'n_estimators': 100, 'random_state': 42}, 'Random Forest'),
        (GradientBoostingClassifier, {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42}, 'Gradient Boosting'),
    ]:
        model = ModelClass(**params)
        t0 = time.time()
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        elapsed = time.time() - t0

        results.append({
            'Dataset': name,
            'Model': model_name,
            'Accuracy': scores.mean(),
            'Std': scores.std(),
            'Time (s)': elapsed,
        })

df = pd.DataFrame(results)
print(df.to_string(index=False))

# Learning curves for one dataset
X, y = load_breast_cancer(return_X_y=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, ModelClass, name in zip(axes,
                                 [RandomForestClassifier(100), GradientBoostingClassifier(100)],
                                 ['Random Forest', 'Gradient Boosting']):
    train_sizes, train_scores, val_scores = learning_curve(
        ModelClass, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10))
    ax.plot(train_sizes, train_scores.mean(1), label='Train')
    ax.plot(train_sizes, val_scores.mean(1), label='Validation')
    ax.fill_between(train_sizes, train_scores.mean(1)-train_scores.std(1),
                    train_scores.mean(1)+train_scores.std(1), alpha=0.1)
    ax.set_title(f'{name} Learning Curve')
    ax.set_xlabel('Training Size')
    ax.set_ylabel('Accuracy')
    ax.legend()
plt.tight_layout()
plt.savefig('learning_curves.png')
```

---

## Project 6: Implement the Attention Mechanism

**From:** Part 1 (Linear Algebra), Part 10 (Deep Learning), Part 11 (LLMs)
**Time:** 5–6 hours
**Level:** Intermediate

### Goal

Implement multi-head attention from scratch in PyTorch. Verify your implementation matches PyTorch's `nn.MultiheadAttention`. Then implement a causal (masked) variant and a complete transformer block.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x):
        """(batch, seq, d_model) -> (batch, n_heads, seq, d_k)"""
        batch, seq, d = x.shape
        x = x.view(batch, seq, self.n_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)

        # Concatenate heads
        batch, _, seq, _ = context.shape
        context = context.transpose(1, 2).contiguous().view(batch, seq, self.d_model)
        return self.W_o(context), attn_weights

# Test: verify against PyTorch
torch.manual_seed(42)
d_model, n_heads, batch, seq = 64, 4, 2, 10
mha_custom = MultiHeadAttention(d_model, n_heads)
mha_torch = nn.MultiheadAttention(d_model, n_heads, bias=False, batch_first=True)

# Copy weights
mha_torch.in_proj_weight.data[:d_model] = mha_custom.W_q.weight.data
mha_torch.in_proj_weight.data[d_model:2*d_model] = mha_custom.W_k.weight.data
mha_torch.in_proj_weight.data[2*d_model:] = mha_custom.W_v.weight.data
mha_torch.out_proj.weight.data = mha_custom.W_o.weight.data

x = torch.randn(batch, seq, d_model)
out_custom, _ = mha_custom(x, x, x)
out_torch, _ = mha_torch(x, x, x)
print(f"Max difference: {(out_custom - out_torch).abs().max().item():.2e}")  # Should be ~1e-6

# TODO: Add causal mask for autoregressive generation
# TODO: Implement full transformer block with pre-norm
# TODO: Train on a toy sequence task (copy task or reverse)
```

---

## Project 7: Variational Autoencoder on MNIST

**From:** Part 3 (Probability), Part 6 (Information Theory), Part 10 (Deep Learning)
**Time:** 6–8 hours
**Level:** Intermediate-Advanced

### Goal

Train a VAE on MNIST. Visualize: (1) the 2D latent space, (2) reconstructions, (3) interpolations between digits.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module):
    def __init__(self, latent_dim=2, hidden_dim=400, output_dim=784):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h))

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=2):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + sigma * eps"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def elbo_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    ELBO = Reconstruction term + KL term
    Reconstruction: Binary cross-entropy (pixel-wise)
    KL: KL(N(mu, sigma^2) || N(0, 1)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    """
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (BCE + beta * KLD) / x.size(0)

# Training loop
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_loader = DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=128, shuffle=True
)

for epoch in range(20):
    model.train()
    total_loss = 0
    for x, _ in train_loader:
        x = x.view(-1, 784).to(device)
        optimizer.zero_grad()
        recon_x, mu, logvar = model(x)
        loss = elbo_loss(recon_x, x, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.2f}")

# TODO: Visualize 2D latent space colored by digit label
# TODO: Sample from prior and decode to generate new digits
# TODO: Interpolate between two digits in latent space
```

---

## Project 8: A/B Test Simulator

**From:** Part 4 (Statistics), Part 5 (A/B Testing)
**Time:** 4–5 hours
**Level:** Intermediate

### Goal

Build an interactive A/B testing simulator that: runs simulations, computes p-values, checks for peeking problem, demonstrates sequential testing, and visualizes power curves.

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class ABTestSimulator:
    def __init__(self, true_control_rate, true_treatment_rate, n_per_group):
        self.p_c = true_control_rate
        self.p_t = true_treatment_rate
        self.n = n_per_group
        self.true_lift = (true_treatment_rate - true_control_rate) / true_control_rate

    def simulate_one_test(self, alpha=0.05):
        """Run one simulated A/B test."""
        control = np.random.binomial(1, self.p_c, self.n)
        treatment = np.random.binomial(1, self.p_t, self.n)

        # Proportions Z-test
        p_hat_c = control.mean()
        p_hat_t = treatment.mean()
        p_hat_pooled = (control.sum() + treatment.sum()) / (2 * self.n)

        se = np.sqrt(p_hat_pooled * (1 - p_hat_pooled) * (2 / self.n))
        z_stat = (p_hat_t - p_hat_c) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        return {
            'p_value': p_value,
            'significant': p_value < alpha,
            'observed_lift': (p_hat_t - p_hat_c) / p_hat_c,
            'z_stat': z_stat
        }

    def estimate_power(self, alpha=0.05, n_simulations=1000):
        """Estimate statistical power via Monte Carlo."""
        results = [self.simulate_one_test(alpha) for _ in range(n_simulations)]
        return np.mean([r['significant'] for r in results])

    def peeking_simulation(self, check_every=50, alpha=0.05):
        """Simulate the peeking problem: check significance repeatedly."""
        control_data = np.random.binomial(1, self.p_c, self.n)
        treatment_data = np.random.binomial(1, self.p_t, self.n)

        false_positives_with_peeking = 0
        stopped_early = False

        for i in range(check_every, self.n + 1, check_every):
            ctrl = control_data[:i]
            trt = treatment_data[:i]
            p_pooled = (ctrl.sum() + trt.sum()) / (2 * i)
            se = np.sqrt(p_pooled * (1 - p_pooled) * (2 / i))
            if se == 0:
                continue
            z = (trt.mean() - ctrl.mean()) / se
            p = 2 * (1 - stats.norm.cdf(abs(z)))
            if p < alpha:
                return True  # Stopped early with significant result

        return False

# Demonstrate power curve
p_c = 0.10
lifts = np.linspace(0.0, 0.30, 20)
n_values = [200, 500, 1000, 2000]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Power vs effect size
ax = axes[0]
for n in n_values:
    powers = []
    for lift in lifts:
        p_t = p_c * (1 + lift)
        sim = ABTestSimulator(p_c, p_t, n)
        powers.append(sim.estimate_power(n_simulations=500))
    ax.plot(lifts * 100, powers, label=f'n={n}')
ax.axhline(0.8, color='gray', linestyle='--', label='80% power target')
ax.set_xlabel('True Lift (%)')
ax.set_ylabel('Power')
ax.set_title('Statistical Power vs Effect Size')
ax.legend()

# Peeking problem
ax = axes[1]
p_values_at_end = []
peeked_significant = []
null_sim = ABTestSimulator(0.10, 0.10, 1000)  # True null effect
for _ in range(1000):
    r = null_sim.simulate_one_test()
    p_values_at_end.append(r['p_value'])
    peeked_significant.append(null_sim.peeking_simulation())

ax.hist(p_values_at_end, bins=20, density=True, alpha=0.7, label='p-value at fixed end')
ax.axhline(1.0, color='r', linestyle='--', label='Uniform (expected under H0)')
ax.set_title(f'p-values under H0\nFixed test FPR: {np.mean(np.array(p_values_at_end) < 0.05):.2%}\nPeeking FPR: {np.mean(peeked_significant):.2%}')
ax.legend()

plt.tight_layout()
plt.savefig('ab_test_analysis.png')
```

---

## Project 9: Information Bottleneck and Representation Learning

**From:** Part 6 (Information Theory), Part 10 (Deep Learning)
**Time:** 5–6 hours
**Level:** Advanced

### Goal

Implement the information bottleneck objective and visualize the "information plane" (I(X;T) vs I(T;Y)) during training a neural network. Reproduce key findings from the Tishby et al. (2017) information bottleneck paper.

```python
import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import KernelDensity

def estimate_mutual_information_kde(X, Y, bandwidth=0.5):
    """
    Estimate MI using kernel density estimation.
    X: (n, d_x), Y: (n, d_y)
    I(X;Y) = H(Y) - H(Y|X) estimated via KDE
    """
    n = len(X)

    # Estimate H(Y) using KDE
    kde_y = KernelDensity(bandwidth=bandwidth).fit(Y)
    log_p_y = kde_y.score_samples(Y)  # log p(y)
    H_Y = -log_p_y.mean()

    # Estimate H(Y|X) using KDE with X features
    XY = np.concatenate([X, Y], axis=1)
    kde_xy = KernelDensity(bandwidth=bandwidth).fit(XY)
    log_p_xy = kde_xy.score_samples(XY)  # log p(x,y)
    kde_x = KernelDensity(bandwidth=bandwidth).fit(X)
    log_p_x = kde_x.score_samples(X)  # log p(x)
    log_p_y_given_x = log_p_xy - log_p_x  # log p(y|x)
    H_Y_given_X = -log_p_y_given_x.mean()

    return max(0, H_Y - H_Y_given_X)

class InformationBottleneckNet(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=10, output_dim=2, beta=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.beta = beta

    def forward(self, x):
        T = self.encoder(x)
        # Add noise for information bottleneck (represents stochastic encoder)
        T_noisy = T + 0.1 * torch.randn_like(T)
        return self.classifier(T_noisy), T

# TODO: Generate XOR or MNIST dataset
# TODO: Train with IB objective: L = H(Y|T) + beta * I(X;T)
# TODO: Track I(X;T) and I(T;Y) at each epoch using KDE
# TODO: Plot information plane showing the compression-prediction tradeoff
```

---

## Project 10: Build a Mini Language Model (Character-Level)

**From:** Part 11 (LLMs), Part 10 (Deep Learning)
**Time:** 8–12 hours
**Level:** Advanced

### Goal

Build and train a character-level language model (Bigram → MLP → Transformer) from scratch, following Karpathy's makemore approach but extended with deeper mathematical understanding.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# Load Shakespeare or any text corpus
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r') as f:
    text = f.read()

# Build vocabulary
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

# Train/val split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

# Hyperparameters
block_size = 64  # Context length
batch_size = 32
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.1

def get_batch(split):
    data_ = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_) - block_size, (batch_size,))
    x = torch.stack([data_[i:i+block_size] for i in ix])
    y = torch.stack([data_[i+1:i+block_size+1] for i in ix])
    return x, y

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        # Weight tying
        self.token_embedding.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # Pre-norm
        x = x + self.ffn(self.ln2(x))
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        # Causal mask
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = q @ k.transpose(-2, -1) / (C // self.n_head) ** 0.5
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        att = self.dropout(F.softmax(att, dim=-1))
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

# Training
model = GPT()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

for step in range(5000):
    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if step % 500 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")

# Generate
context = torch.zeros(1, 1, dtype=torch.long)
generated = model.generate(context, max_new_tokens=200)[0].tolist()
print(decode(generated))
```

---

## Project 11: Implement MCMC Samplers

**From:** Part 3 (Probability), Part 4 (Statistics)
**Time:** 5–6 hours
**Level:** Advanced

### Goal

Implement Metropolis-Hastings and Hamiltonian Monte Carlo from scratch. Sample from a banana-shaped 2D distribution and a multi-modal distribution. Compare mixing rates.

```python
import numpy as np
import matplotlib.pyplot as plt

def log_banana(x, a=1, b=10):
    """Log-density of banana-shaped distribution."""
    return -((x[0]**2 / (2 * a**2)) + (x[1] - b * x[0]**2)**2 / 2)

def grad_log_banana(x, a=1, b=10):
    """Gradient of log-density."""
    dx0 = -x[0] / a**2 - 4 * b * x[0] * (x[1] - b * x[0]**2)
    dx1 = -(x[1] - b * x[0]**2)
    return np.array([dx0, dx1])

def metropolis_hastings(log_density, x0, n_samples, step_size=0.5):
    """Random walk Metropolis-Hastings."""
    d = len(x0)
    samples = [x0.copy()]
    x = x0.copy()
    current_log_p = log_density(x)
    accepted = 0

    for _ in range(n_samples):
        proposal = x + step_size * np.random.randn(d)
        proposal_log_p = log_density(proposal)

        log_accept = min(0.0, proposal_log_p - current_log_p)
        if np.log(np.random.uniform()) < log_accept:
            x = proposal
            current_log_p = proposal_log_p
            accepted += 1

        samples.append(x.copy())

    acceptance_rate = accepted / n_samples
    return np.array(samples), acceptance_rate

def hmc(log_density, grad_log_density, x0, n_samples, step_size=0.1, n_leapfrog=10):
    """Hamiltonian Monte Carlo."""
    d = len(x0)
    samples = [x0.copy()]
    x = x0.copy()
    accepted = 0

    for _ in range(n_samples):
        # Refresh momentum
        p = np.random.randn(d)
        current_H = -log_density(x) + 0.5 * np.dot(p, p)

        # Leapfrog
        x_prop = x.copy()
        p_prop = p.copy()
        p_prop += 0.5 * step_size * grad_log_density(x_prop)
        for _ in range(n_leapfrog):
            x_prop += step_size * p_prop
            p_prop += step_size * grad_log_density(x_prop)
        p_prop += 0.5 * step_size * grad_log_density(x_prop)
        p_prop = -p_prop  # Negate for reversibility

        proposed_H = -log_density(x_prop) + 0.5 * np.dot(p_prop, p_prop)
        if np.log(np.random.uniform()) < current_H - proposed_H:
            x = x_prop
            accepted += 1

        samples.append(x.copy())

    return np.array(samples), accepted / n_samples

# Sample and compare
np.random.seed(42)
x0 = np.array([0.0, 0.0])

mh_samples, mh_rate = metropolis_hastings(log_banana, x0, 5000, step_size=0.5)
hmc_samples, hmc_rate = hmc(log_banana, grad_log_banana, x0, 2000, step_size=0.1, n_leapfrog=10)

print(f"MH acceptance rate: {mh_rate:.1%}")
print(f"HMC acceptance rate: {hmc_rate:.1%}")

# Burn-in removal
mh_samples = mh_samples[500:]
hmc_samples = hmc_samples[200:]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, samples, title in zip(axes,
                               [mh_samples, hmc_samples],
                               ['Metropolis-Hastings', 'HMC']):
    ax.scatter(samples[::5, 0], samples[::5, 1], alpha=0.3, s=10)
    ax.set_title(f'{title}\n(n={len(samples)} samples)')
    ax.set_xlim(-3, 3); ax.set_ylim(-5, 15)

plt.tight_layout()
plt.savefig('mcmc_comparison.png')
```

---

## Project 12: Causal Inference — Estimating Treatment Effects

**From:** Part 9 (Classical ML), Part 14 (Advanced AI Math)
**Time:** 6–8 hours
**Level:** Advanced

### Goal

Implement the doubly robust estimator, propensity score matching, and CausalForest for heterogeneous treatment effect estimation. Compare on simulated data where the true effects are known.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict

def generate_synthetic_rct(n=5000, d=5, seed=42):
    """
    Generate synthetic data with known treatment effects.
    True CATE: tau(x) = 1 + 2 * x[0]  (heterogeneous effect)
    """
    np.random.seed(seed)
    X = np.random.randn(n, d)

    # Propensity score (probability of treatment)
    e = 1 / (1 + np.exp(-0.5 * X[:, 0]))  # Logistic in first feature
    T = np.random.binomial(1, e)

    # Potential outcomes
    mu0 = X[:, 0] + 0.5 * X[:, 1]  # E[Y|X, T=0]
    tau = 1 + 2 * X[:, 0]           # True treatment effect (CATE)
    mu1 = mu0 + tau

    Y = T * mu1 + (1 - T) * mu0 + np.random.randn(n) * 0.5
    return X, T, Y, tau, e

def doubly_robust_aipw(X, T, Y, outcome_model, propensity_model):
    """
    Augmented IPW (doubly robust) ATE estimator.
    Consistent if EITHER outcome_model OR propensity_model is correct.
    """
    n = len(X)

    # Fit propensity model P(T=1|X)
    e_hat = cross_val_predict(propensity_model, X, T, cv=5, method='predict_proba')[:, 1]
    e_hat = np.clip(e_hat, 0.01, 0.99)

    # Fit outcome models
    mu1_hat = cross_val_predict(outcome_model, X[T==1], Y[T==1], cv=5)
    mu0_hat = cross_val_predict(outcome_model, X[T==0], Y[T==0], cv=5)

    # Predict for all units
    mu1_all = outcome_model.fit(X[T==1], Y[T==1]).predict(X)
    mu0_all = outcome_model.fit(X[T==0], Y[T==0]).predict(X)

    # Doubly robust score
    dr_score = (mu1_all - mu0_all
                + T * (Y - mu1_all) / e_hat
                - (1-T) * (Y - mu0_all) / (1 - e_hat))

    ate_dr = dr_score.mean()
    ate_se = dr_score.std() / np.sqrt(n)

    return ate_dr, ate_se, dr_score

X, T, Y, tau_true, e_true = generate_synthetic_rct(n=5000)

# True ATE
true_ate = tau_true.mean()
print(f"True ATE: {true_ate:.3f}")

# Naive comparison (biased)
naive_ate = Y[T==1].mean() - Y[T==0].mean()
print(f"Naive (biased) ATE: {naive_ate:.3f}")

# Doubly robust
ate_dr, ate_se, dr_scores = doubly_robust_aipw(
    X, T, Y,
    outcome_model=GradientBoostingRegressor(n_estimators=100),
    propensity_model=GradientBoostingClassifier(n_estimators=100)
)
print(f"Doubly Robust ATE: {ate_dr:.3f} ± {1.96*ate_se:.3f} (95% CI)")

# Heterogeneous effect estimation (CausalForest-like via T-learner)
model_t1 = GradientBoostingRegressor(n_estimators=200).fit(X[T==1], Y[T==1])
model_t0 = GradientBoostingRegressor(n_estimators=200).fit(X[T==0], Y[T==0])
cate_hat = model_t1.predict(X) - model_t0.predict(X)

from scipy.stats import spearmanr
correlation, _ = spearmanr(tau_true, cate_hat)
print(f"CATE rank correlation (T-learner): {correlation:.3f}")
```

---

## Summary of Projects

| Project | Core Skill | Time | Difficulty |
|---------|-----------|------|------------|
| 1. PCA from scratch | SVD, eigendecomposition | 3–4h | Beginner |
| 2. Backprop from scratch | Chain rule, computation graph | 6–8h | Intermediate |
| 3. K-Means + GMM | EM algorithm, soft assignments | 4–5h | Intermediate |
| 4. Logistic Regression solvers | Newton-Raphson, convergence | 4–5h | Intermediate |
| 5. RF vs GBT comparison | Ensembles, learning curves | 3–4h | Beginner-Int |
| 6. Attention mechanism | Multi-head attention, masking | 5–6h | Intermediate |
| 7. VAE | ELBO, reparameterization | 6–8h | Int-Advanced |
| 8. A/B test simulator | Hypothesis testing, power | 4–5h | Intermediate |
| 9. Information bottleneck | Mutual information, IB objective | 5–6h | Advanced |
| 10. Mini language model | Transformer, generation | 8–12h | Advanced |
| 11. MCMC samplers | MH, HMC, mixing rates | 5–6h | Advanced |
| 12. Causal inference | AIPW, CATE estimation | 6–8h | Advanced |

---

*← [Part 15: Interview Prep](part-15-interview-prep.md) | [Part 17: Study Roadmap →](part-17-roadmap.md)*
