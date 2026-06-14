# Part 13: Numerical Methods

> **Prerequisites:** Parts 0–3 (Linear Algebra, Calculus, Probability)
> **Status:** Complete
> **Lessons:** 10

---

## The Problem This Part Solves

Real computers work with finite-precision floating point numbers, not real numbers. Every computation in a neural network, every matrix multiplication, every gradient — they all accumulate tiny errors. Those errors can compound into **numerical instability**: NaNs, exploding gradients, training failure.

Understanding numerical methods means understanding *why* code fails mysteriously, and *how* to design algorithms that are robust to floating-point reality.

---

## Lesson 13.1: Floating-Point Arithmetic

### IEEE 754 Representation

A 32-bit float (float32) stores:

$$
x = (-1)^s \times 1.f \times 2^{e - 127}
$$

- **Sign** $s$: 1 bit
- **Exponent** $e$: 8 bits (biased by 127)
- **Mantissa** $f$: 23 bits (fractional part after the implicit leading 1)

**Range:** $\approx \pm 3.4 \times 10^{38}$
**Precision:** About 7 decimal digits

For 64-bit double: 11-bit exponent, 52-bit mantissa, ~15 decimal digits.

### Machine Epsilon

The smallest number $\varepsilon$ such that $1 + \varepsilon > 1$ in floating point:

$$
\varepsilon_\text{machine} = 2^{-23} \approx 1.2 \times 10^{-7} \text{ (float32)}
$$

$$
\varepsilon_\text{machine} = 2^{-52} \approx 2.2 \times 10^{-16} \text{ (float64)}
$$

Any two numbers closer than $\varepsilon_\text{machine}$ relative to their magnitude are **identical** in floating point.

### Catastrophic Cancellation

When subtracting two nearly equal numbers, significant digits are lost:

```
a = 1.000001
b = 1.000000
a - b = 0.000001   ← only 1 significant digit from 7-digit inputs!
```

**Example:** Computing variance naively:

$$
\text{Var}(x) = \frac{\sum x_i^2}{n} - \left(\frac{\sum x_i}{n}\right)^2
$$

For large values of $x_i$ with small variance (e.g., $x_i \approx 10^6 \pm 1$), both terms are $\approx 10^{12}$ and their difference $\approx 1$ is lost to cancellation.

**Stable alternative (Welford's algorithm):**

```
M_1 = x_1, S_1 = 0
M_k = M_{k-1} + (x_k - M_{k-1}) / k
S_k = S_{k-1} + (x_k - M_{k-1})(x_k - M_k)
Var = S_n / (n - 1)
```

Never subtracts large numbers.

### Overflow and Underflow

- **Overflow:** Number exceeds max representable value → `inf`
- **Underflow:** Number smaller than min representable value → 0 (gradients silently disappear)

**float16 range:** $\approx \pm 65504$. Common in mixed-precision training. Overflow is frequent — this is why loss scaling is needed.

---

## Lesson 13.2: Numerical Stability — The Log-Sum-Exp Trick

### The Problem

Compute $\log\sum_{i=1}^{n} e^{x_i}$ for large values of $x_i$:

```python
x = [1000, 1001, 1002]
sum(exp(x_i))  # exp(1000) = overflow!
```

### The Solution

$$
\log\sum_{i} e^{x_i} = m + \log\sum_{i} e^{x_i - m}
$$

where $m = \max_i x_i$.

**Proof:** $m + \log\sum_i e^{x_i - m} = \log(e^m \sum_i e^{x_i-m}) = \log\sum_i e^{x_i}$. The subtracted maximum prevents overflow (all exponents are $\leq 0$) while the largest term is $e^0 = 1$, preventing underflow.

**Used everywhere:**
- Stable softmax: first subtract max, then apply softmax
- Log-likelihood computation in log space
- Stable cross-entropy: numerical stable implementations combine softmax and log

```python
def log_sum_exp(x):
    m = max(x)
    return m + np.log(sum(np.exp(xi - m) for xi in x))

def stable_softmax(x):
    x = x - np.max(x)  # shift by max
    e_x = np.exp(x)
    return e_x / e_x.sum()
```

---

## Lesson 13.3: Condition Number and Numerical Stability of Linear Systems

### The Condition Number

For a matrix $\mathbf{A}$, the condition number:

$$
\kappa(\mathbf{A}) = \|\mathbf{A}\| \cdot \|\mathbf{A}^{-1}\| = \frac{\sigma_\text{max}}{\sigma_\text{min}}
$$

(ratio of largest to smallest singular value)

**Interpretation:** When solving $\mathbf{A}\mathbf{x} = \mathbf{b}$, a relative error $\varepsilon$ in $\mathbf{b}$ produces a relative error up to $\kappa(\mathbf{A}) \cdot \varepsilon$ in $\mathbf{x}$.

A condition number of $10^6$ means: 6 digits of precision in $\mathbf{b}$ gives only $7 - 6 = 1$ reliable digit in $\mathbf{x}$ (for float32 with 7 digits).

**Ill-conditioned matrices in ML:**
- Gram matrices $\mathbf{X}^T\mathbf{X}$ with correlated features
- Hessians near saddle points (nearly zero eigenvalues)
- Attention score matrices with extreme values

### Ridge as Conditioning

Adding $\lambda\mathbf{I}$ to $\mathbf{X}^T\mathbf{X}$ improves the condition number:

$$
\kappa(\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I}) = \frac{\sigma_\text{max}^2 + \lambda}{\sigma_\text{min}^2 + \lambda} < \kappa(\mathbf{X}^T\mathbf{X})
$$

Ridge regularization is not just a statistical technique — it is also a numerical stabilization.

---

## Lesson 13.4: Numerical Differentiation

### Finite Differences

**Forward difference:**

$$
f'(x) \approx \frac{f(x + h) - f(x)}{h}
$$

Error: $O(h)$ — halving $h$ halves the error.

**Central difference:**

$$
f'(x) \approx \frac{f(x + h) - f(x - h)}{2h}
$$

Error: $O(h^2)$ — halving $h$ quarters the error. Much more accurate for the same $h$.

**The optimal $h$ tradeoff:** Small $h$ reduces truncation error but increases floating-point error (from subtracting nearly equal numbers). Optimal: $h \approx \sqrt{\varepsilon_\text{machine}}$ for forward differences, $h \approx \varepsilon_\text{machine}^{1/3}$ for central.

### Gradient Checking

In deep learning, verify backpropagation using numerical gradients:

$$
\frac{\partial J}{\partial \theta_i} \approx \frac{J(\theta + h\mathbf{e}_i) - J(\theta - h\mathbf{e}_i)}{2h}
$$

If the ratio $\frac{\|\nabla_\theta J_\text{analytical} - \nabla_\theta J_\text{numerical}\|}{\|\nabla_\theta J_\text{analytical}\| + \|\nabla_\theta J_\text{numerical}\|} < 10^{-5}$, the backprop implementation is likely correct.

---

## Lesson 13.5: Iterative Solvers for Linear Systems

### Why Not Always Use the Normal Equation?

Direct methods (Gaussian elimination, LU decomposition) for $\mathbf{A}\mathbf{x} = \mathbf{b}$ cost $O(n^3)$. For $n = 10^6$ (typical neural network), this is $10^{18}$ operations — infeasible.

Iterative methods exploit structure (sparsity, symmetry) to converge faster.

### Gradient Descent as a Linear Solver

For PSD matrix $\mathbf{A}$, solving $\mathbf{A}\mathbf{x} = \mathbf{b}$ is equivalent to minimizing $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^T\mathbf{A}\mathbf{x} - \mathbf{b}^T\mathbf{x}$.

**Convergence rate:** $\|x_k - x^*\| \leq \left(\frac{\kappa-1}{\kappa+1}\right)^k \|x_0 - x^*\|$

The convergence slows dramatically for ill-conditioned systems (large $\kappa$).

### Conjugate Gradient Method

CG solves $\mathbf{A}\mathbf{x} = \mathbf{b}$ for symmetric PSD $\mathbf{A}$ in at most $n$ steps (exact arithmetic):

```
r_0 = b - A*x_0, p_0 = r_0
For k = 0, 1, ...:
    alpha_k = (r_k^T r_k) / (p_k^T A p_k)
    x_{k+1} = x_k + alpha_k * p_k
    r_{k+1} = r_k - alpha_k * A * p_k
    beta_k = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
    p_{k+1} = r_{k+1} + beta_k * p_k
```

**Convergence:** $\|e_k\|_A \leq 2\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k \|e_0\|_A$ — **square root** of the GD rate. CG is much faster for ill-conditioned problems.

**AI connection:** Newton methods for training require solving $\mathbf{H}\delta\theta = -\nabla J$. CG solves this without forming $\mathbf{H}$ explicitly — only Hessian-vector products $\mathbf{H}\mathbf{v}$ are needed, computable via automatic differentiation.

---

## Lesson 13.6: Automatic Differentiation

### The Three Modes

**Numerical differentiation:** Perturb inputs, finite differences. Cost: $O(d)$ function evaluations for a gradient in $d$ dimensions. Inaccurate.

**Symbolic differentiation:** Apply derivative rules symbolically. Produces exact formulas but can have expression swell.

**Automatic differentiation:** Exact derivatives (to machine precision) via the chain rule applied to elementary operations. Cost: same order as a function evaluation. This is how PyTorch/JAX work.

### Forward Mode AD

Compute directional derivative $\nabla f(\mathbf{x}) \cdot \mathbf{v}$ for a fixed direction $\mathbf{v}$. Cost per direction: $O(1)$ function evaluations. Efficient when $n_\text{inputs} \ll n_\text{outputs}$.

### Reverse Mode AD (Backpropagation)

Compute the full gradient $\nabla f(\mathbf{x})$ in one backward pass. Cost: $O(1)$ function evaluations (constant multiple). Efficient when $n_\text{inputs} \gg n_\text{outputs}$.

**Why backprop is the right choice for neural networks:** A neural network has millions of inputs ($\theta$) and one scalar output (loss). Reverse mode costs the same as one forward pass (roughly). Forward mode would cost $O(|\theta|)$ forward passes.

### The Computation Graph

AD represents computation as a DAG where:
- **Nodes:** variables (intermediate values)
- **Edges:** elementary operations

Forward pass: populate node values. Backward pass: propagate adjoints (partial derivatives of loss with respect to each node) in reverse topological order.

**Memory vs compute tradeoff:** Storing all intermediate activations for backprop costs $O(L)$ memory for $L$ layers. **Gradient checkpointing** recomputes some activations during the backward pass, trading $O(\sqrt{L})$ memory for $O(L)$ extra compute.

---

## Lesson 13.7: Eigenvalue Algorithms

### Power Iteration

Computes the **largest eigenvalue** (in magnitude) and corresponding eigenvector:

```
v_0 = random unit vector
v_{k+1} = A * v_k / ||A * v_k||
```

Converges at rate $|\lambda_2/\lambda_1|$ — fast when largest eigenvalue is well separated.

**AI connection:** PageRank uses power iteration. Neural network spectral norms (used in spectral normalization) use power iteration.

### QR Algorithm

Computes **all eigenvalues** of a matrix:

```
A_0 = A
For k = 0, 1, ...:
    Q_k, R_k = QR(A_k)   # QR decomposition
    A_{k+1} = R_k * Q_k   # reversed product
```

$A_k$ converges to a triangular matrix (Schur form); diagonal entries are eigenvalues. Cost per iteration: $O(n^3)$; converges in $O(n)$ iterations typically.

### SVD Computation

SVD via the relationship $\mathbf{A}^T\mathbf{A}\mathbf{V} = \mathbf{V}\mathbf{\Sigma}^2$ — computing SVD reduces to eigendecomposition of a symmetric matrix.

**Randomized SVD (Halko et al.):** For rank-$r$ approximation of a large matrix $\mathbf{A} \in \mathbb{R}^{m\times n}$:
1. Draw Gaussian random matrix $\mathbf{\Omega} \in \mathbb{R}^{n \times (r+p)}$
2. $\mathbf{Y} = \mathbf{A}\mathbf{\Omega}$ — project onto low-dimensional space
3. $\mathbf{Q}, \_ = \text{QR}(\mathbf{Y})$ — orthonormal basis for range of $\mathbf{A}$
4. Compute SVD of small matrix $\mathbf{B} = \mathbf{Q}^T\mathbf{A}$

Cost: $O(mnr)$ vs $O(mn\min(m,n))$ for full SVD. Used in PCA for large datasets.

---

## Lesson 13.8: Numerical Integration

### Quadrature Rules

**Trapezoid rule:** $\int_a^b f(x)\,dx \approx \frac{h}{2}\sum_{i=0}^{n-1}(f(x_i) + f(x_{i+1}))$. Error: $O(h^2)$.

**Simpson's rule:** $\int_a^b f(x)\,dx \approx \frac{h}{3}[f(x_0) + 4f(x_1) + 2f(x_2) + 4f(x_3) + \ldots + f(x_n)]$. Error: $O(h^4)$.

### Monte Carlo Integration

For high-dimensional integrals:

$$
\int_\Omega f(\mathbf{x})\,d\mathbf{x} \approx \frac{|\Omega|}{N}\sum_{i=1}^{N}f(\mathbf{x}_i), \quad \mathbf{x}_i \sim \text{Uniform}(\Omega)
$$

Error: $O(1/\sqrt{N})$ — independent of dimension. For $d > 3$, Monte Carlo beats deterministic quadrature.

**Importance sampling:** If $f(\mathbf{x})$ is concentrated in a small region, sample from a proposal $q(\mathbf{x})$ focused there:

$$
\int f(\mathbf{x})\,d\mathbf{x} = \int \frac{f(\mathbf{x})}{q(\mathbf{x})} q(\mathbf{x})\,d\mathbf{x} \approx \frac{1}{N}\sum_i \frac{f(\mathbf{x}_i)}{q(\mathbf{x}_i)}, \quad \mathbf{x}_i \sim q
$$

---

## Lesson 13.9: Mixed Precision Training

### The Problem with float16

Training in float16 saves memory and is faster (tensor cores on modern GPUs), but:
- Small gradients underflow to 0 (magnitude < $6 \times 10^{-8}$)
- Large activations overflow (> 65504)

### Loss Scaling

Scale the loss by a large constant $S$ before backprop:

$$
\tilde{J} = S \cdot J
$$

Gradients scale by $S$ too: $\frac{\partial\tilde{J}}{\partial\theta} = S\cdot\frac{\partial J}{\partial\theta}$.

After computing gradients in float16, divide by $S$ before optimizer step. Small gradients are now in the float16 representable range.

**Dynamic loss scaling:** Start with large $S$, reduce when overflow (inf/NaN detected), increase when no overflow for many steps.

### bfloat16

16-bit format with same exponent range as float32 (8 bits) but fewer mantissa bits (7 bits):
- Same numerical range as float32 — no overflow compared to float16
- Less precision — but deep learning is surprisingly robust to reduced mantissa
- No need for loss scaling in most cases

Used in TPUs (Google), A100 GPUs, and most modern LLM training.

---

## Lesson 13.10: Root Finding and Fixed-Point Iteration

### Newton's Method for Root Finding

Find $x$ such that $f(x) = 0$:

$$
x_{k+1} = x_k - \frac{f(x_k)}{f'(x_k)}
$$

**Quadratic convergence:** Near the root, $|x_{k+1} - x^*| \leq C|x_k - x^*|^2$. Doubles the number of correct digits each iteration.

**Newton for optimization:** Minimizing $f(x)$ = finding root of $f'(x) = 0$. Second-order method:

$$
\theta_{k+1} = \theta_k - \mathbf{H}^{-1}\nabla f(\theta_k)
$$

**AI connection:** Natural gradient descent uses the Fisher information matrix $\mathbf{F}$ instead of the Hessian $\mathbf{H}$, but has the same form. K-FAC approximates $\mathbf{F}^{-1}$ using Kronecker products to make second-order optimization feasible for neural networks.

### Python Code

```python
import numpy as np
import torch

# ---- Float precision demonstration ----
print("=== Floating Point Issues ===")
x = np.float32(1e-7)
print(f"float32: 1.0 + 1e-7 = {np.float32(1.0) + x}")  # May equal 1.0
print(f"float64: 1.0 + 1e-7 = {1.0 + 1e-7}")  # 1.0000001

# Catastrophic cancellation
a = np.float32(1.000001)
b = np.float32(1.000000)
print(f"\nCancellation: {a} - {b} = {a - b}")  # Should be ~1e-6

# ---- Log-sum-exp ----
def naive_log_sum_exp(x):
    return np.log(np.sum(np.exp(x)))

def stable_log_sum_exp(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))

x = np.array([1000.0, 1001.0, 1002.0])
print(f"\nNaive log-sum-exp: {naive_log_sum_exp(x)}")  # inf
print(f"Stable log-sum-exp: {stable_log_sum_exp(x):.4f}")  # ~1002.41

# ---- Condition number ----
print("\n=== Condition Numbers ===")
A_good = np.array([[2.0, 1.0], [1.0, 2.0]])
A_bad = np.array([[1.0, 1.0 - 1e-6], [1.0 - 1e-6, 1.0]])

print(f"Good matrix condition: {np.linalg.cond(A_good):.2f}")
print(f"Ill-conditioned matrix: {np.linalg.cond(A_bad):.2e}")

# ---- Gradient checking ----
def numerical_gradient(f, x, h=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy(); x_plus[i] += h
        x_minus = x.copy(); x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

# Test: f(x) = x^T A x / 2 - b^T x, gradient = Ax - b
A = np.array([[3.0, 1.0], [1.0, 2.0]])
b = np.array([1.0, 2.0])
x = np.array([0.5, 0.5])

f = lambda x: 0.5 * x @ A @ x - b @ x
analytical_grad = A @ x - b
numerical_grad = numerical_gradient(f, x)

print(f"\nAnalytical gradient: {analytical_grad}")
print(f"Numerical gradient:  {numerical_grad}")
print(f"Max absolute error:  {np.max(np.abs(analytical_grad - numerical_grad)):.2e}")

# ---- Welford's stable variance ----
def welford_variance(data):
    """Numerically stable single-pass variance."""
    n = 0
    mean = 0.0
    M2 = 0.0
    for x in data:
        n += 1
        delta = x - mean
        mean += delta / n
        M2 += delta * (x - mean)
    return M2 / (n - 1)

# Large offset causes naive variance to fail
data = np.random.randn(1000) + 1e8
naive_var = np.mean(data**2) - np.mean(data)**2
welford_var = welford_variance(data)
true_var = np.var(data, ddof=1)
print(f"\nVariance comparison (data with large offset):")
print(f"Naive:   {naive_var:.4f}")
print(f"Welford: {welford_var:.4f}")
print(f"True:    {true_var:.4f}")
```

---

## Summary

| Issue | Cause | Fix |
|-------|-------|-----|
| Catastrophic cancellation | Subtracting nearly equal floats | Log-sum-exp, Welford's algorithm |
| Softmax overflow | $e^{z_i}$ for large $z_i$ | Subtract max before exp |
| Ill-conditioned systems | Large $\kappa(\mathbf{A})$ | Ridge regularization, preconditioning |
| float16 underflow | Small gradients | Loss scaling |
| Gradient checking | Verify backprop | Central differences with $h \approx 10^{-5}$ |
| Large SVD | $O(mn^2)$ cost | Randomized SVD |
| Backprop memory | All activations stored | Gradient checkpointing |

---

*← [Part 12: Graph Mathematics](part-12-graph-mathematics.md) | [Part 14: Advanced AI Math →](part-14-advanced-ai-math.md)*
