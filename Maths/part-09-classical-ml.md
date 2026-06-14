# Part 9: Classical ML Mathematics

> **Prerequisites:** Parts 0–8 (all of Phase 1)
> **Status:** Complete
> **Lessons:** 14

---

## The Problem This Part Solves

You know the math primitives. You know how vectors work, how gradients flow, how probability works, how to measure uncertainty. Now the question is: **how do those primitives combine into actual learning algorithms?**

This part derives every classical ML algorithm from scratch. Not just "here is the formula" — but *why* this formula, *where* it came from, *what assumptions it bakes in*, and *what breaks when those assumptions fail*.

By the end, you will understand:
- Why linear regression has a closed-form solution but logistic regression doesn't
- What SVMs are *really* doing geometrically
- Why ensembles work (mathematically, not just intuitively)
- What bias-variance tradeoff means as a mathematical identity
- How to think about generalization from first principles

---

## Lesson 9.1: Linear Regression — From Geometry to Gradients

### Why Was This Invented?

Humans observed that when you plot two related quantities (height vs weight, temperature vs crop yield), the points roughly follow a line. The question: *what is the best line?* What does "best" even mean? How do you compute it?

The answer led to one of the most important ideas in statistics: least squares estimation.

### Explain Like I Am 10 Years Old

Imagine you throw darts at a board and miss. Each dart lands a little off. You want to find the "center" of your throws — the spot that is closest to all your darts at once. Linear regression does this, but for fitting a line to data points. The best line is the one where the total squared distance from the line to all points is as small as possible.

### The Setup

Given $n$ data points $\{(\mathbf{x}_i, y_i)\}$ where $\mathbf{x}_i \in \mathbb{R}^d$ and $y_i \in \mathbb{R}$, we want to find $\mathbf{w} \in \mathbb{R}^d$ and $b \in \mathbb{R}$ such that:

$$
\hat{y}_i = \mathbf{w}^T \mathbf{x}_i + b \approx y_i
$$

Absorbing the bias into $\mathbf{w}$ by appending 1 to each $\mathbf{x}_i$:

$$
\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}
$$

where $\mathbf{X} \in \mathbb{R}^{n \times (d+1)}$ is the design matrix.

### The Loss Function: Why Squared Error?

We minimize the **Mean Squared Error (MSE)**:

$$
J(\mathbf{w}) = \frac{1}{n}\|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 = \frac{1}{n}\sum_{i=1}^{n}(y_i - \mathbf{w}^T\mathbf{x}_i)^2
$$

Why squared and not absolute error?
1. Squared error is differentiable everywhere (absolute error is not at 0)
2. Squared error penalizes outliers more heavily — a feature or a bug depending on your data
3. Squared error has a clean closed-form solution

**Probabilistic justification:** If we assume $y_i = \mathbf{w}^T\mathbf{x}_i + \varepsilon_i$ where $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$, then maximizing the log-likelihood is *exactly* minimizing MSE. Squared error loss = Gaussian noise assumption.

### The Normal Equation (Closed-Form Solution)

Take the gradient of $J(\mathbf{w})$:

$$
\nabla_\mathbf{w} J = \frac{2}{n}\mathbf{X}^T(\mathbf{X}\mathbf{w} - \mathbf{y})
$$

Set to zero and solve:

$$
\mathbf{X}^T\mathbf{X}\hat{\mathbf{w}} = \mathbf{X}^T\mathbf{y}
$$

$$
\hat{\mathbf{w}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
$$

This is the **Normal Equation**. The matrix $\mathbf{X}^T\mathbf{X}$ must be invertible (full rank). If it isn't, there are infinitely many solutions — regularization selects a unique one.

**Geometric interpretation:** $\hat{\mathbf{y}} = \mathbf{X}\hat{\mathbf{w}}$ is the **orthogonal projection** of $\mathbf{y}$ onto the column space of $\mathbf{X}$. The residual $\mathbf{y} - \hat{\mathbf{y}}$ is perpendicular to every column of $\mathbf{X}$.

### Ridge Regression (L2 Regularization)

When $\mathbf{X}^T\mathbf{X}$ is near-singular (multicollinearity), or when $d > n$ (more features than samples), the OLS solution blows up. The fix: add a penalty on the magnitude of $\mathbf{w}$.

$$
J_\text{ridge}(\mathbf{w}) = \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \lambda\|\mathbf{w}\|_2^2
$$

The solution is:

$$
\hat{\mathbf{w}}_\text{ridge} = (\mathbf{X}^T\mathbf{X} + \lambda \mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}
$$

Adding $\lambda\mathbf{I}$ to $\mathbf{X}^T\mathbf{X}$ ensures it is always invertible (eigenvalues shift by $\lambda$). Ridge shrinks all coefficients toward zero but does not zero them out.

**Bayesian interpretation:** Ridge = Gaussian prior $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \frac{\sigma^2}{\lambda}\mathbf{I})$ on the weights. MAP estimate = ridge solution.

### Lasso (L1 Regularization)

$$
J_\text{lasso}(\mathbf{w}) = \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \lambda\|\mathbf{w}\|_1
$$

No closed form. Requires iterative algorithms (coordinate descent, proximal gradient). But unlike ridge, **Lasso produces sparse solutions** — it zeros out irrelevant features. This is because the L1 ball has corners at the axes, and gradient descent tends to hit those corners.

**Bayesian interpretation:** Lasso = Laplace (double-exponential) prior on weights. MAP estimate = Lasso solution.

### Elastic Net

Combines both:

$$
J_\text{elastic}(\mathbf{w}) = \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \lambda_1\|\mathbf{w}\|_1 + \lambda_2\|\mathbf{w}\|_2^2
$$

Best of both worlds: groups correlated features (like ridge) and selects sparse solutions (like lasso). Used in genomics, NLP.

### Numerical Example

```
X = [[1, 1], [1, 2], [1, 3], [1, 4]]   # design matrix with bias column
y = [2, 3, 4, 5]

X^T X = [[4, 10], [10, 30]]
X^T y = [14, 40]

Solve: [4, 10; 10, 30] * w = [14, 40]
w = [1, 1]   # intercept=1, slope=1

So: y_hat = 1 + 1*x — a perfect fit (no noise in this example)
```

### Python Code

```python
import numpy as np

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

# Ridge
lam = 1.0
w_ridge = np.linalg.solve(X.T @ X + lam * np.eye(2), X.T @ y)
print(f"Ridge (λ=1): intercept={w_ridge[0]:.3f}, slope={w_ridge[1]:.3f}")

# Compare MSE
y_hat_ols = X @ w_ols
y_hat_ridge = X @ w_ridge
print(f"OLS MSE: {np.mean((y - y_hat_ols)**2):.4f}")
print(f"Ridge MSE: {np.mean((y - y_hat_ridge)**2):.4f}")
```

### AI Connection

- **Embeddings are linear:** The output layer of a language model is a linear regression from hidden state to logits
- **Ridge = weight decay:** The `weight_decay` parameter in Adam/SGD is ridge regularization on all weights
- **Feature importance:** The magnitude of $\hat{\mathbf{w}}$ (after standardizing features) tells you which features drive predictions

### Common Mistakes

1. **Not scaling features:** Ridge/Lasso penalize the *magnitude* of weights. If one feature is in thousands and another in fractions, the penalty is unfair. Always standardize before regularizing.
2. **Forgetting the bias term:** Regularizing the bias (intercept) shrinks it toward zero, which is rarely desired. Usually, exclude the bias from regularization.
3. **Using OLS with $n < d$:** When features exceed samples, $\mathbf{X}^T\mathbf{X}$ is rank-deficient. You need Ridge/Lasso.

### Interview Questions

**Q: Why does Lasso produce sparse solutions but Ridge does not?**
A: Geometrically, the L1 constraint region (a diamond/cross-polytope) has corners on the coordinate axes. The loss function contours are ellipsoids. When the ellipsoid intersects the L1 ball, it tends to hit a corner where some coordinates are exactly zero. The L2 ball (sphere) has no corners, so the intersection point is generically non-zero in all coordinates.

**Q: What is the condition for OLS to have a unique solution?**
A: $\mathbf{X}^T\mathbf{X}$ must be invertible, which requires $\mathbf{X}$ to have full column rank (rank = $d+1$), which requires $n \geq d+1$ and no perfectly collinear features.

**Q: What is the SVD connection to least squares?**
A: Using SVD $\mathbf{X} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$, the OLS solution is $\hat{\mathbf{w}} = \mathbf{V}\mathbf{\Sigma}^{-1}\mathbf{U}^T\mathbf{y}$. The pseudoinverse $\mathbf{X}^+ = \mathbf{V}\mathbf{\Sigma}^+\mathbf{U}^T$ generalizes to the non-full-rank case.

---

## Lesson 9.2: Logistic Regression — Classification via MLE

### Why Was This Invented?

Linear regression predicts real numbers. But sometimes you want to predict a *category* — spam or not spam, cancer or not cancer. Simply thresholding linear regression at 0.5 is unstable and gives poorly calibrated probabilities. Logistic regression gives you a proper probabilistic model.

### The Model

Model the probability that $y = 1$ given features $\mathbf{x}$:

$$
P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T\mathbf{x} + b)}}
$$

The sigmoid function $\sigma(z) = \frac{1}{1+e^{-z}}$ maps any real number to $(0, 1)$, making it a valid probability.

**Why sigmoid?** It arises naturally from the log-odds (logit):

$$
\log\frac{P(y=1|\mathbf{x})}{P(y=0|\mathbf{x})} = \mathbf{w}^T\mathbf{x} + b
$$

If you assume the log-odds is a linear function of $\mathbf{x}$, you get logistic regression.

### Training: Maximum Likelihood Estimation

The likelihood of observing the training labels under the model:

$$
\mathcal{L}(\mathbf{w}) = \prod_{i=1}^{n} P(y_i \mid \mathbf{x}_i) = \prod_{i=1}^{n} \hat{p}_i^{y_i}(1-\hat{p}_i)^{1-y_i}
$$

where $\hat{p}_i = \sigma(\mathbf{w}^T\mathbf{x}_i)$.

Taking the log (negative log-likelihood = cross-entropy loss):

$$
J(\mathbf{w}) = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log\hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\right]
$$

**No closed form exists** because $\sigma$ is nonlinear. We use gradient descent.

### Gradient Derivation

$$
\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{n}\mathbf{X}^T(\hat{\mathbf{p}} - \mathbf{y})
$$

This has an elegant form: the gradient is the design matrix times the prediction errors. Same structure as linear regression!

**Newton-Raphson** converges faster than gradient descent by using the Hessian:

$$
\mathbf{H} = \frac{1}{n}\mathbf{X}^T\mathbf{W}\mathbf{X}
$$

where $\mathbf{W} = \text{diag}(\hat{p}_i(1-\hat{p}_i))$. Newton step: $\mathbf{w} \leftarrow \mathbf{w} - \mathbf{H}^{-1}\nabla J$.

### Decision Boundary

The decision boundary is the hyperplane $\mathbf{w}^T\mathbf{x} + b = 0$. Logistic regression is a **linear classifier** — the boundary is always flat. If the classes are not linearly separable, it cannot achieve perfect classification.

### Multiclass: Softmax Regression

For $K$ classes, replace sigmoid with softmax:

$$
P(y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T\mathbf{x}}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^T\mathbf{x}}}
$$

Loss becomes categorical cross-entropy:

$$
J = -\frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{K} \mathbf{1}[y_i = k]\log P(y=k|\mathbf{x}_i)
$$

### AI Connection

- **Neural network output layers:** Every classification neural network ends with a softmax — it's logistic regression on top of learned features
- **Calibration:** Logistic regression naturally produces calibrated probabilities. Many post-hoc calibration methods (Platt scaling) fit a logistic regression on top of raw model scores

### Common Mistakes

1. **Logistic regression doesn't always converge:** If classes are perfectly separable, the weights diverge to infinity (the sigmoid becomes a step function). Regularization fixes this.
2. **Interpreting coefficients:** A unit increase in $x_j$ increases log-odds by $w_j$. Interpret in terms of odds ratios: $e^{w_j}$ is the multiplicative change in odds.

### Interview Questions

**Q: Why can't you use MSE as the loss for logistic regression?**
A: MSE with sigmoid creates a non-convex loss surface with many local minima. Cross-entropy loss is convex when the model is logistic regression (linear in $\mathbf{w}$), guaranteeing a global minimum.

**Q: How is logistic regression related to Naive Bayes?**
A: Naive Bayes with Gaussian class-conditionals reduces to logistic regression for the posterior. They share the same functional form but estimate parameters differently (generative vs discriminative).

---

## Lesson 9.3: Support Vector Machines — Maximum Margin Classifiers

### Why Was This Invented?

There are infinitely many hyperplanes that can separate two linearly separable classes. Which one should you choose? Intuitively, you want the one that is *farthest* from both classes — the most confident separator. SVMs formalize this intuition.

### The Margin

For a hyperplane $\mathbf{w}^T\mathbf{x} + b = 0$, the **margin** is the total width of the "road" between the two classes:

$$
\text{margin} = \frac{2}{\|\mathbf{w}\|}
$$

Maximizing the margin is equivalent to minimizing $\|\mathbf{w}\|^2$.

The **support vectors** are the data points closest to the hyperplane. The entire decision boundary is determined by only these points — all other training data are irrelevant once you've found the support vectors.

### Hard-Margin SVM (Linearly Separable Case)

Optimization problem:

$$
\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2
$$

subject to $y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1$ for all $i$.

This says: each point must be on the correct side, at least margin/2 away from the hyperplane.

### Soft-Margin SVM (Non-Separable Case)

Allow some misclassification with penalty:

$$
\min_{\mathbf{w}, b, \xi} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{n}\xi_i
$$

subject to $y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 - \xi_i$ and $\xi_i \geq 0$.

$\xi_i$ is the **slack variable** — the amount by which point $i$ violates the margin. $C$ trades off margin width vs misclassification penalty.

### The Dual Problem and Kernel Trick

Using Lagrange multipliers, the dual of the SVM:

$$
\max_{\alpha} \sum_{i=1}^{n}\alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j \mathbf{x}_i^T\mathbf{x}_j
$$

subject to $\alpha_i \geq 0$ and $\sum_i \alpha_i y_i = 0$.

The key insight: the objective and prediction only depend on $\mathbf{x}_i^T\mathbf{x}_j$ — dot products between data points. This opens the door to the **kernel trick**: replace $\mathbf{x}_i^T\mathbf{x}_j$ with $K(\mathbf{x}_i, \mathbf{x}_j)$, which implicitly maps data to a high-dimensional space where it becomes linearly separable.

Common kernels:
- **RBF (Gaussian):** $K(\mathbf{x}, \mathbf{x}') = e^{-\gamma\|\mathbf{x}-\mathbf{x}'\|^2}$ — infinite dimensional feature space
- **Polynomial:** $K(\mathbf{x}, \mathbf{x}') = (\mathbf{x}^T\mathbf{x}' + c)^d$
- **Linear:** $K(\mathbf{x}, \mathbf{x}') = \mathbf{x}^T\mathbf{x}'$ — equivalent to linear SVM

### KKT Conditions

The SVM dual has a beautiful sparsity property via KKT conditions: $\alpha_i > 0$ *only* for the support vectors. This means prediction at a new point $\mathbf{x}$ only requires computing $K(\mathbf{x}, \mathbf{x}_i)$ for the support vectors, not all training points.

### AI Connection

- **Kernel SVM vs neural networks:** Before deep learning, kernel SVMs were state-of-the-art for many tasks. The "neural tangent kernel" (NTK) connects infinitely wide neural networks to kernel methods
- **SVMs in NLP:** Text SVMs with TF-IDF features were strong baselines before BERT

### Common Mistakes

1. **SVMs don't output probabilities natively.** The decision function gives a signed distance. To get calibrated probabilities, use Platt scaling.
2. **Kernel choice matters.** RBF is default but requires tuning $\gamma$. Too large: each point is its own island (overfitting). Too small: all points look the same (underfitting).
3. **Scaling is critical.** SVMs use distances. Unscaled features dominate.

### Interview Questions

**Q: How many support vectors does a linear SVM need?**
A: For a $d$-dimensional feature space, you need at least 1 and at most $n$ support vectors. Typically much fewer than $n$, which is why SVMs are efficient at test time.

**Q: What is the relationship between the SVM C parameter and regularization?**
A: Small $C$ = more regularization (wider margin, more misclassifications allowed). Large $C$ = less regularization (narrower margin, fewer misclassifications allowed). $C = 1/\lambda$ where $\lambda$ is the ridge regularization strength.

---

## Lesson 9.4: Decision Trees — Recursive Partitioning

### Why Was This Invented?

Humans make decisions using rules: "if age > 60 and smoker, then high risk." Decision trees automate the discovery of such rules from data. They are interpretable, handle nonlinearity, and require no feature scaling.

### The Splitting Criterion

At each node, pick the feature $j$ and threshold $t$ that best separates the data. "Best" is measured by **information gain** or **Gini impurity**.

**Information Gain** (used in ID3, C4.5):

$$
\text{IG}(S, j, t) = H(S) - \frac{|S_L|}{|S|}H(S_L) - \frac{|S_R|}{|S|}H(S_R)
$$

where $H(S) = -\sum_k p_k \log_2 p_k$ is the entropy of the node (from Part 6), and $S_L, S_R$ are the left and right child sets.

**Gini Impurity** (used in CART):

$$
G(S) = 1 - \sum_{k=1}^{K} p_k^2
$$

Gini is faster to compute (no logarithm) and tends to give similar results to entropy.

### Tree Construction (CART Algorithm)

```
BuildTree(S):
  if stopping_criterion(S): return Leaf(majority_class(S))
  (j*, t*) = argmax_{j,t} IG(S, j, t)
  S_L = {x in S : x_j <= t*}
  S_R = {x in S : x_j > t*}
  return Node(j*, t*, BuildTree(S_L), BuildTree(S_R))
```

The tree grows greedily — at each step it makes the locally best split, not the globally optimal one. Finding the globally optimal tree is NP-hard.

### Regression Trees

For regression, replace entropy with MSE reduction:

$$
\text{Reduction}(S, j, t) = \text{Var}(S) - \frac{|S_L|}{|S|}\text{Var}(S_L) - \frac{|S_R|}{|S|}\text{Var}(S_R)
$$

Leaf prediction = mean of $y_i$ in the leaf.

### Stopping Criteria and Pruning

**Pre-pruning:** Stop growing if:
- Node has fewer than `min_samples_split` samples
- Tree depth exceeds `max_depth`
- Information gain is below a threshold

**Post-pruning (reduced error pruning):** Grow a full tree, then iteratively remove nodes that don't improve validation accuracy.

### AI Connection

- **Feature importance:** For each feature, sum the weighted impurity reduction across all nodes that split on it. This gives a natural importance score
- **Gradient boosting:** Gradient boosted trees (XGBoost, LightGBM) use decision trees as weak learners. Understanding a single tree is prerequisite for understanding boosting

### Interview Questions

**Q: Why are decision trees high variance?**
A: Small changes in training data (one extra point) can change the entire tree structure since splits are greedy and a different split at the root propagates into completely different subtrees. This is why bagging (random forests) helps.

**Q: What is the difference between Gini and entropy?**
A: Both measure impurity. Gini is bounded in $[0, 0.5]$ for binary classification; entropy in $[0, 1]$. Entropy tends to prefer more balanced splits; Gini tends to isolate the largest class. In practice, the difference is minor.

---

## Lesson 9.5: Ensemble Methods — The Power of Crowds

### Why Was This Invented?

One decision tree is high-variance (wiggly, overfit to noise). One linear model is high-bias (too simple). Can we combine many imperfect models to get something better than any individual?

Yes. This is the core idea behind ensembles: **diversity + aggregation = robustness**.

### Bias-Variance Decomposition (Preview)

For a regression model trained on data from distribution $\mathcal{D}$:

$$
\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}^2[\hat{f}(x)] + \text{Var}[\hat{f}(x)] + \sigma^2
$$

- **Bias:** How wrong the model is on average (underfitting)
- **Variance:** How much the model changes with different training sets (overfitting)
- **Irreducible noise:** $\sigma^2$ — can never be eliminated

Ensembles reduce variance without increasing bias.

### Bagging (Bootstrap Aggregating)

**Algorithm:**
1. Draw $B$ bootstrap samples $S_1, \ldots, S_B$ (sample with replacement from training data)
2. Train model $\hat{f}_b$ on each $S_b$
3. Aggregate: $\hat{f}(x) = \frac{1}{B}\sum_b \hat{f}_b(x)$ (regression) or majority vote (classification)

**Why it reduces variance:** If $B$ models each have variance $\sigma^2$ and correlation $\rho$, the ensemble variance is:

$$
\text{Var}\left(\frac{1}{B}\sum_b \hat{f}_b\right) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2
$$

As $B \to \infty$, variance approaches $\rho\sigma^2$. The lower the correlation between models, the better. This is why diversity is key.

### Random Forests

Bagging + feature randomization: at each split, only consider a random subset of $m$ features (typically $m = \sqrt{d}$ for classification, $m = d/3$ for regression).

Why does this help? It **decorrelates** the trees. Without feature randomization, all trees in a bagged ensemble would be quite similar (they all see the same dominant features), giving high $\rho$.

**Out-of-bag (OOB) error:** Each bootstrap sample leaves out ~36.8% of training data. These out-of-bag samples can be used as a free validation set.

### Boosting

Bagging trains models **in parallel** on resampled data. Boosting trains models **sequentially**, each correcting the errors of the previous ones.

**AdaBoost:**
1. Initialize equal weights $w_i = 1/n$
2. For $t = 1, \ldots, T$:
   - Train weak learner $h_t$ on weighted data
   - Compute weighted error: $\varepsilon_t = \sum_i w_i \mathbf{1}[h_t(x_i) \neq y_i]$
   - Learner weight: $\alpha_t = \frac{1}{2}\ln\frac{1-\varepsilon_t}{\varepsilon_t}$
   - Update data weights: $w_i \leftarrow w_i \exp(-\alpha_t y_i h_t(x_i))$
3. Final: $H(x) = \text{sign}\left(\sum_t \alpha_t h_t(x)\right)$

Points that are misclassified get higher weight, forcing future models to focus on them.

**Theoretical guarantee (AdaBoost):** The training error decreases exponentially in the number of rounds $T$, as long as each weak learner does better than random chance.

### AI Connection

- **Random forests for feature selection:** OOB feature importance is one of the most reliable model-based feature importance methods
- **XGBoost dominates tabular data:** In Kaggle competitions, gradient boosted trees win on structured/tabular data more than any other method

### Common Mistakes

1. **Boosting overfits with enough rounds.** Unlike bagging, adding more boosting rounds can overfit. Use early stopping.
2. **Bagging doesn't help with high-bias models.** Averaging 100 high-bias models still gives a high-bias result. Bagging reduces variance, not bias.

---

## Lesson 9.6: Gradient Boosting — Building Residuals

### Why Was This Invented?

AdaBoost works but is brittle to outliers and limited to classification. Friedman (1999) realized that boosting could be framed as gradient descent in *function space*, enabling a unified framework for any differentiable loss.

### Functional Gradient Descent

We want to find a function $F(x)$ that minimizes:

$$
\mathcal{L} = \sum_{i=1}^{n} L(y_i, F(x_i))
$$

Instead of optimizing parameters, optimize the function itself. Define the **pseudo-residuals**:

$$
r_i^{(t)} = -\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\Big|_{F=F_{t-1}}
$$

This is the negative gradient of the loss with respect to the *model's output* at each data point.

**Algorithm:**
1. Initialize $F_0(x) = \arg\min_c \sum_i L(y_i, c)$ (e.g., the mean of $y$ for MSE)
2. For $t = 1, \ldots, T$:
   - Compute pseudo-residuals $r_i^{(t)}$
   - Fit a decision tree $h_t$ to $(x_i, r_i^{(t)})$
   - Update: $F_t(x) = F_{t-1}(x) + \eta \cdot h_t(x)$

For MSE loss: $L = \frac{1}{2}(y-F)^2$, the pseudo-residuals are just the ordinary residuals: $r_i = y_i - F_{t-1}(x_i)$.

### XGBoost Additions

XGBoost (Chen & Guestrin, 2016) adds:
- **Second-order approximation:** Uses both gradient $g_i = \partial L / \partial \hat{y}_i$ and Hessian $h_i = \partial^2 L / \partial \hat{y}_i^2$ for faster convergence
- **Regularization:** Adds $\Omega(f_t) = \gamma T + \frac{\lambda}{2}\sum_j w_j^2$ to the objective (T = number of leaves, $w_j$ = leaf weights)
- **Column subsampling:** Subsample features per tree (like random forest)
- **Sparsity awareness:** Handles missing values natively

The optimal leaf weight at leaf $j$ given samples $S_j$ in that leaf:

$$
w_j^* = -\frac{\sum_{i \in S_j} g_i}{\sum_{i \in S_j} h_i + \lambda}
$$

### Python Code

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X, y = make_regression(n_samples=500, n_features=10, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                max_depth=3, random_state=42)
gb.fit(X_train, y_train)
print(f"GB RMSE: {np.sqrt(mean_squared_error(y_test, gb.predict(X_test))):.2f}")

# Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
print(f"RF RMSE: {np.sqrt(mean_squared_error(y_test, rf.predict(X_test))):.2f}")

# Staged predictions (how ensemble improves with more trees)
staged_errors = [mean_squared_error(y_test, pred)
                 for pred in gb.staged_predict(X_test)]
best_iter = np.argmin(staged_errors)
print(f"Best iteration: {best_iter}, RMSE: {np.sqrt(staged_errors[best_iter]):.2f}")
```

### Interview Questions

**Q: What is the difference between random forest and gradient boosting?**
A: Random forest trains trees **in parallel** on bootstrap samples, averages them to reduce variance. Gradient boosting trains trees **sequentially**, each fitting the residuals of the previous, to reduce bias. RF is more robust to hyperparameters; GB can achieve lower error but requires careful tuning and early stopping.

**Q: Why use shallow trees (max_depth=3-6) in gradient boosting?**
A: Deep trees capture complex patterns individually but then have high variance. Gradient boosting relies on combining many simple weak learners. Shallow trees (stumps or 3-6 levels) are weak enough to require many of them, which is the regime where boosting's theory guarantees work best.

---

## Lesson 9.7: Kernel Methods — Learning in Infinite Dimensions

### The Kernel Trick

A **kernel function** $K: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ implicitly computes dot products in a (possibly infinite-dimensional) feature space $\phi: \mathcal{X} \to \mathcal{H}$:

$$
K(\mathbf{x}, \mathbf{x}') = \langle \phi(\mathbf{x}), \phi(\mathbf{x}') \rangle_\mathcal{H}
$$

The key: you never need to compute $\phi(\mathbf{x})$ explicitly (which might be infinite-dimensional). You only need the pairwise kernel values.

### Mercer's Theorem

A function $K(\mathbf{x}, \mathbf{x}')$ is a valid kernel (corresponds to some $\phi$) if and only if it is **symmetric positive semi-definite**: for any finite set of points, the Gram matrix $K_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)$ is PSD.

### RBF Kernel

$$
K_\text{RBF}(\mathbf{x}, \mathbf{x}') = e^{-\gamma\|\mathbf{x}-\mathbf{x}'\|^2}
$$

The feature map is infinite-dimensional (Taylor expansion of the exponential). Points that are close in input space have $K \approx 1$; points far apart have $K \approx 0$.

**Bandwidth parameter $\gamma$:** Controls the "reach" of each training point. Large $\gamma$ = narrow Gaussians = complex, high-variance decision boundary. Small $\gamma$ = wide Gaussians = smooth, high-bias decision boundary.

### Kernel Ridge Regression

Replace the dot product in ridge regression with a kernel:

$$
\hat{\mathbf{w}} = (\mathbf{K} + \lambda\mathbf{I})^{-1}\mathbf{y}
$$

Prediction: $\hat{y}(\mathbf{x}) = \sum_{i=1}^{n} \alpha_i K(\mathbf{x}_i, \mathbf{x})$ where $\mathbf{\alpha} = (\mathbf{K} + \lambda\mathbf{I})^{-1}\mathbf{y}$.

### AI Connection

- **Neural Tangent Kernel (NTK):** Infinitely wide neural networks trained with gradient descent converge to kernel regression with a specific kernel (the NTK). This connects deep learning to classical kernel theory
- **Gaussian Processes:** GP regression = kernel ridge regression with a probabilistic interpretation. The kernel defines the covariance function

---

## Lesson 9.8: k-Nearest Neighbors — Distance as Decision

### The Algorithm

Given a test point $\mathbf{x}$, find the $k$ training points closest to $\mathbf{x}$ under some distance metric. Predict by majority vote (classification) or averaging (regression).

No training phase — all computation is deferred to test time. This is called a **lazy learner** or **non-parametric method**.

### Distance Metrics

**Minkowski distance** (family parameterized by $p$):

$$
d_p(\mathbf{x}, \mathbf{x}') = \left(\sum_{j=1}^{d}|x_j - x_j'|^p\right)^{1/p}
$$

- $p=1$: Manhattan distance (L1)
- $p=2$: Euclidean distance (L2)
- $p\to\infty$: Chebyshev distance (maximum coordinate difference)

**Mahalanobis distance:** Accounts for correlations between features:

$$
d_M(\mathbf{x}, \mathbf{x}') = \sqrt{(\mathbf{x}-\mathbf{x}')^T\mathbf{\Sigma}^{-1}(\mathbf{x}-\mathbf{x}')}
$$

where $\mathbf{\Sigma}$ is the covariance matrix of the data.

### The Curse of Dimensionality

In high dimensions, k-NN degrades catastrophically.

**Volume argument:** In $d$ dimensions, the fraction of volume of a unit hypercube within distance $r$ of a corner is $r^d$. To capture 10% of data volume, you need radius $r = 0.1^{1/d}$.

For $d=1$: $r = 0.1$ — you look at 10% of the feature range.
For $d=10$: $r = 0.79$ — you look at 79% of the feature range.
For $d=100$: $r = 0.977$ — you look at essentially all the feature range.

The nearest neighbor is no longer meaningfully "near." All points become equidistant in high dimensions. The ratio of max to min distance approaches 1:

$$
\lim_{d\to\infty} \frac{\max_i d(\mathbf{x}, \mathbf{x}_i) - \min_i d(\mathbf{x}, \mathbf{x}_i)}{\min_i d(\mathbf{x}, \mathbf{x}_i)} \to 0
$$

### Interview Questions

**Q: What is the decision boundary of 1-NN?**
A: The Voronoi diagram of the training points. Each training point "owns" the region of space closer to it than any other training point.

**Q: How does $k$ affect bias and variance?**
A: $k=1$: zero training error, high variance (noisy decision boundary). Large $k$: smoother boundary, higher bias. Optimal $k$ is chosen by cross-validation. Asymptotically, $k$-NN with $k/n \to 0$ and $k \to \infty$ achieves the Bayes error rate.

---

## Lesson 9.9: Dimensionality Reduction — PCA Deep Dive and Beyond

### PCA Revisited: The Optimization View

From Part 1, PCA finds directions of maximum variance. Here we connect this to the SVD and matrix approximation.

**Objective:** Find the rank-$r$ matrix $\hat{\mathbf{X}}$ minimizing:

$$
\|\mathbf{X} - \hat{\mathbf{X}}\|_F^2
$$

By the Eckart-Young theorem, the solution is the truncated SVD: $\hat{\mathbf{X}} = \mathbf{U}_r\mathbf{\Sigma}_r\mathbf{V}_r^T$.

The **proportion of variance explained** by the first $r$ components:

$$
\text{PVE}(r) = \frac{\sum_{i=1}^{r}\sigma_i^2}{\sum_{i=1}^{d}\sigma_i^2}
$$

### t-SNE (t-Distributed Stochastic Neighbor Embedding)

PCA is linear and preserves global structure. t-SNE is nonlinear and preserves *local* structure — it's designed for visualization.

**Algorithm:**
1. Compute pairwise similarities in high-d space: $p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$ where $p_{j|i} \propto e^{-\|\mathbf{x}_i - \mathbf{x}_j\|^2 / 2\sigma_i^2}$ (Gaussian kernel, perplexity controls $\sigma_i$)
2. In the low-d space (2D), use Student-t distribution: $q_{ij} \propto (1 + \|\mathbf{z}_i - \mathbf{z}_j\|^2)^{-1}$
3. Minimize KL divergence: $J = KL(P \| Q) = \sum_{ij} p_{ij}\log\frac{p_{ij}}{q_{ij}}$

The t-distribution in low-d has heavy tails — dissimilar points are placed far apart, creating visual separation between clusters.

**Limitations:** t-SNE is stochastic, non-invertible, expensive ($O(n^2)$ or $O(n\log n)$ with Barnes-Hut), and distances in the embedding are not meaningful — only topology is.

### UMAP

Faster than t-SNE, preserves more global structure. Based on Riemannian geometry and simplicial complexes. Objective:

$$
\min_\mathbf{Z} \sum_{ij}\left[p_{ij}\log\frac{p_{ij}}{q_{ij}} + (1-p_{ij})\log\frac{1-p_{ij}}{1-q_{ij}}\right]
$$

The second term (absent in t-SNE) encourages repulsion of non-neighbors, better preserving global structure.

---

## Lesson 9.10: Clustering — Unsupervised Structure Finding

### K-Means: The Math

**Objective:** Partition $n$ points into $k$ clusters to minimize the within-cluster sum of squares (WCSS):

$$
J = \sum_{c=1}^{k}\sum_{\mathbf{x} \in C_c}\|\mathbf{x} - \mathbf{\mu}_c\|^2
$$

where $\mathbf{\mu}_c = \frac{1}{|C_c|}\sum_{\mathbf{x}\in C_c}\mathbf{x}$ is the centroid of cluster $c$.

**Lloyd's algorithm (EM view):**
- **E-step:** Assign each point to nearest centroid
- **M-step:** Update centroids to mean of assigned points

**Convergence:** WCSS is non-increasing at each step. Since there are only finitely many partitions, convergence is guaranteed. But to a *local* minimum — initialization matters.

**K-Means++ initialization:** Choose first centroid uniformly at random. Choose each subsequent centroid proportional to its squared distance to the nearest existing centroid. This gives an $O(\log k)$ approximation guarantee.

### Gaussian Mixture Models (Soft Clustering)

K-means makes hard assignments. GMM makes soft (probabilistic) assignments.

Model: $\mathbf{x} \sim \sum_{c=1}^{k}\pi_c \mathcal{N}(\mathbf{\mu}_c, \mathbf{\Sigma}_c)$ where $\pi_c$ are mixing weights.

**EM for GMM:**
- **E-step:** Compute responsibilities $r_{ic} = P(z=c|\mathbf{x}_i) \propto \pi_c \mathcal{N}(\mathbf{x}_i; \mathbf{\mu}_c, \mathbf{\Sigma}_c)$
- **M-step:** Update $\pi_c, \mathbf{\mu}_c, \mathbf{\Sigma}_c$ using weighted statistics

K-means is a special case of GMM where $\mathbf{\Sigma}_c = \sigma^2\mathbf{I}$ and the limit $\sigma \to 0$ turns soft responsibilities into hard assignments.

### Choosing K

- **Elbow method:** Plot WCSS vs $k$; look for the "elbow" where adding more clusters gives diminishing returns
- **Silhouette score:** For each point, $s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$ where $a_i$ = average intra-cluster distance and $b_i$ = average distance to nearest other cluster. Range $[-1, 1]$; higher is better.

---

## Lesson 9.11: Bias-Variance Tradeoff — The Fundamental Tension

### The Decomposition

For a regression problem with target $y = f(x) + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, \sigma^2)$, and a model $\hat{f}$ trained on dataset $\mathcal{D}$:

$$
\mathbb{E}_\mathcal{D}[(y - \hat{f}(x))^2] = \text{Bias}^2(\hat{f}(x)) + \text{Var}(\hat{f}(x)) + \sigma^2
$$

where:

$$
\text{Bias}(\hat{f}(x)) = \mathbb{E}_\mathcal{D}[\hat{f}(x)] - f(x)
$$

$$
\text{Var}(\hat{f}(x)) = \mathbb{E}_\mathcal{D}\left[(\hat{f}(x) - \mathbb{E}_\mathcal{D}[\hat{f}(x)])^2\right]
$$

**Derivation sketch:**

$$
\mathbb{E}[(y-\hat{f})^2] = \mathbb{E}[(y - f + f - \mathbb{E}[\hat{f}] + \mathbb{E}[\hat{f}] - \hat{f})^2]
$$

Expanding and using independence of noise $\varepsilon$ from $\hat{f}$:

$$
= \sigma^2 + \underbrace{(f(x) - \mathbb{E}[\hat{f}(x)])^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]}_{\text{Variance}}
$$

### Double Descent

Classical wisdom: as model complexity increases, test error forms a U-shape (bias-variance tradeoff). The optimal complexity balances bias and variance.

Modern deep learning observes **double descent**: as model size continues to grow past the interpolation threshold (zero training error), test error *decreases again*. Ultra-large models that memorize training data can still generalize well.

This challenges the classical U-shape story and is an active research area.

---

## Lesson 9.12: VC Dimension and PAC Learning

### Why Generalization Matters Mathematically

Empirically, a model trained on 1000 examples and tested on a held-out set might generalize well. But why? What guarantees that low training error implies low test error?

PAC (Probably Approximately Correct) learning theory gives mathematical answers.

### VC Dimension

The **VC dimension** of a hypothesis class $\mathcal{H}$ is the size of the largest set of points that can be **shattered** (classified in all $2^n$ possible ways) by some $h \in \mathcal{H}$.

**Examples:**
- Linear classifiers in $\mathbb{R}^d$: VC dim $= d + 1$
- Decision stumps (threshold on 1 feature): VC dim $= 2$
- $k$-NN: VC dim $= \infty$ (in theory can shatter any set)
- Neural networks with $W$ weights: VC dim $= O(W \log W)$

### Generalization Bound

With $n$ training samples and a hypothesis class of VC dimension $d_{VC}$, for any $\delta > 0$, with probability at least $1 - \delta$:

$$
\text{err}_\text{test}(h) \leq \text{err}_\text{train}(h) + O\left(\sqrt{\frac{d_{VC}\log(n/d_{VC}) + \log(1/\delta)}{n}}\right)
$$

This says: if $n \gg d_{VC}$, training error is a good proxy for test error.

**Implication:** Model complexity (VC dim) should be proportional to training set size. This is why regularization (reducing effective complexity) is essential.

### Rademacher Complexity

A more refined measure than VC dimension, based on how well $\mathcal{H}$ can fit random noise:

$$
\hat{\mathcal{R}}_n(\mathcal{H}) = \mathbb{E}_\sigma\left[\sup_{h\in\mathcal{H}}\frac{1}{n}\sum_{i=1}^{n}\sigma_i h(x_i)\right]
$$

where $\sigma_i \in \{-1, +1\}$ are random Rademacher variables.

---

## Lesson 9.13: Causal ML — Beyond Correlation

### Why Causality Matters

Correlation: "Ice cream sales correlate with drowning rates." Causation: "The true cause is summer — it drives both." Building a model on correlation would give wrong interventions: banning ice cream won't prevent drowning.

### Potential Outcomes Framework

For each unit $i$, define:
- $Y_i(1)$: outcome if treated ($T_i = 1$)
- $Y_i(0)$: outcome if not treated ($T_i = 0$)

The **Individual Treatment Effect (ITE):** $\tau_i = Y_i(1) - Y_i(0)$.

We observe only one potential outcome (the fundamental problem of causal inference). The **Average Treatment Effect (ATE):**

$$
\text{ATE} = \mathbb{E}[Y(1) - Y(0)] = \mathbb{E}[Y(1)] - \mathbb{E}[Y(0)]
$$

### Conditional Average Treatment Effect (CATE)

Heterogeneous treatment effects vary by individual:

$$
\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x]
$$

Estimating $\tau(x)$ is the goal of **uplift modeling** — crucial for targeting: who benefits most from an intervention?

### Doubly Robust Estimator

$$
\hat{\tau}_\text{DR} = \frac{1}{n}\sum_i \left[\hat{\mu}_1(x_i) - \hat{\mu}_0(x_i) + \frac{T_i(Y_i - \hat{\mu}_1(x_i))}{\hat{e}(x_i)} - \frac{(1-T_i)(Y_i - \hat{\mu}_0(x_i))}{1-\hat{e}(x_i)}\right]
$$

where $\hat{\mu}_t(x) = \hat{\mathbb{E}}[Y|X=x, T=t]$ (outcome model) and $\hat{e}(x) = \hat{P}(T=1|X=x)$ (propensity model). Doubly robust means: consistent if *either* the outcome model or the propensity model is correctly specified.

---

## Lesson 9.14: Putting It All Together — The ML Algorithm Selection Map

### Decision Framework

```
Is the output continuous?
├── YES → Regression
│   ├── Linear? → Ridge/Lasso/Elastic Net
│   ├── Nonlinear, structured? → Gradient Boosted Trees (XGBoost)
│   └── Nonlinear, unstructured? → Neural Networks
└── NO → Classification
    ├── Need probabilities? → Logistic Regression (calibrated)
    ├── Need interpretability? → Decision Tree
    ├── Many correlated features? → Random Forest
    ├── High dimensional, few samples? → SVM with kernel
    └── Deep features? → Neural Networks

Is the data unlabeled?
├── Find natural groups → K-Means, GMM, DBSCAN
├── Compress for visualization → t-SNE, UMAP
└── Find structure for downstream → PCA, autoencoders

Do you need to understand causes, not just correlations?
└── Use CATE estimation, double ML, or IV regression
```

### When Each Algorithm Excels

| Algorithm | Excels When | Struggles When |
|-----------|------------|----------------|
| Linear Regression | Large n, interpretability needed | Nonlinear relationships |
| Logistic Regression | Baseline classifier, calibrated probabilities | Complex decision boundaries |
| SVM | High-d, small-n, kernel trick needed | Very large n (slow training) |
| Decision Tree | Interpretability required | Data with noise |
| Random Forest | Tabular data, robust baseline | Extrapolation beyond training range |
| Gradient Boosting | Highest accuracy on tabular data | Hyperparameter sensitivity |
| k-NN | Simple baseline, local patterns | High-dimensional data |
| k-Means | Spherical clusters, fast | Non-spherical clusters, unknown k |

---

## Summary

| Concept | Key Formula | Why It Matters |
|---------|-------------|----------------|
| OLS | $\hat{\mathbf{w}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ | Closed-form linear regression |
| Ridge | $(\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$ | Regularized regression |
| Logistic | $P(y=1|x) = \sigma(\mathbf{w}^T\mathbf{x})$ | Classification with probabilities |
| SVM margin | $2/\|\mathbf{w}\|$ | Maximum margin principle |
| Kernel trick | $K(\mathbf{x},\mathbf{x}') = \langle\phi(\mathbf{x}),\phi(\mathbf{x}')\rangle$ | Implicit high-d features |
| Ensemble variance | $\rho\sigma^2 + (1-\rho)\sigma^2/B$ | Why diversity reduces variance |
| Bias-Variance | $\text{Bias}^2 + \text{Var} + \sigma^2$ | Fundamental error decomposition |
| VC bound | $\text{err} \leq \hat{\text{err}} + O(\sqrt{d_{VC}/n})$ | When models generalize |

---

*← [Part 8: Model Evaluation Metrics](part-08-model-evaluation-metrics.md) | [Part 10: Deep Learning Mathematics →](part-10-deep-learning.md)*
