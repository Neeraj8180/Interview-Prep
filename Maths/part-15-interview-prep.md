# Part 15: Interview Prep — 100 Math Questions

> **Prerequisites:** Parts 0–14
> **Status:** Complete

---

## How to Use This Part

Each question has:
- A **core concept** tag
- The **question** exactly as asked in interviews
- The **answer** (what the interviewer actually wants to hear)
- **Follow-ups** that distinguish good answers from excellent ones

These questions are drawn from FAANG/tier-1 AI lab interviews for ML Engineer, Research Scientist, and Applied Scientist roles.

---

## Linear Algebra (Questions 1–20)

**Q1.** What is the rank of a matrix, and why does it matter?

**A:** Rank = the number of linearly independent rows (or columns). It tells you the dimensionality of the output space of the linear map. In ML: rank of the data matrix $\mathbf{X}$ determines whether the normal equation has a unique solution. Low-rank matrices can be compressed (matrix factorization for recommendations, LoRA for LLMs).

**Follow-up:** If $\mathbf{A} \in \mathbb{R}^{m\times n}$ with $m < n$, what is the maximum possible rank? Answer: $m$.

---

**Q2.** What does the determinant of a matrix represent geometrically?

**A:** The signed volume scaling factor of the linear transformation. A matrix with determinant 2 maps unit hypercubes to objects with volume 2. Determinant 0 means the transformation collapses space to a lower dimension (matrix is singular, non-invertible). Negative determinant means orientation is flipped.

---

**Q3.** Why is the covariance matrix always positive semi-definite?

**A:** For any vector $\mathbf{v}$, $\mathbf{v}^T\mathbf{\Sigma}\mathbf{v} = \mathbf{v}^T\mathbb{E}[(\mathbf{x}-\boldsymbol{\mu})(\mathbf{x}-\boldsymbol{\mu})^T]\mathbf{v} = \mathbb{E}[(\mathbf{v}^T(\mathbf{x}-\boldsymbol{\mu}))^2] \geq 0$. It's the variance of the projection of $\mathbf{x}$ onto $\mathbf{v}$ — always non-negative.

---

**Q4.** What is SVD and what are its main applications in ML?

**A:** SVD decomposes any matrix $\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$. The singular values $\sigma_i$ measure the importance of each component. Applications: (1) PCA — principal components are right singular vectors $\mathbf{V}$ of the centered data matrix; (2) Low-rank approximation — truncated SVD gives the best rank-$r$ approximation (Eckart-Young theorem); (3) Pseudoinverse — $\mathbf{A}^+ = \mathbf{V}\mathbf{\Sigma}^+\mathbf{U}^T$; (4) Recommender systems (matrix factorization); (5) LoRA — freeze pretrained weights, add low-rank $\Delta\mathbf{W} = \mathbf{A}\mathbf{B}$.

---

**Q5.** What is an eigenvector and what does it represent physically?

**A:** $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$. An eigenvector is a direction that doesn't rotate under the linear transformation — it only scales by $\lambda$. In PCA, eigenvectors of the covariance matrix are the principal components — directions of maximum variance. For Markov chains, the eigenvector with eigenvalue 1 is the stationary distribution.

---

**Q6.** What is the difference between L1 and L2 regularization?

**A:** L2 (Ridge): adds $\lambda\|\mathbf{w}\|_2^2$. Shrinks all weights toward zero, closed-form solution, no exact sparsity. L1 (Lasso): adds $\lambda\|\mathbf{w}\|_1$. Drives some weights to exactly zero (sparse solutions), no closed form. Geometrically: L1 ball has corners on axes, so the loss function contour hits corners where some coordinates are zero.

---

**Q7.** What is the trace of a matrix, and when is it useful?

**A:** $\text{tr}(\mathbf{A}) = \sum_i A_{ii}$ = sum of diagonal elements = sum of eigenvalues. Key identities: $\text{tr}(\mathbf{AB}) = \text{tr}(\mathbf{BA})$ (cyclic property). In ML: total variance = $\text{tr}(\mathbf{\Sigma})$; the gradient of $\text{tr}(\mathbf{AX})$ with respect to $\mathbf{X}$ is $\mathbf{A}^T$; appears in KL divergence for Gaussians.

---

**Q8.** What is the spectral norm of a matrix, and why does it matter for neural networks?

**A:** $\|\mathbf{A}\|_2 = \sigma_\text{max}(\mathbf{A})$ = largest singular value = maximum stretch factor of the linear map. For neural networks: Lipschitz constant of a linear layer equals its spectral norm. Spectral normalization (divide weights by spectral norm at each step) controls the Lipschitz constant of the network, used in WGAN and other contexts.

---

**Q9.** Explain the geometric interpretation of the dot product.

**A:** $\mathbf{a}\cdot\mathbf{b} = \|\mathbf{a}\|\|\mathbf{b}\|\cos\theta$ where $\theta$ is the angle between them. It measures how much $\mathbf{a}$ and $\mathbf{b}$ "align." Projection of $\mathbf{b}$ onto $\mathbf{a}$: $\frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{a}\|}$. In attention: $\mathbf{q}\cdot\mathbf{k}$ measures how relevant key $\mathbf{k}$ is to query $\mathbf{q}$, based on their alignment in the representation space.

---

**Q10.** If $\mathbf{A}$ is symmetric, what can you say about its eigendecomposition?

**A:** By the spectral theorem: symmetric matrices have real eigenvalues and orthogonal eigenvectors. So $\mathbf{A} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^T$ where $\mathbf{Q}$ is orthogonal and $\mathbf{\Lambda}$ is diagonal with real entries. Additionally, the matrix is PSD iff all eigenvalues $\geq 0$, which makes it valid as a covariance matrix.

---

**Q11.** What is the Kronecker product and where does it appear in deep learning?

**A:** $(\mathbf{A} \otimes \mathbf{B})_{ij,kl} = A_{ik}B_{jl}$. Size: if $\mathbf{A}$ is $m\times n$ and $\mathbf{B}$ is $p\times q$, then $\mathbf{A}\otimes\mathbf{B}$ is $mp\times nq$. Appears in: K-FAC (Kronecker-Factored Approximate Curvature) — approximates the Fisher information matrix as a Kronecker product for efficient second-order optimization; tensor products in quantum computing; Hadamard transforms.

---

**Q12.** What does "orthogonal" mean and why are orthogonal matrices useful?

**A:** Orthogonal matrix: $\mathbf{Q}^T\mathbf{Q} = \mathbf{I}$. Columns (and rows) are orthonormal. Key property: $\|\mathbf{Q}\mathbf{x}\|_2 = \|\mathbf{x}\|_2$ — orthogonal transformations preserve lengths and angles (isometries). Useful: QR decomposition for solving least squares; rotation matrices in 3D; orthogonal initialization avoids vanishing/exploding gradients at initialization.

---

**Q13.** What is the pseudoinverse and when do you need it?

**A:** $\mathbf{A}^+ = \mathbf{V}\mathbf{\Sigma}^+\mathbf{U}^T$ (from SVD, $\Sigma^+$ takes reciprocals of non-zero singular values). Needed when $\mathbf{A}$ is not square or not full rank. For OLS with rank-deficient $\mathbf{X}$, $\hat{\mathbf{w}} = \mathbf{X}^+\mathbf{y}$ gives the minimum-norm least-squares solution.

---

**Q14.** How does PCA relate to SVD?

**A:** PCA of mean-centered data matrix $\mathbf{X}$ (n×d) computes principal components = eigenvectors of $\mathbf{X}^T\mathbf{X}/n = \mathbf{V}\mathbf{\Sigma}^2\mathbf{V}^T/n$. The principal components are the columns of $\mathbf{V}$ from the SVD $\mathbf{X} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$. PC scores (projections) = $\mathbf{U}\mathbf{\Sigma}$. SVD is numerically more stable than forming $\mathbf{X}^T\mathbf{X}$ (avoids squaring the condition number).

---

**Q15.** What are the four fundamental subspaces of a matrix?

**A:** For $\mathbf{A} \in \mathbb{R}^{m\times n}$: (1) Column space = range of $\mathbf{A}$, dim = rank $r$; (2) Null space = ker($\mathbf{A}$), dim = $n - r$; (3) Row space = range of $\mathbf{A}^T$, dim = $r$; (4) Left null space = ker($\mathbf{A}^T$), dim = $m - r$. The column space and left null space are orthogonal complements in $\mathbb{R}^m$; the row space and null space are orthogonal complements in $\mathbb{R}^n$.

---

**Q16.** What is a Gram matrix and why is it important?

**A:** $\mathbf{G} = \mathbf{X}\mathbf{X}^T$ (n×n) or $\mathbf{K} = \mathbf{X}^T\mathbf{X}$ (d×d). Gram matrix $G_{ij} = \mathbf{x}_i^T\mathbf{x}_j$ = inner product between examples. Always PSD. Key role: kernel methods compute $K(\mathbf{x}_i, \mathbf{x}_j)$ — the Gram matrix is the kernel matrix. In style transfer: Gram matrix of feature maps captures texture/style statistics.

---

**Q17.** What is matrix rank and how does low-rank structure appear in LLMs?

**A:** Weight matrices in fine-tuned LLMs often have low intrinsic rank — the difference $\Delta\mathbf{W}$ between a fine-tuned model and the base model lives in a low-dimensional subspace. LoRA (Low-Rank Adaptation) exploits this: freeze $\mathbf{W}_0$, parameterize $\Delta\mathbf{W} = \mathbf{A}\mathbf{B}$ where $\mathbf{A} \in \mathbb{R}^{d\times r}$, $\mathbf{B} \in \mathbb{R}^{r\times d}$, $r \ll d$. Saves parameters by factor $2r/d$ compared to full fine-tuning.

---

**Q18.** Why does matrix multiplication take $O(n^3)$ time and can we do better?

**A:** Naive: for each of $n^2$ output entries, compute a dot product of length $n$ → $O(n^3)$. Strassen: $O(n^{2.81})$ via divide-and-conquer (7 multiplications instead of 8 per level). Best known: $O(n^{2.37})$ (Williams et al.). Practical: modern GPUs use highly optimized BLAS routines with CUDA tensor cores; the constant matters more than the asymptotic for typical sizes.

---

**Q19.** What is the Cholesky decomposition and when is it used?

**A:** For symmetric PSD matrix $\mathbf{A}$: $\mathbf{A} = \mathbf{L}\mathbf{L}^T$ where $\mathbf{L}$ is lower triangular with positive diagonal. Faster than LU for PSD matrices ($O(n^3/3)$ vs $O(n^3/2)$). Used: sampling from multivariate Gaussians ($\mathbf{x} = \boldsymbol{\mu} + \mathbf{L}\mathbf{z}$, $\mathbf{z}\sim\mathcal{N}(0,\mathbf{I})$); solving the normal equation; Gaussian processes.

---

**Q20.** What is the relationship between convolution and matrix multiplication?

**A:** Convolution is equivalent to multiplication by a **Toeplitz matrix** (constant along diagonals) in 1D. In 2D, it's a doubly block circulant matrix. In practice: the im2col trick converts convolution to matrix multiplication, enabling GPU acceleration. Convolution can also be computed as pointwise multiplication in the Fourier domain (convolution theorem).

---

## Calculus (Questions 21–35)

**Q21.** What is the chain rule and how does backpropagation implement it?

**A:** Chain rule: $\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)$. Backpropagation applies the chain rule to the computational graph. For loss $J = f_L \circ f_{L-1} \circ \cdots \circ f_1(\theta)$: $\frac{\partial J}{\partial \theta} = \frac{\partial f_L}{\partial f_{L-1}} \cdot \frac{\partial f_{L-1}}{\partial f_{L-2}} \cdots \frac{\partial f_1}{\partial \theta}$. Backprop computes this in reverse, reusing intermediate results (dynamic programming).

---

**Q22.** What is the gradient of the cross-entropy loss with respect to the logits?

**A:** For $L = -\sum_k y_k\log\hat{p}_k$ where $\hat{p}_k = \text{softmax}(z_k)$: $\frac{\partial L}{\partial z_k} = \hat{p}_k - y_k$. The gradient is just the prediction error. This clean form is one reason cross-entropy is preferred — the gradient never saturates near zero.

---

**Q23.** What is the Jacobian matrix?

**A:** For $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian $\mathbf{J} \in \mathbb{R}^{m\times n}$ where $J_{ij} = \partial f_i / \partial x_j$. The Jacobian generalizes the gradient to vector-valued functions. In neural networks: Jacobian of a layer maps perturbations of inputs to perturbations of outputs. Used in Jacobian-vector products (forward mode AD) and vector-Jacobian products (reverse mode AD).

---

**Q24.** What is the Hessian and why is it expensive to use in deep learning?

**A:** Hessian $\mathbf{H}_{ij} = \partial^2 J / \partial\theta_i\partial\theta_j$. For a model with $N$ parameters: Hessian is $N\times N$ — for $N=10^9$ parameters, that's $10^{18}$ entries. Storing it is impossible. Computing it costs $O(N^2)$ in time and space. Practical alternatives: diagonal Hessian approximation (AdaGrad stores diagonal), low-rank approximations (K-FAC), conjugate gradient (compute $\mathbf{H}\mathbf{v}$ without forming $\mathbf{H}$).

---

**Q25.** What is the gradient of $\text{tr}(\mathbf{A}\mathbf{X})$ with respect to $\mathbf{X}$?

**A:** $\frac{\partial}{\partial \mathbf{X}}\text{tr}(\mathbf{A}\mathbf{X}) = \mathbf{A}^T$.

Key matrix calculus identities: $\frac{\partial}{\partial \mathbf{X}}\text{tr}(\mathbf{X}^T\mathbf{A}) = \mathbf{A}$; $\frac{\partial}{\partial \mathbf{X}}\|\mathbf{X}\|_F^2 = 2\mathbf{X}$; $\frac{\partial}{\partial \mathbf{x}}(\mathbf{x}^T\mathbf{A}\mathbf{x}) = (\mathbf{A}+\mathbf{A}^T)\mathbf{x}$.

---

**Q26.** What is Taylor's theorem and how is it used in optimization?

**A:** $f(x + \delta) \approx f(x) + f'(x)\delta + \frac{1}{2}f''(x)\delta^2 + O(\delta^3)$.

In optimization: gradient descent uses the first-order approximation — step in the direction of negative gradient. Newton's method uses the second-order approximation — the optimal step (minimizing the quadratic approximation) is $\delta = -f''(x)^{-1}f'(x)$. The quality of these approximations determines the optimal learning rate/step size.

---

**Q27.** What is the vanishing gradient problem and what causes it?

**A:** During backpropagation through $L$ layers, gradients involve products of Jacobians. If each Jacobian has spectral norm < 1, the product decreases exponentially. For sigmoid activations: max derivative is 0.25, so 10 layers → gradient multiplied by $\leq 0.25^{10} \approx 10^{-6}$. Fixes: ReLU (gradient = 1 in active region), residual connections (skip connections provide gradient highways), LSTM gating, gradient clipping, careful initialization.

---

**Q28.** What is the difference between a local minimum, global minimum, and saddle point?

**A:** **Local minimum:** $\nabla f = 0$, Hessian PSD (all eigenvalues $\geq 0$). **Global minimum:** lowest $f$ value over all $x$. **Saddle point:** $\nabla f = 0$, Hessian has both positive and negative eigenvalues. Neural networks have many saddle points — gradient descent can slow down near them. Deep networks typically have many local minima of similar quality (the loss landscape is benign), but saddle points are the primary obstacle in training.

---

**Q29.** What is the learning rate and how does it affect training?

**A:** Learning rate $\eta$ in $\theta \leftarrow \theta - \eta\nabla J$: controls step size. Too large: oscillates, diverges. Too small: very slow convergence. Optimal $\eta \approx 1/L$ where $L$ is the Lipschitz constant of the gradient (Lipschitz smoothness). For quadratic functions, optimal learning rate is $2/(\lambda_\text{min}+\lambda_\text{max})$ where $\lambda$s are Hessian eigenvalues. In practice: use learning rate warmup + cosine decay schedule.

---

**Q30.** Derive the gradient of the sigmoid function.

**A:** $\sigma(z) = 1/(1+e^{-z})$. $\sigma'(z) = \frac{e^{-z}}{(1+e^{-z})^2} = \sigma(z)(1-\sigma(z))$. Maximum value = 0.25 at $z=0$. This is why sigmoid causes vanishing gradients for large $|z|$.

---

**Q31.** What is a Lipschitz function?

**A:** $f$ is $L$-Lipschitz if $|f(x) - f(y)| \leq L|x - y|$ for all $x, y$. $L$ is the maximum rate of change. For neural networks: Lipschitz constant controls sensitivity to input perturbations, generalization bounds, and adversarial robustness. The spectral norm of each weight matrix bounds its Lipschitz constant; product of spectral norms bounds the network's Lipschitz constant.

---

**Q32.** What is the difference between convex and non-convex optimization?

**A:** **Convex:** $f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$. Every local minimum is global; gradient descent converges to the global minimum. Loss landscape has no "holes." **Non-convex:** typical of neural networks — multiple local minima, saddle points. Gradient descent may get stuck. In practice, the loss landscape of overparameterized neural networks is benign — local minima are often near-global.

---

**Q33.** What does the gradient of loss with respect to the input mean?

**A:** $\nabla_\mathbf{x} J$ tells you how the loss changes with small perturbations of the input. Used in: (1) Adversarial examples (FGSM: perturb input in gradient direction to maximize loss); (2) Saliency maps (visualize which pixels contribute most to the prediction); (3) Input gradient regularization (penalize large input gradients for smoothness).

---

**Q34.** Explain momentum in gradient descent.

**A:** Momentum accumulates a velocity vector: $\mathbf{v}_t = \beta\mathbf{v}_{t-1} + (1-\beta)\nabla J_t$; $\theta_t = \theta_{t-1} - \eta\mathbf{v}_t$. Typical $\beta=0.9$. Effect: averages gradients over ~$1/(1-\beta)$ steps, damping oscillations in high-curvature directions and accelerating in low-curvature directions. In the direction of consistent gradient, updates accumulate ($v \to \eta\nabla J/(1-\beta)$ = 10× amplification). Across the "valley" of the loss surface, oscillating gradients cancel.

---

**Q35.** What is the difference between Adam and SGD?

**A:** **SGD:** $\theta \leftarrow \theta - \eta\nabla J$. Same learning rate for all parameters. **Adam:** $\theta_j \leftarrow \theta_j - \eta\frac{\hat{m}_j}{\sqrt{\hat{v}_j}+\varepsilon}$ where $\hat{m}_j$ = bias-corrected first moment (moving average of gradient) and $\hat{v}_j$ = bias-corrected second moment (moving average of squared gradient). Adam adapts the learning rate per parameter. Better for sparse gradients (NLP, embeddings). SGD with careful tuning often generalizes better for image classification; Adam is more robust to hyperparameter choice.

---

## Probability and Statistics (Questions 36–60)

**Q36.** What is Bayes' theorem and how is it used in ML?

**A:** $P(H|E) = \frac{P(E|H)P(H)}{P(E)}$. Posterior = likelihood × prior / normalizing constant. In ML: (1) Naive Bayes classifiers — compute $P(\text{class}|\text{features})$; (2) Bayesian neural networks — posterior over weights; (3) MLE vs MAP — MLE ignores the prior, MAP includes it (equivalent to regularization); (4) Online learning — sequentially update posterior as new data arrives.

---

**Q37.** What is the difference between MLE and MAP estimation?

**A:** **MLE:** $\hat\theta = \arg\max_\theta P(\text{data}|\theta) = \arg\max_\theta \log P(\text{data}|\theta)$. No prior. **MAP:** $\hat\theta = \arg\max_\theta P(\theta|\text{data}) = \arg\max_\theta [\log P(\text{data}|\theta) + \log P(\theta)]$. Includes prior as regularizer. With Gaussian prior, MAP = Ridge regression. With Laplace prior, MAP = Lasso. As data → ∞, MAP → MLE (data overwhelms prior).

---

**Q38.** What is the Central Limit Theorem?

**A:** If $X_1, \ldots, X_n$ are i.i.d. with mean $\mu$ and variance $\sigma^2$, then: $\frac{\sqrt{n}(\bar{X} - \mu)}{\sigma} \xrightarrow{d} \mathcal{N}(0, 1)$ as $n\to\infty$. Even if the individual $X_i$ are not normally distributed, their sample mean becomes approximately normal for large $n$. Fundamental to hypothesis testing, confidence intervals, and explaining why Gaussian assumptions work so often in practice.

---

**Q39.** What is the Gaussian distribution and why is it so common?

**A:** $\mathcal{N}(\mu, \sigma^2)$: PDF $= \frac{1}{\sigma\sqrt{2\pi}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$. Ubiquity: (1) Maximum entropy distribution with fixed mean and variance — least informative given those constraints; (2) Sums of many independent random variables converge to it (CLT); (3) Conjugate prior for many likelihoods; (4) Analytically tractable — products of Gaussians are Gaussian, marginalization of joint Gaussians is Gaussian.

---

**Q40.** What is the KL divergence and what properties does it have?

**A:** $KL(P\|Q) = \sum_x P(x)\log\frac{P(x)}{Q(x)}$. Properties: (1) Non-negative: $KL \geq 0$ (Gibbs' inequality); (2) Zero iff $P = Q$; (3) **Not symmetric**: $KL(P\|Q) \neq KL(Q\|P)$; (4) **Not a metric** (no triangle inequality). Used as: loss function (cross-entropy training), variational inference (minimize $KL(q\|p)$), information-theoretic divergence. Forward KL ($KL(P\|Q)$) = zero-avoiding (Q must cover all P). Reverse KL ($KL(Q\|P)$) = zero-forcing (Q focuses on the mode of P).

---

**Q41.** What is the law of large numbers?

**A:** Weak LLN: $\bar{X}_n \xrightarrow{P} \mu$ (converges in probability). Strong LLN: $\bar{X}_n \xrightarrow{a.s.} \mu$ (converges almost surely). As sample size grows, the sample mean converges to the true mean. Foundation for: Monte Carlo estimation (large samples give accurate estimates), empirical risk minimization (sample loss approximates true loss), and all of frequentist statistics.

---

**Q42.** What is p-value and what does it NOT mean?

**A:** p-value = probability of observing data as extreme as the observed, under the null hypothesis. It does NOT mean: probability that $H_0$ is true; probability that the result is due to chance; effect size; probability of replication. Common misuse: p < 0.05 as a binary pass/fail rather than continuous evidence. Proper use: p-value tells you about the data given $H_0$, not about $H_0$ given the data (that requires Bayesian inference).

---

**Q43.** What is the difference between Type I and Type II errors?

**A:** **Type I (false positive):** Reject $H_0$ when it's true. Probability = $\alpha$ (significance level). **Type II (false negative):** Fail to reject $H_0$ when it's false. Probability = $\beta$. Power = $1 - \beta$ = probability of detecting a true effect. Tradeoff: reducing $\alpha$ (stricter threshold) increases $\beta$. Increasing sample size reduces both. In ML: Type I = deploy a model that doesn't work; Type II = miss a model that does work.

---

**Q44.** Explain bias-variance tradeoff in your own words.

**A:** Bias = systematic error (model is consistently wrong in a predictable direction — too simple). Variance = how much the model changes with different training data (too complex, memorizes noise). Total error = Bias² + Variance + Irreducible noise. High bias: linear model for a nonlinear function. High variance: deep decision tree memorizing training data. Fix high bias: more complex model, more features. Fix high variance: regularization, more data, simpler model, ensembles.

---

**Q45.** What is a confidence interval?

**A:** A 95% CI is a procedure that, over repeated experiments, produces intervals containing the true parameter 95% of the time. NOT: the probability that the true value lies in this specific interval is 95% (frequentist parameters are fixed, not random). In ML: report model accuracy with a confidence interval using bootstrap or asymptotic normal approximation. CI narrows as $O(1/\sqrt{n})$.

---

**Q46.** What is the expectation-maximization (EM) algorithm?

**A:** For models with latent variables, maximize the ELBO alternately: **E-step:** compute posterior over latent variables $Q(Z) = P(Z|X, \theta_\text{old})$; **M-step:** maximize $\mathbb{E}_Q[\log P(X, Z|\theta)]$ over $\theta$. Provably non-decreasing log-likelihood at each step. Examples: GMM (E: soft cluster assignments, M: update means/covariances/weights), HMM Baum-Welch, k-means (hard-assignment limit of GMM-EM).

---

**Q47.** What is the difference between generative and discriminative models?

**A:** **Generative:** Models joint $P(X, Y) = P(X|Y)P(Y)$. Can generate samples, compute likelihood, handle missing data. Examples: Naive Bayes, GMM, VAE, diffusion models. **Discriminative:** Models conditional $P(Y|X)$ directly. Doesn't waste capacity modeling the input distribution. Usually better accuracy for classification. Examples: logistic regression, SVM, neural classifiers, CRF. Divide is blurring: GPT is generative but used discriminatively; contrastive learning mixes both.

---

**Q48.** What is mutual information?

**A:** $I(X; Y) = H(X) - H(X|Y) = KL(P(X,Y) \| P(X)P(Y))$. Measures how much knowing $Y$ reduces uncertainty about $X$ (or vice versa). Zero iff $X$ and $Y$ are independent. Non-negative, symmetric. Applications: feature selection (select features with high MI with label), information bottleneck (compress input while preserving MI with label), contrastive learning objectives (maximize MI between augmented views).

---

**Q49.** What is the exponential family of distributions?

**A:** $P(x|\eta) = h(x)\exp(\eta^T T(x) - A(\eta))$ where $\eta$ = natural parameters, $T(x)$ = sufficient statistics, $A(\eta)$ = log-partition function (normalizer). Includes: Gaussian, Bernoulli, Poisson, Gamma, Beta. Key property: $\nabla A(\eta) = \mathbb{E}[T(x)]$ — gradient of log-partition function gives moments. GLMs (generalized linear models) use exponential family outputs. Natural gradient descent in the exponential family = Fisher information metric.

---

**Q50.** Explain the Monte Carlo method.

**A:** Estimate $\mathbb{E}[f(X)] = \int f(x)p(x)dx$ by drawing samples $x_1, \ldots, x_n \sim p$ and computing $\frac{1}{n}\sum_i f(x_i)$. Error: $O(1/\sqrt{n})$ regardless of dimension (unlike quadrature). Used in: stochastic gradient descent (random mini-batch = MC estimate of full gradient), MCMC for Bayesian inference, dropout (averaging over random networks), policy gradient in RL.

---

**Q51.** What is importance sampling and when is it useful?

**A:** Estimate $\mathbb{E}_p[f(X)]$ when sampling from $p$ is hard, but we can sample from proposal $q$: $\mathbb{E}_p[f] = \mathbb{E}_q[f(X)w(X)]$ where $w(X) = p(X)/q(X)$ is the importance weight. Works when $q$ has heavier tails than $p$ (covers all regions where $p > 0$). Used in: off-policy RL (correct for distribution mismatch), variational inference, REINFORCE variance reduction.

---

**Q52.** What is MCMC and why do we need it?

**A:** Markov Chain Monte Carlo: construct a Markov chain whose stationary distribution is the target distribution $\pi(\theta)$. Run the chain long enough; samples approximate $\pi$. Needed when: (1) Can't compute the normalizing constant $Z = \int p^*(\theta)d\theta$; (2) Can't sample directly (no conjugate form). Common algorithms: Metropolis-Hastings (accept/reject based on ratio), Gibbs sampling (sample each variable conditioned on others), HMC (Hamiltonian Monte Carlo — uses gradient information for efficient exploration).

---

**Q53.** What is the multivariate Gaussian and what does its covariance encode?

**A:** $\mathcal{N}(\boldsymbol{\mu}, \mathbf{\Sigma})$: PDF $\propto \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\mathbf{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$. $\mathbf{\Sigma}$ encodes: (1) Diagonal entries = variances of each dimension; (2) Off-diagonal = covariances (correlations); (3) Eigenvectors of $\mathbf{\Sigma}$ = principal axes of the distribution; (4) Eigenvalues = variance along those axes. The distribution's contours are ellipsoids aligned with eigenvectors.

---

**Q54.** What is the beta distribution and when is it used as a prior?

**A:** $\text{Beta}(\alpha, \beta)$: support $[0,1]$, mean $= \frac{\alpha}{\alpha+\beta}$, used as prior over probabilities. **Conjugate prior** for Bernoulli likelihood: if $\theta \sim \text{Beta}(\alpha, \beta)$ and $X|\theta \sim \text{Bernoulli}(\theta)$, then $\theta|X \sim \text{Beta}(\alpha + x, \beta + 1 - x)$. Interpretation: $\alpha$ = prior "heads," $\beta$ = prior "tails." Used in Bayesian A/B testing (Bayesian conversion rate estimation) and bandit algorithms (Thompson sampling).

---

**Q55.** What is the Dirichlet distribution?

**A:** Multivariate generalization of Beta: $\text{Dir}(\boldsymbol{\alpha})$ with $\sum_k\alpha_k = \alpha_0$. Conjugate prior for categorical/multinomial distributions. Used in: LDA (Latent Dirichlet Allocation) — document-topic and topic-word distributions are Dirichlet; softmax outputs can be interpreted as samples from Dirichlet in certain Bayesian models. Higher $\alpha_k$ = more probability concentrated near $k$-th simplex vertex.

---

**Q56.** What is the law of total expectation?

**A:** $\mathbb{E}[X] = \mathbb{E}[\mathbb{E}[X|Y]]$. Also called the tower property or iterated expectation. Used ubiquitously: deriving variance decomposition, computing marginal expectations, REINFORCE baseline derivation (baseline $b(s)$ doesn't change gradient's expectation since $\mathbb{E}_a[\nabla\log\pi(a|s) \cdot b(s)] = 0$).

---

**Q57.** What is the Markov assumption?

**A:** $P(X_t | X_1, \ldots, X_{t-1}) = P(X_t | X_{t-1})$ — the future is conditionally independent of the past given the present. Enables tractable inference and planning. Used in: HMMs, MDPs, RNNs (approximately), Markov Random Fields. The "Markov blanket" of a node in a graphical model = the minimal set of nodes that renders it conditionally independent of all other nodes.

---

**Q58.** What is Simpson's paradox?

**A:** An association can appear, disappear, or reverse when a lurking variable is aggregated or disaggregated. Example: Treatment A has higher success rate than B in both men and women, but lower overall (because treatment assignment is correlated with gender). Implies: always condition on confounders; aggregate statistics can be misleading. In ML: metrics can look good overall but fail for specific subgroups.

---

**Q59.** What is the difference between correlation and covariance?

**A:** $\text{Cov}(X,Y) = \mathbb{E}[(X-\mu_X)(Y-\mu_Y)]$. Units: product of original units. $\text{Corr}(X,Y) = \text{Cov}(X,Y)/(\sigma_X\sigma_Y) \in [-1,1]$. Correlation is scale-invariant. Pearson correlation captures linear relationships only; Spearman (rank correlation) captures monotonic relationships; distance correlation captures any relationship.

---

**Q60.** What is the Cramér-Rao bound?

**A:** For any unbiased estimator $\hat\theta$ of parameter $\theta$: $\text{Var}(\hat\theta) \geq 1/I(\theta)$ where $I(\theta) = \mathbb{E}\left[\left(\frac{\partial}{\partial\theta}\log p(X|\theta)\right)^2\right]$ is the Fisher information. No unbiased estimator can have lower variance than the reciprocal of Fisher information. The MLE achieves this bound asymptotically — it is asymptotically efficient.

---

## Deep Learning (Questions 61–80)

**Q61.** Why do we use mini-batch gradient descent instead of full batch or single sample?

**A:** **Full batch:** accurate gradient but expensive per update ($O(n)$ per step). No noise → may converge to sharp minima that generalize poorly. **Stochastic (n=1):** noisy gradient, cheap per update, but high variance. **Mini-batch:** balances both. The noise from mini-batches acts as regularization, escapes sharp minima, and converges to flatter minima that generalize better. GPU parallelism also makes batch size 32–256 nearly as fast as single-sample on modern hardware.

---

**Q62.** What is batch normalization and what problem does it solve?

**A:** BN normalizes layer inputs to have zero mean and unit variance within each mini-batch, then scales/shifts with learned parameters $\gamma, \beta$. Solves: (1) Internal covariate shift — the distribution of each layer's inputs shifts as weights update, forcing later layers to constantly adapt; (2) Enables higher learning rates (better-conditioned optimization); (3) Mild regularization effect (batch statistics add noise). Limitation: doesn't work with small batches or at inference time (uses running statistics).

---

**Q63.** Why do residual connections work?

**A:** $\mathbf{x}^{(\ell+1)} = \mathbf{x}^{(\ell)} + F(\mathbf{x}^{(\ell)})$. During backpropagation: $\frac{\partial \mathbf{x}^{(L)}}{\partial \mathbf{x}^{(\ell)}} = \prod_{k=\ell}^{L-1}(\mathbf{I} + \frac{\partial F_k}{\partial \mathbf{x}^{(k)}})$. The identity term ensures the gradient has a direct path through the skip connection without being multiplied by (potentially small) Jacobians. Also: at initialization, $F \approx 0$, so the network acts like a shallower network, which is easier to optimize. Residuals allow very deep networks (1000+ layers) that would otherwise fail to train.

---

**Q64.** What is dropout and why does it help?

**A:** During training, randomly zero out each neuron with probability $p$ (typically 0.5). At test time, scale by $(1-p)$ (or equivalently, scale during training with inverted dropout). Why it helps: (1) Prevents co-adaptation — neurons can't rely on specific other neurons; (2) Implicit ensemble — each forward pass trains a different subnetwork; (3) Can be seen as data augmentation on activation patterns. Reduces overfitting significantly on small datasets.

---

**Q65.** What is the difference between pre-norm and post-norm in transformers?

**A:** **Post-norm (original transformer):** LN after the residual addition: $x \leftarrow \text{LN}(x + F(x))$. Gradients can vanish at initialization — the residual stream is small and LN normalizes it aggressively. **Pre-norm (GPT-2+, LLaMA):** LN before the sub-layer: $x \leftarrow x + F(\text{LN}(x))$. More stable training: the identity shortcut is unaffected by LN, allowing larger gradients through the skip connection. Pre-norm is now the standard.

---

**Q66.** What is label smoothing?

**A:** Instead of hard one-hot targets $y_k \in \{0,1\}$, use soft targets: $y_k^{LS} = (1-\varepsilon)y_k + \varepsilon/K$ where $\varepsilon$ is typically 0.1. Prevents overconfidence — the model can't make a logit infinitely large to minimize loss. Improves calibration and generalization. Mathematically equivalent to adding a uniform distribution over labels to the cross-entropy loss.

---

**Q67.** What is the transformer's time and space complexity?

**A:** For sequence length $n$, model dimension $d$, FFN expansion factor 4:
- Self-attention: $O(n^2 d)$ time, $O(n^2)$ space (attention matrix)
- FFN: $O(n d^2)$ time, $O(n d)$ space
- Dominant term: $O(n^2 d)$ for typical sizes ($n < d$)

For $n=512$, $d=768$: attention = $512^2 \times 768 \approx 200$M ops/layer; FFN = $512 \times 768^2 \times 4 \approx 1.2$B ops/layer. FFN dominates in practice for short sequences.

---

**Q68.** What is gradient clipping and when is it needed?

**A:** $\nabla \leftarrow \nabla \cdot \min\left(1, \frac{c}{\|\nabla\|}\right)$ — rescale gradient if its norm exceeds threshold $c$. Needed for: (1) RNNs with BPTT (gradient can grow exponentially through unrolled time); (2) Transformers during warm-up (large gradients at start of training); (3) Any setting with occasional gradient spikes. Common value: $c=1.0$ for LLMs. Without clipping, one bad batch can cause a NaN cascade.

---

**Q69.** What is knowledge distillation?

**A:** Train a small "student" model to mimic a large "teacher" model. Loss combines: (1) Hard targets (true labels); (2) Soft targets (teacher's softmax outputs at temperature $T$): $L = (1-\alpha)\cdot L_\text{CE}(y, \hat{p}_\text{student}) + \alpha\cdot L_\text{CE}(\hat{p}_\text{teacher}^T, \hat{p}_\text{student}^T)$. Soft targets provide richer gradient signal — the distribution over wrong classes encodes structure the teacher learned. Student often exceeds the teacher trained directly on the same data.

---

**Q70.** What is weight tying and where is it used?

**A:** Share parameters between different parts of the model. Most common in LLMs: the input embedding matrix $\mathbf{E} \in \mathbb{R}^{V\times d}$ is reused (transposed) as the output projection matrix. Saves $Vd$ parameters (250M+ for GPT-2 large). Also improves performance: the model learns that the same vector that represents "cat" as an input should also score highly when "cat" is the correct output.

---

**Q71.** What is the difference between sparse and dense attention?

**A:** **Dense attention:** Each query attends to all keys. $O(n^2)$ complexity. **Sparse attention:** Each query attends to a subset of keys (local window, strided patterns, global tokens). $O(n\log n)$ or $O(n\cdot k)$ complexity. Examples: Longformer (sliding window + global tokens), BigBird (random + window + global). FlashAttention achieves dense attention with $O(n^2)$ compute but $O(n)$ memory via tiling.

---

**Q72.** What is a normalizing flow?

**A:** A generative model that transforms a simple distribution $p_z(\mathbf{z})$ into a complex distribution $p_x(\mathbf{x})$ via an invertible mapping $f_\theta: \mathbf{z} \to \mathbf{x}$. By change of variables: $\log p_x(\mathbf{x}) = \log p_z(f_\theta^{-1}(\mathbf{x})) + \log|\det J_{f_\theta^{-1}}(\mathbf{x})|$. Exact log-likelihood computation requires computing the Jacobian determinant — the architecture must make this tractable (e.g., triangular Jacobian → det = product of diagonal).

---

**Q73.** What is contrastive learning?

**A:** Learn representations by pulling together representations of similar (positive) pairs and pushing apart dissimilar (negative) pairs. SimCLR loss (NT-Xent): $\mathcal{L} = -\log\frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_j)/\tau)}{\sum_{k\neq i}\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_k)/\tau)}$. Positive pair: two augmented views of the same image. Negatives: all other images in the batch. Maximizes mutual information between views while being invariant to augmentation.

---

**Q74.** What is the reparameterization trick?

**A:** To differentiate through a sampling operation $\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$: express $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\varepsilon}$ where $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$. Now $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$ are deterministic functions, so gradients can flow through them. Without this: sampling is non-differentiable (REINFORCE needs high-variance estimators). With this: low-variance pathwise gradient estimator. Fundamental to VAE training.

---

**Q75.** What is the perplexity of a language model?

**A:** $\text{PPL} = \exp\left(-\frac{1}{T}\sum_{t=1}^{T}\log P(w_t|w_{<t})\right)$. Geometric mean of inverse probability per token — lower is better. Equivalent to $2^{H(p, q)}$ (exponential of cross-entropy) in bits. A model with PPL 10 assigns average probability 0.1 to each correct next token. Caveats: PPL is sensitive to tokenization (models with smaller vocabulary tend to have lower PPL), and PPL doesn't directly measure text quality or usefulness.

---

**Q76.** What is LoRA?

**A:** Low-Rank Adaptation: freeze pretrained weights $\mathbf{W}_0 \in \mathbb{R}^{d\times k}$, add learnable low-rank matrices: $\mathbf{W}_0 + \Delta\mathbf{W} = \mathbf{W}_0 + \mathbf{A}\mathbf{B}$ where $\mathbf{A} \in \mathbb{R}^{d\times r}$, $\mathbf{B} \in \mathbb{R}^{r\times k}$, $r \ll \min(d,k)$. Trainable parameters: $r(d+k)$ vs $dk$ for full fine-tuning. Ratio: $2r/(d+k)$. For $d=k=4096$, $r=8$: $4 \times 10^{-3}$ = 0.4% of original. Initialization: $\mathbf{A} \sim \mathcal{N}(0, \sigma^2)$, $\mathbf{B} = 0$ so $\Delta\mathbf{W}=0$ at start.

---

**Q77.** What is the difference between pre-training and fine-tuning?

**A:** **Pre-training:** Train on massive unlabeled corpus with self-supervised objective (next-token prediction, masked LM) to learn general representations. Expensive — billions of dollars. **Fine-tuning:** Continue training pre-trained model on smaller task-specific labeled dataset. Fast and cheap. **Why it works:** Pre-training learns transferable features (syntax, semantics, world knowledge). Fine-tuning adapts these features to the specific task. Full fine-tuning: update all weights. PEFT (LoRA, adapters, prefix tuning): update small fraction of weights.

---

**Q78.** What is the attention mask and why is it needed?

**A:** A binary matrix that prevents tokens from attending to certain positions. Uses: (1) **Causal (autoregressive) mask:** each token can only attend to previous tokens (upper triangular mask). Prevents the model from "cheating" by looking at future tokens during training. (2) **Padding mask:** in batched inference, sequences have different lengths. Pad shorter sequences and mask padding tokens so they don't contribute to attention. (3) **Prefix LM mask:** prefix tokens attend bidirectionally, generation tokens attend causally.

---

**Q79.** What are the components of the Adam optimizer?

**A:** $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$ (first moment — moving average of gradient), $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$ (second moment — moving average of squared gradient). Bias correction: $\hat{m}_t = m_t/(1-\beta_1^t)$, $\hat{v}_t = v_t/(1-\beta_2^t)$. Update: $\theta_t = \theta_{t-1} - \eta\hat{m}_t/(\sqrt{\hat{v}_t}+\varepsilon)$. Typical hyperparameters: $\beta_1=0.9$, $\beta_2=0.999$, $\varepsilon=10^{-8}$, $\eta=3\times10^{-4}$.

---

**Q80.** What is the difference between encoder-only, decoder-only, and encoder-decoder models?

**A:** **Encoder-only (BERT):** Bidirectional attention, masked LM pre-training. Captures context from both directions. Best for: classification, NER, sentence embeddings. **Decoder-only (GPT):** Causal attention, next-token prediction. Best for: text generation, in-context learning, instruction following. **Encoder-decoder (T5, BART):** Encoder processes input bidirectionally; decoder generates output causally attending to encoder via cross-attention. Best for: translation, summarization, seq2seq tasks. Trend: decoder-only has come to dominate due to versatility via prompting.

---

## Practical ML (Questions 81–100)

**Q81.** What is data leakage and how do you prevent it?

**A:** Leakage occurs when information from the test set (or future) enters the training process. Types: (1) Target leakage — features that include information about the target after the fact; (2) Train-test contamination — test data used in preprocessing (e.g., fitting scaler on all data). Prevention: strict temporal splits for time-series; fit all preprocessing transforms on train only; feature engineering audit.

---

**Q82.** What is the curse of dimensionality?

**A:** In high dimensions: (1) Volume concentrates on the hypersphere surface (most volume is near the boundary); (2) Nearest neighbor distances become meaningless (all distances converge to the same value); (3) Number of samples needed to cover the space grows exponentially with dimension; (4) Data is sparse in any fixed direction. Mitigations: dimensionality reduction (PCA, UMAP), regularization, feature selection, deep learning (which learns the relevant low-dimensional manifold implicitly).

---

**Q83.** What is cross-validation and when should you use k-fold vs leave-one-out?

**A:** k-fold CV: split data into $k$ equal folds, train on $k-1$, test on remaining, average. k=5 or 10 is standard. **Leave-one-out (LOOCV):** $k=n$. Nearly unbiased estimate of generalization error but high variance and computationally expensive. Use when: $n$ is small (< 100) and you need every sample for training. k=10 is typically the best tradeoff. Time series: use time-series CV (train on past, validate on future — never shuffle).

---

**Q84.** What is the difference between online and batch learning?

**A:** **Batch learning:** Train on entire dataset at once; retrain from scratch for new data. Stable, reproducible. **Online learning:** Update model incrementally as each new example arrives. Adapts to distribution shift. Algorithms: SGD (online variant), Kalman filter (for linear models), follow-the-regularized-leader. Use online learning when: data arrives in a stream, distribution shifts over time, dataset is too large to fit in memory.

---

**Q85.** What is feature importance and how do you compute it for different models?

**A:** **Linear models:** coefficient magnitude (after standardizing features). **Tree-based:** impurity decrease weighted by number of samples at each node that splits on this feature. **Permutation importance:** shuffle feature values randomly; decrease in performance = feature importance. **SHAP (Shapley values):** game-theoretic attribution; each feature's contribution is its average marginal contribution across all subsets. Model-agnostic. Most reliable but expensive.

---

**Q86.** What is distribution shift and how do you detect it?

**A:** Training and deployment data have different distributions. Types: (1) **Covariate shift:** $P_\text{train}(X) \neq P_\text{test}(X)$, but $P(Y|X)$ is the same; (2) **Label shift:** $P_\text{train}(Y) \neq P_\text{test}(Y)$; (3) **Concept drift:** $P(Y|X)$ changes over time. Detection: monitor input feature statistics (PSI, KL divergence), prediction distribution, actual outcomes. Fix: importance weighting (correct for covariate shift), retrain on recent data, domain adaptation.

---

**Q87.** What is the difference between accuracy and F1 score? When to use each?

**A:** **Accuracy:** (TP+TN)/(TP+TN+FP+FN). Misleading when classes are imbalanced — a model predicting always "negative" gets high accuracy if 99% of data is negative. **F1:** $2\text{PR}/(\text{P}+\text{R}) = 2\text{TP}/(2\text{TP}+\text{FP}+\text{FN})$. Harmonic mean of precision and recall. Better for imbalanced classes. Use accuracy when classes are balanced and false positives and false negatives are equally costly. Use F1 (or PR-AUC) for imbalanced classification or when one error type is costlier.

---

**Q88.** What is the AUC-ROC score?

**A:** Area Under the Receiver Operating Characteristic curve. ROC plots TPR (recall) vs FPR at all classification thresholds. AUC = probability that the model ranks a random positive example higher than a random negative example. AUC = 0.5: random; AUC = 1.0: perfect. Threshold-independent metric. Limitation: with severe class imbalance, the many negatives can make AUC misleadingly high even if precision is poor. Prefer PR-AUC in that case.

---

**Q89.** What is model calibration?

**A:** A model is well-calibrated if $P(\hat{Y}=1|\hat{P}=p) = p$ for all $p$. Predicted probabilities are reliable. Logistic regression is naturally well-calibrated. Neural networks and SVMs tend to be overconfident. Check with reliability diagrams (calibration curves). Fix: Platt scaling (fit logistic regression on top of raw scores), isotonic regression, temperature scaling (divide logits by scalar $T$ before softmax).

---

**Q90.** What is data augmentation and why does it work?

**A:** Artificially increase training set size by applying label-preserving transformations (flips, crops, rotations for images; synonym replacement for text; spectrogram warping for audio). Works because: (1) Encodes invariances — if the model sees many augmentations of the same image, it learns that horizontal flips don't change the label; (2) Effectively increases training set size; (3) Acts as regularization (prevents memorization of exact training examples). Mixup: $\tilde{x} = \lambda x_i + (1-\lambda)x_j$, $\tilde{y} = \lambda y_i + (1-\lambda)y_j$ — interpolates between examples.

---

**Q91.** What is the vanishing gradient problem in transformers?

**A:** Less severe than in RNNs due to skip connections, but still present. Symptoms: very deep (100+ layer) transformers train poorly without careful initialization. Mitigation: (1) Pre-norm instead of post-norm (stable gradients through skip connection); (2) Scaled initialization (e.g., $\mathbf{W} \sim \mathcal{N}(0, 0.02)$ for GPT); (3) Gradient clipping; (4) Learning rate warmup (avoid large gradients at start). Deep Scaled Init (DeepNet) multiplies the residual branch by $\alpha = (2L)^{-1/4}$ for $L$-layer transformer.

---

**Q92.** What is early stopping and how does it relate to regularization?

**A:** Stop training when validation loss stops decreasing. Prevents overfitting by limiting the number of gradient steps. Theoretical connection: for linear models trained with gradient descent from zero initialization, early stopping is equivalent to L2 regularization — the optimal stopping iteration $T$ corresponds to regularization strength $\lambda \propto 1/T$. Practical: monitor validation loss every epoch, keep the checkpoint with lowest validation loss, stop if no improvement for $k$ epochs (patience).

---

**Q93.** What is the purpose of the softmax temperature in generation?

**A:** Temperature $T$ in $P(k) \propto e^{z_k/T}$: controls the "confidence" of the sampling distribution. Low T → sharper distribution (more confident, less diverse). High T → flatter distribution (more diverse, potentially incoherent). At T=0 (limit), becomes argmax (greedy). At T=∞, becomes uniform. Practical values: T=0.1–0.5 for factual tasks (be confident), T=0.8–1.2 for creative tasks (allow diversity), T=1.0 during training (standard).

---

**Q94.** What is the difference between recall and precision, and when does each matter?

**A:** **Precision:** TP/(TP+FP) — of all positive predictions, how many are correct? **Recall:** TP/(TP+FN) — of all actual positives, how many did we catch? Tradeoff: lower threshold → higher recall, lower precision. **When precision matters:** spam filter (you don't want to delete legitimate emails = false positives are costly). **When recall matters:** cancer screening (you don't want to miss a cancer case = false negatives are costly). PR curve shows the tradeoff across all thresholds.

---

**Q95.** What is a learning curve and what does it tell you?

**A:** Plot training and validation error vs training set size. Diagnostic: (1) **High bias (underfitting):** training and validation error are both high and converge to the same (high) value — more data won't help, need more complex model; (2) **High variance (overfitting):** large gap between training (low) and validation (high) error — more data would help, or need regularization; (3) **Good fit:** training error slightly lower than validation, gap is small, both converge to low values.

---

**Q96.** What is the exploration-exploitation tradeoff?

**A:** In sequential decision-making (bandits, RL): do you exploit the best known option (to maximize immediate reward) or explore to potentially find better options? Pure exploitation: get stuck in local optima. Pure exploration: no progress. Balance via: $\varepsilon$-greedy (explore with probability $\varepsilon$), UCB (add uncertainty bonus: $\hat{\mu}_k + c\sqrt{\ln t/n_k}$), Thompson sampling (sample from posterior over rewards, play best sample). UCB achieves $O(\sqrt{TK\log T})$ regret — near-optimal.

---

**Q97.** What is the difference between L1 and L2 loss?

**A:** **L1 (MAE):** $\sum|y_i - \hat{y}_i|$. Gradient is constant (sign of error) → less sensitive to outliers. Non-differentiable at zero. **L2 (MSE):** $\sum(y_i - \hat{y}_i)^2$. Gradient is proportional to error → larger errors have larger gradients. Sensitive to outliers. Differentiable everywhere. **Huber loss:** L2 for small errors, L1 for large errors — robust to outliers while differentiable. Probabilistic: L2 = Gaussian noise assumption; L1 = Laplace noise assumption.

---

**Q98.** What is transfer learning and why does it work?

**A:** Pre-train on large dataset (ImageNet, web text), fine-tune on small task-specific dataset. Works because: (1) Lower layers learn general features (edges, textures for images; syntax, semantics for text) that transfer across tasks; (2) High-level features need less data to adapt; (3) The pre-trained weights are a strong initialization that's already near good minima. Fine-tuning converges faster and to better solutions than random initialization.

---

**Q99.** What is the neural tangent kernel?

**A:** For an infinitely wide neural network trained with gradient flow (continuous gradient descent), the function update satisfies: $\dot{f}(x) = -\int K_\text{NTK}(x, x') \frac{\partial L}{\partial f(x')} dx'$ where $K_\text{NTK}(x, x') = \nabla_\theta f(x)^T \nabla_\theta f(x')$ is the neural tangent kernel. This NTK is approximately **constant** throughout training (lazy training regime) for infinitely wide networks. Implication: infinite-width NNs are equivalent to kernel regression with the NTK — they can only represent functions in the RKHS of the NTK, which is linear in the training data.

---

**Q100.** What is the fundamental theorem connecting optimization and generalization?

**A:** There isn't a single theorem — but the key ideas are:
(1) **PAC learning:** sample complexity gives conditions under which training loss ≈ test loss.
(2) **Bias-variance decomposition:** total error = Bias² + Variance + Noise.
(3) **Implicit regularization:** gradient descent on overparameterized models converges to the minimum-norm solution (for linear models) or solutions that generalize well (for neural networks, empirically). The "algorithm" matters: SGD has different implicit regularization than second-order methods.
(4) **Double descent:** very large models can generalize well even after perfectly fitting training data — classical generalization theory (which predicts the U-shaped error curve) is not the whole story.

---

*← [Part 14: Advanced AI Math](part-14-advanced-ai-math.md) | [Part 16: Projects →](part-16-projects.md)*
