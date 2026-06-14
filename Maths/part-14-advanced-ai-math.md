# Part 14: Advanced AI Mathematics

> **Prerequisites:** Parts 0–13
> **Status:** Complete
> **Lessons:** 9

---

## The Problem This Part Solves

You now have a comprehensive toolkit. This part covers the *theoretical foundations* that let you reason rigorously about AI systems: when they work, why they fail, what guarantees you can make, and how to think about adversarial settings.

These topics bridge the gap between practitioner and researcher.

---

## Lesson 14.1: PAC Learning Theory

### The Question

When does a machine learning algorithm **reliably generalize**? Under what conditions can you guarantee that a model trained on $n$ examples will perform well on new examples?

PAC (Probably Approximately Correct) learning theory, developed by Leslie Valiant (1984), provides a formal framework.

### Definitions

A **hypothesis class** $\mathcal{H}$ is **PAC learnable** if there exists an algorithm $A$ and polynomial $\text{poly}(1/\varepsilon, 1/\delta, |x|)$ such that: for any $\varepsilon, \delta > 0$ and any distribution $\mathcal{D}$ over labeled examples, with probability at least $1 - \delta$ over the draw of $n \geq \text{poly}(1/\varepsilon, 1/\delta)$ samples, $A$ outputs a hypothesis $h$ with error $\leq \varepsilon$:

$$
P_\mathcal{D}[h(x) \neq y] \leq \varepsilon
$$

"Probably" ($1-\delta$) and "approximately correct" ($\leq \varepsilon$ error) — hence PAC.

### Sample Complexity

For a finite hypothesis class $|\mathcal{H}|$, the sample complexity to achieve $(\varepsilon, \delta)$-PAC learning:

$$
n \geq \frac{1}{\varepsilon}\left(\ln|\mathcal{H}| + \ln\frac{1}{\delta}\right)
$$

The $\ln|\mathcal{H}|$ term is the "complexity" of the class — how many bits are needed to specify a hypothesis.

**Example:** Binary classifiers over $d$-dimensional binary features. $|\mathcal{H}| = 2^{2^d}$ for all classifiers, $\ln|\mathcal{H}| = 2^d\ln 2$. Not PAC learnable efficiently for general classifiers. But for linear classifiers: $\ln|\mathcal{H}| \approx O(d\log d)$, requiring $O(d\log d / \varepsilon)$ samples.

### Agnostic PAC Learning

The original PAC setting assumes the true concept is in $\mathcal{H}$. **Agnostic** PAC learning relaxes this: even if no $h \in \mathcal{H}$ fits perfectly, the algorithm outputs the best hypothesis in $\mathcal{H}$.

Sample complexity for agnostic PAC:

$$
n \geq O\left(\frac{d_{VC}\log(1/\varepsilon) + \log(1/\delta)}{\varepsilon^2}\right)
$$

The $1/\varepsilon^2$ (vs $1/\varepsilon$ in realizable case) reflects the harder task of approximating the best hypothesis.

---

## Lesson 14.2: Online Learning and Regret

### The Online Setting

Machine learning without a fixed training set. At each round $t$:
1. Receive example $\mathbf{x}_t$
2. Make prediction $\hat{y}_t$
3. Observe true label $y_t$, incur loss $\ell(\hat{y}_t, y_t)$
4. Update model

No distributional assumptions. The adversary can be adaptive.

### Regret

The **regret** of an online algorithm over $T$ rounds:

$$
R_T = \sum_{t=1}^{T}\ell(\hat{y}_t, y_t) - \min_{h \in \mathcal{H}}\sum_{t=1}^{T}\ell(h(x_t), y_t)
$$

How much worse are we than the best fixed hypothesis in hindsight?

**Goal:** Achieve *sublinear regret* $R_T = o(T)$, meaning the per-round regret $R_T/T \to 0$.

### Online Gradient Descent

For convex loss functions, **Online Gradient Descent** (OGD) achieves:

$$
R_T \leq O(\sqrt{T})
$$

This is optimal for general convex losses. The per-round regret $O(1/\sqrt{T}) \to 0$.

OGD update: $\theta_{t+1} = \Pi_\mathcal{C}(\theta_t - \eta_t \nabla_t)$ where $\Pi_\mathcal{C}$ is projection onto the constraint set.

### EXP3 — Exploration-Exploitation Tradeoff

For the multi-armed bandit problem (adversarial): $K$ arms, only observe the reward of the chosen arm (bandit feedback), need to balance exploration and exploitation.

**EXP3 (Exponential Weights for Exploration and Exploitation):**

$$
P_t(i) = (1-\gamma)\frac{w_i^t}{\sum_j w_j^t} + \frac{\gamma}{K}
$$

$$
w_i^{t+1} = w_i^t \cdot \exp\left(\frac{\gamma \cdot \hat{r}_i^t}{K}\right)
$$

where $\hat{r}_i^t = r_i^t/P_t(i)$ is the importance-weighted reward (0 if arm $i$ not selected).

Regret: $R_T \leq O(\sqrt{TK\ln K})$ — scales with $\sqrt{K}$, not $K$ linearly.

---

## Lesson 14.3: Game Theory and Nash Equilibria

### Two-Player Zero-Sum Games

Player 1 chooses row $i$, Player 2 chooses column $j$. Payoff matrix $\mathbf{A} \in \mathbb{R}^{m\times n}$: Player 1 receives $A_{ij}$, Player 2 receives $-A_{ij}$.

A **mixed strategy** for Player 1: probability distribution $\mathbf{p}$ over rows. For Player 2: distribution $\mathbf{q}$ over columns.

Expected payoff: $\mathbf{p}^T\mathbf{A}\mathbf{q}$.

### Minimax Theorem (von Neumann, 1928)

For any finite two-player zero-sum game:

$$
\max_\mathbf{p}\min_\mathbf{q} \mathbf{p}^T\mathbf{A}\mathbf{q} = \min_\mathbf{q}\max_\mathbf{p} \mathbf{p}^T\mathbf{A}\mathbf{q}
$$

Both sides equal the **value of the game** $v^*$. Player 1 can guarantee $\geq v^*$ regardless of Player 2's strategy; Player 2 can guarantee $\leq v^*$ regardless of Player 1.

The optimal strategies $(\mathbf{p}^*, \mathbf{q}^*)$ form a **Nash equilibrium** — neither player can unilaterally improve.

### GAN Connection

GANs formulate the generator-discriminator interaction as:

$$
\min_G\max_D V(D, G) = \mathbb{E}_\mathbf{x}[\log D(\mathbf{x})] + \mathbb{E}_\mathbf{z}[\log(1 - D(G(\mathbf{z})))]
$$

This is a continuous two-player zero-sum game. The Nash equilibrium is $D^*(\mathbf{x}) = 1/2$ (discriminator can't tell real from fake) and $G^*$ generating the true data distribution.

**Training instability** occurs because the minimax theorem requires convex-concave structure; GAN objectives are neither.

### Nash Equilibrium in General Games

For general (non-zero-sum) $n$-player games, Nash equilibrium: a strategy profile $(\sigma_1^*, \ldots, \sigma_n^*)$ where no player can improve their payoff by unilaterally deviating.

**Existence (Nash, 1950):** Every finite game has at least one Nash equilibrium in mixed strategies. Proven using Kakutani's fixed-point theorem.

**Computational hardness:** Finding a Nash equilibrium in general games is PPAD-complete. This is why multi-agent RL is fundamentally harder than single-agent RL.

---

## Lesson 14.4: Adversarial Robustness

### Adversarial Examples

A neural network $f$ is vulnerable if there exists a small perturbation $\boldsymbol{\delta}$ such that:

$$
\|\boldsymbol{\delta}\|_p \leq \varepsilon \quad \text{but} \quad f(\mathbf{x} + \boldsymbol{\delta}) \neq f(\mathbf{x})
$$

These exist because neural networks have large Lipschitz constants — they can change rapidly in directions imperceptible to humans.

### FGSM Attack

**Fast Gradient Sign Method** (Goodfellow et al., 2015):

$$
\boldsymbol{\delta} = \varepsilon \cdot \text{sign}(\nabla_\mathbf{x} J(\mathbf{x}, y))
$$

One-step attack. Move in the direction that maximally increases the loss, bounded by $\varepsilon$ in $L_\infty$ norm.

### PGD Attack

**Projected Gradient Descent** (Madry et al., 2018): $K$ steps of FGSM with projection:

$$
\boldsymbol{\delta}_{t+1} = \Pi_{\|\boldsymbol{\delta}\|_p \leq \varepsilon}\left(\boldsymbol{\delta}_t + \alpha\cdot\text{sign}(\nabla_\mathbf{x} J(\mathbf{x} + \boldsymbol{\delta}_t, y))\right)
$$

Strong attack — finds near-optimal adversarial examples.

### Adversarial Training

Defend by training on adversarial examples:

$$
\min_\theta \mathbb{E}_{(\mathbf{x},y)}\left[\max_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} J(\mathbf{x} + \boldsymbol{\delta}, y; \theta)\right]
$$

This minimax objective is the robust training objective. Inner maximization: find the worst-case perturbation. Outer minimization: update weights to minimize loss under worst case.

**Certified robustness:** Use **interval bound propagation** or **randomized smoothing** to *prove* that no perturbation within a ball can change the prediction, rather than just empirically testing.

---

## Lesson 14.5: Optimal Transport

### The Wasserstein Distance

Given two distributions $p$ and $q$ over $\mathcal{X}$, the Wasserstein-1 distance:

$$
W_1(p, q) = \inf_{\gamma \in \Pi(p,q)} \int_{\mathcal{X}\times\mathcal{X}} d(\mathbf{x}, \mathbf{y}) \,d\gamma(\mathbf{x}, \mathbf{y})
$$

where $\Pi(p,q)$ is the set of all joint distributions with marginals $p$ and $q$.

Intuitively: the minimum "work" to transform distribution $p$ into distribution $q$, where work = mass moved × distance moved. Also called the **Earth Mover's Distance**.

**Kantorovich-Rubinstein duality:**

$$
W_1(p, q) = \sup_{\|f\|_L \leq 1} \left(\mathbb{E}_{x\sim p}[f(x)] - \mathbb{E}_{x\sim q}[f(x)]\right)
$$

The dual form is how Wasserstein GANs implement the distance — learn a 1-Lipschitz critic $f$.

### Why Wasserstein over KL/JS?

**KL divergence:** $KL(p\|q) = \infty$ if $p$ and $q$ have disjoint support (different non-zero regions). Very common in generation: generated distribution and real distribution may not overlap initially.

**Wasserstein distance:** Always finite and meaningful as long as $d(\mathbf{x}, \mathbf{y}) < \infty$. Provides gradient signal even when distributions are disjoint.

### Sinkhorn Distance (Regularized OT)

Entropy-regularized optimal transport:

$$
W_\varepsilon(p, q) = \min_{\gamma \in \Pi(p,q)} \langle \mathbf{C}, \gamma\rangle - \varepsilon H(\gamma)
$$

where $H(\gamma) = -\sum_{ij}\gamma_{ij}\log\gamma_{ij}$ is entropy of the transport plan.

Solved efficiently by the **Sinkhorn algorithm** (matrix scaling):

```
K_ij = exp(-C_ij / epsilon)
Initialize u = 1/n
Repeat:
    v = b / (K^T u)
    u = a / (K v)
Until convergence
Optimal transport: T_ij = u_i K_ij v_j
```

$O(n^2/\varepsilon^2)$ iterations, far faster than exact OT ($O(n^3 \log n)$).

---

## Lesson 14.6: Differential Privacy

### The Privacy Problem

Training on sensitive data (medical records, private messages) risks memorization. If the model "remembers" specific training examples, adversaries can extract them.

### Formal Definition

A randomized algorithm $\mathcal{M}: \mathcal{D} \to \mathcal{R}$ is $(\varepsilon, \delta)$-**differentially private** if for all datasets $D, D'$ differing in one record, and all measurable subsets $S \subseteq \mathcal{R}$:

$$
P[\mathcal{M}(D) \in S] \leq e^\varepsilon P[\mathcal{M}(D') \in S] + \delta
$$

Adding or removing any single person's data changes the output distribution by at most a factor of $e^\varepsilon$ (plus $\delta$ additive slack).

$\varepsilon$ is the **privacy budget**: small $\varepsilon$ = strong privacy = more noise.

### DP-SGD

**Differentially Private SGD** (Abadi et al., 2016):

1. For each sample in mini-batch, compute per-sample gradient $\mathbf{g}_i$
2. **Clip:** $\tilde{\mathbf{g}}_i = \mathbf{g}_i / \max(1, \|\mathbf{g}_i\|_2 / C)$ (clip to $\ell_2$ norm $C$)
3. **Add Gaussian noise:** $\hat{\mathbf{g}} = \frac{1}{B}\left(\sum_i \tilde{\mathbf{g}}_i + \mathcal{N}(\mathbf{0}, \sigma^2 C^2 \mathbf{I})\right)$
4. Update parameters with $\hat{\mathbf{g}}$

Privacy budget: $\varepsilon = O\left(\frac{T}{\sigma^2 B^2/n^2}\right)$ via moments accountant.

**Privacy-utility tradeoff:** More noise = stronger privacy = lower accuracy. DP-SGD degrades accuracy, especially for small datasets. Large language models can be trained with DP at moderate utility cost.

---

## Lesson 14.7: Causal Inference — Deep Dive

### The Causal Hierarchy (Pearl's Ladder)

| Level | Query | Example |
|-------|-------|---------|
| Association ($P(y\|x)$) | Seeing | What is the probability a patient who took drug X recovered? |
| Intervention ($P(y\|do(x))$) | Doing | What would happen if we gave all patients drug X? |
| Counterfactual ($P(y_x\|x', y')$) | Imagining | Would this patient have recovered if they had taken drug X? |

Observational data answers L1 only. Experiments (RCTs) answer L2. Structural causal models answer L3.

### Do-Calculus

Pearl's **do-calculus** provides rules for computing interventional distributions from observational data under assumptions encoded in a **Directed Acyclic Graph (DAG)**:

**Rule 1 (insertion/deletion of observations):**

$$
P(y|do(x), z, w) = P(y|do(x), w) \text{ if } (Z \perp Y | X, W)_{\mathcal{G}_{\bar{X}}}
$$

**Rule 2 (action/observation exchange):**

$$
P(y|do(x), do(z), w) = P(y|do(x), z, w) \text{ if } (Z \perp Y | X, W)_{\mathcal{G}_{\bar{X}\underline{Z}}}
$$

**Rule 3 (insertion/deletion of actions):**

$$
P(y|do(x), do(z), w) = P(y|do(x), w) \text{ if } (Z \perp Y | X, W)_{\mathcal{G}_{\bar{X}\bar{Z}(W)}}
$$

where $\mathcal{G}_{\bar{X}}$ denotes the graph with all arrows into $X$ removed.

### The Backdoor Criterion

A set $Z$ of variables satisfies the **backdoor criterion** relative to $(X, Y)$ if:
1. No node in $Z$ is a descendant of $X$
2. $Z$ blocks all backdoor paths from $X$ to $Y$

If $Z$ satisfies the backdoor criterion:

$$
P(y | do(x)) = \sum_z P(y | x, z) P(z)
$$

This is the **backdoor adjustment** — it removes confounding by adjusting for $Z$.

**Example:** $X$ = smoking, $Y$ = cancer, $Z$ = confounders (genetics, socioeconomic status). If $Z$ blocks all backdoor paths, the interventional distribution (effect of forcing everyone to smoke) is computed by averaging the observational conditional over $Z$.

---

## Lesson 14.8: Algorithmic Fairness

### Definitions

**Demographic parity:** $P(\hat{Y}=1|A=0) = P(\hat{Y}=1|A=1)$. Equal positive prediction rates across groups.

**Equalized odds (Hardt et al., 2016):** Equal TPR and FPR across groups:
$$P(\hat{Y}=1|Y=1,A=0) = P(\hat{Y}=1|Y=1,A=1)$$
$$P(\hat{Y}=1|Y=0,A=0) = P(\hat{Y}=1|Y=0,A=1)$$

**Calibration:** $P(Y=1|\hat{P}=p, A=a) = p$ for all $a$. Predicted probabilities are accurate within each group.

### The Impossibility Theorem

**Chouldechova (2017):** For any predictor with unequal base rates ($P(Y=1|A=0) \neq P(Y=1|A=1)$), it is **impossible** to simultaneously satisfy:
1. Calibration
2. Equal PPV (positive predictive value)
3. Equal FPR and FNR

You cannot have all three fairness criteria at once when base rates differ. Every fairness metric involves a tradeoff.

### Causal Fairness

Individual fairness (Dwork et al.): similar individuals should be treated similarly. $d_f(\hat{Y}(x), \hat{Y}(x')) \leq d_i(x, x')$ for individual distance metrics.

Counterfactual fairness (Kusner et al., 2017): prediction is counterfactually fair if $P(\hat{Y}_{A\leftarrow a}(U) = y | X=x, A=a) = P(\hat{Y}_{A\leftarrow a'}(U) = y | X=x, A=a)$ for all $y$ and $a, a'$. Would the prediction change if the sensitive attribute were different?

---

## Lesson 14.9: Computational Complexity for AI Engineers

### Complexity Classes

| Class | Meaning | Example Problems |
|-------|---------|------------------|
| P | Polynomial time | Sorting, shortest path, linear programming |
| NP | Verifiable in polynomial time | Satisfiability, graph coloring |
| NP-hard | At least as hard as any NP problem | Finding max cut, training optimal decision tree |
| PPAD | Polynomial Parity Arguments on Directed graphs | Nash equilibrium |

**P vs NP:** If P = NP, then any problem whose solution can be verified quickly can also be solved quickly. Most believe P ≠ NP.

### NP-Hardness in ML

- **Optimal decision tree:** NP-hard (Hyafil & Rivest, 1976)
- **Training a 2-layer neural network:** NP-hard in the worst case
- **Feature selection (optimal subset):** NP-hard
- **SVM with hidden units:** NP-hard

In practice, we use approximations and heuristics that work well despite worst-case hardness.

### Approximation Algorithms

An **$\alpha$-approximation algorithm** is guaranteed to find a solution within factor $\alpha$ of optimal in polynomial time.

**Example:** Maximum cut in a graph (partition vertices to maximize edges between parts). Optimal is NP-hard. A random partition achieves 1/2-approximation trivially. The Goemans-Williamson SDP rounding gives a 0.878-approximation.

**AI connection:** Approximate inference in graphical models, approximate nearest neighbor search, and variational methods all trade optimality for tractability.

---

| Concept | Key Result | Implication |
|---------|-----------|-------------|
| PAC learning | Sample complexity $\sim \ln\|\mathcal{H}\|/\varepsilon$ | How many examples you need to generalize |
| Online regret | OGD achieves $O(\sqrt{T})$ regret | Learning without distributional assumptions |
| Minimax theorem | $\max\min = \min\max$ for zero-sum games | GAN equilibrium, game theory |
| Wasserstein distance | Earth mover's distance | Better GAN training, distribution comparison |
| DP-SGD | Clip + noise + moments accountant | Train on private data with guarantees |
| Backdoor criterion | $P(y \mid \text{do}(x)) = \sum_z P(y \mid x,z)P(z)$ | Compute causal effects from observations |
| Fairness impossibility | Cannot satisfy all metrics when base rates differ | Every fairness definition has tradeoffs |
---

*← [Part 13: Numerical Methods](part-13-numerical-methods.md) | [Part 15: Interview Prep →](part-15-interview-prep.md)*
