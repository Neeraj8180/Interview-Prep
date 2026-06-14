# Part 10: Deep Learning Mathematics

> **Prerequisites:** Parts 0–9
> **Status:** Complete
> **Lessons:** 16

---

## The Problem This Part Solves

You understand the building blocks: linear algebra, calculus, optimization, probability. You understand classical ML. Now the question is: **what makes deep learning different?**

Deep learning is not magic. It is function composition + gradient descent. But the specific choices — architectures, activations, normalization, initialization — each have mathematical reasons. Understanding those reasons lets you:
- Debug training failures from first principles
- Design architectures intentionally instead of copying
- Know when to use which component and why

---

## Lesson 10.1: Universal Approximation Theorem

### The Core Claim

A feedforward neural network with a single hidden layer and a non-polynomial activation function can approximate any continuous function on a compact subset of $\mathbb{R}^d$ to arbitrary precision, given enough hidden units.

Formally (Cybenko 1989, Hornik 1991): for any $\varepsilon > 0$ and any continuous $f: [0,1]^d \to \mathbb{R}$, there exist $N$, weights $\mathbf{w}_j$, biases $b_j$, and output weights $\alpha_j$ such that:

$$
\sup_{\mathbf{x} \in [0,1]^d} \left| f(\mathbf{x}) - \sum_{j=1}^{N}\alpha_j\sigma(\mathbf{w}_j^T\mathbf{x} + b_j) \right| < \varepsilon
$$

### What This Does NOT Say

- It does not say *how many* neurons are needed (could be exponential in $d$)
- It does not say gradient descent will *find* the approximating weights
- It does not say *depth* is unnecessary — deeper networks can approximate functions with exponentially fewer neurons than shallow ones

### Depth Efficiency

Functions expressible by depth-$k$ networks with polynomial width may require *exponential* width with depth $k-1$. This is why depth matters in practice: it provides exponential representational efficiency for certain function families (particularly hierarchical functions, like those describing natural images or language).

---

## Lesson 10.2: MLP Forward Pass and Full Backpropagation Derivation

### The Forward Pass

An $L$-layer MLP computes:

$$
\mathbf{a}^{(0)} = \mathbf{x}
$$

$$
\mathbf{z}^{(\ell)} = \mathbf{W}^{(\ell)}\mathbf{a}^{(\ell-1)} + \mathbf{b}^{(\ell)}
$$

$$
\mathbf{a}^{(\ell)} = \sigma(\mathbf{z}^{(\ell)})
$$

for $\ell = 1, \ldots, L$. The output is $\hat{\mathbf{y}} = \mathbf{a}^{(L)}$.

### Backpropagation: The Four Equations

Define the **error** at layer $\ell$: $\boldsymbol{\delta}^{(\ell)} = \frac{\partial J}{\partial \mathbf{z}^{(\ell)}}$.

**BP1: Error at output layer:**

$$
\boldsymbol{\delta}^{(L)} = \nabla_{\mathbf{a}^{(L)}} J \odot \sigma'(\mathbf{z}^{(L)})
$$

**BP2: Error backpropagation:**

$$
\boldsymbol{\delta}^{(\ell)} = \left((\mathbf{W}^{(\ell+1)})^T\boldsymbol{\delta}^{(\ell+1)}\right) \odot \sigma'(\mathbf{z}^{(\ell)})
$$

**BP3: Gradient for biases:**

$$
\frac{\partial J}{\partial \mathbf{b}^{(\ell)}} = \boldsymbol{\delta}^{(\ell)}
$$

**BP4: Gradient for weights:**

$$
\frac{\partial J}{\partial \mathbf{W}^{(\ell)}} = \boldsymbol{\delta}^{(\ell)} (\mathbf{a}^{(\ell-1)})^T
$$

$\odot$ is element-wise (Hadamard) product.

### Why Backprop is Efficient

Naive computation of $\frac{\partial J}{\partial \mathbf{W}^{(\ell)}}$ via finite differences costs $O(|\mathbf{W}|)$ forward passes. Backprop computes all gradients in $O(1)$ forward passes + $O(1)$ backward passes. This is the chain rule applied with dynamic programming (memoization of intermediate quantities).

---

## Lesson 10.3: Activation Functions — The Nonlinearity That Makes Depth Useful

### Why Nonlinearity?

Without nonlinear activations, any composition of linear layers collapses to a single linear layer: $\mathbf{W}^{(L)}\cdots\mathbf{W}^{(1)}\mathbf{x} = \mathbf{W}\mathbf{x}$. Depth without nonlinearity = zero benefit.

### Sigmoid

$$
\sigma(z) = \frac{1}{1+e^{-z}}, \quad \sigma'(z) = \sigma(z)(1-\sigma(z)) \in (0, 0.25]
$$

**Vanishing gradient problem:** $|\sigma'(z)| \leq 0.25$. After $L$ layers, the gradient shrinks by at most $(0.25)^L$. For $L=10$, this is $\approx 10^{-6}$ — negligibly small. Weights in early layers barely update.

### ReLU

$$
\text{ReLU}(z) = \max(0, z), \quad \text{ReLU}'(z) = \mathbf{1}[z > 0]
$$

**Advantages:**
- Gradient is either 0 or 1 — no vanishing in the active region
- Computationally cheap
- Induces sparsity (many units output 0)

**Dying ReLU problem:** If $z < 0$ for all training examples, the gradient is 0 and the neuron never updates. Fixed by Leaky ReLU: $\max(0.01z, z)$.

### GELU

$$
\text{GELU}(z) = z \cdot \Phi(z)
$$

where $\Phi$ is the standard normal CDF. GELU ≈ $z\sigma(1.702z)$ (fast approximation). Used in BERT, GPT. Smoother than ReLU, empirically better for transformers.

### Softmax

$$
\text{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_j e^{z_j}}
$$

**Numerical stability:** Subtract $\max_j z_j$ before exponentiation:

$$
\text{softmax}(\mathbf{z})_k = \frac{e^{z_k - z_\text{max}}}{\sum_j e^{z_j - z_\text{max}}}
$$

This prevents overflow without changing the output (numerator and denominator divided by $e^{z_\text{max}}$).

---

## Lesson 10.4: Convolutional Networks — Convolution as Matrix Multiplication

### The Convolution Operation

A 2D convolution of input $\mathbf{X} \in \mathbb{R}^{H\times W}$ with kernel $\mathbf{K} \in \mathbb{R}^{k\times k}$:

$$
(\mathbf{X} * \mathbf{K})_{i,j} = \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} \mathbf{X}_{i+m, j+n} \cdot \mathbf{K}_{m,n}
$$

Output size: $\left(\frac{H - k + 2p}{s} + 1\right) \times \left(\frac{W - k + 2p}{s} + 1\right)$ where $p$ = padding and $s$ = stride.

### Convolution = Structured Weight Sharing

A fully connected layer has $H \times W \times H' \times W'$ weights. A convolutional layer has $k^2$ weights, shared across all spatial positions. This is a dramatic reduction in parameters, encoding the **translation equivariance** inductive bias.

**As matrix multiply:** Unroll input patches into rows of a matrix (im2col), then multiply by the flattened kernel. This turns convolution into a matrix multiplication, enabling GPU acceleration.

### Receptive Field

The receptive field of a neuron in layer $\ell$ is the region of the input that can influence it. For $k \times k$ convolutions stacked $L$ layers deep with stride 1:

$$
\text{RF} = 1 + L(k-1)
$$

Deeper networks have larger receptive fields without increasing kernel size. Dilated convolutions expand RF exponentially: RF grows as $O(r^L)$ where $r$ is the dilation rate.

---

## Lesson 10.5: Recurrent Networks and BPTT

### The RNN Equations

$$
\mathbf{h}_t = \sigma(\mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{W}_{xh}\mathbf{x}_t + \mathbf{b}_h)
$$

$$
\hat{\mathbf{y}}_t = \mathbf{W}_{hy}\mathbf{h}_t + \mathbf{b}_y
$$

The same weight matrices $\mathbf{W}_{hh}$ and $\mathbf{W}_{xh}$ are reused at every time step — this is the parameter sharing across time.

### Backpropagation Through Time (BPTT)

The gradient of loss $J = \sum_t J_t$ with respect to $\mathbf{W}_{hh}$ involves summing over all time steps:

$$
\frac{\partial J}{\partial \mathbf{W}_{hh}} = \sum_{t=1}^{T}\frac{\partial J_t}{\partial \mathbf{W}_{hh}} = \sum_{t=1}^{T}\sum_{k=1}^{t}\frac{\partial J_t}{\partial \mathbf{h}_t}\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k}\frac{\partial \mathbf{h}_k}{\partial \mathbf{W}_{hh}}
$$

The chain product $\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} = \prod_{\tau=k+1}^{t}\frac{\partial \mathbf{h}_\tau}{\partial \mathbf{h}_{\tau-1}} = \prod_{\tau=k+1}^{t}\mathbf{W}_{hh}^T\text{diag}(\sigma'(\mathbf{z}_\tau))$.

**Vanishing gradients:** If the largest eigenvalue of $\mathbf{W}_{hh}$ is $< 1$, the product of $t-k$ matrices shrinks exponentially. Gradients from far back in time vanish. Long-range dependencies cannot be learned.

**Exploding gradients:** If the largest eigenvalue is $> 1$, the product grows exponentially. Gradients overflow. Fix: **gradient clipping** $\nabla \leftarrow \nabla \cdot \min\left(1, \frac{c}{\|\nabla\|}\right)$ where $c$ is the clip threshold.

---

## Lesson 10.6: LSTM — Solving Vanishing Gradients

### The LSTM Equations

$$
\mathbf{f}_t = \sigma(\mathbf{W}_f[\mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_f) \quad \text{(forget gate)}
$$

$$
\mathbf{i}_t = \sigma(\mathbf{W}_i[\mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_i) \quad \text{(input gate)}
$$

$$
\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c[\mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_c) \quad \text{(candidate memory)}
$$

$$
\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \quad \text{(cell state update)}
$$

$$
\mathbf{o}_t = \sigma(\mathbf{W}_o[\mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_o) \quad \text{(output gate)}
$$

$$
\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t) \quad \text{(hidden state)}
$$

### Why LSTMs Solve Vanishing Gradients

The **cell state** $\mathbf{c}_t$ has an additive update: $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$.

The gradient through the cell state path:

$$
\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} = \mathbf{f}_t
$$

The forget gate $\mathbf{f}_t \in (0,1)$. If $\mathbf{f}_t \approx \mathbf{1}$, the gradient flows through time without shrinking. The additive structure (not multiplicative like vanilla RNN) allows gradient highways.

**ResNets apply the same principle** to depth: $\mathbf{a}^{(\ell)} = \mathbf{a}^{(\ell-1)} + F(\mathbf{a}^{(\ell-1)})$ so $\frac{\partial \mathbf{a}^{(\ell)}}{\partial \mathbf{a}^{(\ell-1)}} = \mathbf{I} + \frac{\partial F}{\partial \mathbf{a}^{(\ell-1)}}$. The identity term prevents vanishing.

---

## Lesson 10.7: Attention Mechanism — Full Mathematical Derivation

### The Motivation

RNNs encode the entire sequence into a fixed-size vector $\mathbf{h}_T$. For long sequences, early information gets diluted. **Attention** allows the decoder to directly access all encoder states, weighting them by relevance.

### Scaled Dot-Product Attention

Given queries $\mathbf{Q} \in \mathbb{R}^{n \times d_k}$, keys $\mathbf{K} \in \mathbb{R}^{m \times d_k}$, values $\mathbf{V} \in \mathbb{R}^{m \times d_v}$:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}
$$

**Why divide by $\sqrt{d_k}$?** The dot product $\mathbf{q}^T\mathbf{k}$ has variance $d_k$ when $\mathbf{q}$ and $\mathbf{k}$ have unit variance components. Dividing by $\sqrt{d_k}$ brings variance back to 1, preventing the softmax from saturating (which would zero out gradients).

**Step by step:**
1. $\mathbf{S} = \mathbf{Q}\mathbf{K}^T / \sqrt{d_k}$: similarity matrix, shape $n \times m$
2. $\mathbf{A} = \text{softmax}(\mathbf{S})$: attention weights (each row sums to 1), shape $n \times m$
3. Output $= \mathbf{A}\mathbf{V}$: weighted sum of values, shape $n \times d_v$

### Multi-Head Attention

$$
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{concat}(\text{head}_1, \ldots, \text{head}_h)\mathbf{W}^O
$$

$$
\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q, \mathbf{K}\mathbf{W}_i^K, \mathbf{V}\mathbf{W}_i^V)
$$

where $\mathbf{W}_i^Q \in \mathbb{R}^{d_\text{model} \times d_k}$, $\mathbf{W}_i^K \in \mathbb{R}^{d_\text{model} \times d_k}$, $\mathbf{W}_i^V \in \mathbb{R}^{d_\text{model} \times d_v}$, $\mathbf{W}^O \in \mathbb{R}^{hd_v \times d_\text{model}}$.

Each head attends to different aspects of the sequence. With $h$ heads and $d_\text{model} = 512$, typically $d_k = d_v = d_\text{model}/h = 64$.

**Parameter count for multi-head attention:** $4d_\text{model}^2$ (four weight matrices: $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V, \mathbf{W}^O$, each $d_\text{model} \times d_\text{model}$).

---

## Lesson 10.8: Transformer Architecture

### Full Transformer Block

Each Transformer block applies:

$$
\mathbf{X}' = \text{LayerNorm}(\mathbf{X} + \text{MultiHead}(\mathbf{X}, \mathbf{X}, \mathbf{X}))
$$

$$
\mathbf{X}'' = \text{LayerNorm}(\mathbf{X}' + \text{FFN}(\mathbf{X}'))
$$

The FFN:

$$
\text{FFN}(\mathbf{x}) = \mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2
$$

where $\mathbf{W}_1 \in \mathbb{R}^{4d \times d}$ and $\mathbf{W}_2 \in \mathbb{R}^{d \times 4d}$. The FFN expands to $4d$ dimension in the middle.

**Parameter count per block:** $8d^2$ for attention + $8d^2$ for FFN = $16d^2$. For $d=768$ (BERT-base): $16 \times 768^2 \approx 9.4$M per block, 12 blocks = 113M total.

### Self-Attention Complexity

For sequence length $n$ and model dimension $d$:
- **Time:** $O(n^2 d)$ — quadratic in sequence length (the $\mathbf{Q}\mathbf{K}^T$ product)
- **Memory:** $O(n^2)$ — the attention matrix must be stored

This quadratic bottleneck motivates efficient attention variants (Linformer, Longformer, FlashAttention, etc.).

### Encoder vs Decoder

**Encoder (BERT-style):** Uses bidirectional (full) self-attention. Each token attends to all other tokens. Good for understanding.

**Decoder (GPT-style):** Uses causal (masked) self-attention. Each token attends only to previous tokens. The mask sets $S_{ij} = -\infty$ for $j > i$ (future positions), which after softmax gives weight 0.

---

## Lesson 10.9: Positional Encoding

### Why Positional Encoding?

Self-attention is **permutation equivariant**: if you shuffle the input sequence, the output shuffles the same way. But order matters in language! "The cat sat on the mat" ≠ "On the mat sat the cat." Positional encodings inject position information.

### Sinusoidal Positional Encoding

$$
\text{PE}(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_\text{model}}}\right)
$$

$$
\text{PE}(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_\text{model}}}\right)
$$

**Why sinusoids?** For any offset $k$, $\text{PE}(pos+k)$ can be written as a linear function of $\text{PE}(pos)$ — the model can learn relative position from the encoding. Also, the encoding extends naturally to positions beyond those seen during training.

### Rotary Position Embeddings (RoPE)

Instead of additive positional encoding, rotate the query and key vectors:

$$
(\mathbf{q}_m, \mathbf{k}_n) \to (\mathbf{R}_m\mathbf{q}_m, \mathbf{R}_n\mathbf{k}_n)
$$

where $\mathbf{R}_m$ is a rotation matrix at angle $\theta m$. The dot product $(\mathbf{R}_m\mathbf{q}_m)^T(\mathbf{R}_n\mathbf{k}_n)$ depends only on $m - n$ (relative position), not absolute positions. This property enables better length generalization. Used in LLaMA, GPT-NeoX.

---

## Lesson 10.10: Normalization Layers

### Batch Normalization

For a mini-batch of activations $\{z_i\}_{i=1}^{B}$ at a layer:

$$
\hat{z}_i = \frac{z_i - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}}, \quad \text{BN}(z_i) = \gamma\hat{z}_i + \beta
$$

where $\mu_B = \frac{1}{B}\sum_i z_i$, $\sigma_B^2 = \frac{1}{B}\sum_i(z_i - \mu_B)^2$, and $\gamma, \beta$ are learnable scale and shift.

**Why it works:** Reduces internal covariate shift (the distribution of each layer's inputs changing as weights update). Allows higher learning rates. Acts as mild regularization (due to noise from batch statistics).

**Limitations:** Fails with small batch sizes (batch stats become noisy). Cannot be used with variable-length sequences. Not appropriate for RNNs.

### Layer Normalization

Normalizes across the feature dimension (not the batch dimension):

$$
\text{LN}(\mathbf{x}) = \frac{\mathbf{x} - \mu(\mathbf{x})}{\sqrt{\sigma(\mathbf{x})^2 + \varepsilon}} \odot \gamma + \beta
$$

where $\mu(\mathbf{x}) = \frac{1}{d}\sum_j x_j$ and $\sigma^2(\mathbf{x}) = \frac{1}{d}\sum_j(x_j - \mu)^2$ are computed per example.

Layer norm is batch-size independent — works for sequence models, small batches, online learning. Standard in Transformers.

### RMSNorm

Simpler than LayerNorm — removes the centering step (only scales):

$$
\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})} \odot \gamma, \quad \text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_j x_j^2}
$$

Empirically similar performance, computationally cheaper. Used in LLaMA, T5.

---

## Lesson 10.11: Weight Initialization — Variance Preservation

### Why Initialization Matters

Random initialization avoids symmetry (all zeros → all gradients identical). But poorly scaled initialization leads to vanishing or exploding activations before training even starts.

### Xavier / Glorot Initialization

For a layer with $n_\text{in}$ inputs and $n_\text{out}$ outputs, using tanh activation:

$$
W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_\text{in} + n_\text{out}}}, \sqrt{\frac{6}{n_\text{in} + n_\text{out}}}\right)
$$

**Derivation:** We want $\text{Var}(z^{(\ell)}) \approx \text{Var}(z^{(\ell-1)})$ (preserve variance through layers). For a linear layer:

$$
\text{Var}(z_j^{(\ell)}) = n_\text{in} \cdot \text{Var}(W) \cdot \text{Var}(a^{(\ell-1)})
$$

Setting $\text{Var}(z^{(\ell)}) = \text{Var}(z^{(\ell-1)})$ requires $\text{Var}(W) = 1/n_\text{in}$ (forward pass). For the backward pass, it's $1/n_\text{out}$. Xavier compromises: $\text{Var}(W) = 2/(n_\text{in} + n_\text{out})$.

### He / Kaiming Initialization

For ReLU (which kills half the variance):

$$
W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_\text{in}}}\right)
$$

**Derivation:** ReLU zeros out negative inputs, halving the variance: $\text{Var}(\text{ReLU}(z)) \approx \frac{1}{2}\text{Var}(z)$. Compensate by doubling initial weight variance: $\text{Var}(W) = 2/n_\text{in}$.

---

## Lesson 10.12: Generalization and Double Descent

### Classical View: Bias-Variance U-Curve

As model complexity increases:
- Bias decreases (more expressive model fits the function better)
- Variance increases (more complex model is more sensitive to data)
- Test error forms a U-shape with an optimal complexity

### Modern View: Double Descent

Belkin et al. (2019) showed that past the interpolation threshold (where train error = 0), as model size grows further, test error *decreases again*:

```
Test
Error
  |      *
  |     * *
  |    *   *
  |   *     *                       *
  |  *       *          *
  | *         *   *  *
  |*            **
  +-----------------------------------------> Model Complexity
                 ^ Interpolation threshold
```

The intuition: very large overparameterized models can interpolate training data while still generalizing because the gradient descent solution among all interpolating solutions prefers *smooth* functions (implicit regularization of SGD).

---

## Lesson 10.13: Variational Autoencoders

### The ELBO Derivation

We want to model the distribution $p(\mathbf{x})$ of data. Introduce latent variable $\mathbf{z}$:

$$
p(\mathbf{x}) = \int p(\mathbf{x}|\mathbf{z})p(\mathbf{z})\, d\mathbf{z}
$$

This integral is intractable. Instead, maximize the **Evidence Lower BOund (ELBO)**:

$$
\log p(\mathbf{x}) \geq \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}[\log p(\mathbf{x}|\mathbf{z})] - KL(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))
$$

**Derivation:**

$$
\log p(\mathbf{x}) = \log \int p(\mathbf{x}|\mathbf{z})p(\mathbf{z})\, d\mathbf{z}
$$

$$
= \log \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}\left[\frac{p(\mathbf{x}|\mathbf{z})p(\mathbf{z})}{q(\mathbf{z}|\mathbf{x})}\right]
$$

$$
\geq \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}\left[\log\frac{p(\mathbf{x}|\mathbf{z})p(\mathbf{z})}{q(\mathbf{z}|\mathbf{x})}\right] \quad \text{(Jensen's inequality)}
$$

$$
= \mathbb{E}_{q}[\log p(\mathbf{x}|\mathbf{z})] - KL(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))
$$

### Reparameterization Trick

The expectation $\mathbb{E}_{q(\mathbf{z}|\mathbf{x})}[\cdot]$ cannot be differentiated through directly (sampling is not differentiable). Reparameterize:

$$
\mathbf{z} = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

Now $\nabla_\phi \mathbb{E}_{q}[f(\mathbf{z})] = \mathbb{E}_{\boldsymbol{\varepsilon}}[\nabla_\phi f(\boldsymbol{\mu}_\phi + \boldsymbol{\sigma}_\phi \odot \boldsymbol{\varepsilon})]$ — gradients can flow through $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$.

### KL Term (Closed Form)

For $q = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2\mathbf{I})$ and $p = \mathcal{N}(\mathbf{0}, \mathbf{I})$:

$$
KL(q \| p) = \frac{1}{2}\sum_j\left(\mu_j^2 + \sigma_j^2 - 1 - \log\sigma_j^2\right)
$$

---

## Lesson 10.14: Diffusion Models

### Forward Process

Add Gaussian noise gradually:

$$
q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I})
$$

Let $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s$. Then:

$$
q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})
$$

At large $T$, $\bar{\alpha}_T \approx 0$ and $\mathbf{x}_T \approx \mathcal{N}(\mathbf{0}, \mathbf{I})$ — pure noise.

### Reverse Process

A neural network learns to denoise:

$$
p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))
$$

**Training objective (simplified):** Predict the noise $\boldsymbol{\varepsilon}$ that was added:

$$
\mathcal{L} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\varepsilon}}\left[\|\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\varepsilon}, t)\|^2\right]
$$

---

## Lesson 10.15: GANs — The Minimax Game

### The Objective

Generator $G$ and Discriminator $D$ play:

$$
\min_G \max_D V(D, G) = \mathbb{E}_{\mathbf{x}\sim p_\text{data}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z}\sim p_\mathbf{z}}[\log(1 - D(G(\mathbf{z})))]
$$

At the optimal $D^*$ for fixed $G$: $D^*(\mathbf{x}) = \frac{p_\text{data}(\mathbf{x})}{p_\text{data}(\mathbf{x}) + p_G(\mathbf{x})}$.

Substituting $D^*$ into $V$, the generator objective becomes:

$$
\min_G C(G) = -\log 4 + 2 \cdot JSD(p_\text{data} \| p_G)
$$

The minimum $C(G) = -\log 4$ is achieved iff $p_G = p_\text{data}$. Training converges when the generator matches the data distribution.

### Mode Collapse and Wasserstein GAN

Vanilla GAN training is unstable. If $D$ becomes too good, the generator gradient vanishes. Wasserstein GAN replaces JS divergence with Wasserstein-1 distance:

$$
W(p, q) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{\mathbf{x}\sim p}[f(\mathbf{x})] - \mathbb{E}_{\mathbf{x}\sim q}[f(\mathbf{x})]
$$

where the sup is over 1-Lipschitz functions. WGAN provides meaningful gradients even when $p$ and $q$ have disjoint support.

---

## Lesson 10.16: Reinforcement Learning Mathematics

### MDP Formulation

A Markov Decision Process (MDP) is a tuple $(S, A, P, R, \gamma)$:
- $S$: state space
- $A$: action space
- $P(s'|s, a)$: transition probability
- $R(s, a)$: expected reward
- $\gamma \in [0,1)$: discount factor

### Bellman Equations

The **value function** $V^\pi(s)$ under policy $\pi$:

$$
V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty}\gamma^t R(s_t, a_t) \mid s_0 = s\right]
$$

**Bellman expectation equation:**

$$
V^\pi(s) = \sum_a \pi(a|s)\left[R(s,a) + \gamma\sum_{s'} P(s'|s,a)V^\pi(s')\right]
$$

**Bellman optimality equation:**

$$V^{\ast}(s) = \max_{a} \left[ R(s,a) + \gamma \sum_{s'} P(s' \mid s,a) V^{\ast}(s') \right]$$

### Policy Gradient Theorem

For a parameterized policy $\pi_\theta$, the policy gradient theorem states:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta\log\pi_\theta(a|s) \cdot Q^{\pi_\theta}(s,a)\right]
$$

This is the foundation of REINFORCE, PPO, and other policy gradient methods. The gradient pushes up the log-probability of actions that lead to high $Q$-values.

### REINFORCE (Monte Carlo Policy Gradient)

$$
\nabla_\theta J(\theta) \approx \frac{1}{T}\sum_{t=0}^{T}\nabla_\theta\log\pi_\theta(a_t|s_t) \cdot G_t
$$

where $G_t = \sum_{k=t}^{T}\gamma^{k-t}r_k$ is the return from time $t$.

**Variance reduction:** Subtract a baseline $b(s_t)$ from the return. The advantage $A_t = G_t - b(s_t)$ reduces variance without changing the gradient's expected value (since $\mathbb{E}[\nabla_\theta\log\pi_\theta(a|s) \cdot b(s)] = 0$).

---

## Python Code

```python
import torch
import torch.nn as nn
import numpy as np

# ---- Backpropagation: Manual vs Autograd ----
def manual_backprop():
    """2-layer MLP: manual BP4 equations."""
    np.random.seed(42)
    X = np.random.randn(4, 2)  # 4 samples, 2 features
    y = np.array([0, 1, 1, 0])

    W1 = np.random.randn(2, 3) * 0.1
    b1 = np.zeros(3)
    W2 = np.random.randn(3, 1) * 0.1
    b2 = np.zeros(1)

    def sigmoid(z): return 1 / (1 + np.exp(-z))

    # Forward
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)

    loss = -np.mean(y.reshape(-1,1) * np.log(a2 + 1e-8) + (1-y.reshape(-1,1)) * np.log(1-a2+1e-8))

    # Backward (BP equations)
    n = len(X)
    delta2 = (a2 - y.reshape(-1,1)) / n          # BP1
    dW2 = a1.T @ delta2                            # BP4
    db2 = delta2.sum(axis=0)                       # BP3
    delta1 = (delta2 @ W2.T) * a1 * (1 - a1)     # BP2
    dW1 = X.T @ delta1                             # BP4
    db1 = delta1.sum(axis=0)                       # BP3

    return loss, dW1, dW2

loss, dW1, dW2 = manual_backprop()
print(f"Loss: {loss:.4f}, ||dW1||: {np.linalg.norm(dW1):.4f}")

# ---- Attention from scratch ----
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, V), weights

# Example: 2 queries, 3 keys/values, d_k=4, d_v=4
Q = torch.randn(2, 4)
K = torch.randn(3, 4)
V = torch.randn(3, 4)
output, attn_weights = scaled_dot_product_attention(Q, K, V)
print(f"Attention output shape: {output.shape}")   # (2, 4)
print(f"Attention weights:\n{attn_weights.detach().numpy()}")

# ---- Transformer block ----
class TransformerBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-norm variant (used in modern LLMs)
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x

model = TransformerBlock()
x = torch.randn(2, 10, 64)  # batch=2, seq_len=10, d_model=64
out = model(x)
print(f"Transformer block output shape: {out.shape}")  # (2, 10, 64)
params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {params:,}")  # ~66K for d_model=64

# ---- He initialization check ----
def check_variance_preservation():
    torch.manual_seed(42)
    x = torch.randn(1000, 256)

    # Xavier init (for tanh/sigmoid)
    W_xavier = torch.empty(256, 256)
    nn.init.xavier_uniform_(W_xavier)
    z_xavier = x @ W_xavier
    print(f"Xavier: input var={x.var():.3f}, output var={z_xavier.var():.3f}")

    # He init (for ReLU)
    W_he = torch.empty(256, 256)
    nn.init.kaiming_normal_(W_he, nonlinearity='relu')
    z_he = torch.relu(x @ W_he)
    print(f"He+ReLU: input var={x.var():.3f}, output var={z_he.var():.3f}")

check_variance_preservation()
```

---

## Summary

| Concept | Key Insight | Formula |
|---------|------------|---------|
| Universal Approximation | 1 hidden layer suffices; depth is about efficiency | — |
| Backprop | Chain rule + memoization | $\boldsymbol{\delta}^{(\ell)} = (\mathbf{W}^{(\ell+1)T}\boldsymbol{\delta}^{(\ell+1)}) \odot \sigma'(\mathbf{z}^{(\ell)})$ |
| Vanishing gradients | Product of small numbers shrinks | $\prod \|\mathbf{W}_{hh}\| \to 0$ |
| LSTM fix | Additive cell state = gradient highway | $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$ |
| Attention scaling | Prevent softmax saturation | Divide by $\sqrt{d_k}$ |
| Layer Norm | Normalize per-example, not per-batch | $(\mathbf{x}-\mu)/\sigma \odot \gamma + \beta$ |
| He init | Compensate for ReLU variance halving | $\text{Var}(W) = 2/n_\text{in}$ |
| VAE ELBO | Lower bound on log-likelihood | $\mathbb{E}_q[\log p(\mathbf{x}|\mathbf{z})] - KL(q \| p)$ |
| Policy gradient | Increase prob of good actions | $\mathbb{E}[\nabla\log\pi \cdot Q]$ |

---

*← [Part 9: Classical ML](part-09-classical-ml.md) | [Part 11: LLM Mathematics →](part-11-llm-mathematics.md)*
