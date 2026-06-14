# Part 11: LLM Mathematics

> **Prerequisites:** Parts 0–10
> **Status:** Complete
> **Lessons:** 12

---

## The Problem This Part Solves

You understand transformers mathematically. Now the question is: **what makes a Language Model?** How do you train a system that generates coherent text? What mathematics underlies prompting, fine-tuning, RLHF, and the emergent capabilities that appear at scale?

This part covers the math that is unique to large language models — the pieces that go beyond the generic deep learning tools of Part 10.

---

## Lesson 11.1: Autoregressive Language Modeling

### The Task

A language model assigns probability to sequences of tokens. For a sequence $w_1, w_2, \ldots, w_T$, the autoregressive factorization via the chain rule of probability:

$$
P(w_1, w_2, \ldots, w_T) = \prod_{t=1}^{T} P(w_t \mid w_1, w_2, \ldots, w_{t-1})
$$

The model is trained to maximize log-likelihood over the training corpus:

$$
\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T}\log P(w_t \mid w_1, \ldots, w_{t-1})
$$

This is exactly **categorical cross-entropy** at each position, averaged over the sequence.

### Perplexity

The standard metric for language models is **perplexity**:

$$
\text{PPL} = \exp\left(-\frac{1}{T}\sum_{t=1}^{T}\log P(w_t \mid w_{<t})\right)
$$

Perplexity is the exponential of the average negative log-likelihood. A model with perplexity $K$ is as "confused" as if it had to choose uniformly among $K$ equally likely options at each step.

- **PPL = 1:** Perfect prediction (knows exactly what comes next)
- **PPL = V:** Uniform distribution over vocabulary (knows nothing)
- **Human-level English:** PPL ≈ 15–30 depending on domain
- **GPT-4:** PPL < 10 on standard benchmarks

### Autoregressive Generation

At inference time, sample token by token:

$$
w_t \sim P(\cdot \mid w_1, \ldots, w_{t-1})
$$

**Greedy decoding:** $w_t = \arg\max P(w_t \mid w_{<t})$ — fast but repetitive.

**Beam search:** Maintain top-$B$ partial sequences at each step. Output the sequence with highest total log-probability.

**Nucleus (top-p) sampling:** At each step, sample from the smallest set of tokens whose cumulative probability $\geq p$. Adapts to the distribution's entropy dynamically.

---

## Lesson 11.2: Tokenization Mathematics — BPE

### Why Not Just Characters or Words?

- **Characters:** Very long sequences, no word-level semantics
- **Words:** Vocabulary explosion with morphological variants ("run", "runs", "running", "ran"); unknown words at test time
- **BPE (Byte-Pair Encoding):** Sub-word units that balance vocabulary size vs sequence length

### BPE Algorithm

```
1. Start with character-level vocabulary
2. Count all adjacent symbol pairs in the corpus
3. Merge the most frequent pair into a new symbol
4. Repeat step 2-3 for K merges (K = vocab size - initial chars)
```

**Mathematical framing:** BPE is greedy compression. Each merge reduces total corpus length by (frequency of pair - 1). After $K$ merges, we have a vocabulary of size $|\text{chars}| + K$ that minimizes corpus length greedily.

**Fertility:** The number of tokens a word maps to. English words typically have fertility 1–3 with a 50K vocabulary. Non-English languages often have higher fertility with English-trained tokenizers (up to 10x for some scripts), which is a form of representation inequality.

### Tokenization and the Vocabulary Distribution

The distribution of token IDs is extremely skewed — a few tokens are very common (articles, punctuation, common words) and most are rare. The softmax at the output layer must handle a vocabulary of 32K–100K tokens.

**Tied embeddings:** The input embedding matrix $\mathbf{E} \in \mathbb{R}^{V \times d}$ is often reused (transposed) as the output projection. This reduces parameters by $Vd$ and empirically improves performance.

---

## Lesson 11.3: Softmax Temperature

### The Temperature Parameter

At inference, scale logits before softmax:

$$
P(w_t = k \mid w_{\lt t}) = \frac{e^{z_k/T}}{\sum_j e^{z_j/T}}
$$

where $T > 0$ is the temperature.

**Effect of temperature:**
- $T \to 0$: distribution collapses to argmax (greedy decoding)
- $T = 1$: standard training distribution
- $T \to \infty$: approaches uniform distribution over vocabulary

**Why temperature controls "creativity":** High temperature flattens the distribution, giving more probability mass to low-probability tokens (unexpected, creative choices). Low temperature sharpens it, making the model conservative and repetitive.

**Mathematical characterization:** Scaling logits by $1/T$ is equivalent to applying a power $1/T$ to the probabilities (before renormalization). This is why temperature is also called "sharpness" in some literature.

### Top-k and Top-p Sampling

**Top-k:** Sample from the $k$ tokens with highest probability. Fixed $k$ doesn't adapt to entropy — at some positions the distribution is peaked (one obvious next word), at others it is flat.

**Top-p (nucleus):** Find smallest set $V_p$ such that $\sum_{k \in V_p} P(k) \geq p$. Adaptive: at low-entropy positions, $|V_p|$ is small; at high-entropy positions, it is large.

---

## Lesson 11.4: KV Cache

### The Memory Savings

In autoregressive decoding, at each step $t$ you compute attention over all previous tokens. Without caching:
- Keys and values are recomputed for all previous tokens at every step
- Cost per step: $O(t \cdot d)$
- Total cost for sequence of length $T$: $O(T^2 \cdot d)$

With KV cache:
- Store $\mathbf{K}_t$ and $\mathbf{V}_t$ for each token as it is generated
- At step $t$, only compute query for the new token, attending to cached KV
- Cost per step: $O(d)$ (just the new query-key dot products)
- Memory: $O(T \cdot d \cdot 2 \cdot L)$ where $L$ is number of layers

For a 7B parameter model with 32 layers, $d=4096$, sequence length 4096, using float16:

$$
\text{KV cache size} = 4096 \times 4096 \times 2 \times 32 \times 2\text{ bytes} \approx 2\text{ GB}
$$

This is why long context is memory-expensive.

### Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)

**MQA:** Share keys and values across all heads, only queries differ. Reduces KV cache size by factor $h$ (number of heads). Some accuracy loss.

**GQA:** Share keys and values within groups of heads. Tradeoff between MQA and full MHA. Used in LLaMA 2/3, Gemma.

KV cache size with GQA: reduced by factor $h/g$ where $g$ is number of groups.

---

## Lesson 11.5: Mixture of Experts (MoE)

### Sparse Gating

A MoE layer replaces the dense FFN with $E$ expert FFNs and a router:

$$
\text{MoE}(\mathbf{x}) = \sum_{i=1}^{E} G(\mathbf{x})_i \cdot E_i(\mathbf{x})
$$

where $G(\mathbf{x})$ is the gating function that selects which experts activate.

**Top-k sparse gating:** Activate only the top-$k$ experts by gating score:

$$
G(\mathbf{x}) = \text{Softmax}(\text{TopK}(\mathbf{W}_g\mathbf{x}, k))
$$

Typically $k=2$ (top-2 routing). 98% of experts are zero — only 2 expert FFNs compute.

**Parameters vs FLOPS:** With $E$ experts and top-2 routing, parameters scale as $E\times$, but FLOPS per token stay at $2\times$ expert FLOPS. MoE achieves large model capacity at constant inference cost.

### Load Balancing Loss

Without explicit regularization, routers collapse — all tokens route to 1–2 experts. Add a load balancing auxiliary loss:

$$
\mathcal{L}_\text{aux} = \alpha \sum_{i=1}^{E} f_i \cdot P_i
$$

where $f_i$ = fraction of tokens routed to expert $i$, $P_i$ = mean routing probability for expert $i$.

Maximizing entropy of the routing distribution prevents collapse. $\alpha$ is a small hyperparameter (typically 0.01).

---

## Lesson 11.6: RLHF — Reinforcement Learning from Human Feedback

### The Three-Stage Pipeline

1. **Supervised Fine-Tuning (SFT):** Fine-tune the base LM on demonstration data from human experts. Gets you a model $\pi_\text{SFT}$.

2. **Reward Modeling (RM):** Train a reward model $r_\phi$ to predict human preferences. Given pairs $(x, y_w, y_l)$ where $y_w$ is preferred over $y_l$:

$$
\mathcal{L}_\text{RM} = -\log\sigma(r_\phi(x, y_w) - r_\phi(x, y_l))
$$

This is the Bradley-Terry model for pairwise preferences.

3. **RL Fine-Tuning (PPO):** Optimize the LM to maximize reward while staying close to $\pi_\text{SFT}$:

$$
\mathcal{L}_\text{PPO} = \mathbb{E}_{(x,y)\sim\pi_\theta}\left[r_\phi(x, y) - \beta \log\frac{\pi_\theta(y|x)}{\pi_\text{SFT}(y|x)}\right]
$$

The KL penalty $\beta \cdot KL(\pi_\theta \| \pi_\text{SFT})$ prevents reward hacking (the model exploiting the reward model's errors).

### KL Penalty Derivation

The KL term:

$$
KL(\pi_\theta \| \pi_\text{SFT}) = \mathbb{E}_{y\sim\pi_\theta}\left[\log\frac{\pi_\theta(y|x)}{\pi_\text{SFT}(y|x)}\right]
$$

This is computed token-by-token during generation. Large deviation from $\pi_\text{SFT}$ is penalized. Without this term, models learn to generate short, high-reward outputs (reward hacking) or drift to degenerate text.

---

## Lesson 11.7: DPO — Direct Preference Optimization

### Avoiding the Reward Model

RLHF is complex: separate reward model training, PPO implementation, KV cache for reference model. DPO (Rafailov et al., 2023) shows you can skip the reward model entirely.

**Key insight:** The optimal policy under the RLHF objective has a closed form:

$$
\pi^{\ast}(y|x) = \frac{1}{Z(x)}\pi_\text{ref}(y|x)\exp\left(\frac{1}{\beta}r^{\ast}(x,y)\right)
$$

This implies:

$$
r^{\ast}(x,y) = \beta\log\frac{\pi^{\ast}(y \mid x)}{\pi_\text{ref}(y \mid x)} + \beta\log Z(x)
$$

Substituting into the Bradley-Terry preference model and noting that $Z(x)$ cancels in the pairwise comparison:

$$
\mathcal{L}_\text{DPO} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)\right]
$$

This trains the policy directly from preference data without any RL. The model is pushed to increase the (relative log-probability) gap between preferred and dispreferred completions.

---

## Lesson 11.8: Scaling Laws

### Chinchilla Scaling Laws

Hoffmann et al. (2022) derived compute-optimal training: for a given FLOP budget $C$, what model size $N$ and training tokens $D$ should you use?

**Empirical fits:**
- Loss as function of $N$ (parameters): $L(N) \approx \frac{A}{N^\alpha}$
- Loss as function of $D$ (tokens): $L(D) \approx \frac{B}{D^\beta}$
- Combined: $L(N, D) \approx \frac{A}{N^\alpha} + \frac{B}{D^\beta} + E$

where $E$ is the irreducible entropy of language.

**Compute budget:** Training FLOPS $\approx 6ND$ (forward + backward, approximate).

**Compute-optimal allocation (Chinchilla):** At fixed compute $C$:

$$
N_\text{opt} \propto C^{0.5}, \quad D_\text{opt} \propto C^{0.5}
$$

This means: **tokens ≈ 20 × parameters** for compute-optimal training. The original GPT-3 (175B params, 300B tokens) was undertrained; Chinchilla (70B params, 1.4T tokens) achieved the same performance with 4× less compute.

### Emergent Capabilities

Some capabilities appear abruptly at certain scales — near zero performance below a threshold, then sudden improvement. Mathematical explanation: many-shot tasks require "routing" through multiple reasoning steps. Each step has some failure probability $p$. For $k$ steps, success probability is $(1-p)^k$. If $p$ depends on model scale and crosses some threshold, the overall capability appears to emerge discontinuously.

---

## Lesson 11.9: In-Context Learning (ICL)

### What Is ICL?

At inference time, provide examples in the prompt: `[Q1, A1], [Q2, A2], ..., [Qtest, ?]`. The model answers the new question by "learning" from examples — without any weight updates.

### Theoretical Perspectives

**Bayesian inference view:** ICL approximates Bayesian posterior inference over a distribution of tasks. The LM represents a prior over task functions; few-shot examples update the posterior.

**Gradient descent view:** Akyürek et al. (2023) showed that transformers can implement gradient descent in their forward pass. The self-attention layers can simulate one step of gradient descent on the demonstrated examples. The keys/values represent a linear model; the query updates the model.

**Meta-learning view:** Large-scale pretraining is pretraining a meta-learner. The model has seen so many diverse tasks that it has learned to learn from examples.

---

## Lesson 11.10: Chain of Thought — Expressiveness of Multi-Step Reasoning

### The Problem

Some tasks require computational steps that cannot be compressed into a single feed-forward pass. For example, solving a 10-step arithmetic problem requires 10 sequential operations; a transformer of fixed depth cannot implement arbitrary-depth recursion in one forward pass.

### Formal Expressiveness

**Without CoT:** Transformers with $L$ layers and $d$-dimensional hidden states can compute functions in $TC^0$ (threshold circuits of constant depth). This class does not include many reasoning problems.

**With CoT (scratchpad):** Each token generated extends the "memory" of the computation. With $T$ reasoning steps, the model can implement $TC^0$ computations sequentially, accessing up to $T$ "memory slots." This effectively gives the model $T$-layer depth at inference time.

**Formal result (Feng et al., 2024):** A constant-depth transformer with $O(1)$ scratchpad tokens per step can solve any problem solvable in $O(T)$ time — including problems requiring $T$ sequential reasoning steps.

### Why CoT Improves Accuracy

Numerical estimates for arithmetic:

Without CoT: "What is 23 × 47?"
→ Single forward pass must compress the multiplication into weights

With CoT: "23 × 47 = 23 × 40 + 23 × 7 = 920 + 161 = 1081"
→ Each token in the scratchpad is a "layer" of computation

---

## Lesson 11.11: FlashAttention — Exact Attention with Subquadratic Memory

### The Memory Bottleneck

Standard attention: $\mathbf{A} = \text{softmax}(\mathbf{Q}\mathbf{K}^T/\sqrt{d_k})$, then output $= \mathbf{A}\mathbf{V}$.

The attention matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$ requires $O(n^2)$ memory. For $n=8192$, $\mathbf{A}$ is 256M entries × 2 bytes = 512MB per head per batch.

GPU memory (VRAM) is limited. For long contexts, the bottleneck is storing and accessing $\mathbf{A}$.

### FlashAttention (Dao et al., 2022)

**Key insight:** Use tiling to compute attention in tiles that fit in fast SRAM (L1/L2 cache on GPU), never materializing the full $\mathbf{A}$ matrix in slow HBM (GPU VRAM).

**Online softmax trick:** Compute softmax in a streaming fashion:
- Maintain running max $m$ and running sum $d$ of exponentials
- For each new tile, update: $m_\text{new} = \max(m, m_\text{tile})$, rescale previous and add new

**Result:** Exact same output as standard attention, but:
- Memory: $O(n)$ instead of $O(n^2)$ (no attention matrix stored)
- Speed: 2–4× faster than standard attention on modern GPUs (better memory access patterns)

---

## Lesson 11.12: Long Context — Mathematical Challenges

### Length Extrapolation

Sinusoidal PE: if trained on length $n$, the model has never seen position indices $> n$. Empirically degrades.

RoPE + linear scaling: scale position indices by $n/n_\text{train}$ to fit longer sequences into the trained range. Simple but introduces small errors.

**YaRN (Yet Another RoPE extensioN):** Scale different frequency components of RoPE differently — low frequencies (long-range dependencies) are scaled less than high frequencies (short-range).

### Attention Sink Phenomenon

In LLMs, the first token (often a special `[BOS]` token) accumulates disproportionately large attention weights. This is an "attention sink" — all tokens attend to it as a no-op. This enables the KV cache to be managed more aggressively: the first few tokens must never be evicted.

**StreamingLLM** exploits this: keep only the first $k$ tokens (sinks) and the most recent $w$ tokens. This enables infinite-context generation with bounded memory.

---

## Python Code

```python
import torch
import torch.nn.functional as F
import numpy as np

# ---- Perplexity computation ----
def compute_perplexity(logits, targets):
    """
    logits: (batch, seq_len, vocab_size)
    targets: (batch, seq_len)
    """
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        reduction='mean'
    )
    return torch.exp(loss).item()

# Simulate a 3-step sequence, vocab=5
logits = torch.randn(1, 3, 5)
targets = torch.randint(0, 5, (1, 3))
ppl = compute_perplexity(logits, targets)
print(f"Perplexity: {ppl:.2f}")

# ---- Temperature sampling ----
def temperature_sample(logits, temperature=1.0, top_p=0.9):
    """Top-p sampling with temperature."""
    scaled = logits / temperature
    probs = F.softmax(scaled, dim=-1)

    # Sort probabilities descending
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above top_p
    sorted_probs[cumulative - sorted_probs > top_p] = 0.0
    sorted_probs /= sorted_probs.sum()

    # Sample
    sampled_idx = torch.multinomial(sorted_probs, 1)
    return sorted_idx[sampled_idx].item()

logits = torch.tensor([1.0, 2.0, 0.5, 0.1, 3.0])
print("Sampled token (T=1.0, top-p=0.9):", temperature_sample(logits, 1.0))
print("Sampled token (T=0.1, greedy-ish):", temperature_sample(logits, 0.1))
print("Sampled token (T=2.0, creative):", temperature_sample(logits, 2.0))

# ---- Scaling law simulation ----
def predict_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    """Chinchilla scaling law prediction."""
    return A / (N ** alpha) + B / (D ** beta) + E

# Compare GPT-3 vs Chinchilla training
N_gpt3 = 175e9
D_gpt3 = 300e9
N_chinchilla = 70e9
D_chinchilla = 1.4e12  # 20x params

C_gpt3 = 6 * N_gpt3 * D_gpt3
C_chinchilla = 6 * N_chinchilla * D_chinchilla

print(f"\nCompute (GPT-3): {C_gpt3:.2e} FLOPs")
print(f"Compute (Chinchilla): {C_chinchilla:.2e} FLOPs")
print(f"GPT-3 predicted loss: {predict_loss(N_gpt3, D_gpt3):.3f}")
print(f"Chinchilla predicted loss: {predict_loss(N_chinchilla, D_chinchilla):.3f}")

# ---- DPO loss ----
def dpo_loss(policy_logprobs_w, policy_logprobs_l,
             ref_logprobs_w, ref_logprobs_l, beta=0.1):
    """
    policy_logprobs_w: log P_theta(y_w | x)
    policy_logprobs_l: log P_theta(y_l | x)
    ref_logprobs_w: log P_ref(y_w | x)
    ref_logprobs_l: log P_ref(y_l | x)
    """
    pi_logratios = policy_logprobs_w - policy_logprobs_l
    ref_logratios = ref_logprobs_w - ref_logprobs_l
    losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))
    return losses.mean()

# Example: preferred completion more likely than dispreferred
policy_w = torch.tensor(-2.0)   # log prob of preferred
policy_l = torch.tensor(-4.0)   # log prob of dispreferred
ref_w = torch.tensor(-2.5)
ref_l = torch.tensor(-3.0)

loss = dpo_loss(policy_w, policy_l, ref_w, ref_l)
print(f"\nDPO loss: {loss.item():.4f}")
```

---

## Summary

| Concept | Key Formula | Significance |
|---------|-------------|--------------|
| Perplexity | $\exp(-\frac{1}{T}\sum \log P(w_t\|w_{<t}))$ | Language model quality metric |
| Temperature | $P(k) \propto e^{z_k/T}$ | Controls generation diversity |
| KV cache | $O(T \cdot d \cdot 2 \cdot L)$ memory | Enables efficient autoregressive decoding |
| RLHF objective | $\mathbb{E}[r(x,y)] - \beta \cdot KL(\pi \| \pi_\text{SFT})$ | Align LM to human preferences |
| DPO | Reparameterized RLHF as supervised learning | No RL needed for preference training |
| Chinchilla | Tokens $\approx 20\times$ params for compute-optimal | Data scaling law |
| CoT expressiveness | $O(1)$ depth → $O(T)$ depth with $T$ scratchpad tokens | Why reasoning improves with CoT |
| FlashAttention | Tiled computation, $O(n)$ memory | Long-context efficient attention |

---

*← [Part 10: Deep Learning](part-10-deep-learning.md) | [Part 12: Graph Mathematics →](part-12-graph-mathematics.md)*
