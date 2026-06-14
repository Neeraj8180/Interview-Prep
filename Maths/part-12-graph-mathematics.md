# Part 12: Graph Mathematics

> **Prerequisites:** Parts 0–10 (Part 11 helpful but not required)
> **Status:** Complete
> **Lessons:** 10

---

## The Problem This Part Solves

Many of the most interesting data structures are not tables — they are graphs. Social networks, molecules, knowledge bases, road maps, dependency trees. Classical ML treats each example independently. **Graph neural networks** extend deep learning to relational data.

But GNNs rest on mathematical foundations: graph theory, spectral theory, and message passing. Understanding these foundations lets you design GNN architectures intentionally and diagnose their failure modes.

---

## Lesson 12.1: Graph Theory Foundations

### Definitions

A **graph** $G = (V, E)$ consists of a vertex set $V$ (with $|V| = n$) and an edge set $E \subseteq V \times V$.

- **Undirected:** $(u,v) \in E \Leftrightarrow (v,u) \in E$
- **Directed:** $(u,v) \neq (v,u)$
- **Weighted:** Each edge has a weight $w_{uv} \in \mathbb{R}$
- **Bipartite:** $V = A \cup B$, all edges go between $A$ and $B$

### Representations

**Adjacency matrix** $\mathbf{A} \in \{0,1\}^{n \times n}$:

$$
A_{ij} = \begin{cases} 1 & (i,j) \in E \\ 0 & \text{otherwise} \end{cases}
$$

**Degree matrix** $\mathbf{D}$: diagonal matrix with $D_{ii} = \deg(v_i) = \sum_j A_{ij}$.

**Adjacency list:** For each node, store list of neighbors. Memory $O(n + |E|)$ vs $O(n^2)$ for adjacency matrix.

### Key Graph Properties

| Property | Definition | Relevance |
|----------|-----------|-----------|
| Degree $d_v$ | Number of neighbors of $v$ | Highly connected nodes (hubs) |
| Diameter | Max shortest path between any two nodes | Information propagation depth |
| Clustering coefficient | Fraction of triangles among neighbors | Local density |
| Connected component | Maximal subgraph with paths between all pairs | Isolated subgraphs |
| Bipartiteness | No odd-length cycles | Two-sided matching problems |

---

## Lesson 12.2: Spectral Graph Theory

### The Graph Laplacian

The **unnormalized Laplacian**:

$$
\mathbf{L} = \mathbf{D} - \mathbf{A}
$$

Properties:
- Symmetric PSD: $\mathbf{L} = \mathbf{L}^T$ and $\mathbf{x}^T\mathbf{L}\mathbf{x} \geq 0$ for all $\mathbf{x}$
- Smallest eigenvalue is 0 with eigenvector $\mathbf{1}$ (all-ones)
- Number of 0 eigenvalues = number of connected components

The quadratic form has an elegant form:

$$
\mathbf{x}^T\mathbf{L}\mathbf{x} = \sum_{(i,j)\in E}(x_i - x_j)^2
$$

This measures the **smoothness** of a signal $\mathbf{x}$ on the graph: large when neighboring nodes have very different values, small when the signal varies slowly.

### Normalized Laplacian

$$
\tilde{\mathbf{L}} = \mathbf{D}^{-1/2}\mathbf{L}\mathbf{D}^{-1/2} = \mathbf{I} - \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}
$$

Eigenvalues of $\tilde{\mathbf{L}}$ lie in $[0, 2]$. Eigenvectors form the **graph Fourier basis** — the analog of the Fourier transform for graph-structured signals.

### Spectral Clustering

1. Compute the $k$ smallest eigenvectors $\mathbf{U} = [\mathbf{u}_1, \ldots, \mathbf{u}_k]$ of $\mathbf{L}$
2. Row-normalize $\mathbf{U}$
3. Run k-means on the rows of $\mathbf{U}$

**Why it works:** The $k$ smallest eigenvectors of $\mathbf{L}$ capture the $k$-way partition of the graph that minimizes cut (the sum of edge weights between clusters). Clustering in eigenvector space approximates this optimal partition.

---

## Lesson 12.3: Random Walks on Graphs

### The Random Walk

At each step, move from current node to a uniformly random neighbor:

$$
P(X_{t+1} = j \mid X_t = i) = \frac{A_{ij}}{d_i}
$$

The transition matrix is $\mathbf{P} = \mathbf{D}^{-1}\mathbf{A}$.

**Stationary distribution:** $\boldsymbol{\pi} = \mathbf{d}/(2|E|)$ where $\mathbf{d}$ is the degree vector. High-degree nodes are visited more often.

### PageRank

Google's original algorithm: model web surfers as random walkers with a "teleportation" probability $\alpha$:

$$
\text{PR}(v) = \frac{\alpha}{n} + (1-\alpha)\sum_{u:(u,v)\in E}\frac{\text{PR}(u)}{d_u}
$$

In matrix form: $\mathbf{r} = \alpha\frac{\mathbf{1}}{n} + (1-\alpha)\mathbf{P}^T\mathbf{r}$.

Solved iteratively (power iteration): $\mathbf{r}_{t+1} = \alpha\frac{\mathbf{1}}{n} + (1-\alpha)\mathbf{P}^T\mathbf{r}_t$.

**Convergence:** Guaranteed by the Perron-Frobenius theorem since the teleportation ensures the Markov chain is ergodic.

---

## Lesson 12.4: Graph Neural Networks — Message Passing Framework

### The Core Abstraction

All modern GNNs follow the **message passing** framework. At each layer, each node:
1. **Aggregates** messages from its neighbors
2. **Updates** its own representation

Formally, for node $v$ at layer $\ell$:

$$
\mathbf{m}_v^{(\ell)} = \text{AGGREGATE}^{(\ell)}\left(\{\mathbf{h}_u^{(\ell-1)} : u \in \mathcal{N}(v)\}\right)
$$

$$
\mathbf{h}_v^{(\ell)} = \text{UPDATE}^{(\ell)}\left(\mathbf{h}_v^{(\ell-1)}, \mathbf{m}_v^{(\ell)}\right)
$$

After $k$ layers, each node's representation captures its $k$-hop neighborhood.

### Graph Convolutional Network (GCN)

Kipf & Welling (2017) define:

$$
\mathbf{H}^{(\ell+1)} = \sigma\left(\tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}\mathbf{H}^{(\ell)}\mathbf{W}^{(\ell)}\right)
$$

where $\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}$ (adjacency with self-loops) and $\tilde{\mathbf{D}}$ is the corresponding degree matrix.

The term $\hat{\mathbf{A}} = \tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}$ is the **symmetrically normalized adjacency** with self-loops. Each layer computes a weighted average of neighbor features and applies a linear transformation.

**Spectral interpretation:** GCN implements a first-order approximation of spectral graph convolution, learned via the parameter matrix $\mathbf{W}$.

### Graph Attention Network (GAT)

Instead of fixed normalization, learn attention weights between neighbors:

$$
\mathbf{h}_v^{(\ell+1)} = \sigma\left(\sum_{u \in \mathcal{N}(v) \cup \{v\}} \alpha_{vu}^{(\ell)}\mathbf{W}^{(\ell)}\mathbf{h}_u^{(\ell)}\right)
$$

Attention weight:

$$
\alpha_{vu} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^T[\mathbf{W}\mathbf{h}_v \| \mathbf{W}\mathbf{h}_u]))}{\sum_{k\in\mathcal{N}(v)}\exp(\text{LeakyReLU}(\mathbf{a}^T[\mathbf{W}\mathbf{h}_v \| \mathbf{W}\mathbf{h}_k]))}
$$

where $\|$ denotes concatenation. Multi-head attention with $H$ heads and concatenation:

$$
\mathbf{h}_v^{(\ell+1)} = \text{concat}_{h=1}^{H}\sigma\left(\sum_{u\in\mathcal{N}(v)}\alpha_{vu}^h\mathbf{W}^h\mathbf{h}_u^{(\ell)}\right)
$$

---

## Lesson 12.5: Theoretical Limits of GNNs — The WL Test

### The Weisfeiler-Lehman (WL) Isomorphism Test

WL is a classical algorithm for graph isomorphism:
1. Each node starts with the same label
2. Iteratively: each node gets a new label = hash of its current label + multiset of neighbor labels
3. Two graphs are non-isomorphic if their label distributions differ at any iteration

**Theorem (Xu et al., 2019):** No GNN with sum aggregation and injective update functions can distinguish graphs that WL cannot distinguish. MPNNs are exactly as powerful as 1-WL.

**What GNNs cannot distinguish:** Cycles of even length from two disconnected paths of the same length; regular graphs of different structures (if every node has the same degree $d$, all nodes look identical to a GNN regardless of global structure).

### Going Beyond WL: Higher-Order GNNs

2-WL, 3-WL: color pairs/triples of nodes instead of individual nodes. More powerful but $O(n^2)$, $O(n^3)$ memory — expensive.

**Practical alternatives:**
- **PNA:** Multiple aggregators (sum, mean, max, std) with degree scalers
- **NGNN:** Add unique IDs (breaks permutation equivariance but increases power)
- **Subgraph GNNs:** Compute GNN on subgraphs centered at each node

---

## Lesson 12.6: Graph-Level Predictions — Pooling

### The Readout Problem

Node classification: output per node. Graph classification: need a single vector for the whole graph. This requires a **pooling** (readout) operation.

**Global pooling:**

$$
\mathbf{h}_G = \text{READOUT}(\{\mathbf{h}_v^{(L)} : v \in V\})
$$

Common choices: $\text{sum}$, $\text{mean}$, $\text{max}$.

- **Sum:** Sensitive to graph size (larger graphs → larger vectors). But expressive: can count nodes with certain properties.
- **Mean:** Normalized by graph size. Cannot count — can only express averages.
- **Max:** Takes most prominent feature. Ignores how many nodes have it.

**Theorem (Xu et al., 2019):** Sum pooling is strictly more expressive than mean or max pooling for graph classification.

### Hierarchical Pooling (DiffPool)

Learn a soft cluster assignment matrix $\mathbf{S}^{(\ell)} \in \mathbb{R}^{n_\ell \times n_{\ell+1}}$:

$$
\mathbf{X}^{(\ell+1)} = \mathbf{S}^{(\ell)T}\mathbf{Z}^{(\ell)}, \quad \mathbf{A}^{(\ell+1)} = \mathbf{S}^{(\ell)T}\mathbf{A}^{(\ell)}\mathbf{S}^{(\ell)}
$$

Coarsens the graph at each level, building a hierarchical representation.

---

## Lesson 12.7: Knowledge Graph Embeddings

### The Task

A knowledge graph is a set of triples $(h, r, t)$: head entity, relation, tail entity. Example: (Paris, capital_of, France).

Goal: **link prediction** — predict missing triples. Does (Berlin, capital_of, Germany) hold?

### Translation-Based Models

**TransE:** Embed entities and relations in $\mathbb{R}^d$. Encode the triple as a translation:

$$
\mathbf{h} + \mathbf{r} \approx \mathbf{t} \text{ if } (h, r, t) \text{ is true}
$$

Score: $f(h, r, t) = -\|\mathbf{h} + \mathbf{r} - \mathbf{t}\|$. Minimize contrastive loss over true vs corrupted triples.

**Limitation:** TransE cannot model symmetric relations (if $h+r \approx t$ and $t+r \approx h$, then $r \approx 0$).

**RotatE:** Model relations as rotations in complex space:

$$
\mathbf{h} \circ \mathbf{r} \approx \mathbf{t}, \quad r_i \in \mathbb{C}, |r_i| = 1
$$

Can model symmetric, antisymmetric, inverse, and composition relations.

---

## Lesson 12.8: Graph Transformers

### Motivating Limitation of MPNNs

MPNNs have limited receptive field (diameter must be ≥ number of layers for global information). Long-range dependencies require very deep GNNs, which suffer from:
- **Over-smoothing:** Repeated averaging makes all node representations similar
- **Over-squashing:** Information from exponentially many nodes compressed into fixed-size vectors

### Combining Attention with Graphs

**Graphformer (Microsoft, 2021):** Add structural biases to standard transformer attention:

$$
A_{ij} = \frac{(\mathbf{x}_i\mathbf{W}_Q)(\mathbf{x}_j\mathbf{W}_K)^T}{\sqrt{d}} + b_{\phi(i,j)} + c_{d(i,j)}
$$

where $b_{\phi(i,j)}$ is a learnable bias based on edge features and $c_{d(i,j)}$ is a bias based on shortest-path distance. The model attends over all node pairs (quadratic) but uses structural biases to encode graph topology.

---

## Lesson 12.9: Applications in AI

### Molecular Property Prediction

Atoms = nodes, bonds = edges. Node features: atom type, charge, aromaticity. Edge features: bond type, stereochemistry.

GNNs (particularly MPNN variants) are state-of-the-art for:
- Drug toxicity prediction
- Molecular solubility
- Reaction outcome prediction

**ChemBERTa / MolBERT:** Pretrain transformers on SMILES strings; fine-tune for molecular properties.

### Recommendation Systems

Users and items form a bipartite graph. User-item interactions = edges. GNNs (LightGCN, PinSage) propagate information through this graph:

$$
\mathbf{e}_u^{(k+1)} = \sum_{i\in\mathcal{N}(u)}\frac{1}{\sqrt{|\mathcal{N}(u)||\mathcal{N}(i)|}} \mathbf{e}_i^{(k)}
$$

(LightGCN — simplified GCN without self-loops or activation functions)

Multi-hop propagation captures collaborative filtering signals: if user A liked items $\{x, y, z\}$ and user B liked items $\{x, y, w\}$, their embeddings become similar after propagation, so item $z$ is recommended to B.

---

## Lesson 12.10: Putting It Together — GNN Design Principles

### Depth vs Over-smoothing

After $k$ GNN layers, node features average over $k$-hop neighborhoods. If graph diameter $\leq k$, all nodes converge to the same representation.

**Fix:** Use **residual connections** (add original features), **initial residuals** (add input features at each layer), or **Jumping Knowledge Networks** (concatenate all layers' representations).

### Python Code

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Graph Laplacian computation ----
def graph_laplacian(adj_matrix):
    """Compute normalized graph Laplacian."""
    A = np.array(adj_matrix, dtype=float)
    D = np.diag(A.sum(axis=1))
    L = D - A

    # Normalized: D^{-1/2} L D^{-1/2}
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-10))
    L_norm = d_inv_sqrt @ L @ d_inv_sqrt
    return L, L_norm

# Triangle graph: 3 nodes fully connected
adj = [[0,1,1],[1,0,1],[1,1,0]]
L, L_norm = graph_laplacian(adj)
eigenvalues = np.linalg.eigvalsh(L)
print("Laplacian eigenvalues:", eigenvalues.round(3))
# [0, 3, 3] — one zero (connected), two equal

# ---- Simple GCN layer ----
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, X, A_hat):
        """
        X: (n, in_features) node features
        A_hat: (n, n) normalized adjacency with self-loops
        """
        return F.relu(A_hat @ X @ self.linear.weight.T)

def normalize_adj(A):
    """Compute A_hat = D_tilde^{-1/2} * A_tilde * D_tilde^{-1/2}"""
    A_tilde = A + np.eye(len(A))  # Add self-loops
    D_tilde = np.diag(A_tilde.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D_tilde)))
    return D_inv_sqrt @ A_tilde @ D_inv_sqrt

# Karate club graph (simplified: 5 nodes)
A = np.array([[0,1,1,0,0],
              [1,0,1,1,0],
              [1,1,0,0,1],
              [0,1,0,0,1],
              [0,0,1,1,0]], dtype=float)

A_hat = normalize_adj(A)
A_hat_t = torch.tensor(A_hat, dtype=torch.float32)

# Random initial features
X = torch.randn(5, 8)  # 5 nodes, 8-dim features

gcn = GCNLayer(8, 4)
H = gcn(X, A_hat_t)
print(f"GCN output shape: {H.shape}")  # (5, 4)

# ---- Message passing: average neighbor features ----
def message_passing(node_features, adj_matrix, num_steps=2):
    """Simple mean aggregation message passing."""
    H = np.array(node_features, dtype=float)
    A = np.array(adj_matrix, dtype=float)
    D_inv = np.diag(1.0 / (A.sum(axis=1) + 1e-10))
    P = D_inv @ A  # Row-normalized adjacency

    for step in range(num_steps):
        H = P @ H  # Average neighbor features
    return H

X_np = X.detach().numpy()
H_mp = message_passing(X_np, A, num_steps=2)
print(f"Message passing output shape: {H_mp.shape}")

# ---- PageRank ----
def pagerank(A, alpha=0.15, n_iter=100):
    n = len(A)
    D_inv = np.diag(1.0 / (A.sum(axis=1) + 1e-10))
    P = D_inv @ A  # Transition matrix

    r = np.ones(n) / n  # Start uniform
    for _ in range(n_iter):
        r = alpha / n + (1 - alpha) * P.T @ r

    return r / r.sum()

pr = pagerank(A)
print(f"PageRank scores: {pr.round(3)}")
```

---

## Summary

| Concept | Formula/Tool | Application |
|---------|-------------|-------------|
| Graph Laplacian | $\mathbf{L} = \mathbf{D} - \mathbf{A}$ | Smoothness, spectral clustering |
| Spectral clustering | Top-$k$ eigenvectors of $\mathbf{L}$ | Community detection |
| GCN update | $\hat{\mathbf{A}}\mathbf{H}\mathbf{W}$ | Node/graph classification |
| GAT | Attention-weighted neighbor aggregation | Heterogeneous graphs |
| WL expressiveness | MPNNs ≡ 1-WL | Theoretical limits of GNNs |
| Sum pooling | Strictly more expressive than mean/max | Graph-level tasks |
| TransE | $\mathbf{h} + \mathbf{r} \approx \mathbf{t}$ | Knowledge graph completion |
| Over-smoothing | $k$ layers → $k$-hop mean | Pathology of deep GNNs |

---

*← [Part 11: LLM Mathematics](part-11-llm-mathematics.md) | [Part 13: Numerical Methods →](part-13-numerical-methods.md)*
