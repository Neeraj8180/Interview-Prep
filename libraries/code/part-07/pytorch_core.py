# PyTorch core concepts — Part 7 code examples
# PyTorch 2.6.x | Python 3.11+
#
# Covers:
#   1. Tensors and autograd
#   2. Custom nn.Module
#   3. Full training loop on MNIST
#   4. Mixed precision training

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import math

# ── 1. Tensors and basic autograd ─────────────────────────────────────────────

def demo_tensors():
    print("=" * 50)
    print("1. Tensors")
    print("=" * 50)

    # Creating tensors
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.zeros(3, 4)
    z = torch.randn(3, 4)
    print(f"randn(3,4): shape={z.shape}, dtype={z.dtype}")

    # Shape ops
    a = torch.arange(12).reshape(3, 4)
    print(f"arange(12).reshape(3,4):\n{a}")
    print(f"transpose:\n{a.T}")

    # Device movement
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    z = z.to(device)

    # NumPy interop (CPU only, zero-copy)
    import numpy as np
    arr = np.array([1.0, 2.0, 3.0])
    t = torch.from_numpy(arr)   # shares memory
    t[0] = 99.0
    print(f"NumPy array after tensor modification: arr[0]={arr[0]}")   # 99.0


def demo_autograd():
    print("\n" + "=" * 50)
    print("2. Autograd")
    print("=" * 50)

    # Basic gradient computation
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    y = x ** 2       # y = [4, 9]
    z = y.sum()      # z = 13

    z.backward()     # dz/dx = 2x
    print(f"x: {x.data}")
    print(f"x.grad (should be [4, 6]): {x.grad}")   # [2*2, 2*3]

    # Gradient accumulates — zero before each backward
    x.grad.zero_()
    z2 = (x * 3).sum()
    z2.backward()
    print(f"After zero, new grad (should be [3, 3]): {x.grad}")

    # torch.no_grad for inference
    with torch.no_grad():
        out = x ** 2
    print(f"out.requires_grad inside no_grad: {out.requires_grad}")  # False


# ── 2. Custom nn.Module ───────────────────────────────────────────────────────

class MLP(nn.Module):
    """Three-layer MLP for binary classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)   # (batch,)


def demo_module():
    print("\n" + "=" * 50)
    print("3. nn.Module")
    print("=" * 50)

    model = MLP(input_dim=20)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MLP parameters: {n_params:,}")
    print(f"Layers:\n{model}")

    # Forward pass
    x = torch.randn(4, 20)   # batch of 4
    output = model(x)
    print(f"Input: {x.shape} → Output: {output.shape}")

    # Training vs eval mode
    model.train()   # dropout active
    out_train = model(x)
    model.eval()    # dropout inactive
    with torch.no_grad():
        out_eval = model(x)

    print(f"Train output (dropout active): {out_train[:2]}")
    print(f"Eval output (dropout off):     {out_eval[:2]}")


# ── 3. Full training loop ─────────────────────────────────────────────────────

def make_synthetic_data(n_samples: int = 2000, n_features: int = 20):
    """Create a synthetic binary classification problem."""
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=n_samples, n_features=n_features, random_state=42)
    X = StandardScaler().fit_transform(X).astype("float32")
    y = y.astype("float32")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_tr, X_te, y_tr, y_te


def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)    # zero gradients
        logits = model(X_batch)                  # forward pass
        loss   = criterion(logits, y_batch)      # compute loss
        loss.backward()                          # compute gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()                         # update weights
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            probs = torch.sigmoid(model(X_batch.to(device))).cpu()
            all_probs.append(probs)
            all_labels.append(y_batch)
    probs  = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    return probs, labels


def demo_training_loop():
    print("\n" + "=" * 50)
    print("4. Training Loop")
    print("=" * 50)

    try:
        X_tr, X_te, y_tr, y_te = make_synthetic_data()
    except ImportError:
        print("sklearn not available — skipping training demo")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
    test_ds  = TensorDataset(torch.FloatTensor(X_te), torch.FloatTensor(y_te))
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_dl  = DataLoader(test_ds,  batch_size=64)

    model     = MLP(input_dim=20).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, 21):
        loss = train(model, train_dl, optimizer, criterion, device)
        if epoch % 5 == 0:
            probs, labels = evaluate(model, test_dl, device)
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(labels, probs)
            print(f"Epoch {epoch:2d} | train loss: {loss:.4f} | test AUC: {auc:.4f}")

    # Save model
    torch.save(model.state_dict(), "/tmp/mlp_model.pt")
    print("Model saved to /tmp/mlp_model.pt")

    # Load model
    model2 = MLP(input_dim=20).to(device)
    model2.load_state_dict(torch.load("/tmp/mlp_model.pt", weights_only=True))
    model2.eval()
    print("Model loaded and ready for inference")


# ── 4. Mixed precision demo ───────────────────────────────────────────────────

def demo_mixed_precision():
    print("\n" + "=" * 50)
    print("5. Mixed Precision")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("CUDA not available — skipping mixed precision demo")
        return

    device = "cuda"
    model  = MLP(input_dim=20).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    scaler    = torch.cuda.amp.GradScaler()

    X = torch.randn(64, 20, device=device)
    y = torch.randint(0, 2, (64,), device=device).float()

    model.train()
    optimizer.zero_grad()

    # Forward pass in bfloat16 — faster, less memory
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(X)
        loss   = criterion(logits, y)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    print(f"Mixed precision loss: {loss.item():.4f}")
    print("AMP training step completed successfully")


if __name__ == "__main__":
    demo_tensors()
    demo_autograd()
    demo_module()
    demo_training_loop()
    demo_mixed_precision()
