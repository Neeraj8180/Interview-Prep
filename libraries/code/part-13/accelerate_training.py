# Accelerate — Part 13 code examples
# accelerate 0.35+ | PyTorch 2.6+ | Python 3.11+
#
# Covers:
#   1. Single-GPU → multi-GPU conversion (the 4-line change)
#   2. Gradient accumulation with accelerator.accumulate()
#   3. Mixed precision training
#   4. Checkpoint save/load
#
# To run on multiple GPUs:
#   accelerate launch --num_processes 4 accelerate_training.py
# To run on single GPU:
#   python accelerate_training.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator


# ── Model and data factory ────────────────────────────────────────────────────

def make_model():
    return nn.Sequential(
        nn.Linear(20, 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )


def make_data(n=500, dim=20):
    torch.manual_seed(42)
    X = torch.randn(n, dim)
    y = X.sum(dim=1, keepdim=True) + 0.1 * torch.randn(n, 1)   # y ≈ sum(X)
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=32, shuffle=True)


# ── Before Accelerate (single GPU only) ──────────────────────────────────────

def train_single_gpu():
    """Standard PyTorch training — single GPU only."""
    print("=" * 50)
    print("Training WITHOUT Accelerate (single GPU)")
    print("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = make_model().to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dl     = make_data()
    crit   = nn.MSELoss()

    for epoch in range(3):
        for X, y in dl:
            X, y = X.to(device), y.to(device)   # manual device placement
            opt.zero_grad()
            loss = crit(model(X), y)
            loss.backward()
            opt.step()
        print(f"  Epoch {epoch+1}: loss={loss.item():.4f}")


# ── After Accelerate (works on 1, 4, 8 GPUs, TPU) ────────────────────────────

def train_with_accelerate():
    """Same logic — works on any number of GPUs/TPUs with 4 changes."""
    print("\n" + "=" * 50)
    print("Training WITH Accelerate (any hardware)")
    print("=" * 50)

    # CHANGE 1: Create Accelerator
    accelerator = Accelerator(
        mixed_precision="bf16",          # bfloat16 on A100/H100
        gradient_accumulation_steps=2,   # accumulate 2 batches before update
    )

    model = make_model()
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dl    = make_data()
    crit  = nn.MSELoss()

    # CHANGE 2: prepare() wraps model (DDP/FSDP), dataloader (DistributedSampler)
    model, opt, dl = accelerator.prepare(model, opt, dl)

    for epoch in range(3):
        model.train()
        epoch_loss = 0.0

        for X, y in dl:
            # No .to(device) needed — Accelerate handles device placement

            with accelerator.accumulate(model):   # handles grad accum correctly
                loss = crit(model(X), y)
                # CHANGE 3: accelerator.backward instead of loss.backward()
                accelerator.backward(loss)
                opt.step()
                opt.zero_grad()

            epoch_loss += loss.item()

        # CHANGE 4: is_main_process guards logging/saving
        if accelerator.is_main_process:
            print(f"  Epoch {epoch+1}: loss={epoch_loss/len(dl):.4f}")

    # Checkpoint
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        torch.save(unwrapped.state_dict(), "/tmp/accelerate_model.pt")
        print("Checkpoint saved to /tmp/accelerate_model.pt")
        print(f"Device: {accelerator.device} | "
              f"Processes: {accelerator.num_processes} | "
              f"Mixed precision: {accelerator.mixed_precision}")


# ── Gradient accumulation demo ────────────────────────────────────────────────

def demo_gradient_accumulation():
    """Show why accelerator.accumulate() is better than dividing loss."""
    print("\n" + "=" * 50)
    print("Gradient Accumulation Demo")
    print("=" * 50)

    accelerator = Accelerator(gradient_accumulation_steps=4)
    model = nn.Linear(5, 1)
    opt   = torch.optim.SGD(model.parameters(), lr=0.01)
    model, opt = accelerator.prepare(model, opt)

    X = torch.randn(4, 5)
    y = torch.randn(4, 1)
    from torch.utils.data import TensorDataset
    dl = DataLoader(TensorDataset(X, y), batch_size=1)
    dl = accelerator.prepare(dl)

    # With accumulate() context — correct DDP behavior (no_sync on accum steps)
    opt.zero_grad()
    for step, (x_b, y_b) in enumerate(dl):
        with accelerator.accumulate(model):
            loss = nn.MSELoss()(model(x_b), y_b)
            accelerator.backward(loss)
            opt.step()
            opt.zero_grad()
        print(f"  Step {step}: loss={loss.item():.4f} | "
              f"sync={'yes' if (step+1) % 4 == 0 else 'no'}")


if __name__ == "__main__":
    train_single_gpu()
    train_with_accelerate()
    demo_gradient_accumulation()
