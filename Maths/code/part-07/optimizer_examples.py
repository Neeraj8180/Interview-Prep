"""
Part 7: Optimization — Key Optimizer Implementations
Run: python part-07/optimizer_examples.py
"""

import numpy as np
import torch
import torch.nn as nn

print("=" * 60)
print("Manual Adam Implementation")
print("=" * 60)

class ManualAdam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.lr, self.betas, self.eps = lr, betas, eps
        self.params = list(params)
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        b1, b2 = self.betas
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g**2
            m_hat = self.m[i] / (1 - b1**self.t)
            v_hat = self.v[i] / (1 - b2**self.t)
            p.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


torch.manual_seed(42)
model = nn.Linear(10, 1)
opt = ManualAdam(model.parameters(), lr=0.01)

losses = []
for step in range(200):
    x = torch.randn(32, 10)
    y = x.sum(dim=1, keepdim=True)
    loss = ((model(x) - y)**2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())

print(f"Initial loss: {losses[0]:.4f}")
print(f"Final loss: {losses[-1]:.6f}")
assert losses[-1] < losses[0] * 0.01, "Optimizer not converging!"
print("Adam converges correctly.")

print("\n" + "=" * 60)
print("LR Schedule: Warmup + Cosine Decay")
print("=" * 60)

import math

def warmup_cosine_decay(step, warmup_steps=1000, total_steps=10000, min_lr=0.0):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (1.0 - min_lr) * (1 + math.cos(math.pi * progress))

steps = list(range(0, 10001, 100))
lrs = [warmup_cosine_decay(s) * 1e-3 for s in steps]
print(f"LR at step 0: {lrs[0]:.6f}")
print(f"LR at step 1000 (end warmup): {lrs[10]:.6f}")
print(f"LR at step 5000 (mid decay): {lrs[50]:.6f}")
print(f"LR at step 10000 (end): {lrs[100]:.6f}")

print("\n" + "=" * 60)
print("Gradient Clipping Demo")
print("=" * 60)

model2 = nn.LSTM(32, 64, num_layers=4, batch_first=True)
x = torch.randn(8, 50, 32)
h = model2(x)[0]
loss = h.sum()
loss.backward()

total_norm_before = 0
for p in model2.parameters():
    if p.grad is not None:
        total_norm_before += p.grad.norm()**2
total_norm_before = total_norm_before**0.5
print(f"Gradient norm before clipping: {total_norm_before:.4f}")

clip_value = 1.0
clipped_norm = nn.utils.clip_grad_norm_(model2.parameters(), max_norm=clip_value)
print(f"Gradient norm after clipping: {min(total_norm_before.item(), clip_value):.4f}")
print(f"Actual clipped norm: {clipped_norm:.4f}")

print("\n" + "=" * 60)
print("BatchNorm vs LayerNorm Comparison")
print("=" * 60)

x = torch.randn(32, 128)    # batch_size=32, features=128

bn = nn.BatchNorm1d(128)
ln = nn.LayerNorm(128)

x_bn = bn(x)
x_ln = ln(x)

print(f"BatchNorm — mean: {x_bn.mean().item():.4f}, std: {x_bn.std().item():.4f}")
print(f"LayerNorm — mean: {x_ln.mean().item():.4f}, std: {x_ln.std().item():.4f}")

print("\nAll Part 7 examples ran successfully!")
