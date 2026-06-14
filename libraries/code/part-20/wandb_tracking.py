# Weights & Biases (wandb) — Part 20 code examples
# wandb 0.18.x | Python 3.11+
# Install: pip install wandb
# For offline use (no account needed): set mode="offline"

import os
import time
import numpy as np

# Use offline mode so examples run without W&B account
os.environ.setdefault("WANDB_MODE", "offline")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed. Run: pip install wandb")


# ── 1. Basic run with metric logging ─────────────────────────────────────────

def demo_basic_run():
    if not WANDB_AVAILABLE:
        print("Skipping W&B demo — not installed"); return

    print("=" * 55)
    print("1. Basic W&B Run (offline mode)")
    print("=" * 55)

    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    X = torch.randn(150, 4); y = (X[:, 0] > 0).long()

    with wandb.init(
        project="wandb-demo",
        name="basic-mlp",
        config={"lr": 0.01, "hidden": 32, "epochs": 50},
        mode="offline",   # no upload — local only
    ) as run:
        cfg = wandb.config
        print(f"  Run ID: {run.id}")

        model   = nn.Sequential(nn.Linear(4, cfg.hidden), nn.ReLU(), nn.Linear(cfg.hidden, 2))
        opt     = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(cfg.epochs):
            logits = model(X)
            loss   = loss_fn(logits, y)
            acc    = (logits.argmax(1) == y).float().mean().item()

            opt.zero_grad(); loss.backward(); opt.step()

            # Log every step
            run.log({"train/loss": loss.item(), "train/accuracy": acc}, step=epoch)

            if epoch % 10 == 0:
                print(f"  Epoch {epoch:2d}: loss={loss.item():.4f} acc={acc:.3f}")

    print("  Offline run saved. Upload with: wandb sync ./wandb/")


# ── 2. Sweep: automated hyperparameter search ─────────────────────────────────

def demo_sweep():
    if not WANDB_AVAILABLE:
        print("Skipping sweep demo — wandb not installed"); return

    print("\n" + "=" * 55)
    print("2. W&B Hyperparameter Sweep (random search)")
    print("=" * 55)

    from sklearn.datasets import make_moons
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import cross_val_score

    X, y = make_moons(n_samples=300, noise=0.15, random_state=42)

    sweep_config = {
        "method": "random",
        "metric": {"name": "val_accuracy", "goal": "maximize"},
        "parameters": {
            "lr":     {"min": 0.001, "max": 0.1},
            "hidden": {"values": [16, 32, 64]},
        },
    }

    def train():
        with wandb.init(mode="offline") as run:
            config = wandb.config
            model  = MLPClassifier(
                hidden_layer_sizes=(config.hidden,),
                learning_rate_init=config.lr,
                max_iter=200, random_state=0,
            )
            scores = cross_val_score(model, X, y, cv=3)
            run.log({"val_accuracy": float(scores.mean())})
            print(f"    lr={config.lr:.4f} hidden={config.hidden}: "
                  f"acc={scores.mean():.3f}")

    sweep_id = wandb.sweep(sweep_config, project="demo-sweep")
    print(f"  Sweep ID: {sweep_id}")
    wandb.agent(sweep_id, function=train, count=6)   # run 6 trials
    print("  Sweep complete. Best config visible in W&B dashboard after sync.")


# ── 3. Artifact versioning ────────────────────────────────────────────────────

def demo_artifacts():
    if not WANDB_AVAILABLE:
        print("Skipping artifacts demo — wandb not installed"); return

    print("\n" + "=" * 55)
    print("3. W&B Artifacts (versioning)")
    print("=" * 55)

    import tempfile, json, pathlib

    with wandb.init(project="artifacts-demo", name="dataset-v1", mode="offline") as run:
        # Create a temp "dataset"
        tmpdir = pathlib.Path(tempfile.mkdtemp())
        (tmpdir / "train.json").write_text(json.dumps([{"text": "hello", "label": 1}]))
        (tmpdir / "val.json").write_text(json.dumps([{"text": "world", "label": 0}]))

        # Log artifact
        artifact = wandb.Artifact(
            name="instruction-data",
            type="dataset",
            description="Sample instruction tuning data",
            metadata={"num_train": 1, "num_val": 1, "version": "v1"},
        )
        artifact.add_dir(str(tmpdir))
        run.log_artifact(artifact)
        print(f"  Artifact logged: instruction-data (type=dataset)")

    with wandb.init(project="artifacts-demo", name="model-v1", mode="offline") as run:
        artifact = wandb.Artifact(
            name="tiny-model",
            type="model",
            metadata={"accuracy": 0.92, "trained_on": "instruction-data:v0"},
        )
        # In production: artifact.add_file("./model.pkl")
        # For demo, create a dummy file
        tmpfile = pathlib.Path(tempfile.mktemp(suffix=".txt"))
        tmpfile.write_text("model weights placeholder")
        artifact.add_file(str(tmpfile))
        run.log_artifact(artifact)
        print(f"  Artifact logged: tiny-model (type=model)")


# ── 4. W&B vs MLflow comparison ──────────────────────────────────────────────

def print_comparison():
    print("\n" + "=" * 55)
    print("W&B vs MLflow — Practical Comparison")
    print("=" * 55)
    rows = [
        ("Real-time dashboard",    "✓ Cloud, live",         "✓ Local UI only"),
        ("Team collaboration",     "✓ Shared workspace",    "✗ Shared server needed"),
        ("Hyperparameter sweeps",  "✓ Bayesian/ASHA native","✓ (via MLflow + Optuna)"),
        ("Model registry",         "✓ W&B Models",          "✓ MLflow Registry (mature)"),
        ("On-premises support",    "✓ W&B Server (paid)",   "✓ Free self-hosted"),
        ("LLM tracing",            "✓ Weave",               "✓ MLflow AI Gateway"),
        ("HuggingFace integration","✓ report_to='wandb'",   "✓ report_to='mlflow'"),
        ("Cost",                   "Free for individuals",   "Free (Apache 2.0)"),
        ("Enterprise adoption",    "OpenAI, StabilityAI",   "Databricks, Azure ML"),
    ]
    print(f"  {'Feature':<30} {'W&B':>20} {'MLflow':>20}")
    print("  " + "-" * 70)
    for feature, wb, mlf in rows:
        print(f"  {feature:<30} {wb:>20} {mlf:>20}")


if __name__ == "__main__":
    demo_basic_run()
    demo_artifacts()
    print_comparison()

    # Sweep demo is optional (takes longer)
    # demo_sweep()

    print("\n  To view offline runs in W&B: wandb sync ./wandb/")
    print("  To use W&B with HuggingFace Trainer:")
    print("    os.environ['WANDB_PROJECT'] = 'my-project'")
    print("    TrainingArguments(..., report_to='wandb')")
