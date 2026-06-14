# MLflow experiment tracking — Part 19 code examples
# mlflow 2.18.x | Python 3.11+
# Install: pip install mlflow scikit-learn

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from mlflow.models import infer_signature


# ── 1. Basic experiment tracking ──────────────────────────────────────────────

def demo_basic_tracking():
    print("=" * 55)
    print("1. MLflow Experiment Tracking")
    print("=" * 55)

    X, y = make_classification(n_samples=800, n_features=15,
                               n_informative=8, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_experiment("sklearn-comparison")

    # Track multiple models
    models = [
        ("Logistic", LogisticRegression(C=1.0, max_iter=500)),
        ("RF-100",   RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)),
        ("RF-200",   RandomForestClassifier(n_estimators=200, max_depth=7, random_state=0)),
    ]

    run_ids = []
    for name, model in models:
        with mlflow.start_run(run_name=name) as run:
            # Log hyperparameters
            mlflow.log_params({
                "model_type": type(model).__name__,
                **{k: v for k, v in model.get_params().items()
                   if isinstance(v, (int, float, str, bool, type(None)))}
            })

            # Train
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            # Log metrics
            acc = accuracy_score(y_test, preds)
            f1  = f1_score(y_test, preds)
            mlflow.log_metrics({"accuracy": acc, "f1_score": f1})

            # Log model with signature
            sig = infer_signature(X_train[:5], model.predict(X_train[:5]))
            mlflow.sklearn.log_model(model, "model", signature=sig)

            run_ids.append(run.info.run_id)
            print(f"  {name:12}: acc={acc:.4f}  f1={f1:.4f}  run_id={run.info.run_id[:8]}")

    # Query best run
    runs_df = mlflow.search_runs(
        experiment_names=["sklearn-comparison"],
        order_by=["metrics.f1_score DESC"],
    )
    best = runs_df.iloc[0]
    print(f"\n  Best: {best['tags.mlflow.runName']} "
          f"f1={best['metrics.f1_score']:.4f}")

    return run_ids


# ── 2. Epoch-level metric logging ─────────────────────────────────────────────

def demo_epoch_logging():
    print("\n" + "=" * 55)
    print("2. Epoch-Level Metric Logging (step parameter)")
    print("=" * 55)

    import torch
    import torch.nn as nn

    torch.manual_seed(42)
    X = torch.randn(200, 8); y = (X[:, 0] + X[:, 1] > 0).long()

    mlflow.set_experiment("pytorch-epoch-tracking")

    with mlflow.start_run(run_name="two-layer-net"):
        mlflow.log_params({"lr": 0.01, "hidden": 32, "epochs": 30})

        model   = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 2))
        opt     = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(30):
            logits = model(X)
            loss   = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()

            acc = (logits.argmax(1) == y).float().mean().item()

            # step= enables time-series view in MLflow UI
            mlflow.log_metrics({
                "train/loss":     loss.item(),
                "train/accuracy": acc,
            }, step=epoch)

            if epoch % 10 == 0:
                print(f"  Epoch {epoch:2d}: loss={loss.item():.4f} acc={acc:.3f}")

        # Log final model
        sample_in  = X[:2].detach().numpy()
        sample_out = model(X[:2]).detach().numpy()
        mlflow.pytorch.log_model(model, "model",
                                  signature=infer_signature(sample_in, sample_out))


# ── 3. MLflow autologging ─────────────────────────────────────────────────────

def demo_autolog():
    print("\n" + "=" * 55)
    print("3. MLflow AutoLog (zero boilerplate)")
    print("=" * 55)

    mlflow.sklearn.autolog(log_input_examples=True)   # one line!

    X, y = make_classification(n_samples=500, n_features=10, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

    mlflow.set_experiment("autolog-demo")
    with mlflow.start_run(run_name="auto-rf"):
        model = RandomForestClassifier(n_estimators=50, random_state=0)
        model.fit(X_tr, y_tr)   # MLflow auto-logs: params, train metrics, model

    runs = mlflow.search_runs(experiment_names=["autolog-demo"])
    print(f"  Auto-logged metrics: {[c for c in runs.columns if c.startswith('metrics.')]}")
    print(f"  Auto-logged params:  {[c for c in runs.columns if c.startswith('params.')]}")


# ── 4. Model registry (requires MLflow tracking server) ──────────────────────

def demo_registry(run_id: str = None):
    print("\n" + "=" * 55)
    print("4. Model Registry")
    print("=" * 55)

    if run_id is None:
        print("  Run a tracking experiment first to get a run_id.")
        print("  Demo code:")
        print("""
  # Register model from a run
  result = mlflow.register_model(
      model_uri=f"runs:/RUN_ID/model",
      name="my-classifier",
  )
  print(f"Version: {result.version}")
  
  # Transition to production
  client = mlflow.tracking.MlflowClient()
  client.transition_model_version_stage(
      name="my-classifier", version=result.version, stage="Production",
  )
  
  # Load latest production
  model = mlflow.sklearn.load_model("models:/my-classifier/Production")
""")
    else:
        try:
            result = mlflow.register_model(f"runs:/{run_id}/model", "demo-classifier")
            print(f"  Registered model version: {result.version}")
        except Exception as e:
            print(f"  Registry operation: {e}")


if __name__ == "__main__":
    # Start MLflow tracking server first for persistent storage:
    # mlflow server --host 0.0.0.0 --port 5000
    # For this demo, use local file store (default)

    run_ids = demo_basic_tracking()
    demo_epoch_logging()
    demo_autolog()
    demo_registry(run_ids[0] if run_ids else None)

    print("\n  Run 'mlflow ui' to see results at http://localhost:5000")
