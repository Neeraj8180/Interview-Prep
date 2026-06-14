# BentoML model packaging and serving — Part 33 code examples
# bentoml 1.4.x | Python 3.11+
# Install: pip install bentoml scikit-learn numpy

import numpy as np


def demo_model_save_load():
    """Save and retrieve a model using BentoML model store."""
    print("=" * 60)
    print("1. BentoML Model Save and Load")
    print("=" * 60)

    try:
        import bentoml
        from sklearn.datasets import make_classification
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import train_test_split

        # Train
        X, y    = make_classification(n_samples=1000, n_features=10, random_state=42)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
        model   = GradientBoostingClassifier(n_estimators=100, random_state=0).fit(X_tr, y_tr)
        acc     = model.score(X_te, y_te)
        print(f"  Test accuracy: {acc:.4f}")

        # Save to BentoML
        saved = bentoml.sklearn.save_model(
            "fraud-detector",
            model,
            signatures={
                "predict":       {"batchable": True, "batch_dim": 0},
                "predict_proba": {"batchable": True, "batch_dim": 0},
            },
            metadata={"accuracy": acc, "n_features": 10, "framework": "sklearn"},
        )
        print(f"  Saved: {saved.tag}")

        # Retrieve
        loaded    = bentoml.sklearn.load_model("fraud-detector:latest")
        sample    = X_te[:5]
        preds     = loaded.predict(sample)
        proba     = loaded.predict_proba(sample)[:, 1]
        print(f"  Predictions:   {preds}")
        print(f"  Probabilities: {proba.round(3)}")

        # List models
        models = bentoml.models.list()
        print(f"\n  Saved models ({len(models)} total):")
        for m in models[:3]:
            print(f"    {m.tag}  metadata={dict(m.info.metadata or {})}")

    except ImportError:
        print("  bentoml not installed: pip install bentoml")
        _show_bentoml_api()


def _show_bentoml_api():
    print("""
  BentoML API pattern:
    import bentoml

    # Save
    bentoml.sklearn.save_model("name", model,
        signatures={"predict": {"batchable": True, "batch_dim": 0}},
        metadata={"accuracy": 0.94})

    # Load
    loaded = bentoml.sklearn.load_model("name:latest")
    loaded.predict(X)

    # As runner for service
    runner = bentoml.sklearn.get("name:latest").to_runner()
    svc = bentoml.Service("api", runners=[runner])

    @svc.api(input=bentoml.io.NumpyNdarray(), output=bentoml.io.NumpyNdarray())
    async def predict(X: np.ndarray) -> np.ndarray:
        return await runner.predict.async_run(X)

    # Serve: bentoml serve service:svc
    # Build: bentoml build
    # Containerize: bentoml containerize name:version
""")


def demo_service_code():
    """Show what a BentoML service looks like."""
    print("\n" + "=" * 60)
    print("2. BentoML Service Code (service.py)")
    print("=" * 60)

    service_code = '''# service.py
import bentoml
import numpy as np
from bentoml.io import NumpyNdarray, JSON
from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    features: List[List[float]]

class PredictResponse(BaseModel):
    predictions: List[int]
    probabilities: List[float]

# Load model runner (scales independently from web server)
runner = bentoml.sklearn.get("fraud-detector:latest").to_runner()

svc = bentoml.Service("fraud-detector-api", runners=[runner])

@svc.api(
    input=JSON(pydantic_model=PredictRequest),
    output=JSON(pydantic_model=PredictResponse),
    route="/predict",
)
async def predict(request: PredictRequest) -> PredictResponse:
    """Async inference — non-blocking, handles concurrent requests."""
    X       = np.array(request.features, dtype=np.float32)
    preds   = await runner.predict.async_run(X)
    proba   = await runner.predict_proba.async_run(X)
    return PredictResponse(
        predictions=preds.tolist(),
        probabilities=proba[:, 1].tolist(),
    )
'''
    print(service_code)
    print("  Start server: bentoml serve service:svc --port 3000")
    print("  Test:         curl -X POST http://localhost:3000/predict \\")
    print('                  -H "Content-Type: application/json" \\')
    print('                  -d \'{"features": [[0.1, 0.2, ...10 features...]]}\' ')


def demo_batching_config():
    """Demonstrate BentoML batching configuration."""
    print("\n" + "=" * 60)
    print("3. Adaptive Batching for GPU Efficiency")
    print("=" * 60)

    print("""
  Without batching:
    Request 1 → model_forward(X1)   → 15ms latency
    Request 2 → model_forward(X2)   → 15ms latency
    Request 3 → model_forward(X3)   → 15ms latency
    Total: 45ms for 3 requests

  With BentoML adaptive batching (batchable=True):
    Request 1, 2, 3 arrive → collected into batch
    model_forward([X1, X2, X3])     → 20ms latency
    All 3 respond simultaneously
    Total: 20ms for 3 requests (2.25× faster, GPU utilization: 3×)

  Configuration:
    bentoml.sklearn.save_model("name", model, signatures={
        "predict": {
            "batchable":     True,
            "batch_dim":     0,          # batch along dimension 0
            "max_batch_size": 64,        # max batch before forced dispatch
            "max_latency_ms": 50,        # max wait time to collect batch
        }
    })
""")


def demo_pytorch_bentoml():
    """Save and serve a PyTorch model."""
    print("\n" + "=" * 60)
    print("4. PyTorch Model with BentoML")
    print("=" * 60)

    try:
        import bentoml
        import torch
        import torch.nn as nn

        class TinyClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2)
                )
            def forward(self, x): return self.net(x)

        model = TinyClassifier()
        # Train briefly
        X = torch.randn(100, 10)
        y = torch.randint(0, 2, (100,))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(20):
            loss = nn.CrossEntropyLoss()(model(X), y)
            opt.zero_grad(); loss.backward(); opt.step()

        # Save to BentoML
        saved = bentoml.pytorch.save_model(
            "tiny-classifier",
            model,
            signatures={"__call__": {"batchable": True, "batch_dim": 0}},
        )
        print(f"  Saved PyTorch model: {saved.tag}")

        # Load and run
        loaded_model = bentoml.pytorch.load_model("tiny-classifier:latest")
        sample  = torch.randn(3, 10)
        with torch.no_grad():
            logits = loaded_model(sample)
        print(f"  Logits shape: {logits.shape}")
        print(f"  Predictions: {logits.argmax(1).tolist()}")

    except ImportError as e:
        print(f"  {e}")


if __name__ == "__main__":
    demo_model_save_load()
    demo_service_code()
    demo_batching_config()
    demo_pytorch_bentoml()
