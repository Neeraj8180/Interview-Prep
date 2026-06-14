# Ray distributed computing — Part 18 code examples
# ray 2.40.x | Python 3.11+
# Install: pip install "ray[default,tune,train]"

import time
import numpy as np

try:
    import ray
    from ray import tune
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("Ray not installed. Run: pip install 'ray[default,tune,train]'")


# ── 1. Basic remote tasks ─────────────────────────────────────────────────────

def demo_parallel_tasks():
    if not RAY_AVAILABLE:
        print("Skipping task demo — Ray not installed"); return

    ray.init(ignore_reinit_error=True, num_cpus=4)
    print("=" * 50)
    print("1. Parallel Remote Tasks")
    print("=" * 50)

    @ray.remote
    def cpu_work(n: int) -> float:
        """Simulate CPU-intensive work (e.g., fold training)."""
        total = sum(i ** 2 for i in range(n))
        return float(total)

    # Sequential
    t0 = time.time()
    seq_results = [cpu_work.remote(500_000) for _ in range(4)]
    seq_results = ray.get(seq_results)  # wait for ALL
    t_parallel = time.time() - t0

    print(f"  4 tasks in parallel: {t_parallel:.2f}s")
    print(f"  Results: {[f'{r:.0f}' for r in seq_results]}")


# ── 2. Remote actors ──────────────────────────────────────────────────────────

def demo_actors():
    if not RAY_AVAILABLE:
        print("Skipping actor demo — Ray not installed"); return

    ray.init(ignore_reinit_error=True)
    print("\n" + "=" * 50)
    print("2. Remote Actors (stateful workers)")
    print("=" * 50)

    @ray.remote
    class ParameterServer:
        """Maintains model parameters — actors keep state between calls."""
        def __init__(self, n_params: int):
            self.params = np.zeros(n_params)
            self.updates = 0

        def apply_gradient(self, grad: np.ndarray):
            self.params -= 0.01 * grad
            self.updates += 1

        def get_params(self) -> np.ndarray:
            return self.params.copy()

        def get_updates(self) -> int:
            return self.updates

    ps = ParameterServer.remote(100)

    # Multiple workers send gradients
    grads = [np.random.randn(100) for _ in range(10)]
    for grad in grads:
        ray.get(ps.apply_gradient.remote(grad))

    params  = ray.get(ps.get_params.remote())
    updates = ray.get(ps.get_updates.remote())
    print(f"  Applied {updates} gradient updates")
    print(f"  Param mean: {params.mean():.4f} (non-zero = updates applied)")


# ── 3. Object store (put/get for large data) ──────────────────────────────────

def demo_object_store():
    if not RAY_AVAILABLE:
        print("Skipping object store demo — Ray not installed"); return

    ray.init(ignore_reinit_error=True)
    print("\n" + "=" * 50)
    print("3. Distributed Object Store")
    print("=" * 50)

    large_data = np.random.randn(1_000_000)   # 8 MB array

    @ray.remote
    def process_chunk(data_ref, start: int, end: int) -> float:
        data = ray.get(data_ref)             # zero-copy read within same node
        return float(data[start:end].mean())

    # Store large array ONCE in object store
    data_ref = ray.put(large_data)

    # All tasks share the same object store entry — no copying
    n_chunks = 8
    chunk_size = len(large_data) // n_chunks
    refs = [process_chunk.remote(data_ref, i * chunk_size, (i+1) * chunk_size)
            for i in range(n_chunks)]

    results = ray.get(refs)
    print(f"  Chunk means: {[f'{r:.4f}' for r in results]}")
    print(f"  Grand mean:  {np.mean(results):.4f}")
    print(f"  Expected:    {large_data.mean():.4f}")


# ── 4. Ray Tune hyperparameter search ─────────────────────────────────────────

def demo_tune():
    if not RAY_AVAILABLE:
        print("Skipping Tune demo — Ray not installed"); return

    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from sklearn.datasets import make_classification
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import cross_val_score

    print("\n" + "=" * 50)
    print("4. Ray Tune Hyperparameter Search")
    print("=" * 50)

    X, y = make_classification(n_samples=300, n_features=10, random_state=42)

    def train_mlp(config: dict):
        model = MLPClassifier(
            hidden_layer_sizes=(config["hidden"],),
            learning_rate_init=config["lr"],
            max_iter=100, random_state=0,
        )
        scores = cross_val_score(model, X, y, cv=3, scoring="accuracy")
        tune.report({"val_accuracy": float(scores.mean())})

    tuner = tune.Tuner(
        train_mlp,
        param_space={
            "lr":     tune.loguniform(1e-4, 1e-1),
            "hidden": tune.choice([16, 32, 64, 128]),
        },
        tune_config=tune.TuneConfig(
            metric="val_accuracy",
            mode="max",
            num_samples=16,
        ),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="val_accuracy", mode="max")
    print(f"  Best config: lr={best.config['lr']:.5f}, "
          f"hidden={best.config['hidden']}")
    print(f"  Best val accuracy: {best.metrics['val_accuracy']:.4f}")


if __name__ == "__main__":
    demo_parallel_tasks()
    demo_actors()
    demo_object_store()
    demo_tune()

    if RAY_AVAILABLE:
        ray.shutdown()
