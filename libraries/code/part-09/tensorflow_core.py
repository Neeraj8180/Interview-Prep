# TensorFlow / Keras — Part 9 code examples
# TensorFlow 2.18 / Keras 3.x | Python 3.11+
#
# Covers:
#   1. Tensors and Variables
#   2. GradientTape — manual gradient computation
#   3. Sequential and Functional API models
#   4. compile + fit training
#   5. Custom training loop with tf.function
#   6. tf.data pipelines

import tensorflow as tf
import keras
from keras import layers
import numpy as np


# ── 1. Tensors and Variables ──────────────────────────────────────────────────

def demo_tensors():
    print("=" * 50)
    print("1. TensorFlow Tensors and Variables")
    print("=" * 50)

    # Constants are immutable
    x = tf.constant([1.0, 2.0, 3.0])
    y = tf.zeros((3, 4))
    z = tf.random.normal((3, 4))
    print(f"x: {x.numpy()}")
    print(f"z shape: {z.shape}, dtype: {z.dtype}")

    # Operations return new tensors (eager)
    result = x * 2 + 1
    print(f"x*2+1: {result.numpy()}")

    # Variables are mutable — use for model parameters
    v = tf.Variable([1.0, 2.0, 3.0])
    v.assign(tf.zeros(3))
    print(f"After assign: {v.numpy()}")
    v.assign_add([1.0, 2.0, 3.0])
    print(f"After assign_add: {v.numpy()}")

    # NumPy interop
    arr = np.array([4.0, 5.0])
    t = tf.constant(arr)
    back = t.numpy()
    print(f"Round-trip numpy: {back}")


# ── 2. GradientTape ───────────────────────────────────────────────────────────

def demo_gradient_tape():
    print("\n" + "=" * 50)
    print("2. GradientTape — manual gradients")
    print("=" * 50)

    # Basic gradient
    x = tf.Variable([2.0, 3.0])
    with tf.GradientTape() as tape:
        y = tf.reduce_sum(x ** 2)   # y = 4+9 = 13

    grad = tape.gradient(y, x)
    print(f"dy/dx (should be [4,6]): {grad.numpy()}")

    # Gradient descent on f(x) = (x-3)^2, minimum at x=3
    x = tf.Variable(10.0)
    for step in range(50):
        with tf.GradientTape() as tape:
            loss = (x - 3.0) ** 2
        g = tape.gradient(loss, x)
        x.assign_sub(0.3 * g)   # x -= 0.3 * gradient
        if step % 15 == 0:
            print(f"  Step {step}: x={x.numpy():.4f}, loss={loss.numpy():.4f}")
    print(f"Final x (should be ~3.0): {x.numpy():.4f}")


# ── 3. Sequential API ─────────────────────────────────────────────────────────

def build_sequential_model():
    model = keras.Sequential([
        layers.Dense(64, activation="relu", input_shape=(20,)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ], name="binary_clf")
    return model


def demo_sequential():
    print("\n" + "=" * 50)
    print("3. Sequential API")
    print("=" * 50)

    model = build_sequential_model()
    model.summary()

    n_params = sum(np.prod(v.shape) for v in model.trainable_variables)
    print(f"\nTrainable parameters: {n_params:,}")


# ── 4. compile + fit ──────────────────────────────────────────────────────────

def demo_compile_fit():
    print("\n" + "=" * 50)
    print("4. compile + fit (batteries-included training)")
    print("=" * 50)

    try:
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("sklearn not available — skipping")
        return

    X, y = make_classification(n_samples=3000, n_features=20, random_state=42)
    X = StandardScaler().fit_transform(X).astype("float32")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = build_sequential_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )

    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=64,
        validation_split=0.15,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, verbose=1),
        ],
        verbose=0,
    )

    loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test — loss: {loss:.4f}, accuracy: {acc:.4f}, AUC: {auc:.4f}")
    print(f"Trained for {len(history.history['loss'])} epochs")

    # Predictions
    probs = model.predict(X_test[:5], verbose=0)
    print(f"Sample predictions: {probs.flatten()}")


# ── 5. Custom training loop with tf.function ──────────────────────────────────

def demo_custom_loop():
    print("\n" + "=" * 50)
    print("5. Custom training loop with @tf.function")
    print("=" * 50)

    # Simple regression model
    inputs  = keras.Input(shape=(10,))
    x       = layers.Dense(32, activation="relu")(inputs)
    outputs = layers.Dense(1)(x)
    model   = keras.Model(inputs, outputs, name="regressor")

    optimizer = keras.optimizers.Adam(1e-3)
    loss_fn   = keras.losses.MeanSquaredError()
    train_metric = keras.metrics.MeanAbsoluteError()

    @tf.function   # compile to graph on first call
    def train_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            preds = model(x_batch, training=True)
            loss  = loss_fn(y_batch, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_metric.update_state(y_batch, preds)
        return loss

    # Data: y = sum(X) + noise
    rng = np.random.default_rng(0)
    X   = rng.normal(size=(500, 10)).astype("float32")
    y   = X.sum(axis=1, keepdims=True).astype("float32") + rng.normal(0, 0.1, (500, 1)).astype("float32")

    dataset = (
        tf.data.Dataset.from_tensor_slices((X, y))
        .shuffle(200)
        .batch(32)
        .prefetch(tf.data.AUTOTUNE)
    )

    for epoch in range(10):
        train_metric.reset_state()
        for x_batch, y_batch in dataset:
            loss = train_step(x_batch, y_batch)
        if epoch % 3 == 0:
            print(f"  Epoch {epoch+1:2d}: loss={loss:.4f}, MAE={train_metric.result():.4f}")


# ── 6. tf.data pipelines ─────────────────────────────────────────────────────

def demo_tf_data():
    print("\n" + "=" * 50)
    print("6. tf.data pipeline")
    print("=" * 50)

    n = 1000
    X = np.random.randn(n, 20).astype("float32")
    y = np.random.randint(0, 2, n).astype("int32")

    ds = (
        tf.data.Dataset.from_tensor_slices((X, y))
        .shuffle(buffer_size=500, reshuffle_each_iteration=True)
        .batch(64)
        .prefetch(tf.data.AUTOTUNE)
        .cache()         # cache entire dataset in memory
    )

    # Count batches
    n_batches = sum(1 for _ in ds)
    print(f"Dataset: {n} samples → {n_batches} batches of 64")

    # Inspect first batch
    for x_batch, y_batch in ds.take(1):
        print(f"Batch shape: x={x_batch.shape}, y={y_batch.shape}")


if __name__ == "__main__":
    demo_tensors()
    demo_gradient_tape()
    demo_sequential()
    demo_compile_fit()
    demo_custom_loop()
    demo_tf_data()
