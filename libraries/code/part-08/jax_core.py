# JAX core concepts — Part 8 code examples
# JAX 0.4.30+ | Python 3.11+
#
# Covers:
#   1. Immutable arrays and .at[].set()
#   2. jax.grad and value_and_grad
#   3. jax.jit compilation and tracing
#   4. jax.vmap — vectorization
#   5. Manual training loop (no Flax/Equinox)

import jax
import jax.numpy as jnp
import time


# ── 1. Immutable arrays ───────────────────────────────────────────────────────

def demo_arrays():
    print("=" * 50)
    print("1. JAX Arrays (immutable)")
    print("=" * 50)

    # Basic creation
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.zeros((3, 4))
    z = jnp.arange(12.0).reshape(3, 4)
    print(f"z:\n{z}")

    # Immutable — no in-place modification
    try:
        z[0, 0] = 99.0
    except TypeError as e:
        print(f"Cannot modify in-place: {type(e).__name__}")

    # Use .at[].set() for functional updates
    z_new = z.at[0, 0].set(99.0)
    print(f"Original z[0,0]: {z[0,0]}")   # unchanged
    print(f"New      z[0,0]: {z_new[0,0]}")

    # Other .at[] operations
    x = jnp.zeros(5)
    x = x.at[2].set(1.0)      # set
    x = x.at[1:3].add(0.5)    # add
    x = x.at[0].mul(10.0)     # multiply
    print(f"After functional updates: {x}")


# ── 2. PRNG and randomness ────────────────────────────────────────────────────

def demo_random():
    print("\n" + "=" * 50)
    print("2. Explicit PRNG (key splitting)")
    print("=" * 50)

    # Keys must be split for independent random streams
    key = jax.random.key(42)

    # Wrong: reusing same key gives same values
    x1 = jax.random.normal(key, (3,))
    x2 = jax.random.normal(key, (3,))
    print(f"Same key: identical? {jnp.allclose(x1, x2)}")   # True

    # Correct: split key for each use
    key, subkey1 = jax.random.split(key)
    key, subkey2 = jax.random.split(key)
    x1 = jax.random.normal(subkey1, (3,))
    x2 = jax.random.normal(subkey2, (3,))
    print(f"Split keys: identical? {jnp.allclose(x1, x2)}")  # False

    # Generate multiple keys at once
    keys = jax.random.split(key, num=5)
    samples = [jax.random.uniform(k, (2,)) for k in keys]
    print(f"5 independent samples created")


# ── 3. jax.grad ───────────────────────────────────────────────────────────────

def demo_grad():
    print("\n" + "=" * 50)
    print("3. jax.grad — automatic differentiation")
    print("=" * 50)

    # Basic gradient
    def f(x):
        return jnp.sum(x ** 2)

    df = jax.grad(f)                    # returns gradient function
    x  = jnp.array([2.0, 3.0])
    print(f"f(x)    = {f(x)}")         # 13.0
    print(f"df/dx   = {df(x)}")        # [4.0, 6.0]

    # value_and_grad: one forward pass for both value and gradient
    val, grad = jax.value_and_grad(f)(x)
    print(f"val={val}, grad={grad}")

    # Gradient descent on f(x) = (x-2)^2 → minimum at x=2
    def objective(x):
        return (x - 2.0) ** 2

    x = jnp.float32(10.0)
    grad_obj = jax.grad(objective)

    for step in range(30):
        g = grad_obj(x)
        x = x - 0.3 * g
        if step % 10 == 0:
            print(f"  step {step}: x={x:.4f}, loss={objective(x):.4f}")

    print(f"Final x (should be ~2.0): {x:.4f}")

    # Differentiating w.r.t. nested pytree
    def linear_loss(params, X, y):
        pred = X @ params["w"] + params["b"]
        return jnp.mean((pred - y) ** 2)

    params = {"w": jnp.ones(4), "b": jnp.zeros(1)}
    X = jnp.ones((8, 4))
    y = jnp.ones(8) * 2.0

    grads = jax.grad(linear_loss)(params, X, y)
    print(f"Gradient pytree keys: {list(grads.keys())}")
    print(f"Grad w shape: {grads['w'].shape}, Grad b shape: {grads['b'].shape}")


# ── 4. jax.jit ────────────────────────────────────────────────────────────────

def demo_jit():
    print("\n" + "=" * 50)
    print("4. jax.jit — XLA compilation")
    print("=" * 50)

    def matmul_chain(A, B, C):
        return A @ B @ C

    fast_fn = jax.jit(matmul_chain)

    key = jax.random.key(0)
    A = jax.random.normal(key, (256, 256))
    B = jax.random.normal(key, (256, 256))
    C = jax.random.normal(key, (256, 256))

    # First call: trace + compile (slower)
    t0 = time.time()
    r = fast_fn(A, B, C)
    r.block_until_ready()
    print(f"First call (includes trace+compile): {(time.time()-t0)*1000:.1f}ms")

    # Second call: compiled code (faster)
    t0 = time.time()
    r = fast_fn(A, B, C)
    r.block_until_ready()
    print(f"Second call (compiled): {(time.time()-t0)*1000:.1f}ms")

    # Avoid Python control flow on values inside jit
    @jax.jit
    def safe_relu(x):
        return jnp.where(x > 0, x, jnp.zeros_like(x))  # not: if x > 0

    x = jnp.array([-1.0, 2.0, -3.0, 4.0])
    print(f"safe_relu({x}): {safe_relu(x)}")


# ── 5. jax.vmap ───────────────────────────────────────────────────────────────

def demo_vmap():
    print("\n" + "=" * 50)
    print("5. jax.vmap — vectorization")
    print("=" * 50)

    # Write function for a single example
    def single_predict(w, x):
        return jnp.dot(w, x)

    # Batch over x axis 0 (w is shared — None means don't vectorize)
    batch_predict = jax.vmap(single_predict, in_axes=(None, 0))

    w = jnp.ones(8)
    X = jnp.ones((32, 8))
    preds = batch_predict(w, X)
    print(f"Input X: {X.shape} | Predictions: {preds.shape}")  # (32,)

    # Per-sample gradients using vmap(grad(f))
    def single_loss(w, x, y):
        pred = jnp.dot(w, x)
        return (pred - y) ** 2

    per_sample_grad = jax.vmap(jax.grad(single_loss), in_axes=(None, 0, 0))

    y = jnp.ones(32)
    grads = per_sample_grad(w, X, y)
    print(f"Per-sample gradient shape: {grads.shape}")   # (32, 8)
    print(f"Average grad equals batch grad: "
          f"{jnp.allclose(grads.mean(0), jax.grad(lambda w: jnp.mean(jax.vmap(single_loss, in_axes=(None,0,0))(w, X, y)))(w))}")


# ── 6. Full training loop (no Flax/Equinox) ──────────────────────────────────

def demo_training_loop():
    print("\n" + "=" * 50)
    print("6. Full JAX Training Loop")
    print("=" * 50)

    # Init
    key = jax.random.key(0)

    def init_params(in_dim, hidden, out_dim, key):
        k1, k2, k3 = jax.random.split(key, 3)
        return {
            "W1": jax.random.normal(k1, (in_dim, hidden)) * 0.01,
            "b1": jnp.zeros(hidden),
            "W2": jax.random.normal(k2, (hidden, out_dim)) * 0.01,
            "b2": jnp.zeros(out_dim),
        }

    def forward(params, x):
        h = jax.nn.relu(x @ params["W1"] + params["b1"])
        return (h @ params["W2"] + params["b2"]).squeeze()

    def loss_fn(params, x, y):
        pred = forward(params, x)
        return jnp.mean((pred - y) ** 2)

    @jax.jit
    def train_step(params, x, y, lr=0.01):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
        return params, loss

    # Data: y = sum(x)
    key, dk = jax.random.split(key)
    X = jax.random.normal(dk, (200, 10))
    y = X.sum(axis=1)

    params = init_params(10, 32, 1, key)

    for step in range(300):
        params, loss = train_step(params, X, y)
        if step % 60 == 0:
            print(f"  Step {step:3d}: loss={loss:.4f}")

    print("Training complete.")


if __name__ == "__main__":
    demo_arrays()
    demo_random()
    demo_grad()
    demo_jit()
    demo_vmap()
    demo_training_loop()
