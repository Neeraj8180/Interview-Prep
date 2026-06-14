# Megatron-LM 3D Parallelism — Part 15 code examples
# megatron-core 0.8+ | PyTorch 2.6+ | Python 3.11+
#
# Covers:
#   1. Conceptual tensor-parallel matmul simulation (no GPU cluster needed)
#   2. 3D parallelism planning utility
#   3. Megatron-Core API demonstration (requires megatron-core)

import torch
import math


# ── 1. Column-parallel and row-parallel linear simulation ─────────────────────

def simulate_tensor_parallel(d_in: int, d_out: int, t: int, batch: int):
    """
    Simulate column-parallel + row-parallel linear (Megatron's MLP pattern).
    This runs on a single device to demonstrate the math — a real implementation
    uses multiple GPUs with NCCL AllGather/ReduceScatter.
    """
    print("=" * 55)
    print(f"Tensor-Parallel Linear (t={t}-way parallelism)")
    print("=" * 55)
    print(f"  Input:    ({batch}, {d_in})")
    print(f"  W1 (col): ({d_in}, {d_out}) → split to {t}×({d_in}, {d_out//t})")
    print(f"  W2 (row): ({d_out}, {d_in}) → split to {t}×({d_out//t}, {d_in})")

    torch.manual_seed(0)
    X  = torch.randn(batch, d_in)
    W1 = torch.randn(d_in, d_out)       # column-parallel weight
    W2 = torch.randn(d_out, d_in)       # row-parallel weight

    # ── Full (non-parallel) computation ──────────────────────────────────────
    act = torch.nn.functional.gelu
    Y_full = act(X @ W1) @ W2
    print(f"\n  Full output shape: {Y_full.shape}")

    # ── Tensor-parallel simulation ────────────────────────────────────────────
    # Each GPU holds 1/t slice of W1 and W2
    W1_shards = W1.chunk(t, dim=1)    # (d_in, d_out/t) per GPU
    W2_shards = W2.chunk(t, dim=0)    # (d_out/t, d_in) per GPU

    partial_outputs = []
    for i in range(t):
        # Column-parallel forward (no communication):
        h_i = act(X @ W1_shards[i])         # (batch, d_out/t)
        # Row-parallel forward (partial output):
        y_i = h_i @ W2_shards[i]            # (batch, d_in)
        partial_outputs.append(y_i)

    # AllReduce: sum partial outputs across all GPUs
    Y_parallel = sum(partial_outputs)        # (batch, d_in)

    match = torch.allclose(Y_full, Y_parallel, atol=1e-5)
    print(f"  Parallel output matches full: {match} ✓" if match else
          f"  MISMATCH! Max diff: {(Y_full - Y_parallel).abs().max():.2e}")

    # Memory saved per GPU
    total_params = W1.numel() + W2.numel()
    params_per_gpu = total_params // t
    print(f"\n  Weight memory savings:")
    print(f"    Full model:      {total_params:>10,} params")
    print(f"    Per GPU (t={t}): {params_per_gpu:>10,} params ({100/t:.0f}% of total)")
    print(f"    Communication:   2 AllReduce ops per MLP layer (output only)")


# ── 2. 3D Parallelism planning utility ───────────────────────────────────────

def plan_3d_parallelism(
    num_gpus: int,
    param_billions: float,
    num_layers: int,
    gpus_per_node: int = 8,
) -> dict:
    """
    Suggest 3D parallelism configs for a given model and GPU count.
    
    Rules:
    - tensor_parallel <= gpus_per_node (NVLink constraint)
    - pipeline_parallel: num_layers must be divisible
    - data_parallel = num_gpus / (tensor_parallel * pipeline_parallel)
    """
    model_gb = param_billions * 1e9 * 12 / 1e9  # 12 bytes/param (bf16 AdamW)
    configs = []

    for t in [1, 2, 4, 8]:
        if t > gpus_per_node:
            continue
        for p in [1, 2, 4, 8, 16, 32]:
            if num_gpus % (t * p) != 0:
                continue
            if num_layers % p != 0:
                continue
            d = num_gpus // (t * p)
            if d < 1:
                continue
            # Memory per GPU (ZeRO-1 for data parallel)
            mem_per_gpu = model_gb / (t * p) + (model_gb * 8 / num_gpus)  # approx
            configs.append({
                "tensor_parallel": t,
                "pipeline_parallel": p,
                "data_parallel": d,
                "total_gpus": t * p * d,
                "mem_per_gpu_GB": round(mem_per_gpu, 1),
            })

    configs.sort(key=lambda x: abs(x["mem_per_gpu_GB"] - 40))  # prefer 40GB fit
    return configs[:5]


def print_3d_plan(param_billions: float, num_gpus: int, num_layers: int):
    print(f"\n{'=' * 60}")
    print(f"3D Parallelism Plans: {param_billions}B model, {num_gpus} GPUs, {num_layers} layers")
    print(f"{'=' * 60}")
    print(f"{'TP':>4} {'PP':>4} {'DP':>4} {'Mem/GPU GB':>12}")
    print("-" * 60)
    plans = plan_3d_parallelism(num_gpus, param_billions, num_layers)
    for p in plans:
        fits = "✓" if p["mem_per_gpu_GB"] < 80 else "✗"
        print(f"  t={p['tensor_parallel']:<2} p={p['pipeline_parallel']:<2} "
              f"d={p['data_parallel']:<3} → "
              f"{p['mem_per_gpu_GB']:>8.1f} GB  {fits}")


# ── 3. Megatron-Core API demo ─────────────────────────────────────────────────

def demo_megatron_core():
    print(f"\n{'=' * 55}")
    print("Megatron-Core API (requires: pip install megatron-core)")
    print("=" * 55)

    try:
        from megatron.core import parallel_state
        from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear

        # Initialize (requires distributed environment)
        # parallel_state.initialize_model_parallel(
        #     tensor_model_parallel_size=4,
        #     pipeline_model_parallel_size=2,
        # )

        print("Megatron-Core imports successful!")
        print("Key APIs:")
        print("  ColumnParallelLinear(in, out, gather_output=False)")
        print("  RowParallelLinear(in, out, input_is_parallel=True)")
        print("  parallel_state.initialize_model_parallel(t=4, p=2)")
        print("  parallel_state.get_tensor_model_parallel_rank()")

    except ImportError:
        print("megatron-core not installed.")
        print("Install: pip install megatron-core")
        print()
        print("Core concepts (no installation needed):")
        print("  ColumnParallelLinear: W split along output cols, no AllReduce in fwd")
        print("  RowParallelLinear:    W split along input rows, AllReduce in fwd")
        print("  initialize_model_parallel(t, p): sets up process groups")
        print("  get_tensor_model_parallel_rank(): this GPU's rank in TP group")


if __name__ == "__main__":
    # Demonstrate tensor parallelism math
    simulate_tensor_parallel(d_in=512, d_out=2048, t=4, batch=8)

    # Planning utility
    print_3d_plan(param_billions=7.0,   num_gpus=8,  num_layers=32)
    print_3d_plan(param_billions=13.0,  num_gpus=16, num_layers=40)
    print_3d_plan(param_billions=70.0,  num_gpus=64, num_layers=80)
    print_3d_plan(param_billions=175.0, num_gpus=512, num_layers=96)

    # Megatron-Core API
    demo_megatron_core()
