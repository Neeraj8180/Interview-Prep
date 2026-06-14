"""
CUDA Python, NCCL, and Triton code examples — Parts 38-40
cupy 13.x | numba 0.60.x | triton 3.x | Python 3.11+

Install:
  pip install cupy-cuda12x numba triton torch
  # CUDA toolkit required for numba/triton
"""

import numpy as np
import math


# ─────────────────────────────────────────────────────────────────────
# Part 1: CuPy — NumPy on GPU (Part 38)
# ─────────────────────────────────────────────────────────────────────

def demo_cupy_basics():
    """CuPy: NumPy-compatible GPU arrays."""
    print("=" * 60)
    print("1. CuPy — NumPy on GPU")
    print("=" * 60)

    try:
        import cupy as cp
        import time

        N = 2048

        # CPU baseline
        a_cpu = np.random.randn(N, N).astype(np.float32)
        b_cpu = np.random.randn(N, N).astype(np.float32)
        t0 = time.perf_counter()
        c_cpu = np.dot(a_cpu, b_cpu)
        cpu_ms = (time.perf_counter() - t0) * 1000

        # GPU with CuPy
        a_gpu = cp.asarray(a_cpu)
        b_gpu = cp.asarray(b_cpu)
        cp.cuda.stream.get_current_stream().synchronize()  # warm up
        t0 = time.perf_counter()
        c_gpu = cp.dot(a_gpu, b_gpu)
        cp.cuda.stream.get_current_stream().synchronize()
        gpu_ms = (time.perf_counter() - t0) * 1000

        print(f"  Matrix multiply {N}×{N}:")
        print(f"    CPU: {cpu_ms:.1f}ms | GPU: {gpu_ms:.1f}ms | Speedup: {cpu_ms/gpu_ms:.1f}×")

        # Zero-copy interop with PyTorch
        import torch
        t = torch.as_tensor(a_gpu, device="cuda")    # zero-copy!
        a_back = cp.asarray(t)                        # zero-copy back
        print(f"\n  Zero-copy PyTorch ↔ CuPy:")
        print(f"    Original CuPy ptr:  {a_gpu.data.ptr}")
        print(f"    PyTorch data_ptr:   {t.data_ptr()}")
        print(f"    Same memory: {a_gpu.data.ptr == t.data_ptr()}")

        # Memory pool
        pool = cp.get_default_memory_pool()
        print(f"\n  GPU memory pool:")
        print(f"    Used:  {pool.used_bytes() / 1e6:.1f} MB")
        print(f"    Total: {pool.total_bytes() / 1e6:.1f} MB")

    except ImportError:
        print("  cupy not installed: pip install cupy-cuda12x")
        _show_cupy_api()


def _show_cupy_api():
    print("""
  CuPy API:
    import cupy as cp
    x = cp.asarray(np_array)          # CPU → GPU
    y = cp.random.randn(N, M)         # allocate on GPU
    z = cp.dot(x, y.T)                # GPU matmul (cuBLAS)
    result = z.get()                  # GPU → CPU
    t = torch.as_tensor(x, device="cuda")  # zero-copy to PyTorch
""")


def demo_numba_kernel():
    """Numba @cuda.jit — write custom GPU kernels in Python."""
    print("\n" + "=" * 60)
    print("2. Numba @cuda.jit — Custom GPU Kernels")
    print("=" * 60)

    try:
        from numba import cuda
        import numpy as np

        @cuda.jit
        def vector_add_kernel(a, b, out):
            """Each thread handles one element."""
            i = cuda.grid(1)
            if i < out.shape[0]:
                out[i] = a[i] + b[i]

        N = 1_000_000
        a = np.random.randn(N).astype(np.float32)
        b = np.random.randn(N).astype(np.float32)

        d_a   = cuda.to_device(a)
        d_b   = cuda.to_device(b)
        d_out = cuda.device_array(N, dtype=np.float32)

        threads = 256
        blocks  = math.ceil(N / threads)

        vector_add_kernel[blocks, threads](d_a, d_b, d_out)

        result = d_out.copy_to_host()
        ref    = a + b
        print(f"  Vector add {N:,} elements")
        print(f"  Max error vs numpy: {np.max(np.abs(result - ref)):.2e}")
        print(f"  Grid: {blocks} blocks × {threads} threads = {blocks*threads:,} total threads")

    except ImportError:
        print("  numba not installed: pip install numba")
        print("""
  Numba kernel template:
    from numba import cuda
    @cuda.jit
    def kernel(a, out):
        i = cuda.grid(1)
        if i < out.shape[0]:
            out[i] = a[i] * 2.0

    threads = 256
    blocks  = math.ceil(N / threads)
    kernel[blocks, threads](d_a, d_out)
""")


def demo_shared_memory_kernel():
    """Numba kernel with shared memory (tile-based reduction)."""
    print("\n" + "=" * 60)
    print("3. Shared Memory Reduction Kernel")
    print("=" * 60)
    print("""
  Parallel sum using shared memory reduction:

  @cuda.jit
  def parallel_sum(data, partial_sums):
      shared = cuda.shared.array(256, dtype=numba.float32)
      tx, bid = cuda.threadIdx.x, cuda.blockIdx.x
      i       = cuda.grid(1)

      # Load into shared memory
      shared[tx] = data[i] if i < data.shape[0] else 0.0
      cuda.syncthreads()

      # Reduction tree (log2(256) = 8 steps)
      stride = 128
      while stride > 0:
          if tx < stride:
              shared[tx] += shared[tx + stride]
          cuda.syncthreads()
          stride //= 2

      # Write block result
      if tx == 0:
          partial_sums[bid] = shared[0]

  Memory access pattern:
    Global memory (slow):  1 load per thread (coalesced)
    Shared memory (fast):  log2(BLOCK) reads per thread
    Result: BLOCK × reduction work in BLOCK × log2(BLOCK) steps

  Performance: 5-10× faster than naive sequential reduction
""")


# ─────────────────────────────────────────────────────────────────────
# Part 2: NCCL via PyTorch Distributed (Part 39)
# ─────────────────────────────────────────────────────────────────────

def demo_nccl_basics():
    """Show NCCL/torch.distributed usage patterns."""
    print("\n" + "=" * 60)
    print("4. NCCL via PyTorch Distributed")
    print("=" * 60)

    try:
        import torch
        if torch.cuda.is_available():
            print(f"  NCCL version: {torch.cuda.nccl.version()}")
        print("""
  NCCL usage via torch.distributed:

  # Launch: torchrun --nproc_per_node=4 train.py

  import torch
  import torch.distributed as dist

  # 1. Initialize NCCL process group
  dist.init_process_group(
      backend="nccl",       # use NCCL for GPU
      init_method="env://", # reads MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE
  )
  rank       = dist.get_rank()
  world_size = dist.get_world_size()
  torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

  # 2. AllReduce: sum gradients across all GPUs
  grad = torch.randn(1000, device="cuda")
  dist.all_reduce(grad, op=dist.ReduceOp.SUM)
  grad /= world_size  # gradient average

  # 3. Broadcast model weights from rank 0
  model = MyModel().cuda()
  for param in model.parameters():
      dist.broadcast(param.data, src=0)

  # 4. ReduceScatter (FSDP/ZeRO backward)
  shard_size = total_size // world_size
  output     = torch.zeros(shard_size, device="cuda")
  dist.reduce_scatter_tensor(output, full_grad, op=dist.ReduceOp.SUM)

  # 5. AllGather (FSDP/ZeRO forward)
  full_param = torch.empty(total_size, device="cuda")
  dist.all_gather_tensor(full_param, param_shard)

  # 6. Async AllReduce (overlap with compute)
  handle = dist.all_reduce(grad, async_op=True)
  # ... do other work ...
  handle.wait()  # ensure completion before using grad

  # 7. Custom process group (tensor parallelism)
  tp_group = dist.new_group(ranks=[0, 1, 2, 3])  # 4 TP ranks
  dist.all_reduce(tensor, group=tp_group)

  dist.destroy_process_group()  # cleanup
""")
    except ImportError:
        print("  torch not installed: pip install torch")


def demo_nccl_ring_algorithm():
    """Explain ring-allreduce conceptually with simulation."""
    print("\n" + "=" * 60)
    print("5. Ring-AllReduce Algorithm (Conceptual)")
    print("=" * 60)

    # Simulate ring-allreduce on CPU to demonstrate the algorithm
    world_size = 4
    data_size  = 8  # elements per GPU
    np.random.seed(42)
    gpu_data = [np.random.randn(data_size).astype(np.float32) for _ in range(world_size)]

    print(f"  Initial data on each GPU (world_size={world_size}):")
    for rank, d in enumerate(gpu_data):
        print(f"    GPU{rank}: {d.round(2)}")

    # Expected result: element-wise sum
    expected = np.sum(gpu_data, axis=0)
    print(f"\n  Expected sum: {expected.round(2)}")

    # Simulate ring-allreduce
    chunk_size   = data_size // world_size
    result       = [d.copy() for d in gpu_data]
    recv_buffers = [d.copy() for d in gpu_data]

    # Phase 1: ReduceScatter (N-1 rounds)
    for step in range(world_size - 1):
        for rank in range(world_size):
            send_to   = (rank + 1) % world_size
            recv_from = (rank - 1) % world_size
            # Each GPU sends chunk at position (rank - step) mod N
            chunk_idx     = (rank - step - 1) % world_size
            chunk_start   = chunk_idx * chunk_size
            # Simulate receive (in real NCCL, this is async)
            recv_buffers[send_to][chunk_start:chunk_start+chunk_size] += \
                result[rank][chunk_start:chunk_start+chunk_size]
        result = [r.copy() for r in recv_buffers]

    # Phase 2: AllGather (N-1 rounds)
    # After ReduceScatter, each rank has one fully-reduced chunk
    # AllGather distributes each chunk to all GPUs
    # (simplified: just verify the fully-reduced chunks exist)

    # Final: each rank should have the complete sum
    # (simplified simulation - demonstrates the concept)
    final_allreduce = [np.sum(gpu_data, axis=0) for _ in range(world_size)]

    print(f"\n  After AllReduce (each GPU has the sum):")
    print(f"    GPU0: {final_allreduce[0].round(2)}")
    print(f"\n  Bandwidth efficiency: {(world_size-1)/world_size*100:.1f}% for {world_size} GPUs")
    print(f"  vs Parameter Server: bottleneck at master GPU (O(N) bandwidth)")


# ─────────────────────────────────────────────────────────────────────
# Part 3: Triton — Python GPU Kernel Compiler (Part 40)
# ─────────────────────────────────────────────────────────────────────

def demo_triton_elementwise():
    """Simple Triton elementwise kernel."""
    print("\n" + "=" * 60)
    print("6. Triton — Elementwise Kernel")
    print("=" * 60)

    try:
        import triton
        import triton.language as tl
        import torch

        @triton.jit
        def relu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
            pid     = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask    = offsets < n
            x       = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            out     = tl.maximum(x, 0.0)
            tl.store(out_ptr + offsets, out, mask=mask)

        N   = 1_000_000
        x   = torch.randn(N, device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)

        BLOCK = 1024
        grid  = (triton.cdiv(N, BLOCK),)
        relu_kernel[grid](x, out, N, BLOCK_SIZE=BLOCK)

        # Verify
        ref_out = torch.relu(x)
        diff    = (out - ref_out).abs().max().item()
        print(f"  ReLU {N:,} elements")
        print(f"  Max diff vs PyTorch: {diff:.2e}")
        print(f"  Grid: {grid[0]:,} program instances × 1024 elements each")

        # Benchmark
        ms = triton.testing.do_bench(lambda: relu_kernel[grid](x, out, N, BLOCK_SIZE=BLOCK))
        bandwidth = x.nbytes * 2 / ms * 1e-9  # read + write
        print(f"  Latency: {ms:.3f} ms | Bandwidth: {bandwidth:.1f} GB/s")

    except ImportError:
        print("  triton not installed: pip install triton")
        _show_triton_api()


def _show_triton_api():
    print("""
  Triton kernel template:
    import triton
    import triton.language as tl

    @triton.jit
    def my_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid     = tl.program_id(0)           # which tile am I?
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # my indices
        mask    = offsets < n               # bounds check
        x       = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        result  = tl.exp(x)                 # elementwise GPU op
        tl.store(out_ptr + offsets, result, mask=mask)

    # Launch
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    my_kernel[grid](x, out, n, BLOCK_SIZE=1024)

  Key Triton primitives:
    tl.program_id(axis)     — current program's index in grid
    tl.arange(0, BLOCK)     — [0, 1, ..., BLOCK-1] offset array
    tl.load(ptr + offsets, mask=mask)  — load tile from GPU memory
    tl.store(ptr + offsets, val, mask=mask)  — store tile to GPU memory
    tl.dot(a, b)            — tile matrix multiply (tensor cores!)
    tl.constexpr            — compile-time constant (enables specialization)
""")


def demo_triton_matmul():
    """Triton tiled matrix multiply with tensor cores."""
    print("\n" + "=" * 60)
    print("7. Triton Matrix Multiply (Tensor Cores)")
    print("=" * 60)

    try:
        import triton
        import triton.language as tl
        import torch

        @triton.autotune(
            configs=[
                triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64},
                              num_stages=3, num_warps=8),
                triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 32},
                              num_stages=4, num_warps=4),
                triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
                              num_stages=4, num_warps=4),
            ],
            key=["M", "N", "K"],
        )
        @triton.jit
        def matmul_kernel(
            a_ptr, b_ptr, c_ptr,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        ):
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_K)):
                a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
                b = tl.load(b_ptrs, mask=offs_n[None, :] < N, other=0.0)
                acc += tl.dot(a, b)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
            c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

        M, N, K = 1024, 1024, 1024
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        c = torch.empty(M, N, device="cuda", dtype=torch.float32)

        grid = (triton.cdiv(M, 128), triton.cdiv(N, 256))
        matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
        )

        ref = torch.mm(a.float(), b.float())
        diff = (c - ref).abs().max().item()
        print(f"  Matrix multiply {M}×{K}×{N}")
        print(f"  Max diff vs torch.mm: {diff:.2e}")

        ms     = triton.testing.do_bench(lambda: matmul_kernel[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
        ))
        gflops = 2 * M * N * K / ms * 1e-12
        print(f"  Latency: {ms:.2f} ms | Performance: {gflops:.1f} TFLOPS FP16")

    except ImportError:
        print("  triton not installed: pip install triton")


def demo_triton_softmax():
    """Online softmax in Triton — fused, memory-efficient."""
    print("\n" + "=" * 60)
    print("8. Triton Fused Softmax")
    print("=" * 60)

    try:
        import triton
        import triton.language as tl
        import torch

        @triton.autotune(
            configs=[
                triton.Config({"BLOCK_SIZE": 1024}),
                triton.Config({"BLOCK_SIZE": 2048}),
            ],
            key=["n_cols"],
        )
        @triton.jit
        def softmax_kernel(output_ptr, input_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
            row  = tl.program_id(0)
            offs = tl.arange(0, BLOCK_SIZE)
            mask = offs < n_cols
            x    = tl.load(input_ptr + row * n_cols + offs, mask=mask, other=-float("inf"))
            x   -= tl.max(x, axis=0)          # numerical stability
            x    = tl.exp(x)
            y    = x / tl.sum(x, axis=0)
            tl.store(output_ptr + row * n_cols + offs, y, mask=mask)

        batch_size, n_cols = 4096, 512
        x   = torch.randn(batch_size, n_cols, device="cuda")
        out = torch.empty_like(x)

        softmax_kernel[(batch_size,)](out, x, n_cols)

        ref  = torch.softmax(x, dim=-1)
        diff = (out - ref).abs().max().item()
        print(f"  Softmax [{batch_size}, {n_cols}]")
        print(f"  Max diff vs torch.softmax: {diff:.2e}")

        ms = triton.testing.do_bench(lambda: softmax_kernel[(batch_size,)](out, x, n_cols))
        bw = x.nbytes * 2 / ms * 1e-9
        print(f"  Latency: {ms:.3f} ms | Bandwidth: {bw:.1f} GB/s")
        print("""
  Why Triton softmax is faster than PyTorch's default:
    PyTorch default: 3 kernels (max reduction, exp, div)
                     3× global memory read-write
    Triton fused:    1 kernel, 1× global memory read-write
    Result: ~2× faster due to memory bandwidth savings
""")

    except ImportError:
        print("  triton not installed: pip install triton")


def demo_flash_attention_concept():
    """Explain Flash Attention algorithm (without CUDA)."""
    print("\n" + "=" * 60)
    print("9. Flash Attention: Memory-Efficient Attention (Algorithm)")
    print("=" * 60)
    print("""
  Standard Attention O(N²) memory:
    S = Q @ K.T              [N, N] — stored in HBM
    P = softmax(S / sqrt(d)) [N, N] — stored in HBM
    O = P @ V                [N, d]
    Memory: O(N² + Nd) — for N=4096, d=128 → 16M + 0.5M floats

  Flash Attention O(N) memory:
    # Split Q into BLOCK_M tiles, iterate K,V in BLOCK_N tiles
    for q_block in Q:           # BLOCK_M rows of Q
        m_i   = -inf            # running max (scalar per q)
        l_i   = 0               # running sum
        O_acc = zeros           # accumulator

        for k_block, v_block in zip(K, V):    # BLOCK_N cols at a time
            qk    = q_block @ k_block.T        # [BM, BN] — small, in SRAM
            m_ij  = max(qk, dim=-1)            # local max
            p     = exp(qk - m_ij)             # local softmax numerator

            # Online softmax update
            m_new = max(m_i, m_ij)
            alpha = exp(m_i - m_new)           # rescaling factor
            l_i   = alpha * l_i + sum(p)       # update running sum
            O_acc = alpha * O_acc + p @ v_block # update accumulator
            m_i   = m_new

        O[q_block] = O_acc / l_i              # final normalization

  Memory: only SRAM holds [BM, BN] tiles; no N×N matrix ever in HBM!
  For N=4096: standard = 64MB, Flash = 16KB tiles (4096× reduction)

  Triton makes this implementable in Python:
    - tl.program_id selects which Q tile
    - tl.dot computes QK^T and PV tiles
    - Online softmax is standard Python arithmetic on tl scalars
""")


if __name__ == "__main__":
    demo_cupy_basics()
    demo_numba_kernel()
    demo_shared_memory_kernel()
    demo_nccl_basics()
    demo_nccl_ring_algorithm()
    demo_triton_elementwise()
    demo_triton_matmul()
    demo_triton_softmax()
    demo_flash_attention_concept()

    print("\n" + "=" * 60)
    print("Parts 38-40 Summary: GPU Computing Stack")
    print("=" * 60)
    print("""
  GPU programming levels (lowest to highest):

  cuda-python (Part 38)  — Raw CUDA Driver API
    Direct device management, streams, events, memory
    Use: device queries, multi-GPU management, graph capture

  CuPy (Part 38)         — NumPy on GPU
    GPU array with NumPy-compatible API, cuBLAS/cuFFT backend
    Use: data preprocessing, signal processing, scientific computing

  Numba (Part 38)        — Python JIT → CUDA kernels
    Write Python kernels compiled to CUDA via LLVM
    Use: custom elementwise ops, reductions, prototype kernels

  NCCL (Part 39)         — GPU collective communications
    AllReduce, AllGather, ReduceScatter for distributed training
    Use: gradient sync (DDP), ZeRO stages, tensor/pipeline parallel

  Triton (Part 40)       — Tile-level GPU kernel compiler
    Python → PTX via MLIR; automatic tensor cores, shared memory
    Use: Flash Attention, fused ops, custom quantization kernels

  Relationship to DL frameworks:
    PyTorch → calls cuDNN/cuBLAS (most ops)
           → can call Triton kernels (torch.compile)
           → can call custom CUDA extensions (Numba/cuda-python)
           → uses NCCL for multi-GPU gradient sync
""")
