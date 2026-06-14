# vLLM inference — Part 16 code examples
# vllm 0.6.x | Python 3.11+
#
# Covers:
#   1. Batch inference with LLM + SamplingParams
#   2. OpenAI-compatible server client
#   3. Performance comparison vs HuggingFace generate()
#   4. LoRA adapter serving

from typing import Optional
import time


# ── 1. Basic vLLM inference ───────────────────────────────────────────────────

def demo_vllm_inference():
    """Batch inference with vLLM."""
    try:
        from vllm import LLM, SamplingParams

        print("=" * 50)
        print("1. vLLM Batch Inference")
        print("=" * 50)

        # Load model (downloads from HuggingFace Hub if not cached)
        llm = LLM(
            model="facebook/opt-125m",   # tiny model for demo
            dtype="float32",             # use float32 for CPU compatibility
            gpu_memory_utilization=0.85,
            max_model_len=512,
            enforce_eager=True,          # skip CUDA graph for small models
        )

        params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=50,
        )

        prompts = [
            "Machine learning is a field of",
            "Python is a programming language that",
            "The transformer architecture was introduced",
        ]

        t0 = time.time()
        outputs = llm.generate(prompts, params)
        elapsed = time.time() - t0

        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        print(f"\nGenerated {total_tokens} tokens in {elapsed:.2f}s")
        print(f"Throughput: {total_tokens/elapsed:.1f} tokens/sec\n")

        for output in outputs:
            print(f"Prompt: {output.prompt[:40]}...")
            print(f"Output: {output.outputs[0].text[:80]}")
            print()

    except ImportError:
        print("vllm not installed. Run: pip install vllm")
        print("Install requires NVIDIA GPU with CUDA 12.1+")


# ── 2. Sampling strategies ────────────────────────────────────────────────────

def demo_sampling_strategies():
    """Show different sampling parameter combinations."""
    try:
        from vllm import LLM, SamplingParams
        print("=" * 50)
        print("2. Sampling Strategies")
        print("=" * 50)

        llm   = LLM("facebook/opt-125m", dtype="float32", max_model_len=256)
        prompt = "The most important aspect of artificial intelligence is"

        strategies = [
            ("Greedy",    SamplingParams(temperature=0.0, max_tokens=30)),
            ("Temp 0.8",  SamplingParams(temperature=0.8, max_tokens=30)),
            ("Temp 1.5",  SamplingParams(temperature=1.5, max_tokens=30)),
            ("Top-p 0.5", SamplingParams(temperature=1.0, top_p=0.5, max_tokens=30)),
            ("Top-k 10",  SamplingParams(temperature=1.0, top_k=10, max_tokens=30)),
        ]

        for name, params in strategies:
            output = llm.generate([prompt], params)[0]
            print(f"{name:12}: {output.outputs[0].text[:60]}")

    except ImportError:
        print("vllm not installed — skipping sampling demo")


# ── 3. OpenAI-compatible client (works with running vLLM server) ──────────────

def demo_openai_client(base_url: str = "http://localhost:8000/v1"):
    """
    Client for a running vLLM server.
    
    Start server first:
        vllm serve facebook/opt-125m --port 8000 --dtype float32
    """
    print("=" * 50)
    print("3. OpenAI-Compatible Client")
    print("=" * 50)
    print(f"Connecting to: {base_url}")

    try:
        from openai import OpenAI
        import httpx

        client = OpenAI(base_url=base_url, api_key="none",
                        http_client=httpx.Client(timeout=5.0))

        # Check available models
        models = client.models.list()
        print(f"Available models: {[m.id for m in models.data]}")

        response = client.completions.create(
            model=models.data[0].id if models.data else "facebook/opt-125m",
            prompt="Explain transformers in one sentence:",
            max_tokens=60,
            temperature=0.7,
        )
        print(f"\nResponse: {response.choices[0].text}")
        print(f"Usage: {response.usage}")

    except Exception as e:
        print(f"Server not available ({type(e).__name__})")
        print("Start server with: vllm serve facebook/opt-125m --port 8000")

        # Show the code pattern
        print("\nCode pattern for when server is running:")
        print("""
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

# Chat completions
resp = client.chat.completions.create(
    model="meta-llama/Llama-3.2-1B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "Explain PagedAttention."},
    ],
    max_tokens=200,
    stream=False,
)
print(resp.choices[0].message.content)
""")


# ── 4. Performance estimation ─────────────────────────────────────────────────

def estimate_throughput(
    model_size_b: float,
    gpu_memory_gb: float,
    gpu_bandwidth_tb: float = 3.35,  # H100 HBM3
) -> dict:
    """
    Rough throughput estimate for LLM inference.
    
    Memory-bandwidth-bound regime (small batch, decode-dominated):
    throughput ≈ GPU_bandwidth / (model_size × bytes_per_param)
    """
    bytes_per_param_fp16 = 2.0
    bytes_per_param_fp8  = 1.0

    params = model_size_b * 1e9
    bandwidth_bytes = gpu_bandwidth_tb * 1e12

    tokens_per_sec_fp16 = bandwidth_bytes / (params * bytes_per_param_fp16)
    tokens_per_sec_fp8  = bandwidth_bytes / (params * bytes_per_param_fp8)

    return {
        "model_size_B":           model_size_b,
        "fp16_tokens_per_sec":    round(tokens_per_sec_fp16),
        "fp8_tokens_per_sec":     round(tokens_per_sec_fp8),
        "fits_in_gpu":            model_size_b * 2 < gpu_memory_gb,
        "fits_in_gpu_fp8":        model_size_b * 1 < gpu_memory_gb,
    }


def print_throughput_table():
    print("\n" + "=" * 55)
    print("LLM Throughput Estimates (H100-80GB, memory-bound)")
    print("=" * 55)
    print(f"{'Model':>10} {'FP16 tok/s':>12} {'FP8 tok/s':>12} {'Fits FP16':>10}")
    print("-" * 55)
    for size_b in [1, 7, 13, 70, 175]:
        est = estimate_throughput(size_b, gpu_memory_gb=80)
        fits = "✓" if est["fits_in_gpu"] else "✗"
        print(f"  {size_b}B{'':<7} {est['fp16_tokens_per_sec']:>10,}  "
              f"{est['fp8_tokens_per_sec']:>10,}   {fits}")


if __name__ == "__main__":
    demo_vllm_inference()
    demo_sampling_strategies()
    demo_openai_client()
    print_throughput_table()
