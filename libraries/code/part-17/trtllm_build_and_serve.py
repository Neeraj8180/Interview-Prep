# TensorRT-LLM — Part 17 code examples
# tensorrt-llm 0.14.x | Python 3.11+
#
# TensorRT-LLM requires NVIDIA GPU with CUDA 12.x
# Install: pip install tensorrt-llm
#
# Covers:
#   1. Build script generation (convert + trtllm-build commands)
#   2. Python inference API
#   3. FP8 vs FP16 memory comparison
#   4. Performance benchmarking utility

import json
import shutil


# ── 1. Build script generators ────────────────────────────────────────────────

def generate_build_commands(
    model_name: str,
    hf_model_dir: str,
    output_dir: str,
    tp_size: int = 1,
    dtype: str = "float16",
    fp8: bool = False,
    max_batch_size: int = 32,
    max_seq_len: int = 4096,
) -> str:
    """Generate TensorRT-LLM build commands for a given model."""
    ckpt_dir = output_dir + "/ckpt"
    engine_dir = output_dir + "/engine"

    if fp8:
        quant_step = f"""
# Step 1a: FP8 quantization calibration
python tensorrt_llm/examples/quantization/quantize.py \\
    --model_dir {hf_model_dir} \\
    --dtype float16 \\
    --qformat fp8 \\
    --kv_cache_dtype fp8 \\
    --calib_size 512 \\
    --output_dir {ckpt_dir}
"""
        gemm_plugin = "fp8"
    else:
        quant_step = f"""
# Step 1: Convert HuggingFace model to TensorRT-LLM checkpoint
python tensorrt_llm/examples/{model_name}/convert_checkpoint.py \\
    --model_dir {hf_model_dir} \\
    --output_dir {ckpt_dir} \\
    --dtype {dtype}
"""
        gemm_plugin = dtype

    build_cmd = f"""
# Step 2: Build the TensorRT-LLM engine (COMPILATION STEP — takes minutes)
trtllm-build \\
    --checkpoint_dir {ckpt_dir} \\
    --output_dir {engine_dir} \\
    --gemm_plugin {gemm_plugin} \\
    --tp_size {tp_size} \\
    --max_batch_size {max_batch_size} \\
    --max_input_len 2048 \\
    --max_seq_len {max_seq_len} \\
    --use_paged_context_fmha enable \\
    --workers {tp_size}

# Step 3: Serve (OpenAI-compatible)
{"mpirun -n " + str(tp_size) + " " if tp_size > 1 else ""}trtllm-serve \\
    {engine_dir} \\
    --model_name {model_name} \\
    --port 8000
"""
    return quant_step + build_cmd


# ── 2. Engine memory estimator ────────────────────────────────────────────────

def estimate_engine_memory(
    params_billions: float,
    dtype: str = "float16",
    tp_size: int = 1,
) -> dict:
    """Estimate TRT-LLM engine memory requirements per GPU."""
    bytes_per_param = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "fp8":     1,
        "int8":    1,
        "int4":    0.5,
    }
    bpp = bytes_per_param.get(dtype, 2)
    params = params_billions * 1e9
    total_gb = (params * bpp) / 1e9
    per_gpu_gb = total_gb / tp_size

    return {
        "model_params_B":     params_billions,
        "dtype":              dtype,
        "tp_size":            tp_size,
        "total_weight_GB":    round(total_gb, 1),
        "per_gpu_weight_GB":  round(per_gpu_gb, 1),
        "fits_A100_40GB":     per_gpu_gb < 35,
        "fits_A100_80GB":     per_gpu_gb < 75,
        "fits_H100_80GB":     per_gpu_gb < 75,
    }


def print_memory_comparison():
    print("=" * 65)
    print("TensorRT-LLM Engine Memory vs Precision")
    print("=" * 65)
    print(f"{'Model':>8} {'TP':>4} {'FP16 GB/GPU':>12} {'FP8 GB/GPU':>12} {'INT4 GB/GPU':>12}")
    print("-" * 65)
    configs = [(7, 1), (13, 1), (70, 4), (70, 8), (175, 8)]
    for (params, tp) in configs:
        fp16 = estimate_engine_memory(params, "float16", tp)
        fp8  = estimate_engine_memory(params, "fp8", tp)
        int4 = estimate_engine_memory(params, "int4", tp)
        print(f"  {params}B{'':<3}  tp={tp}  "
              f"{fp16['per_gpu_weight_GB']:>10.1f}  "
              f"{fp8['per_gpu_weight_GB']:>10.1f}  "
              f"{int4['per_gpu_weight_GB']:>10.1f}")


# ── 3. TensorRT-LLM inference API ────────────────────────────────────────────

def demo_trtllm_inference(engine_dir: str = "./engine"):
    """
    Demo TensorRT-LLM Python inference API.
    Requires a compiled .engine file (built with trtllm-build).
    """
    print("\n" + "=" * 50)
    print("TensorRT-LLM Inference API")
    print("=" * 50)

    if not shutil.which("trtllm-build"):
        print("TensorRT-LLM not installed.")
        print("Install: pip install tensorrt-llm")
        print()
        print("Python inference API (requires compiled engine):")
        print("""
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner, SamplingConfig

# Load compiled engine
runner = ModelRunner.from_dir(
    engine_dir="./engine",
    rank=tensorrt_llm.mpi_rank(),
)

# Tokenize
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
input_ids = tokenizer(["Explain PagedAttention:"], return_tensors="pt")["input_ids"]

# Generate
sampling_config = SamplingConfig(
    end_id=tokenizer.eos_token_id,
    pad_id=tokenizer.pad_token_id,
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.95,
)
outputs = runner.generate(input_ids, sampling_config=sampling_config)

# Decode
for output in outputs:
    print(tokenizer.decode(output[input_ids.shape[1]:], skip_special_tokens=True))
""")
        return

    try:
        import tensorrt_llm
        from tensorrt_llm.runtime import ModelRunner, SamplingConfig
        from transformers import AutoTokenizer

        runner = ModelRunner.from_dir(engine_dir=engine_dir,
                                      rank=tensorrt_llm.mpi_rank())
        tokenizer = AutoTokenizer.from_pretrained(engine_dir)

        prompts = ["Explain the difference between vLLM and TensorRT-LLM:"]
        input_ids = tokenizer(prompts, return_tensors="pt", padding=True)["input_ids"]

        sampling_config = SamplingConfig(
            end_id=tokenizer.eos_token_id,
            pad_id=tokenizer.pad_token_id,
            max_new_tokens=100,
            temperature=0.7,
        )

        import time
        t0 = time.time()
        outputs = runner.generate(input_ids, sampling_config=sampling_config)
        elapsed = time.time() - t0

        output_ids = outputs[0][input_ids.shape[1]:]
        text = tokenizer.decode(output_ids, skip_special_tokens=True)
        print(f"Output: {text}")
        print(f"Latency: {elapsed*1000:.0f}ms")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Build script examples
    print("=" * 55)
    print("TensorRT-LLM Build Commands")
    print("=" * 55)
    print("\n--- Llama-3.2-1B (FP16, single GPU) ---")
    print(generate_build_commands(
        "llama", "./hf-models/llama-3.2-1b", "./trtllm-1b",
        tp_size=1, dtype="float16", fp8=False,
    ))
    print("\n--- Llama-3.1-70B (FP8, 4-GPU tensor parallel) ---")
    print(generate_build_commands(
        "llama", "./hf-models/llama-3.1-70b", "./trtllm-70b",
        tp_size=4, dtype="float16", fp8=True,
    ))

    # Memory comparison table
    print_memory_comparison()

    # Inference API demo
    demo_trtllm_inference()
