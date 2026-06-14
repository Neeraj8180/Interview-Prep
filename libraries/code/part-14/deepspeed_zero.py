# DeepSpeed ZeRO — Part 14 code examples
# deepspeed 0.16.x | PyTorch 2.6+ | Python 3.11+
#
# Covers:
#   1. ZeRO Stage 2 config generation
#   2. ZeRO Stage 3 config generation
#   3. DeepSpeed initialization and training step
#   4. Memory calculation utility
#
# To run with DeepSpeed:
#   deepspeed --num_gpus 4 deepspeed_training.py
# Or with Accelerate:
#   accelerate launch --config_file deepspeed_accel.yaml deepspeed_training.py

import json
import os


# ── 1. Config generators ──────────────────────────────────────────────────────

def make_zero2_config(
    micro_batch_size: int = 4,
    grad_accum_steps: int = 4,
    use_bf16: bool = True,
) -> dict:
    """ZeRO Stage 2: partitions optimizer states + gradients."""
    return {
        "bf16": {"enabled": use_bf16},
        "fp16": {"enabled": not use_bf16},
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": True,
            "allgather_bucket_size": 2e8,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
        },
        "gradient_clipping": 1.0,
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": grad_accum_steps,
        "steps_per_print": 50,
        "wall_clock_breakdown": False,
    }


def make_zero3_config(
    micro_batch_size: int = 1,
    grad_accum_steps: int = 8,
    cpu_offload: bool = False,
) -> dict:
    """ZeRO Stage 3: partitions optimizer states + gradients + parameters."""
    config = {
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
            "offload_optimizer": {"device": "none"},
            "offload_param": {"device": "none"},
        },
        "gradient_clipping": 1.0,
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": grad_accum_steps,
        "steps_per_print": 10,
    }
    if cpu_offload:
        config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
        }
        config["zero_optimization"]["offload_param"] = {
            "device": "cpu",
            "pin_memory": True,
        }
    return config


# ── 2. Memory calculator ──────────────────────────────────────────────────────

def calculate_zero_memory(
    param_billions: float,
    num_gpus: int,
    zero_stage: int,
    bytes_per_param_fp16: float = 2.0,
) -> dict:
    """
    Estimate per-GPU memory for ZeRO training.
    
    AdamW memory per parameter:
        params (bf16):    2 bytes
        grads  (bf16):    2 bytes
        Adam m (fp32):    4 bytes
        Adam v (fp32):    4 bytes
        Total:           12 bytes
    """
    params = param_billions * 1e9
    bytes_per_param = 12.0  # full AdamW in mixed precision

    if zero_stage == 0:   # DDP
        mem_bytes = params * bytes_per_param
    elif zero_stage == 1:  # partition optimizer states
        mem_bytes = params * (2 + 2) + params * 8 / num_gpus
    elif zero_stage == 2:  # partition optimizer + gradients
        mem_bytes = params * 2 + params * (2 + 8) / num_gpus
    elif zero_stage == 3:  # partition everything
        mem_bytes = params * bytes_per_param / num_gpus

    mem_gb = mem_bytes / 1e9
    return {
        "stage": zero_stage,
        "num_gpus": num_gpus,
        "params_B": param_billions,
        "per_gpu_GB": round(mem_gb, 1),
        "fits_A100_40GB": mem_gb < 40,
        "fits_A100_80GB": mem_gb < 80,
        "fits_H100_80GB": mem_gb < 80,
    }


def print_memory_table(param_billions: float = 7.0):
    print(f"\nMemory estimates for {param_billions}B parameter model")
    print("(params + grads + AdamW m,v — no activations)")
    print("-" * 60)
    print(f"{'Config':<30} {'Per-GPU GB':>12}")
    print("-" * 60)
    for n_gpu in [1, 4, 8]:
        for stage in [0, 1, 2, 3]:
            if stage == 0 and n_gpu > 1:
                continue  # DDP shows same on each GPU
            result = calculate_zero_memory(param_billions, n_gpu, stage)
            label = f"ZeRO-{stage}, {n_gpu} GPU{'s' if n_gpu > 1 else ''}"
            fits = "✓" if result["fits_A100_80GB"] else "✗"
            print(f"  {label:<28} {result['per_gpu_GB']:>8.1f} GB  {fits}")
    print()


# ── 3. DeepSpeed training demo ────────────────────────────────────────────────

def demo_deepspeed_training():
    """Demonstrates DeepSpeed training when deepspeed is available."""
    try:
        import deepspeed
        import torch
        import torch.nn as nn

        print("=" * 50)
        print("DeepSpeed Training Demo")
        print("=" * 50)

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(256, 1024),
                    nn.GELU(),
                    nn.Linear(1024, 256),
                )
                self.head = nn.Linear(256, 2)

            def forward(self, x, labels=None):
                out = self.head(self.layers(x))
                if labels is not None:
                    loss = nn.CrossEntropyLoss()(out, labels)
                    return loss, out
                return out

        model = SimpleModel()
        ds_config = make_zero2_config(micro_batch_size=4, grad_accum_steps=1)

        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            config=ds_config,
            model_parameters=model.parameters(),
        )

        device = model_engine.device
        for step in range(5):
            X      = torch.randn(4, 256, device=device)
            labels = torch.randint(0, 2, (4,), device=device)
            loss, _ = model_engine(X, labels)
            model_engine.backward(loss)
            model_engine.step()
            if model_engine.global_rank == 0:
                print(f"  Step {step+1}: loss={loss.item():.4f}")

        print("Training complete (ZeRO Stage 2)")
        model_engine.save_checkpoint("/tmp/deepspeed_ckpt")
        print("Checkpoint saved to /tmp/deepspeed_ckpt")

    except ImportError:
        print("deepspeed not installed. Run: pip install deepspeed")
        print("Training demo skipped — config generation still works below.")


# ── 4. HuggingFace Trainer integration example ───────────────────────────────

def show_hf_integration():
    print("\n" + "=" * 50)
    print("HuggingFace Trainer + DeepSpeed Integration")
    print("=" * 50)

    config = make_zero3_config(micro_batch_size=1, grad_accum_steps=8)
    config_path = "/tmp/ds_zero3_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"ZeRO Stage 3 config written to {config_path}")

    integration_code = '''
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM

args = TrainingArguments(
    output_dir="./output",
    deepspeed="/tmp/ds_zero3_config.json",   # ← enable DeepSpeed ZeRO-3
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    bf16=True,
    num_train_epochs=1,
)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
trainer = Trainer(model=model, args=args, train_dataset=dataset)
trainer.train()

# Launch with:
# deepspeed --num_gpus 8 train.py
# OR: accelerate launch train.py (with DeepSpeed accelerate config)
'''
    print("\nIntegration code:")
    print(integration_code)


if __name__ == "__main__":
    # Memory analysis (no GPU required)
    print_memory_table(param_billions=7.0)
    print_memory_table(param_billions=70.0)

    # Save sample configs
    z2 = make_zero2_config()
    z3 = make_zero3_config()
    z3_cpu = make_zero3_config(cpu_offload=True)

    print("ZeRO Stage 2 config:")
    print(json.dumps(z2, indent=2))
    print("\nZeRO Stage 3 config (no offload):")
    print(json.dumps(z3["zero_optimization"], indent=2))

    # DeepSpeed training (if installed)
    demo_deepspeed_training()
    show_hf_integration()
