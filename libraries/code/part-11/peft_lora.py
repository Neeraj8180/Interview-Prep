# PEFT / LoRA — Part 11 code examples
# peft 0.13+ | transformers 4.45+ | Python 3.11+
#
# Covers:
#   1. LoRA config and wrapping a model
#   2. Inspecting trainable parameters
#   3. Training a LoRA adapter
#   4. Saving and loading adapters

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model, PeftModel


# ── 1. Create a LoRA-wrapped model ────────────────────────────────────────────

def demo_lora_config():
    print("=" * 50)
    print("1. LoRA Configuration")
    print("=" * 50)

    model_name = "gpt2"   # small model for demo
    model = AutoModelForCausalLM.from_pretrained(model_name)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Base model parameters: {total_params:,}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=["c_attn"],   # GPT-2 uses c_attn for Q, K, V combined
        lora_dropout=0.05,
        bias="none",
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    # Verify frozen base parameters
    non_lora_frozen = all(
        not p.requires_grad
        for name, p in peft_model.named_parameters()
        if "lora_" not in name
    )
    print(f"All non-LoRA parameters frozen: {non_lora_frozen}")

    return peft_model


# ── 2. Simple LoRA training step ──────────────────────────────────────────────

def demo_lora_training(peft_model):
    print("\n" + "=" * 50)
    print("2. LoRA Training Step")
    print("=" * 50)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    peft_model.to(device)

    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=2e-4)

    texts = [
        "The transformer architecture uses self-attention mechanisms.",
        "Large language models are trained on vast amounts of text data.",
        "PyTorch enables dynamic computation graphs for research.",
    ]
    encoding = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                         max_length=32).to(device)

    peft_model.train()
    optimizer.zero_grad()
    output = peft_model(**encoding, labels=encoding["input_ids"])
    loss = output.loss
    loss.backward()
    optimizer.step()

    print(f"Training loss: {loss.item():.4f}")
    print(f"Gradients computed only for LoRA params — base model unchanged")


# ── 3. Save and reload adapter ────────────────────────────────────────────────

def demo_save_load(peft_model):
    print("\n" + "=" * 50)
    print("3. Save and Load Adapter")
    print("=" * 50)

    adapter_dir = "/tmp/lora-adapter-demo"
    peft_model.save_pretrained(adapter_dir)
    print(f"Adapter saved to {adapter_dir}")

    import os
    files = os.listdir(adapter_dir)
    print(f"Saved files: {files}")
    for f in files:
        path = os.path.join(adapter_dir, f)
        size = os.path.getsize(path)
        print(f"  {f}: {size/1024:.1f} KB")

    # Reload
    base_model = AutoModelForCausalLM.from_pretrained("gpt2")
    loaded = PeftModel.from_pretrained(base_model, adapter_dir)
    loaded.eval()
    print(f"\nAdapter reloaded successfully")

    # Merge and unload (remove adapter overhead)
    merged = loaded.merge_and_unload()
    print(f"Merged model type: {type(merged).__name__}")
    print(f"Merged parameters: {sum(p.numel() for p in merged.parameters()):,}")


if __name__ == "__main__":
    peft_model = demo_lora_config()
    demo_lora_training(peft_model)
    demo_save_load(peft_model)
