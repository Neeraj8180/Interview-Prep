# TRL (SFT + DPO) — Part 12 code examples
# trl 0.12+ | transformers 4.45+ | Python 3.11+
#
# Covers:
#   1. SFTTrainer: instruction fine-tuning on a toy dataset
#   2. DPOTrainer: direct preference optimization on toy pairs

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig
import torch


# ── 1. SFT on toy instruction data ───────────────────────────────────────────

def demo_sft():
    print("=" * 50)
    print("1. SFT (Supervised Fine-Tuning)")
    print("=" * 50)

    # Toy instruction-following dataset (chat format)
    data = {
        "messages": [
            [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "2+2 = 4."}],
            [{"role": "user", "content": "Name a planet."}, {"role": "assistant", "content": "Mars is a planet."}],
            [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "The sky is blue."}],
        ] * 20  # repeat to have enough data
    }
    dataset = Dataset.from_dict(data)

    model_name = "gpt2"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    config = SFTConfig(
        output_dir="/tmp/sft-output",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        max_seq_length=64,
        logging_steps=5,
        no_cuda=not torch.cuda.is_available(),
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    print("SFT training complete")
    trainer.save_model("/tmp/sft-model")


# ── 2. DPO on toy preference data ────────────────────────────────────────────

def demo_dpo():
    print("\n" + "=" * 50)
    print("2. DPO (Direct Preference Optimization)")
    print("=" * 50)

    # Preference pairs: chosen = better response, rejected = worse response
    data = {
        "prompt": [
            "What is the capital of France?",
            "How do you sort a list in Python?",
            "What is machine learning?",
        ] * 10,
        "chosen": [
            "The capital of France is Paris.",
            "You can sort a list with list.sort() or sorted(list).",
            "Machine learning is a method of teaching computers to learn from data.",
        ] * 10,
        "rejected": [
            "I'm not sure.",
            "I don't know Python.",
            "It's a complex topic.",
        ] * 10,
    }
    dataset = Dataset.from_dict(data)

    model_name = "gpt2"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model     = AutoModelForCausalLM.from_pretrained(model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)

    config = DPOConfig(
        output_dir="/tmp/dpo-output",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        learning_rate=5e-7,
        beta=0.1,
        max_length=64,
        max_prompt_length=32,
        logging_steps=5,
        no_cuda=not torch.cuda.is_available(),
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    print("DPO training complete")
    print("Key hyperparameter: beta=0.1 (KL penalty — how far from reference)")


if __name__ == "__main__":
    demo_sft()
    demo_dpo()
