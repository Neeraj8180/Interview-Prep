# HuggingFace Transformers — Part 10 code examples
# transformers 4.45+ | Python 3.11+
#
# Covers:
#   1. AutoTokenizer and encoding
#   2. AutoModel inference (embeddings)
#   3. Pipeline API for common tasks
#   4. text generation with model.generate()

from transformers import AutoTokenizer, AutoModel, pipeline
import torch


# ── 1. Tokenization ───────────────────────────────────────────────────────────

def demo_tokenization():
    print("=" * 50)
    print("1. Tokenization")
    print("=" * 50)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Basic tokenization
    text = "Machine learning is a subset of artificial intelligence."
    tokens = tokenizer.tokenize(text)
    print(f"Tokens: {tokens}")

    ids = tokenizer.encode(text)
    print(f"IDs (with [CLS] and [SEP]): {ids}")
    print(f"Decoded: {tokenizer.decode(ids, skip_special_tokens=True)}")

    # Batch encoding with padding
    texts = [
        "Short sentence.",
        "A much longer sentence that requires more tokens to encode completely.",
    ]
    encoding = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,        # pad to longest
        truncation=True,
        max_length=32,
    )
    print(f"\nBatch input_ids shape: {encoding['input_ids'].shape}")
    print(f"attention_mask (1=real, 0=pad):\n{encoding['attention_mask']}")


# ── 2. BERT embeddings ────────────────────────────────────────────────────────

def demo_embeddings():
    print("\n" + "=" * 50)
    print("2. Sentence Embeddings (BERT)")
    print("=" * 50)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model     = AutoModel.from_pretrained("bert-base-uncased")
    model.eval()

    sentences = [
        "A dog is playing with a ball.",
        "A puppy is fetching a toy.",
        "The economy grew by 3% last quarter.",
    ]
    encoding = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        output = model(**encoding)

    # Mean pooling over non-padding tokens
    hidden = output.last_hidden_state   # (B, S, 768)
    mask   = encoding["attention_mask"].unsqueeze(-1).float()
    emb    = (hidden * mask).sum(1) / mask.sum(1)   # (B, 768)

    # L2 normalize
    emb = emb / emb.norm(dim=1, keepdim=True)
    sim = emb @ emb.T

    print(f"Embedding shape: {emb.shape}")
    print(f"\nCosine similarities:")
    for i, s1 in enumerate(sentences):
        for j, s2 in enumerate(sentences):
            if j > i:
                print(f"  '{s1[:35]}...' <-> '{s2[:35]}...': {sim[i,j].item():.3f}")


# ── 3. Pipelines ──────────────────────────────────────────────────────────────

def demo_pipelines():
    print("\n" + "=" * 50)
    print("3. Pipeline API")
    print("=" * 50)

    # Sentiment analysis
    sa = pipeline("sentiment-analysis")
    results = sa(["I love this library!", "This API is confusing."])
    for r in results:
        print(f"  {r['label']}: {r['score']:.3f}")

    # Zero-shot classification
    zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    result = zs(
        "PyTorch 2.6 was released with major performance improvements.",
        candidate_labels=["technology", "sports", "politics", "science"],
    )
    print(f"\nZero-shot top label: {result['labels'][0]} ({result['scores'][0]:.3f})")

    # Feature extraction (embeddings)
    fe = pipeline("feature-extraction", model="bert-base-uncased", framework="pt")
    features = fe("Hello world")
    print(f"\nFeature shape: {len(features[0])} tokens × {len(features[0][0])} dims")


# ── 4. Text generation ────────────────────────────────────────────────────────

def demo_generation():
    print("\n" + "=" * 50)
    print("4. Text Generation (GPT-2)")
    print("=" * 50)

    from transformers import AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()

    prompt  = "The key advantage of transformer models is"
    inputs  = tokenizer(prompt, return_tensors="pt")

    # Greedy
    with torch.no_grad():
        greedy = model.generate(**inputs, max_new_tokens=30, do_sample=False)
    print(f"Greedy: {tokenizer.decode(greedy[0], skip_special_tokens=True)}")

    # Sampling with temperature and top-p
    with torch.no_grad():
        sampled = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
        )
    print(f"Sample: {tokenizer.decode(sampled[0], skip_special_tokens=True)}")


if __name__ == "__main__":
    demo_tokenization()
    demo_embeddings()
    demo_pipelines()
    demo_generation()
