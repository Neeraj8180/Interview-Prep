# FAISS + Qdrant vector search — Parts 25-26 code examples
# faiss-cpu 1.8.x | qdrant-client 1.12.x | Python 3.11+
# Install: pip install faiss-cpu qdrant-client numpy

import numpy as np
import time


# ── FAISS demos ────────────────────────────────────────────────────────────────

def demo_faiss_indices():
    """Compare FAISS index types: Flat, IVF, HNSW."""
    print("=" * 60)
    print("FAISS Index Types Comparison")
    print("=" * 60)

    try:
        import faiss

        d, N = 64, 50_000
        np.random.seed(42)
        vecs  = np.random.rand(N, d).astype("float32")
        query = np.random.rand(1, d).astype("float32")
        k     = 5

        # ── 1. Flat (exact search) ─────────────────────────────────────────────
        t0   = time.time()
        flat = faiss.IndexFlatL2(d)
        flat.add(vecs)
        t_build_flat = time.time() - t0

        t1 = time.time()
        D_flat, I_flat = flat.search(query, k)
        t_search_flat  = time.time() - t1

        # ── 2. IVF (clustered) ────────────────────────────────────────────────
        nlist = 200
        quant = faiss.IndexFlatL2(d)
        ivf   = faiss.IndexIVFFlat(quant, d, nlist)

        t0 = time.time()
        ivf.train(vecs)
        ivf.add(vecs)
        t_build_ivf = time.time() - t0
        ivf.nprobe  = 20

        t1 = time.time()
        D_ivf, I_ivf = ivf.search(query, k)
        t_search_ivf = time.time() - t1

        # ── 3. HNSW (graph-based) ─────────────────────────────────────────────
        M    = 16
        hnsw = faiss.IndexHNSWFlat(d, M)
        hnsw.hnsw.efConstruction = 200

        t0 = time.time()
        hnsw.add(vecs)
        t_build_hnsw = time.time() - t0
        hnsw.hnsw.efSearch = 64

        t1 = time.time()
        D_hnsw, I_hnsw = hnsw.search(query, k)
        t_search_hnsw  = time.time() - t1

        # ── Results ───────────────────────────────────────────────────────────
        print(f"\n{'Index':<20} {'Build ms':>10} {'Search ms':>10} {'Recall@1':>10}")
        print("-" * 55)
        true_idx = I_flat[0][0]
        for name, build, search, I in [
            ("Flat (exact)",   t_build_flat*1e3,  t_search_flat*1e3,  I_flat),
            (f"IVF-{nlist},nprobe=20", t_build_ivf*1e3,   t_search_ivf*1e3,   I_ivf),
            (f"HNSW(M={M})",   t_build_hnsw*1e3, t_search_hnsw*1e3, I_hnsw),
        ]:
            recall = "✓" if I[0][0] == true_idx else "✗"
            print(f"  {name:<18} {build:>8.1f}  {search:>8.3f}  {recall:>10}")

        # Memory usage
        print(f"\nMemory (flat vectors): {N * d * 4 / 1024**2:.1f} MB")
        print(f"flat.ntotal = {flat.ntotal}")

    except ImportError:
        print("faiss-cpu not installed: pip install faiss-cpu")
        _show_faiss_concepts()


def _show_faiss_concepts():
    print("\nFAISS key concepts:")
    print("  IndexFlatL2(d):           exact L2 search — O(N×d) per query")
    print("  IndexIVFFlat(q, d, nlist): cluster → search nprobe clusters")
    print("  IndexHNSWFlat(d, M):      graph-based ANN — no training needed")
    print("  IndexPQ(d, M, nbits):     compressed vectors — 8× smaller")
    print()
    print("  Always use float32:")
    print("    vecs = np.random.rand(N, d).astype('float32')")
    print("  Always train before add (IVF/PQ):")
    print("    ivf.train(vecs); ivf.add(vecs)")


def demo_faiss_cosine():
    """FAISS cosine similarity with normalized vectors."""
    try:
        import faiss

        d = 128
        N = 10_000
        np.random.seed(0)

        # Generate and L2-normalize (enables cosine via inner product)
        vecs = np.random.rand(N, d).astype("float32")
        faiss.normalize_L2(vecs)   # in-place normalization

        index = faiss.IndexFlatIP(d)  # inner product = cosine when normalized
        index.add(vecs)

        query = np.random.rand(1, d).astype("float32")
        faiss.normalize_L2(query)

        D, I = index.search(query, k=5)
        print(f"\nCosine similarity results (scores in [-1, 1]):")
        for i, (idx, score) in enumerate(zip(I[0], D[0])):
            print(f"  #{i+1}: idx={idx:5d} cosine={score:.4f}")

    except ImportError:
        pass


# ── Qdrant demos ───────────────────────────────────────────────────────────────

def demo_qdrant_filtered_search():
    """Qdrant: vector search with payload filtering."""
    print("\n" + "=" * 60)
    print("Qdrant: Filtered Vector Search")
    print("=" * 60)

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import (
            Distance, VectorParams, PointStruct,
            Filter, FieldCondition, MatchValue, Range,
        )

        client = QdrantClient(":memory:")

        # Create collection
        client.create_collection(
            "tech_docs",
            vectors_config=VectorParams(size=32, distance=Distance.COSINE),
        )

        # Sample data
        corpus = [
            {"id": 0, "text": "Transformers attention mechanism",     "category": "ml",       "year": 2017},
            {"id": 1, "text": "BERT pre-training objectives",         "category": "ml",       "year": 2019},
            {"id": 2, "text": "PostgreSQL B-tree indexing",           "category": "database", "year": 2020},
            {"id": 3, "text": "vLLM PagedAttention serving",          "category": "ml",       "year": 2023},
            {"id": 4, "text": "Qdrant filtered vector search",        "category": "ml",       "year": 2023},
            {"id": 5, "text": "MySQL query optimization",             "category": "database", "year": 2021},
            {"id": 6, "text": "LLM fine-tuning with PEFT",            "category": "ml",       "year": 2023},
        ]

        def embed(text: str, d: int = 32) -> list:
            rng = np.random.default_rng(sum(ord(c) for c in text) % (2**32))
            v = rng.random(d).astype("float32")
            return (v / np.linalg.norm(v)).tolist()

        client.upsert("tech_docs", points=[
            PointStruct(id=d["id"], vector=embed(d["text"]),
                        payload={"text": d["text"], "category": d["category"], "year": d["year"]})
            for d in corpus
        ])

        query_vec = embed("neural network transformer model")

        # No filter
        all_results = client.search("tech_docs", query_vector=query_vec, limit=3)
        print("Top-3 (no filter):")
        for r in all_results:
            print(f"  [{r.score:.4f}] {r.payload['text']} ({r.payload['category']}, {r.payload['year']})")

        # Filter: ml only, year >= 2023
        filtered = client.search(
            "tech_docs",
            query_vector=query_vec,
            query_filter=Filter(must=[
                FieldCondition("category", match=MatchValue(value="ml")),
                FieldCondition("year",     range=Range(gte=2023)),
            ]),
            limit=3,
        )
        print("\nTop-3 (category=ml, year>=2023):")
        for r in filtered:
            print(f"  [{r.score:.4f}] {r.payload['text']}")

    except ImportError:
        print("qdrant-client not installed: pip install qdrant-client")
        print("\nQdrant pattern:")
        print("  client = QdrantClient(':memory:')")
        print("  client.create_collection('c', vectors_config=VectorParams(size=384, distance=Distance.COSINE))")
        print("  client.upsert('c', points=[PointStruct(id=0, vector=v, payload={'cat': 'ml'})])")
        print("  results = client.search('c', query_vector=q, limit=5,")
        print("                          query_filter=Filter(must=[FieldCondition('cat', match=MatchValue('ml'))]))")


# ── Comparison table ──────────────────────────────────────────────────────────

def print_vector_db_comparison():
    print("\n" + "=" * 70)
    print("Vector Database Comparison")
    print("=" * 70)
    rows = [
        ("FAISS",    "Library (no server)", "ANN only", "No",  "No",   "~200M (RAM-bound)"),
        ("Qdrant",   "Server (Rust)",        "HNSW+filter", "Yes", "Yes",  "~500M/node"),
        ("Milvus",   "Distributed (K8s)",    "HNSW/IVF/DISKANN", "Yes", "Yes", "Billions"),
        ("Weaviate", "Server + cloud",       "HNSW+BM25", "Yes", "Yes", "~500M/node"),
        ("Pinecone", "SaaS",                "Managed", "Yes",  "Yes", "Billions (managed)"),
    ]
    print(f"  {'System':<12} {'Type':<22} {'Index':<22} {'Filter':>6} {'API':>4} {'Scale'}")
    print("  " + "-" * 75)
    for r in rows:
        print(f"  {r[0]:<12} {r[1]:<22} {r[2]:<22} {r[3]:>6} {r[4]:>4} {r[5]}")


if __name__ == "__main__":
    demo_faiss_indices()
    demo_faiss_cosine()
    demo_qdrant_filtered_search()
    print_vector_db_comparison()
