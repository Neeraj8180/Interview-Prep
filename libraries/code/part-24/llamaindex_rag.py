# LlamaIndex RAG — Part 24 code examples
# llama-index 0.12.x | Python 3.11+
# Install: pip install llama-index llama-index-llms-openai llama-index-embeddings-openai

from typing import List


def demo_documents_and_nodes():
    """Show Document → Node → Index pipeline."""
    print("=" * 55)
    print("1. Documents → Nodes → Index")
    print("=" * 55)

    try:
        from llama_index.core import Document, VectorStoreIndex, Settings
        from llama_index.core.node_parser import SentenceSplitter

        # Sample documents
        docs = [
            Document(
                text=(
                    "The Transformer model introduced in 'Attention is All You Need' (2017) "
                    "uses multi-head self-attention to process sequences in parallel. "
                    "The attention formula is: softmax(QK^T / sqrt(d_k)) * V. "
                    "Transformers replaced RNNs in most NLP tasks by 2020."
                ),
                metadata={"title": "Transformers", "year": 2017},
            ),
            Document(
                text=(
                    "RAG (Retrieval-Augmented Generation) combines a retrieval system "
                    "with a language model. Documents are split into chunks, embedded, "
                    "and stored in a vector database. At query time, the top-k most similar "
                    "chunks are retrieved and provided as context to the LLM."
                ),
                metadata={"title": "RAG Overview"},
            ),
            Document(
                text=(
                    "LlamaIndex provides multiple index types: VectorStoreIndex for semantic "
                    "search, SummaryIndex for synthesizing whole documents, and KeywordTableIndex "
                    "for BM25 keyword matching. It also supports hierarchical node parsing for "
                    "retrieval with parent-context."
                ),
                metadata={"title": "LlamaIndex Indices"},
            ),
        ]

        # Parse into nodes
        parser = SentenceSplitter(chunk_size=150, chunk_overlap=30)
        nodes  = parser.get_nodes_from_documents(docs)

        print(f"Documents: {len(docs)}")
        print(f"Nodes:     {len(nodes)}")
        print("\nSample nodes:")
        for n in nodes[:3]:
            print(f"  [{n.metadata.get('title', '?')}]: {n.text[:70]}...")

        return docs, nodes

    except ImportError:
        print("llama-index not installed: pip install llama-index")
        return [], []


def demo_vector_query(docs: list):
    """Build VectorStoreIndex and query it."""
    print("\n" + "=" * 55)
    print("2. VectorStoreIndex Query")
    print("=" * 55)

    if not docs:
        print("Skipping — docs not available")
        return

    try:
        from llama_index.core import VectorStoreIndex, Settings
        from llama_index.llms.openai import OpenAI
        from llama_index.embeddings.openai import OpenAIEmbedding

        Settings.llm         = OpenAI(model="gpt-4o-mini")
        Settings.embed_model = OpenAIEmbedding()
        Settings.chunk_size  = 200

        index = VectorStoreIndex.from_documents(docs)
        qe    = index.as_query_engine(similarity_top_k=3)

        queries = [
            "What is the attention formula?",
            "How does RAG work?",
        ]
        for q in queries:
            resp = qe.query(q)
            print(f"\nQ: {q}")
            print(f"A: {resp.response[:150]}")
            print(f"Sources: {[n.metadata.get('title') for n in resp.source_nodes]}")

    except ImportError:
        print("OpenAI integration not available.")
        print("Install: pip install llama-index-llms-openai llama-index-embeddings-openai")
        _show_pattern()
    except Exception as e:
        print(f"API error: {e}")
        _show_pattern()


def _show_pattern():
    print("\nLlamaIndex query pattern:")
    print("""
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.llm        = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding()

index    = VectorStoreIndex.from_documents(docs)
qe       = index.as_query_engine(similarity_top_k=5)
response = qe.query("Your question here?")

print(response.response)
for node in response.source_nodes:
    print(f"  - {node.metadata['title']}: score={node.score:.3f}")
""")


def demo_index_comparison():
    """Show the different index types conceptually."""
    print("\n" + "=" * 55)
    print("3. LlamaIndex Index Types Comparison")
    print("=" * 55)

    comparison = {
        "VectorStoreIndex": {
            "query_type": "Semantic similarity",
            "best_for":   "Find relevant chunks for a specific question",
            "mechanism":  "Embed query → cosine similarity → top-k nodes",
            "example_q":  "'What is attention?' → finds attention-related chunks",
        },
        "SummaryIndex": {
            "query_type": "Synthesize all content",
            "best_for":   "Summarize or compare multiple documents",
            "mechanism":  "Read all nodes → LLM synthesizes",
            "example_q":  "'Summarize the document' → LLM reads all chunks",
        },
        "KeywordTableIndex": {
            "query_type": "Keyword / BM25 matching",
            "best_for":   "Exact term matching, acronyms, specific codes",
            "mechanism":  "Extract keywords → inverted index → match",
            "example_q":  "'RFC 7231' → exact keyword match",
        },
    }

    for name, info in comparison.items():
        print(f"\n{name}")
        for k, v in info.items():
            print(f"  {k:15}: {v}")


def demo_node_relationships():
    """Show hierarchical node structure."""
    print("\n" + "=" * 55)
    print("4. Hierarchical Node Parsing")
    print("=" * 55)

    print("Node hierarchy for better RAG:")
    print("""
  Document (2048 tokens)
  ├── Chunk A (512 tokens)
  │   ├── Sentence A1 (128 tokens)  ← leaf: retrieved by semantic search
  │   └── Sentence A2 (128 tokens)
  └── Chunk B (512 tokens)
      ├── Sentence B1 (128 tokens)
      └── Sentence B2 (128 tokens)

AutoMergingRetriever logic:
  - Match at sentence level (128 tokens) → precise similarity
  - If 3+ sibling sentences from same parent retrieved → return PARENT (512 tokens)
  - Parent gives LLM full context without retrieving random paragraphs

Result: Better answer quality with same retrieval cost.
""")
    print("Code:")
    print("""
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.retrievers import AutoMergingRetriever

parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])
nodes  = parser.get_nodes_from_documents(documents)
index  = VectorStoreIndex(nodes)

base_retriever = index.as_retriever(similarity_top_k=12)
retriever      = AutoMergingRetriever(base_retriever, index.storage_context)
""")


if __name__ == "__main__":
    docs, nodes = demo_documents_and_nodes()
    demo_vector_query(docs)
    demo_index_comparison()
    demo_node_relationships()
