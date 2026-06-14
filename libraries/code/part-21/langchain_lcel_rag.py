# LangChain LCEL + RAG — Part 21 code examples
# langchain 0.3.x | Python 3.11+
# Install: pip install langchain langchain-openai langchain-community

from typing import List


# ── 1. LCEL chain composition ─────────────────────────────────────────────────

def demo_lcel_chain():
    print("=" * 55)
    print("1. LCEL Chain Composition")
    print("=" * 55)

    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_openai import ChatOpenAI

        prompt = ChatPromptTemplate.from_template(
            "You are an ML expert. Explain in 2 sentences: {topic}"
        )
        llm   = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        chain = prompt | llm | StrOutputParser()

        topics = ["attention mechanism", "gradient descent", "vector embeddings"]
        for topic in topics:
            print(f"\n{topic.title()}:")
            for chunk in chain.stream({"topic": topic}):
                print(chunk, end="", flush=True)
            print()

    except ImportError:
        print("langchain-openai not installed")
        _show_lcel_concept()

    except Exception as e:
        print(f"API error: {e}")
        _show_lcel_concept()


def _show_lcel_concept():
    print("\nLCEL pattern (requires API key):")
    print("""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Compose with | operator
chain = (
    ChatPromptTemplate.from_template("Explain {topic} in 2 sentences.")
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

# Invoke
result = chain.invoke({"topic": "transformers"})

# Stream tokens
for chunk in chain.stream({"topic": "RAG"}):
    print(chunk, end="", flush=True)

# Batch (parallel)
results = chain.batch([{"topic": "RLHF"}, {"topic": "LoRA"}])

# Async
import asyncio
result = asyncio.run(chain.ainvoke({"topic": "embeddings"}))
""")


# ── 2. Prompt templates ───────────────────────────────────────────────────────

def demo_prompt_templates():
    print("\n" + "=" * 55)
    print("2. Prompt Templates")
    print("=" * 55)

    try:
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.messages import HumanMessage, AIMessage

        # System + user
        basic_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a {role}. Respond in {language}."),
            ("user", "{question}"),
        ])
        formatted = basic_prompt.format_messages(
            role="Python expert", language="English",
            question="What is a list comprehension?"
        )
        print("Basic prompt messages:")
        for msg in formatted:
            print(f"  [{msg.type}]: {msg.content[:80]}")

        # With history placeholder
        history_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder("history"),
            ("user", "{input}"),
        ])
        formatted2 = history_prompt.format_messages(
            history=[
                HumanMessage("My name is Alice"),
                AIMessage("Hello Alice!"),
            ],
            input="What is my name?",
        )
        print(f"\nHistory prompt: {len(formatted2)} messages")
        print(f"  Last: {formatted2[-1].content}")

    except ImportError:
        print("langchain-core not installed: pip install langchain-core")


# ── 3. RAG pipeline (without API calls) ──────────────────────────────────────

def demo_rag_pipeline():
    print("\n" + "=" * 55)
    print("3. RAG Pipeline")
    print("=" * 55)

    # Demonstrate the data flow without API calls
    sample_docs = [
        {"text": "Transformers use attention mechanism to relate tokens.", "id": "doc1"},
        {"text": "RAG combines retrieval with generation for factual accuracy.", "id": "doc2"},
        {"text": "Vector embeddings represent text as dense numerical vectors.", "id": "doc3"},
        {"text": "LangChain provides composable building blocks for LLM apps.", "id": "doc4"},
    ]

    query = "What is RAG?"

    # Simulate embedding similarity (using simple keyword overlap for demo)
    def simple_similarity(query: str, doc: dict) -> float:
        q_words = set(query.lower().split())
        d_words = set(doc["text"].lower().split())
        return len(q_words & d_words) / max(len(q_words), 1)

    scored = sorted(sample_docs, key=lambda d: simple_similarity(query, d), reverse=True)
    top_k  = scored[:2]

    print(f"Query: '{query}'")
    print("\nTop-2 retrieved docs:")
    for doc in top_k:
        print(f"  [{doc['id']}]: {doc['text']}")

    context = "\n".join(d["text"] for d in top_k)
    rag_prompt = f"""Answer based on context:

Context: {context}

Question: {query}
Answer:"""
    print(f"\nFormatted RAG prompt:\n{rag_prompt}")

    # With LangChain (requires API key)
    print("\nLangChain RAG chain pattern:")
    print("""
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())
retriever   = vectorstore.as_retriever(search_kwargs={"k": 4})

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | ChatOpenAI()
    | StrOutputParser()
)
answer = rag_chain.invoke("What is RAG?")
""")


# ── 4. Tool calling pattern ───────────────────────────────────────────────────

def demo_tools():
    print("\n" + "=" * 55)
    print("4. Tool Calling Pattern")
    print("=" * 55)

    try:
        from langchain_core.tools import tool

        @tool
        def get_weather(city: str) -> str:
            """Get the current weather for a city."""
            return f"Weather in {city}: 22°C, sunny"

        @tool
        def calculate(expression: str) -> str:
            """Evaluate a math expression. Example: '2 + 2' or '10 * 5'."""
            try:
                allowed = {k: v for k, v in __builtins__.items()
                           if k in ('abs', 'round', 'min', 'max')} if isinstance(__builtins__, dict) else {}
                return str(eval(expression, {"__builtins__": {}}, {}))
            except Exception as e:
                return f"Error: {e}"

        tools = [get_weather, calculate]
        print(f"Defined {len(tools)} tools:")
        for t in tools:
            print(f"  @tool {t.name}: {t.description[:60]}")
            print(f"    Args: {list(t.args.keys())}")

        # Test tools directly
        print(f"\nDirect invocation:")
        print(f"  get_weather('Paris') = {get_weather.invoke({'city': 'Paris'})}")
        print(f"  calculate('2 ** 10') = {calculate.invoke({'expression': '2 ** 10'})}")

    except ImportError:
        print("langchain-core not installed")


if __name__ == "__main__":
    demo_lcel_chain()
    demo_prompt_templates()
    demo_rag_pipeline()
    demo_tools()
