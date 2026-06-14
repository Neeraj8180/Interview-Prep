"""
Ragas, DeepEval, and Arize Phoenix evaluation examples — Parts 35-37
ragas 0.2.x | deepeval 1.x | arize-phoenix 4.x | Python 3.11+

Install:
  pip install ragas deepeval arize-phoenix openinference-instrumentation-langchain
  pip install langchain langchain-openai faiss-cpu
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Part 1: Ragas — reference-free RAG evaluation (no real LLM needed)
# ─────────────────────────────────────────────────────────────────────

def demo_ragas_dataset():
    """Build and inspect a Ragas EvaluationDataset."""
    print("=" * 60)
    print("1. Ragas EvaluationDataset")
    print("=" * 60)

    try:
        from ragas import EvaluationDataset, SingleTurnSample

        samples = [
            SingleTurnSample(
                user_input="What is PagedAttention in vLLM?",
                retrieved_contexts=[
                    "PagedAttention stores the KV cache in non-contiguous "
                    "memory blocks, similar to OS virtual memory paging.",
                    "This allows vLLM to serve more requests simultaneously "
                    "by reducing memory fragmentation.",
                ],
                response=(
                    "PagedAttention is a memory management technique in vLLM "
                    "that divides the KV cache into non-contiguous blocks, "
                    "reducing waste and increasing throughput."
                ),
                reference=(
                    "PagedAttention manages KV cache in paged blocks to "
                    "reduce memory fragmentation in LLM serving."
                ),
            ),
            SingleTurnSample(
                user_input="What is ZeRO Stage 3 in DeepSpeed?",
                retrieved_contexts=[
                    "ZeRO Stage 3 partitions optimizer states, gradients, "
                    "AND model parameters across all GPUs."
                ],
                response=(
                    "ZeRO Stage 3 distributes optimizer states, gradients, "
                    "and model parameters across all GPUs, enabling training "
                    "of trillion-parameter models."
                ),
                reference=(
                    "ZeRO Stage 3 partitions all three memory components "
                    "(optimizer states, gradients, parameters) across GPUs."
                ),
            ),
        ]

        dataset = EvaluationDataset(samples=samples)
        df = dataset.to_pandas()
        print(f"  Dataset size: {len(df)} samples")
        print(f"  Columns: {list(df.columns)}")
        print(f"\n  Sample 1 query: {df['user_input'].iloc[0][:60]}...")
        print(f"  Sample 1 response chars: {len(df['response'].iloc[0])}")
        print(f"\n  Ragas metrics available:")
        print("    - Faithfulness         (reference-free)")
        print("    - AnswerRelevancy      (reference-free)")
        print("    - ContextPrecision     (needs reference)")
        print("    - ContextRecall        (needs reference)")
        print("    - AnswerCorrectness    (needs reference)")
    except ImportError:
        print("  ragas not installed: pip install ragas")

    _show_ragas_evaluate_api()


def _show_ragas_evaluate_api():
    print("""
  Full Ragas evaluate() pattern:
    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    llm   = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    embed = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    result = evaluate(
        dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextRecall()],
        llm=llm, embeddings=embed,
    )
    df = result.to_pandas()
    print(df[["user_input","faithfulness","answer_relevancy"]].describe())

  Faithfulness formula:
    score = verified_claims / total_claims
    (verified = can be inferred from retrieved_contexts)

  CI gate pattern:
    assert df["faithfulness"].mean() >= 0.85, "Faithfulness below threshold"
""")


def demo_ragas_testset_generator():
    """Show TestsetGenerator pattern for synthetic test creation."""
    print("\n" + "=" * 60)
    print("2. Ragas TestsetGenerator (conceptual)")
    print("=" * 60)
    print("""
  TestsetGenerator automatically creates diverse evaluation datasets
  from your document corpus — no manual annotation required.

  Pattern:
    from ragas.testset import TestsetGenerator
    from langchain_community.document_loaders import DirectoryLoader

    loader    = DirectoryLoader("./my_docs/", glob="**/*.md")
    documents = loader.load()

    generator = TestsetGenerator(llm=llm, embedding_model=embed)
    testset   = generator.generate_with_langchain_docs(
        documents,
        testset_size=100,
        # distributions control question variety:
        # simple(0.5) = factoid, reasoning(0.3) = multi-step, multi_context(0.2)
    )
    testset.to_pandas().to_csv("eval_testset.csv")

  Generated test case example:
    {
      "user_input": "How does PagedAttention reduce memory fragmentation?",
      "reference": "PagedAttention uses paged KV cache blocks, preventing ...",
      "reference_contexts": ["PagedAttention stores KV cache in non-contiguous..."]
    }

  This testset can then be used with evaluate() to benchmark your RAG pipeline.
""")


# ─────────────────────────────────────────────────────────────────────
# Part 2: DeepEval — pytest-style LLM testing
# ─────────────────────────────────────────────────────────────────────

def demo_deepeval_test_case():
    """Show DeepEval LLMTestCase and metric patterns."""
    print("\n" + "=" * 60)
    print("3. DeepEval LLMTestCase")
    print("=" * 60)

    try:
        from deepeval.test_case import LLMTestCase
        tc = LLMTestCase(
            input="What is the difference between ZeRO Stage 1, 2, and 3?",
            actual_output=(
                "ZeRO Stage 1 partitions optimizer states. "
                "Stage 2 also partitions gradients. "
                "Stage 3 additionally partitions model parameters."
            ),
            expected_output=(
                "ZeRO stages progressively partition more model states: "
                "optimizer states (S1), + gradients (S2), + parameters (S3)."
            ),
            context=[
                "ZeRO Stage 1: optimizer state partitioning only.",
                "ZeRO Stage 2: optimizer state + gradient partitioning.",
                "ZeRO Stage 3: full partitioning including model parameters.",
            ],
        )
        print(f"  input:          {tc.input[:60]}...")
        print(f"  actual_output:  {tc.actual_output[:60]}...")
        print(f"  context chunks: {len(tc.context)}")
    except ImportError:
        print("  deepeval not installed: pip install deepeval")

    _show_deepeval_metrics_api()


def _show_deepeval_metrics_api():
    print("""
  DeepEval metric patterns:

  1. FaithfulnessMetric — claims verified against context
    from deepeval.metrics import FaithfulnessMetric
    m = FaithfulnessMetric(threshold=0.85, model="gpt-4o-mini")
    m.measure(test_case)
    print(m.score, m.reason, m.success)

  2. GEval — custom natural language rubric
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCaseParams
    m = GEval(
        name="Conciseness",
        criteria="Answer is complete but under 3 sentences.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.INPUT],
        threshold=0.7,
    )

  3. Safety metrics
    from deepeval.metrics import BiasMetric, ToxicityMetric
    bias_m    = BiasMetric(threshold=0.5, model="gpt-4o")
    toxicity_m = ToxicityMetric(threshold=0.5, model="gpt-4o")

  4. Agent evaluation
    from deepeval.test_case import ToolCall
    from deepeval.metrics import ToolCorrectnessMetric
    agent_tc = LLMTestCase(
        input="Get weather in NYC",
        actual_output="It is 72°F",
        tools_called=[ToolCall(name="get_weather",
                               input_parameters={"location": "NYC"})],
        expected_tools=[ToolCall(name="get_weather",
                                 input_parameters={"location": "NYC"})],
    )
    m = ToolCorrectnessMetric(threshold=1.0)

  5. Batch evaluate
    from deepeval import evaluate
    results = evaluate([tc1, tc2], [faithfulness_m, geval_m])

  6. pytest integration
    import deepeval
    def test_rag():
        deepeval.assert_test(test_case, [FaithfulnessMetric(threshold=0.85)])
    # Run: deepeval test run test_rag.py
""")


def demo_deepeval_custom_metric():
    """Demonstrate a custom BaseMetric without LLM dependency."""
    print("\n" + "=" * 60)
    print("4. DeepEval Custom Metric (no LLM)")
    print("=" * 60)

    try:
        from deepeval.metrics import BaseMetric
        from deepeval.test_case import LLMTestCase

        class ResponseLengthMetric(BaseMetric):
            """Simple heuristic metric: score = 1 if 50 <= len <= 300 chars."""
            name = "ResponseLengthMetric"

            def __init__(self, min_chars=50, max_chars=300, threshold=0.5):
                self.min_chars = min_chars
                self.max_chars = max_chars
                self.threshold = threshold

            def measure(self, test_case: LLMTestCase) -> float:
                output = test_case.actual_output or ""
                length = len(output)
                if self.min_chars <= length <= self.max_chars:
                    self.score   = 1.0
                    self.reason  = f"Length {length} in range [{self.min_chars}, {self.max_chars}]"
                else:
                    self.score   = 0.0
                    self.reason  = f"Length {length} outside range [{self.min_chars}, {self.max_chars}]"
                self.success = self.score >= self.threshold
                return self.score

            async def a_measure(self, test_case, _callbacks=None):
                return self.measure(test_case)

            def is_successful(self):
                return self.success

        from deepeval.test_case import LLMTestCase
        tc = LLMTestCase(
            input="What is FAISS?",
            actual_output="FAISS is a library for efficient similarity search of dense vectors.",
        )
        metric = ResponseLengthMetric(min_chars=20, max_chars=200)
        metric.measure(tc)
        print(f"  Score:   {metric.score}")
        print(f"  Reason:  {metric.reason}")
        print(f"  Success: {metric.success}")

    except ImportError:
        print("  deepeval not installed: pip install deepeval")
        print("""
  Custom metric pattern:
    from deepeval.metrics import BaseMetric
    from deepeval.test_case import LLMTestCase

    class MyMetric(BaseMetric):
        name = "MyMetric"
        def __init__(self, threshold=0.5): self.threshold = threshold
        def measure(self, tc: LLMTestCase) -> float:
            self.score   = ...  # compute score 0.0 - 1.0
            self.reason  = "..."
            self.success = self.score >= self.threshold
            return self.score
        async def a_measure(self, tc, _=None): return self.measure(tc)
        def is_successful(self): return self.success
""")


# ─────────────────────────────────────────────────────────────────────
# Part 3: Arize Phoenix — LLM observability
# ─────────────────────────────────────────────────────────────────────

def demo_phoenix_instrumentation():
    """Show Phoenix setup and instrumentation patterns."""
    print("\n" + "=" * 60)
    print("5. Arize Phoenix — Instrumentation")
    print("=" * 60)

    try:
        import phoenix as px
        print(f"  Phoenix version: {px.__version__}")
        print("""
  Setup pattern:
    import phoenix as px
    from phoenix.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor

    # Start local Phoenix UI at http://localhost:6006
    px.launch_app()

    # Configure OTel tracer provider
    tracer_provider = register(
        project_name="rag-v2",
        endpoint="http://localhost:6006/v1/traces",
    )

    # Auto-instrument LangChain (MUST be before any LangChain usage)
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

    # Now ALL LangChain calls automatically emit spans — no other changes!
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini")
    result = llm.invoke("Hello")  # → span appears in Phoenix UI
""")
    except ImportError:
        print("  arize-phoenix not installed: pip install arize-phoenix")
        print("""
  Phoenix setup pattern (shown without install):

    import phoenix as px
    from phoenix.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor

    px.launch_app()  # start local UI at http://localhost:6006
    tracer_provider = register(
        project_name="my-rag-app",
        endpoint="http://localhost:6006/v1/traces",
    )
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    # → all LangChain calls now emit OTel spans to Phoenix
""")


def demo_manual_spans():
    """Show manual span creation for custom code paths."""
    print("\n" + "=" * 60)
    print("6. Manual OTel Spans for Custom Code")
    print("=" * 60)

    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

        # Create a provider with console output (for demo)
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        tracer   = provider.get_tracer("demo")

        import time
        with tracer.start_as_current_span("rag-pipeline") as root:
            root.set_attribute("user.query", "What is FAISS?")
            root.set_attribute("user.tier", "enterprise")

            with tracer.start_as_current_span("retriever") as retriever_span:
                retriever_span.set_attribute("retriever.type", "dense")
                retriever_span.set_attribute("retriever.top_k", 5)
                time.sleep(0.01)  # simulate retrieval
                retriever_span.set_attribute("retriever.docs_returned", 4)

            with tracer.start_as_current_span("llm-call") as llm_span:
                llm_span.set_attribute("llm.model_name", "gpt-4o-mini")
                llm_span.set_attribute("llm.token_count.prompt", 320)
                time.sleep(0.01)  # simulate LLM call
                llm_span.set_attribute("llm.token_count.completion", 85)

        print("  Span tree emitted to console (would go to Phoenix in production)")
        print("  Span kinds captured: rag-pipeline (CHAIN), retriever (RETRIEVER), llm-call (LLM)")

    except ImportError:
        print("  opentelemetry not installed: pip install opentelemetry-sdk")
        print("""
  Manual span pattern:
    from opentelemetry import trace
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("retriever") as span:
        span.set_attribute("retriever.type", "dense")
        span.set_attribute("retriever.top_k", 5)
        docs = retriever.search(query)
        span.set_attribute("retriever.docs_returned", len(docs))
""")


def demo_phoenix_inline_eval():
    """Show Phoenix inline evaluation pattern."""
    print("\n" + "=" * 60)
    print("7. Phoenix Inline Evaluations")
    print("=" * 60)
    print("""
  After collecting traces, run LLM judge evaluations inline:

    from phoenix.evals import (
        HallucinationEvaluator,
        QAEvaluator,
        RelevanceEvaluator,
        run_evals,
    )
    from phoenix.evals.models import OpenAIModel
    import phoenix as px

    # 1. Get collected traces as DataFrame
    client   = px.Client()
    spans_df = client.get_spans_dataframe(project_name="rag-v2")

    # 2. Run evaluations (judge LLM scores each span)
    judge = OpenAIModel(model="gpt-4o-mini")
    evals = run_evals(
        dataframe=spans_df,
        evaluators=[
            HallucinationEvaluator(judge),
            QAEvaluator(judge),
            RelevanceEvaluator(judge),
        ],
        provide_explanation=True,  # store judge reasoning
    )

    # 3. Upload scores back to Phoenix UI
    client.log_evaluations(evals)

    # 4. Inspect in Python
    print(evals["hallucination"].describe())
    print(evals["qa_correctness"].mean())

  Evaluators available out-of-the-box:
    HallucinationEvaluator  — detects claims not in context
    QAEvaluator             — checks if answer addresses question
    RelevanceEvaluator      — context relevance to query
    SummarizationEvaluator  — summary coverage + faithfulness
    ToxicityEvaluator       — harmful content detection
""")


def demo_phoenix_export_to_ragas():
    """Show the Phoenix → Ragas evaluation pipeline."""
    print("\n" + "=" * 60)
    print("8. Phoenix → Ragas Pipeline (Trace to Eval)")
    print("=" * 60)
    print("""
  Complete production evaluation loop:

  Step 1: Collect traces in Phoenix
    LangChainInstrumentor().instrument(tracer_provider=tp)
    # All production RAG calls auto-traced

  Step 2: Export traces as Ragas dataset
    import phoenix as px
    from ragas import EvaluationDataset, SingleTurnSample

    client   = px.Client()
    spans_df = client.get_spans_dataframe(project_name="rag-prod")

    # Filter to LLM spans with full context
    llm_spans = spans_df[spans_df["span_kind"] == "LLM"]

    # Convert to Ragas samples (field names depend on your span attributes)
    samples = []
    for _, row in llm_spans.iterrows():
        samples.append(SingleTurnSample(
            user_input=row.get("input.value", ""),
            retrieved_contexts=[row.get("retrieval.documents", "")],
            response=row.get("output.value", ""),
        ))
    dataset = EvaluationDataset(samples=samples)

  Step 3: Run Ragas evaluation
    from ragas import evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy
    result = evaluate(dataset, metrics=[Faithfulness(), AnswerRelevancy()],
                      llm=llm, embeddings=embed)
    df = result.to_pandas()

  Step 4: CI gate
    assert df["faithfulness"].mean() >= 0.85, "Regression detected!"

  Step 5: Log scores back to Phoenix
    client.log_evaluations(result_as_phoenix_format)
""")


if __name__ == "__main__":
    demo_ragas_dataset()
    demo_ragas_testset_generator()
    demo_deepeval_test_case()
    demo_deepeval_custom_metric()
    demo_phoenix_instrumentation()
    demo_manual_spans()
    demo_phoenix_inline_eval()
    demo_phoenix_export_to_ragas()

    print("\n" + "=" * 60)
    print("Parts 35-37 Summary: RAG Evaluation Stack")
    print("=" * 60)
    print("""
  Full evaluation stack:

  Ragas (Part 35)        — RAG-specific metrics (Faithfulness, ContextRecall)
    ↑ offline benchmark evaluation on curated testsets
    ↑ TestsetGenerator for synthetic benchmarks from corpus

  DeepEval (Part 36)     — Comprehensive pytest-style LLM testing
    ↑ GEval for custom domain rubrics
    ↑ BiasMetric + ToxicityMetric for safety CI gates
    ↑ ToolCorrectnessMetric for agent evaluation

  Arize Phoenix (Part 37) — Production tracing + observability
    ↑ OTel-based distributed tracing for all LLM components
    ↑ Inline evaluations on real production traces
    ↑ Export traces → Ragas dataset for offline analysis

  Recommended pipeline:
    1. Phoenix traces ALL production calls (automatic instrumentation)
    2. Phoenix inline evals score 1% sampled traffic nightly
    3. Ragas evaluates fixed testset in CI (every PR)
    4. DeepEval pytest tests block merges on quality/safety regression
    5. Low-scoring Phoenix traces fed back to fine-tuning pipeline
""")
