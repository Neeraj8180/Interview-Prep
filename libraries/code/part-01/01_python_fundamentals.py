# Python 3.11+ | Part 1: Python Fundamentals
# Topic: Reference semantics, generators, decorators, async, profiling
# Run: python 01_python_fundamentals.py

"""
This script demonstrates the core Python concepts every ML engineer
needs before touching NumPy, PyTorch, or any other ML library.
"""

from __future__ import annotations

import asyncio
import copy
import functools
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, TypeVar

T = TypeVar("T")


# =============================================================
# 1. REFERENCE SEMANTICS: the most important thing to internalize
# =============================================================

def demo_references():
    print("\n--- 1. Reference Semantics ---")

    a = [1, 2, 3]
    b = a               # b is a NAME pointing at the same list object
    b.append(4)
    print(f"a after b.append(4): {a}")    # [1, 2, 3, 4] — same object!

    c = a.copy()        # shallow copy: new list, same element objects
    c.append(5)
    print(f"a after c.append(5): {a}")    # [1, 2, 3, 4] — unchanged

    # For nested structures, shallow copy shares inner objects
    nested = [[1, 2], [3, 4]]
    shallow = copy.copy(nested)
    deep = copy.deepcopy(nested)

    shallow[0].append(99)
    print(f"nested[0] after shallow[0].append: {nested[0]}")  # [1, 2, 99]

    deep[0].append(100)
    print(f"nested[0] after deep[0].append:    {nested[0]}")  # [1, 2, 99] — unchanged

    # id() shows the memory address — same id means same object
    x = "hello"
    y = x
    print(f"\n  id(x) == id(y): {id(x) == id(y)}")           # True


# =============================================================
# 2. GENERATORS: lazy evaluation for memory-efficient pipelines
# =============================================================

def demo_generators():
    print("\n--- 2. Generators ---")

    def squares(n: int) -> Iterator[int]:
        """Yield squares one at a time — O(1) memory regardless of n."""
        for i in range(n):
            yield i * i

    # Generator is consumed once; use list() if you need to replay
    gen = squares(5)
    print(f"Generator object: {gen}")           # <generator object ...>
    print(f"First value:  {next(gen)}")          # 0
    print(f"Second value: {next(gen)}")          # 1
    print(f"Rest as list: {list(gen)}")          # [4, 9, 16]

    # Generator expression (inline syntax)
    total = sum(x * x for x in range(1_000_000))  # never builds a list
    print(f"\nSum of squares to 1M: {total}")        # 333332833333500000

    # Batch generator — used in data loading pipelines
    def batched(items: list[T], size: int) -> Iterator[list[T]]:
        for i in range(0, len(items), size):
            yield items[i : i + size]

    for batch in batched(list(range(7)), 3):
        print(f"  Batch: {batch}")
    # Batch: [0, 1, 2]
    # Batch: [3, 4, 5]
    # Batch: [6]


# =============================================================
# 3. DECORATORS: extend behavior without modifying code
# =============================================================

def demo_decorators():
    print("\n--- 3. Decorators ---")

    def timer(func):
        """Wrap a function to print its execution time."""
        @functools.wraps(func)         # preserve __name__ and __doc__
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            print(f"  {func.__name__} took {elapsed*1000:.2f}ms")
            return result
        return wrapper

    @timer
    def compute(n: int) -> int:
        return sum(i for i in range(n))

    result = compute(100_000)
    print(f"  Result: {result}")

    # Decorator with arguments (factory pattern)
    def retry(max_attempts: int = 3):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except ValueError as e:
                        if attempt == max_attempts - 1:
                            raise
                        print(f"  Attempt {attempt + 1} failed: {e}, retrying...")
            return wrapper
        return decorator

    call_count = 0

    @retry(max_attempts=3)
    def flaky_api():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError(f"API error on attempt {call_count}")
        return "success"

    result = flaky_api()
    print(f"  Flaky API result: {result}")   # success (after 2 retries)


# =============================================================
# 4. CONTEXT MANAGERS: guaranteed setup and teardown
# =============================================================

def demo_context_managers():
    print("\n--- 4. Context Managers ---")

    @contextmanager
    def timer_ctx(label: str):
        """Time a block of code."""
        start = time.perf_counter()
        yield                                          # code inside with block runs here
        elapsed = time.perf_counter() - start
        print(f"  {label}: {elapsed*1000:.2f}ms")

    with timer_ctx("list comprehension"):
        data = [x * x for x in range(100_000)]

    with timer_ctx("generator expression"):
        total = sum(x * x for x in range(100_000))

    # Class-based context manager (same thing, more control)
    class ResourceManager:
        def __enter__(self):
            print("  Resource acquired")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            print("  Resource released")
            # return True to suppress exceptions
            return False

    with ResourceManager() as rm:
        print("  Using resource...")
    # Resource released even if an exception occurs inside


# =============================================================
# 5. DATACLASSES: clean configuration objects
# =============================================================

def demo_dataclasses():
    print("\n--- 5. Dataclasses ---")

    @dataclass
    class TrainingConfig:
        model_name: str = "bert-base"
        learning_rate: float = 2e-5
        batch_size: int = 32
        num_epochs: int = 3
        fp16: bool = False
        label_names: list[str] = field(default_factory=list)  # mutable default

        def __post_init__(self):
            if self.learning_rate <= 0:
                raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")

        @property
        def description(self) -> str:
            return f"{self.model_name} | lr={self.learning_rate} | bs={self.batch_size}"

    config = TrainingConfig(model_name="gpt2", fp16=True)
    print(f"  Config: {config}")
    print(f"  Description: {config.description}")

    # Immutable (frozen) dataclass
    from dataclasses import dataclass as dc

    @dc(frozen=True)
    class ModelKey:
        name: str
        version: int

    key = ModelKey("bert", 2)
    print(f"  ModelKey (hashable): {key}")
    print(f"  Can be used as dict key: {hash(key)}")


# =============================================================
# 6. ASYNC / AWAIT: concurrent I/O without threads
# =============================================================

async def demo_async():
    print("\n--- 6. Async / Await ---")

    async def fake_api_call(doc_id: str, latency: float = 0.05) -> str:
        """Simulates a network call to an embedding API."""
        await asyncio.sleep(latency)               # releases control to event loop
        return f"embedding({doc_id})"

    # Sequential: 5 × 50ms = 250ms
    start = time.perf_counter()
    results_seq = []
    for doc_id in [f"doc_{i}" for i in range(5)]:
        result = await fake_api_call(doc_id)
        results_seq.append(result)
    seq_time = time.perf_counter() - start

    # Concurrent: max(50ms, 50ms, ...) ≈ 50ms
    start = time.perf_counter()
    tasks = [fake_api_call(f"doc_{i}") for i in range(5)]
    results_conc = await asyncio.gather(*tasks)
    conc_time = time.perf_counter() - start

    print(f"  Sequential: {seq_time*1000:.0f}ms | Concurrent: {conc_time*1000:.0f}ms")
    print(f"  Speedup: {seq_time/conc_time:.1f}x")
    # Sequential: ~250ms | Concurrent: ~50ms | Speedup: ~5x


# =============================================================
# 7. MEMORY: understanding size differences
# =============================================================

def demo_memory():
    print("\n--- 7. Memory ---")

    # Python list vs what NumPy would use
    n = 1000
    py_list = list(range(n))

    list_size = sys.getsizeof(py_list)
    element_size = sys.getsizeof(py_list[0]) if py_list else 0
    total_py = list_size + n * element_size      # pointers + each int object

    print(f"  Python list of {n} ints: {total_py / 1024:.1f} KB")
    # "NumPy array of {n} float32 would be: {n * 4 / 1024:.1f} KB"
    print(f"  NumPy float32 array would be: {n * 4 / 1024:.1f} KB")
    print(f"  Ratio: {total_py / (n * 4):.1f}x larger for Python list")

    # __slots__ demonstration
    class WithDict:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    class WithSlots:
        __slots__ = ("x", "y")
        def __init__(self, x, y):
            self.x = x
            self.y = y

    obj_dict  = WithDict(1.0, 2.0)
    obj_slots = WithSlots(1.0, 2.0)
    print(f"\n  Instance with __dict__:  {sys.getsizeof(obj_dict)} bytes")
    print(f"  Instance with __slots__: {sys.getsizeof(obj_slots)} bytes")
    # Savings per instance: ~120-150 bytes
    # At 1 million instances: ~120-150 MB saved


# =============================================================
# 8. COLLECTIONS: the right tool for the right job
# =============================================================

def demo_collections():
    print("\n--- 8. Collections ---")

    # defaultdict: avoid KeyError when building frequency tables
    label_counts: defaultdict[str, int] = defaultdict(int)
    for label in ["cat", "dog", "cat", "cat", "bird", "dog"]:
        label_counts[label] += 1
    print(f"  Label counts: {dict(label_counts)}")

    # Counter: purpose-built frequency counting
    vocab = Counter("the quick brown fox jumps over the lazy dog".split())
    print(f"  Top 3 words: {vocab.most_common(3)}")

    # Membership testing: set vs list
    import random
    items = [str(i) for i in range(10_000)]
    targets = [str(random.randint(0, 9999)) for _ in range(1000)]

    items_list = items
    items_set  = set(items)

    start = time.perf_counter()
    list_hits = sum(1 for t in targets if t in items_list)
    list_time = time.perf_counter() - start

    start = time.perf_counter()
    set_hits = sum(1 for t in targets if t in items_set)
    set_time = time.perf_counter() - start

    print(f"\n  Membership test (1000 lookups in 10k items):")
    print(f"    list: {list_time*1000:.2f}ms | set: {set_time*1000:.2f}ms")
    print(f"    Speedup: {list_time/set_time:.0f}x")


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Part 1: Python Fundamentals — Runnable Demonstrations")
    print("=" * 60)

    demo_references()
    demo_generators()
    demo_decorators()
    demo_context_managers()
    demo_dataclasses()
    asyncio.run(demo_async())
    demo_memory()
    demo_collections()

    print("\n" + "=" * 60)
    print("All demonstrations complete.")
    print("=" * 60)
