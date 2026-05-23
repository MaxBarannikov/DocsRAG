from prometheus_client import Counter, Histogram

rag_requests_total = Counter(
    "rag_requests_total",
    "Total RAG requests by endpoint",
    ["endpoint"],
)

rag_retrieval_duration_seconds = Histogram(
    "rag_retrieval_duration_seconds",
    "Retrieval stage duration",
    ["endpoint"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

rag_generation_duration_seconds = Histogram(
    "rag_generation_duration_seconds",
    "Generation stage duration",
    ["endpoint"],
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

rag_top_k = Histogram(
    "rag_top_k",
    "Number of chunks requested (top_k)",
    buckets=[1, 2, 3, 5, 10, 20],
)

rag_answer_length_chars = Histogram(
    "rag_answer_length_chars",
    "Answer length in characters",
    buckets=[100, 250, 500, 1000, 2000, 5000],
)
