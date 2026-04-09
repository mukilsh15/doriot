import sys
sys.path.insert(0, "/Users/mukils/brainsync")

from brainsync.agents.formatter import formatter
from brainsync.state import Signal, Thesis
import pathlib

signals = [
    Signal(title="vllm-project/vllm: 10x throughput via speculative decoding", summary="New async prefill scheduling achieves 10x throughput on LLaMA-3-70B with 4x GPU parallelism.", url="https://github.com/vllm-project/vllm", source="github", date="2026-04-01T00:00:00", relevance_score=0.95),
    Signal(title="Show HN: Prefect 3.0 — AI-native pipeline orchestration", summary="Prefect ships a major version with native support for LLM pipelines, async task execution, and built-in retry semantics for flaky model calls.", url="https://news.ycombinator.com/item?id=100001", source="hn", date="2026-04-01T09:00:00", relevance_score=0.9),
    Signal(title="Anyscale raises $100M to scale Ray-based AI infrastructure", summary="Anyscale expands its managed Ray platform, targeting AI teams running distributed training and inference at scale.", url="https://techcrunch.com/2026/04/01/anyscale-raises/", source="techcrunch", date="2026-04-01T10:00:00", relevance_score=0.88),
    Signal(title="Why I switched from Airflow to DuckDB + dbt for our ML feature pipeline", summary="For sub-1B row feature tables, DuckDB in-process is 10x faster to operate than Airflow with none of the operational overhead.", url="https://reddit.com/r/dataengineering/post/duckdb", source="reddit", date="2026-04-02T14:00:00", relevance_score=0.85),
]

theses = [
    Thesis(title="The Inference Cost Squeeze Creates a $1B Tooling Market", category="inference infrastructure", point_of_view="Every AI product team is bleeding on inference costs, and the gap between what naive serving delivers and what's achievable with speculative decoding and async prefill is now 5-10x. The next infrastructure winner abstracts this complexity behind a simple API. This is the Datadog moment for LLM serving.", why_now="Speculative decoding only became production-stable in late 2025. Model sizes crossed a threshold where naive serving is economically unviable for any product at scale. The window for tooling is open now, before the hyperscalers absorb this layer.", signals=[signals[0], signals[2]]),
    Thesis(title="AI-Native Orchestration Displaces Airflow for the ML Stack", category="MLOps", point_of_view="Airflow was built for ETL, not for workflows where tasks are probabilistic, latencies are variable, and retries require semantic awareness. The ML stack needs an orchestrator that treats LLM calls as first-class primitives — with built-in handling for rate limits, token budgets, and partial outputs.", why_now="The volume of LLM-in-production deployments hit a critical mass in 2025 that made the impedance mismatch with existing orchestration intolerable. Prefect 3.0 and similar launches signal that the market is ready to pay for a purpose-built replacement.", signals=[signals[1], signals[3]]),
]

state = {"run_date": "2026-04-07", "raw_signals": signals, "synthesized_trends": [], "investment_theses": theses, "newsletter_html": "", "newsletter_text": "", "sent": False}
result = formatter(state)

pathlib.Path("/Users/mukils/brainsync/examples").mkdir(exist_ok=True)
pathlib.Path("/Users/mukils/brainsync/examples/sample-newsletter.html").write_text(result["newsletter_html"])
print("Written to examples/sample-newsletter.html")
