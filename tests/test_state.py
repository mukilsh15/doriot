from brainsync.state import Signal, Trend, Thesis, NewsletterState
import operator
from typing import get_type_hints, get_args, Annotated


def test_signal_fields():
    s: Signal = {
        "title": "Test",
        "summary": "A summary",
        "url": "https://example.com",
        "source": "reddit",
        "date": "2026-04-01T00:00:00",
        "relevance_score": 0.8,
    }
    assert s["source"] == "reddit"


def test_trend_fields():
    t: Trend = {
        "name": "Inference Optimization",
        "description": "Tooling to reduce inference latency",
        "signal_ids": [0, 1, 2],
    }
    assert t["signal_ids"] == [0, 1, 2]


def test_thesis_fields():
    th: Thesis = {
        "title": "The Inference Cost Squeeze",
        "point_of_view": "Inference is the new compute bottleneck.",
        "why_now": "GPU costs plateaued while model sizes exploded.",
        "signals": [],
        "category": "inference infrastructure",
    }
    assert th["category"] == "inference infrastructure"


def test_raw_signals_reducer():
    # raw_signals must use operator.add so parallel nodes can append without overwrite
    hints = get_type_hints(NewsletterState, include_extras=True)
    raw = hints["raw_signals"]
    args = get_args(raw)
    assert args[1] is operator.add, "raw_signals must use operator.add reducer"
