from brainsync.state import Signal, Thesis

SAMPLE_SIGNALS: list[Signal] = [
    {
        "title": "vllm achieves 10x throughput",
        "summary": "Speculative decoding",
        "url": "https://github.com/vllm-project/vllm",
        "source": "github",
        "date": "2026-04-01T00:00:00",
        "relevance_score": 0.95,
    }
]

SAMPLE_THESES: list[Thesis] = [
    {
        "title": "The Inference Cost Squeeze",
        "category": "inference infrastructure",
        "point_of_view": "Every AI team is bleeding on inference costs.",
        "why_now": "Speculative decoding only became practical in 2025.",
        "signals": SAMPLE_SIGNALS,
    }
]


def test_formatter_returns_html_and_text():
    from brainsync.agents.formatter import formatter

    state = {
        "run_date": "2026-04-05",
        "raw_signals": SAMPLE_SIGNALS,
        "synthesized_trends": [],
        "investment_theses": SAMPLE_THESES,
        "newsletter_html": "",
        "newsletter_text": "",
        "sent": False,
    }
    result = formatter(state)

    assert "newsletter_html" in result
    assert "newsletter_text" in result
    assert len(result["newsletter_html"]) > 100
    assert len(result["newsletter_text"]) > 50


def test_formatter_html_contains_thesis_title():
    from brainsync.agents.formatter import formatter

    state = {
        "run_date": "2026-04-05",
        "raw_signals": SAMPLE_SIGNALS,
        "synthesized_trends": [],
        "investment_theses": SAMPLE_THESES,
        "newsletter_html": "",
        "newsletter_text": "",
        "sent": False,
    }
    result = formatter(state)

    assert "The Inference Cost Squeeze" in result["newsletter_html"]
    assert "2026-04-05" in result["newsletter_html"]
    assert "vllm" in result["newsletter_html"]


def test_formatter_text_contains_why_now():
    from brainsync.agents.formatter import formatter

    state = {
        "run_date": "2026-04-05",
        "raw_signals": SAMPLE_SIGNALS,
        "synthesized_trends": [],
        "investment_theses": SAMPLE_THESES,
        "newsletter_html": "",
        "newsletter_text": "",
        "sent": False,
    }
    result = formatter(state)

    assert "Speculative decoding only became practical" in result["newsletter_text"]
