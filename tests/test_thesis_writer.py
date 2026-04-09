import json
from unittest.mock import MagicMock, patch

from brainsync.state import Signal, Trend

SAMPLE_SIGNALS: list[Signal] = [
    {
        "title": "vllm achieves 10x throughput improvement",
        "summary": "New speculative decoding approach",
        "url": "https://github.com/vllm-project/vllm",
        "source": "github",
        "date": "2026-04-01T00:00:00",
        "relevance_score": 0.95,
    },
    {
        "title": "Inference cost is the new capex",
        "summary": "Discussion on inference optimization",
        "url": "https://news.ycombinator.com/item?id=12345",
        "source": "hn",
        "date": "2026-04-02T00:00:00",
        "relevance_score": 0.9,
    },
]

SAMPLE_TRENDS: list[Trend] = [
    {
        "name": "Inference Efficiency Tooling",
        "description": "Multiple signals point to growing activity in LLM inference optimization.",
        "signal_ids": [0, 1],
    }
]

MOCK_CLAUDE_RESPONSE = json.dumps({
    "theses": [
        {
            "title": "The Inference Cost Squeeze Creates a $1B Tooling Market",
            "category": "inference infrastructure",
            "point_of_view": "Every AI product team is bleeding on inference costs. The next infrastructure winner abstracts away the complexity of speculative decoding, batching, and KV cache management behind a simple API.",
            "why_now": "Model sizes crossed a threshold where naive serving approaches are economically unviable. Speculative decoding only became practical in 2025 — the window for tooling is now.",
            "signal_ids": [0, 1],
        }
    ]
})


@patch("brainsync.agents.thesis_writer.anthropic.Anthropic")
def test_thesis_writer_returns_theses(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text=MOCK_CLAUDE_RESPONSE)]
    mock_client.messages.create.return_value = mock_message

    from brainsync.agents.thesis_writer import thesis_writer

    state = {
        "run_date": "2026-04-05",
        "raw_signals": SAMPLE_SIGNALS,
        "synthesized_trends": SAMPLE_TRENDS,
        "investment_theses": [],
        "newsletter_html": "",
        "newsletter_text": "",
        "sent": False,
    }
    result = thesis_writer(state)

    assert "investment_theses" in result
    theses = result["investment_theses"]
    assert len(theses) == 1
    assert theses[0]["title"] == "The Inference Cost Squeeze Creates a $1B Tooling Market"
    assert theses[0]["category"] == "inference infrastructure"


@patch("brainsync.agents.thesis_writer.anthropic.Anthropic")
def test_thesis_writer_resolves_signal_ids_to_signals(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text=MOCK_CLAUDE_RESPONSE)]
    mock_client.messages.create.return_value = mock_message

    from brainsync.agents.thesis_writer import thesis_writer

    state = {
        "run_date": "2026-04-05",
        "raw_signals": SAMPLE_SIGNALS,
        "synthesized_trends": SAMPLE_TRENDS,
        "investment_theses": [],
        "newsletter_html": "",
        "newsletter_text": "",
        "sent": False,
    }
    result = thesis_writer(state)

    thesis = result["investment_theses"][0]
    # signal_ids [0, 1] should be resolved to actual Signal objects
    assert len(thesis["signals"]) == 2
    assert thesis["signals"][0]["title"] == "vllm achieves 10x throughput improvement"
    assert thesis["signals"][1]["source"] == "hn"
