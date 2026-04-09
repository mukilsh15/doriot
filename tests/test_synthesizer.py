import json
from unittest.mock import MagicMock, patch

from brainsync.state import Signal

SAMPLE_SIGNALS: list[Signal] = [
    {
        "title": "vllm achieves 10x throughput improvement",
        "summary": "New speculative decoding approach reduces latency",
        "url": "https://github.com/vllm-project/vllm",
        "source": "github",
        "date": "2026-04-01T00:00:00",
        "relevance_score": 0.95,
    },
    {
        "title": "Inference cost is the new capex",
        "summary": "Discussion on why inference optimization is the next big opportunity",
        "url": "https://news.ycombinator.com/item?id=12345",
        "source": "hn",
        "date": "2026-04-02T00:00:00",
        "relevance_score": 0.9,
    },
    {
        "title": "DuckDB-based feature stores gaining traction",
        "summary": "Several startups adopting DuckDB for in-process feature computation",
        "url": "https://reddit.com/r/dataengineering/post",
        "source": "reddit",
        "date": "2026-04-03T00:00:00",
        "relevance_score": 0.85,
    },
]

MOCK_CLAUDE_RESPONSE = json.dumps({
    "trends": [
        {
            "name": "Inference Efficiency Tooling",
            "description": "Multiple signals point to growing activity in LLM inference optimization.",
            "signal_ids": [0, 1],
        },
        {
            "name": "In-Process Data Compute",
            "description": "DuckDB and similar tools displacing heavyweight orchestration for ML data pipelines.",
            "signal_ids": [2],
        },
    ]
})


@patch("brainsync.agents.synthesizer.OpenAI")
def test_synthesizer_returns_trends(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_message = MagicMock()
    mock_message.choices = [MagicMock(message=MagicMock(content=MOCK_CLAUDE_RESPONSE))]
    mock_client.chat.completions.create.return_value = mock_message

    from brainsync.agents.synthesizer import synthesizer

    state = {
        "run_date": "2026-04-05",
        "raw_signals": SAMPLE_SIGNALS,
        "synthesized_trends": [],
        "investment_theses": [],
        "newsletter_html": "",
        "newsletter_text": "",
        "sent": False,
    }
    result = synthesizer(state)

    assert "synthesized_trends" in result
    trends = result["synthesized_trends"]
    assert len(trends) == 2
    assert trends[0]["name"] == "Inference Efficiency Tooling"
    assert 0 in trends[0]["signal_ids"]


@patch("brainsync.agents.synthesizer.OpenAI")
def test_synthesizer_formats_signals_for_prompt(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_message = MagicMock()
    mock_message.choices = [MagicMock(message=MagicMock(content=MOCK_CLAUDE_RESPONSE))]
    mock_client.chat.completions.create.return_value = mock_message

    from brainsync.agents.synthesizer import synthesizer

    state = {
        "run_date": "2026-04-05",
        "raw_signals": SAMPLE_SIGNALS,
        "synthesized_trends": [],
        "investment_theses": [],
        "newsletter_html": "",
        "newsletter_text": "",
        "sent": False,
    }
    synthesizer(state)

    call_args = mock_client.chat.completions.create.call_args
    prompt_content = call_args.kwargs["messages"][0]["content"]
    assert "vllm achieves" in prompt_content
    assert "DuckDB" in prompt_content
