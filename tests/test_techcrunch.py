from pathlib import Path
from unittest.mock import patch, MagicMock
import feedparser

FIXTURES = Path(__file__).parent / "fixtures"

# Parse the fixture XML once at import time, before any mocking occurs
_PARSED_FEED = feedparser.parse((FIXTURES / "techcrunch_feed.xml").read_text())


@patch("brainsync.agents.researchers.techcrunch._score_relevance")
@patch("brainsync.agents.researchers.techcrunch.feedparser.parse")
def test_techcrunch_researcher_returns_signals(mock_parse, mock_score):
    mock_parse.return_value = _PARSED_FEED
    mock_score.side_effect = [0.9, 0.85, 0.05]

    from brainsync.agents.researchers.techcrunch import techcrunch_researcher

    result = techcrunch_researcher({"run_date": "2026-04-05", "raw_signals": []})

    assert "raw_signals" in result
    signals = result["raw_signals"]
    assert len(signals) == 2
    assert all(s["source"] == "techcrunch" for s in signals)
    assert all(s["relevance_score"] >= 0.4 for s in signals)


@patch("brainsync.agents.researchers.techcrunch._score_relevance")
@patch("brainsync.agents.researchers.techcrunch.feedparser.parse")
def test_techcrunch_researcher_includes_link(mock_parse, mock_score):
    mock_parse.return_value = _PARSED_FEED
    mock_score.return_value = 0.9

    from brainsync.agents.researchers.techcrunch import techcrunch_researcher

    result = techcrunch_researcher({"run_date": "2026-04-05", "raw_signals": []})
    assert all(s["url"].startswith("https://") for s in result["raw_signals"])
