import json
from pathlib import Path
from unittest.mock import MagicMock, patch

FIXTURES = Path(__file__).parent / "fixtures"


@patch("brainsync.agents.researchers.hn._score_relevance")
@patch("brainsync.agents.researchers.hn.httpx.get")
def test_hn_researcher_returns_signals(mock_get, mock_score):
    payload = json.loads((FIXTURES / "hn_items.json").read_text())
    mock_response = MagicMock()
    mock_response.json.return_value = payload
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    mock_score.side_effect = [0.9, 0.85, 0.0]

    from brainsync.agents.researchers.hn import hn_researcher

    result = hn_researcher({"run_date": "2026-04-05", "raw_signals": []})

    assert "raw_signals" in result
    signals = result["raw_signals"]
    assert len(signals) == 2
    assert all(s["source"] == "hn" for s in signals)
    assert all(s["relevance_score"] >= 0.4 for s in signals)


@patch("brainsync.agents.researchers.hn._score_relevance")
@patch("brainsync.agents.researchers.hn.httpx.get")
def test_hn_researcher_uses_item_url_as_fallback(mock_get, mock_score):
    payload = json.loads((FIXTURES / "hn_items.json").read_text())
    # Second item has no url — should fall back to HN item URL
    mock_response = MagicMock()
    mock_response.json.return_value = payload
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response
    mock_score.return_value = 0.9

    from brainsync.agents.researchers.hn import hn_researcher

    result = hn_researcher({"run_date": "2026-04-05", "raw_signals": []})
    urls = [s["url"] for s in result["raw_signals"]]
    assert any("news.ycombinator.com" in u for u in urls)
