import json
from pathlib import Path
from unittest.mock import MagicMock, patch

FIXTURES = Path(__file__).parent / "fixtures"


@patch("brainsync.agents.researchers.github._score_relevance")
@patch("brainsync.agents.researchers.github.httpx.get")
def test_github_researcher_returns_signals(mock_get, mock_score):
    payload = json.loads((FIXTURES / "github_repos.json").read_text())
    mock_response = MagicMock()
    mock_response.json.return_value = payload
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response
    mock_score.side_effect = [0.95, 0.9, 0.05]

    from brainsync.agents.researchers.github import github_researcher

    result = github_researcher({"run_date": "2026-04-05", "raw_signals": []})

    assert "raw_signals" in result
    signals = result["raw_signals"]
    assert len(signals) == 2
    assert all(s["source"] == "github" for s in signals)


@patch("brainsync.agents.researchers.github._score_relevance")
@patch("brainsync.agents.researchers.github.httpx.get")
def test_github_researcher_signal_includes_stars_in_summary(mock_get, mock_score):
    payload = json.loads((FIXTURES / "github_repos.json").read_text())
    mock_response = MagicMock()
    mock_response.json.return_value = payload
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response
    mock_score.return_value = 0.9

    from brainsync.agents.researchers.github import github_researcher

    result = github_researcher({"run_date": "2026-04-05", "raw_signals": []})
    assert all("stars" in s["summary"] for s in result["raw_signals"])
