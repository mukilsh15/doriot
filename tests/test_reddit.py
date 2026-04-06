import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _make_mock_post(data: dict) -> MagicMock:
    post = MagicMock()
    post.title = data["title"]
    post.selftext = data["selftext"]
    post.permalink = data["permalink"]
    post.score = data["score"]
    post.created_utc = data["created_utc"]
    post.url = f"https://reddit.com{data['permalink']}"
    return post


@patch("brainsync.agents.researchers.reddit._score_relevance")
@patch("brainsync.agents.researchers.reddit.praw.Reddit")
def test_reddit_researcher_returns_signals(mock_reddit_cls, mock_score):
    posts_data = json.loads((FIXTURES / "reddit_posts.json").read_text())
    mock_subreddit = MagicMock()
    mock_subreddit.hot.return_value = [_make_mock_post(p) for p in posts_data]
    mock_reddit = MagicMock()
    mock_reddit.subreddit.return_value = mock_subreddit
    mock_reddit_cls.return_value = mock_reddit

    # First two posts score high, third scores low
    mock_score.side_effect = [0.9, 0.8, 0.0, 0.9, 0.8, 0.0, 0.9, 0.8, 0.0, 0.9, 0.8, 0.0]

    from brainsync.agents.researchers.reddit import reddit_researcher

    result = reddit_researcher({"run_date": "2026-04-05", "raw_signals": []})

    assert "raw_signals" in result
    signals = result["raw_signals"]
    # Low-relevance post filtered out
    assert all(s["relevance_score"] >= 0.4 for s in signals)
    assert all(s["source"] == "reddit" for s in signals)
    assert all("url" in s for s in signals)


@patch("brainsync.agents.researchers.reddit._score_relevance")
@patch("brainsync.agents.researchers.reddit.praw.Reddit")
def test_reddit_researcher_filters_low_score_posts(mock_reddit_cls, mock_score):
    posts_data = json.loads((FIXTURES / "reddit_posts.json").read_text())
    # Make first post have a low karma score
    posts_data[0]["score"] = 2

    mock_subreddit = MagicMock()
    mock_subreddit.hot.return_value = [_make_mock_post(p) for p in posts_data]
    mock_reddit = MagicMock()
    mock_reddit.subreddit.return_value = mock_subreddit
    mock_reddit_cls.return_value = mock_reddit
    mock_score.return_value = 0.9

    from brainsync.agents.researchers.reddit import reddit_researcher

    result = reddit_researcher({"run_date": "2026-04-05", "raw_signals": []})
    titles = [s["title"] for s in result["raw_signals"]]
    assert posts_data[0]["title"] not in titles
