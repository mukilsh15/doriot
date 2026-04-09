import json
from unittest.mock import MagicMock, patch

from brainsync.state import Signal


MOCK_TRENDS_RESPONSE = json.dumps({
    "trends": [{"name": "Inference Efficiency", "description": "LLM serving tooling.", "signal_ids": []}]
})

MOCK_THESES_RESPONSE = json.dumps({
    "theses": [
        {
            "title": "The Inference Cost Squeeze",
            "category": "inference infrastructure",
            "point_of_view": "Every AI team is bleeding on inference costs.",
            "why_now": "Speculative decoding only became practical in 2025.",
            "signal_ids": [],
        }
    ]
})


@patch("brainsync.agents.delivery.httpx.post")
@patch("httpx.get")
@patch("brainsync.agents.researchers.techcrunch.feedparser.parse")
@patch("brainsync.agents.researchers.reddit.praw.Reddit")
@patch("brainsync.agents.researchers.reddit._score_relevance")
@patch("brainsync.agents.researchers.hn._score_relevance")
@patch("brainsync.agents.researchers.techcrunch._score_relevance")
@patch("brainsync.agents.researchers.github._score_relevance")
@patch("brainsync.agents.synthesizer.OpenAI")
@patch("brainsync.agents.thesis_writer.OpenAI")
def test_full_graph_runs_end_to_end(
    mock_thesis_openai_cls,
    mock_synth_openai_cls,
    mock_gh_score,
    mock_tc_score,
    mock_hn_score,
    mock_reddit_score,
    mock_reddit_cls,
    mock_tc_parse,
    mock_httpx_get,
    mock_resend_post,
):
    # All researchers return empty signals
    mock_reddit_score.return_value = 0.0
    mock_hn_score.return_value = 0.0
    mock_tc_score.return_value = 0.0
    mock_gh_score.return_value = 0.0

    mock_reddit_subreddit = MagicMock()
    mock_reddit_subreddit.hot.return_value = []
    mock_reddit_inst = MagicMock()
    mock_reddit_inst.subreddit.return_value = mock_reddit_subreddit
    mock_reddit_cls.return_value = mock_reddit_inst

    import feedparser as fp
    mock_tc_parse.return_value = fp.parse("")

    # Both HN and GitHub use httpx.get — route by URL
    def httpx_get_side_effect(url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        if "algolia" in url:
            resp.json.return_value = {"hits": []}
        else:
            resp.json.return_value = {"items": []}
        return resp

    mock_httpx_get.side_effect = httpx_get_side_effect

    mock_synth_client = MagicMock()
    mock_synth_resp = MagicMock()
    mock_synth_resp.choices = [MagicMock(message=MagicMock(content=MOCK_TRENDS_RESPONSE))]
    mock_synth_client.chat.completions.create.return_value = mock_synth_resp
    mock_synth_openai_cls.return_value = mock_synth_client

    mock_thesis_client = MagicMock()
    mock_thesis_resp = MagicMock()
    mock_thesis_resp.choices = [MagicMock(message=MagicMock(content=MOCK_THESES_RESPONSE))]
    mock_thesis_client.chat.completions.create.return_value = mock_thesis_resp
    mock_thesis_openai_cls.return_value = mock_thesis_client

    mock_resend_resp = MagicMock()
    mock_resend_resp.raise_for_status.return_value = None
    mock_resend_resp.json.return_value = {"id": "email_123"}
    mock_resend_post.return_value = mock_resend_resp

    import os
    with patch.dict(os.environ, {"RESEND_API_KEY": "test_key", "NEWSLETTER_RECIPIENT_EMAIL": "test@example.com"}):
        from brainsync.graph import build_graph

        graph = build_graph()
        final_state = graph.invoke({"run_date": "2026-04-05", "raw_signals": []})

    assert final_state["sent"] is True
    assert len(final_state["investment_theses"]) == 1
    assert final_state["newsletter_html"] != ""


def test_dry_run_excludes_delivery_node():
    from brainsync.graph import build_graph

    graph_with = build_graph(include_delivery=True)
    graph_without = build_graph(include_delivery=False)

    nodes_with = set(graph_with.nodes)
    nodes_without = set(graph_without.nodes)

    assert "delivery" in nodes_with
    assert "delivery" not in nodes_without
