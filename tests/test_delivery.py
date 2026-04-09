import os
from unittest.mock import MagicMock, patch


def test_delivery_sends_email(monkeypatch):
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    monkeypatch.setenv("NEWSLETTER_RECIPIENT_EMAIL", "test@example.com")

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"id": "email_123"}

    with patch("brainsync.agents.delivery.httpx.post", return_value=mock_response) as mock_post:
        from brainsync.agents.delivery import delivery

        state = {
            "run_date": "2026-04-05",
            "raw_signals": [],
            "synthesized_trends": [],
            "investment_theses": [],
            "newsletter_html": "<h1>Test</h1>",
            "newsletter_text": "Test newsletter",
            "sent": False,
        }
        result = delivery(state)

        assert result["sent"] is True
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["json"]["to"] == ["test@example.com"]
        assert call_kwargs["json"]["html"] == "<h1>Test</h1>"


def test_delivery_uses_correct_resend_endpoint(monkeypatch):
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    monkeypatch.setenv("NEWSLETTER_RECIPIENT_EMAIL", "test@example.com")

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"id": "email_123"}

    with patch("brainsync.agents.delivery.httpx.post", return_value=mock_response) as mock_post:
        from brainsync.agents.delivery import delivery

        delivery({
            "run_date": "2026-04-05",
            "raw_signals": [],
            "synthesized_trends": [],
            "investment_theses": [],
            "newsletter_html": "<h1>Test</h1>",
            "newsletter_text": "Test",
            "sent": False,
        })

        url = mock_post.call_args.args[0]
        assert url == "https://api.resend.com/emails"
