import json

import anthropic
import feedparser

from brainsync.prompts import RELEVANCE_SCORER
from brainsync.state import NewsletterState, Signal

FEED_URL = "https://techcrunch.com/feed/"
MIN_RELEVANCE = 0.4


def techcrunch_researcher(state: NewsletterState) -> dict:
    client = anthropic.Anthropic()
    feed = feedparser.parse(FEED_URL)
    signals: list[Signal] = []

    for entry in feed.entries:
        title = entry.get("title", "")
        summary = entry.get("summary", title)[:500]
        url = entry.get("link", "")
        date = entry.get("published", "")
        score = _score_relevance(client, title, summary)
        if score < MIN_RELEVANCE:
            continue
        signals.append(
            Signal(
                title=title,
                summary=summary,
                url=url,
                source="techcrunch",
                date=date,
                relevance_score=score,
            )
        )

    return {"raw_signals": signals}


def _score_relevance(client: anthropic.Anthropic, title: str, summary: str) -> float:
    prompt = RELEVANCE_SCORER.format(title=title, summary=summary)
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=50,
        messages=[{"role": "user", "content": prompt}],
    )
    result = json.loads(message.content[0].text)
    return float(result["score"])
