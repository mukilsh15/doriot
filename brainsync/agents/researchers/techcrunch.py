import json
import os

import feedparser
from openai import OpenAI

from brainsync.prompts import RELEVANCE_SCORER
from brainsync.state import NewsletterState, Signal

FEED_URL = "https://techcrunch.com/feed/"
MIN_RELEVANCE = 0.4


def techcrunch_researcher(state: NewsletterState) -> dict:
    client = OpenAI(api_key=os.environ.get("GROQ_API_KEY", ""), base_url="https://api.groq.com/openai/v1")
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


def _score_relevance(client: OpenAI, title: str, summary: str) -> float:
    prompt = RELEVANCE_SCORER.format(title=title, summary=summary)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=50,
        messages=[{"role": "user", "content": prompt}],
    )
    result = json.loads(response.choices[0].message.content)
    return float(result["score"])
