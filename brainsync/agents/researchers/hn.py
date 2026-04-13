import json
import os
import time
from datetime import datetime, timedelta, timezone

import httpx
from openai import OpenAI, RateLimitError

from brainsync.prompts import RELEVANCE_SCORER
from brainsync.state import NewsletterState, Signal

HN_API = "https://hn.algolia.com/api/v1/search"
MIN_POINTS = 50
MIN_RELEVANCE = 0.4
TAGS = "(story,show_hn,ask_hn)"


def hn_researcher(state: NewsletterState) -> dict:
    client = OpenAI(api_key=os.environ.get("GROQ_API_KEY", ""), base_url="https://api.groq.com/openai/v1")
    response = httpx.get(
        HN_API,
        params={
            "tags": TAGS,
            "numericFilters": f"points>{MIN_POINTS},created_at_i>{_week_ago_ts()}",
            "hitsPerPage": 50,
        },
    )
    response.raise_for_status()
    hits = response.json()["hits"]
    signals: list[Signal] = []

    for hit in hits:
        title = hit.get("title", "")
        url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit['objectID']}"
        summary = title
        score = _score_relevance(client, title, summary)
        if score < MIN_RELEVANCE:
            continue
        signals.append(
            Signal(
                title=title,
                summary=summary,
                url=url,
                source="hn",
                date=hit.get("created_at", ""),
                relevance_score=score,
            )
        )

    return {"raw_signals": signals}


def _week_ago_ts() -> int:
    return int((datetime.now(tz=timezone.utc) - timedelta(days=7)).timestamp())


def _score_relevance(client: OpenAI, title: str, summary: str) -> float:
    prompt = RELEVANCE_SCORER.format(title=title, summary=summary)
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                max_tokens=50,
                messages=[{"role": "user", "content": prompt}],
            )
            result = json.loads(response.choices[0].message.content)
            return float(result["score"])
        except RateLimitError:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt + 2)
