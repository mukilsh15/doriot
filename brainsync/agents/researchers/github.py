import json
import os
from datetime import datetime, timedelta, timezone

import httpx
from openai import OpenAI

from brainsync.prompts import RELEVANCE_SCORER
from brainsync.state import NewsletterState, Signal

GITHUB_SEARCH_URL = "https://api.github.com/search/repositories"
LOOKBACK_DAYS = 7
MIN_STARS = 50
MIN_RELEVANCE = 0.4


def github_researcher(state: NewsletterState) -> dict:
    client = OpenAI(api_key=os.environ.get("GROQ_API_KEY", ""), base_url="https://api.groq.com/openai/v1")
    cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=LOOKBACK_DAYS)).strftime(
        "%Y-%m-%d"
    )
    headers = {"Authorization": f"Bearer {os.environ.get('GITHUB_TOKEN', '')}"}
    query = f"created:>{cutoff} stars:>{MIN_STARS}"

    response = httpx.get(
        GITHUB_SEARCH_URL,
        params={"q": query, "sort": "stars", "order": "desc", "per_page": 30},
        headers=headers,
    )
    response.raise_for_status()
    items = response.json()["items"]
    signals: list[Signal] = []

    for repo in items:
        title = repo["full_name"]
        description = repo.get("description") or ""
        stars = repo.get("stargazers_count", 0)
        summary = f"{description} ({stars:,} stars)"
        score = _score_relevance(client, title, summary)
        if score < MIN_RELEVANCE:
            continue
        signals.append(
            Signal(
                title=title,
                summary=summary,
                url=repo["html_url"],
                source="github",
                date=repo.get("created_at", ""),
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
