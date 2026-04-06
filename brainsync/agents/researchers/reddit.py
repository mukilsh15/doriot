import json
import os
from datetime import datetime, timedelta, timezone

import anthropic
import praw

from brainsync.prompts import RELEVANCE_SCORER
from brainsync.state import NewsletterState, Signal

SUBREDDITS = ["MachineLearning", "dataengineering", "LocalLLaMA", "mlops"]
MIN_KARMA = 10
LOOKBACK_DAYS = 7
MIN_RELEVANCE = 0.4


def reddit_researcher(state: NewsletterState) -> dict:
    reddit = praw.Reddit(
        client_id=os.environ.get("REDDIT_CLIENT_ID", ""),
        client_secret=os.environ.get("REDDIT_CLIENT_SECRET", ""),
        user_agent="brainsync/1.0",
    )
    client = anthropic.Anthropic()
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    signals: list[Signal] = []

    for sub in SUBREDDITS:
        for post in reddit.subreddit(sub).hot(limit=25):
            if post.score < MIN_KARMA:
                continue
            created = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
            if created < cutoff:
                continue
            summary = post.selftext[:500] if post.selftext else post.url
            score = _score_relevance(client, post.title, summary)
            if score < MIN_RELEVANCE:
                continue
            signals.append(
                Signal(
                    title=post.title,
                    summary=summary,
                    url=f"https://reddit.com{post.permalink}",
                    source="reddit",
                    date=created.isoformat(),
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
