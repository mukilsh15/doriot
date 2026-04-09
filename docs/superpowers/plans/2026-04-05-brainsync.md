# BrainSync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a LangGraph multi-agent system that scrapes AI/data infrastructure signals weekly, synthesizes investment theses via Claude, and emails a newsletter.

**Architecture:** Four researcher nodes run in parallel (Reddit, HN, TechCrunch, GitHub), feeding into a synthesizer → thesis writer → formatter → email delivery pipeline. Shared `NewsletterState` uses an `operator.add` reducer so parallel researchers can write to `raw_signals` without conflicts.

**Tech Stack:** Python 3.12, LangGraph, Anthropic SDK (claude-opus-4-6), PRAW, feedparser, httpx, Jinja2, pytest, GitHub Actions

---

## File Map

| File | Responsibility |
|------|---------------|
| `brainsync/state.py` | `Signal`, `Trend`, `Thesis`, `NewsletterState` TypedDicts |
| `brainsync/prompts.py` | All Claude prompt templates as string constants |
| `brainsync/agents/researchers/reddit.py` | PRAW-based researcher, returns `{"raw_signals": [...]}` |
| `brainsync/agents/researchers/hn.py` | HN Algolia API researcher |
| `brainsync/agents/researchers/techcrunch.py` | RSS feedparser researcher |
| `brainsync/agents/researchers/github.py` | GitHub Search API researcher |
| `brainsync/agents/synthesizer.py` | Clusters signals into `Trend` objects via Claude |
| `brainsync/agents/thesis_writer.py` | Produces `Thesis` objects with POV + why-now via Claude |
| `brainsync/agents/formatter.py` | Renders HTML + plain text newsletter via Jinja2 |
| `brainsync/agents/delivery.py` | Sends email via Resend REST API |
| `brainsync/graph.py` | LangGraph `StateGraph` wiring + `--dry-run` CLI entry point |
| `brainsync/templates/newsletter.html.j2` | Jinja2 HTML email template |
| `tests/fixtures/reddit_posts.json` | Recorded Reddit API response for tests |
| `tests/fixtures/hn_items.json` | Recorded HN Algolia API response |
| `tests/fixtures/techcrunch_feed.xml` | Recorded RSS feed |
| `tests/fixtures/github_repos.json` | Recorded GitHub Search API response |
| `tests/test_reddit.py` | Tests for reddit researcher |
| `tests/test_hn.py` | Tests for HN researcher |
| `tests/test_techcrunch.py` | Tests for TechCrunch researcher |
| `tests/test_github.py` | Tests for GitHub researcher |
| `tests/test_synthesizer.py` | Tests for synthesizer |
| `tests/test_thesis_writer.py` | Tests for thesis writer |
| `tests/test_formatter.py` | Tests for formatter |
| `tests/test_delivery.py` | Tests for delivery |
| `tests/test_graph.py` | Integration test for full graph with all agents mocked |
| `.github/workflows/newsletter.yml` | Monday 8am UTC cron trigger |
| `pyproject.toml` | Dependencies and project config |
| `.env.example` | Required environment variables |
| `README.md` | Technical blog-post-style documentation |
| `examples/sample-newsletter.html` | Committed sample output |

---

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `brainsync/__init__.py`
- Create: `brainsync/agents/__init__.py`
- Create: `brainsync/agents/researchers/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/fixtures/` (directory)
- Create: `.env.example`

- [ ] **Step 1: Initialize git repo**

```bash
cd /Users/mukils/brainsync
git init
```

- [ ] **Step 2: Write pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "brainsync"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "langgraph>=0.2",
    "anthropic>=0.40",
    "praw>=7.7",
    "feedparser>=6.0",
    "httpx>=0.27",
    "jinja2>=3.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-mock>=3.12",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Create package init files**

`brainsync/__init__.py` — empty file.
`brainsync/agents/__init__.py` — empty file.
`brainsync/agents/researchers/__init__.py` — empty file.
`tests/__init__.py` — empty file.

- [ ] **Step 4: Create .env.example**

```bash
ANTHROPIC_API_KEY=sk-ant-...
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
GITHUB_TOKEN=ghp_...
RESEND_API_KEY=re_...
NEWSLETTER_RECIPIENT_EMAIL=you@example.com
```

- [ ] **Step 5: Install dependencies**

```bash
pip install -e ".[dev]"
```

Expected: All packages install without errors.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml brainsync/ tests/ .env.example
git commit -m "chore: project scaffold"
```

---

## Task 2: State Types

**Files:**
- Create: `brainsync/state.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_state.py`:

```python
from brainsync.state import Signal, Trend, Thesis, NewsletterState
import operator
from typing import get_type_hints, get_args, Annotated


def test_signal_fields():
    s: Signal = {
        "title": "Test",
        "summary": "A summary",
        "url": "https://example.com",
        "source": "reddit",
        "date": "2026-04-01T00:00:00",
        "relevance_score": 0.8,
    }
    assert s["source"] == "reddit"


def test_trend_fields():
    t: Trend = {
        "name": "Inference Optimization",
        "description": "Tooling to reduce inference latency",
        "signal_ids": [0, 1, 2],
    }
    assert t["signal_ids"] == [0, 1, 2]


def test_thesis_fields():
    th: Thesis = {
        "title": "The Inference Cost Squeeze",
        "point_of_view": "Inference is the new compute bottleneck.",
        "why_now": "GPU costs plateaued while model sizes exploded.",
        "signals": [],
        "category": "inference infrastructure",
    }
    assert th["category"] == "inference infrastructure"


def test_raw_signals_reducer():
    # raw_signals must use operator.add so parallel nodes can append without overwrite
    hints = get_type_hints(NewsletterState, include_extras=True)
    raw = hints["raw_signals"]
    args = get_args(raw)
    assert args[1] is operator.add, "raw_signals must use operator.add reducer"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_state.py -v
```

Expected: `ImportError: cannot import name 'Signal' from 'brainsync.state'`

- [ ] **Step 3: Write state.py**

```python
import operator
from typing import Annotated, TypedDict


class Signal(TypedDict):
    title: str
    summary: str
    url: str
    source: str  # "reddit" | "hn" | "techcrunch" | "github"
    date: str
    relevance_score: float  # 0.0-1.0


class Trend(TypedDict):
    name: str
    description: str
    signal_ids: list[int]  # indices into NewsletterState.raw_signals


class Thesis(TypedDict):
    title: str
    point_of_view: str
    why_now: str
    signals: list[Signal]
    category: str


class NewsletterState(TypedDict):
    run_date: str
    raw_signals: Annotated[list[Signal], operator.add]
    synthesized_trends: list[Trend]
    investment_theses: list[Thesis]
    newsletter_html: str
    newsletter_text: str
    sent: bool
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_state.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add brainsync/state.py tests/test_state.py
git commit -m "feat: state types with operator.add reducer for parallel fan-out"
```

---

## Task 3: Prompt Templates

**Files:**
- Create: `brainsync/prompts.py`

- [ ] **Step 1: Write prompts.py**

No test needed — these are string constants with no logic.

```python
RELEVANCE_SCORER = """\
You are evaluating whether content is relevant to AI and data infrastructure investing.

Title: {title}
Summary: {summary}

Rate relevance to AI/data infrastructure on a scale of 0.0 to 1.0:
- 1.0: Directly about AI infrastructure, ML systems, data engineering, developer tools, inference
- 0.5: Tangentially related (general cloud computing, software tooling)
- 0.0: Unrelated (consumer apps, politics, hardware consumer products)

Respond with only valid JSON: {{"score": <float between 0.0 and 1.0>}}"""

SYNTHESIZER = """\
You are an AI infrastructure investment analyst. Below are signals collected from technical \
communities this week.

Signals (index: title | source):
{signals}

Identify 3-7 distinct trends. Rules:
- Each trend must have a specific, descriptive name (not "AI is growing")
- Each trend must be backed by signals from at least 2 different sources
- Trends should represent categories where a company could be built

Respond with only valid JSON:
{{
  "trends": [
    {{
      "name": "<trend name>",
      "description": "<2-3 sentences on what is happening and why it matters>",
      "signal_ids": [<list of signal indices from the list above>]
    }}
  ]
}}"""

THESIS_WRITER = """\
You are a seed-stage venture investor at a top-tier firm focused on AI and data infrastructure. \
You are opinionated, specific, and think in terms of market timing.

This week's trends:
{trends}

For each trend, write an investment thesis memo to your partners. Be direct. Do not hedge. Cover:
1. What specific company archetype could be built here?
2. What technical or market shift makes this fundable NOW vs 2 years ago?
3. What is the wedge product (the first thing a founding team ships)?

Respond with only valid JSON:
{{
  "theses": [
    {{
      "title": "<sharp, specific thesis title>",
      "category": "<one of: inference infrastructure | data engineering | developer tools | MLOps | AI agents | other>",
      "point_of_view": "<2-3 sentences, opinionated, no hedging>",
      "why_now": "<1-2 sentences on what has changed to create this opportunity today>",
      "signal_ids": [<signal indices from the trend's signal_ids that support this thesis>]
    }}
  ]
}}"""
```

- [ ] **Step 2: Commit**

```bash
git add brainsync/prompts.py
git commit -m "feat: Claude prompt templates for scoring, synthesis, and thesis writing"
```

---

## Task 4: Reddit Researcher

**Files:**
- Create: `tests/fixtures/reddit_posts.json`
- Create: `tests/test_reddit.py`
- Create: `brainsync/agents/researchers/reddit.py`

- [ ] **Step 1: Create fixture**

Create `tests/fixtures/reddit_posts.json`:

```json
[
  {
    "title": "We open-sourced our tensor parallelism inference engine — 8x throughput on LLaMA-3",
    "selftext": "After 8 months building at our startup we're releasing the core engine. It uses pipeline parallelism across 4 GPUs with async prefill scheduling. Benchmarks in the README.",
    "permalink": "/r/MachineLearning/comments/abc123/inference_engine/",
    "score": 892,
    "created_utc": 1743724800,
    "subreddit": "MachineLearning"
  },
  {
    "title": "Why I switched from Airflow to DuckDB + dbt for our ML feature pipeline",
    "selftext": "Airflow was overkill. For sub-1B row feature tables, DuckDB in-process is faster and way simpler to operate. Here's our architecture.",
    "permalink": "/r/dataengineering/comments/def456/duckdb_features/",
    "score": 445,
    "created_utc": 1743638400,
    "subreddit": "dataengineering"
  },
  {
    "title": "My cat knocked over my coffee",
    "selftext": "It was a Monday.",
    "permalink": "/r/funny/comments/xyz/cat/",
    "score": 50000,
    "created_utc": 1743638400,
    "subreddit": "funny"
  }
]
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_reddit.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/test_reddit.py -v
```

Expected: `ModuleNotFoundError: No module named 'brainsync.agents.researchers.reddit'`

- [ ] **Step 4: Write reddit.py**

```python
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
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
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
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_reddit.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add brainsync/agents/researchers/reddit.py tests/test_reddit.py tests/fixtures/reddit_posts.json
git commit -m "feat: Reddit researcher with relevance scoring"
```

---

## Task 5: HN Researcher

**Files:**
- Create: `tests/fixtures/hn_items.json`
- Create: `tests/test_hn.py`
- Create: `brainsync/agents/researchers/hn.py`

- [ ] **Step 1: Create fixture**

Create `tests/fixtures/hn_items.json`:

```json
{
  "hits": [
    {
      "title": "Show HN: Prefect 3.0 – open-source orchestration for AI pipelines",
      "url": "https://github.com/PrefectHQ/prefect",
      "points": 312,
      "created_at": "2026-04-01T09:00:00.000Z",
      "objectID": "100001"
    },
    {
      "title": "Ask HN: What does your ML feature store look like in 2026?",
      "url": null,
      "points": 198,
      "created_at": "2026-04-02T14:00:00.000Z",
      "objectID": "100002"
    },
    {
      "title": "I made a chocolate cake",
      "url": "https://myblog.com/cake",
      "points": 5,
      "created_at": "2026-04-03T08:00:00.000Z",
      "objectID": "100003"
    }
  ]
}
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_hn.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/test_hn.py -v
```

Expected: `ModuleNotFoundError: No module named 'brainsync.agents.researchers.hn'`

- [ ] **Step 4: Write hn.py**

```python
import json
import os

import anthropic
import httpx

from brainsync.prompts import RELEVANCE_SCORER
from brainsync.state import NewsletterState, Signal

HN_API = "https://hn.algolia.com/api/v1/search"
MIN_POINTS = 50
MIN_RELEVANCE = 0.4
TAGS = "(story,show_hn,ask_hn)"


def hn_researcher(state: NewsletterState) -> dict:
    client = anthropic.Anthropic()
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
    from datetime import datetime, timedelta, timezone

    return int((datetime.now(tz=timezone.utc) - timedelta(days=7)).timestamp())


def _score_relevance(client: anthropic.Anthropic, title: str, summary: str) -> float:
    prompt = RELEVANCE_SCORER.format(title=title, summary=summary)
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=50,
        messages=[{"role": "user", "content": prompt}],
    )
    result = json.loads(message.content[0].text)
    return float(result["score"])
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_hn.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add brainsync/agents/researchers/hn.py tests/test_hn.py tests/fixtures/hn_items.json
git commit -m "feat: HN researcher via Algolia API"
```

---

## Task 6: TechCrunch Researcher

**Files:**
- Create: `tests/fixtures/techcrunch_feed.xml`
- Create: `tests/test_techcrunch.py`
- Create: `brainsync/agents/researchers/techcrunch.py`

- [ ] **Step 1: Create fixture**

Create `tests/fixtures/techcrunch_feed.xml`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>TechCrunch</title>
    <item>
      <title>Anyscale raises $100M to scale Ray-based AI infrastructure</title>
      <link>https://techcrunch.com/2026/04/01/anyscale-raises/</link>
      <description>Anyscale, the company behind the Ray distributed computing framework, announced a $100M Series C to expand its managed infrastructure platform for AI workloads.</description>
      <pubDate>Tue, 01 Apr 2026 10:00:00 +0000</pubDate>
    </item>
    <item>
      <title>New startup promises to fix data observability with AI agents</title>
      <link>https://techcrunch.com/2026/04/02/data-observability-startup/</link>
      <description>DataWatch emerges from stealth with a $12M seed round to bring autonomous data quality monitoring to modern data stacks.</description>
      <pubDate>Wed, 02 Apr 2026 14:00:00 +0000</pubDate>
    </item>
    <item>
      <title>Celebrity opens new restaurant in New York</title>
      <link>https://techcrunch.com/2026/04/03/celebrity-restaurant/</link>
      <description>A famous actor opened a new dining establishment.</description>
      <pubDate>Thu, 03 Apr 2026 08:00:00 +0000</pubDate>
    </item>
  </channel>
</rss>
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_techcrunch.py`:

```python
from pathlib import Path
from unittest.mock import patch, MagicMock
import feedparser

FIXTURES = Path(__file__).parent / "fixtures"


@patch("brainsync.agents.researchers.techcrunch._score_relevance")
@patch("brainsync.agents.researchers.techcrunch.feedparser.parse")
def test_techcrunch_researcher_returns_signals(mock_parse, mock_score):
    xml = (FIXTURES / "techcrunch_feed.xml").read_text()
    mock_parse.return_value = feedparser.parse(xml)
    mock_score.side_effect = [0.9, 0.85, 0.05]

    from brainsync.agents.researchers.techcrunch import techcrunch_researcher

    result = techcrunch_researcher({"run_date": "2026-04-05", "raw_signals": []})

    assert "raw_signals" in result
    signals = result["raw_signals"]
    assert len(signals) == 2
    assert all(s["source"] == "techcrunch" for s in signals)
    assert all(s["relevance_score"] >= 0.4 for s in signals)


@patch("brainsync.agents.researchers.techcrunch._score_relevance")
@patch("brainsync.agents.researchers.techcrunch.feedparser.parse")
def test_techcrunch_researcher_includes_link(mock_parse, mock_score):
    xml = (FIXTURES / "techcrunch_feed.xml").read_text()
    mock_parse.return_value = feedparser.parse(xml)
    mock_score.return_value = 0.9

    from brainsync.agents.researchers.techcrunch import techcrunch_researcher

    result = techcrunch_researcher({"run_date": "2026-04-05", "raw_signals": []})
    assert all(s["url"].startswith("https://") for s in result["raw_signals"])
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/test_techcrunch.py -v
```

Expected: `ModuleNotFoundError: No module named 'brainsync.agents.researchers.techcrunch'`

- [ ] **Step 4: Write techcrunch.py**

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_techcrunch.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add brainsync/agents/researchers/techcrunch.py tests/test_techcrunch.py tests/fixtures/techcrunch_feed.xml
git commit -m "feat: TechCrunch researcher via RSS"
```

---

## Task 7: GitHub Researcher

**Files:**
- Create: `tests/fixtures/github_repos.json`
- Create: `tests/test_github.py`
- Create: `brainsync/agents/researchers/github.py`

- [ ] **Step 1: Create fixture**

Create `tests/fixtures/github_repos.json`:

```json
{
  "items": [
    {
      "full_name": "vllm-project/vllm",
      "description": "A high-throughput and memory-efficient inference and serving engine for LLMs",
      "html_url": "https://github.com/vllm-project/vllm",
      "stargazers_count": 28000,
      "language": "Python",
      "created_at": "2026-03-28T00:00:00Z",
      "topics": ["llm", "inference", "serving"]
    },
    {
      "full_name": "apache/iceberg",
      "description": "A high-performance format for huge analytic tables",
      "html_url": "https://github.com/apache/iceberg",
      "stargazers_count": 6200,
      "language": "Java",
      "created_at": "2026-03-30T00:00:00Z",
      "topics": ["data-lake", "table-format", "analytics"]
    },
    {
      "full_name": "someuser/my-wedding-website",
      "description": "A website for my wedding",
      "html_url": "https://github.com/someuser/my-wedding-website",
      "stargazers_count": 12,
      "language": "HTML",
      "created_at": "2026-04-01T00:00:00Z",
      "topics": []
    }
  ]
}
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_github.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/test_github.py -v
```

Expected: `ModuleNotFoundError: No module named 'brainsync.agents.researchers.github'`

- [ ] **Step 4: Write github.py**

```python
import json
import os
from datetime import datetime, timedelta, timezone

import anthropic
import httpx

from brainsync.prompts import RELEVANCE_SCORER
from brainsync.state import NewsletterState, Signal

GITHUB_SEARCH_URL = "https://api.github.com/search/repositories"
LOOKBACK_DAYS = 7
MIN_STARS = 50
MIN_RELEVANCE = 0.4
LANGUAGES = ["Python", "Go", "Rust"]


def github_researcher(state: NewsletterState) -> dict:
    client = anthropic.Anthropic()
    cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=LOOKBACK_DAYS)).strftime(
        "%Y-%m-%d"
    )
    headers = {"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"}
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


def _score_relevance(client: anthropic.Anthropic, title: str, summary: str) -> float:
    prompt = RELEVANCE_SCORER.format(title=title, summary=summary)
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=50,
        messages=[{"role": "user", "content": prompt}],
    )
    result = json.loads(message.content[0].text)
    return float(result["score"])
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_github.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add brainsync/agents/researchers/github.py tests/test_github.py tests/fixtures/github_repos.json
git commit -m "feat: GitHub researcher via Search API"
```

---

## Task 8: Synthesizer

**Files:**
- Create: `tests/test_synthesizer.py`
- Create: `brainsync/agents/synthesizer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_synthesizer.py`:

```python
import json
from unittest.mock import MagicMock, patch

from brainsync.state import Signal

SAMPLE_SIGNALS: list[Signal] = [
    {
        "title": "vllm achieves 10x throughput improvement",
        "summary": "New speculative decoding approach reduces latency",
        "url": "https://github.com/vllm-project/vllm",
        "source": "github",
        "date": "2026-04-01T00:00:00",
        "relevance_score": 0.95,
    },
    {
        "title": "Inference cost is the new capex",
        "summary": "Discussion on why inference optimization is the next big opportunity",
        "url": "https://news.ycombinator.com/item?id=12345",
        "source": "hn",
        "date": "2026-04-02T00:00:00",
        "relevance_score": 0.9,
    },
    {
        "title": "DuckDB-based feature stores gaining traction",
        "summary": "Several startups adopting DuckDB for in-process feature computation",
        "url": "https://reddit.com/r/dataengineering/post",
        "source": "reddit",
        "date": "2026-04-03T00:00:00",
        "relevance_score": 0.85,
    },
]

MOCK_CLAUDE_RESPONSE = json.dumps({
    "trends": [
        {
            "name": "Inference Efficiency Tooling",
            "description": "Multiple signals point to growing activity in LLM inference optimization.",
            "signal_ids": [0, 1],
        },
        {
            "name": "In-Process Data Compute",
            "description": "DuckDB and similar tools displacing heavyweight orchestration for ML data pipelines.",
            "signal_ids": [2],
        },
    ]
})


@patch("brainsync.agents.synthesizer.anthropic.Anthropic")
def test_synthesizer_returns_trends(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text=MOCK_CLAUDE_RESPONSE)]
    mock_client.messages.create.return_value = mock_message

    from brainsync.agents.synthesizer import synthesizer

    state = {
        "run_date": "2026-04-05",
        "raw_signals": SAMPLE_SIGNALS,
        "synthesized_trends": [],
        "investment_theses": [],
        "newsletter_html": "",
        "newsletter_text": "",
        "sent": False,
    }
    result = synthesizer(state)

    assert "synthesized_trends" in result
    trends = result["synthesized_trends"]
    assert len(trends) == 2
    assert trends[0]["name"] == "Inference Efficiency Tooling"
    assert 0 in trends[0]["signal_ids"]


@patch("brainsync.agents.synthesizer.anthropic.Anthropic")
def test_synthesizer_formats_signals_for_prompt(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text=MOCK_CLAUDE_RESPONSE)]
    mock_client.messages.create.return_value = mock_message

    from brainsync.agents.synthesizer import synthesizer

    state = {
        "run_date": "2026-04-05",
        "raw_signals": SAMPLE_SIGNALS,
        "synthesized_trends": [],
        "investment_theses": [],
        "newsletter_html": "",
        "newsletter_text": "",
        "sent": False,
    }
    synthesizer(state)

    call_args = mock_client.messages.create.call_args
    prompt_content = call_args.kwargs["messages"][0]["content"]
    assert "vllm achieves" in prompt_content
    assert "DuckDB" in prompt_content
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_synthesizer.py -v
```

Expected: `ModuleNotFoundError: No module named 'brainsync.agents.synthesizer'`

- [ ] **Step 3: Write synthesizer.py**

```python
import json

import anthropic

from brainsync.prompts import SYNTHESIZER
from brainsync.state import NewsletterState, Trend


def synthesizer(state: NewsletterState) -> dict:
    client = anthropic.Anthropic()
    signals = state["raw_signals"]

    signal_list = "\n".join(
        f"{i}: {s['title']} | {s['source']}" for i, s in enumerate(signals)
    )
    prompt = SYNTHESIZER.format(signals=signal_list)

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    result = json.loads(message.content[0].text)
    trends: list[Trend] = result["trends"]
    return {"synthesized_trends": trends}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_synthesizer.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add brainsync/agents/synthesizer.py tests/test_synthesizer.py
git commit -m "feat: synthesizer node — clusters signals into trends via Claude"
```

---

## Task 9: Thesis Writer

**Files:**
- Create: `tests/test_thesis_writer.py`
- Create: `brainsync/agents/thesis_writer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_thesis_writer.py`:

```python
import json
from unittest.mock import MagicMock, patch

from brainsync.state import Signal, Trend

SAMPLE_SIGNALS: list[Signal] = [
    {
        "title": "vllm achieves 10x throughput improvement",
        "summary": "New speculative decoding approach",
        "url": "https://github.com/vllm-project/vllm",
        "source": "github",
        "date": "2026-04-01T00:00:00",
        "relevance_score": 0.95,
    },
    {
        "title": "Inference cost is the new capex",
        "summary": "Discussion on inference optimization",
        "url": "https://news.ycombinator.com/item?id=12345",
        "source": "hn",
        "date": "2026-04-02T00:00:00",
        "relevance_score": 0.9,
    },
]

SAMPLE_TRENDS: list[Trend] = [
    {
        "name": "Inference Efficiency Tooling",
        "description": "Multiple signals point to growing activity in LLM inference optimization.",
        "signal_ids": [0, 1],
    }
]

MOCK_CLAUDE_RESPONSE = json.dumps({
    "theses": [
        {
            "title": "The Inference Cost Squeeze Creates a $1B Tooling Market",
            "category": "inference infrastructure",
            "point_of_view": "Every AI product team is bleeding on inference costs. The next infrastructure winner abstracts away the complexity of speculative decoding, batching, and KV cache management behind a simple API.",
            "why_now": "Model sizes crossed a threshold where naive serving approaches are economically unviable. Speculative decoding only became practical in 2025 — the window for tooling is now.",
            "signal_ids": [0, 1],
        }
    ]
})


@patch("brainsync.agents.thesis_writer.anthropic.Anthropic")
def test_thesis_writer_returns_theses(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text=MOCK_CLAUDE_RESPONSE)]
    mock_client.messages.create.return_value = mock_message

    from brainsync.agents.thesis_writer import thesis_writer

    state = {
        "run_date": "2026-04-05",
        "raw_signals": SAMPLE_SIGNALS,
        "synthesized_trends": SAMPLE_TRENDS,
        "investment_theses": [],
        "newsletter_html": "",
        "newsletter_text": "",
        "sent": False,
    }
    result = thesis_writer(state)

    assert "investment_theses" in result
    theses = result["investment_theses"]
    assert len(theses) == 1
    assert theses[0]["title"] == "The Inference Cost Squeeze Creates a $1B Tooling Market"
    assert theses[0]["category"] == "inference infrastructure"


@patch("brainsync.agents.thesis_writer.anthropic.Anthropic")
def test_thesis_writer_resolves_signal_ids_to_signals(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text=MOCK_CLAUDE_RESPONSE)]
    mock_client.messages.create.return_value = mock_message

    from brainsync.agents.thesis_writer import thesis_writer

    state = {
        "run_date": "2026-04-05",
        "raw_signals": SAMPLE_SIGNALS,
        "synthesized_trends": SAMPLE_TRENDS,
        "investment_theses": [],
        "newsletter_html": "",
        "newsletter_text": "",
        "sent": False,
    }
    result = thesis_writer(state)

    thesis = result["investment_theses"][0]
    # signal_ids [0, 1] should be resolved to actual Signal objects
    assert len(thesis["signals"]) == 2
    assert thesis["signals"][0]["title"] == "vllm achieves 10x throughput improvement"
    assert thesis["signals"][1]["source"] == "hn"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_thesis_writer.py -v
```

Expected: `ModuleNotFoundError: No module named 'brainsync.agents.thesis_writer'`

- [ ] **Step 3: Write thesis_writer.py**

```python
import json

import anthropic

from brainsync.prompts import THESIS_WRITER
from brainsync.state import NewsletterState, Thesis


def thesis_writer(state: NewsletterState) -> dict:
    client = anthropic.Anthropic()
    trends = state["synthesized_trends"]
    raw_signals = state["raw_signals"]

    trend_text = "\n\n".join(
        f"Trend: {t['name']}\n{t['description']}\nSignal IDs: {t['signal_ids']}"
        for t in trends
    )
    prompt = THESIS_WRITER.format(trends=trend_text)

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = json.loads(message.content[0].text)
    theses: list[Thesis] = []

    for t in raw["theses"]:
        signal_ids = t.get("signal_ids", [])
        resolved_signals = [
            raw_signals[i] for i in signal_ids if i < len(raw_signals)
        ]
        theses.append(
            Thesis(
                title=t["title"],
                category=t["category"],
                point_of_view=t["point_of_view"],
                why_now=t["why_now"],
                signals=resolved_signals,
            )
        )

    return {"investment_theses": theses}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_thesis_writer.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add brainsync/agents/thesis_writer.py tests/test_thesis_writer.py
git commit -m "feat: thesis writer node — forms investment POVs with why-now via Claude"
```

---

## Task 10: Newsletter Formatter

**Files:**
- Create: `brainsync/templates/newsletter.html.j2`
- Create: `tests/test_formatter.py`
- Create: `brainsync/agents/formatter.py`

- [ ] **Step 1: Write the Jinja2 template**

Create `brainsync/templates/newsletter.html.j2`:

```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body { font-family: Georgia, serif; max-width: 680px; margin: 40px auto; color: #1a1a1a; line-height: 1.6; }
    h1 { font-size: 1.4em; border-bottom: 2px solid #1a1a1a; padding-bottom: 8px; }
    h2 { font-size: 1.1em; margin-top: 2em; color: #333; }
    .thesis { margin-bottom: 2.5em; }
    .thesis-title { font-weight: bold; font-size: 1.05em; }
    .why-now { font-style: italic; color: #555; margin: 0.5em 0; }
    .signals { font-size: 0.9em; color: #444; }
    .signals a { color: #0066cc; }
    .category { font-size: 0.8em; text-transform: uppercase; letter-spacing: 0.05em; color: #888; }
    .appendix { border-top: 1px solid #ddd; margin-top: 3em; padding-top: 1em; font-size: 0.85em; }
    .appendix a { color: #444; }
  </style>
</head>
<body>
  <h1>BrainSync — AI & Data Infrastructure Signal</h1>
  <p style="color:#888; font-size:0.9em;">Week of {{ run_date }}</p>

  <h2>This Week's Theses</h2>

  {% for thesis in theses %}
  <div class="thesis">
    <div class="category">{{ thesis.category }}</div>
    <div class="thesis-title">{{ thesis.title }}</div>
    <p>{{ thesis.point_of_view }}</p>
    <div class="why-now">Why now: {{ thesis.why_now }}</div>
    <div class="signals">
      Supporting signals:
      {% for signal in thesis.signals %}
        <a href="{{ signal.url }}">{{ signal.title }}</a> ({{ signal.source }}){% if not loop.last %}, {% endif %}
      {% endfor %}
    </div>
  </div>
  {% endfor %}

  <div class="appendix">
    <strong>Raw Signals Appendix</strong><br>
    {% for signal in all_signals %}
      <a href="{{ signal.url }}">{{ signal.title }}</a> — {{ signal.source }}<br>
    {% endfor %}
  </div>
</body>
</html>
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_formatter.py`:

```python
from brainsync.state import Signal, Thesis

SAMPLE_SIGNALS: list[Signal] = [
    {
        "title": "vllm achieves 10x throughput",
        "summary": "Speculative decoding",
        "url": "https://github.com/vllm-project/vllm",
        "source": "github",
        "date": "2026-04-01T00:00:00",
        "relevance_score": 0.95,
    }
]

SAMPLE_THESES: list[Thesis] = [
    {
        "title": "The Inference Cost Squeeze",
        "category": "inference infrastructure",
        "point_of_view": "Every AI team is bleeding on inference costs.",
        "why_now": "Speculative decoding only became practical in 2025.",
        "signals": SAMPLE_SIGNALS,
    }
]


def test_formatter_returns_html_and_text():
    from brainsync.agents.formatter import formatter

    state = {
        "run_date": "2026-04-05",
        "raw_signals": SAMPLE_SIGNALS,
        "synthesized_trends": [],
        "investment_theses": SAMPLE_THESES,
        "newsletter_html": "",
        "newsletter_text": "",
        "sent": False,
    }
    result = formatter(state)

    assert "newsletter_html" in result
    assert "newsletter_text" in result
    assert len(result["newsletter_html"]) > 100
    assert len(result["newsletter_text"]) > 50


def test_formatter_html_contains_thesis_title():
    from brainsync.agents.formatter import formatter

    state = {
        "run_date": "2026-04-05",
        "raw_signals": SAMPLE_SIGNALS,
        "synthesized_trends": [],
        "investment_theses": SAMPLE_THESES,
        "newsletter_html": "",
        "newsletter_text": "",
        "sent": False,
    }
    result = formatter(state)

    assert "The Inference Cost Squeeze" in result["newsletter_html"]
    assert "2026-04-05" in result["newsletter_html"]
    assert "vllm" in result["newsletter_html"]


def test_formatter_text_contains_why_now():
    from brainsync.agents.formatter import formatter

    state = {
        "run_date": "2026-04-05",
        "raw_signals": SAMPLE_SIGNALS,
        "synthesized_trends": [],
        "investment_theses": SAMPLE_THESES,
        "newsletter_html": "",
        "newsletter_text": "",
        "sent": False,
    }
    result = formatter(state)

    assert "Speculative decoding only became practical" in result["newsletter_text"]
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/test_formatter.py -v
```

Expected: `ModuleNotFoundError: No module named 'brainsync.agents.formatter'`

- [ ] **Step 4: Write formatter.py**

```python
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from brainsync.state import NewsletterState

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


def formatter(state: NewsletterState) -> dict:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
    template = env.get_template("newsletter.html.j2")

    html = template.render(
        run_date=state["run_date"],
        theses=state["investment_theses"],
        all_signals=state["raw_signals"],
    )
    text = _html_to_text(state)
    return {"newsletter_html": html, "newsletter_text": text}


def _html_to_text(state: NewsletterState) -> str:
    lines = [
        f"BRAINSYNC — AI & Data Infrastructure Signal",
        f"Week of {state['run_date']}",
        "",
        "=== THIS WEEK'S THESES ===",
        "",
    ]
    for thesis in state["investment_theses"]:
        lines.append(f"[{thesis['category'].upper()}]")
        lines.append(thesis["title"])
        lines.append(thesis["point_of_view"])
        lines.append(f"Why now: {thesis['why_now']}")
        signal_refs = ", ".join(
            f"{s['title']} ({s['url']})" for s in thesis["signals"]
        )
        lines.append(f"Supporting signals: {signal_refs}")
        lines.append("")

    lines.append("=== RAW SIGNALS ===")
    for signal in state["raw_signals"]:
        lines.append(f"- {signal['title']} [{signal['source']}] {signal['url']}")

    return "\n".join(lines)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_formatter.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add brainsync/agents/formatter.py brainsync/templates/ tests/test_formatter.py
git commit -m "feat: newsletter formatter with Jinja2 HTML template and plain text"
```

---

## Task 11: Email Delivery

**Files:**
- Create: `tests/test_delivery.py`
- Create: `brainsync/agents/delivery.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_delivery.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_delivery.py -v
```

Expected: `ModuleNotFoundError: No module named 'brainsync.agents.delivery'`

- [ ] **Step 3: Write delivery.py**

```python
import os

import httpx

from brainsync.state import NewsletterState

RESEND_URL = "https://api.resend.com/emails"
FROM_ADDRESS = "BrainSync <newsletter@brainsync.dev>"


def delivery(state: NewsletterState) -> dict:
    api_key = os.environ["RESEND_API_KEY"]
    recipient = os.environ["NEWSLETTER_RECIPIENT_EMAIL"]

    response = httpx.post(
        RESEND_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "from": FROM_ADDRESS,
            "to": [recipient],
            "subject": f"BrainSync — AI & Data Infrastructure Signal ({state['run_date']})",
            "html": state["newsletter_html"],
            "text": state["newsletter_text"],
        },
    )
    response.raise_for_status()
    return {"sent": True}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_delivery.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add brainsync/agents/delivery.py tests/test_delivery.py
git commit -m "feat: email delivery via Resend API"
```

---

## Task 12: LangGraph StateGraph

**Files:**
- Create: `tests/test_graph.py`
- Create: `brainsync/graph.py`

- [ ] **Step 1: Write the failing integration test**

Create `tests/test_graph.py`:

```python
import json
from unittest.mock import MagicMock, patch

from brainsync.state import Signal


MOCK_SIGNALS: list[Signal] = [
    {
        "title": "vllm 10x throughput",
        "summary": "Speculative decoding improvement",
        "url": "https://github.com/vllm-project/vllm",
        "source": "github",
        "date": "2026-04-01T00:00:00",
        "relevance_score": 0.95,
    }
]

MOCK_TRENDS_RESPONSE = json.dumps({
    "trends": [{"name": "Inference Efficiency", "description": "LLM serving tooling.", "signal_ids": [0]}]
})

MOCK_THESES_RESPONSE = json.dumps({
    "theses": [
        {
            "title": "The Inference Cost Squeeze",
            "category": "inference infrastructure",
            "point_of_view": "Every AI team is bleeding on inference costs.",
            "why_now": "Speculative decoding only became practical in 2025.",
            "signal_ids": [0],
        }
    ]
})


@patch("brainsync.agents.delivery.httpx.post")
@patch("brainsync.agents.researchers.github.httpx.get")
@patch("brainsync.agents.researchers.hn.httpx.get")
@patch("brainsync.agents.researchers.techcrunch.feedparser.parse")
@patch("brainsync.agents.researchers.reddit.praw.Reddit")
@patch("brainsync.agents.researchers.reddit._score_relevance")
@patch("brainsync.agents.researchers.hn._score_relevance")
@patch("brainsync.agents.researchers.techcrunch._score_relevance")
@patch("brainsync.agents.researchers.github._score_relevance")
@patch("brainsync.agents.synthesizer.anthropic.Anthropic")
@patch("brainsync.agents.thesis_writer.anthropic.Anthropic")
def test_full_graph_runs_end_to_end(
    mock_thesis_anthropic,
    mock_synth_anthropic,
    mock_gh_score,
    mock_tc_score,
    mock_hn_score,
    mock_reddit_score,
    mock_reddit_cls,
    mock_tc_parse,
    mock_hn_get,
    mock_gh_get,
    mock_resend_post,
):
    # Researchers return empty signals
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

    mock_hn_resp = MagicMock()
    mock_hn_resp.json.return_value = {"hits": []}
    mock_hn_resp.raise_for_status.return_value = None
    mock_hn_get.return_value = mock_hn_resp

    mock_gh_resp = MagicMock()
    mock_gh_resp.json.return_value = {"items": []}
    mock_gh_resp.raise_for_status.return_value = None
    mock_gh_get.return_value = mock_gh_resp

    # Synthesizer and thesis writer return valid responses
    mock_synth_client = MagicMock()
    mock_synth_anthropic.return_value = mock_synth_client
    mock_synth_msg = MagicMock()
    mock_synth_msg.content = [MagicMock(text=MOCK_TRENDS_RESPONSE)]
    mock_synth_client.messages.create.return_value = mock_synth_msg

    mock_thesis_client = MagicMock()
    mock_thesis_anthropic.return_value = mock_thesis_client
    mock_thesis_msg = MagicMock()
    mock_thesis_msg.content = [MagicMock(text=MOCK_THESES_RESPONSE)]
    mock_thesis_client.messages.create.return_value = mock_thesis_msg

    mock_resend_resp = MagicMock()
    mock_resend_resp.raise_for_status.return_value = None
    mock_resend_resp.json.return_value = {"id": "email_123"}
    mock_resend_post.return_value = mock_resend_resp

    from brainsync.graph import build_graph

    graph = build_graph()
    final_state = graph.invoke({"run_date": "2026-04-05", "raw_signals": []})

    assert final_state["sent"] is True
    assert len(final_state["investment_theses"]) == 1
    assert final_state["newsletter_html"] != ""


def test_dry_run_excludes_delivery_node(monkeypatch):
    # build_graph(include_delivery=False) should not include the delivery node
    from brainsync.graph import build_graph

    graph_with = build_graph(include_delivery=True)
    graph_without = build_graph(include_delivery=False)

    nodes_with = set(graph_with.nodes)
    nodes_without = set(graph_without.nodes)

    assert "delivery" in nodes_with
    assert "delivery" not in nodes_without
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_graph.py -v
```

Expected: `ModuleNotFoundError: No module named 'brainsync.graph'`

- [ ] **Step 3: Write graph.py**

```python
import argparse
from datetime import date

from langgraph.graph import END, START, StateGraph

from brainsync.agents.delivery import delivery
from brainsync.agents.formatter import formatter
from brainsync.agents.researchers.github import github_researcher
from brainsync.agents.researchers.hn import hn_researcher
from brainsync.agents.researchers.reddit import reddit_researcher
from brainsync.agents.researchers.techcrunch import techcrunch_researcher
from brainsync.agents.synthesizer import synthesizer
from brainsync.agents.thesis_writer import thesis_writer
from brainsync.state import NewsletterState


def build_graph(include_delivery: bool = True):
    graph = StateGraph(NewsletterState)

    graph.add_node("reddit_researcher", reddit_researcher)
    graph.add_node("hn_researcher", hn_researcher)
    graph.add_node("techcrunch_researcher", techcrunch_researcher)
    graph.add_node("github_researcher", github_researcher)
    graph.add_node("synthesizer", synthesizer)
    graph.add_node("thesis_writer", thesis_writer)
    graph.add_node("formatter", formatter)

    # Parallel fan-out
    graph.add_edge(START, "reddit_researcher")
    graph.add_edge(START, "hn_researcher")
    graph.add_edge(START, "techcrunch_researcher")
    graph.add_edge(START, "github_researcher")

    # Fan-in — all researchers must complete before synthesizer starts
    graph.add_edge("reddit_researcher", "synthesizer")
    graph.add_edge("hn_researcher", "synthesizer")
    graph.add_edge("techcrunch_researcher", "synthesizer")
    graph.add_edge("github_researcher", "synthesizer")

    graph.add_edge("synthesizer", "thesis_writer")
    graph.add_edge("thesis_writer", "formatter")

    if include_delivery:
        graph.add_node("delivery", delivery)
        graph.add_edge("formatter", "delivery")
        graph.add_edge("delivery", END)
    else:
        graph.add_edge("formatter", END)

    return graph.compile()


def run(dry_run: bool = False) -> None:
    graph = build_graph(include_delivery=not dry_run)
    initial_state: NewsletterState = {
        "run_date": date.today().isoformat(),
        "raw_signals": [],
        "synthesized_trends": [],
        "investment_theses": [],
        "newsletter_html": "",
        "newsletter_text": "",
        "sent": False,
    }
    final = graph.invoke(initial_state)
    if dry_run:
        print(final["newsletter_text"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_graph.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Run the full test suite**

```bash
pytest -v
```

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add brainsync/graph.py tests/test_graph.py
git commit -m "feat: LangGraph StateGraph with parallel researcher fan-out and dry-run CLI"
```

---

## Task 13: GitHub Actions Workflow

**Files:**
- Create: `.github/workflows/newsletter.yml`

- [ ] **Step 1: Write the workflow**

```bash
mkdir -p .github/workflows
```

Create `.github/workflows/newsletter.yml`:

```yaml
name: Weekly Newsletter

on:
  schedule:
    - cron: "0 8 * * 1"  # Monday 8am UTC
  workflow_dispatch:       # Manual trigger for testing

jobs:
  send-newsletter:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: "pip"

      - name: Install dependencies
        run: pip install -e .

      - name: Run BrainSync
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          REDDIT_CLIENT_ID: ${{ secrets.REDDIT_CLIENT_ID }}
          REDDIT_CLIENT_SECRET: ${{ secrets.REDDIT_CLIENT_SECRET }}
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          RESEND_API_KEY: ${{ secrets.RESEND_API_KEY }}
          NEWSLETTER_RECIPIENT_EMAIL: ${{ secrets.NEWSLETTER_RECIPIENT_EMAIL }}
        run: python -m brainsync.graph
```

- [ ] **Step 2: Commit**

```bash
git add .github/
git commit -m "chore: GitHub Actions cron workflow — Monday 8am UTC"
```

---

## Task 14: README and Sample Newsletter

**Files:**
- Create: `README.md`
- Create: `examples/sample-newsletter.html`

- [ ] **Step 1: Write README.md**

```markdown
# BrainSync

A weekly AI agent that monitors technical communities for AI and data infrastructure signals, synthesizes them into investment theses, and delivers a newsletter to your inbox every Monday morning.

Built to think like a seed-stage investor: not just summarizing what happened, but forming a point of view on what category is forming, who would build it, and why now.

## Why I Built This

Seed investing in AI infrastructure is a signal-processing problem. The most interesting opportunities don't show up in TechCrunch — they surface first in GitHub commits, HN Show HN threads, and Reddit posts from engineers who just got frustrated enough to build something. BrainSync automates the signal collection and applies a VC mental model on top.

## How It Works

```
┌──────────────────────────────────────┐
│           Researcher Nodes           │
│  Reddit  │  HN/YC  │  TechCrunch  │ GitHub │
└──────────────────┬───────────────────┘
                   │ parallel fan-out (LangGraph)
                   ▼
          [Synthesizer]
          Clusters signals into named trends
                   │
                   ▼
          [Thesis Writer]
          Forms investment POVs with why-now
                   │
                   ▼
          [Formatter → Email]
```

Four researcher agents run in parallel every Monday morning, each scoring content for relevance to AI/data infrastructure using Claude. The synthesizer clusters signals into trends (requiring cross-source corroboration to reduce noise). The thesis writer prompts Claude to reason as a seed-stage investor — forming specific, opinionated theses with a "why now" grounded in current technical shifts.

## Stack

- **Orchestration:** LangGraph (parallel fan-out StateGraph)
- **LLM:** Claude claude-opus-4-6 (Anthropic SDK) — for scoring, synthesis, and thesis writing
- **Sources:** Reddit (PRAW), HN Algolia API, TechCrunch RSS, GitHub Search API
- **Email:** Resend
- **Scheduling:** GitHub Actions cron

## Sample Output

See [`examples/sample-newsletter.html`](examples/sample-newsletter.html).

## Setup

**1. Clone and install:**
```bash
git clone https://github.com/yourusername/brainsync
cd brainsync
pip install -e .
```

**2. Set environment variables:**
```bash
cp .env.example .env
# Fill in your API keys
```

Required:
- `ANTHROPIC_API_KEY` — [console.anthropic.com](https://console.anthropic.com)
- `REDDIT_CLIENT_ID` + `REDDIT_CLIENT_SECRET` — [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps), create a "script" app
- `GITHUB_TOKEN` — Personal access token with `public_repo` scope
- `RESEND_API_KEY` — [resend.com](https://resend.com)
- `NEWSLETTER_RECIPIENT_EMAIL` — Where to send the newsletter

**3. Test locally (no email sent):**
```bash
python -m brainsync.graph --dry-run
```

**4. Deploy:**
Add secrets to your GitHub repository under Settings → Secrets. The workflow runs automatically every Monday at 8am UTC, or trigger manually via Actions → Weekly Newsletter → Run workflow.

## Tests

```bash
pytest -v
```

All tests use fixtures and mocked API calls — no live requests, no API keys needed.
```

- [ ] **Step 2: Generate a sample newsletter**

Run the formatter with hardcoded sample data to produce `examples/sample-newsletter.html`:

```python
# Run this once to generate the sample
import sys
sys.path.insert(0, ".")
from brainsync.agents.formatter import formatter
from brainsync.state import Signal, Thesis

signals = [
    Signal(title="vllm-project/vllm: 10x throughput via speculative decoding", summary="New async prefill scheduling achieves 10x throughput on LLaMA-3-70B with 4x GPU parallelism.", url="https://github.com/vllm-project/vllm", source="github", date="2026-04-01T00:00:00", relevance_score=0.95),
    Signal(title="Show HN: Prefect 3.0 — AI-native pipeline orchestration", summary="Prefect ships a major version with native support for LLM pipelines, async task execution, and built-in retry semantics for flaky model calls.", url="https://news.ycombinator.com/item?id=100001", source="hn", date="2026-04-01T09:00:00", relevance_score=0.9),
    Signal(title="Anyscale raises $100M to scale Ray-based AI infrastructure", summary="Anyscale expands its managed Ray platform, targeting AI teams running distributed training and inference at scale.", url="https://techcrunch.com/2026/04/01/anyscale-raises/", source="techcrunch", date="2026-04-01T10:00:00", relevance_score=0.88),
    Signal(title="Why I switched from Airflow to DuckDB + dbt for our ML feature pipeline", summary="For sub-1B row feature tables, DuckDB in-process is 10x faster to operate than Airflow with none of the operational overhead.", url="https://reddit.com/r/dataengineering/post/duckdb", source="reddit", date="2026-04-02T14:00:00", relevance_score=0.85),
]

theses = [
    Thesis(title="The Inference Cost Squeeze Creates a $1B Tooling Market", category="inference infrastructure", point_of_view="Every AI product team is bleeding on inference costs, and the gap between what naive serving delivers and what's achievable with speculative decoding and async prefill is now 5-10x. The next infrastructure winner abstracts this complexity behind a simple API. This is the Datadog moment for LLM serving.", why_now="Speculative decoding only became production-stable in late 2025. Model sizes crossed a threshold where naive serving is economically unviable for any product at scale. The window for tooling is open now, before the hyperscalers absorb this layer.", signals=[signals[0], signals[2]]),
    Thesis(title="AI-Native Orchestration Displaces Airflow for the ML Stack", category="MLOps", point_of_view="Airflow was built for ETL, not for workflows where tasks are probabilistic, latencies are variable, and retries require semantic awareness. The ML stack needs an orchestrator that treats LLM calls as first-class primitives — with built-in handling for rate limits, token budgets, and partial outputs.", why_now="The volume of LLM-in-production deployments hit a critical mass in 2025 that made the impedance mismatch with existing orchestration intolerable. Prefect 3.0 and similar launches signal that the market is ready to pay for a purpose-built replacement.", signals=[signals[1], signals[3]]),
]

state = {"run_date": "2026-04-07", "raw_signals": signals, "synthesized_trends": [], "investment_theses": theses, "newsletter_html": "", "newsletter_text": "", "sent": False}
result = formatter(state)

import pathlib
pathlib.Path("examples").mkdir(exist_ok=True)
pathlib.Path("examples/sample-newsletter.html").write_text(result["newsletter_html"])
print("Written to examples/sample-newsletter.html")
```

Save this as `scripts/generate_sample.py` and run:

```bash
mkdir -p scripts
# save the above as scripts/generate_sample.py
python scripts/generate_sample.py
```

Expected: `Written to examples/sample-newsletter.html`

- [ ] **Step 3: Commit**

```bash
git add README.md examples/ scripts/
git commit -m "docs: README with architecture overview and sample newsletter output"
```

---

## Task 15: Final Verification

- [ ] **Step 1: Run the full test suite**

```bash
pytest -v
```

Expected: All tests pass, 0 failures.

- [ ] **Step 2: Verify dry-run works end-to-end**

With real API keys set in `.env`:

```bash
export $(cat .env | xargs)
python -m brainsync.graph --dry-run
```

Expected: Newsletter text printed to stdout with actual theses.

- [ ] **Step 3: Verify project structure**

```bash
find . -name "*.py" | sort
```

Expected output includes:
```
./brainsync/__init__.py
./brainsync/agents/__init__.py
./brainsync/agents/delivery.py
./brainsync/agents/formatter.py
./brainsync/agents/researchers/__init__.py
./brainsync/agents/researchers/github.py
./brainsync/agents/researchers/hn.py
./brainsync/agents/researchers/reddit.py
./brainsync/agents/researchers/techcrunch.py
./brainsync/agents/synthesizer.py
./brainsync/agents/thesis_writer.py
./brainsync/graph.py
./brainsync/prompts.py
./brainsync/state.py
./scripts/generate_sample.py
./tests/test_delivery.py
./tests/test_formatter.py
./tests/test_github.py
./tests/test_graph.py
./tests/test_hn.py
./tests/test_reddit.py
./tests/test_state.py
./tests/test_synthesizer.py
./tests/test_techcrunch.py
./tests/test_thesis_writer.py
```

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: final verification pass"
```
