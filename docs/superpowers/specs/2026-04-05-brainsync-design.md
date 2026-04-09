# BrainSync — AI Infrastructure Investment Intelligence Newsletter

**Date:** 2026-04-05  
**Author:** Mukil  
**Purpose:** Demonstrate technical depth and investment thesis thinking for BVP Associate (AI & Data Infrastructure) application

---

## Overview

BrainSync is a fully automated, LangGraph-based multi-agent system that monitors AI and data infrastructure signals across the web, synthesizes them into investment theses, and delivers a weekly newsletter via email. It is designed to think like a seed-stage investor: not just summarizing what happened, but forming a point of view on what category is forming, who would build it, and why now.

---

## Architecture

A LangGraph `StateGraph` with 5 node types. Researcher nodes run in parallel (fan-out), then results flow sequentially through synthesis, thesis writing, formatting, and delivery.

```
[GitHub Actions Cron — Monday 8am]
        │
        ▼
┌──────────────────────────────────────┐
│           Researcher Nodes           │
│  Reddit  │  HN/YC  │  TechCrunch  │ GitHub Trending │
└──────────────────┬───────────────────┘
                   │ parallel fan-out → join
                   ▼
          [Synthesizer Node]
          Clusters signals into named trends
                   │
                   ▼
          [Thesis Writer Node]
          Forms 3-5 investment POVs
                   │
                   ▼
          [Newsletter Formatter Node]
          Renders HTML + plain text email
                   │
                   ▼
          [Email Delivery Node]
          Sends via Resend API
```

---

## Shared State

```python
class Signal(TypedDict):
    title: str
    summary: str
    url: str
    source: str        # "reddit" | "hn" | "techcrunch" | "github"
    date: str
    relevance_score: float  # 0.0-1.0, scored by researcher agent

class Trend(TypedDict):
    name: str
    description: str
    signal_ids: list[int]   # indices into raw_signals

class Thesis(TypedDict):
    title: str
    point_of_view: str      # 2-3 sentences, opinionated
    why_now: str            # what has changed to make this fundable today
    signals: list[Signal]   # supporting evidence with sources + links
    category: str           # e.g. "inference infrastructure", "data lineage"

class NewsletterState(TypedDict):
    run_date: str
    raw_signals: list[Signal]
    synthesized_trends: list[Trend]
    investment_theses: list[Thesis]
    newsletter_html: str
    newsletter_text: str
    sent: bool
```

---

## Agent Responsibilities

### Researcher Nodes (run in parallel)

Each researcher fetches its source, filters for relevance to AI/data infrastructure, and returns a list of `Signal` objects. Relevance scoring uses Claude to rate each item 0-1 against the domain.

| Node | Source | Method |
|------|--------|--------|
| `reddit_researcher` | r/MachineLearning, r/dataengineering, r/LocalLLaMA, r/mlops | PRAW (free) |
| `hn_researcher` | Hacker News front page + Show HN | HN Algolia API (free) |
| `techcrunch_researcher` | TechCrunch AI/enterprise RSS | feedparser (free) |
| `github_researcher` | GitHub Trending (Python, Go, Rust) | GitHub API (free) |

### Synthesizer Node

Takes all raw signals, deduplicates by semantic similarity, and clusters into named `Trend` objects. Each trend must be backed by signals from at least 2 different sources to reduce noise.

### Thesis Writer Node

Takes trends and prompts Claude to reason as a seed-stage investor:
- What category is forming?
- What company could be built here?
- What has changed technically or in the market to make this fundable now?
- What would the wedge product look like?

Produces 3-5 `Thesis` objects per run. Quality over quantity.

### Newsletter Formatter Node

Renders the final email with two sections:
1. **"This Week's Signal"** — 3-5 theses, each with POV, why-now, and linked evidence
2. **"Raw Signals Appendix"** — full list of sources for the reader's own exploration

Format: HTML email + plain text fallback.

### Email Delivery Node

Sends via Resend API. Simple REST call. Recipient configured via environment variable.

---

## Project Structure

```
brainsync/
├── brainsync/
│   ├── agents/
│   │   ├── researchers/
│   │   │   ├── reddit.py
│   │   │   ├── hn.py
│   │   │   ├── techcrunch.py
│   │   │   └── github.py
│   │   ├── synthesizer.py
│   │   ├── thesis_writer.py
│   │   ├── formatter.py
│   │   └── delivery.py
│   ├── graph.py          # LangGraph StateGraph definition
│   ├── state.py          # NewsletterState, Signal, Trend, Thesis types
│   └── prompts.py        # All Claude prompt templates
├── .github/
│   └── workflows/
│       └── newsletter.yml
├── examples/
│   └── sample-newsletter.html
├── tests/
│   ├── test_researchers.py
│   ├── test_synthesizer.py
│   └── test_thesis_writer.py
├── pyproject.toml
├── .env.example
└── README.md
```

---

## LLM

Claude (via Anthropic SDK) throughout. Chosen intentionally — BVP is an Anthropic investor, and Claude's extended thinking is well-suited for synthesis and thesis-writing steps.

---

## Scheduling

GitHub Actions cron workflow, Monday 8am UTC. Secrets:
- `ANTHROPIC_API_KEY`
- `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET`
- `GITHUB_TOKEN`
- `RESEND_API_KEY`
- `NEWSLETTER_RECIPIENT_EMAIL`

---

## Testing Strategy

- Unit tests for each researcher using recorded fixtures (no live API calls in CI)
- Integration test for the full graph using a mocked LLM
- Dry-run mode: `python -m brainsync.graph --dry-run` prints newsletter to stdout without sending

---

## README Goals

The README should read like a technical blog post:
- Sample newsletter output committed as `examples/sample-newsletter.html`
- Architecture diagram
- Brief "why I built this" connecting the tool to investment thesis work
- Clear setup instructions for self-hosting
