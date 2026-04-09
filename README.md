# BrainSync

A weekly AI agent that monitors technical communities for AI and data infrastructure signals, synthesizes them into investment theses, and delivers a newsletter to your inbox every Monday morning.

Built to think like a seed-stage investor: not just summarizing what happened, but forming a point of view on what category is forming, who would build it, and why now.

## Why I Built This

Seed investing in AI infrastructure is a signal-processing problem. The most interesting opportunities don't show up in TechCrunch first — they surface in GitHub commits, HN Show HN threads, and Reddit posts from engineers who just got frustrated enough to build something. BrainSync automates the signal collection and applies a VC mental model on top.

The thesis: if you can track what engineers are actually building and talking about — not what founders are pitching — you get a 2-4 week head start on which categories are forming. That's the edge this tool tries to create.

## How It Works

```
┌──────────────────────────────────────────────────┐
│                  Researcher Nodes                │
│  Reddit (PRAW)  │  HN Algolia  │  TechCrunch RSS │  GitHub Search  │
└──────────────────────────┬───────────────────────┘
                           │ parallel fan-out → join (LangGraph)
                           ▼
                    [Synthesizer]
                    Clusters signals into named trends
                    (requires cross-source corroboration)
                           │
                           ▼
                    [Thesis Writer]
                    Forms investment POVs with why-now
                    (prompted to reason as a seed investor)
                           │
                           ▼
                    [Formatter → Email via Resend]
```

Four researcher agents run in parallel every Monday morning via LangGraph's fan-out pattern. Each agent scores content for relevance to AI/data infrastructure using Claude before returning signals. The synthesizer clusters signals into trends — requiring at least two independent sources to reduce noise. The thesis writer then prompts Claude to reason as a seed-stage investor, forming specific, opinionated theses grounded in a "why now."

## Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph (parallel StateGraph with `operator.add` reducer) |
| LLM | Claude claude-opus-4-6 (Anthropic SDK) |
| Sources | Reddit via PRAW, HN Algolia API, TechCrunch RSS, GitHub Search API |
| Email | Resend |
| Scheduling | GitHub Actions cron |

Claude was chosen deliberately — BVP is an Anthropic investor, and Claude's extended context and structured output reliability make it well-suited for the synthesis and thesis-writing steps.

## Sample Output

See [`examples/sample-newsletter.html`](examples/sample-newsletter.html) for a rendered example.

## Setup

**1. Clone and install:**
```bash
git clone https://github.com/yourusername/brainsync
cd brainsync
pip install -e .
```

**2. Configure credentials:**
```bash
cp .env.example .env
# Edit .env with your API keys
```

| Variable | Where to get it |
|----------|----------------|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) |
| `REDDIT_CLIENT_ID` + `REDDIT_CLIENT_SECRET` | [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) — create a "script" app |
| `GITHUB_TOKEN` | Personal access token, `public_repo` scope |
| `RESEND_API_KEY` | [resend.com](https://resend.com) |
| `NEWSLETTER_RECIPIENT_EMAIL` | Where to send the newsletter |

**3. Preview locally (no email sent):**
```bash
export $(cat .env | xargs)
python -m brainsync.graph --dry-run
```

**4. Deploy:**

Add the variables above as secrets in your GitHub repository (Settings → Secrets and variables → Actions). The workflow triggers automatically every Monday at 8am UTC, or manually via Actions → Weekly Newsletter → Run workflow.

## Tests

```bash
pytest -v
```

All tests use fixtures and mocked API calls — no live requests or API keys required.

## Architecture Notes

**Why LangGraph over a simple script?** The parallel fan-out is real: 4 researchers run concurrently, each scoring ~25 items via Claude. Sequential execution would be ~4x slower. LangGraph's `Annotated[list, operator.add]` reducer pattern handles the merge cleanly without any coordination code.

**Why cross-source corroboration in the synthesizer?** A single Reddit post about a tool isn't a signal. The same tool appearing on GitHub trending, HN, and Reddit in the same week is. The synthesizer prompt enforces this — trends must be backed by multiple sources before they become thesis candidates.

**Why Claude for relevance scoring instead of keyword filtering?** Keywords miss context. "Kubernetes" in a data engineering post is a different signal than "Kubernetes" in a DevOps post. Claude's relevance scores are more expensive but meaningfully better at identifying what's actually relevant to AI infrastructure investing.
