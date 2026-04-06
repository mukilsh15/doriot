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
