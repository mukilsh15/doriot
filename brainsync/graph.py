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

    # Parallel fan-out from START
    graph.add_edge(START, "reddit_researcher")
    graph.add_edge(START, "hn_researcher")
    graph.add_edge(START, "techcrunch_researcher")
    graph.add_edge(START, "github_researcher")

    # Fan-in: all 4 researchers must complete before synthesizer runs
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
