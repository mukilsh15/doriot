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
        "BRAINSYNC — AI & Data Infrastructure Signal",
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
