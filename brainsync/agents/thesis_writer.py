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
