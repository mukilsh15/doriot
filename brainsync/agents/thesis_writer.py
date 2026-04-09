import json
import os

from openai import OpenAI

from brainsync.prompts import THESIS_WRITER
from brainsync.state import NewsletterState, Thesis


def thesis_writer(state: NewsletterState) -> dict:
    client = OpenAI(api_key=os.environ.get("GROQ_API_KEY", ""), base_url="https://api.groq.com/openai/v1")
    trends = state["synthesized_trends"]
    raw_signals = state["raw_signals"]

    trend_text = "\n\n".join(
        f"Trend: {t['name']}\n{t['description']}\nSignal IDs: {t['signal_ids']}"
        for t in trends
    )
    prompt = THESIS_WRITER.format(trends=trend_text)

    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = json.loads(response.choices[0].message.content)
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
