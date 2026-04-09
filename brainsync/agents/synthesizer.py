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
