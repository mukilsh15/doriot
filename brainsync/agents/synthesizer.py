import json
import os

from openai import OpenAI

from brainsync.prompts import SYNTHESIZER
from brainsync.state import NewsletterState, Trend


def synthesizer(state: NewsletterState) -> dict:
    client = OpenAI(api_key=os.environ.get("GROQ_API_KEY", ""), base_url="https://api.groq.com/openai/v1")
    signals = state["raw_signals"]

    signal_list = "\n".join(
        f"{i}: {s['title']} | {s['source']}" for i, s in enumerate(signals)
    )
    prompt = SYNTHESIZER.format(signals=signal_list)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    result = json.loads(response.choices[0].message.content)
    trends: list[Trend] = result["trends"]
    return {"synthesized_trends": trends}
