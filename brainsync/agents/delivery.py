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
