"""Shared Gemini chat client using Google's OpenAI-compatible endpoint."""

from functools import lru_cache

from config import GEMINI_API_KEY, GEMINI_BASE_URL, GEMINI_MODEL


@lru_cache(maxsize=1)
def _client():
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is unavailable")
    from openai import OpenAI

    return OpenAI(
        api_key=GEMINI_API_KEY,
        base_url=GEMINI_BASE_URL,
        timeout=30.0,
        max_retries=1,
    )


def chat(system: str, user: str, *, json_mode: bool = False, max_tokens: int = 1024) -> str:
    options = {"response_format": {"type": "json_object"}} if json_mode else {}
    response = _client().chat.completions.create(
        model=GEMINI_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        reasoning_effort="low",
        max_tokens=max_tokens,
        **options,
    )
    content = (response.choices[0].message.content or "").strip()
    if not content:
        raise RuntimeError("Gemini returned an empty answer")
    return content
