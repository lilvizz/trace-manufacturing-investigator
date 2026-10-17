import os
import random
import time

from google import genai
from google.genai import errors, types

from app.config import load_environment

load_environment()

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class GeminiReasoningClient:
    """
    Thin adapter around Gemini for investigation planning/reasoning.

    max_retries controls TRACE's adapter-level retries.
    The Google GenAI HTTP client is also configured explicitly so
    transient API errors do not silently trigger its default 5 attempts.
    """

    def __init__(
        self,
        model: str = "gemini-3.6-flash",
        max_retries: int = 5,
        base_delay_seconds: float = 1.0,
        timeout_seconds: float | None = None,
    ):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not set."
            )

        # The SDK defaults to 5 HTTP attempts. Make this explicit so the
        # caller can choose fast-fail behavior for the UI.
        sdk_attempts = max(1, max_retries + 1)

        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                # Per-request timeout (milliseconds). None keeps the SDK
                # default; the UI passes a bound so a stalled network call
                # surfaces as an error instead of hanging the app.
                timeout=int(timeout_seconds * 1000) if timeout_seconds else None,
                retry_options=types.HttpRetryOptions(
                    attempts=sdk_attempts,
                    initial_delay=max(0.0, base_delay_seconds),
                    max_delay=max(0.0, base_delay_seconds),
                    jitter=0.0,
                )
            ),
        )
        self.model = model
        self.max_retries = max_retries
        self.base_delay_seconds = base_delay_seconds

    def generate(self, prompt: str) -> str:
        attempt = 0

        while True:
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1,
                    ),
                )

                if not response.text:
                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )

                return response.text

            except errors.APIError as exc:
                attempt += 1

                if (
                    exc.code not in RETRYABLE_STATUS_CODES
                    or attempt > self.max_retries
                ):
                    raise

                delay = (
                    self.base_delay_seconds
                    * (2 ** (attempt - 1))
                )

                delay += random.uniform(
                    0,
                    self.base_delay_seconds,
                )

                time.sleep(delay)
