import os
import random
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class GeminiReasoningClient:
    """
    Thin adapter around Gemini for investigation reasoning.

    The investigation layer owns the prompt and response validation.
    This class only handles Gemini API communication.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        max_retries: int = 5,
        base_delay_seconds: float = 1.0,
    ):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not set."
            )

        self.client = genai.Client(api_key=api_key)
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