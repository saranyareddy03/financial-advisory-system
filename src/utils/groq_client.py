"""
Minimal Groq chat client used as an LLM fallback.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

from src.config.settings import config as settings


class GroqAPIError(RuntimeError):
    """Raised when the Groq API request fails."""


class GroqChatClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model = model or settings.GROQ_SQL_MODEL

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.model)

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1200,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not self.enabled:
            raise GroqAPIError("Groq API key or model is not configured.")

        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/json",
                    "User-Agent": "financial-advisory-system/1.0",
                },
                timeout=45,
            )
        except Exception as exc:  # pragma: no cover - network/runtime
            raise GroqAPIError(str(exc)) from exc

        if not response.ok:
            body = response.text[:800]
            raise GroqAPIError(f"HTTP {response.status_code}: {body}")

        try:
            body = response.json()
        except Exception as exc:  # pragma: no cover - defensive
            raise GroqAPIError(f"Invalid JSON response: {response.text[:800]}") from exc

        try:
            return body["choices"][0]["message"]["content"]
        except Exception as exc:  # pragma: no cover - defensive
            raise GroqAPIError(f"Unexpected Groq response shape: {body}") from exc
