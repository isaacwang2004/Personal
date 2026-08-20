from __future__ import annotations

import os
from dataclasses import dataclass

import requests


class Translator:
    def translate(self, prompt: str) -> str:
        raise NotImplementedError


@dataclass
class HTTPChatTranslator(Translator):
    """Minimal client for an OpenAI-compatible chat-completions endpoint."""

    api_key: str
    base_url: str
    model: str
    timeout: int = 60

    @classmethod
    def from_env(cls) -> "HTTPChatTranslator":
        api_key = os.getenv("LLM_API_KEY", "")
        base_url = os.getenv("LLM_BASE_URL", "")
        model = os.getenv("LLM_MODEL", "")
        if not all([api_key, base_url, model]):
            raise ValueError("Set LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL in the environment.")
        return cls(api_key=api_key, base_url=base_url.rstrip("/"), model=model)

    def translate(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "temperature": 0.1,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a careful professional translator. Follow terminology constraints exactly.",
                    },
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"].strip()
