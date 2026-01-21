"""LLM backends: local (Ollama), DeepSeek, OpenAI, Gemini."""

import os
import sys
import requests
from typing import Protocol
from pathlib import Path

# Добавляем путь к utils для импорта retry
sys.path.insert(0, str(Path(__file__).parent.parent))

from settings import LLM_MODEL, OLLAMA_BASE
from settings import DEEPSEEK_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY
from utils.retry import retry_with_backoff

class LLMBackend(Protocol):
    def generate(self, prompt: str) -> str: ...

class OllamaBackend:
    """Текущий локальный бэкенд через Ollama /api/generate."""

    def generate(self, prompt: str) -> str:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "top_k": 30,
                    "num_predict": 450,
                    "repeat_penalty": 1.5,
                },
            },
            timeout=150,
        )
        if resp.status_code != 200:
            return f"Ollama HTTP {resp.status_code}: {resp.text[:200]}"
        data = resp.json()
        return data.get("response", "").strip() or "Нет ответа от локальной модели"


class DeepSeekBackend:
    """Внешний LLM через DeepSeek Chat API."""

    def __init__(self, model: str = "deepseek-reasoner") -> None:
        """
        Args:
            model: Модель DeepSeek. По умолчанию "deepseek-reasoner" (для ответов).
                   Для саммари используйте "deepseek-chat" (быстрее).
        """
        if not DEEPSEEK_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY не задан в settings.py")
        self.api_key = DEEPSEEK_API_KEY
        self.base_url = "https://api.deepseek.com/v1"
        self.model = model

    @retry_with_backoff(
        max_retries=3,
        initial_delay=1.0,
        max_delay=30.0,
        exceptions=(requests.exceptions.RequestException, requests.exceptions.HTTPError),
    )
    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Ты кратко и технически точно отвечаешь по данному контексту.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": 0.1,
            "max_tokens": 3000,
        }
        resp = requests.post(url, headers=headers, json=data, timeout=150)
        resp.raise_for_status()
        j = resp.json()
        return j["choices"][0]["message"]["content"].strip()


class OpenAIBackend:
    """Внешний LLM через OpenAI Chat Completions (gpt‑4o / gpt‑4o-mini)."""

    def __init__(self) -> None:
        self.base_url = "https://api.openai.com/v1"
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY не задан в settings.py")
        self.api_key = OPENAI_API_KEY

    @retry_with_backoff(
        max_retries=3,
        initial_delay=1.0,
        max_delay=30.0,
        exceptions=(requests.exceptions.RequestException, requests.exceptions.HTTPError),
    )
    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "gpt-4o-mini",  # быстрый/дешевый чат‑модели OpenAI [web:64][web:70]
            "messages": [
                {
                    "role": "system",
                    "content": "Ты кратко и без галлюцинаций отвечаешь по данному контексту.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": 0.1,
            "max_tokens": 1024,
        }
        resp = requests.post(url, headers=headers, json=data, timeout=150)
        resp.raise_for_status()
        j = resp.json()
        return j["choices"][0]["message"]["content"].strip()


class GeminiBackend:
    """Внешний LLM через Gemini API (gemini-3-flash-preview)."""

    def __init__(self, model: str = "gemini-3-flash-preview") -> None:
        """
        Args:
            model: Модель Gemini. По умолчанию "gemini-3-flash-preview".
                   Доступные модели: gemini-3-flash-preview, gemini-2.5-flash, gemini-1.5-flash, gemini-1.5-pro
        """
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY не задан в settings.py")
        self.api_key = GEMINI_API_KEY
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.model = model

    @retry_with_backoff(
        max_retries=3,
        initial_delay=1.0,
        max_delay=30.0,
        exceptions=(requests.exceptions.RequestException, requests.exceptions.HTTPError, RuntimeError),
    )
    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/models/{self.model}:generateContent"
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "systemInstruction": {
                "parts": [
                    {
                        "text": "Ты кратко и технически точно отвечаешь по данному контексту."
                    }
                ]
            },
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 3000,
            }
        }
        resp = requests.post(url, headers=headers, json=data, timeout=150)
        
        # Детальное логирование ошибок для отладки
        if resp.status_code != 200:
            error_detail = resp.text
            try:
                error_json = resp.json()
                error_detail = str(error_json)
            except:
                pass
            raise RuntimeError(
                f"Gemini API error {resp.status_code}: {error_detail[:500]}"
            )
        
        j = resp.json()
        
        # Извлекаем текст из ответа Gemini
        if "candidates" in j and len(j["candidates"]) > 0:
            candidate = j["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                if len(parts) > 0 and "text" in parts[0]:
                    return parts[0]["text"].strip()
        
        raise RuntimeError(f"Неожиданный формат ответа Gemini: {j}")
