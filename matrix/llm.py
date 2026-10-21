"""Provider-neutral LLM wrapper: structured JSON output, disk cache, logged fallbacks.

Every response is cached by (model, prompt_version, params, system, user, sample_idx, attempt) so
runs are reproducible and the demo can replay offline. API errors are NOT cached.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional, TypeVar

import numpy as np

from .schemas import HYPOTHESES

T = TypeVar("T")


class LLMError(RuntimeError):
    pass


class EmptyResponseError(LLMError):
    pass


class LLMClient(ABC):
    model: str = "unknown"
    params: dict = {}

    @abstractmethod
    def complete(self, system: str, user: str) -> str: ...



class GeminiClient(LLMClient):
    """Gemini API client. Requires google-genai and GEMINI_API_KEY."""

    def __init__(self, model: str, temperature: float = 1.0,
                 max_tokens: int = 800):
        import os
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")

        self._c = genai.Client(api_key=api_key)
        self.model = model
        self.params = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "response_mime_type": "application/json",
        }

    def complete(self, system: str, user: str) -> str:
        from google.genai import types

        response = self._c.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=self.params["temperature"],
                max_output_tokens=self.params["max_output_tokens"],
                response_mime_type=self.params["response_mime_type"],
            ),
        )
        text = response.text
        if not text:
            candidates = getattr(response, "candidates", None) or []
            candidate_details = [
                {
                    "finish_reason": str(getattr(candidate, "finish_reason", None)),
                    "finish_message": getattr(candidate, "finish_message", None),
                }
                for candidate in candidates
            ]
            prompt_feedback = getattr(response, "prompt_feedback", None)
            usage = getattr(response, "usage_metadata", None)
            diagnostics = {
                "candidates": candidate_details,
                "prompt_block_reason": str(
                    getattr(prompt_feedback, "block_reason", None)
                ),
                "usage": {
                    name: getattr(usage, name, None)
                    for name in (
                        "prompt_token_count",
                        "candidates_token_count",
                        "thoughts_token_count",
                        "total_token_count",
                    )
                },
            }
            raise EmptyResponseError(
                "Gemini returned an empty response; diagnostics="
                + json.dumps(diagnostics, default=str, sort_keys=True)
            )
        return text


class CachedLLM:
    def __init__(self, client: LLMClient, cache_dir, prompt_version: str):
        self.client, self.prompt_version = client, prompt_version
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.stats = {
            "calls": 0,
            "cache_hits": 0,
            "api_errors": 0,
            "empty_responses": 0,
            "invalid_outputs": 0,
        }

    def _key(self, system, user, sample_idx, attempt) -> str:
        blob = json.dumps([self.client.model, self.prompt_version, self.client.params, system, user,
                           sample_idx, attempt], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def complete(self, system: str, user: str, sample_idx: int = 0, attempt: int = 0) -> str:
        f = self.dir / f"{self._key(system, user, sample_idx, attempt)}.json"
        if f.exists():
            self.stats["cache_hits"] += 1
            return json.loads(f.read_text())["text"]
        self.stats["calls"] += 1
        try:
            text = self.client.complete(system, user)
        except EmptyResponseError:
            self.stats["empty_responses"] += 1
            raise
        except Exception as e:  # noqa: BLE001
            self.stats["api_errors"] += 1
            raise LLMError(str(e)) from e
        f.write_text(json.dumps({"text": text, "model": self.client.model, "params": self.client.params,
                                 "prompt_version": self.prompt_version, "ts": time.time()}))
        return text

    def get_json(self, system: str, user: str, validate: Callable[[dict], T], sample_idx: int = 0,
                 retries: int = 1) -> tuple[Optional[T], dict]:
        """Returns (parsed or None, meta). One retry on invalid output; never raises."""
        meta = {"attempts": 0, "errors": []}
        for attempt in range(retries + 1):
            meta["attempts"] += 1
            try:
                text = self.complete(system, user, sample_idx, attempt)
                return validate(extract_json(text)), meta
            except LLMError as e:
                meta["errors"].append(f"api: {e}")
            except (ValueError, KeyError, TypeError) as e:
                self.stats["invalid_outputs"] += 1
                meta["errors"].append(f"invalid: {e}")
        return None, meta


def extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("no JSON object")
    return json.loads(m.group(0))


def parse_distribution(d: dict, tol: float = 0.05) -> np.ndarray:
    """Validate and normalize a distribution over HYPOTHESES. Rejects wrong keys, negatives, bad sums."""
    if set(d) != set(HYPOTHESES):
        raise ValueError(f"keys {sorted(d)} != {list(HYPOTHESES)}")
    p = np.array([float(d[h]) for h in HYPOTHESES])
    if (p < 0).any() or not np.isfinite(p).all():
        raise ValueError("negative or non-finite probability")
    if abs(p.sum() - 1) > tol:
        raise ValueError(f"sums to {p.sum():.3f}")
    return p / p.sum()