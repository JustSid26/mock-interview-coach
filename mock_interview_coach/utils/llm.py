"""LLM client wrapper — Groq (default: llama-3.1-8b-instant)."""

from __future__ import annotations

import json
import os
import re
import time
from functools import lru_cache
from typing import Any, Callable

from groq import Groq

from mock_interview_coach.utils.parser import extract_json, repair_json_text

# Lighter default — fewer tokens; separate TPD bucket from 70b on free tier
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant")
REQUEST_TIMEOUT_SEC = float(os.getenv("GROQ_TIMEOUT_SEC", "45"))
MAX_RETRIES = 3
JSON_MAX_RETRIES = 4
RETRY_DELAY_SEC = 0.8
_HARD_FAIL_CODES = {401, 403, 404}


def _api_key() -> str:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your Groq API key."
        )
    return key


@lru_cache(maxsize=1)
def get_client() -> Groq:
    return Groq(api_key=_api_key(), timeout=REQUEST_TIMEOUT_SEC)


def _models_to_try(requested: str | None) -> list[str]:
    primary = requested or DEFAULT_MODEL
    chain = [primary]
    fallback = FALLBACK_MODEL.strip()
    if fallback and fallback not in chain:
        chain.append(fallback)
    return chain


def _error_code(exc: Exception) -> int | None:
    code = getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


def _is_rate_limit_error(exc: Exception) -> bool:
    if _error_code(exc) == 429:
        return True
    text = str(exc).lower()
    return "rate_limit" in text or "rate limit" in text


def rate_limit_user_message(exc: Exception) -> str:
    """Human-readable rate limit guidance."""
    text = str(exc)
    wait = re.search(r"try again in (\d+m\d+(?:\.\d+)?s)", text, re.IGNORECASE)
    if wait:
        return (
            f"Daily token limit reached for this model. "
            f"Groq says to retry in **{wait.group(1)}**, or set "
            f"`GROQ_MODEL=llama-3.1-8b-instant` in `.env` (uses a lighter quota)."
        )
    return (
        "Groq rate limit reached. Wait a few minutes and retry, or set "
        "`GROQ_MODEL=llama-3.1-8b-instant` in your `.env` file."
    )


def _format_api_error(exc: Exception) -> str:
    code = _error_code(exc)
    msg = str(exc).strip() or repr(exc)
    return f"[{code}] {msg}" if code else msg


def _is_json_validate_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "json_validate_failed" in text
        or "failed to generate json" in text
        or "failed_generation" in text
    )


def _salvage_from_error(exc: Exception) -> str | None:
    text = str(exc)
    if "failed_generation" not in text:
        return None

    marker_idx = text.find("failed_generation")
    start = text.find("{", marker_idx)
    if start == -1:
        start = text.find("{")
    if start == -1:
        return None

    chunk = text[start:]
    repaired = repair_json_text(chunk)
    if repaired:
        return repaired

    for end_marker in ("'}", '"}', "})", "\n}"):
        pos = chunk.rfind(end_marker)
        if pos != -1:
            trial = chunk[: pos + len(end_marker)].rstrip("'\"")
            try:
                json.loads(trial)
                return trial
            except json.JSONDecodeError:
                pass
    return None


def chat_completion(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.7,
    response_format: dict[str, Any] | None = None,
    max_retries: int | None = None,
) -> str:
    """Call Groq with retries; auto-fallback model on 429."""
    client = get_client()
    want_json = bool(
        response_format and response_format.get("type") == "json_object"
    )
    attempts = max_retries or (JSON_MAX_RETRIES if want_json else MAX_RETRIES)
    all_errors: list[str] = []

    for model_name in _models_to_try(model):
        errors: list[str] = []

        for attempt in range(attempts):
            use_strict_json = want_json and attempt == 0
            temp = max(0.2, temperature - 0.15 * attempt) if want_json else temperature

            try:
                kwargs: dict[str, Any] = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": temp,
                }
                if use_strict_json:
                    kwargs["response_format"] = response_format

                response = client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                if not content or not str(content).strip():
                    raise ValueError("Empty response from LLM")
                return str(content).strip()

            except Exception as exc:
                formatted = _format_api_error(exc)
                errors.append(f"{model_name}: {formatted}")
                code = _error_code(exc)

                if _is_json_validate_error(exc):
                    salvaged = _salvage_from_error(exc)
                    if salvaged:
                        return salvaged
                    if attempt < attempts - 1:
                        time.sleep(RETRY_DELAY_SEC)
                        continue

                if _is_rate_limit_error(exc):
                    break  # try next model in chain

                if code in _HARD_FAIL_CODES:
                    break
                if attempt < attempts - 1:
                    time.sleep(RETRY_DELAY_SEC)

        all_errors.extend(errors)

    detail = all_errors[-1] if all_errors else "unknown error"
    last_exc_text = detail.lower()
    if "429" in last_exc_text or "rate_limit" in last_exc_text:
        raise RuntimeError(
            f"Groq rate limit exceeded. {rate_limit_user_message(Exception(detail))}"
        ) from None
    raise RuntimeError(f"Groq API call failed: {detail}") from None


def chat_completion_json(
    messages: list[dict[str, str]],
    validator: Callable[[dict[str, Any]], Any],
    *,
    model: str | None = None,
    temperature: float = 0.5,
    max_parse_attempts: int = 2,
) -> Any:
    """JSON completion: strict mode once, then plain text + parse/repair."""
    last_parse_error: Exception | None = None
    msgs = list(messages)

    for parse_try in range(max_parse_attempts):
        use_strict = parse_try == 0
        raw = chat_completion(
            msgs,
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"} if use_strict else None,
            max_retries=2 if use_strict else 3,
        )
        try:
            return validator(extract_json(raw))
        except (ValueError, json.JSONDecodeError) as exc:
            last_parse_error = exc
            repaired = repair_json_text(raw)
            if repaired:
                try:
                    return validator(extract_json(repaired))
                except (ValueError, json.JSONDecodeError):
                    pass
            msgs = msgs + [
                {
                    "role": "user",
                    "content": (
                        "Return ONLY valid JSON matching the schema. "
                        "Keep all string values concise."
                    ),
                }
            ]

    raise ValueError(f"Could not parse JSON: {last_parse_error}") from last_parse_error
