"""JSON extraction and validation for agent outputs."""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

T = TypeVar("T")


def repair_json_text(text: str) -> str | None:
    """Best-effort repair for truncated or slightly malformed JSON."""
    start = text.find("{")
    if start == -1:
        return None
    chunk = text[start:]

    for candidate in (chunk, chunk.rstrip("'\"}") + "}"):
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    # Close open strings and brackets (common with truncated Groq output)
    repaired = chunk
    if repaired.count('"') % 2 == 1:
        repaired += '"'
    open_brackets = repaired.count("[") - repaired.count("]")
    open_braces = repaired.count("{") - repaired.count("}")
    repaired += "]" * max(0, open_brackets)
    repaired += "}" * max(0, open_braces)

    try:
        json.loads(repaired)
        return repaired
    except json.JSONDecodeError:
        return None


def extract_json(text: str) -> dict[str, Any]:
    """Extract JSON object from LLM output, handling markdown fences."""
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError(f"Could not parse JSON from response: {text[:200]}...")


def validate_interviewer_output(data: dict[str, Any]) -> dict[str, Any]:
    required = {"question", "intent", "difficulty", "follow_up_needed"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Interviewer output missing fields: {missing}")

    difficulty = str(data["difficulty"]).lower()
    if difficulty not in ("easy", "medium", "hard"):
        data["difficulty"] = "medium"
    else:
        data["difficulty"] = difficulty

    data["follow_up_needed"] = bool(data["follow_up_needed"])
    data["question"] = str(data["question"]).strip()
    data["intent"] = str(data["intent"]).strip()
    return data


def validate_evaluator_output(data: dict[str, Any]) -> dict[str, Any]:
    dimensions = [
        "clarity",
        "technical_accuracy",
        "communication",
        "confidence",
        "depth",
        "structure",
        "problem_solving",
        "relevance",
    ]
    scores = data.get("scores", {})
    if not isinstance(scores, dict):
        raise ValueError("Evaluator output must include 'scores' object")

    normalized_scores: dict[str, float] = {}
    for dim in dimensions:
        val = scores.get(dim, 5)
        try:
            normalized_scores[dim] = max(0.0, min(10.0, float(val)))
        except (TypeError, ValueError):
            normalized_scores[dim] = 5.0

    data["scores"] = normalized_scores
    if "overall_score" not in data:
        data["overall_score"] = round(
            sum(normalized_scores.values()) / len(normalized_scores), 1
        )
    else:
        data["overall_score"] = round(float(data["overall_score"]), 1)

    data["strengths"] = list(data.get("strengths", []))[:5]
    data["weaknesses"] = list(data.get("weaknesses", []))[:5]
    data["follow_up_recommended"] = bool(data.get("follow_up_recommended", False))
    data["follow_up_focus"] = str(data.get("follow_up_focus", ""))
    adj = str(data.get("difficulty_adjustment", "maintain")).lower()
    if adj not in ("increase", "decrease", "maintain"):
        adj = "maintain"
    data["difficulty_adjustment"] = adj
    return data


def validate_single_model_answer(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "turn_number": int(data.get("turn_number", 1)),
        "ideal_answer": str(data.get("ideal_answer", "")).strip(),
        "key_points": [
            str(p).strip() for p in data.get("key_points", []) if p
        ][:6],
    }


def validate_model_answers_output(data: dict[str, Any]) -> list[dict[str, Any]]:
    turns = data.get("turns", [])
    if not isinstance(turns, list):
        raise ValueError("Model answers output must include 'turns' list")

    validated: list[dict[str, Any]] = []
    for entry in turns:
        if not isinstance(entry, dict):
            continue
        validated.append(
            {
                "turn_number": int(entry.get("turn_number", len(validated) + 1)),
                "ideal_answer": str(entry.get("ideal_answer", "")).strip(),
                "key_points": [
                    str(p).strip() for p in entry.get("key_points", []) if p
                ][:6],
            }
        )
    return validated


def parse_with_retry(
    text: str,
    validator: Any,
    *,
    max_attempts: int = 1,
) -> dict[str, Any]:
    """Parse JSON and validate; re-raises on failure."""
    last_error: Exception | None = None
    for _ in range(max_attempts):
        try:
            data = extract_json(text)
            return validator(data)
        except (ValueError, json.JSONDecodeError, KeyError) as exc:
            last_error = exc
    raise ValueError(f"Failed to parse agent output: {last_error}") from last_error
