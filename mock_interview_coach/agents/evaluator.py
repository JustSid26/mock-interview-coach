"""Evaluator Agent — scores each candidate response."""

from __future__ import annotations

from pathlib import Path

from mock_interview_coach.memory.conversation_state import ConversationState
from mock_interview_coach.utils.llm import chat_completion_json
from mock_interview_coach.utils.parser import validate_evaluator_output

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "evaluator_prompt.txt"


class EvaluatorAgent:
    def __init__(self, prompt_path: Path | None = None):
        self.system_prompt = (prompt_path or PROMPT_PATH).read_text(encoding="utf-8")

    def evaluate_response(
        self,
        state: ConversationState,
        question: str,
        answer: str,
    ) -> dict:
        config = state.config
        user_content = f"""## Session
Role: {config.role}
Background: {config.background or "Not provided"}
Focus: {config.focus}

## Question
{question}

## Candidate answer
{answer}

## Prior turns (summary)
{state.conversation_summary_for_agents()}

Evaluate this answer. Return JSON only."""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        return chat_completion_json(
            messages,
            validate_evaluator_output,
            temperature=0.3,
        )
