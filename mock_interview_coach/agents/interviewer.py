"""Interviewer Agent — conducts adaptive mock interviews."""

from __future__ import annotations

from pathlib import Path

from mock_interview_coach.memory.conversation_state import ConversationState
from mock_interview_coach.utils.llm import chat_completion_json
from mock_interview_coach.utils.parser import validate_interviewer_output

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "interviewer_prompt.txt"


class InterviewerAgent:
    def __init__(self, prompt_path: Path | None = None):
        self.system_prompt = (prompt_path or PROMPT_PATH).read_text(encoding="utf-8")

    def generate_question(
        self,
        state: ConversationState,
        orchestrator_directive: str,
        last_evaluation: dict | None = None,
    ) -> dict:
        config = state.config
        eval_summary = ""
        if last_evaluation:
            eval_summary = (
                f"Latest evaluation (internal): overall={last_evaluation.get('overall_score')}, "
                f"weaknesses={last_evaluation.get('weaknesses')}, "
                f"follow_up_focus={last_evaluation.get('follow_up_focus')}"
            )

        user_content = f"""## Session
Role: {config.role}
Background: {config.background or "Not provided"}
Focus: {config.focus}
Current difficulty: {state.current_difficulty}
Turn: {state.current_turn + 1} of {config.max_turns}
Topics covered: {", ".join(state.topics_covered) or "none"}
Detected weaknesses: {", ".join(state.detected_weaknesses) or "none"}

## Orchestrator directive
{orchestrator_directive}

{eval_summary}

## Prior conversation
{state.conversation_summary_for_agents()}

Generate the next interview question as JSON."""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        return chat_completion_json(
            messages,
            validate_interviewer_output,
            temperature=0.6,
        )
