"""Coach Agent — final debrief and coaching feedback."""

from __future__ import annotations

from pathlib import Path
from statistics import mean

from mock_interview_coach.memory.conversation_state import ConversationState
from mock_interview_coach.utils.llm import chat_completion, chat_completion_json
from mock_interview_coach.utils.parser import validate_single_model_answer

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "coach_prompt.txt"
MODEL_ANSWER_PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "model_answer_prompt.txt"
)

DEFAULT_IDEAL = (
    "Review core concepts for this topic and practice a structured 60–90 second answer."
)


class CoachAgent:
    def __init__(
        self,
        prompt_path: Path | None = None,
        model_answer_prompt_path: Path | None = None,
    ):
        self.system_prompt = (prompt_path or PROMPT_PATH).read_text(encoding="utf-8")
        self.model_answer_prompt = (
            model_answer_prompt_path or MODEL_ANSWER_PROMPT_PATH
        ).read_text(encoding="utf-8")

    def _dimension_averages(self, state: ConversationState) -> dict[str, float]:
        if not state.turn_history:
            return {}
        dims = [
            "clarity",
            "technical_accuracy",
            "communication",
            "confidence",
            "depth",
            "structure",
            "problem_solving",
            "relevance",
        ]
        averages: dict[str, float] = {}
        for dim in dims:
            values = [
                turn.evaluation.get("scores", {}).get(dim, 0)
                for turn in state.turn_history
            ]
            if values:
                averages[dim] = round(mean(values), 1)
        return averages

    def generate_feedback(self, state: ConversationState) -> str:
        config = state.config
        dim_avgs = self._dimension_averages(state)

        turns_detail = []
        for turn in state.turn_history:
            turns_detail.append(
                f"### Turn {turn.turn_number} ({turn.difficulty})\n"
                f"**Q:** {turn.question}\n"
                f"**A:** {turn.candidate_answer}\n"
                f"**Overall score:** {turn.evaluation.get('overall_score')}\n"
                f"**Strengths:** {turn.evaluation.get('strengths')}\n"
                f"**Weaknesses:** {turn.evaluation.get('weaknesses')}\n"
                f"**Orchestrator:** {turn.orchestrator_decision}\n"
            )

        user_content = f"""## Session metadata
Role: {config.role}
Background: {config.background or "Not provided"}
Focus: {config.focus}
Session average score: {state.average_score()}
Dimension averages: {dim_avgs}
Topics covered: {state.topics_covered}
Detected weaknesses: {state.detected_weaknesses}

## Full transcript
{"".join(turns_detail)}

Produce the final coaching debrief in Markdown."""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        return chat_completion(messages, temperature=0.6)

    def _generate_one_model_answer(
        self, state: ConversationState, turn
    ) -> dict[str, str | list[str]]:
        config = state.config
        user_content = f"""## Session
Role: {config.role}
Background: {config.background or "Not provided"}
Focus: {config.focus}

## This turn only
Turn number: {turn.turn_number}
Question: {turn.question}
Candidate answer: {turn.candidate_answer}
Intent: {turn.question_intent}
Difficulty: {turn.difficulty}

Write one exemplar answer for THIS question only.
Return JSON:
{{
  "turn_number": {turn.turn_number},
  "ideal_answer": "concise strong interview answer (under 120 words)",
  "key_points": ["point 1", "point 2", "point 3"]
}}"""

        messages = [
            {"role": "system", "content": self.model_answer_prompt},
            {"role": "user", "content": user_content},
        ]
        return chat_completion_json(
            messages,
            validate_single_model_answer,
            temperature=0.4,
        )

    def generate_model_answers(self, state: ConversationState) -> list[dict]:
        """Exemplar answers per turn (one API call each for reliable JSON)."""
        reviews: list[dict] = []
        for turn in state.turn_history:
            try:
                model = self._generate_one_model_answer(state, turn)
            except Exception:
                model = {
                    "turn_number": turn.turn_number,
                    "ideal_answer": DEFAULT_IDEAL,
                    "key_points": [],
                }

            reviews.append(
                {
                    "turn_number": turn.turn_number,
                    "question": turn.question,
                    "candidate_answer": turn.candidate_answer,
                    "intent": turn.question_intent,
                    "difficulty": turn.difficulty,
                    "score": turn.evaluation.get("overall_score"),
                    "ideal_answer": model.get("ideal_answer") or DEFAULT_IDEAL,
                    "key_points": model.get("key_points", []),
                }
            )
        return reviews
