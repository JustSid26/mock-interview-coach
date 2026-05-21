"""Orchestrator — coordinates agents and adaptive interview flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mock_interview_coach.agents.coach import CoachAgent
from mock_interview_coach.agents.evaluator import EvaluatorAgent
from mock_interview_coach.agents.interviewer import InterviewerAgent
from mock_interview_coach.memory.conversation_state import (
    ConversationState,
    InterviewConfig,
)


@dataclass
class OrchestratorDecision:
    action: str  # follow_up | new_topic | increase_difficulty | decrease_difficulty | wrap_up
    reasoning: str
    probe_deeper: bool
    change_topic: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reasoning": self.reasoning,
            "probe_deeper": self.probe_deeper,
            "change_topic": self.change_topic,
        }


class InterviewOrchestrator:
    """Coordinates interviewer, evaluator, and coach through adaptive logic."""

    DIFFICULTY_ORDER = ("easy", "medium", "hard")

    def __init__(
        self,
        interviewer: InterviewerAgent | None = None,
        evaluator: EvaluatorAgent | None = None,
        coach: CoachAgent | None = None,
    ):
        self.interviewer = interviewer or InterviewerAgent()
        self.evaluator = evaluator or EvaluatorAgent()
        self.coach = coach or CoachAgent()
        self.state: ConversationState | None = None

    def start_session(
        self,
        role: str,
        background: str,
        focus: str,
        min_turns: int = 5,
        max_turns: int = 7,
    ) -> ConversationState:
        focus = focus.lower().strip()
        if focus not in ("behavioral", "technical", "case", "mixed"):
            focus = "mixed"

        min_turns = min(min_turns, max_turns)
        config = InterviewConfig(
            role=role.strip(),
            background=background.strip(),
            focus=focus,
            min_turns=min_turns,
            max_turns=max_turns,
        )
        self.state = ConversationState(config=config, status="in_progress")
        return self.state

    def _adjust_difficulty(self, evaluation: dict[str, Any]) -> None:
        assert self.state is not None
        adj = evaluation.get("difficulty_adjustment", "maintain")
        idx = self.DIFFICULTY_ORDER.index(self.state.current_difficulty)

        if adj == "increase" and idx < len(self.DIFFICULTY_ORDER) - 1:
            self.state.current_difficulty = self.DIFFICULTY_ORDER[idx + 1]
        elif adj == "decrease" and idx > 0:
            self.state.current_difficulty = self.DIFFICULTY_ORDER[idx - 1]

    def decide_next_step(self, evaluation: dict[str, Any]) -> OrchestratorDecision:
        """Rule-based orchestration using evaluator signals and session state."""
        assert self.state is not None
        state = self.state

        overall = float(evaluation.get("overall_score", 5.0))
        follow_up = evaluation.get("follow_up_recommended", False)
        follow_focus = evaluation.get("follow_up_focus", "")

        if state.must_continue_for_min_turns():
            if follow_up and overall < 7.0:
                return OrchestratorDecision(
                    action="follow_up",
                    reasoning=f"Minimum turns not met; probing: {follow_focus}",
                    probe_deeper=True,
                    change_topic=False,
                )
            return OrchestratorDecision(
                action="new_topic",
                reasoning="Building coverage toward minimum turn count",
                probe_deeper=False,
                change_topic=True,
            )

        if state.should_end_interview():
            return OrchestratorDecision(
                action="wrap_up",
                reasoning="Maximum turns reached",
                probe_deeper=False,
                change_topic=False,
            )

        if follow_up and overall < 6.5:
            return OrchestratorDecision(
                action="follow_up",
                reasoning=f"Weak or shallow answer; focus: {follow_focus}",
                probe_deeper=True,
                change_topic=False,
            )

        if overall >= 7.5:
            self._adjust_difficulty(evaluation)
            return OrchestratorDecision(
                action="increase_difficulty",
                reasoning="Strong performance — advancing difficulty",
                probe_deeper=False,
                change_topic=True,
            )

        if overall <= 4.5:
            self._adjust_difficulty(evaluation)
            return OrchestratorDecision(
                action="decrease_difficulty",
                reasoning="Struggling — simplifying and redirecting",
                probe_deeper=False,
                change_topic=True,
            )

        if follow_up and 5.0 <= overall < 7.0:
            return OrchestratorDecision(
                action="follow_up",
                reasoning="Moderate answer with gaps worth one probe",
                probe_deeper=True,
                change_topic=False,
            )

        return OrchestratorDecision(
            action="new_topic",
            reasoning="Adequate answer — moving to new topic",
            probe_deeper=False,
            change_topic=True,
        )

    def _directive_text(self, decision: OrchestratorDecision) -> str:
        return (
            f"Action: {decision.action}\n"
            f"Reasoning: {decision.reasoning}\n"
            f"Probe deeper: {decision.probe_deeper}\n"
            f"Change topic: {decision.change_topic}"
        )

    def get_opening_question(self) -> dict[str, Any]:
        assert self.state is not None
        decision = OrchestratorDecision(
            action="new_topic",
            reasoning="Session opening",
            probe_deeper=False,
            change_topic=True,
        )
        question = self.interviewer.generate_question(
            self.state,
            self._directive_text(decision),
            last_evaluation=None,
        )
        self.state.current_difficulty = question.get("difficulty", "medium")
        self.set_pending_question(question)
        return question

    def process_answer(
        self, answer: str, question_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Evaluate answer, orchestrate, record turn. Returns next step info."""
        assert self.state is not None

        question_text = question_data["question"]
        evaluation = self.evaluator.evaluate_response(
            self.state, question_text, answer
        )
        self._adjust_difficulty(evaluation)

        # Stop before recording if we already hit the question limit
        if self.state.current_turn >= self.state.config.max_turns:
            result: dict[str, Any] = {
                "evaluation": evaluation,
                "decision": {
                    "action": "wrap_up",
                    "reasoning": "Maximum turns already reached",
                    "probe_deeper": False,
                    "change_topic": False,
                },
                "interview_complete": True,
                "next_question": None,
            }
            self.state.status = "completed"
            return result

        decision = self.decide_next_step(evaluation)

        self.state.add_turn(
            question_data={
                "question": question_text,
                "intent": question_data.get("intent") or "interview",
                "difficulty": question_data.get(
                    "difficulty", self.state.current_difficulty
                ),
            },
            answer=answer,
            evaluation=evaluation,
            decision=decision.to_dict(),
        )
        self.state.pending_question = None

        result: dict[str, Any] = {
            "evaluation": evaluation,
            "decision": decision.to_dict(),
            "interview_complete": False,
            "next_question": None,
        }

        if self.state.should_end_interview():
            result["interview_complete"] = True
            self.state.status = "completed"
            return result

        next_q = self.interviewer.generate_question(
            self.state,
            self._directive_text(decision),
            last_evaluation=evaluation,
        )
        self.state.current_difficulty = next_q.get(
            "difficulty", self.state.current_difficulty
        )
        self.state.pending_question = next_q
        result["next_question"] = next_q
        return result

    def set_pending_question(self, question_data: dict[str, Any]) -> None:
        assert self.state is not None
        self.state.pending_question = question_data
        intent = question_data.get("intent", "")
        if intent and intent not in self.state.topics_covered:
            self.state.topics_covered.append(intent)

    def finalize_session(self) -> tuple[str, str, list[dict[str, Any]]]:
        """Generate coach feedback, model answers, and save transcript."""
        assert self.state is not None
        self.state.status = "completed"
        feedback = self.coach.generate_feedback(self.state)
        turn_reviews = self.coach.generate_model_answers(self.state)
        self.state.coach_feedback = feedback
        self.state.turn_reviews = turn_reviews
        path = self.state.save_transcript()
        return feedback, str(path), turn_reviews
