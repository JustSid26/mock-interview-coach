"""Interview session state and transcript management."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class TurnRecord:
    turn_number: int
    question: str
    question_intent: str
    difficulty: str
    candidate_answer: str
    evaluation: dict[str, Any]
    orchestrator_decision: dict[str, Any]


@dataclass
class InterviewConfig:
    role: str
    background: str
    focus: str  # behavioral | technical | case | mixed
    min_turns: int = 5
    max_turns: int = 7


@dataclass
class ConversationState:
    config: InterviewConfig
    current_turn: int = 0
    current_difficulty: str = "medium"
    topics_covered: list[str] = field(default_factory=list)
    detected_weaknesses: list[str] = field(default_factory=list)
    turn_history: list[TurnRecord] = field(default_factory=list)
    aggregated_scores: list[float] = field(default_factory=list)
    turn_reviews: list[dict[str, Any]] = field(default_factory=list)
    coach_feedback: str = ""
    session_id: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    )
    status: str = "setup"  # setup | in_progress | completed
    pending_question: dict[str, Any] | None = None

    def add_turn(
        self,
        question_data: dict[str, Any],
        answer: str,
        evaluation: dict[str, Any],
        decision: dict[str, Any],
    ) -> None:
        self.current_turn += 1
        self.turn_history.append(
            TurnRecord(
                turn_number=self.current_turn,
                question=question_data["question"],
                question_intent=question_data.get("intent", ""),
                difficulty=question_data.get("difficulty", self.current_difficulty),
                candidate_answer=answer,
                evaluation=evaluation,
                orchestrator_decision=decision,
            )
        )
        self.aggregated_scores.append(float(evaluation.get("overall_score", 5.0)))

        for weakness in evaluation.get("weaknesses", []):
            if weakness and weakness not in self.detected_weaknesses:
                self.detected_weaknesses.append(weakness)

        intent = question_data.get("intent", "")
        if intent and intent not in self.topics_covered:
            self.topics_covered.append(intent)

    def should_end_interview(self) -> bool:
        return self.current_turn >= self.config.max_turns

    def question_progress(self) -> tuple[int, int]:
        """(current question number, total) — 1-based, capped at max_turns."""
        total = self.config.max_turns
        if self.status == "completed":
            current = min(self.current_turn, total)
        else:
            current = min(self.current_turn + 1, total)
        return current, total

    def must_continue_for_min_turns(self) -> bool:
        return self.current_turn < self.config.min_turns

    def average_score(self) -> float:
        if not self.aggregated_scores:
            return 0.0
        return round(sum(self.aggregated_scores) / len(self.aggregated_scores), 1)

    def transcript_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "config": asdict(self.config),
            "status": self.status,
            "current_difficulty": self.current_difficulty,
            "topics_covered": self.topics_covered,
            "detected_weaknesses": self.detected_weaknesses,
            "average_score": self.average_score(),
            "coach_feedback": self.coach_feedback,
            "turn_reviews": self.turn_reviews,
            "turns": [asdict(t) for t in self.turn_history],
        }

    def save_transcript(self, transcripts_dir: Path | None = None) -> Path:
        base = transcripts_dir or Path(__file__).resolve().parents[1] / "transcripts"
        base.mkdir(parents=True, exist_ok=True)
        path = base / f"session_{self.session_id}.json"
        path.write_text(json.dumps(self.transcript_dict(), indent=2), encoding="utf-8")
        return path

    def conversation_summary_for_agents(self) -> str:
        if not self.turn_history:
            return "No prior turns yet."
        lines = []
        for turn in self.turn_history:
            lines.append(f"Turn {turn.turn_number} [{turn.difficulty}]")
            lines.append(f"Q: {turn.question}")
            lines.append(f"A: {turn.candidate_answer}")
            scores = turn.evaluation.get("scores", {})
            lines.append(f"Score: {turn.evaluation.get('overall_score')} | {scores}")
            lines.append("")
        return "\n".join(lines)
