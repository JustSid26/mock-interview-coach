"""Streamlit UI for the AI Mock Interview Coach."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from mock_interview_coach.agents.orchestrator import InterviewOrchestrator  # noqa: E402
from mock_interview_coach.utils.llm import rate_limit_user_message  # noqa: E402
from mock_interview_coach.utils.pdf_report import build_session_pdf  # noqa: E402


def init_session_state() -> None:
    defaults = {
        "orchestrator": None,
        "phase": "setup",
        "current_question": None,
        "messages": [],
        "coach_feedback": None,
        "turn_reviews": None,
        "transcript_path": None,
        "error": None,
        "pending_start": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar() -> None:
    st.sidebar.title("Session")
    if st.session_state.orchestrator and st.session_state.orchestrator.state:
        state = st.session_state.orchestrator.state
        q_cur, q_total = state.question_progress()
        st.sidebar.metric("Question", f"{q_cur} / {q_total}")
        avg = state.average_score()
        st.sidebar.metric(
            "Avg score",
            f"{avg} / 10" if avg else "—",
        )
        st.sidebar.write(f"**Difficulty:** {state.current_difficulty}")
        if state.topics_covered:
            st.sidebar.caption("Topics: " + ", ".join(state.topics_covered[:5]))


def render_setup() -> None:
    st.header("Configure your mock interview")
    col1, col2 = st.columns(2)
    with col1:
        role = st.text_input(
            "Target role",
            placeholder="e.g., Software Engineering Intern",
        )
        focus = st.selectbox(
            "Interview focus",
            ["mixed", "behavioral", "technical", "case"],
            index=0,
        )
    with col2:
        background = st.text_area(
            "Background / resume snippet (optional)",
            height=120,
            placeholder="Projects, skills, experience...",
        )
        turns = st.selectbox(
            "Number of questions",
            options=[5, 6, 7],
            index=1,
        )

    if st.button("Start interview", type="primary", disabled=not role.strip()):
        st.session_state.pending_start = {
            "role": role,
            "background": background,
            "focus": focus,
            "turns": turns,
        }
        st.rerun()

    pending = st.session_state.get("pending_start")
    if pending:
        try:
            with st.spinner("Interviewer is preparing your first question..."):
                orch = InterviewOrchestrator()
                orch.start_session(
                    role=pending["role"],
                    background=pending["background"],
                    focus=pending["focus"],
                    min_turns=pending["turns"],
                    max_turns=pending["turns"],
                )
                opening = orch.get_opening_question()
            st.session_state.orchestrator = orch
            st.session_state.current_question = opening
            st.session_state.phase = "interview"
            st.session_state.messages = [
                {
                    "role": "interviewer",
                    "content": opening["question"],
                    "meta": opening,
                }
            ]
            st.session_state.error = None
            del st.session_state.pending_start
            st.rerun()
        except Exception as exc:
            del st.session_state.pending_start
            msg = str(exc)
            st.session_state.error = msg
            st.error(f"Failed to start: {msg}")
            if "401" in msg or "403" in msg or "invalid" in msg.lower():
                st.info(
                    "API key invalid or denied. Create a key at "
                    "[Groq Console](https://console.groq.com/keys), "
                    "set `GROQ_API_KEY` in `.env`, and restart Streamlit."
                )
            elif "429" in msg or "rate" in msg.lower():
                st.info(rate_limit_user_message(Exception(msg)))
            elif "timeout" in msg.lower():
                st.info(
                    "Request timed out. Try `GROQ_MODEL=llama-3.1-8b-instant` in `.env` "
                    "or increase `GROQ_TIMEOUT_SEC`."
                )


def render_interview() -> None:
    st.header("Mock interview in progress")
    render_sidebar()

    for msg in st.session_state.messages:
        if msg["role"] == "interviewer":
            meta = msg.get("meta", {})
            badge = f"{meta.get('difficulty', 'medium')} · {meta.get('intent', '')}"
            with st.chat_message("assistant", avatar="🎤"):
                st.caption(badge)
                st.markdown(msg["content"])
        else:
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg["content"])

    answer = st.chat_input("Your answer...")
    if answer and st.session_state.current_question:
        st.session_state.messages.append({"role": "candidate", "content": answer})
        orch: InterviewOrchestrator = st.session_state.orchestrator
        q_data = st.session_state.current_question

        try:
            with st.spinner("Evaluating and preparing next question..."):
                result = orch.process_answer(answer, q_data)

            if result.get("interview_complete"):
                with st.spinner("Coach is preparing your debrief..."):
                    feedback, path, turn_reviews = orch.finalize_session()
                st.session_state.coach_feedback = feedback
                st.session_state.turn_reviews = turn_reviews
                st.session_state.transcript_path = path
                st.session_state.phase = "debrief"
                st.rerun()
            else:
                next_q = result["next_question"]
                st.session_state.current_question = next_q
                st.session_state.messages.append(
                    {
                        "role": "interviewer",
                        "content": next_q["question"],
                        "meta": next_q,
                    }
                )
                st.rerun()
        except Exception as exc:
            st.session_state.error = str(exc)
            st.error(f"Error: {exc}")


def _build_pdf_bytes() -> bytes | None:
    orch = st.session_state.orchestrator
    if not orch or not orch.state:
        return None
    state = orch.state
    turn_history = [asdict(t) for t in state.turn_history]
    return build_session_pdf(
        session_id=state.session_id,
        role=state.config.role,
        focus=state.config.focus,
        background=state.config.background,
        average_score=state.average_score(),
        coach_feedback=st.session_state.coach_feedback or "",
        turn_reviews=st.session_state.turn_reviews or [],
        turn_history=turn_history,
    )


def render_debrief() -> None:
    st.header("Interview complete")
    render_sidebar()
    st.success("Session saved.")

    # if st.session_state.transcript_path:
    #     st.caption(f"Transcript: `{st.session_state.transcript_path}`")

    pdf_bytes = _build_pdf_bytes()
    if pdf_bytes and st.session_state.orchestrator:
        sid = st.session_state.orchestrator.state.session_id
        st.download_button(
            label="Download PDF report",
            data=pdf_bytes,
            file_name=f"interview_report_{sid}.pdf",
            mime="application/pdf",
        )

    if st.session_state.coach_feedback:
        st.subheader("Session review")
        st.markdown(st.session_state.coach_feedback)

    reviews = st.session_state.turn_reviews or []
    if reviews:
        st.subheader("Questions, your answers & model answers")
        for item in reviews:
            n = item.get("turn_number", "?")
            with st.expander(f"Question {n}", expanded=True):
                st.markdown(f"**Question:** {item.get('question', '')}")
                st.markdown(f"**Your answer:** {item.get('candidate_answer', '')}")
                st.markdown(f"**Strong answer (exemplar):** {item.get('ideal_answer', '')}")
                points = item.get("key_points") or []
                if points:
                    st.markdown("**Key points to hit:**")
                    for pt in points:
                        st.markdown(f"- {pt}")

    with st.expander("Full conversation"):
        for msg in st.session_state.messages:
            role = "Interviewer" if msg["role"] == "interviewer" else "You"
            st.markdown(f"**{role}:** {msg['content']}")

    if st.button("Start new session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="AI Mock Interview Coach",
        page_icon="🎯",
        layout="wide",
    )
    init_session_state()

    st.title("AI Mock Interview Coach")
    st.caption(
        "Multi-agent adaptive interviews · Interviewer · Evaluator · Coach · Orchestrator"
    )

    if not os.getenv("GROQ_API_KEY"):
        st.warning(
            "Set `GROQ_API_KEY` in `.env` (see `.env.example`). "
            "Default model: llama-3.1-8b-instant (set GROQ_MODEL in .env)."
        )

    if st.session_state.error:
        st.error(st.session_state.error)

    phase = st.session_state.phase
    if phase == "setup":
        render_setup()
    elif phase == "interview":
        render_interview()
    else:
        render_debrief()


if __name__ == "__main__":
    main()
