"""AI Quadratic Functions Explorer: Gemini-assisted differentiated learning.

Gemini suggests practice-problem parameters and offers hints. Python creates
and independently verifies all mathematical answers. Keep API keys in Secrets.
"""

from __future__ import annotations

import json
import math
import os
import random
import re

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from google import genai
from google.genai import types

from math_core import (
    LEVELS, build_problem, candidates, choose_unique, validate_ai_parameters,
)

MODEL = "gemini-3.5-flash-lite"  # Change only if your account cannot access this model.

st.set_page_config(page_title="AI Quadratic Functions Explorer", page_icon="📈", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1120px; padding-top: 1.5rem;}
.hero {border: 1px solid rgba(126,126,126,.18); border-radius: 18px;
       padding: 20px 24px; margin-bottom: 14px; background: rgba(120,125,255,.06);}
.hero h1 {margin: 0 0 .3rem 0; font-size: 2.05rem;}
.hero p {margin: 0; opacity: .84;}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<div class="hero">
<h1>📈 AI Quadratic Functions Explorer</h1>
<p>One mathematics topic · Three learning levels · Interactive parabola graphs</p>
</div>
""", unsafe_allow_html=True)


def get_secret_key() -> str:
    try:
        return str(st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))).strip()
    except FileNotFoundError:
        return os.getenv("GEMINI_API_KEY", "").strip()


def generate_ai_parameters(client: genai.Client, level: str, recent: list[str]) -> dict[str, int]:
    if level == "Basic":
        spec = 'Return JSON with exactly one integer field: {"r": number}. Use r from 1 to 12.'
    elif level == "Standard":
        spec = ('Return JSON with exactly two integer fields: {"p": number, "q": number}. '
                'Choose distinct nonzero integers p and q from -7 to 7; p+q must not be 0.')
    else:
        spec = ('Return JSON with exactly two integer fields: {"m": number, "T": number}. '
                'm is an integer from 1 to 4; T is one of 4, 6, 8, 10, or 12.')
    prompt = f"""You are helping a teacher differentiate practice in quadratic functions.
Difficulty level: {level}.
{spec}
Choose varied values, preferably different from these recent parameter IDs: {recent}.
Return ONLY valid JSON. Do not provide equations, answers, or explanations.
Python independently constructs and verifies the problem from your parameters.
"""
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    if not response.text:
        raise ValueError("Gemini returned an empty response")
    return validate_ai_parameters(level, json.loads(response.text))


def hint_reveals_answer(hint: str, problem: dict) -> bool:
    """Reject explicit answers; still treat AI text as unverified teaching support."""
    patterns = [
        r"\b(?:the\s+)?(?:answer|solution|root|roots|landing\s+time|maximum\s+height)\s*(?:is|are|=|:)\s*[-+]?\d",
        r"\b[xt]\s*=\s*[-+]?\d",
        r"\b(?:vertex|intercepts?)\s*(?:is|are|at)\s*\(\s*[-+]?\d",
    ]
    return any(re.search(pattern, hint, flags=re.I) for pattern in patterns)


def get_ai_hint(client: genai.Client, problem: dict) -> str:
    prompt = f"""Act as a supportive mathematics tutor.
Level: {problem['level']}
Equation: {problem['equation']}
Student question: {problem['statement']}
Write ONE short strategy hint in plain English, appropriate to the student's level.
Basic: suggest using y = 0 and square roots.
Standard: suggest factoring and symmetry of the roots.
Advanced: suggest finding the roots and vertex in the real-world context.
Do NOT provide numerical roots, vertex coordinates, landing time, maximum height,
or a complete solution. Do not reveal any final answer.
"""
    response = client.models.generate_content(model=MODEL, contents=prompt)
    hint = (response.text or "").strip()
    if not hint or hint_reveals_answer(hint, problem):
        return problem["fallback_hint"]
    return hint


def parabola_chart(problem: dict, show_key_points: bool) -> go.Figure:
    a, b, c = problem["a"], problem["b"], problem["c"]
    roots = problem["roots"]
    left, right = (0, roots[1]) if problem["level"] == "Advanced" else (min(roots)-3, max(roots)+3)
    xs = np.linspace(left, right, 361)
    ys = a*xs*xs + b*xs + c
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Quadratic function",
                             line=dict(color="#6750D8", width=4)))
    fig.add_hline(y=0, line_color="gray", opacity=0.5)
    if problem["level"] != "Advanced":
        fig.add_vline(x=0, line_color="gray", opacity=0.35)
    if show_key_points:
        vx, vy = problem["vertex"]
        fig.add_trace(go.Scatter(x=list(roots), y=[0, 0], mode="markers+text",
                                 name="Roots", text=[f"x={r:g}" for r in roots],
                                 textposition="bottom center", marker=dict(size=11, color="#17A398")))
        fig.add_trace(go.Scatter(x=[vx], y=[vy], mode="markers+text", name="Vertex",
                                 text=[f"({vx:g}, {vy:g})"], textposition="top center",
                                 marker=dict(size=12, color="#F59E0B")))
    fig.update_layout(height=405, margin=dict(l=10, r=15, t=22, b=10),
                      xaxis_title="Time t (seconds)" if problem["level"] == "Advanced" else "x",
                      yaxis_title="Height h(t) (metres)" if problem["level"] == "Advanced" else "y",
                      template="plotly_white", hovermode="x unified",
                      legend=dict(orientation="h", y=1.12))
    if problem["level"] == "Advanced":
        fig.update_xaxes(range=[0, roots[1]])
        fig.update_yaxes(rangemode="tozero")
    return fig


if "history" not in st.session_state or not isinstance(st.session_state.history, dict):
    st.session_state.history = {name: [] for name in LEVELS}
for name in LEVELS:
    if name not in st.session_state.history or not isinstance(st.session_state.history[name], list):
        st.session_state.history[name] = []
if "problem" not in st.session_state or not isinstance(st.session_state.problem, dict):
    st.session_state.problem = None
if "hint" not in st.session_state:
    st.session_state.hint = None
if "serial" not in st.session_state:
    st.session_state.serial = 0

api_key = get_secret_key()
client = genai.Client(api_key=api_key) if api_key else None
if client:
    st.success("Gemini AI is connected. AI generates problem parameters and provides hints.", icon="✅")
else:
    st.warning("DEMO MODE: No Gemini API key found. Problems and hints are generated locally, NOT by AI.")

left_col, right_col = st.columns([2, 3], gap="large")
with left_col:
    level = st.selectbox("Choose difficulty", LEVELS,
                         help="Basic: roots | Standard: roots and vertex | Advanced: ball flight")
    st.caption({"Basic": "🌱 Roots of y = x² − r²",
                "Standard": "📘 Factoring and vertex",
                "Advanced": "🚀 Real-world projectile motion"}[level])
    if st.session_state.get("last_level") != level:
        st.session_state.last_level = level
        st.session_state.problem = None
        st.session_state.hint = None
    generate = st.button("✨ Generate AI Problem" if client else "🎲 Generate Demo Problem",
                         type="primary", use_container_width=True)
    if st.button("Reset practice history", use_container_width=True):
        st.session_state.history = {name: [] for name in LEVELS}
        st.session_state.problem = None
        st.session_state.hint = None
        st.rerun()
    st.caption(f"This level: {len(st.session_state.history[level])}/{len(candidates(level))} unique examples used")

if generate:
    try:
        with st.spinner("Preparing a mathematically verified problem..."):
            proposed = (generate_ai_parameters(client, level, st.session_state.history[level][-5:])
                        if client else random.choice(candidates(level)))
            params = choose_unique(level, proposed, st.session_state.history[level])
            problem = build_problem(level, params)
            st.session_state.history[level].append(problem["signature"])
            st.session_state.problem = problem
            st.session_state.hint = None
            st.session_state.serial += 1
    except Exception as exc:
        st.error("Could not generate a new problem. Please try again or reset the practice history.")
        st.caption(f"Error type: {type(exc).__name__}. {str(exc)[:180]}")

problem = st.session_state.problem
if problem and problem.get("level") == level:
    with right_col:
        st.markdown(f"**Learning goal:** {problem['learning']}")
        st.latex(problem["equation"])
        st.write(problem["statement"])
        with st.form(key=f"answer_{st.session_state.serial}"):
            entries = []
            for idx, label in enumerate(problem["labels"]):
                entries.append(st.number_input(label, value=0.0, step=0.25,
                                               key=f"field_{st.session_state.serial}_{idx}"))
            check = st.form_submit_button("✅ Check Answer", use_container_width=True)
        if check:
            if all(math.isfinite(got) and math.isclose(got, expected, abs_tol=1e-6)
                   for got, expected in zip(entries, problem["answers"])):
                st.success("Correct — well done! 🎉")
            else:
                st.warning("Not quite. Explore the graph or request a hint, then try again.")
        if st.button("💡 Get AI Hint" if client else "💡 Get Demo Hint"):
            if client:
                try:
                    with st.spinner("Gemini is preparing a hint..."):
                        st.session_state.hint = get_ai_hint(client, problem)
                except Exception:
                    st.session_state.hint = problem["fallback_hint"]
                    st.caption("Gemini is unavailable. Showing a predefined hint instead.")
            else:
                st.session_state.hint = problem["fallback_hint"]
        if st.session_state.hint:
            st.info(st.session_state.hint)

    st.divider()
    chart_col, expl_col = st.columns([3, 2], gap="large")
    with chart_col:
        st.subheader("Explore the parabola")
        reveal = st.checkbox("Reveal roots and vertex on the graph", value=False,
                             key=f"reveal_{st.session_state.serial}")
        st.plotly_chart(parabola_chart(problem, reveal), use_container_width=True,
                        config={"scrollZoom": False})
        st.caption("Drag to zoom, use the toolbar to reset. Key points stay hidden until you reveal them.")
    with expl_col:
        st.subheader("Guided solution")
        with st.expander("Show Step-by-Step Solution"):
            for i, step in enumerate(problem["solution"], 1):
                st.write(f"**Step {i}.** {step}")
            st.success("Python verified the roots, vertex, and equation before displaying it.")
        st.markdown("**How AI helps:** Gemini selects new practice parameters and supplies a level-specific hint. "
                    "Python independently calculates and checks the mathematics.")
else:
    with right_col:
        st.info("Choose a difficulty level and click Generate to start. The interactive graph appears below the question.")

st.divider()
st.caption("Educational prototype | Gemini API key is never included in the public code | Demo mode is not AI integration")

