"""Pure mathematics for AI Quadratic Functions Explorer (no AI dependency)."""

from __future__ import annotations

import math
from typing import Any

LEVELS = ("Basic", "Standard", "Advanced")
ADVANCED_TIMES = (4, 6, 8, 10, 12)


def polynomial_latex(a: int, b: int, c: int, variable: str = "x", label: str = "y") -> str:
    """Convert integer quadratic coefficients into readable LaTeX."""
    if a == 0:
        raise ValueError("A quadratic must have a nonzero leading coefficient")
    lead_abs = "" if abs(a) == 1 else str(abs(a))
    lead = ("-" if a < 0 else "") + lead_abs + variable + "^2"
    terms = [lead]
    if b:
        linear_abs = "" if abs(b) == 1 else str(abs(b))
        terms.append(("+" if b > 0 else "-") + " " + linear_abs + variable)
    if c:
        terms.append(("+" if c > 0 else "-") + " " + str(abs(c)))
    return label + " = " + " ".join(terms)


def candidates(level: str) -> list[dict[str, int]]:
    """Valid candidate parameters; finite pools allow no-repeat practice."""
    if level == "Basic":
        return [{"r": r} for r in range(1, 13)]
    if level == "Standard":
        return [
            {"p": p, "q": q}
            for p in range(-7, 8)
            for q in range(p + 1, 8)
            if p != 0 and q != 0 and p + q != 0
        ]
    if level == "Advanced":
        return [{"m": m, "T": t} for m in range(1, 5) for t in ADVANCED_TIMES]
    raise ValueError("Unknown difficulty level")


def validate_ai_parameters(level: str, data: Any) -> dict[str, int]:
    """Reject malformed or invalid responses; never trust generated math."""
    if not isinstance(data, dict):
        raise ValueError("Gemini must return a JSON object")
    expected = {"Basic": ("r",), "Standard": ("p", "q"), "Advanced": ("m", "T")}
    if level not in expected:
        raise ValueError("Unknown difficulty level")
    if set(data) != set(expected[level]) or any(type(data[k]) is not int for k in expected[level]):
        raise ValueError("Gemini returned the wrong integer parameters")
    clean = {key: data[key] for key in expected[level]}
    if level == "Standard" and clean["p"] > clean["q"]:
        clean = {"p": clean["q"], "q": clean["p"]}
    if clean not in candidates(level):
        raise ValueError("Gemini parameters are outside the allowed values")
    return clean


def choose_unique(level: str, preferred: dict[str, int], seen: list[str]) -> dict[str, int]:
    """Prefer AI's parameters; use a new valid combination when it repeats."""
    options = candidates(level)
    if preferred not in options:
        raise ValueError("Invalid candidate")
    start = options.index(preferred)
    for index in range(len(options)):
        value = options[(start + index) % len(options)]
        if signature(level, value) not in seen:
            return value
    raise ValueError("All examples at this level have been used. Restart the session.")


def signature(level: str, params: dict[str, int]) -> str:
    if level == "Basic":
        return f"B:{params['r']}"
    if level == "Standard":
        return f"S:{params['p']}:{params['q']}"
    if level == "Advanced":
        return f"A:{params['m']}:{params['T']}"
    raise ValueError("Unknown difficulty level")


def build_problem(level: str, params: dict[str, int]) -> dict[str, Any]:
    """Compute roots and vertex from coefficients, then cross-check the design."""
    params = validate_ai_parameters(level, params)
    if level == "Basic":
        r = params["r"]
        a, b, c = 1, 0, -(r * r)
        expected_roots = (-r, r)
        expected_vertex = (0.0, float(-r * r))
        statement = "Find the two x-intercepts of the parabola (the values of x when y = 0)."
        labels = ("Left root (x₁)", "Right root (x₂)")
        answers = expected_roots
        solution = [
            f"Set y = 0: 0 = x² − {r*r}.",
            f"Add {r*r} to both sides: x² = {r*r}.",
            f"Take the square root: x = −{r} or x = {r}.",
            f"The vertex is (0, −{r*r}).",
        ]
        fallback_hint = "Set y equal to zero, then isolate x² before taking square roots."
        learning = "One-step reasoning with square roots and intercepts"
    elif level == "Standard":
        p, q = params["p"], params["q"]
        a, b, c = 1, -(p + q), p * q
        expected_roots = (p, q)
        expected_vertex = ((p + q) / 2, -((q - p) ** 2) / 4)
        statement = "Find the two roots and the x-coordinate of the vertex."
        labels = ("Smaller root (x₁)", "Larger root (x₂)", "Vertex x-coordinate")
        answers = (p, q, expected_vertex[0])
        solution = [
            f"Factor the quadratic: y = (x − ({p}))(x − ({q})).",
            f"Set each factor to zero: x = {p} or x = {q}.",
            f"The vertex lies halfway between the roots: xᵥ = ({p} + {q}) / 2 = {expected_vertex[0]:g}.",
            f"Evaluate the function at xᵥ: yᵥ = {expected_vertex[1]:g}.",
        ]
        fallback_hint = "Factor the quadratic to find the two roots. The vertex lies halfway between them."
        learning = "Factoring and finding a parabola's vertex"
    else:
        m, t = params["m"], params["T"]
        a, b, c = -m, m * t, 0
        expected_roots = (0, t)
        expected_vertex = (t / 2, m * (t**2) / 4)
        statement = (
            "A ball is launched from ground level. Its height h(t) is measured in metres, "
            "and t is time in seconds. Find when it returns to the ground (after launch) "
            "and its maximum height."
        )
        labels = ("Landing time (seconds)", "Maximum height (metres)")
        answers = (t, expected_vertex[1])
        solution = [
            f"Set the height to zero: 0 = −{m}t² + {m*t}t = t(−{m}t + {m*t}).",
            f"The solutions are t = 0 at launch and t = {t} when the ball lands.",
            f"The maximum occurs at the vertex: t = −b/(2a) = {t/2:g} seconds.",
            f"Substitute this time: maximum height = {expected_vertex[1]:g} metres.",
            f"Only 0 ≤ t ≤ {t} is physically meaningful in this model.",
        ]
        fallback_hint = (
            "Set height to zero to find the landing time. Use the vertex to find the maximum height."
        )
        learning = "Quadratic modelling: flight duration and maximum height"

    # Independently check the roots/vertex from the coefficients.
    discriminant = b * b - 4 * a * c
    if discriminant <= 0:
        raise ValueError("Expected two distinct real roots")
    computed = sorted(((-b - math.sqrt(discriminant)) / (2 * a),
                       (-b + math.sqrt(discriminant)) / (2 * a)))
    vx = -b / (2 * a)
    vy = a * vx * vx + b * vx + c
    if not all(math.isclose(v, w, abs_tol=1e-10) for v, w in zip(computed, expected_roots)):
        raise ValueError("Root verification failed")
    if not all(math.isclose(v, w, abs_tol=1e-10) for v, w in zip((vx, vy), expected_vertex)):
        raise ValueError("Vertex verification failed")
    if not all(math.isclose(a * root**2 + b * root + c, 0, abs_tol=1e-10)
               for root in expected_roots):
        raise ValueError("Equation verification failed")

    variable, function = ("t", "h(t)") if level == "Advanced" else ("x", "y")
    return {
        "level": level,
        "params": params,
        "signature": signature(level, params),
        "a": a, "b": b, "c": c,
        "equation": polynomial_latex(a, b, c, variable, function),
        "roots": expected_roots,
        "vertex": expected_vertex,
        "statement": statement,
        "labels": labels,
        "answers": answers,
        "solution": solution,
        "fallback_hint": fallback_hint,
        "learning": learning,
    }
