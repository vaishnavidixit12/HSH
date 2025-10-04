from __future__ import annotations
import os
import json
import re
from typing import Dict, Any
from pydantic import ValidationError
from openai import OpenAI
from .models import Plan

SYSTEM_PROMPT = (
    "You are a rigorous planner for a math/physics agent. "
    "Extract domain, choose ONE tool from the provided catalog, "
    "normalize variables (with units indicated), and list unknowns. "
    "Never perform calculations yourself; the Python tools will do that. "
    "Return ONLY valid JSON per the schema."
)

TOOL_CATALOG = [
    # Math
    "sqrt", "quadratic", "poly_roots", "linear_system", "area", "volume",
    "right_triangle", "law_of_cosines", "differentiate", "integrate_basic",
    "nCr", "binom_pmf", "z_score",
    # Physics
    "projectile_horizontal", "projectile_vertical", "time_of_flight",
    "newton2", "friction", "energy_mech", "power", "collision_1d",
    "centripetal", "orbital_velocity", "spring_k", "wave_relation",
    "ohms_law", "series_resistance", "parallel_resistance",
    "thin_lens", "pressure", "buoyancy", "ideal_gas"
]

CATALOG_DESC = (
    "Available tools: " + ", ".join(TOOL_CATALOG) + ". "
    "Choose the single best entry-point tool; solvers may orchestrate internally."
)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_client: OpenAI | None = None

def _client_instance() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client

def plan_with_llm(user_text: str, unit_system: str = "SI") -> Plan:
    client = _client_instance()
    prompt = (
        f"{SYSTEM_PROMPT}\n{CATALOG_DESC}\n\n"
        f"User question: {user_text}\n"
        f"Preferred units: {unit_system}\n\n"
        "Schema keys: {domain, tool, variables, unknowns, unit_system}."
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    content = resp.choices[0].message.content
    data: Dict[str, Any] = json.loads(content)
    if "unit_system" not in data:
        data["unit_system"] = unit_system
    try:
        return Plan(**data)
    except ValidationError as e:
        text = user_text.lower()
        tool = "quadratic" if re.search(r"x\^2|quadratic|ax\^2", text) else (
            "projectile_horizontal" if ("throw" in text and "window" in text) else (
            "projectile_vertical" if ("throw" in text and "vert" in text) else (
            "sqrt" if "square root" in text or "sqrt" in text else "poly_roots")))
        return Plan(domain="math/algebra" if tool in ("sqrt","quadratic","poly_roots") else "physics/kinematics",
                    tool=tool, variables={}, unknowns=[], unit_system=unit_system, notes=f"fallback due to parse error: {e}")
