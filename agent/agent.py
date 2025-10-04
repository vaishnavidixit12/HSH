from __future__ import annotations
import os
import platform
from typing import Any, Dict
from .models import Plan, ToolCall, VerificationResult, FinalAnswer
from .parsing import plan_with_llm
from .router import select_and_run
from .verify import dimensional_check
from .formatters import paraphrase_answer, steps_from_result
from .tracing import save_trace
from .tools.units import UnitSession

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def solve(user_text: str, preferred_units: str = None) -> Dict[str, Any]:
    unit_system = preferred_units or os.getenv("UNIT_SYSTEM", "SI")
    us = UnitSession(target_system=unit_system)

    plan: Plan = plan_with_llm(user_text, unit_system)
    normalized_inputs, result = select_and_run(plan.tool, plan.variables, plan.unit_system, us)

    dim_ok = True
    v = VerificationResult(
        dimensional_check=dimensional_check(dim_ok),
        residuals={},
        sanity=result.get("checks", {})
    )

    final = FinalAnswer(
        primary=result.get("primary", result | {}),
        alternates=result.get("alternates", []),
        rounding_policy="3 sig figs"
    )

    paraphrase = paraphrase_answer(plan.tool, result)
    steps = steps_from_result(result)

    meta = {
        "version": "1.0.0",
        "model": MODEL,
        "platform": platform.platform(),
    }
    tool_calls = [
        ToolCall(name=plan.tool, args=plan.variables, result=result)
    ]

    trace_path = save_trace(meta, {"user_text": user_text, "preferences": {"unit_system": unit_system}}, plan, {
        "normalized_inputs": normalized_inputs
    }, tool_calls, v, final)

    if v.dimensional_check == "failed" or (False in v.sanity.values() if v.sanity else False):
        return {
            "ok": False,
            "message": "Cannot verify solution with current assumptions.",
            "paraphrase": paraphrase,
            "steps": steps,
            "trace_path": trace_path,
        }

    return {
        "ok": True,
        "paraphrase": paraphrase,
        "steps": steps,
        "trace_path": trace_path,
    }
