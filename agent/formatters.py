from __future__ import annotations
from typing import Dict, Any, List

ROUND_SIG = 3

def _sig(x: float, n: int = ROUND_SIG) -> float:
    if x == 0:
        return 0.0
    from math import log10, floor
    p = -int(floor(log10(abs(x)))) + (n - 1)
    return round(x, p)

def paraphrase_answer(plan_tool: str, result: Dict[str, Any]) -> str:
    if plan_tool == "projectile_horizontal":
        x = result.get("range", {}).get("value")
        t = result.get("time", {}).get("value")
        xu = result.get("range", {}).get("unit")
        tu = result.get("time", {}).get("unit")
        return f"The ball lands about {_sig(x)} {xu} from the building after {_sig(t)} {tu}."
    if plan_tool == "projectile_vertical":
        h = result.get("h_max", {}).get("value")
        hu = result.get("h_max", {}).get("unit")
        return f"The ball reaches a maximum height of approximately {_sig(h)} {hu}."
    if plan_tool in ("quadratic", "poly_roots"):
        roots = result.get("roots", [])
        return f"The roots are {roots}."
    if plan_tool == "sqrt":
        val = result.get("sqrt", {}).get("numeric")
        return f"The square root is approximately {_sig(val)}."
    return "Computation completed."

def steps_from_result(result: Dict[str, Any]) -> List[str]:
    return result.get("steps", [])
