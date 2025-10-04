from __future__ import annotations
from typing import Any, Dict, Tuple
from .tools.algebra import sqrt_tool, quadratic_tool, poly_roots_tool
from .tools.units import UnitSession
from .tools.physics.kinematics import (
    projectile_horizontal_tool,
    projectile_vertical_tool,
)

REGISTRY = {
    "sqrt": sqrt_tool,
    "quadratic": quadratic_tool,
    "poly_roots": poly_roots_tool,
    "projectile_horizontal": projectile_horizontal_tool,
    "projectile_vertical": projectile_vertical_tool,
}

def select_and_run(tool_name: str, variables: Dict[str, Any], units: str, us: UnitSession) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if tool_name not in REGISTRY:
        raise ValueError(f"Unknown tool: {tool_name}")
    fn = REGISTRY[tool_name]
    normalized_inputs, result = fn(variables, units, us)
    return normalized_inputs, result
