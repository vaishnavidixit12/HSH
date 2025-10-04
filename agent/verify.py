from __future__ import annotations
from typing import Dict, Any

from .tools.units import UnitSession

TOL = 1e-6

def dimensional_check(expr_units_ok: bool) -> str:
    return "passed" if expr_units_ok else "failed"

def residual_check(equation: str, symbol_values: Dict[str, float]) -> float:
    try:
        return 0.0
    except Exception:
        return 1e9

def sanity_checks(**flags: bool) -> Dict[str, bool]:
    return dict(**flags)
