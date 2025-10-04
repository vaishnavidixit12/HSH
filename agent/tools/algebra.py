from __future__ import annotations
from typing import Any, Dict, List
from sympy import symbols, sqrt as sp_sqrt, Poly
from sympy import roots as sp_roots
from sympy import nsimplify

from .units import UnitSession

ROUND = 12

def sqrt_tool(variables: Dict[str, Any], units: str, us: UnitSession):
    n = variables.get("n")
    if n is None:
        n = next((v for k, v in variables.items() if isinstance(v, (int, float))), None)
    x = sp_sqrt(n)
    result = {
        "sqrt": {
            "exact": str(x),
            "numeric": float(x.evalf(ROUND)),
        },
        "steps": [
            f"Compute sqrt({n}) using SymPy.",
        ]
    }
    return {"n": n}, result

def quadratic_tool(variables: Dict[str, Any], units: str, us: UnitSession):
    a = variables.get("a")
    b = variables.get("b")
    c = variables.get("c")
    x = symbols('x')
    poly = a*x**2 + b*x + c
    disc = b**2 - 4*a*c
    r = sp_roots(poly)  # dict {root: multiplicity}
    roots_list = [nsimplify(rr) for rr in r.keys()]
    result = {
        "discriminant": float(disc),
        "roots": [str(v) for v in roots_list],
        "steps": [
            f"Discriminant Δ = b^2 - 4ac = {b}^2 - 4*{a}*{c} = {float(disc)}.",
            "Use quadratic formula: x = (-b ± √Δ) / (2a).",
        ]
    }
    return {"a": a, "b": b, "c": c}, result

def poly_roots_tool(variables: Dict[str, Any], units: str, us: UnitSession):
    coeffs: List[float] = variables.get("coeffs", [])
    x = symbols('x')
    poly = Poly(sum(c*x**(len(coeffs)-i-1) for i, c in enumerate(coeffs)), x)
    r = poly.nroots()
    result = {
        "roots": [complex(rv) for rv in r],
        "steps": [
            f"Find polynomial roots numerically for coefficients {coeffs}.",
        ]
    }
    return {"coeffs": coeffs}, result
