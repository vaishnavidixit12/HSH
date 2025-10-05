# -----------------------------------------------------------------------------
# Direct Math Fast-Path Utilities
# Purpose: Provide quick, deterministic handling for common "direct math"
# questions before invoking the heavier catalog/solver pipeline.
# Supported fast paths:
#   1) Quadratic roots extraction/solution (including complex roots)
#   2) Parity checks (even/odd) for integers
#   3) Square roots with guard for negatives
#   4) Safe calculator for simple arithmetic expressions (AST-based sandbox)

# -----------------------------------------------------------------------------

# src/sps/direct_math.py
from __future__ import annotations
import ast
import math
import re
from typing import Any, Dict, Optional, Tuple, List
from sympy import symbols, sympify, Poly

# Regex fragment for a numeric literal: integers, decimals, scientific notation
_NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"

def _mk_step(kind: str, detail: Dict[str, Any]) -> Dict[str, Any]:
    # Trace step helper to keep consistent shape across fast-path returns
    return {"kind": kind, "detail": detail}

# --------------------------
# Safe arithmetic evaluator
# --------------------------
# This section safely evaluates arithmetic using Python's AST, whitelisting
# allowed nodes, operators, and math functions to avoid code execution risks.

# Whitelisted math functions; only these names are callable from expressions
_ALLOWED_FUNCS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "log": math.log,   # natural log
    "ln": math.log,    # alias for natural log
    "log10": math.log10,
    "exp": math.exp,
}
# Whitelisted math constants
_ALLOWED_CONSTS = {
    "pi": math.pi,
    "e": math.e,
}

# Allowed AST operator node types
_ALLOWED_BINOPS = {ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow}
_ALLOWED_UNARYOPS = {ast.UAdd, ast.USub}

def _eval_ast(node: ast.AST) -> float:
    """
    Recursively evaluate a parsed AST expression under a strict whitelist.
    Supports: numbers, + - * / // % **, parentheses, unary +/-,
    function calls to _ALLOWED_FUNCS, and names in _ALLOWED_CONSTS.
    """
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Num):  # py<3.8
        return float(node.n)
    if isinstance(node, ast.Constant):  # py>=3.8
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("Unsupported constant type.")
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _ALLOWED_BINOPS:
            raise ValueError("Unsupported operator.")
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        if isinstance(node.op, ast.Add):   return left + right
        if isinstance(node.op, ast.Sub):   return left - right
        if isinstance(node.op, ast.Mult):  return left * right
        if isinstance(node.op, ast.Div):   return left / right
        if isinstance(node.op, ast.FloorDiv): return left // right
        if isinstance(node.op, ast.Mod):   return left % right
        if isinstance(node.op, ast.Pow):   return left ** right
        raise ValueError("Unsupported BinOp.")
    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in _ALLOWED_UNARYOPS:
            raise ValueError("Unsupported unary operator.")
        val = _eval_ast(node.operand)
        return +val if isinstance(node.op, ast.UAdd) else -val
    if isinstance(node, ast.Call):
        # Only bare function names allowed; no attribute access or lambdas
        if not isinstance(node.func, ast.Name):
            raise ValueError("Unsupported call.")
        name = node.func.id
        if name not in _ALLOWED_FUNCS:
            raise ValueError(f"Unsupported function: {name}")
        if len(node.keywords) != 0:
            raise ValueError("Keywords not allowed.")
        args = [_eval_ast(a) for a in node.args]
        return float(_ALLOWED_FUNCS[name](*args))
    if isinstance(node, ast.Name):
        # Resolve known constants only
        if node.id in _ALLOWED_CONSTS:
            return float(_ALLOWED_CONSTS[node.id])
        raise ValueError(f"Unknown name: {node.id}")
    if isinstance(node, ast.Paren):
        # Rare in practice; included for completeness
        return _eval_ast(node.value)  # rarely produced
    # Anything else is rejected to preserve safety guarantees
    raise ValueError("Unsupported syntax.")

def _safe_calc_expr(expr: str) -> float:
    """
    Safely evaluate a numeric expression using Python AST.
    Normalizations:
      - Commas are stripped for readability (e.g., "1,000" → "1000")
      - Caret (^) is treated as exponent by converting to ** for Python
    """
    expr = expr.strip()
    # mild normalization
    expr = expr.replace(",", "")
    expr = expr.replace("^", "**")
    # parse and eval
    tree = ast.parse(expr, mode="eval")
    return float(_eval_ast(tree))

# ----------------------------------
# Dedicated pattern handlers
# ----------------------------------

def _try_quadratic_roots(question: str) -> Optional[Dict[str, Any]]:
    """
    Extract and solve a quadratic:
      - "roots of 3x^2 + 4x + 5"
      - "roots of 3x^2 + 4x + 5 = 0"
      - "solve 3x^2+4x+5"
      - "zeros of ..."
    Uses sympy to parse, ensures degree 2, then applies quadratic formula.
    Returns complex roots when discriminant < 0 (as Python complex numbers).
    """
    q = (question or "").strip()
    low = q.lower()

    if not any(kw in low for kw in ["roots of", "zeros of", "solve", "root of"]):
        return None

    # Heuristic: capture the polynomial tail after the keyword that still has 'x'
    tail = None
    for kw in ["roots of", "zeros of", "root of", "solve"]:
        i = low.find(kw)
        if i >= 0:
            cand = q[i + len(kw):].strip().rstrip("?").strip()
            if "x" in cand:
                tail = cand
                break
    if tail is None:
        # Fallback: try whole question if we didn't find a clean tail
        tail = q

    # Strip a trailing "= 0" if present
    tail = re.sub(r"\s*=\s*0\s*$", "", tail)

    # Parse expression and coerce to a polynomial in x
    x = symbols("x")
    try:
        expr = sympify(tail)
        # Handle "solve 3x^2+4x+5=0" by isolating lhs
        if getattr(expr, "is_Relational", False) and expr.rhs == 0:
            expr = expr.lhs
        poly = Poly(expr, x)
    except Exception:
        return None

    if poly.degree() != 2:
        return None

    # Extract coefficients a, b, c
    coeffs = poly.all_coeffs()
    # Ensure exactly 3 coefficients
    if len(coeffs) != 3:
        return None
    a, b, c = [float(co) for co in coeffs]
    D = b*b - 4*a*c  # discriminant

    if D >= 0:
        r1 = (-b + math.sqrt(D)) / (2*a)
        r2 = (-b - math.sqrt(D)) / (2*a)
        roots: List[Any] = [r1, r2]
        note = "real"
    else:
        # Complex roots using i*sqrt(|D|)
        sqrtD = complex(0.0, math.sqrt(-D))
        r1 = (-b + sqrtD) / (2*a)
        r2 = (-b - sqrtD) / (2*a)
        roots = [r1, r2]
        note = "complex"

    steps = [
        f"Parsed quadratic: a={a}, b={b}, c={c}",
        f"Discriminant D={D}",
        f"Roots are {note}.",
    ]
    trace = [_mk_step("direct_math", {
        "op": "quadratic_roots",
        "a": a, "b": b, "c": c, "D": D,
        "roots": [str(r) for r in roots]
    })]
    return {
        "name": "Direct Math",
        "eq": "roots = quadratic_roots(a,b,c)",
        "answer": {"symbol": "roots", "value": roots, "unit": ""},
        "trace": trace,
        "steps": steps,
    }

def _try_parity(lower_q: str) -> Optional[Dict[str, Any]]:
    """
    Detect phrases like "is 123 even" or "is -57 odd".
    - Requires a numeric literal per _NUM.
    - Returns an error if the number is not an integer.
    """
    m = re.search(rf"\bis\s+({_NUM})\s+(even|odd)\b", lower_q)
    if not m:
        return None
    n_str, which = m.group(1), m.group(2)
    try:
        n = float(n_str)
    except Exception:
        return None
    if not n.is_integer():
        return {
            "name": "Direct Math",
            "eq": "answer = parity(int(n))",
            "answer": {"symbol": "answer", "value": None, "unit": ""},
            "error": "Parity requested on a non-integer.",
            "trace": [_mk_step("direct_math", {"reason": "non-integer parity", "n": n})],
            "steps": ["Requested odd/even on a non-integer; no result."],
        }
    n_i = int(n)
    is_even = (n_i % 2 == 0)
    val = "even" if is_even else "odd"
    return {
        "name": "Direct Math",
        "eq": "answer = parity(int(n))",
        "answer": {"symbol": "answer", "value": val, "unit": ""},
        "trace": [_mk_step("direct_math", {"op": "parity", "n": n_i, "result": val})],
        "steps": [f"Checked parity of {n_i}."],
    }

def _try_sqrt(lower_q: str) -> Optional[Dict[str, Any]]:
    """
    Detect "sqrt of N" / "square root of N" variants and evaluate.
    Guards against negative inputs (returns error with no real result).
    """
    m = re.search(rf"\b(sqrt|square\s*root)\s*(?:of)?\s*\(?\s*({_NUM})\s*\)?", lower_q)
    if not m:
        return None
    x_str = m.group(2)
    try:
        x = float(x_str)
    except Exception:
        return None
    if x < 0:
        return {
            "name": "Direct Math",
            "eq": "answer = sqrt(x)",
            "answer": {"symbol": "answer", "value": None, "unit": ""},
            "error": "Square root of a negative is not real.",
            "trace": [_mk_step("direct_math", {"op": "sqrt", "x": x, "error": "negative"})],
            "steps": ["Square root of negative number has no real result."],
        }
    ans = math.sqrt(x)
    return {
        "name": "Direct Math",
        "eq": "answer = sqrt(x)",
        "answer": {"symbol": "answer", "value": ans, "unit": ""},
        "trace": [_mk_step("direct_math", {"op": "sqrt", "x": x, "result": ans})],
        "steps": [f"Computed sqrt({x})."],
    }

def _extract_calc_expr(lower_q: str) -> Optional[str]:
    """
    Extract a 'calculator' expression from natural language, or return None.
    Accepted prompts (case-insensitive):
      - "what is 2 + 3 * 4"
      - "compute 10^3 - 5"
      - "evaluate sqrt(144) + 7"
    Also accepts a bare arithmetic line (digits/operators/func letters) if it
    doesn't look like a physics word problem (ban-list guards).
    """
    m = re.search(r"\b(what\s+is|compute|calculate|eval(?:uate)?)\b[:\s]*(.+)", lower_q)
    if m:
        return m.group(2).strip().rstrip("?")
    # Bare expression path: tight allowlist to avoid swallowing word problems
    if re.fullmatch(r"[0-9\.\+\-\*/%\^\(\)\sA-Za-z,]+", lower_q):
        # Avoid hijacking domain questions that contain physics terms
        banned = ["velocity", "projectile", "range", "ohm", "resistor", "force", "work", "energy",
                  "height", "angle", "speed", "acceleration"]
        if any(w in lower_q for w in banned):
            return None
        return lower_q.strip().rstrip("?")
    return None

def try_simple_math(question: str) -> Optional[Dict[str, Any]]:
    """
    Entry point for the direct-math fast path.
    Order of attempts:
      1) Quadratic roots
      2) Parity (even/odd)
      3) Square root
      4) AST-safe calculator
    Returns a standardized result dict on success; otherwise None to let the
    main planner/solver take over.
    """
    q = (question or "").strip()
    if not q:
        return None
    lower = q.lower()

    # 1) Quadratic roots (handles complex)
    qr = _try_quadratic_roots(q)
    if qr is not None:
        return qr

    # 2) Parity checks
    pr = _try_parity(lower)
    if pr is not None:
        return pr

    # 3) Square roots
    sr = _try_sqrt(lower)
    if sr is not None:
        return sr

    # 4) Simple calculator evaluator
    expr = _extract_calc_expr(lower)
    if expr:
        try:
            val = _safe_calc_expr(expr)
        except Exception as e:
            # Surface parse/eval errors transparently for debugging/tracing
            return {
                "name": "Direct Math",
                "eq": f"answer = {expr}",
                "answer": {"symbol": "answer", "value": None, "unit": ""},
                "error": f"Invalid expression: {e}",
                "trace": [_mk_step("direct_math", {"op": "eval_error", "expr": expr, "error": str(e)})],
                "steps": [f"Failed to evaluate expression: {expr}"],
            }
        return {
            "name": "Direct Math",
            "eq": f"answer = {expr}",
            "answer": {"symbol": "answer", "value": val, "unit": ""},
            "trace": [_mk_step("direct_math", {"op": "eval", "expr": expr, "result": val})],
            "steps": [f"Evaluated expression: {expr}"],
        }

    # No direct match; fall back to the main system
    return None
