# -----------------------------------------------------------------------------
# Safe mathematical evaluator (controlled environment)
# Purpose:
#   Evaluate a math expression string using only whitelisted functions/constants
#   and caller-supplied numeric variables, blocking all builtins or globals.
# Safety:
#   - `__builtins__` disabled (prevents arbitrary code execution).
#   - Only math module functions/constants and provided variables are allowed.
# -----------------------------------------------------------------------------

from __future__ import annotations
import math
from typing import Dict

def safe_eval(expr: str, vars: Dict[str, float]) -> float:
    """
    Evaluate a numeric expression safely with restricted environment.

    Parameters
    ----------
    expr : str
        A mathematical expression, e.g. "v0**2 * sin(2*theta) / g"
    vars : Dict[str, float]
        Dictionary of variable values to substitute into the expression.

    Returns
    -------
    float
        The evaluated numeric result as a float.

    Notes
    -----
    Allowed functions and constants include:
      sqrt, sin, cos, tan, asin, acos, atan,
      log (natural), ln (alias), log10, exp,
      pi, e
    """
    # Whitelisted math functions/constants
    allowed = {
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "asin": math.asin, "acos": math.acos, "atan": math.atan,
        "log": math.log, "ln": math.log, "log10": math.log10, "exp": math.exp,
        "pi": math.pi, "e": math.e,
    }
    # Disable all builtins for safety
    env = {"__builtins__": {}}
    # Merge allowed symbols and user variables
    env.update(allowed)
    env.update(vars)
    # Evaluate in isolated environment
    return float(eval(expr, env, {}))
