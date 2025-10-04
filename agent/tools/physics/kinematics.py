from __future__ import annotations
from typing import Any, Dict
from math import sqrt
from ..units import UnitSession, convert_scalar

def _mph_to_fts(us: UnitSession, v_mph: float) -> float:
    v, _ = convert_scalar(us, v_mph, "mile/hour", "ft/s")
    return v

def _mph_to_ms(us: UnitSession, v_mph: float) -> float:
    v, _ = convert_scalar(us, v_mph, "mile/hour", "m/s")
    return v

def projectile_horizontal_tool(variables: Dict[str, Any], units: str, us: UnitSession):
    h = float(variables.get("height"))
    h_unit = variables.get("height_unit", "ft")
    v0 = float(variables.get("v0"))
    v0_unit = variables.get("v0_unit", "mph")
    target_units = units

    if v0_unit == "mph":
        vx_fts = _mph_to_fts(us, v0)
    elif v0_unit in ("ft/s", "fts"):
        vx_fts = v0
    elif v0_unit in ("m/s", "ms"):
        if target_units == "imperial":
            vx_fts, _ = convert_scalar(us, v0, "m/s", "ft/s")
        else:
            vx_fts = v0
    else:
        vx_fts = _mph_to_fts(us, v0)

    if h_unit in ("m", "meter", "meters"):
        if target_units == "imperial":
            h_ft, _ = convert_scalar(us, h, "m", "ft")
            h_m = None
        else:
            h_ft = None
            h_m = h
    else:
        h_ft = h
        if target_units == "SI":
            h_m, _ = convert_scalar(us, h, "ft", "m")
        else:
            h_m = None

    g_fts2 = us.g_imperial()
    g_ms2 = us.g()

    if h_ft is not None:
        t_s = sqrt(2*h_ft / g_fts2.magnitude)
        range_ft = vx_fts * t_s
    else:
        t_s = sqrt(2*h_m / g_ms2.magnitude)
        if v0_unit == "m/s" and target_units == "SI":
            range_m = v0 * t_s
        else:
            v_ms, _ = convert_scalar(us, vx_fts, "ft/s", "m/s")
            range_m = v_ms * t_s

    if target_units == "imperial":
        result = {
            "time": {"value": t_s, "unit": "s"},
            "range": {"value": range_ft, "unit": "ft"},
            "steps": [
                f"Convert {v0} {v0_unit} to {vx_fts:.4f} ft/s.",
                f"Time to fall: t = sqrt(2*h/g) = sqrt(2*{h_ft}/32.174) ≈ {t_s:.4f} s.",
                f"Horizontal range: x = v_x * t = {vx_fts:.4f} * {t_s:.4f} ≈ {range_ft:.4f} ft.",
            ],
            "checks": {"time_nonnegative": t_s >= 0},
        }
        normalized = {
            "height_ft": h_ft if h_ft is not None else None,
            "v0_fts": vx_fts,
            "g_fts2": g_fts2.magnitude,
        }
        return normalized, result
    else:
        result = {
            "time": {"value": t_s, "unit": "s"},
            "range": {"value": range_m, "unit": "m"},
            "steps": [
                "Normalize units; compute time t = sqrt(2*h/g).",
                f"Range x = v_x * t in SI; result ≈ {range_m:.4f} m.",
            ],
            "checks": {"time_nonnegative": t_s >= 0},
        }
        normalized = {
            "height_m": h_m if h_m is not None else None,
            "v0_ms": v0 if v0_unit == "m/s" else None,
            "g_ms2": g_ms2.magnitude,
        }
        return normalized, result

def projectile_vertical_tool(variables: Dict[str, Any], units: str, us: UnitSession):
    v0 = float(variables.get("v0"))
    v0_unit = variables.get("v0_unit", "mph")
    h0 = float(variables.get("h0", 0.0))
    h0_unit = variables.get("h0_unit", "ft")
    target_units = units

    if v0_unit == "mph":
        v_ms = _mph_to_ms(us, v0)
    elif v0_unit in ("m/s", "ms"):
        v_ms = v0
    elif v0_unit in ("ft/s", "fts"):
        v_ms, _ = convert_scalar(us, v0, "ft/s", "m/s")
    else:
        v_ms = _mph_to_ms(us, v0)

    if h0_unit in ("ft", "feet"):
        h0_m, _ = convert_scalar(us, h0, "ft", "m")
    else:
        h0_m = h0

    g = us.g().magnitude
    h_max_m = h0_m + (v_ms**2) / (2*g)

    if target_units == "imperial":
        h_max_ft, _ = convert_scalar(us, h_max_m, "m", "ft")
        steps = [
            f"Convert {v0} {v0_unit} to {v_ms:.4f} m/s.",
            f"Max height: h_max = h0 + v^2/(2g) = {h0_m:.4f} + {v_ms**2:.4f}/(2*{g:.5f})."
        ]
        return (
            {"v0_ms": v_ms, "h0_m": h0_m, "g_ms2": g},
            {"h_max": {"value": h_max_ft, "unit": "ft"}, "steps": steps, "checks": {"height_nonnegative": h_max_m >= 0}}
        )
    else:
        steps = [
            f"Convert {v0} {v0_unit} to {v_ms:.4f} m/s.",
            f"Max height: h_max = h0 + v^2/(2g) = {h0_m:.4f} + {v_ms**2:.4f}/(2*{g:.5f})."
        ]
        return (
            {"v0_ms": v_ms, "h0_m": h0_m, "g_ms2": g},
            {"h_max": {"value": h_max_m, "unit": "m"}, "steps": steps, "checks": {"height_nonnegative": h_max_m >= 0}}
        )
