# -----------------------------------------------------------------------------
# Units & Conversion Utilities
# Purpose:
#   Canonicalize loose user-provided unit strings (aliases/phrases/patterns)
#   and convert numeric values to/from a simplified SI-like internal system.
# Scope:
#   - Lightweight mapper (not a full dimensional analysis engine).
#   - Handles common HS contexts: m, s, kg, rad/deg, ft, mph/kph/km/h, ft/s, etc.
# Safety:
#   - Raises UnitError on unknown or malformed units.
# -----------------------------------------------------------------------------

# src/sps/units.py
from __future__ import annotations
import math
import re
from typing import Tuple

class UnitError(Exception): pass

# Base canonical units (SI or accepted derived)
# Keys here are *canonical keys*, not raw user strings.
# Values: (canonical_SI_label, scale_to_that_label)
UNIT_TABLE = {
    "dimensionless": ("dimensionless", 1.0),

    # SI base / derived we use
    "m": ("m", 1.0),
    "s": ("s", 1.0),
    "kg": ("kg", 1.0),
    "n": ("N", 1.0),
    "j": ("J", 1.0),
    "w": ("W", 1.0),
    "a": ("A", 1.0),
    "v": ("V", 1.0),
    "ohm": ("ohm", 1.0),

    # compound SI
    "m/s": ("m/s", 1.0),
    "m/s^2": ("m/s^2", 1.0),
    "m^2": ("m^2", 1.0),
    "1/s": ("1/s", 1.0),
    "kg*m/s": ("kg*m/s", 1.0),
    "n*s": ("N*s", 1.0),
    "j/(kg*k)": ("J/(kg*K)", 1.0),

    # angles
    "rad": ("rad", 1.0),
    "deg": ("rad", math.pi/180.0),  # canonicalizes to radians internally

    # common imperial & convenience
    "ft": ("m", 0.3048),
    "mph": ("m/s", 0.44704),
    "kph": ("m/s", 1000.0/3600.0),
    "km/h": ("m/s", 1000.0/3600.0),
    "ft/s": ("m/s", 0.3048),
}

# Phrase/alias canonicalization map → canonical key above
# (Runs prior to UNIT_TABLE lookups and regex patterns.)
ALIASES = {
    # feet
    "foot": "ft", "feet": "ft", "ft.": "ft",

    # time variants
    "sec": "s", "secs": "s", "second": "s", "seconds": "s",
    "hr": "s", "hrs": "s", "hour": "s", "hours": "s",  # stand-alone time words scaled later

    # angles
    "degree": "deg", "degrees": "deg",

    # velocities (phrases → mph/kph/m/s/ft/s)
    "mile per hour": "mph", "miles per hour": "mph", "mi/h": "mph", "mi/hr": "mph",
    "mile/hour": "mph", "miles/hour": "mph", "mph": "mph",
    "kilometer per hour": "kph", "kilometers per hour": "kph", "km/h": "km/h", "kph": "kph",
    "meter per second": "m/s", "meters per second": "m/s", "m per s": "m/s", "m/sec": "m/s",
    "foot per second": "ft/s", "feet per second": "ft/s", "ft per s": "ft/s", "ft/sec": "ft/s",
}

# Regex patterns that we map to a canonical compound key.
# Helps normalize free-form "X/Y" or "X per Y" style inputs.
PATTERNS = [
    # e.g., "miles/hour", "mile/hour", "mi/hr"
    (re.compile(r"^(?:mi(?:le)?s?)\s*/\s*hr$"), "mph"),
    (re.compile(r"^(?:mi(?:le)?s?)\s*/\s*hour[s]?$"), "mph"),
    (re.compile(r"^kilometers?\s*/\s*hour[s]?$"), "kph"),
    (re.compile(r"^meters?\s*/\s*second[s]?$"), "m/s"),
    (re.compile(r"^(?:feet|foot)\s*/\s*second[s]?$"), "ft/s"),
    # "m per s", "ft per s" (with optional plurals/abbrev)
    (re.compile(r"^m(?:eters?)?\s+per\s+s(?:ec(?:ond)?s?)?$"), "m/s"),
    (re.compile(r"^(?:ft|feet|foot)\s+per\s+s(?:ec(?:ond)?s?)?$"), "ft/s"),
    (re.compile(r"^km\s*/\s*h(?:ours?)?$"), "km/h"),
]

def _canonicalize(raw: str) -> str:
    """
    Normalize a user-provided unit string to a canonical key.
    - lowercases
    - collapses whitespace
    - replaces unicode degree symbol with 'deg'
    - maps phrases like "miles per hour" → "mph"
    - maps "feet" → "ft"
    - keeps compound forms like "m/s", "m/s^2"
    Returns a canonical key (possibly new), which is then resolved via UNIT_TABLE.
    """
    if raw is None:
        raise UnitError("Unit is None")
    u = str(raw).strip().lower()
    u = u.replace("°", "deg")
    u = re.sub(r"\s+", " ", u)

    # Fast direct hits
    if u in ALIASES:
        return ALIASES[u]
    if u in UNIT_TABLE:
        return u

    # Try pattern rules (regex-based canonicalization)
    for pat, canon in PATTERNS:
        if pat.match(u):
            return canon

    # Try simple "X per Y" → "x/y" normalization (then re-check alias/table)
    if " per " in u:
        parts = u.split(" per ")
        if len(parts) == 2:
            left = parts[0].strip()
            right = parts[1].strip()
            # normalize sides again through ALIASES
            left = ALIASES.get(left, left)
            right = ALIASES.get(right, right)
            fused = f"{left}/{right}"
            if fused in ALIASES:
                return ALIASES[fused]
            if fused in UNIT_TABLE:
                return fused

    # Tidy composited spaces around slash, then final aliasing pass
    u = u.replace(" / ", "/")
    u = ALIASES.get(u, u)
    return u

def _simple_unit_to_si(value: float, unit: str, expected_dim: str) -> Tuple[float, str]:
    """
    Convert a scalar with unit to the solver's canonical SI-like base.
    - expected_dim is advisory only here (not enforcing dimensions).
    - Returns (value_in_SI, canonical_SI_unit_label)
    """
    if not unit or unit.strip() == "":
        return value, "dimensionless"
    key = _canonicalize(unit)

    # Handle lonely time words like "hour" → seconds (special-case scaling)
    if key in ("hour", "hours", "hr", "hrs"):
        # canonicalize to seconds and scale numeric value accordingly
        key = "s"
        return value * 3600.0, "s"

    if key not in UNIT_TABLE:
        raise UnitError(f"Unknown unit: {unit}")

    si_u, scale = UNIT_TABLE[key]
    return value * scale, si_u

class UnitConverter:
    @staticmethod
    def to_si(value: float, unit: str, expected_dim: str) -> Tuple[float, str]:
        # Public facade for inbound normalization (value, unit) → (value_SI, SI_unit)
        return _simple_unit_to_si(value, unit, expected_dim)

    @staticmethod
    def from_si(value_si: float, display_unit: str | None, si_unit: str) -> Tuple[float, str]:
        """
        Convert an internal SI value to a requested display unit.
        - If display_unit is None/empty, return the SI pair as-is.
        - If the requested unit is unknown or incompatible (different SI base),
          return the SI pair as-is (no conversion thrown).
        """
        if not display_unit:
            return value_si, si_unit
        key = _canonicalize(display_unit)
        if key not in UNIT_TABLE:
            raise UnitError(f"Unknown display unit: {display_unit}")
        display_si, scale = UNIT_TABLE[key]
        if display_si != si_unit:
            # keep simple: if incompatible, return SI (avoid incorrect scaling)
            return value_si, si_unit
        return value_si/scale, key
