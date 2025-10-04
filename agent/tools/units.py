from __future__ import annotations
from typing import Any, Dict, Tuple
from pint import UnitRegistry

_UR = UnitRegistry(autoconvert_offset_to_baseunit=True)
_Q_ = _UR.Quantity

class UnitSession:
    def __init__(self, target_system: str = "SI"):
        self.ur = _UR
        self.target_system = target_system

    def q(self, value: float, unit: str):
        return _Q_(value, unit)

    def to(self, quantity, unit: str):
        return quantity.to(unit)

    def number(self, quantity) -> float:
        return quantity.magnitude

    def unit(self, quantity) -> str:
        return f"{quantity.units}"

    def g(self):
        return self.q(9.80665, "m/s^2")

    def g_imperial(self):
        return self.q(32.174, "ft/s^2")

def to_si(us: UnitSession, qty, default_unit: str):
    return us.to(qty, default_unit)

def to_imperial(us: UnitSession, qty, default_unit: str):
    return us.to(qty, default_unit)

def convert_scalar(us: UnitSession, value: float, from_unit: str, to_unit: str) -> Tuple[float, str]:
    q = us.q(value, from_unit)
    tgt = us.to(q, to_unit)
    return tgt.magnitude, f"{tgt.units}"
