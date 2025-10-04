from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field

Domain = Literal[
    "math/arithmetic",
    "math/algebra",
    "math/geometry",
    "math/trigonometry",
    "math/calculus",
    "math/probability",
    "math/statistics",
    "physics/kinematics",
    "physics/dynamics",
    "physics/energy",
    "physics/momentum",
    "physics/circular",
    "physics/shm",
    "physics/electricity",
    "physics/optics",
    "physics/fluids",
    "physics/thermo",
]

class Plan(BaseModel):
    domain: Domain
    tool: str
    variables: Dict[str, Any] = Field(default_factory=dict)
    unknowns: List[str] = Field(default_factory=list)
    unit_system: Literal["SI", "imperial"] = "SI"
    notes: Optional[str] = None

class ToolCall(BaseModel):
    name: str
    args: Dict[str, Any]
    result: Any | None = None

class VerificationResult(BaseModel):
    dimensional_check: Literal["passed", "failed"]
    residuals: Dict[str, float] = Field(default_factory=dict)
    sanity: Dict[str, bool] = Field(default_factory=dict)
    reason: Optional[str] = None

class FinalAnswer(BaseModel):
    primary: Dict[str, Any]
    alternates: List[Dict[str, Any]] = Field(default_factory=list)
    rounding_policy: str = "3 sig figs"

class Trace(BaseModel):
    meta: Dict[str, Any]
    input: Dict[str, Any]
    parse_intent: Plan
    unit_normalize: Dict[str, Any]
    tool_calls: List[ToolCall]
    verification: VerificationResult
    final: FinalAnswer
