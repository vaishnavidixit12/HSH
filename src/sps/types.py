# -----------------------------------------------------------------------------
# Types module: Shared dataclasses for the Solver ecosystem
# Purpose:
#   Define structured representations for formulas, variables, and constants
#   used across the catalog, selector, planner, and solver.
# -----------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class VariableSpec:
    """
    Metadata for a variable in a formula.
    - unit: canonical SI dimension (e.g., 'm', 's', 'm/s^2')
    - aliases: alternate symbols or names accepted in parsing
    - source: optional label (e.g., 'constant' if derived from catalog constants)
    - key: if source=='constant', lookup key in the constants section
    - normalize: flag for parser normalization or preprocessing
    """
    unit: str
    aliases: List[str] = field(default_factory=list)
    source: str | None = None     # "constant" if constant
    key: str | None = None        # constant key name
    normalize: bool = False

@dataclass
class Formula:
    """
    Represents a single mathematical relation within a topic.
    Example:
        id: "proj_range"
        name: "Projectile range"
        eq: "R = (v0**2 * sin(2*theta)) / g"
        rhs: "R"
    Attributes:
        - variables: mapping from symbol → VariableSpec
        - tags: semantic tags for keyword matching
        - topic_path: [domain, topic] hierarchy
        - rearrangements: whether symbolic rearrangements are allowed
        - priority: heuristic weight for selector scoring
    """
    id: str
    name: str
    eq: str                # 'RHS = <expr>'
    rhs: str               # RHS symbol name
    variables: Dict[str, VariableSpec]
    tags: List[str]
    topic_path: List[str]  # [domain, topic]
    rearrangements: bool = True
    priority: float = 0.5

@dataclass
class Constant:
    """
    Physical or mathematical constant defined in the catalog.
    Example: g = 9.81 m/s^2
    """
    key: str
    value: float
    unit: str
    tags: List[str] = field(default_factory=list)
    notes: str = ""
