# -----------------------------------------------------------------------------
# Planner: Simple BFS-based formula sequencing
# Goal: Given a domain, a target symbol to produce, and a set of known symbols,
#       find a short sequence of formulas (plan) that allows producing the target.
# -----------------------------------------------------------------------------

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
from .types import Formula, VariableSpec
from .catalog import Catalog

@dataclass
class PlanStep:
    # One application of a formula to produce a specific symbol.
    formula: Formula
    target_symbol: str  # which symbol this step will produce

@dataclass
class Plan:
    # Ordered list of plan steps, from first to last.
    steps: List[PlanStep]

class Planner:
    """
    Simple BFS planner:
      - Nodes are sets of 'known' symbols.
      - Using a formula can 'produce' any symbol in that formula (RHS or any via rearrangement).
      - We prefer producing exactly the requested target, but allow intermediate symbols.
    """
    def __init__(self, catalog: Catalog):
        # Keep a reference to the global catalog with all formulas.
        self.catalog = catalog

    def _requirements_for(self, f: Formula, produce_symbol: str) -> Set[str]:
        """
        Compute which input variables are required to use formula `f`
        in order to produce `produce_symbol`.
        - Constants are excluded from requirements.
        - The symbol being produced is not a requirement itself.
        """
        reqs = set()
        for sym, vs in f.variables.items():
            if vs.source == "constant":
                continue
            if sym == produce_symbol:
                continue
            reqs.add(sym)
        return reqs

    def plan(self, domain: str, target: str, known_symbols: Set[str]) -> Optional[Plan]:
        """
        Breadth-first search over sets of known symbols.
        - Start from the initial set of known symbols.
        - At each step, try to apply any formula to produce any symbol in that formula
          (assuming rearrangement is permitted elsewhere in the system).
        - A formula is applicable if all its non-constant, non-produced variables are known.
        - Return the first plan that yields the target symbol (shortest in steps).
        """
        # Narrow formulas to the requested domain (e.g., "physics" or "math").
        formulas = [f for f in self.catalog.formulas if f.topic_path[0] == domain]

        # Each BFS state is (known_symbols_set, plan_so_far)
        start_state = (frozenset(known_symbols), [])
        q = deque([start_state])
        seen = {frozenset(known_symbols)}  # Avoid revisiting identical known-sets
        max_depth = 4  # Prevent explosion; tune as needed

        while q:
            known, steps = q.popleft()

            # Success condition: target symbol is in the known set
            if target in known:
                return Plan(steps)

            # Depth limit guard
            if len(steps) >= max_depth:
                continue

            # Try expanding by producing any variable from any formula in-domain
            for f in formulas:
                produce_candidates = list(f.variables.keys())  # produce any symbol the formula mentions
                for prod in produce_candidates:
                    reqs = self._requirements_for(f, prod)
                    # If all requirements are already known, we can produce `prod`
                    if reqs.issubset(set(known)):
                        new_known = set(known)
                        new_known.add(prod)
                        new_steps = steps + [PlanStep(formula=f, target_symbol=prod)]
                        key = frozenset(new_known)
                        if key not in seen:
                            seen.add(key)
                            q.append((key, new_steps))

        # No plan within depth/space constraints
        return None
