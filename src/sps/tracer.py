# -----------------------------------------------------------------------------
# Tracing utility
# Purpose:
#   Lightweight, append-only trace collector to record structured steps
#   (events, decisions, calculations) during solving. Produces a JSON-friendly
#   list suitable for API responses and debugging.
# -----------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class TraceStep:
    # One trace record with a short 'kind' label and free-form structured detail.
    kind: str
    detail: Dict[str, Any]

class Tracer:
    def __init__(self): self._steps: List[TraceStep] = []
    def add(self, kind: str, detail: Dict[str, Any]): self._steps.append(TraceStep(kind, detail))
    def steps(self) -> List[Dict[str, Any]]:
        # Export in plain dict form for easy JSON serialization.
        return [{"kind": s.kind, "detail": s.detail} for s in self._steps]
