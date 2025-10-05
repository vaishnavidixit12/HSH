# -----------------------------------------------------------------------------
# Formula Selector
# Purpose: Score and choose the most appropriate formula from the catalog
# given a domain, keyword hints, and user-provided variable values.
# -----------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from .catalog import Catalog, CatalogError
from .types import Formula

@dataclass
class ProvidedValue:
    # Normalized input value coming from the parser/LLM or user.
    # One of (symbol|name) may be present; 'symbol' is preferred when resolved.
    symbol: str | None
    name: str | None
    value: float
    unit: str
    src_text_span: str | None = None  # optional provenance for explainability

@dataclass
class Scored:
    # Scoring breakdown for a candidate formula (used for audit/explain).
    formula_id: str
    total: float              # final score used for ranking
    keyword_overlap: int      # count of matching tags with provided keywords
    var_coverage: int         # how many required variables are already provided
    dimensional_ready: bool   # whether all non-produced vars are available/constant
    priority: float           # formula-level prior (0..1 heuristic)

class Selector:
    def __init__(self, catalog: Catalog):
        # Keep a handle to the loaded catalog (formulas + constants).
        self.catalog = catalog

    def _score(self, f: Formula, keywords: List[str], provided: List[ProvidedValue]) -> Scored:
        # ---- Keyword overlap: tags ∩ keywords (case-insensitive)
        kwset = set(k.lower() for k in keywords)
        overlap = sum(1 for t in f.tags if t.lower() in kwset)

        # ---- Variable coverage: how many required inputs we already have
        provided_symbols = {p.symbol for p in provided if p.symbol}
        # Required variables are all variables except the formula's 'rhs' (produced symbol)
        vars_need = set(f.variables.keys()) - {f.rhs}
        var_cov = len(vars_need & (provided_symbols or set()))

        # ---- Ready flag: all inputs known or constant (ignoring the produced symbol)
        ready = all((v.source == "constant" or sym in provided_symbols or sym == f.rhs)
                    for sym, v in f.variables.items())

        # ---- Total score: simple linear combo (tag overlap + 0.5*coverage + prior)
        total = overlap + 0.5*var_cov + f.priority

        return Scored(f.id, total, overlap, var_cov, ready, f.priority)

    def select(self, domain: str, keywords: List[str], provided: List[ProvidedValue],
               prefer_formula_id: str | None = None) -> Tuple[Formula, List[Scored]]:
        # Filter formulas by domain (e.g., "physics" or "math")
        cands = [f for f in self.catalog.formulas if f.topic_path[0] == domain]

        # Score all in-domain candidates
        scored = [self._score(f, keywords, provided) for f in cands]
        scored.sort(key=lambda s: s.total, reverse=True)

        # Hard preference override if caller provides a specific formula id
        if prefer_formula_id:
            chosen = next((f for f in cands if f.id == prefer_formula_id), None)
            if chosen: return chosen, scored

        # If no candidates, the catalog/domain is empty → raise
        if not scored:
            raise CatalogError("No formulas in domain.")

        # Ties at the top are treated as ambiguity requiring more context
        top = [s for s in scored if s.total == scored[0].total]
        if len(top) > 1:
            raise CatalogError("Ambiguous: multiple formulas tie; provide more context.")

        # Return the highest-scoring formula and the full scored list for explainability
        chosen = next(f for f in cands if f.id == scored[0].formula_id)
        return chosen, scored
