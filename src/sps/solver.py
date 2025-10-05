# -----------------------------------------------------------------------------
# Solver: End-to-end numeric pipeline for HS Helper
# Responsibilities:
#   • Normalize/ingest parsed inputs (values/keywords/paraphrase/domain)
#   • Optional direct-math fast path (sqrt, parity, arithmetic, quadratics)
#   • Formula selection (lightweight ranking for context/explainability)
#   • Target inference via reachability planning when target_symbol is absent
#   • Unit-safe numeric evaluation with constants + user values
#   • Symbolic rearrangement when the desired output isn't the formula's RHS
#   • Result formatting (unit conversion to display_unit), tracing, and errors
# -----------------------------------------------------------------------------

# src/sps/solver.py
from __future__ import annotations
import re, math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from sympy import symbols, sympify, Eq, solve, sqrt, sin, cos, tan, asin, acos, atan, pi as SPI, N

from .catalog import Catalog, CatalogError
from .units import UnitConverter, UnitError
from .selector import Selector, ProvidedValue
from .safe_eval import safe_eval
from .tracer import Tracer
from .planner import Planner
from .direct_math import try_simple_math


@dataclass
class SolverResult:
    # Structured response used by the API layer
    ok: bool
    paraphrase: str
    selected_formula: Dict[str, Any]
    substitutions: List[Dict[str, Any]]
    steps: List[str]
    answer: Dict[str, Any] | None
    full_trace: List[Dict[str, Any]]
    candidates: List[Dict[str, Any]]
    chain: List[Dict[str, Any]]
    error: str | None = None
    error_kind: str | None = None  # "user_input" | "agent"


class Solver:
    def __init__(self, catalog: Catalog):
        # Inject catalog and compose core subsystems
        self.catalog = catalog
        self.selector = Selector(catalog)  # ranks formulas (for explainability + fallback)
        self.planner = Planner(catalog)    # BFS reachability over symbols

    # ---------------- internal helpers ----------------

    def _sym_solve_for(self, eq_str: str, rhs_symbol: str, target_symbol: str) -> str:
        """Symbolically rearrange eq_str to solve target_symbol = ...
        - Parses 'RHS = <expr>' into a sympy Eq
        - Solves for the desired target symbol
        - Returns a string 'target = <sympy_expr>'
        """
        m = re.match(r"\s*([A-Za-z_]\w*)\s*=\s*(.+)$", eq_str)
        if not m:
            raise CatalogError("Eq format error; expected 'RHS = <expr>'")
        lhs, expr = m.group(1), m.group(2)
        local = {"sqrt": sqrt, "sin": sin, "cos": cos, "tan": tan, "asin": asin, "acos": acos, "atan": atan, "pi": SPI}
        rhs_sym = symbols(lhs)
        tgt_sym = symbols(target_symbol)
        expr_sym = sympify(expr, locals=local)
        equation = Eq(rhs_sym, expr_sym)
        sol = solve(equation, tgt_sym, dict=True)
        if not sol:
            raise CatalogError(f"Could not solve symbolically for {target_symbol}")
        tgt_expr = sol[0][tgt_sym]
        return f"{target_symbol} = {str(tgt_expr)}"

    @staticmethod
    def _normsym(s: Optional[str]) -> Optional[str]:
        # Minimal normalization: remove underscores/spaces (e.g., 'h_0' → 'h0')
        if s is None:
            return None
        return s.replace("_", "").replace(" ", "")

    def _canonical_hint(self, sym_norm: str) -> Optional[str]:
        """
        Best-effort canonicalization to the ontology for planning only (not for value lookup).
        Rules are intentionally minimal & deterministic (no natural language):
          - v0 stays v0; v → v0 (common initial speed symbol in kinematics)
          - any h* containing '0' or 'init' → h0
          - h initial variants: hinitial, hinit, hstart → h0
        """
        s = sym_norm.lower()
        if s == "v0":
            return "v0"
        if s == "v":
            return "v0"
        if s.startswith("h"):
            if "0" in s or "init" in s or s in ("hinitial", "hinit", "hstart"):
                return "h0"
        return None

    def _build_known_symbols(self, provided: List[ProvidedValue], constants: Dict[str, Any]) -> set[str]:
        """
        Build the set of symbols known to the planner. Includes:
          - raw provided symbols (normalized)
          - catalog constants
          - canonical hints (e.g., v -> v0, hinitial -> h0) to improve reachability
        """
        known_raw = {(p.symbol or "").strip() for p in provided if p.symbol}
        known_raw = {s for s in known_raw if s}
        # add constants
        known = set(known_raw) | set(constants.keys())

        # augment with canonical hints for planning
        aug = set()
        for s in known_raw:
            s_norm = self._normsym(s)  # already normalized earlier, but safe
            hint = self._canonical_hint(s_norm)
            if hint:
                aug.add(hint)

        return known | aug

    def _const_si_map_for_formula(self, formula) -> Dict[str, float]:
        """
        Build a substitution map for constants used by a given formula
        (converted into the formula's declared SI units).
        """
        subs: Dict[str, float] = {}
        for sym, vs in formula.variables.items():
            if vs.source == "constant":
                c = self.catalog.constants.get(vs.key or "")
                if c is None:
                    raise CatalogError(f"Missing constant: {vs.key}")
                v_si, _ = UnitConverter.to_si(c.value, c.unit, expected_dim=vs.unit)
                subs[sym] = v_si
        return subs

    def _value_lookup(self, sym: str, formula, provided: List[ProvidedValue]) -> Optional[Tuple[float, str]]:
        """
        Find provided value for symbol `sym`.
        Tolerant to aliases and underscore/space differences (e.g., h_0 vs h0).
        Also accepts 'v' as 'v0' for simple vertical throws.
        Returns (value_in_SI, si_unit_str) or None if not found.
        """
        def _norm(s: str | None) -> str:
            return (s or "").strip().lower().replace("_", "").replace(" ", "")

        vs = formula.variables[sym]
        candidates = [sym] + list(vs.aliases or [])
        cand_norm = {_norm(c): c for c in candidates}

        by_sym_norm = {_norm(p.symbol): p for p in provided if p.symbol}
        by_name_norm = {_norm(p.name): p for p in provided if p.name}

        # try symbol matches first
        for key in cand_norm.keys():
            if key in by_sym_norm:
                p = by_sym_norm[key]
                v_si, u_si = UnitConverter.to_si(p.value, p.unit, expected_dim=vs.unit)
                return v_si, u_si

        # then names (LLM sometimes puts value under 'name')
        for key in cand_norm.keys():
            if key in by_name_norm:
                p = by_name_norm[key]
                v_si, u_si = UnitConverter.to_si(p.value, p.unit, expected_dim=vs.unit)
                return v_si, u_si

        # last-chance: 'v' as 'v0'
        if sym == "v0":
            p = by_sym_norm.get(_norm("v")) or by_name_norm.get(_norm("v"))
            if p:
                v_si, u_si = UnitConverter.to_si(p.value, p.unit, expected_dim=vs.unit)
                return v_si, u_si

        return None

    def _eval_step(
        self,
        step_formula,
        produce_symbol: str,
        provided: List[ProvidedValue],
        trace: Tracer
    ) -> Tuple[float, str, List[Dict[str, Any]], List[str]]:
        """
        Execute a single formula application:
          - Gather constants (SI), then gather required provided inputs (SI)
          - Either evaluate RHS directly, or symbolically rearrange to the target
          - Return (result_value_SI, unit_SI, substitution_rows, step_messages)
        """
        rows: List[Dict[str, Any]] = []
        steps: List[str] = []
        subs: Dict[str, float] = self._const_si_map_for_formula(step_formula)

        # log constants used
        for sym, vs in step_formula.variables.items():
            if vs.source == "constant":
                c = self.catalog.constants[vs.key]
                rows.append({
                    "symbol": sym,
                    "value_original": c.value,
                    "unit_original": c.unit,
                    "value_si": subs[sym],
                    "unit_si": vs.unit,
                    "source": f"constant:{vs.key}",
                })

        # collect provided inputs except target
        for sym, vs in step_formula.variables.items():
            if vs.source == "constant" or sym == produce_symbol:
                continue
            if sym in subs:
                continue
            got = self._value_lookup(sym, step_formula, provided)
            if got is None:
                trace.add("missing_variable", {"formula": step_formula.id, "symbol": sym})
                raise CatalogError(f"Missing required variable '{sym}' for formula {step_formula.id}")
            v_si, u_si = got
            subs[sym] = v_si
            rows.append({
                "symbol": sym,
                "value_original": None,
                "unit_original": None,
                "value_si": v_si,
                "unit_si": u_si,
                "source": "provided/previous",
            })

        # compute
        if produce_symbol == step_formula.rhs:
            # Direct RHS numeric evaluation using safe_eval
            m = re.match(r"\s*([A-Za-z_]\w*)\s*=\s*(.+)$", step_formula.eq)
            if not m:
                raise CatalogError("Eq format error")
            rhs_expr = m.group(2)
            pretty = rhs_expr
            # Pretty-print a substituted expression for trace/steps (non-executed)
            for k, v in sorted(subs.items(), key=lambda kv: -len(kv[0])):
                pretty = re.sub(rf"\b{k}\b", f"({v})", pretty)
            result_si = safe_eval(rhs_expr, {**subs, "pi": math.pi})
            unit_si = step_formula.variables[produce_symbol].unit
            trace.add("numeric_eval", {
                "formula": step_formula.id,
                "target": produce_symbol,
                "expr": rhs_expr,
                "expr_subs": pretty,
                "result_si": result_si,
                "unit": unit_si
            })
            steps.append(f"{step_formula.id}: computed {produce_symbol} from RHS.")
            return result_si, unit_si, rows, steps
        else:
            # Rearrangement path: derive target_expr via sympy then evaluate numerically
            eq_s = self._sym_solve_for(step_formula.eq, step_formula.rhs, produce_symbol)
            m = re.match(rf"\s*{re.escape(produce_symbol)}\s*=\s*(.+)$", eq_s)
            if not m:
                raise CatalogError("Rearrangement parse error.")
            target_expr = m.group(1)
            expr_sym = sympify(target_expr, locals={
                "sqrt": sqrt, "sin": sin, "cos": cos, "tan": tan,
                "asin": asin, "acos": acos, "atan": atan, "pi": SPI
            })
            sub_map = {symbols(k): float(v) for k, v in subs.items()}
            val_num = N(expr_sym.subs(sub_map))
            result_si = float(val_num)
            unit_si = step_formula.variables[produce_symbol].unit
            trace.add("numeric_eval", {
                "formula": step_formula.id,
                "target": produce_symbol,
                "expr": target_expr,
                "result_si": result_si,
                "unit": unit_si
            })
            steps.append(f"{step_formula.id}: rearranged to solve {produce_symbol}.")
            return result_si, unit_si, rows, steps

    def _classify_error(self, e: Exception) -> str:
        """
        Map raw exceptions to a coarse error_kind for the API:
          - 'user_input': missing vars, ties, no plan, OOS, etc.
          - 'agent': unit resolution issues, constants missing, eq format errors, etc.
        """
        msg = str(e).lower()
        if any(s in msg for s in [
            "missing required variable",
            "ambiguous: multiple formulas tie",
            "cannot infer target",
            "planner could not find a chain",
            "out of scope",
            "no formulas in domain",
        ]):
            return "user_input"
        if isinstance(e, UnitError) or "missing constant" in msg or "eq format error" in msg:
            return "agent"
        return "agent"

    def _symbol_in_keywords(self, sym: str, keywords: List[str]) -> bool:
        # Lightweight ontology hit: exact symbol token appears in keywords
        ks = {(k or "").lower() for k in keywords}
        return sym.lower() in ks

    def _alias_hit(self, sym: str, keywords: List[str]) -> bool:
        # Alias hit: any alias for `sym` appears in keywords (cross-catalog search)
        ks = {(k or "").lower() for k in keywords}
        for f in self.catalog.formulas:
            if sym in f.variables:
                aliases = (f.variables[sym].aliases or [])
                for a in aliases:
                    if (a or "").lower() in ks:
                        return True
        return False

    def _score_target(self, target: str, plan_steps: int, keywords: List[str]) -> float:
        # Target ranking heuristic:
        #   - Prefer fewer steps (shorter plans)
        #   - Bonus for ontology hits (symbol in keywords/aliases)
        score = 0.0
        score += 2.0 / (1.0 + plan_steps)  # 0 steps → 2.0; 1 step → ~1.0
        if self._symbol_in_keywords(target, keywords):
            score += 1.0
        if self._alias_hit(target, keywords):
            score += 0.5
        return score

    # ---------------- main entry ----------------

    def solve(
        self,
        paraphrase: str,
        domain: str,
        keywords: List[str],
        values: List[Dict[str, Any]],
        display_unit: Optional[str] = None,
        candidate_formula_id: Optional[str] = None,
        target_symbol: Optional[str] = None,
        original_question: Optional[str] = None,
    ) -> SolverResult:
        """
        Main orchestration:
          0) Try direct-math handlers first.
          1) Do formula selection for context (non-binding).
          2) Build known symbol set and infer a target via planning if needed.
          3) Trivial-known fast path: if the target is already provided, convert & return.
          4) Plan a chain; if no plan but we have a chosen candidate, fallback to 1-step plan.
          5) Execute chain with unit-correct substitutions; record trace/steps.
          6) Convert result to display_unit and return a rich, explainable payload.
        """
        trace = Tracer()
        candidates: List[Dict[str, Any]] = []
        chain_view: List[Dict[str, Any]] = []
        substitutions_all: List[Dict[str, Any]] = []
        steps_view: List[str] = []
        selected_json: Dict[str, Any] = {}
        answer: Dict[str, Any] | None = None
        direct = try_simple_math(original_question or paraphrase)

        # normalize incoming symbols (e.g., v_0 -> v0, h_0/h_initial -> h0 if possible)
        provided = [
            ProvidedValue(
                symbol=self._normsym(v.get("symbol")),
                name=self._normsym(v.get("name")),
                value=float(v["value"]),
                unit=v["unit"],
                src_text_span=v.get("text_span") or v.get("src_text_span"),
            )
            for v in values
        ]

        # Initial trace of raw inputs for reproducibility/debugging
        trace.add("inputs_raw", {
            "paraphrase": paraphrase,
            "domain": domain,
            "keywords": keywords,
            "values": values,
            "display_unit": display_unit,
            "candidate_formula_id": candidate_formula_id,
            "target_symbol": target_symbol,
            "question": original_question or paraphrase,
        })

        try:
            # 0) direct math (sqrt, arithmetic, etc.)
            if direct is not None:
                for s in direct.get("trace", []):
                    trace.add(s["kind"], s["detail"])
                if "error" in direct:
                    # direct path produced a user-visible error (e.g., sqrt of negative)
                    err = direct["error"]
                    trace.add("direct_math_error", {"message": err})
                    return SolverResult(
                        ok=False,
                        paraphrase=paraphrase,
                        selected_formula={"id": "direct_math", "name": direct["name"], "eq": direct["eq"], "rhs": "answer",
                                          "tags": ["direct_math"], "topic_path": [domain, "direct"]},
                        substitutions=[],
                        steps=direct.get("steps", []),
                        answer=None,
                        full_trace=trace.steps(),
                        candidates=[],
                        chain=[{"id": "direct_math", "target": "answer"}],
                        error=err,
                        error_kind="user_input",
                    )
                # Successful direct result
                return SolverResult(
                    ok=True,
                    paraphrase=paraphrase,
                    selected_formula={"id": "direct_math", "name": direct["name"], "eq": direct["eq"], "rhs": "answer",
                                      "tags": ["direct_math"], "topic_path": [domain, "direct"]},
                    substitutions=[],
                    steps=direct.get("steps", []),
                    answer=direct["answer"],
                    full_trace=trace.steps(),
                    candidates=[],
                    chain=[{"id": "direct_math", "target": "answer"}],
                )

            # 1) selection (gives candidates list only; we won't trust its RHS blindly)
            chosen = None
            if not target_symbol:
                try:
                    chosen, scored = self.selector.select(domain, keywords, provided, candidate_formula_id)
                    candidates = [{
                        "formula_id": s.formula_id, "score_total": s.total,
                        "keyword_overlap": s.keyword_overlap, "var_coverage": s.var_coverage,
                        "priority": s.priority,
                    } for s in scored[:5]]
                    trace.add("formula_selection", {"chosen": chosen.id if chosen else None, "top": candidates})
                except Exception as e_sel:
                    # Selection is non-fatal; planning may still succeed without it
                    trace.add("selection_error", {"message": str(e_sel)})

            # 2) reachability-based target inference if parser didn't give one
            known_symbols = self._build_known_symbols(provided, self.catalog.constants)
            trace.add("planner_knowns", {"symbols": sorted(known_symbols)})

            if not target_symbol:
                # Consider RHS symbols present in-domain; skip ones already known (unless user implied)
                rhs_all = {f.rhs for f in self.catalog.formulas if f.topic_path[0] == domain}
                # Skip targets that are already known, unless the user explicitly mentions them in keywords
                rhs_options = []
                skipped_known = []
                kw_lower = { (k or "").lower() for k in keywords }
                for t in sorted(rhs_all):
                    if t in known_symbols and (t.lower() not in kw_lower):
                        skipped_known.append(t)
                        continue
                    rhs_options.append(t)
                if skipped_known:
                    trace.add("skipped_known_targets", {"symbols": skipped_known})

                # Score reachable targets via planning + symbol/alias hints
                scored_targets: List[Tuple[str, float, int]] = []
                for t in rhs_options:
                    plan = self.planner.plan(domain=domain, target=t, known_symbols=known_symbols)
                    if plan is None:
                        continue
                    steps_count = len(getattr(plan, "steps", []))
                    scored_targets.append((t, self._score_target(t, steps_count, keywords), steps_count))

                trace.add("target_candidates", [
                    {"symbol": t, "score": sc, "steps": st}
                    for (t, sc, st) in sorted(scored_targets, key=lambda x: (-x[1], x[2], x[0]))[:10]
                ])

                if scored_targets:
                    # Take best by (score desc, fewer steps asc)
                    target_symbol = max(scored_targets, key=lambda x: (x[1], -x[2]))[0]
                    trace.add("target_selected", {"target": target_symbol})
                else:
                    # Fallback: if we have a chosen formula, use its RHS as target
                    if chosen:
                        target_symbol = chosen.rhs
                        trace.add("target_fallback_top_candidate_rhs", {"target": target_symbol})
                    else:
                        raise CatalogError("Planner could not find any reachable target from known inputs. Provide target_symbol or more values.")

            # 3) trivial-known fast path (if the target was provided directly)
            provided_by_sym = {(p.symbol or "").strip(): p for p in provided if p.symbol}
            if target_symbol in provided_by_sym:
                p = provided_by_sym[target_symbol]
                # try to get expected unit from any formula containing target_symbol
                expected_unit = None
                for f in self.catalog.formulas:
                    if target_symbol in f.variables:
                        expected_unit = f.variables[target_symbol].unit
                        break
                v_si, u_si = UnitConverter.to_si(p.value, p.unit, expected_dim=expected_unit or "dimensionless")
                val, unit = UnitConverter.from_si(v_si, display_unit, u_si)
                trace.add("trivial_known_return", {"symbol": target_symbol})
                return SolverResult(
                    ok=True,
                    paraphrase=paraphrase,
                    selected_formula={},
                    substitutions=[],
                    steps=["Returned known variable."],
                    answer={"symbol": target_symbol, "value": val, "unit": unit},
                    full_trace=trace.steps(),
                    candidates=candidates,
                    chain=[],
                )

            # 4) plan a chain for the chosen target
            plan = self.planner.plan(domain=domain, target=target_symbol, known_symbols=known_symbols)
            if plan is None:
                if chosen:
                    # Fallback: wrap the chosen formula as a single-step "plan"
                    plan = type("Plan", (), {"steps": [type("PS", (), {"formula": chosen, "target_symbol": target_symbol})]})
                    trace.add("planner_fallback_single_step", {"formula": chosen.id, "target": target_symbol})
                else:
                    raise CatalogError(f"Planner could not find a chain to produce '{target_symbol}' from known inputs.")

            # 5) execute chain
            produced: Dict[str, Tuple[float, str]] = {}

            def inject_produced(sym: str, val: float, unit: str):
                # Make newly produced symbols available to later steps
                provided.append(ProvidedValue(symbol=sym, name=None, value=val, unit=unit))

            for st in plan.steps:
                # Record human-readable chain for the response
                chain_view.append({
                    "id": st.formula.id,
                    "name": st.formula.name,
                    "eq": st.formula.eq,
                    "target": st.target_symbol
                })
                # Perform numeric evaluation for this step (may rearrange)
                val_si, unit_si, rows, step_msgs = self._eval_step(st.formula, st.target_symbol, provided, trace)
                substitutions_all.extend(rows)
                steps_view.extend(step_msgs)
                produced[st.target_symbol] = (val_si, unit_si)
                inject_produced(st.target_symbol, val_si, unit_si)

            if target_symbol not in produced:
                # Defensive guard: planner said reachable, but execution missed target
                raise CatalogError(f"Execution finished but target '{target_symbol}' was not produced.")

            # 6) format result (display unit conversion + selected formula metadata)
            val_si, unit_si = produced[target_symbol]
            out_value, out_unit = UnitConverter.from_si(val_si, display_unit, unit_si)
            selected = plan.steps[-1].formula if plan.steps else (chosen or None)
            if selected:
                selected_json = {
                    "id": selected.id,
                    "name": selected.name,
                    "eq": selected.eq,
                    "rhs": selected.rhs,
                    "tags": selected.tags,
                    "topic_path": selected.topic_path,
                }

            answer = {"symbol": target_symbol, "value": out_value, "unit": out_unit}
            return SolverResult(
                ok=True,
                paraphrase=paraphrase,
                selected_formula=selected_json,
                substitutions=substitutions_all,
                steps=steps_view,
                answer=answer,
                full_trace=trace.steps(),
                candidates=candidates,
                chain=chain_view,
            )

        except Exception as e:
            # Uniform error funnel with typed error_kind and full trace
            kind = self._classify_error(e)
            trace.add("error", {"kind": kind, "message": str(e)})
            return SolverResult(
                ok=False,
                paraphrase=paraphrase,
                selected_formula=selected_json,
                substitutions=substitutions_all,
                steps=steps_view,
                answer=None,
                full_trace=trace.steps(),
                candidates=candidates,
                chain=chain_view,
                error=str(e),
                error_kind=kind,
            )
