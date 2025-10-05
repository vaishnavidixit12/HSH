# --- High School Helper: Single-Pipe Solver API (FastAPI) ---------------------
# Purpose: Minimal API that (1) parses a natural-language question with an LLM
# into structured variables, then (2) solves it with a catalog-driven solver.
# ------------------------------------------------------------------------------

from __future__ import annotations
import os, json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import re
from typing import Optional, List, Dict, Any
from sps.catalog import Catalog, CatalogError
from sps.solver import Solver

# Load .env for external configuration (API keys, model, catalog path)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
CATALOG_PATH = os.getenv("CATALOG_PATH", "examples/catalog_hs.yaml")
# api/main.py (top helpers; put after imports and before app = FastAPI)


def _extract_json_block(text: str) -> str:
    """
    Helper: Robustly extract a JSON object from an LLM response.
    - Prefers fenced ```json blocks, but falls back to first {...} span.
    - Raises if nothing resembling JSON is found.
    """
    import re, json
    if not text: raise ValueError("Empty completion text.")
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    start = text.find("{"); end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end+1].strip()
    if text.strip().startswith("{") and text.strip().endswith("}"):
        return text.strip()
    raise ValueError("No JSON object found in model output.")

def _missing_var_from_error(err: str) -> Optional[str]:
    """
    Parse solver error messages to detect a missing required variable.
    Expected format: "Missing required variable 'X'..."
    Returns the symbol (e.g., 'v0') or None if not found.
    """
    m = re.search(r"Missing required variable '([A-Za-z_]\w*)'", err)
    return m.group(1) if m else None

def _find_var_aliases(formula_id: Optional[str], symbol: str) -> List[str]:
    """Collect aliases for `symbol` from the formula if given; else from all formulas."""
    aliases: List[str] = []
    if formula_id:
        for f in _catalog.formulas:
            if f.id == formula_id and symbol in f.variables:
                aliases = list(f.variables[symbol].aliases or [])
                break
    if not aliases:
        # fallback: search across catalog
        for f in _catalog.formulas:
            if symbol in f.variables:
                aliases.extend(f.variables[symbol].aliases or [])
        # dedupe
        aliases = sorted(set(aliases))
    return aliases

def _retry_extract_variable(question: str, symbol: str, aliases: List[str]) -> Optional[Dict[str, Any]]:
    """
    Last-mile extraction: if solver fails due to exactly one missing variable,
    ask the LLM to extract ONLY that variable from the original question.
    Expected return: {symbol, value, unit, text_span}. Returns None on failure.
    """
    if not _client:
        return None

    alias_note = ""
    if aliases:
        alias_note = " Treat the following as synonyms for this variable: " + ", ".join(aliases) + "."

    system = (
    "You are an input parser for high school physics/math.\n"
    "Return STRICT JSON with keys ONLY:\n"
    "  domain: 'physics' | 'math' | 'out_of_scope'\n"
    "  paraphrase: short, keyword-only, no prose (e.g., 'projectile horizontal; v0=..; h0=..; find R')\n"
    "  keywords: array of ontology tokens drawn from the catalog (formula ids, variable symbols, and tags). No English phrases.\n"
    "  target_symbol: one symbol to solve for, chosen from the catalog variables (e.g., R, h_top, t, v0, etc.). If not applicable, null.\n"
    "  values: array of objects with fields: symbol or name, value (number), unit (string), text_span (verbatim from question).\n"
    "Do NOT include any commentary.\n"
    "Use units exactly as they appear in the question (e.g., 'miles per hour', 'feet')."
    )

    user = (
        f"Question: {question}\n"
        f"Target variable symbol: {symbol}.\n"
        f"{alias_note}\n"
        "Only extract the numeric value present in the text and its unit string exactly as written."
    )

    try:
        # Chat completion call; expects a STRICT JSON object in the response.
        chat = _client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = chat.choices[0].message.content
        js = _extract_json_block(content)
        data = json.loads(js)
        if data.get("missing"):
            return None
        # basic sanity
        if "symbol" not in data or "value" not in data or "unit" not in data:
            return None
        # normalize the symbol to the target variable requested
        data["symbol"] = symbol
        # keep a consistent key name used by solver
        if "text_span" not in data:
            data["text_span"] = None
        return data
    except Exception:
        # Any parsing/LLM error leads to "no extraction"
        return None

# FastAPI app with two main endpoints: /parse and /solve
app = FastAPI(title="HS Single-Pipe Solver API")

# Initialize catalog + solver from YAML; and OpenAI client (optional if no key)
_catalog = Catalog.from_file(CATALOG_PATH)
_solver = Solver(_catalog)
_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ----------------------------- Schemas ----------------------------------------
class ParseRequest(BaseModel):
    # Natural language question from the user.
    question: str

class Parsed(BaseModel):
    # Output of /parse: structured parse used by /solve.
    domain: str
    paraphrase: str
    keywords: list[str]
    values: list[dict]

class SolveRequest(Parsed):
    # /solve accepts the Parsed payload plus optional hints/overrides.
    display_unit: str | None = None
    candidate_formula_id: str | None = None
    target_symbol: str | None = None
    question: str | None = None

# ----------------------------- Routes -----------------------------------------
@app.get("/health")
def health(): return {"ok": True}

@app.get("/catalog")
def list_catalog():
    """
    Quick introspection: list formulas from the loaded catalog.
    Useful for demo UI to let users see available equations/variables.
    """
    return {"count": len(_catalog.formulas), "items": _catalog.list_formulas()}

@app.post("/parse", response_model=Parsed)
def parse_input(req: ParseRequest):
    """
    LLM-only step: Convert a raw question to a structured Parsed object.
    - Enforces strict JSON format via system prompt.
    - Extracts domain, paraphrase, keywords, and variable values.
    """
    if not _client:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set.")
    system = (
        "You are an input parser for high school physics/math. "
        "Output strict JSON with keys: domain ('physics'|'math'|'out_of_scope'), paraphrase, keywords (list), "
        "values (list of objects with symbol OR name, value (number), unit (string), and the exact text span). "
        "Do not include any commentary."
    )
    user = f"Question: {req.question}"
    try:
        chat = _client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = chat.choices[0].message.content or ""
        text = _extract_json_block(content)
        data = json.loads(text)
        return Parsed(**data)
    except Exception as e:
        # 502: Upstream model error or invalid JSON returned by the LLM.
        raise HTTPException(status_code=502, detail=f"OpenAI call failed: {e}")

@app.post("/solve")
def solve(req: SolveRequest):
    """
    Core solving path:
    1) If out-of-scope → early return (traced).
    2) Attempt solve with provided parse.
    3) If exactly one variable is missing:
       - Try to re-extract that variable with the LLM using aliases.
       - Merge & re-solve; return second attempt with retry metadata.
    """
    # Out-of-scope early return (still traced style)
    if req.domain not in ("physics","math"):
        return {
            "ok": False,
            "error": "Out of scope or invalid domain.",
            "error_kind": "user_input",
            "paraphrase": req.paraphrase,
            "selected_formula": {},
            "substitutions": [],
            "steps": [],
            "answer": None,
            "trace": [{"kind":"error","detail":{"kind":"user_input","message":"invalid domain"}}],
            "candidates": [],
            "chain": [],
            "retry": {"attempted": False}
        }

    # --- First attempt
    res = _solver.solve(
        paraphrase=req.paraphrase,
        domain=req.domain,
        keywords=req.keywords,
        values=req.values,
        display_unit=req.display_unit,
        candidate_formula_id=req.candidate_formula_id,
        target_symbol=req.target_symbol,
        original_question=req.question or req.paraphrase,
    )

    if res.ok:
        # Happy path: return solver artifacts (selected formula, steps, answer, full trace)
        return {
            "ok": res.ok,
            "paraphrase": res.paraphrase,
            "selected_formula": res.selected_formula,
            "substitutions": res.substitutions,
            "steps": res.steps,
            "answer": res.answer,
            "trace": res.full_trace,
            "candidates": res.candidates,
            "chain": res.chain,
            "retry": {"attempted": False}
        }

    # --- Retry path: specifically for "Missing required variable 'X'..."
    missing_sym = _missing_var_from_error(res.error or "")
    if not missing_sym:
        # nothing to retry; return original failure
        return {
            "ok": res.ok,
            "paraphrase": res.paraphrase,
            "selected_formula": res.selected_formula,
            "substitutions": res.substitutions,
            "steps": res.steps,
            "answer": res.answer,
            "trace": res.full_trace,
            "candidates": res.candidates,
            "chain": res.chain,
            "error": res.error,
            "error_kind": res.error_kind,
            "retry": {"attempted": False}
        }

    # Try to discover formula_id from the trace to collect aliases
    formula_id = None
    try:
        for t in res.full_trace:
            if t.get("kind") == "missing_variable":
                formula_id = t.get("detail", {}).get("formula")
                if formula_id:
                    break
    except Exception:
        formula_id = None

    # Resolve variable synonyms to improve targeted extraction prompt
    aliases = _find_var_aliases(formula_id, missing_sym)

    # Ask LLM to extract only this variable
    extracted = _retry_extract_variable(req.question or req.paraphrase, missing_sym, aliases)
    if not extracted:
        # return original failure, but annotate retry attempt
        return {
            "ok": res.ok,
            "paraphrase": res.paraphrase,
            "selected_formula": res.selected_formula,
            "substitutions": res.substitutions,
            "steps": res.steps,
            "answer": res.answer,
            "trace": res.full_trace,
            "candidates": res.candidates,
            "chain": res.chain,
            "error": res.error,
            "error_kind": res.error_kind,
            "retry": {"attempted": True, "success": False, "reason": "extraction_failed"}
        }

    # Merge and re-solve
    merged_values = list(req.values) + [extracted]
    res2 = _solver.solve(
        paraphrase=req.paraphrase,
        domain=req.domain,
        keywords=req.keywords,
        values=merged_values,
        display_unit=req.display_unit,
        candidate_formula_id=req.candidate_formula_id,
        target_symbol=req.target_symbol,
        original_question=req.question or req.paraphrase,
    )

    # Return second attempt (success or failure), with retry annotation
    payload = {
        "ok": res2.ok,
        "paraphrase": res2.paraphrase,
        "selected_formula": res2.selected_formula,
        "substitutions": res2.substitutions,
        "steps": res2.steps,
        "answer": res2.answer,
        "trace": res2.full_trace,
        "candidates": res2.candidates,
        "chain": res2.chain,
        "retry": {
            "attempted": True,
            "success": bool(res2.ok),
            "filled_symbol": missing_sym,
            "aliases_used": aliases,
            "extracted": extracted,
        }
    }
    if not res2.ok:
        payload["error"] = res2.error
        payload["error_kind"] = res2.error_kind
    return payload
