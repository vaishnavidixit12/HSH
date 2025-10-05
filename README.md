---

# High School Helper

*(For internal review: Vaishnavi A. Dixit & Cailum Blues Reviewing Facility)*

---

## 1. Concept Overview

High School Helper is an AI-driven solver for high-school math and physics.
It pairs a small LLM for input understanding with a deterministic Python engine
for units, formulas, symbolic rearrangement, and numeric evaluation.

It records a trace showing the exact steps the system took,
so reviewers can verify tool use and correctness.

---

## 2. Design Philosophy

* Initially explored orchestration frameworks (e.g., LangChain with local inference).
* Found that too many handlers and chains diluted control over reasoning.
* Adopted a **single-pipeline** approach instead:

```
GPT-4o-mini → JSON parsing → Deterministic Solver → Trace Output
```

* GPT-4o-mini handles only structured extraction.
* Solver performs all symbolic + numeric computation deterministically.
* The separation keeps behavior predictable and debugging transparent.

---

## 3. System Architecture (Modules Overview)

| Module          | Path                                             | Responsibility                                                                                                |
| --------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| **API**         | `api/main.py`                                    | `/parse` for LLM extraction, `/solve` for orchestration and retry on missing variables                        |
| **Solver**      | `src/sps/solver.py`                              | Core orchestrator; calls planner, selector, units, tracer; does symbolic rearrangement and numeric evaluation |
| **Catalog**     | `src/sps/catalog.py`, `examples/catalog_hs.yaml` | Stores constants, formulas, variable metadata, and aliases                                                    |
| **Units**       | `src/sps/units.py`                               | Canonicalizes input units such as "miles per hour" → `m/s`, "ft" → `m`                                        |
| **Planner**     | `src/sps/planner.py`                             | Determines dependency chains and target reachability                                                          |
| **Selector**    | `src/sps/selector.py`                            | Scores candidate formulas by keyword overlap and variable coverage                                            |
| **Direct Math** | `src/sps/direct_math.py`                         | Fast path for arithmetic, square roots, parity, and quadratic roots (real/complex)                            |
| **Tracer**      | `src/sps/tracer.py`                              | Structured logging for auditability and grading                                                               |

---

## 4. External Workflow (User → Answer)

1. User enters a natural-language question.
2. `/parse` calls GPT-4o-mini → returns strict JSON:

   ```json
   { "domain": "physics", "paraphrase": "...", "keywords": [...], "values": [...] }
   ```
3. `/solve`:

   * Try direct math first (e.g., sqrt, quadratic roots).
   * If not applicable, run full formula path (selector → planner → solver).
   * If a required variable is missing, re-prompt GPT-4o-mini for that symbol and re-solve.
4. Returns the answer, selected formula, substitutions, steps, and trace.

---

## 4.1 Internal Execution Flow (Files and Functions)

### A. Parsing (`api/main.py`)

* `parse_input()` → sends question to GPT-4o-mini.
* GPT returns domain, paraphrase, and extracted numeric values.
* The validated JSON feeds into `/solve`.

### B. Solving (`src/sps/solver.py`)

* Direct-math shortcut: `direct_math.try_simple_math()`

  * Handles sqrt, parity, arithmetic, quadratic equations.
* Formula pipeline:

  1. `selector.select()` → candidate formulas
  2. Build knowns (provided + constants)
  3. `planner.plan()` → infer target variable
  4. `_eval_step()` → evaluate formula symbolically or numerically
  5. `UnitConverter` converts to/from SI units

### C. Errors and Recovery

* Missing variable → `_retry_extract_variable()` re-asks GPT-4o-mini.
* Unknown units → structured error (not crash).

### D. Output

Returns:

```
{ paraphrase, selected_formula, substitutions, steps, answer, trace, candidates }
```

---

## 5. Edge Cases (Formal Handling)

| Case               | Behavior                                                                   |
| ------------------ | -------------------------------------------------------------------------- |
| Arithmetic-only    | Solved directly by `direct_math.py`, supports complex roots                |
| Unit variations    | Canonicalized before computation (`feet`, `ft`, `mi/hour` → standard form) |
| Missing inputs     | Automatically re-queried once for the missing symbol                       |
| Ambiguous formulas | Planner picks shortest valid path; ambiguity surfaced in trace             |
| Out-of-catalog     | Returns clean trace error (limitation = formula coverage)                  |

---

## 6. Code Structure and Responsibilities

* Each module handles a single concern.
* LLM is used **only** for parsing and variable extraction.
* Numeric and symbolic math handled formally via Sympy and `safe_eval`.
* Tracer records every event for deterministic replay.

---

## 7. Development Perspectives

**Designer:**
Emphasized clarity — the trace itself is the user experience.

**AI Engineer:**
Maintained separation of probabilistic parsing and deterministic computation.

**Automation / DevOps:**
Single-file startup, easy to containerize, structured logs, auto-kill and restart scripts.

**Product Manager:**
Focused on accuracy and explainability; retry logic raises completion rate safely.

---

## 8. Limitations and Scalability

* Current limits stem from catalog scope (formulas, constants).
* Architecture itself scales by simply expanding `catalog_hs.yaml`.
* New STEM domains can be added without changing the solver design.

---

## 9. Vision

A small, auditable reasoning service for STEM learning:

* GPT-4o-mini for low-cost structured parsing.
* Deterministic solver for correctness and transparency.
* Optional integration with `n8n` for QA regression or coverage testing.
* With more time: automatic fallback routing for planner failures and confidence heuristics.

---

## 10. How to Run (Reviewer-Friendly)

### A. Clone and Install

```bash
git clone <GITHUB_LINK_TO_BE_ADDED>
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### B. Create `.env`

```bash
# .env (sample)
OPENAI_API_KEY=sk-proj-XXXXXXXXXXXXXXXXXXXXXXXX
OPENAI_MODEL=gpt-4o-mini
CATALOG_PATH=examples/catalog_hs.yaml
API_URL=http://localhost:8000
```

### C. One-Shot Dev Runner

Use the helper script to auto-start both API and UI:

```bash
python dev_run.py
```

**What this does:**

* Loads `.env` if present (or uses defaults).
* Kills existing processes on ports `8000` (API) and `8501` (UI).
* Starts Uvicorn + Streamlit automatically.
* Warns if `OPENAI_API_KEY` is missing (you can still test direct math).

### D. Token / Cost Guardrails

Limit usage manually:

```bash
unset OPENAI_API_KEY
```

To estimate how many problems $20 covers:

```
problems ≈ $20 / (price_per_1K_tokens * avg_tokens_per_problem / 1000)
```

### E. Reviewer Notes

* `dev_run.py` helps avoid full container rebuilds.
* `.env` is excluded via `.gitignore` for security.
* Include a `.env.example` file for reviewers.

---

## 11. Final Notes

* LLM used **only** for parsing; solver is fully deterministic.
* Trace-first design ensures transparency and reproducibility.
* Failures stem from missing formulas, not system design.
* Architecture is intentionally lightweight but production-ready in principle.

---


