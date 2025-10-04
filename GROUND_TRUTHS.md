# High School Helper — Ground Truths

## 1. Purpose
An AI agent that solves high‑school math and physics problems **accurately**, with **visible tool use** and **traceable reasoning**.

## 2. Scoring Truths
- **Accuracy > aesthetics**
- **Tool use required** (deterministic Python/SymPy/Pint)
- **Transparent reasoning** (JSON traces)
- **Subjective**: clarity, robustness to curveballs, trust

## 3. Expectations
- Start from examples, build **general tools** (not hard-coded)
- Handle curveballs (mixed units, odd phrasing, impossible setups)
- Runnable repo; no paid API required (OpenAI optional but integrated)

## 4. Tooling
- ChatGPT API (planner/paraphraser)
- SymPy for math; Pint for units
- Verification: dimensional checks + numeric re‑substitution

## 5. Constraints
- Do not let LLM guess numerical results
- Always log trace JSON
- Prefer SI; handle imperial consistently

## 6. Canonical Logic Snippets
- **sqrt(N)** → SymPy exact & float
- **Quadratic** → discriminant, real/complex
- **Horizontal projectile** → convert units; t=√(2h/g); range=v_x t
- **Vertical throw** → h_max=h0+v^2/(2g)

## 7. Deliverables
- Agent, tools, UI, tests, traces, reviews, README
