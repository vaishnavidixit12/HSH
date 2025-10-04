# High School Helper

Deterministic tool‑based agent with ChatGPT planning + Python solvers. Logs JSON traces, shows steps, and supports user reviews.

## Quickstart
1. **Clone** and open in **Dev Container** (or local Python 3.11).
2. Copy `.env.example` to `.env` and set your `OPENAI_API_KEY` (optional but recommended).
3. Install deps: `pip install -r requirements.txt`.
4. Run UI: `streamlit run ui/app_streamlit.py`.
5. Run tests: `pytest`.

## Modes
- **Planner/Paraphraser**: OpenAI model via `OPENAI_MODEL` (e.g., `gpt-4o-mini`).
- **Solvers**: Python tools (SymPy/Pint) — no API required.

## Traces & Reviews
- Traces: `traces/run_*.json`
- Reviews: `reviews/review_*.json`

## Example Problems
- `sqrt(9843765983475)`
- Projectile from 30 ft at 10 mph (no air)
- Roots of `3x^2 + 4x + 5`
- Vertical throw at 10 mph from 6 ft — max height

## Policy
- Default **3 significant figures**, units printed
- SI preferred; imperial supported
- "No answer > wrong answer" — verification failure yields a safe refusal with rationale
