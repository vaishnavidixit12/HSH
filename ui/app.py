# -----------------------------------------------------------------------------
# Streamlit Frontend for HS Single-Pipe Solver
# Purpose:
#   Minimal UI to (1) browse catalog, (2) parse a natural-language question,
#   and (3) call the deterministic solver, rendering traces and results.
#
#---------------------------------------------------------------------------

import os, json, requests, streamlit as st
from dotenv import load_dotenv

# Load .env to pick API_URL at runtime for local/remote backends
load_dotenv()
API_URL = os.getenv("API_URL","http://127.0.0.1:8000")

# Page setup and header
st.set_page_config(page_title="HS Single-Pipe Solver", layout="centered")
st.title("HS Math & Physics — Single-Pipe Solver")

# ---------------- Sidebar: Catalog Explorer -----------------------------------
with st.sidebar:
    st.subheader("Agent Scope (Catalog)")
    if st.checkbox("Show catalog"):
        # Simple GET to list formulas; displays a compact table
        r = requests.get(f"{API_URL}/catalog")
        if r.status_code == 200:
            data = r.json()
            st.caption(f"{data['count']} formulas")
            rows = [{
                "domain": it["domain"], "topic": it["topic"], "id": it["id"],
                "name": it["name"], "eq": it["eq"], "tags": ", ".join(it.get("tags", []))
            } for it in data["items"]]
            st.dataframe(rows, use_container_width=True, height=400)
        else:
            st.error(f"Catalog error: {r.text}")

# ---------------- Main Form: Parse + Solve ------------------------------------
with st.form("qform"):
    # Seeded example question to make demo-ready
    q = st.text_area("Enter your question", height=120, value="Please Add Your Question Here")
    target = st.text_input("Target symbol (optional)")
    disp = st.text_input("Display unit (optional)", value="")
    submit = st.form_submit_button("Solve")

if submit and q.strip():
    # First phase: /parse → uses LLM to structure the input
    with st.spinner("Parsing with ChatGPT..."):
        r = requests.post(f"{API_URL}/parse", json={"question": q})
        if r.status_code != 200:
            # Render parse error as-is from the API
            st.error(f"Parse error: {r.text}")
        else:
            parsed = r.json()
            # Compose payload for the solver (the deterministic engine)
            payload = {
                "question": q,
                "domain": parsed["domain"],
                "paraphrase": parsed["paraphrase"],
                "keywords": parsed.get("keywords", []),
                "values": parsed.get("values", []),
                "candidate_formula_id": parsed.get("candidate_formula_id"),
                "target_symbol": target or None,
                "display_unit": disp or None,
            }
            # Second phase: /solve → deterministic, unit-aware numeric pipeline
            with st.spinner("Solving deterministically..."):
                r2 = requests.post(f"{API_URL}/solve", json=payload)
                # NOTE: The following branch checks r.status_code instead of r2.status_code
                # and references 'endpoint' which is not defined. Left unchanged (per instruction).
                if r.status_code != 200:
                    try:
                        err = r.json()
                        tb = "\n".join(err.get("traceback", []))
                        st.error(f"{('Parse' if endpoint=='/parse' else 'Solve')} error: {err.get('type')}: {err.get('error')}")
                        if tb:
                            with st.expander("API traceback"):
                                st.code(tb, language="text")
                    except Exception:
                        st.error(f"{('Parse' if endpoint=='/parse' else 'Solve')} error: {r.text}")
                    st.stop()
                else:
                    res = r2.json()
                    # Failure path: show paraphrase, chain, trace, and candidates for debugging
                    if not res.get("ok", False):
                        st.error(f"Solver error ({res.get('error_kind','unknown')}): {res.get('error')}")
                        st.subheader("Paraphrased")
                        st.write(res.get("paraphrase",""))
                        st.subheader("Planner Chain")
                        st.json(res.get("chain", []))
                        st.subheader("Selected Formula (if any)")
                        st.code(str(res.get("selected_formula", {})))
                        st.subheader("Trace")
                        st.code(json.dumps(res.get("trace", []), indent=2))
                        st.subheader("Top Candidates")
                        st.write(res.get("candidates", []))
                    else:
                        # Success path: render selected formula, chain, substitutions, steps, answer
                        st.subheader("Paraphrased")
                        st.write(res["paraphrase"])
                        st.subheader("Selected Formula")
                        st.code(f"{res['selected_formula'].get('name','?')}  ({res['selected_formula'].get('eq','')})")
                        st.subheader("Planner Chain")
                        st.json(res.get("chain", []))
                        st.subheader("Substitutions (SI)")
                        st.table(res["substitutions"])
                        st.subheader("Steps")
                        st.write("\n".join("• " + s for s in res["steps"]))
                        st.subheader("Answer")
                        st.success(f"{res['answer']['symbol']} = {res['answer']['value']} {res['answer']['unit']}")
                        # Extra details for explainability
                        with st.expander("Trace"):
                            st.code(json.dumps(res["trace"], indent=2))
                        with st.expander("Top Candidates"):
                            st.write(res.get("candidates", []))
