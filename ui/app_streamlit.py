import os
import json
import streamlit as st
from agent.agent import solve
from agent.tracing import save_review

st.set_page_config(page_title="High School Helper", page_icon="🧮", layout="centered")
st.title("High School Helper 🧮⚙️")

with st.sidebar:
    st.markdown("### Settings")
    units = st.selectbox("Preferred units", ["SI", "imperial"], index=0)
    st.markdown("---")
    st.markdown("**Traces** are written to `traces/`. Reviews go to `reviews/`.")

user_q = st.text_area("Enter your question", height=150,
                      placeholder="e.g., If I throw a ball out a window that is 30 feet high at 10 mph, how far will it land?")

if st.button("Solve", type="primary"):
    if not user_q.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking with tools..."):
            out = solve(user_q, preferred_units=units)
        if out["ok"]:
            st.success(out["paraphrase"])
        else:
            st.error(out["message"])
        st.markdown("### Steps")
        for s in out.get("steps", []):
            st.write("- ", s)
        st.markdown("### Trace")
        trace_path = out.get("trace_path")
        if trace_path and os.path.exists(trace_path):
            with open(trace_path, "r", encoding="utf-8") as f:
                st.code(f.read(), language="json")
            st.download_button("Download trace JSON", data=open(trace_path, "rb"), file_name=os.path.basename(trace_path))
        st.markdown("---")
        st.markdown("### Not satisfied?")
        fb = st.text_area("Tell us what's off, and we'll log it for review.")
        if st.button("Submit review") and fb.strip():
            rid = save_review(trace_id=os.path.basename(trace_path) if trace_path else "", comment=fb, context={"units": units})
            st.info(f"Thanks! Logged: {os.path.basename(rid)}")
