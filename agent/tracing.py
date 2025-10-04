from __future__ import annotations
import os
import json
import time
from typing import Any, Dict, List
from .models import ToolCall, VerificationResult, FinalAnswer, Plan

TRACE_DIR = os.getenv("TRACE_DIR", "traces")
REVIEW_DIR = os.getenv("REVIEW_DIR", "reviews")

os.makedirs(TRACE_DIR, exist_ok=True)
os.makedirs(REVIEW_DIR, exist_ok=True)

def ts() -> str:
    return time.strftime("%Y%m%d_%H%M%SZ", time.gmtime())

def save_trace(meta: Dict[str, Any], user_input: Dict[str, Any], plan: Plan,
               unit_norm: Dict[str, Any], tool_calls: List[ToolCall],
               verification: VerificationResult, final: FinalAnswer) -> str:
    data = {
        "meta": meta,
        "input": user_input,
        "parse_intent": plan.model_dump(),
        "unit_normalize": unit_norm,
        "tool_calls": [tc.model_dump() for tc in tool_calls],
        "verification": verification.model_dump(),
        "final": final.model_dump(),
    }
    fname = f"run_{ts()}.json"
    fpath = os.path.join(TRACE_DIR, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return fpath

def save_review(trace_id: str, comment: str, context: Dict[str, Any] | None = None) -> str:
    data = {
        "trace_id": trace_id,
        "user_feedback": comment,
        "ui_context": context or {},
        "action_taken": "queued_for_review"
    }
    fname = f"review_{ts()}.json"
    fpath = os.path.join(REVIEW_DIR, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return fpath
