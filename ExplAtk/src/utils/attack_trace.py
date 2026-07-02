"""
Attack trace logging utilities for ExplAtk.

This module is intentionally standalone so the attack implementation only needs
small hook points. It writes one JSON object per actual model query, which is
convenient for later vulnerability-space profiling and explanation-guidance
analysis.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


_DEFAULT_TRACE_DIR = "attack_traces"


def _safe_filename(value: Any, max_len: int = 160) -> str:
    text = str(value if value is not None else "unknown")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    if not text:
        text = "unknown"
    return text[:max_len]


class AttackTraceLogger:
    """JSONL logger for query-level attack traces."""

    def __init__(
        self,
        sample_id: str,
        model_name: str = "unknown",
        attack_name: str = "expl_atk",
        trace_dir: Optional[str] = None,
        enabled: Optional[bool] = None,
        original_true_conf: Optional[float] = None,
        guidance_mode: Optional[str] = None,
    ):

        if enabled is None:
            # Default OFF for large-scale attack runs. Enable tracing explicitly
            # by passing trace_dir or setting EXPLATK_TRACE=1/true/yes/on.
            env_flag = os.environ.get("EXPLATK_TRACE", "").lower()
            enabled = trace_dir is not None or env_flag in {"1", "true", "yes", "on"}
        self.enabled = enabled
        self.sample_id = sample_id
        self.model_name = model_name
        self.attack_name = attack_name
        self.file = None
        self.path = None

        if not self.enabled:
            return

        root = Path(trace_dir or os.environ.get("EXPLATK_TRACE_DIR", _DEFAULT_TRACE_DIR))
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{_safe_filename(model_name)}__{_safe_filename(sample_id)}__{run_tag}.jsonl"
        self.path = root / _safe_filename(model_name) / filename
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", encoding="utf-8")

        start_payload = {
            "sample_id": sample_id,
            "model_name": model_name,
            "attack_name": attack_name,
            "trace_path": str(self.path),
        }
        if original_true_conf is not None:
            start_payload["original_true_conf"] = float(original_true_conf)
        if guidance_mode is not None:
            start_payload["guidance_mode"] = guidance_mode
        self.write_event("start", start_payload)


    def write_event(self, event_type: str, payload: Dict[str, Any]):
        if not self.enabled or self.file is None:
            return
        record = {
            "event": event_type,
            "time": datetime.now().isoformat(timespec="microseconds"),
            **payload,
        }
        self.file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.file.flush()

    def log_query(
        self,
        *,
        phase: str,
        generation: Optional[int],
        individual_index: Optional[int],
        local_query_index: int,
        global_query_count: Optional[int],
        replacements,
        pred: int,
        true_label: int,
        true_conf: float,
        margin: float,
        success: bool,
        guidance_mode: Optional[str] = None,
    ):
        payload = {
            "sample_id": self.sample_id,
            "model_name": self.model_name,
            "attack_name": self.attack_name,
            "phase": phase,
            "generation": generation,
            "individual_index": individual_index,
            "local_query_index": local_query_index,
            "global_query_count": global_query_count,
            "num_replacements": len(replacements),
            "replacements": replacements,
            "pred": int(pred),
            "true_label": int(true_label),
            "true_conf": float(true_conf),
            "margin": float(margin),
            "success": bool(success),
        }
        if guidance_mode is not None:
            payload["guidance_mode"] = guidance_mode
        self.write_event("query", payload)


    def close(self, final_payload: Optional[Dict[str, Any]] = None):
        if not self.enabled or self.file is None:
            return
        self.write_event("end", final_payload or {})
        self.file.close()
        self.file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close({"exception": repr(exc) if exc else None})
