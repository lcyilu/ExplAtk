from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def to_text(code: str | bytes | None) -> str:
    if code is None:
        return ""
    if isinstance(code, bytes):
        return code.decode("utf-8", errors="ignore")
    return code


@dataclass
class AttackResult:
    sample_id: str | int
    attack_name: str
    model_name: str
    true_label: int
    original_pred: int
    original_true_conf: float
    is_attackable: bool
    success: bool
    query_count: int
    original_code: str
    final_variant: str
    best_variant_by_conf_drop: str
    first_success_variant: str | None
    final_pred: int
    final_true_conf: float
    best_true_conf: float
    success_true_conf: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
