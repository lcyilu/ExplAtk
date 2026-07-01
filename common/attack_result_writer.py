from __future__ import annotations

import csv
import json
from pathlib import Path

from common.attack_result import AttackResult


class AttackResultWriter:
    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def write_jsonl(self, result: AttackResult) -> None:
        with self.output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

    def write_csv(self, results: list[AttackResult]) -> None:
        if not results:
            return
        with self.output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].to_dict().keys()))
            writer.writeheader()
            for item in results:
                writer.writerow(item.to_dict())
