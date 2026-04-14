from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd


def dataclass_to_json(instance: Any) -> str:
    """Serialize a dataclass-like object to a JSON string."""
    return json.dumps(asdict(instance), ensure_ascii=False)


def export_csv(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save a DataFrame to CSV, creating parent directories if needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
