"""Structured CSV logging for AlphaScale experiments."""

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


FIELDNAMES: List[str] = [
    "domain",
    "dataset_fraction",
    "scale_id",
    "params",
    "compute",
    "energy",
    "val_accuracy",
    "test_accuracy",
    "train_time",
    "generalization_gap",
]


class ExperimentLogger:
    """Logs experiment results to a structured CSV file.

    Args:
        log_path: Path to the output CSV file.
    """

    def __init__(self, log_path: str = "results/experiments.csv") -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_file()

    def _initialize_file(self) -> None:
        """Create the CSV file with headers if it does not exist."""
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writeheader()

    def log(self, record: Dict[str, Any]) -> None:
        """Append a single experiment record to the CSV.

        Args:
            record: Dictionary containing experiment metrics.
                    Missing keys are filled with empty string.
        """
        row = {field: record.get(field, "") for field in FIELDNAMES}
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writerow(row)

    def load(self) -> List[Dict[str, Any]]:
        """Load all logged records from the CSV.

        Returns:
            List of record dictionaries.
        """
        if not self.log_path.exists():
            return []
        with open(self.log_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)
