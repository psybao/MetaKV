"""Derived logical-position weighting helpers; raw third-party corpora are not bundled."""
import csv
from pathlib import Path

def normalized_weights(counts):
    values = [float(x) for x in counts]
    total = sum(values)
    if total <= 0:
        raise ValueError("positive total required")
    return [x / total for x in values]

def read_position_counts(csv_path, column='count'):
    with Path(csv_path).open(newline='', encoding='utf-8-sig') as handle:
        return [float(row[column]) for row in csv.DictReader(handle)]
