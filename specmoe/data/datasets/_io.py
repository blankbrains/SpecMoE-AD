"""Strict parsing helpers shared by dataset loaders."""

import csv
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np


def load_numeric_matrix(
    path: Path,
    *,
    delimiter: str = ",",
    skiprows: int = 0,
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Required data file not found: {path}")
    try:
        values = np.loadtxt(
            path,
            delimiter=delimiter,
            skiprows=skiprows,
            dtype=np.float64,
            ndmin=2,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot parse numeric data file: {path}") from exc
    if values.size == 0:
        raise ValueError(f"Data file is empty: {path}")
    return values


def load_label_vector(path: Path) -> np.ndarray:
    """Load labels encoded as one column, equivalent to [T] or [T, 1]."""
    labels = load_numeric_matrix(path)
    if labels.shape[1] != 1:
        raise ValueError(
            f"Label file must have shape [T] or [T, 1] with one column: {path}"
        )
    return labels[:, 0]


def first_row_is_header(path: Path, *, delimiter: str = ",") -> bool:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        row = next(csv.reader(handle, delimiter=delimiter), None)
    if row is None:
        raise ValueError(f"Data file is empty: {path}")
    try:
        for value in row:
            float(value)
    except ValueError:
        return True
    return False


def normalized_fieldnames(fieldnames: Sequence[str]) -> dict:
    result = {}
    for original in fieldnames:
        normalized = str(original).strip().casefold()
        if not normalized:
            raise ValueError("CSV contains an empty column name")
        if normalized in result:
            raise ValueError(f"CSV contains duplicate column {normalized!r}")
        result[normalized] = original
    return result


def find_field(
    fields: dict,
    aliases: Iterable[str],
    *,
    purpose: str,
) -> str:
    for alias in aliases:
        match = fields.get(alias.casefold())
        if match is not None:
            return match
    raise ValueError(
        f"CSV is missing required {purpose} column; accepted names: "
        f"{sorted(set(aliases))}"
    )


def read_value_label_csv(
    path: Path,
    *,
    value_aliases: Tuple[str, ...] = ("value",),
    label_aliases: Tuple[str, ...] = ("label", "is_anomaly", "anomaly"),
) -> Tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV file not found: {path}")
    values = []
    labels = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV is empty or has no header: {path}")
        fields = normalized_fieldnames(reader.fieldnames)
        value_field = find_field(fields, value_aliases, purpose="value")
        label_field = find_field(fields, label_aliases, purpose="label")
        for row_number, row in enumerate(reader, start=2):
            try:
                value = float(str(row[value_field]).strip())
                label = float(str(row[label_field]).strip())
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid numeric value/label in {path} row {row_number}"
                ) from exc
            values.append(value)
            labels.append(label)
    if not values:
        raise ValueError(f"CSV contains no data rows: {path}")
    return (
        np.asarray(values, dtype=np.float64)[:, None],
        np.asarray(labels, dtype=np.float64),
    )


def read_value_label_csv(
    path: Path,
    *,
    value_aliases: Tuple[str, ...] = ("value",),
    label_aliases: Tuple[str, ...] = ("label", "is_anomaly", "anomaly"),
) -> Tuple[np.ndarray, np.ndarray]:
    """Read a labeled curve while preserving rows with blank benchmark fields."""
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV file not found: {path}")
    values = []
    labels = []
    value_missing = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV is empty or has no header: {path}")
        fields = normalized_fieldnames(reader.fieldnames)
        value_field = find_field(fields, value_aliases, purpose="value")
        label_field = find_field(fields, label_aliases, purpose="label")
        for row_number, row in enumerate(reader, start=2):
            raw_value = str(row[value_field]).strip()
            raw_label = str(row[label_field]).strip()
            if raw_value == "":
                value = np.nan
                value_missing.append(True)
            else:
                try:
                    value = float(raw_value)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid numeric value/label in {path} row {row_number}"
                    ) from exc
                if not np.isfinite(value):
                    raise ValueError(
                        f"CSV value must be finite in {path} row {row_number}"
                    )
                value_missing.append(False)

            if raw_label == "":
                label = 0.0
            else:
                try:
                    label = float(raw_label)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid numeric value/label in {path} row {row_number}"
                    ) from exc
                if not np.isfinite(label):
                    raise ValueError(
                        f"CSV label must be finite in {path} row {row_number}"
                    )
            values.append(value)
            labels.append(label)

    if not values:
        raise ValueError(f"CSV contains no data rows: {path}")
    value_array = np.asarray(values, dtype=np.float64)
    missing_mask = np.asarray(value_missing, dtype=bool)
    valid_positions = np.flatnonzero(~missing_mask)
    if valid_positions.size == 0:
        raise ValueError(f"CSV value column is entirely blank: {path}")
    if missing_mask.any():
        value_array[missing_mask] = np.interp(
            np.flatnonzero(missing_mask),
            valid_positions,
            value_array[valid_positions],
        )
    label_array = np.asarray(labels, dtype=np.float64)
    if not np.isin(label_array, [0, 1]).all():
        raise ValueError(f"CSV labels must be binary 0/1: {path}")
    return value_array[:, None], label_array
