"""Helpers for visualising ROC curves from stored evaluation results."""

from __future__ import annotations

import csv
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


RunData = Dict[str, Tuple[np.ndarray, np.ndarray]]


def load_run_probabilities(
    csv_path: str | Path,
    *,
    split: str = "test",
    run_names: Optional[Sequence[str]] = None,
) -> RunData:
    """
    Load probabilities + labels for the requested split and runs.

    Args:
        csv_path: Path to a CSV produced by ``src/train/pruning.py``.
        split: Dataset split to filter on (defaults to ``"test"``).
        run_names: Optional subset of runs to keep; when omitted all runs present
            in the CSV for the chosen split are returned.

    Returns:
        Mapping of ``run_name`` to ``(probabilities, labels)`` numpy arrays.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the CSV is missing required columns or no rows match.
    """

    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Results CSV not found at '{csv_file}'.")

    requested = set(run_names) if run_names is not None else None
    runs: Dict[str, Dict[str, list]] = defaultdict(lambda: {"prob": [], "label": []})

    with csv_file.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"run_name", "split", "probability", "label"}
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            missing_joined = ", ".join(sorted(missing))
            raise ValueError(
                f"Results CSV is missing required columns: {missing_joined}."
            )

        for row in reader:
            if row.get("split") != split:
                continue
            run_name = row.get("run_name")
            if not run_name:
                continue
            if requested is not None and run_name not in requested:
                continue

            try:
                prob = float(row["probability"])
                label = int(row["label"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid probability/label values for run '{run_name}' in '{csv_file}'."
                ) from exc

            runs[run_name]["prob"].append(prob)
            runs[run_name]["label"].append(label)

    if not runs:
        if requested:
            missing_runs = ", ".join(sorted(requested))
            raise ValueError(
                f"No rows found for split '{split}' matching runs: {missing_runs}."
            )
        raise ValueError(f"No rows found for split '{split}' in '{csv_file}'.")

    output: RunData = {}
    for run_name, payload in runs.items():
        probs = np.asarray(payload["prob"], dtype=np.float64)
        labels = np.asarray(payload["label"], dtype=np.int64)
        output[run_name] = (probs, labels)
    return output


def plot_roc_curves(
    csv_path: str | Path,
    *,
    split: str = "test",
    run_names: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    output_path: Optional[str | Path] = None,
) -> plt.Axes:
    """
    Plot ROC curves for one or more runs loaded from a CSV file.

    Args:
        csv_path: Path to the CSV with stored probabilities.
        split: Dataset split to filter on (defaults to ``"test"``).
        run_names: Optional iterable specifying which runs to plot.
        ax: Existing matplotlib axis to draw on (a new figure is created when
            omitted).
        title: Optional plot title; defaults to ``"ROC Curves (<split>)"``.
        output_path: Optional path to save the rendered figure.

    Returns:
        The matplotlib axis containing the ROC plot.
    """

    runs = load_run_probabilities(csv_path, split=split, run_names=run_names)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    plotted = False
    for run_name in sorted(runs):
        probs, labels = runs[run_name]
        if probs.size == 0:
            continue
        if np.unique(labels).size < 2:
            warnings.warn(
                f"Skipping run '{run_name}' for split '{split}' because only a single class is present.",
                RuntimeWarning,
            )
            continue

        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{run_name} (AUC={roc_auc:.3f})")
        plotted = True

    if not plotted:
        raise ValueError(
            "Unable to plot ROC curves because no run had both positive and negative samples."
        )

    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        color="grey",
        linewidth=1.0,
        label="baseline",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title or f"ROC Curves ({split})")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, bbox_inches="tight")

    return ax


if __name__ == "__main__":
    import sys

    # Default to ./results.csv if no argument is given
    if len(sys.argv) < 2:
        csv_path = "results.csv"
        print("No CSV path provided. Using default: results.csv")
    else:
        csv_path = sys.argv[1]
    plot_roc_curves(csv_path, output_path="roc_curve.pdf")
    plt.show()
