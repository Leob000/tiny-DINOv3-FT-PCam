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
            # Ignore runs that containing some word
            if any(word in run_name.lower() for word in ("int8", "bf16")):
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
    # Collect plotted lines with their AUC so we can sort the legend by AUC later.
    plotted_entries = []  # list of tuples (auc, Line2D)
    for idx, run_name in enumerate(sorted(runs)):
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
        # Cycle through linestyles/markers so overlapping curves are distinguishable.
        linestyles = ["-", "--", "-.", ":"]
        markers = [None, "o", "s", "D"]
        ls = linestyles[idx % len(linestyles)]
        marker = markers[idx % len(markers)]
        markevery = max(1, len(fpr) // 20)

        # Better labels for common run types
        lowered = run_name.lower()
        if lowered.startswith("eval_fullft"):
            base_label = "Full fine-tune"
        elif lowered.startswith("eval_lora"):
            base_label = "LoRA"
        elif lowered.startswith("eval_head_only"):
            base_label = "Head-only"
        else:
            base_label = run_name

        if "noprune" in lowered:
            prune_label = "no pruning"
        elif any(k in lowered for k in ("attn-mlp", "tsvd", "prune")):
            prune_label = "pruned"
        else:
            prune_label = None

        nice_label = f"{base_label}, {prune_label}" if prune_label else base_label

        (line,) = ax.plot(
            fpr,
            tpr,
            # label=f"{run_name} (AUC={roc_auc:.3f})",
            label=f"(AUC={roc_auc:.3f}) " + nice_label,
            linestyle=ls,
            marker=marker,
            markevery=markevery,
            markersize=4,
            linewidth=1.25,
            alpha=0.8,
        )
        plotted_entries.append((roc_auc, line))
        plotted = True
    if not plotted:
        raise ValueError(
            "Unable to plot ROC curves because no run had both positive and negative samples."
        )

    (baseline_line,) = ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        color="grey",
        label="baseline",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    # Ensure the ROC plot uses equal scaling on X and Y so the plotted area is square.
    if hasattr(ax, "set_box_aspect"):
        try:
            ax.set_box_aspect(1.0)
        except Exception:
            ax.set_aspect("equal", adjustable="box")
    else:
        ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title or f"ROC Curves ({split} set, no quantization)")
    # Sort legend entries by AUC (descending) so highest AUC appears at the top.
    # Keep the baseline entry at the bottom.
    plotted_entries.sort(key=lambda t: t[0], reverse=True)
    handles = [entry[1] for entry in plotted_entries]
    labels = [h.get_label() for h in handles]
    ax.legend(handles=handles, labels=labels, loc="lower right")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, bbox_inches="tight")

    return ax


if __name__ == "__main__":
    import sys

    # Default to reports/results_probs.csv if no argument is given
    if len(sys.argv) < 2:
        csv_path = "reports/results_probs.csv"
        print("No CSV path provided. Using default: reports/results_probs.csv")
    else:
        csv_path = sys.argv[1]
    plot_roc_curves(csv_path, output_path="reports/roc_curve.pdf")
    plt.show()
