#!/usr/bin/env python3
"""
Compute cross-set TM-scores between two CIF directories and write a heatmap.
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List

# Limit OpenMP/BLAS threading to avoid resource errors on constrained systems.
for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
    os.environ.setdefault(_env, "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute all-vs-all cross-set TM-scores between two CIF directories "
            "using TMalign and save a heatmap."
        )
    )
    parser.add_argument("set_a_dir", help="Directory for set A CIF files.")
    parser.add_argument("set_b_dir", help="Directory for set B CIF files.")
    parser.add_argument(
        "--tm-bin",
        default="TMalign",
        help="TM-align executable name/path (default: TMalign).",
    )
    parser.add_argument(
        "--out-dir",
        default="cross_set_tm_analysis",
        help="Output directory for CSV/NPY and heatmap (default: cross_set_tm_analysis).",
    )
    parser.add_argument(
        "--output-prefix",
        default="cross_set_tm",
        help="Prefix for output files (default: cross_set_tm).",
    )
    parser.add_argument(
        "--max-labels",
        type=int,
        default=60,
        help="Maximum number of axis labels to draw per axis (default: 60).",
    )
    parser.add_argument(
        "--recursive",
        dest="recursive",
        action="store_true",
        help="Recursively discover CIF files (default).",
    )
    parser.add_argument(
        "--non-recursive",
        dest="recursive",
        action="store_false",
        help="Only scan top-level directory for CIF files.",
    )
    parser.set_defaults(recursive=True)
    return parser.parse_args()


def find_cif_files(directory: Path, recursive: bool) -> List[Path]:
    if not directory.exists() or not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    if recursive:
        candidates = directory.rglob("*")
    else:
        candidates = directory.iterdir()

    files = sorted(p for p in candidates if p.is_file() and p.suffix.lower() == ".cif")
    if not files:
        raise FileNotFoundError(f"No CIF files found in {directory}")
    return files


def to_relative_labels(paths: List[Path], root: Path) -> List[str]:
    labels: List[str] = []
    for path in paths:
        try:
            labels.append(str(path.relative_to(root)))
        except ValueError:
            labels.append(path.name)
    return labels


def parse_tm_score(output: str) -> float:
    matches: List[float] = []
    for line in output.splitlines():
        line = line.strip()
        if "TM-score" not in line:
            continue
        match = re.search(r"TM-score\s*=\s*([0-9]*\.?[0-9]+)", line)
        if match:
            matches.append(float(match.group(1)))

    if matches:
        # TM-align prints two directional TM-scores; use the max.
        return max(matches)
    raise ValueError("TM-score not found in aligner output.")


def run_tm_alignment(bin_path: str, model_a: Path, model_b: Path) -> float:
    """Run TM-align/US-align and return TM-score; NaN on failure."""
    cmd = [bin_path, str(model_a), str(model_b)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    combined_output = (result.stdout or "") + (result.stderr or "")

    if result.returncode != 0:
        print(
            f"Warning: TM-align failed for {model_a} vs {model_b}: "
            f"return code {result.returncode}",
            file=sys.stderr,
        )
        return np.nan
    try:
        return parse_tm_score(combined_output)
    except ValueError:
        print(
            f"Warning: could not parse TM-score for {model_a} vs {model_b}",
            file=sys.stderr,
        )
        return np.nan


def build_cross_tm_matrix(models_a: List[Path], models_b: List[Path], tm_bin: str) -> np.ndarray:
    n_a = len(models_a)
    n_b = len(models_b)
    matrix = np.full((n_a, n_b), np.nan, dtype=float)

    total = n_a * n_b
    done = 0
    for i, model_a in enumerate(models_a):
        print(f"Processing set A structure {i + 1}/{n_a}: {model_a.name}", file=sys.stderr)
        for j, model_b in enumerate(models_b):
            matrix[i, j] = run_tm_alignment(tm_bin, model_a, model_b)
            done += 1
            if done == 1 or done == total or done % 100 == 0:
                print(f"  TM-align progress: {done}/{total}", file=sys.stderr)
    return matrix


def save_outputs(
    matrix: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    out_dir: Path,
    output_prefix: str,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{output_prefix}.npy", matrix)
    df = pd.DataFrame(matrix, index=row_labels, columns=col_labels)
    df.to_csv(out_dir / f"{output_prefix}.csv")
    return df


def plot_heatmap(df: pd.DataFrame, output_path: Path, max_labels: int) -> None:
    n_rows, n_cols = df.shape
    fig_w = max(8.0, min(24.0, n_cols * 0.22 + 4.0))
    fig_h = max(6.0, min(24.0, n_rows * 0.22 + 3.0))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    img = ax.imshow(df.values, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("TM-score (max of TM-align 1/2)")

    ax.set_xlabel("Set B CIF files")
    ax.set_ylabel("Set A CIF files")
    ax.set_title("Cross-set TM-score heatmap")

    if n_cols <= max_labels:
        ax.set_xticks(np.arange(n_cols))
        ax.set_xticklabels(df.columns, rotation=90, fontsize=6)
    else:
        ax.set_xticks([])

    if n_rows <= max_labels:
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels(df.index, fontsize=6)
    else:
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    start = time.monotonic()
    args = parse_args()

    set_a_dir = Path(args.set_a_dir)
    set_b_dir = Path(args.set_b_dir)
    out_dir = Path(args.out_dir)

    set_a_files = find_cif_files(set_a_dir, args.recursive)
    set_b_files = find_cif_files(set_b_dir, args.recursive)

    print(f"Found {len(set_a_files)} CIF files in set A: {set_a_dir}", file=sys.stderr)
    print(f"Found {len(set_b_files)} CIF files in set B: {set_b_dir}", file=sys.stderr)

    row_labels = to_relative_labels(set_a_files, set_a_dir)
    col_labels = to_relative_labels(set_b_files, set_b_dir)

    tm_matrix = build_cross_tm_matrix(set_a_files, set_b_files, args.tm_bin)
    df = save_outputs(tm_matrix, row_labels, col_labels, out_dir, args.output_prefix)

    heatmap_path = out_dir / f"{args.output_prefix}_heatmap.png"
    plot_heatmap(df, heatmap_path, args.max_labels)

    elapsed = time.monotonic() - start
    print(f"Wrote matrix: {out_dir / (args.output_prefix + '.csv')}", file=sys.stderr)
    print(f"Wrote matrix: {out_dir / (args.output_prefix + '.npy')}", file=sys.stderr)
    print(f"Wrote heatmap: {heatmap_path}", file=sys.stderr)
    print(f"Completed in {elapsed:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
