#!/usr/bin/env python3
"""visualize_sweep.py — sweep visualization for the Edmondson 1999 model.

Reads one row per trial from the sweep's cell child runs (`events.jsonl` の `terminal`
行．旧 `sweep_summary.csv` の 1 行がこの 1 行にあたる) and produces:
  - sweep_icc_heatmap.png        : α × δ heatmap of mean ICC(ψ) (anchor .39)
  - sweep_mediation_heatmap.png  : α × δ heatmap of mean mediation ratio
  - sweep_r2_heatmap.png         : α × δ heatmap of mean ψ→L R² (anchor .63)

Usage:
    uv run edmondson-tools visualize-sweep
    uv run edmondson-tools visualize-sweep --results-dir results/<ts>_sweep   # legacy も可
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from edmondson_tools import runs

COLOR_BG = "#FAFAF8"


def _heatmap(df: pd.DataFrame, value: str, title: str, anchor, output_dir: str, fname: str) -> None:
    if value not in df.columns:
        print(f"[visualize-sweep] no column {value}; skipping {fname}")
        return
    pivot = df.pivot_table(index="delta", columns="alpha", values=value, aggfunc="mean")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(COLOR_BG)
    im = ax.imshow(pivot.values, cmap="viridis", origin="lower", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.2f}" for v in pivot.index])
    ax.set_xlabel("α (context-support weight)")
    ax.set_ylabel("δ (shared-belief convergence weight)")
    title_full = title
    if anchor is not None:
        title_full += f"  (paper anchor ≈ {anchor})"
    ax.set_title(title_full)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(im, ax=ax, label=value)
    fig.tight_layout()
    out = os.path.join(output_dir, fname)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[visualize-sweep] wrote {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="edmondson-tools visualize-sweep",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", "--results_dir", default=None)
    parser.add_argument("--output-dir", "--output_dir", default=None)
    args = parser.parse_args(argv)

    sweep_dir = runs.resolve_run_dir(args.results_dir, subcommand="sweep")
    output_dir = str(runs.analysis_output_dir(sweep_dir, args.output_dir))
    df = runs.sweep_trials(sweep_dir)

    _heatmap(df, "icc_psi", "Mean ICC(ψ) over α × δ", 0.39, output_dir, "sweep_icc_heatmap.png")
    _heatmap(
        df, "mediation_ratio", "Mean mediation ratio over α × δ", 0.5, output_dir,
        "sweep_mediation_heatmap.png",
    )
    _heatmap(df, "r2_psi_l", "Mean ψ→L R² over α × δ", 0.63, output_dir, "sweep_r2_heatmap.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
