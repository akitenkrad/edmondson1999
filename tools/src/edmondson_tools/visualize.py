#!/usr/bin/env python3
"""visualize.py — single-run visualization for the Edmondson 1999 model.

Reads `results/latest` (or `--results-dir`) and produces:
  - psi_learning_perf_timeseries.png : mean ψ̄ / L / Π per step (the causal chain)
  - mediation_scatter.png            : team ψ̄ → L → Π scatter (final-tick cross-section)
  - icc_trace.png                    : ICC(ψ) / ICC(L) over time + mediation ratio

Usage:
    uv run edmondson-tools visualize
    uv run edmondson-tools visualize --results-dir results/latest --output-dir out
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLOR_BG = "#FAFAF8"
C_PSI = "#534AB7"
C_L = "#0F6E56"
C_PI = "#F4A259"


def load_config(results_dir: str) -> dict | None:
    path = os.path.join(results_dir, "config.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def plot_timeseries(results_dir: str, output_dir: str, cfg: dict | None) -> None:
    path = os.path.join(results_dir, "teams.csv")
    if not os.path.exists(path):
        print(f"[visualize] no teams.csv at {results_dir}; skipping time series")
        return
    df = pd.read_csv(path)
    g = df.groupby("t").agg(
        psi=("psi", "mean"),
        learning=("learning", "mean"),
        performance=("performance", "mean"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.plot(g["t"], g["psi"], color=C_PSI, lw=2.4, label="ψ̄ (psychological safety)")
    ax.plot(g["t"], g["learning"], color=C_L, lw=2.0, label="L (learning behavior)")
    ax.plot(g["t"], g["performance"], color=C_PI, lw=2.0, ls="--", label="Π (team performance)")
    ax.set_xlabel("step t")
    ax.set_ylabel("team mean")
    title = "Causal chain over time: support/coaching → ψ → L → Π"
    if cfg:
        psi = cfg.get("psi", {})
        title += f"  (α={psi.get('alpha')}, δ={psi.get('delta')})"
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = os.path.join(output_dir, "psi_learning_perf_timeseries.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[visualize] wrote {out}")


def plot_mediation_scatter(results_dir: str, output_dir: str) -> None:
    path = os.path.join(results_dir, "teams.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    t_last = df["t"].max()
    t_half = t_last // 2
    tail = df[df["t"] >= t_half]
    agg = tail.groupby("team_id").agg(
        psi=("psi", "mean"), learning=("learning", "mean"), performance=("performance", "mean")
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(COLOR_BG)
    for ax in axes:
        ax.set_facecolor(COLOR_BG)

    def _fit_line(ax, x, y, color):
        if len(x) >= 2 and np.std(x) > 1e-9:
            b, a = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, a + b * xs, color=color, lw=1.6)
            r = np.corrcoef(x, y)[0, 1]
            ax.text(0.05, 0.92, f"B={b:.2f}  r={r:.2f}", transform=ax.transAxes, fontsize=9)

    axes[0].scatter(agg["psi"], agg["learning"], s=20, color=C_PSI, alpha=0.6)
    _fit_line(axes[0], agg["psi"].values, agg["learning"].values, C_PSI)
    axes[0].set_xlabel("team ψ̄ (psychological safety)")
    axes[0].set_ylabel("team L (learning behavior)")
    axes[0].set_title("ψ → L  (Table 5A: B≈.76)")

    axes[1].scatter(agg["learning"], agg["performance"], s=20, color=C_L, alpha=0.6)
    _fit_line(axes[1], agg["learning"].values, agg["performance"].values, C_L)
    axes[1].set_xlabel("team L (learning behavior)")
    axes[1].set_ylabel("team Π (performance)")
    axes[1].set_title("L → Π  (Table 4: R²≈.26)")

    fig.suptitle("Mediation cross-section (run second-half team averages)")
    fig.tight_layout()
    out = os.path.join(output_dir, "mediation_scatter.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[visualize] wrote {out}")


def plot_icc_trace(results_dir: str, output_dir: str) -> None:
    path = os.path.join(results_dir, "metrics.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    if "icc_psi" in df:
        ax.plot(df["t"], df["icc_psi"], color=C_PSI, lw=2.2, label="ICC(ψ)")
    if "icc_learning" in df:
        ax.plot(df["t"], df["icc_learning"], color=C_L, lw=1.8, label="ICC(L) snapshot")
    if "mediation_ratio" in df:
        ax.plot(df["t"], df["mediation_ratio"], color=C_PI, lw=1.6, ls="--", label="mediation ratio")
    ax.axhline(0.39, color="gray", lw=1, ls="-.", label="ICC(ψ) anchor .39")
    ax.set_xlabel("step t")
    ax.set_ylabel("value")
    ax.set_title("ICC(ψ) / ICC(L) / mediation-ratio trace")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = os.path.join(output_dir, "icc_trace.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[visualize] wrote {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="edmondson-tools visualize",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", "--results_dir", default="results/latest")
    parser.add_argument("--output-dir", "--output_dir", default=None)
    args = parser.parse_args(argv)

    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir
    os.makedirs(output_dir, exist_ok=True)

    cfg = load_config(results_dir)
    plot_timeseries(results_dir, output_dir, cfg)
    plot_mediation_scatter(results_dir, output_dir)
    plot_icc_trace(results_dir, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
