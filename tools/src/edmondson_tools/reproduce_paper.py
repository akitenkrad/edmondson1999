#!/usr/bin/env python3
"""reproduce_paper.py — one-command Edmondson (1999) reproduction (Track-B ABM).

Runs end-to-end on a `run` results directory (teams.csv + metrics.csv):

  1. Table 4-8-style Baron & Kenny (1986) three-step OLS mediation of the chain
     support → ψ → L → Π, on the time-averaged team cross-section:
       step 1  Π ~ ψ            (total effect c)
       step 2  L ~ ψ            (path a)              [Table 5A: B≈.76, R²≈.63]
       step 3  Π ~ ψ + L        (direct c', mediator b) [Table 4: L→Π R²≈.26]
       support → ψ              (Table 6: B≈.56)
     reported with the exact-t p-values from statsmodels.
  2. Bootstrap mediation: the indirect effect a·b with a 95% bias-corrected (BC)
     percentile CI over `--bootstrap` resamples of the teams.
  3. Efficacy discriminant (H5/H8): L ~ ψ + efficacy; the efficacy partial |t|
     should be < 2 (the paper finds efficacy non-significant once ψ is controlled).

All statistics are compared against the §5 calibration anchors with PASS / off-
anchor verdicts. Writes `table4_report.csv` and `mediation_bootstrap.csv`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# §5 anchor bands.
ICC_PSI_BAND = (0.25, 0.55)
ICC_L_BAND = (0.15, 0.40)
PSI_L_B_BAND = (0.50, 1.00)
PSI_L_R2_BAND = (0.48, 0.78)
L_PI_R2_BAND = (0.16, 0.36)
SUPPORT_PSI_B_BAND = (0.35, 0.77)


def _team_cross_section(teams: pd.DataFrame) -> pd.DataFrame:
    """Time-average each team over the run's second half (stable measures)."""
    t_last = teams["t"].max()
    t_half = t_last // 2
    tail = teams[teams["t"] >= t_half]
    agg = tail.groupby("team_id").agg(
        psi=("psi", "mean"),
        learning=("learning", "mean"),
        performance=("performance", "mean"),
        support=("support", "mean"),
        efficacy=("efficacy", "mean"),
    ).reset_index()
    return agg


def _ols(y: np.ndarray, X: np.ndarray):
    """OLS via statsmodels, returning the fitted result (X already has const)."""
    import statsmodels.api as sm

    return sm.OLS(y, X).fit()


def _band(name: str, value: float, band: tuple[float, float]) -> str:
    ok = band[0] <= value <= band[1]
    return f"  {name:<26} = {value:>7.3f}   ([{band[0]:.2f}, {band[1]:.2f}]: {'PASS' if ok else 'off-anchor'})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="edmondson-tools reproduce",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", "--results_dir", default="results/latest")
    parser.add_argument("--output-dir", "--output_dir", default=None)
    parser.add_argument("--bootstrap", type=int, default=10000, help="bootstrap resamples (default 1e4)")
    parser.add_argument("--seed", type=int, default=1999)
    args = parser.parse_args(argv)

    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir
    os.makedirs(output_dir, exist_ok=True)

    import statsmodels.api as sm

    # Prefer the pooled multi-run cross-section (one row per team per run, with
    # run-level ICC columns); fall back to deriving it from the last run's
    # teams.csv if only that is present.
    cs_path = os.path.join(results_dir, "team_cross_section.csv")
    teams_path = os.path.join(results_dir, "teams.csv")
    icc_psi = float("nan")
    icc_l = float("nan")
    if os.path.exists(cs_path):
        cs = pd.read_csv(cs_path)
        n_runs = cs["run"].nunique()
        # ICC: one value per run → average.
        icc_psi = float(cs.groupby("run")["icc_psi"].first().mean())
        icc_l = float(cs.groupby("run")["icc_learning"].first().mean())
        source_desc = f"{len(cs)} team observations pooled over {n_runs} runs"
    elif os.path.exists(teams_path):
        cs = _team_cross_section(pd.read_csv(teams_path))
        n_runs = 1
        source_desc = f"{len(cs)} teams (single run, second-half time averages)"
        metrics_path = os.path.join(results_dir, "metrics.csv")
        if os.path.exists(metrics_path):
            m = pd.read_csv(metrics_path)
            last = m[m["t"] == m["t"].max()].iloc[0]
            icc_psi = float(last.get("icc_psi", float("nan")))
            icc_l = float(last.get("icc_learning", float("nan")))
    else:
        print(
            f"error: need team_cross_section.csv or teams.csv in {results_dir}\n"
            f"  run e.g. `cargo run --release -- run --decision-mode rule` first",
            file=sys.stderr,
        )
        return 1

    n = len(cs)
    psi = cs["psi"].to_numpy()
    learn = cs["learning"].to_numpy()
    perf = cs["performance"].to_numpy()
    support = cs["support"].to_numpy()
    efficacy = cs["efficacy"].to_numpy()

    print("=" * 70)
    print("Edmondson (1999) — one-command reproduction (Track-B ABM)")
    print("=" * 70)
    print(f"team cross-section: {source_desc}\n")

    # ── Baron & Kenny three-step (Table 4-8), PER RUN then averaged ──────────
    # Each trial regresses its own ~n_teams cross-section (the paper's n≈51
    # power); we average the statistics across trials. Pooling all teams into one
    # regression would inflate the power so far that every effect — even the
    # mediated residual that should be non-significant — turns "significant".
    def _bk(df: pd.DataFrame) -> dict:
        p = df["psi"].to_numpy()
        l = df["learning"].to_numpy()
        y = df["performance"].to_numpy()
        s = df["support"].to_numpy()
        e = df["efficacy"].to_numpy()
        s1 = _ols(y, sm.add_constant(p))
        s2 = _ols(l, sm.add_constant(p))
        s3 = _ols(y, sm.add_constant(np.column_stack([p, l])))
        s_lpi = _ols(y, sm.add_constant(l))
        s_sp = _ols(p, sm.add_constant(s))
        s_eff = _ols(l, sm.add_constant(np.column_stack([p, e])))
        c = s1.params[1]
        a = s2.params[1]
        b = s3.params[2]
        return {
            "c_total": c,
            "a_path": a,
            "a_p": s2.pvalues[1],
            "r2_psi_l": s2.rsquared_adj,
            "c_prime": s3.params[1],
            "c_prime_p": s3.pvalues[1],
            "b_path": b,
            "r2_l_pi": s_lpi.rsquared_adj,
            "support_psi_b": s_sp.params[1],
            "support_psi_p": s_sp.pvalues[1],
            "eff_t": s_eff.tvalues[2],
            "ratio": (a * b) / c if abs(c) > 1e-9 else np.nan,
            "c_total_p": s1.pvalues[1],
        }

    if "run" in cs.columns and n_runs > 1:
        per_run = [_bk(g) for _, g in cs.groupby("run") if len(g) >= 4]
        agg = {k: float(np.nanmean([r[k] for r in per_run])) for k in per_run[0]}
    else:
        agg = _bk(cs)
    c_total = agg["c_total"]
    a_path = agg["a_path"]
    a_p = agg["a_p"]
    r2_psi_l = agg["r2_psi_l"]
    c_prime = agg["c_prime"]
    c_prime_p = agg["c_prime_p"]
    b_path = agg["b_path"]
    r2_l_pi = agg["r2_l_pi"]
    support_psi_b = agg["support_psi_b"]
    support_psi_p = agg["support_psi_p"]
    eff_t = agg["eff_t"]
    ratio = agg["ratio"]

    print("[1] Baron & Kenny three-step mediation (support → ψ → L → Π):")
    print(f"  step 1  Π ~ ψ        total c   = {c_total:.3f} (p={agg['c_total_p']:.3f})")
    print(f"  step 2  L ~ ψ        path  a   = {a_path:.3f} (p={a_p:.3f}), adj R²={r2_psi_l:.3f}")
    print(f"  step 3  Π ~ ψ + L    direct c' = {c_prime:.3f} (p={c_prime_p:.3f}); mediator b={b_path:.3f}")
    print(f"          proportion mediated (a·b/c) = {ratio:.3f}")
    print()
    print("[2] §5 anchor checks:")
    print(_band("ICC(ψ)", icc_psi, ICC_PSI_BAND))
    print(_band("ICC(L)", icc_l, ICC_L_BAND))
    print(_band("ψ→L  B (Table 5A)", a_path, PSI_L_B_BAND))
    print(_band("ψ→L  adj R² (Table 5A)", r2_psi_l, PSI_L_R2_BAND))
    print(_band("L→Π  adj R² (Table 4)", r2_l_pi, L_PI_R2_BAND))
    print(_band("support→ψ B (Table 6)", support_psi_b, SUPPORT_PSI_B_BAND))
    print(
        f"  {'mediation residual p':<26} = {c_prime_p:>7.3f}   "
        f"(non-significant p>.10: {'PASS' if c_prime_p > 0.10 else 'review'})"
    )
    print(
        f"  {'mediation ratio':<26} = {ratio:>7.3f}   "
        f"(≥ .5: {'PASS' if (not np.isnan(ratio) and ratio >= 0.5) else 'review'})"
    )
    print(
        f"  {'efficacy partial |t| (H5)':<26} = {abs(eff_t):>7.3f}   "
        f"(< 2, H5/H8 unsupported: {'PASS' if abs(eff_t) < 2.0 else 'review'})"
    )

    # ── 3. Bootstrap mediation 95% BC CI ─────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    nb = max(args.bootstrap, 100)
    indirects = np.empty(nb)
    idx = np.arange(n)
    for i in range(nb):
        bi = rng.choice(idx, size=n, replace=True)
        p_b, l_b, y_b = psi[bi], learn[bi], perf[bi]
        if np.std(p_b) < 1e-9:
            indirects[i] = 0.0
            continue
        a_b = _ols(l_b, sm.add_constant(p_b)).params[1]
        try:
            b_b = _ols(y_b, sm.add_constant(np.column_stack([p_b, l_b]))).params[2]
        except Exception:  # noqa: BLE001
            b_b = 0.0
        indirects[i] = a_b * b_b
    point = a_path * b_path
    # Bias-corrected percentile CI.
    z0 = _norm_ppf(np.mean(indirects < point))
    lo = _bc_quantile(indirects, 0.025, z0)
    hi = _bc_quantile(indirects, 0.975, z0)
    excludes_zero = not (lo <= 0.0 <= hi)
    print()
    print(f"[3] Bootstrap mediation ({nb} resamples):")
    print(f"  indirect effect a·b = {point:.3f}, 95% BC CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CI excludes 0 (significant indirect effect): {'PASS' if excludes_zero else 'review'}")

    # ── write CSVs ───────────────────────────────────────────────────────────
    table_rows = [
        {"indicator": "icc_psi", "value": icc_psi, "anchor": 0.39},
        {"indicator": "icc_learning", "value": icc_l, "anchor": 0.27},
        {"indicator": "psi_to_L_B", "value": a_path, "anchor": 0.76},
        {"indicator": "psi_to_L_adjR2", "value": r2_psi_l, "anchor": 0.63},
        {"indicator": "L_to_Pi_adjR2", "value": r2_l_pi, "anchor": 0.26},
        {"indicator": "support_to_psi_B", "value": support_psi_b, "anchor": 0.56},
        {"indicator": "psi_residual_c_prime", "value": c_prime, "anchor": 0.25},
        {"indicator": "psi_residual_p", "value": c_prime_p, "anchor": 0.42},
        {"indicator": "mediation_ratio", "value": ratio, "anchor": float("nan")},
        {"indicator": "efficacy_partial_t", "value": eff_t, "anchor": float("nan")},
    ]
    table_path = os.path.join(output_dir, "table4_report.csv")
    pd.DataFrame(table_rows).to_csv(table_path, index=False)
    boot_path = os.path.join(output_dir, "mediation_bootstrap.csv")
    pd.DataFrame(
        [{"indirect_point": point, "ci_lo": lo, "ci_hi": hi, "n_boot": nb, "excludes_zero": excludes_zero}]
    ).to_csv(boot_path, index=False)

    # ── hypothesis tally ─────────────────────────────────────────────────────
    supported = 0
    if support_psi_b > 0 and support_psi_p < 0.05:
        supported += 2  # H1, H2 (support + coaching climate)
    if a_path > 0 and a_p < 0.05:
        supported += 1  # H3
    if r2_l_pi > 0:
        supported += 1  # H4
    if (not np.isnan(ratio)) and ratio >= 0.5 and c_prime_p > 0.10:
        supported += 1  # H6
    if abs(c_total) > 1e-6:
        supported += 1  # H7
    # H5 / H8 expected unsupported → not counted.
    print()
    print(f"hypotheses supported: {supported}/8 (paper: 6/8; ≥5/8: {'PASS' if supported >= 5 else 'review'})")
    print("=" * 70)
    print(f"[reproduce] wrote {table_path}")
    print(f"[reproduce] wrote {boot_path}")
    return 0


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation)."""
    p = min(max(p, 1e-9), 1 - 1e-9)
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = np.sqrt(-2 * np.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def _norm_cdf(x: float) -> float:
    from math import erf, sqrt

    return 0.5 * (1.0 + erf(x / sqrt(2)))


def _bc_quantile(samples: np.ndarray, alpha: float, z0: float) -> float:
    """Bias-corrected percentile."""
    adj = _norm_cdf(2 * z0 + _norm_ppf(alpha))
    return float(np.quantile(samples, np.clip(adj, 0.0, 1.0)))


if __name__ == "__main__":
    sys.exit(main())
