#!/usr/bin/env python3
"""show_experiment_settings.py — print a results directory's settings.

Reads `config.json` (run) or `sweep_config.json` (sweep) plus `llm_meta.json`
and renders them as a readable table, or as JSON with `--json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(path: Path) -> dict | None:
    if path.exists():
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    return None


def _find_config_file(results_dir: Path) -> tuple[Path, str]:
    run_cfg = results_dir / "config.json"
    sweep_cfg = results_dir / "sweep_config.json"
    if run_cfg.exists():
        return run_cfg, "run"
    if sweep_cfg.exists():
        return sweep_cfg, "sweep"
    raise FileNotFoundError(
        f"no settings file in: {results_dir}\n"
        f"  expected: config.json (run) or sweep_config.json (sweep)"
    )


def render_run_config(cfg: dict, source: Path) -> str:
    psi = cfg.get("psi", {})
    vb = cfg.get("voice_beta", {})
    lw = cfg.get("learning_weights", {})
    lines = [
        "=" * 72,
        "experiment settings (run)",
        "=" * 72,
        f"settings file: {source}",
        "-" * 72,
        f"decision_mode     : {cfg.get('decision_mode', '-')}",
        f"n_individuals     : {cfg.get('n_individuals', '-')} "
        f"({cfg.get('n_teams', '-')} teams × {cfg.get('team_size', '-')})",
        f"network           : {cfg.get('network_kind', '-')} "
        f"(k={cfg.get('network_k', '-')}, β={cfg.get('network_beta', '-')})",
        f"ψ-update λ/α/β/γ/δ : {psi.get('lambda', '-')} / {psi.get('alpha', '-')} / "
        f"{psi.get('beta', '-')} / {psi.get('gamma', '-')} / {psi.get('delta', '-')}",
        f"voice β0/β_ψ/β_f   : {vb.get('intercept', '-')} / {vb.get('beta_psafety', '-')} / "
        f"{vb.get('beta_fear', '-')}",
        f"learning w_v/w_h/w_e: {lw.get('w_voice', '-')} / {lw.get('w_help', '-')} / "
        f"{lw.get('w_error', '-')}",
        f"γ_L / γ_K / σ_obs  : {cfg.get('gamma_l', '-')} / {cfg.get('gamma_k', '-')} / "
        f"{cfg.get('sigma_obs', '-')}",
        f"p_retaliate        : {cfg.get('p_retaliate', '-')}",
        f"t_max / runs       : {cfg.get('t_max', '-')} / {cfg.get('runs', '-')}",
        f"seed (core)        : {cfg.get('seed', '-')}",
        f"LLM temp / seed    : {cfg.get('llm_temperature', '-')} / {cfg.get('llm_seed', '-')}",
        f"output_dir         : {cfg.get('output_dir', '-')}",
        "=" * 72,
    ]
    return "\n".join(lines)


def render_sweep_config(cfg: dict, source: Path) -> str:
    lines = [
        "=" * 72,
        f"experiment settings ({cfg.get('command', 'sweep')})",
        "=" * 72,
        f"settings file: {source}",
        "-" * 72,
        f"decision_mode     : {cfg.get('decision_mode', '-')}",
        f"n_teams × team    : {cfg.get('n_teams', '-')} × {cfg.get('team_size', '-')}",
        f"α values          : {cfg.get('alpha_values', '-')}",
        f"δ values          : {cfg.get('delta_values', '-')}",
        f"λ                 : {cfg.get('lambda', '-')}",
        f"runs/cell         : {cfg.get('runs', '-')}",
        f"t_max             : {cfg.get('t_max', '-')}",
        f"seed              : {cfg.get('seed', '-')}",
        "=" * 72,
    ]
    return "\n".join(lines)


def render_llm_meta(meta: dict) -> str:
    lines = [
        "LLM / determinism metadata",
        "-" * 72,
        f"decision_mode     : {meta.get('decision_mode', '-')}",
        f"model / endpoint  : {meta.get('llm_model', '-')} @ {meta.get('llm_endpoint', '-')}",
        f"temperature / seed: {meta.get('llm_temperature', '-')} / {meta.get('llm_seed', '-')}",
        f"LLM calls         : {meta.get('total_calls', '-')} "
        f"(cache-hit {meta.get('cache_hits', '-')}, "
        f"{100 * meta.get('cache_hit_rate', 0):.1f}%)",
        f"final_round       : {meta.get('final_round', '-')}",
        f"convergence_step  : {meta.get('convergence_step', '-')}",
        "=" * 72,
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="edmondson-tools show-experiment-settings",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", "--results_dir", default="results/latest")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table.")
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"error: directory does not exist: {results_dir}", file=sys.stderr)
        return 1

    try:
        cfg_path, kind = _find_config_file(results_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    cfg = _load(cfg_path)
    meta = _load(results_dir / "llm_meta.json")

    if args.json:
        payload = {"source": str(cfg_path), "kind": kind, "config": cfg, "llm_meta": meta}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        if kind == "run":
            print(render_run_config(cfg, cfg_path))
        else:
            print(render_sweep_config(cfg, cfg_path))
        if meta is not None:
            print(render_llm_meta(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
