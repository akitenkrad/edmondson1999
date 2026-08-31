#!/usr/bin/env python3
"""show_experiment_settings.py — print a run's conditions and what it recorded.

runvault の run ディレクトリは `config.json` (封筒 + `parameters`) に条件を，
`run.json` に同一性 (`run_uid` / `config_hash` / `master_seed` / `llm` ブロック) を，
`metrics.csv` の step を持たない行に run 全体を 1 つの値で表す指標を持つ．旧
`config.json` / `sweep_config.json` / `llm_meta.json` の中身はこの 3 つに分かれた．

legacy な `results/<timestamp>/` はそのまま渡せば従来どおり読める．

Usage:
    uv run edmondson-tools show-experiment-settings
    uv run edmondson-tools show-experiment-settings --subcommand sweep
    uv run edmondson-tools show-experiment-settings --results-dir results/20260530_000000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from edmondson_tools import runs


def _load(path: Path) -> dict | None:
    if path.exists():
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    return None


def _parameters(run_dir: Path) -> tuple[dict, str]:
    """条件と，それがどの形の run から来たか．"""
    doc = _load(run_dir / "config.json")
    if doc is None:
        legacy = _load(run_dir / "sweep_config.json")
        if legacy is None:
            raise FileNotFoundError(
                f"no settings file in: {run_dir}\n"
                f"  expected: config.json (runvault / legacy run) or sweep_config.json (legacy sweep)"
            )
        return legacy, "sweep"
    if "parameters" in doc:
        meta = runs.load_run_meta(run_dir)
        return doc["parameters"], str(meta["subcommand"])
    return doc, "run"


def _get(cfg: dict, *keys, default="-"):
    """runvault の平坦な parameters と legacy の入れ子 `psi` の両方から引く．"""
    for key in keys:
        cur: object = cfg
        for part in key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                cur = None
                break
            cur = cur[part]
        if cur is not None:
            return cur
    return default


def render_conditions(cfg: dict, run_dir: Path, kind: str) -> str:
    vb = cfg.get("voice_beta", {})
    lw = cfg.get("learning_weights", {})
    # 親 / セル / reproduce は反復シードの元 (`base_seed`) を，反復 1 本の子 run は
    # 自分が実際に使った `seed` を持つ．どちらかしか無いので名前で区別する．
    seed_label = "base seed" if "base_seed" in cfg else "seed"
    lines = [
        "=" * 72,
        f"experiment conditions ({kind})",
        "=" * 72,
        f"run directory: {run_dir}",
        "-" * 72,
    ]
    if kind == "sweep":
        # 親 run (と legacy の sweep) は格子そのものを持つ．
        lines += [
            f"decision_mode      : {_get(cfg, 'decision_mode')}",
            f"n_teams × team     : {_get(cfg, 'n_teams')} × {_get(cfg, 'team_size')}",
            f"α values           : {_get(cfg, 'alpha_values')}",
            f"δ values           : {_get(cfg, 'delta_values')}",
            f"λ                  : {_get(cfg, 'lambda')}",
            f"runs/cell          : {_get(cfg, 'runs')}",
            f"t_max              : {_get(cfg, 't_max')}",
            f"{seed_label:<19}: {_get(cfg, 'base_seed', 'seed')}",
        ]
    else:
        lines += [
            f"decision_mode      : {_get(cfg, 'decision_mode')}",
            f"n_individuals      : {_get(cfg, 'n_individuals')} "
            f"({_get(cfg, 'n_teams')} teams × {_get(cfg, 'team_size')})",
            f"network            : {_get(cfg, 'network_kind')} "
            f"(k={_get(cfg, 'network_k')}, β={_get(cfg, 'network_beta')})",
            "ψ-update λ/α/β/γ/δ : "
            f"{_get(cfg, 'lambda', 'psi.lambda')} / {_get(cfg, 'alpha', 'psi.alpha')} / "
            f"{_get(cfg, 'beta', 'psi.beta')} / {_get(cfg, 'gamma', 'psi.gamma')} / "
            f"{_get(cfg, 'delta', 'psi.delta')}",
            f"voice β0/β_ψ/β_f   : {vb.get('intercept', '-')} / {vb.get('beta_psafety', '-')} / "
            f"{vb.get('beta_fear', '-')}",
            f"learning w_v/w_h/w_e: {lw.get('w_voice', '-')} / {lw.get('w_help', '-')} / "
            f"{lw.get('w_error', '-')}",
            f"γ_L / γ_K / σ_obs  : {_get(cfg, 'gamma_l')} / {_get(cfg, 'gamma_k')} / "
            f"{_get(cfg, 'sigma_obs')}",
            f"knowledge_decay    : {_get(cfg, 'knowledge_decay')}",
            f"p_retaliate        : {_get(cfg, 'p_retaliate')}",
            f"t_max              : {_get(cfg, 't_max')}",
            f"runs (replicates)  : {_get(cfg, 'runs')}",
            f"{seed_label:<19}: {_get(cfg, 'base_seed', 'seed')}",
            f"LLM temp / seed    : {_get(cfg, 'llm_temperature')} / {_get(cfg, 'llm_seed')}",
            f"LLM cache path     : {_get(cfg, 'llm_cache_path')}",
        ]
    lines.append("=" * 72)
    return "\n".join(lines)


def render_identity(meta: dict) -> str:
    """同一性 — 旧 `llm_meta.json` の model / endpoint / temperature はここ．"""
    rng = meta.get("rng") or {}
    llm = meta.get("llm")
    lines = [
        "run identity",
        "-" * 72,
        f"run_uid / slug     : {meta.get('run_uid', '-')} / {meta.get('run_slug', '-')}",
        f"subcommand         : {meta.get('subcommand', '-')}",
        f"config_hash        : {meta.get('config_hash', '-')}",
        f"execution_hash     : {meta.get('execution_hash', '-')}",
        f"master_seed        : {rng.get('master_seed', '-')}",
        f"replicate_index    : {rng.get('replicate_index', '-')}",
    ]
    if llm is not None:
        lines += [
            f"LLM provider/model : {llm.get('provider', '-')} / {llm.get('model_snapshot', '-')}",
            f"LLM temperature    : {llm.get('temperature', '-')}",
        ]
    else:
        # LLM を 1 回も叩かない rule モードの run に «モデル none» を名乗らせない．
        lines.append("LLM                : (none — rule mode makes zero LLM calls)")
    lines.append("=" * 72)
    return "\n".join(lines)


def render_scope_metrics(scoped: dict[str, float]) -> str:
    """run 全体を 1 つの値で表す指標 (旧 `llm_meta.json` の呼び出し数もここ)．"""
    if not scoped:
        return ""
    lines = ["run-scope metrics", "-" * 72]
    for name in sorted(scoped):
        lines.append(f"{name:<19}: {scoped[name]}")
    lines.append("=" * 72)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="edmondson-tools show-experiment-settings",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", "--results_dir", default=None)
    parser.add_argument(
        "--subcommand", default="run",
        help="which subcommand's latest run to show when --results-dir is omitted",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table.")
    args = parser.parse_args(argv)

    run_dir = runs.resolve_run_dir(args.results_dir, subcommand=args.subcommand)
    if not run_dir.exists():
        print(f"error: directory does not exist: {run_dir}", file=sys.stderr)
        return 1

    try:
        cfg, kind = _parameters(run_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    meta = runs.load_run_meta(run_dir, required=False)
    # legacy な wide `metrics.csv` に run スコープの行は無い (step 列すら無い)．
    scoped = runs.scope_metrics(run_dir) if meta is not None else {}

    if args.json:
        payload = {
            "run_dir": str(run_dir),
            "kind": kind,
            "parameters": cfg,
            "run": meta,
            "run_scope_metrics": scoped,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print(render_conditions(cfg, run_dir, kind))
    if meta is not None:
        print(render_identity(meta))
    body = render_scope_metrics(scoped)
    if body:
        print(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
