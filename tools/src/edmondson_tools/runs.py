"""runs.py — どの run を読むか，そして 4 枚の旧 CSV をどう組み直すか．

Rust 側は `run` 1 回を «親 run 1 本 + 反復ごとの子 run» として，`sweep` 1 回を
«親 run 1 本 + セルごとの子 run» として記録する．旧 `results/<timestamp>/` に並んで
いた 4 枚の CSV は，粒度ごとに別の置き場へ移った:

| 旧ファイル | 新しい置き場 |
|---|---|
| `metrics.csv` | 子 run の `metrics.csv` (`scope=run` のステップ指標，long 形式) |
| `teams.csv` | 子 run の `events.jsonl` の `observation` 行 (`unit_id` がチーム) |
| `team_cross_section.csv` | 子 run の `events.jsonl` の `terminal` 行 + run スコープ指標 |
| `individuals.csv` | 子 run の `artifacts/individuals.csv` (表のまま) |
| `sweep_summary.csv` | セル子 run の `terminal` 行 + 子の `parameters` |

このモジュールはその表を読み側で組み直す．旧形式と同じ列を返すので，`visualize` /
`reproduce` の計算そのものは移行前後で 1 セルも変わらない．

run ディレクトリの解決は `runvault path` に任せる — `results/` を走査して新しそうな
ディレクトリを当てにいかない．run_slug には条件と環境のハッシュが入るので，こちらで
名前を組み立てることはできないし，できたとしてもすべきでない．

legacy な `results/<timestamp>/` は `--results-dir` に直接渡せば従来どおり読める．
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from runvault.read import artifacts_dir, load_run_meta, run_subcommand, sweep_children
from runvault.read import runvault_path as _runvault_path

EXPERIMENT = "edmondson-psafety"

#: 子 run を持つ («親» の) サブコマンド．
PARENT_SUBCOMMANDS = ("run", "sweep")


# --------------------------------------------------------------------------- #
# どの run を読むか
# --------------------------------------------------------------------------- #

def resolve_run_dir(
    results_dir: str | os.PathLike | None,
    *,
    subcommand: str = "run",
    results_root: str = "results",
) -> Path:
    """読む run ディレクトリを決める．

    `results_dir` が与えられていればそれを使う (legacy な `results/<timestamp>/` も
    そのまま渡せる)．与えられていなければ `runvault path --latest` に聞く．
    """
    if results_dir is not None:
        path = Path(results_dir)
        # legacy の `results/latest` シンボリックリンクは実体へ解決する．
        if path.is_symlink():
            return Path(os.path.realpath(path))
        return path
    return Path(_runvault_path(EXPERIMENT, results_root, subcommand=subcommand))


def is_runvault_run(run_dir: str | os.PathLike) -> bool:
    """runvault の run ディレクトリか (`run.json` があるか)．"""
    return load_run_meta(run_dir, required=False) is not None


def replicate_dirs(run_dir: str | os.PathLike) -> list[Path]:
    """反復 (またはセル) 1 本ずつの run ディレクトリ．

    親 run を渡すとその子を `replicate_index` の順に，単独の run を渡すとそれ自身
    1 本を返す．legacy なディレクトリもそれ自身 1 本になる (1 枚の CSV に反復が
    まとまっているため)．
    """
    run_dir = Path(run_dir)
    if not is_runvault_run(run_dir):
        return [run_dir]
    if run_subcommand(run_dir) not in PARENT_SUBCOMMANDS:
        return [run_dir]
    children = [Path(c) for c in sweep_children(run_dir)]
    if not children:
        raise SystemExit(
            f"エラー: 親 run に子がありません: {run_dir}\n"
            "  子は lineage.parent_run_uid で親を指します．親と子が同じ"
            " results root にあるか確認してください．"
        )
    return sorted(children, key=replicate_index)


def replicate_index(run_dir: str | os.PathLike) -> int:
    """その run が何本目の反復か．旧 `team_cross_section.csv` の `run` 列にあたる．"""
    meta = load_run_meta(run_dir)
    assert meta is not None  # required=True raises rather than returning None
    return int((meta.get("rng") or {}).get("replicate_index") or 0)


def master_seed(run_dir: str | os.PathLike) -> int:
    """その run を駆動したシード．"""
    meta = load_run_meta(run_dir)
    assert meta is not None
    seed = (meta.get("rng") or {}).get("master_seed")
    if seed is None:
        raise SystemExit(f"エラー: master_seed を持たない run です: {run_dir}")
    return int(seed)


def analysis_output_dir(run_dir: str | os.PathLike, override: str | None) -> Path:
    """図や表の置き場．

    `manifest.csv` は `finish()` が確定させるので，run が終わったあとに作ったものを
    `artifacts/` に置くとハッシュを持たない (＝記録の一部でない) ファイルが混ざる．
    runvault の run には `runvault.read.figures_dir` が示す run ディレクトリの外を使う．
    """
    if override is not None:
        out = Path(override)
    elif is_runvault_run(run_dir):
        from runvault.read import figures_dir

        out = Path(figures_dir(run_dir))
    else:
        out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


# --------------------------------------------------------------------------- #
# metrics.csv
# --------------------------------------------------------------------------- #

def _metrics_wide(metrics_path: Path) -> pd.DataFrame:
    """long 形式の `metrics.csv` を «1 ステップ 1 行» に戻す．

    `runvault.read.metrics_wide` と同じ形を返すが，読み込みに
    `float_precision="round_trip"` を渡す点だけが違う．pandas の既定の C パーサは
    f64 を 1 ULP 落とすことがあり (例: `0.26666666666666666` → `0.2666666666666666`)，
    移行前後の値の照合が «実際には一致しているのに一致しない» と出る．
    """
    df = pd.read_csv(metrics_path, float_precision="round_trip")
    stepped = df[df["step"].notna()]
    return (
        stepped.pivot_table(index="step", columns="name", values="value", aggfunc="last")
        .reset_index()
        .rename_axis(None, axis=1)
        .astype({"step": int})
        .sort_values("step")
        .reset_index(drop=True)
    )


def scope_metrics(run_dir: str | os.PathLike) -> dict[str, float]:
    """step を持たない run スコープの指標．

    `runvault.read.run_scope_metrics` と同じものを返すが，`float_precision=
    "round_trip"` で読む点だけが違う (`_metrics_wide` と同じ理由)．
    """
    run_dir = Path(run_dir)
    # 親 run は指標を書かない (条件と反復リストを宣言するだけ)．無いことは答えである．
    if not (run_dir / "metrics.csv").exists():
        return {}
    df = pd.read_csv(run_dir / "metrics.csv", float_precision="round_trip")
    rows = df[df["step"].isna()]
    return {str(r["name"]): float(r["value"]) for _, r in rows.iterrows()}


def replicate_metrics(run_dir: str | os.PathLike) -> pd.DataFrame:
    """反復 1 本のステップ指標．旧 `metrics.csv` と同じ列 (`t` + 5 指標) を返す．"""
    run_dir = Path(run_dir)
    if not is_runvault_run(run_dir):
        return pd.read_csv(run_dir / "metrics.csv", float_precision="round_trip")
    return _metrics_wide(run_dir / "metrics.csv").rename(columns={"step": "t"})


def pooled_metrics(run_dir: str | os.PathLike) -> pd.DataFrame:
    """反復をプールしたステップ指標．`seed` 列で反復を区別する．"""
    run_dir = Path(run_dir)
    if not is_runvault_run(run_dir):
        return pd.read_csv(run_dir / "metrics.csv", float_precision="round_trip")
    frames = []
    for child in replicate_dirs(run_dir):
        wide = replicate_metrics(child)
        wide.insert(0, "seed", master_seed(child))
        frames.append(wide)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# events.jsonl
# --------------------------------------------------------------------------- #

def _events(run_dir: Path, kind: str) -> pd.DataFrame:
    """`events.jsonl` の 1 種別．

    `runvault.read.events_table` と同じものを返すが，dict → DataFrame の際に
    int64 を超える値 (シード) が float64 へ降格するのを避けるため，自分で読む．
    """
    path = run_dir / "events.jsonl"
    if not path.exists():
        raise SystemExit(f"エラー: events.jsonl がありません: {path}")
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") == kind:
                rows.append(row)
    if not rows:
        raise SystemExit(f"エラー: {path} に schema={kind} の行がありません．")
    df = pd.DataFrame(rows)
    if "seed" in df.columns:
        df["seed"] = pd.array([r["seed"] for r in rows], dtype="UInt64")
    return df


TEAM_PANEL_COLUMNS = ["t", "team_id", "psi", "learning", "performance", "efficacy", "support", "coaching"]


def teams_panel(run_dir: str | os.PathLike) -> pd.DataFrame:
    """反復 1 本のチームパネル．旧 `teams.csv` と同じ列を返す．"""
    run_dir = Path(run_dir)
    if not is_runvault_run(run_dir):
        return pd.read_csv(run_dir / "teams.csv", float_precision="round_trip")
    df = _events(run_dir, "observation")
    df["team_id"] = df["unit_id"].str.removeprefix("team-").astype(int)
    return (
        df[TEAM_PANEL_COLUMNS]
        .sort_values(["t", "team_id"])
        .reset_index(drop=True)
    )


CROSS_SECTION_COLUMNS = [
    "run", "team_id", "psi", "learning", "performance", "support", "efficacy",
    "icc_psi", "icc_learning",
]


def cross_section(run_dir: str | os.PathLike) -> pd.DataFrame:
    """反復をプールしたチーム断面．旧 `team_cross_section.csv` と同じ列を返す．

    1 チーム 1 行の値は子 run の `terminal` 行がそのまま持つ (Rust が計算した数を
    読むだけで，後半平均をこちらで計算し直さない — 同じ集約を Python と Rust の
    2 箇所に置くと食い違う余地ができる)．run 内で一定の 2 列は指標から採る:
    `icc_psi` は最終ステップのステップ指標，`icc_learning` は run スコープの
    `icc_learning_stable` である．
    """
    run_dir = Path(run_dir)
    if not is_runvault_run(run_dir):
        return pd.read_csv(run_dir / "team_cross_section.csv", float_precision="round_trip")
    frames = []
    for child in replicate_dirs(run_dir):
        df = _events(child, "terminal")
        df["team_id"] = df["unit_id"].str.removeprefix("team-").astype(int)
        df["run"] = replicate_index(child)
        wide = replicate_metrics(child)
        df["icc_psi"] = float(wide.loc[wide["t"].idxmax(), "icc_psi"])
        df["icc_learning"] = scope_metrics(child)["icc_learning_stable"]
        frames.append(df[CROSS_SECTION_COLUMNS].sort_values("team_id"))
    return pd.concat(frames, ignore_index=True)


SWEEP_COLUMNS = [
    "decision_mode", "alpha", "delta", "lambda", "run", "seed", "icc_psi", "beta_psi_l",
    "r2_psi_l", "r2_l_pi", "mediation_ratio", "beta_support_psi", "efficacy_t",
    "hypotheses_supported",
]


def sweep_trials(sweep_dir: str | os.PathLike) -> pd.DataFrame:
    """セルの試行を «1 試行 1 行» に戻す．旧 `sweep_summary.csv` と同じ列を返す．

    条件の列 (`decision_mode` / `alpha` / `delta` / `lambda`) は子 run の
    `parameters` から，試行ごとの値は `terminal` 行から来る．`run` 列 (試行の番号)
    は `unit_id` が持っているので，イベントに数として書き足していない．
    """
    sweep_dir = Path(sweep_dir)
    if not is_runvault_run(sweep_dir):
        return pd.read_csv(sweep_dir / "sweep_summary.csv", float_precision="round_trip")
    frames = []
    for cell in replicate_dirs(sweep_dir):
        with (cell / "config.json").open(encoding="utf-8") as f:
            params = json.load(f)["parameters"]
        df = _events(cell, "terminal")
        df["run"] = df["unit_id"].str.removeprefix("trial-").astype(int)
        df = df.rename(columns={"efficacy_partial_t": "efficacy_t"})
        for key in ("decision_mode", "alpha", "delta", "lambda"):
            df[key] = params[key]
        frames.append(df[SWEEP_COLUMNS].sort_values("run"))
    return pd.concat(frames, ignore_index=True)


def individual_panel_path(run_dir: str | os.PathLike) -> Path:
    """反復 1 本の個人パネル (`individuals.csv`) の場所．"""
    run_dir = Path(run_dir)
    if not is_runvault_run(run_dir):
        return run_dir / "individuals.csv"
    return Path(artifacts_dir(run_dir)) / "individuals.csv"
