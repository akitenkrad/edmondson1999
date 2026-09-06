[English](visualization.md) | **日本語**

# 可視化

すべてのツールは読む run を `runvault path --latest` に聞く (`--results-dir` で上書きでき，
legacy な `results/<timestamp>/` もそのまま渡せる)．出力先は
`<results-root>/<experiment>/figures/<run_slug>/` — `manifest.csv` は run の終了時に確定する
ので，あとから描いた図を run ディレクトリに置くとハッシュを持たないファイルが記録に混ざる．
`--output-dir` でリダイレクトできる．

## `edmondson-tools visualize` (単一実行)

反復 1 本のチームパネル (`observation` イベント) とステップ指標から．既定は **最後の**
反復で，これは runvault 以前のレイアウトが残せていたものに合わせてある (`--replicate N` で
選べる)．反復をまたいでチームをプールしない — 回帰の検出力が実験計画より大きくなり，
`reproduce` が避けている過大評価と同じことが起きるからである．

- `psi_learning_perf_timeseries.png` — チーム平均 ψ̄ / L / Π の時系列 (因果連鎖がともに上昇)．
- `mediation_scatter.png` — 実行後半のチーム横断データの 2 パネル: ψ̄ → L と L → Π．各々に推定傾き B と相関 r を注記．
- `icc_trace.png` — ICC(ψ) / ICC(L) スナップショットと媒介比率の時系列．ICC(ψ) ≈ .39 アンカーを表示．

## `edmondson-tools visualize-sweep`

セル子 run の `terminal` 行 (1 試行 1 行．条件の列は子の `parameters` から) から α × δ の 3 つのヒートマップ:

- `sweep_icc_heatmap.png` — 平均 ICC(ψ) (アンカー .39)．
- `sweep_mediation_heatmap.png` — 平均媒介比率 (≥ .5 基準)．
- `sweep_r2_heatmap.png` — 平均 ψ→L 調整 R² (アンカー .63)．

## `edmondson-tools show-experiment-settings`

run の実験条件 (`config.json` の `parameters`)・同一性 (`run.json` の `run_uid` /
`config_hash` / `master_seed` / `replicate_index` / `llm` ブロック)・run スコープ指標を
整形表示する．`--subcommand` でどのサブコマンドの最新 run を見るかを選び，`--json` で
3 つをまとめて JSON 出力する．legacy な `config.json` / `sweep_config.json` のディレクトリも
そのまま表示できる．

## `edmondson-tools reproduce`

[再現](reproduction.ja.md) を参照 — Table 4-8 風 Baron & Kenny レポートとブートストラップ媒介 CI．
