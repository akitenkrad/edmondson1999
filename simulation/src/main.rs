//! Edmondson (1999) — Psychological Safety & Team Learning CLI.
//!
//! `run`       : single configuration; `--decision-mode {rule|llm}`.
//! `sweep`     : Cartesian product over ψ-update params × seeds; one child run per cell.
//! `reproduce` : team-level cross-section against the §5 calibration anchors.
//!
//! 出力の置き場と同一性は runvault が持つ．タイムスタンプ付きディレクトリも
//! `latest` シンボリックリンクもこちらでは作らず，`Run::start` が決めた run
//! ディレクトリへ書く．
//!
//! `run` は «条件 + その反復» なので，親 run が反復リストを宣言し，反復 1 本ずつが
//! 子 run になる．`sweep` はセル 1 つが子 run で，そのセルの試行は `events.jsonl` の
//! `terminal` 行になる．`reproduce` は試行しか見ないので run 1 本で足りる．
//! どこに何を置いたかは `edmondson_team::record` の冒頭にまとめてある．

use std::fs;
use std::path::Path;

use clap::{Parser, Subcommand};
use runvault::{Lineage, Run, RunOptions, Stage};
use serde::Serialize;

use edmondson_team::config::{
    parse_decision_mode, parse_network_kind, Config, LearningWeights, LlmSettings, NetworkKind,
    PsiParams, VoiceBeta,
};
use edmondson_team::llm::{build_live_client, VoiceClient};
use edmondson_team::record::{
    self, ConditionParameters, ReplicateGroupParameters, ReplicateParameters, DOMAIN, EXPERIMENT,
    GROUP_SEED_POINTERS, HASH_EXCLUDE, REPLICATE_SEED_POINTERS, REPO_ID,
};
use edmondson_team::simulation::{
    anchor_report_from_result, run_with_client_observed, save_individuals, AnchorReport,
    SimulationResult,
};

use socsim_llm::LlmClient;

// --------------------------------------------------------------------------- //
// CLI
// --------------------------------------------------------------------------- //

#[derive(Parser, Debug)]
#[command(
    name = "edmondson",
    about = "Edmondson (1999) — Psychological Safety & Team Learning (support/coaching → ψ → L → Π)"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    /// Development run: write it under results/_scratch/ so it is never synced to the vault.
    #[arg(long, global = true)]
    scratch: bool,
    /// Ollama 接続先 URL（指定時は環境変数 OLLAMA_HOST を上書きする）．
    #[arg(long, global = true)]
    ollama_host: Option<String>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run a single configuration; one child run per replicate.
    Run(RunArgs),
    /// Sweep ψ-update parameters across seeds; one child run per cell.
    Sweep(SweepArgs),
    /// Team-level cross-section against the design's §5 anchors.
    Reproduce(ReproduceArgs),
}

#[derive(Parser, Debug)]
struct RunArgs {
    /// Decision mechanism (rule / llm).
    #[arg(long, default_value = "rule")]
    decision_mode: String,
    /// Number of teams.
    #[arg(long, default_value_t = 90)]
    n_teams: usize,
    /// Individuals per team.
    #[arg(long, default_value_t = 8)]
    team_size: usize,
    /// Within-team network family.
    #[arg(long, default_value = "watts-strogatz")]
    network_model: String,
    /// Watts–Strogatz `k`.
    #[arg(long, default_value_t = 6)]
    network_k: usize,
    /// Watts–Strogatz β / Erdős–Rényi p.
    #[arg(long, default_value_t = 0.15)]
    network_beta: f64,
    /// ψ-update learning rate λ.
    #[arg(long, default_value_t = 0.10)]
    lambda: f64,
    /// ψ-update support weight α.
    #[arg(long, default_value_t = 0.30)]
    alpha: f64,
    /// ψ-update coaching weight β.
    #[arg(long, default_value_t = 0.25)]
    beta: f64,
    /// ψ-update retaliation-shock weight γ.
    #[arg(long, default_value_t = 0.50)]
    gamma: f64,
    /// ψ-update shared-belief convergence weight δ.
    #[arg(long, default_value_t = 0.35)]
    delta: f64,
    /// Observer-rating noise sd σ_obs.
    #[arg(long, default_value_t = 0.22)]
    sigma_obs: f64,
    /// Maximum simulation step.
    #[arg(long, default_value_t = 24)]
    t_max: u64,
    /// Number of independent replicates (one child run each).
    #[arg(long, default_value_t = 30)]
    runs: usize,
    /// Base random seed (per-replicate seeds are derived from it).
    #[arg(long, default_value_t = 1999)]
    seed: u64,
    /// LLM generation temperature.
    #[arg(long, default_value_t = 0.0)]
    llm_temperature: f32,
    /// LLM generation seed (offset; per-(agent, t) seed derived from it).
    #[arg(long, default_value_t = 0)]
    llm_seed: u64,
    /// Prompt → response cache path (LLM mode only).
    #[arg(long, default_value = ".llm_cache/cache.json")]
    cache_path: String,
    /// runvault results root.
    #[arg(long, default_value = "results")]
    output_dir: String,
}

#[derive(Parser, Debug)]
struct SweepArgs {
    /// Decision mode (the parameter sweep is meaningful only for rule).
    #[arg(long, default_value = "rule")]
    decision_mode: String,
    /// Number of teams.
    #[arg(long, default_value_t = 90)]
    n_teams: usize,
    /// Individuals per team.
    #[arg(long, default_value_t = 8)]
    team_size: usize,
    /// α sweep min / max / step.
    #[arg(long, default_value_t = 0.10)]
    alpha_min: f64,
    #[arg(long, default_value_t = 0.50)]
    alpha_max: f64,
    #[arg(long, default_value_t = 0.10)]
    alpha_step: f64,
    /// δ sweep min / max / step.
    #[arg(long, default_value_t = 0.10)]
    delta_min: f64,
    #[arg(long, default_value_t = 0.60)]
    delta_max: f64,
    #[arg(long, default_value_t = 0.10)]
    delta_step: f64,
    /// λ (held fixed across the sweep unless overridden).
    #[arg(long, default_value_t = 0.10)]
    lambda: f64,
    /// Runs (seeds) per cell.
    #[arg(long, default_value_t = 30)]
    runs: usize,
    /// Maximum simulation step.
    #[arg(long, default_value_t = 24)]
    t_max: u64,
    /// Base seed.
    #[arg(long, default_value_t = 1999)]
    seed: u64,
    /// runvault results root.
    #[arg(long, default_value = "results")]
    output_dir: String,
}

#[derive(Parser, Debug)]
struct ReproduceArgs {
    /// Decision mode to report.
    #[arg(long, default_value = "rule")]
    decision_mode: String,
    /// Number of teams.
    #[arg(long, default_value_t = 90)]
    n_teams: usize,
    /// Individuals per team.
    #[arg(long, default_value_t = 8)]
    team_size: usize,
    /// Maximum simulation step.
    #[arg(long, default_value_t = 24)]
    t_max: u64,
    /// Base seed.
    #[arg(long, default_value_t = 1999)]
    seed: u64,
    /// Trials (pooled into the cross-section).
    #[arg(long, default_value_t = 30)]
    runs: usize,
    /// runvault results root.
    #[arg(long, default_value = "results")]
    output_dir: String,
}

// --------------------------------------------------------------------------- //
// sweep parent parameters
// --------------------------------------------------------------------------- //

/// The sweep parent's own conditions: the grid definition itself.
///
/// 個別セルの条件は子 run が持つ．親はどの格子を掃いたかだけを宣言する．
#[derive(Serialize)]
struct SweepParameters {
    decision_mode: &'static str,
    n_teams: usize,
    team_size: usize,
    alpha_values: Vec<f64>,
    delta_values: Vec<f64>,
    lambda: f64,
    runs: usize,
    t_max: u64,
    base_seed: u64,
}

// --------------------------------------------------------------------------- //
// helpers
// --------------------------------------------------------------------------- //

fn frange(min: f64, max: f64, step: f64) -> Vec<f64> {
    let mut out = Vec::new();
    let mut v = min;
    while v <= max + 1e-9 {
        out.push((v * 1000.0).round() / 1000.0);
        v += step.max(1e-6);
    }
    out
}

fn cfg_from_run_args(args: &RunArgs) -> Config {
    Config {
        n_teams: args.n_teams,
        team_size: args.team_size,
        network_kind: parse_network_kind(&args.network_model).unwrap_or(NetworkKind::WattsStrogatz),
        network_k: args.network_k,
        network_beta: args.network_beta,
        decision_mode: parse_decision_mode(&args.decision_mode).unwrap_or_else(|e| panic!("{e}")),
        psi: PsiParams {
            lambda: args.lambda,
            alpha: args.alpha,
            beta: args.beta,
            gamma: args.gamma,
            delta: args.delta,
        },
        voice_beta: VoiceBeta::default(),
        learning_weights: LearningWeights::default(),
        sigma_obs: args.sigma_obs,
        t_max: args.t_max,
        runs: args.runs,
        seed: args.seed,
        llm: LlmSettings {
            temperature: args.llm_temperature,
            seed: args.llm_seed,
            cache_path: Some(args.cache_path.clone()),
        },
        output_dir: args.output_dir.clone(),
        ..Config::default()
    }
}

/// Mean of a slice (0 on empty).
fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

/// LLM モードなら本番クライアントを組み立てる．rule モードでは `None`．
///
/// `Run::start` より前に組み立てるのは，`llm` ブロックに書く model / endpoint を
/// クライアント自身から採るためである (名前を推測で書かない)．
fn build_client(cfg: &Config) -> Option<VoiceClient> {
    if !cfg.decision_mode.is_llm() {
        return None;
    }
    Some(build_live_client(&cfg.llm).unwrap_or_else(|e| panic!("LLM client build failed: {e}")))
}

/// LLM キャッシュの置き場を用意する (LLM モードのみ)．
fn ensure_cache_dir(cfg: &Config, cache_path: &str) {
    if cfg.decision_mode.is_llm() {
        if let Some(parent) = Path::new(cache_path).parent() {
            let _ = fs::create_dir_all(parent);
        }
    }
}

/// 試行 1 本を回す (`sweep` / `reproduce`)．run ディレクトリは作らない．
///
/// 進捗の 1 単位は 1 ステップ．費用がそこにあるからで，1 ステップは全チームの
/// 全メンバーについて決定を出し，LLM モードではその 1 つ 1 つがモデル呼び出しに
/// なる．試行を単位にすると，ライブの 1 本は 0/1 と出したきり終わりまで黙る．
/// `stage` は呼び出し側が開ける — コマンド全体で 1 つにすることで，条件をまたいでも
/// 割合が途中で 100% に戻らない．
fn run_trial(cfg: &Config, stage: &mut Stage) -> SimulationResult {
    let client = build_client(cfg);
    run_with_client_observed(cfg, client, |_| stage.tick())
        .unwrap_or_else(|e| panic!("trial run failed: {e}"))
}

// --------------------------------------------------------------------------- //
// run
// --------------------------------------------------------------------------- //

/// 反復 1 本を子 run として回し，記録する．
fn run_replicate(
    results_root: &str,
    cfg: &Config,
    seed: u64,
    replicate_index: usize,
    lineage: &Lineage,
    stage: &mut Stage,
    scratch: bool,
) -> SimulationResult {
    let client = build_client(cfg);
    let llm = client
        .as_ref()
        .map(|c| record::llm_block(c.inner().model(), c.inner().endpoint(), cfg.llm.temperature));

    let parameters = ReplicateParameters {
        condition: ConditionParameters::from_config(cfg),
        seed,
    };

    let mut options = RunOptions::new(EXPERIMENT, "run-replicate")
        .scratch(scratch)
        .repo_id(REPO_ID)
        .domain(DOMAIN)
        .results_root(results_root)
        .parameters(&parameters)
        .expect("runvault: parameters の組み立てに失敗")
        .hash_exclude(HASH_EXCLUDE)
        .seed_pointers(REPLICATE_SEED_POINTERS)
        .master_seed(seed)
        .replicate_index(replicate_index as u64)
        .lineage(lineage.clone())
        .replication(record::replication());
    if let Some(llm) = llm {
        options = options.llm(llm);
    }

    let mut child = Run::start(options).expect("runvault: 子 run の開始に失敗");

    let result = run_with_client_observed(cfg, client, |_| stage.tick())
        .unwrap_or_else(|e| panic!("replicate run failed: {e}"));

    record::log_replicate(&mut child, cfg, &result);
    if cfg.decision_mode.is_llm() {
        record::log_llm_usage(&mut child, &result);
    }
    save_individuals(
        &result.individual_rows,
        &child.dir().join("artifacts").to_string_lossy(),
    );
    child.finish().expect("runvault: 子 run の完了に失敗");

    result
}

fn cmd_run(args: RunArgs, scratch: bool) {
    let base_cfg = cfg_from_run_args(&args);
    ensure_cache_dir(&base_cfg, &args.cache_path);
    let runs = base_cfg.runs.max(1);

    // 親 run: 条件と反復リストを宣言するだけで，シミュレーションは回さない．
    // 反復ごとの派生シードで駆動されるので単一の master_seed は名乗らない
    // (base seed は /base_seed と seed_pointers 経由で execution_hash に残る)．
    let parent = Run::start(
        RunOptions::new(EXPERIMENT, "run")
            .scratch(scratch)
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&args.output_dir)
            .parameters(&ReplicateGroupParameters {
                condition: ConditionParameters::from_config(&base_cfg),
                runs,
                base_seed: base_cfg.seed,
            })
            .expect("runvault: parameters の組み立てに失敗")
            .hash_exclude(HASH_EXCLUDE)
            .seed_pointers(GROUP_SEED_POINTERS)
            .sweep_parent()
            .replication(record::replication()),
    )
    .expect("runvault: 親 run の開始に失敗");

    let lineage = Lineage {
        sweep_id: parent.sweep_id().map(str::to_string),
        parent_run_uid: Some(parent.run_uid().to_string()),
        ..Default::default()
    };

    println!("=== Edmondson (1999) — Psychological Safety & Team Learning ===");
    println!(
        "decision-mode: {} | teams: {}×{} (={}) | network: {:?} k={} β={:.2}",
        base_cfg.decision_mode.label(),
        base_cfg.n_teams,
        base_cfg.team_size,
        base_cfg.n_individuals(),
        base_cfg.network_kind,
        base_cfg.network_k,
        base_cfg.network_beta,
    );
    println!(
        "ψ-update: λ={:.2} α={:.2} β={:.2} γ={:.2} δ={:.2} | σ_obs={:.2} | t_max={} runs={} seed={}",
        base_cfg.psi.lambda,
        base_cfg.psi.alpha,
        base_cfg.psi.beta,
        base_cfg.psi.gamma,
        base_cfg.psi.delta,
        base_cfg.sigma_obs,
        base_cfg.t_max,
        runs,
        base_cfg.seed,
    );
    println!("output: {}", parent.dir().display());
    println!("----------------------------------------------------------------------");

    let (mut pooled_icc, mut pooled_beta_psi_l, mut pooled_r2, mut pooled_med) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut last: Option<SimulationResult> = None;
    // 親 run に stage を 1 つ．反復はすべて同じ条件・同じ t_max なので重みでは
    // なく数える．子 run ごとに開け直すと小さな 100% が並ぶだけになる．
    let mut stage = parent.stage("steps", runs * base_cfg.t_max as usize);
    for run_idx in 0..runs {
        let seed = record::replicate_seed(base_cfg.seed, run_idx);
        let cfg = Config {
            seed,
            ..base_cfg.clone()
        };
        let result = run_replicate(
            &args.output_dir,
            &cfg,
            seed,
            run_idx,
            &lineage,
            &mut stage,
            scratch,
        );
        let rep = anchor_report_from_result(&result);
        pooled_icc.push(rep.icc_psi);
        pooled_beta_psi_l.push(rep.beta_psi_l);
        pooled_r2.push(rep.r2_psi_l);
        pooled_med.push(rep.mediation_ratio);
        if run_idx + 1 == runs || runs <= 5 {
            println!(
                "[{}/{}] seed={} icc_ψ={:.3} ψ→L B={:.3} R²={:.3} med_ratio={:.3}",
                run_idx + 1,
                runs,
                seed,
                rep.icc_psi,
                rep.beta_psi_l,
                rep.r2_psi_l,
                rep.mediation_ratio,
            );
        }
        last = Some(result);
    }
    // manifest.csv は finish() で封をされる．その後に 1 行足せば，manifest が
    // 食い違うダイジェストを持つことになる．
    stage.close();

    let dir = parent.finish().expect("runvault: 親 run の完了に失敗");

    println!("----------------------------------------------------------------------");
    println!(
        "pooled over {} replicates: icc_ψ={:.3} ψ→L B={:.3} R²={:.3} med_ratio={:.3}",
        runs,
        mean(&pooled_icc),
        mean(&pooled_beta_psi_l),
        mean(&pooled_r2),
        mean(&pooled_med),
    );
    if let Some(result) = &last {
        if base_cfg.decision_mode.is_llm() {
            println!(
                "LLM calls (最後の反復): {} | cache-hit: {} ({:.1}%) | model: {}",
                result.metadata.total(),
                result.metadata.cache_hits(),
                result.metadata.cache_hit_rate() * 100.0,
                result.llm_model,
            );
        }
    }
    println!("親 run   → {}", dir.display());
    println!(
        "反復 {runs} 本 → 子 run (subcommand=run-replicate)．metrics.csv がステップごとの時系列，\
         events.jsonl の observation がチームのパネル，terminal がチームの断面，\
         artifacts/individuals.csv が個人パネル．"
    );
}

// --------------------------------------------------------------------------- //
// sweep
// --------------------------------------------------------------------------- //

fn cmd_sweep(args: SweepArgs, scratch: bool) {
    let mode = parse_decision_mode(&args.decision_mode).unwrap_or_else(|e| panic!("{e}"));

    let alphas = frange(args.alpha_min, args.alpha_max, args.alpha_step);
    let deltas = frange(args.delta_min, args.delta_max, args.delta_step);
    let n_cells = alphas.len() * deltas.len();
    let n_total = n_cells * args.runs;

    // 親 run: グリッド定義そのものを parameters に持つ．個別セルの指標は書かない．
    let parent = Run::start(
        RunOptions::new(EXPERIMENT, "sweep")
            .scratch(scratch)
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&args.output_dir)
            .parameters(&SweepParameters {
                decision_mode: mode.label(),
                n_teams: args.n_teams,
                team_size: args.team_size,
                alpha_values: alphas.clone(),
                delta_values: deltas.clone(),
                lambda: args.lambda,
                runs: args.runs,
                t_max: args.t_max,
                base_seed: args.seed,
            })
            .expect("runvault: sweep の parameters の組み立てに失敗")
            .seed_pointers(GROUP_SEED_POINTERS)
            .sweep_parent()
            .replication(record::replication()),
    )
    .expect("runvault: sweep 親 run の開始に失敗");

    let lineage = Lineage {
        sweep_id: parent.sweep_id().map(str::to_string),
        parent_run_uid: Some(parent.run_uid().to_string()),
        ..Default::default()
    };

    println!("=== edmondson-sweep ===");
    println!(
        "mode: {} | α={:?} | δ={:?} | λ={:.2} | runs/cell={} | total {} runs",
        mode.label(),
        alphas,
        deltas,
        args.lambda,
        args.runs,
        n_total,
    );
    println!("base seed: {}", args.seed);
    println!("output: {}", parent.dir().display());
    println!("------------------------------------------------------------");

    let mut idx = 0usize;
    // グリッド全体で stage を 1 つ．掃引しているのは α と δ の係数で，どちらも
    // 仕事の量を変えない (チーム数も t_max も固定) ので，重みではなく数える．
    let mut stage = parent.stage("steps", n_total * args.t_max as usize);

    for &alpha in &alphas {
        for &delta in &deltas {
            let cell_cfg = Config {
                n_teams: args.n_teams,
                team_size: args.team_size,
                decision_mode: mode,
                psi: PsiParams {
                    alpha,
                    delta,
                    lambda: args.lambda,
                    ..PsiParams::default()
                },
                t_max: args.t_max,
                runs: args.runs,
                seed: args.seed,
                ..Config::default()
            };

            // 子は «そのセルの試行群» そのもの．base seed とセル座標からすべての
            // 試行シードが決まるので master_seed は base seed であり，同一セルの
            // 繰り返しは無いので replicate_index は 0．
            let mut child = Run::start(
                RunOptions::new(EXPERIMENT, "sweep-point")
                    .scratch(scratch)
                    .repo_id(REPO_ID)
                    .domain(DOMAIN)
                    .results_root(&args.output_dir)
                    .parameters(&ReplicateGroupParameters {
                        condition: ConditionParameters::from_config(&cell_cfg),
                        runs: args.runs,
                        base_seed: args.seed,
                    })
                    .expect("runvault: 子 run の parameters の組み立てに失敗")
                    .hash_exclude(HASH_EXCLUDE)
                    .seed_pointers(GROUP_SEED_POINTERS)
                    .master_seed(args.seed)
                    .replicate_index(0)
                    .lineage(lineage.clone())
                    .replication(record::replication()),
            )
            .expect("runvault: sweep 子 run の開始に失敗");

            let mut reps: Vec<AnchorReport> = Vec::with_capacity(args.runs);
            for run_idx in 0..args.runs {
                idx += 1;
                let seed = record::trial_seed(args.seed, alpha, delta, run_idx);
                let cfg = Config {
                    seed,
                    runs: 1,
                    ..cell_cfg.clone()
                };
                let result = run_trial(&cfg, &mut stage);
                let rep = anchor_report_from_result(&result);
                record::log_trial(
                    &mut child,
                    run_idx,
                    seed,
                    result.final_round,
                    args.t_max,
                    &rep,
                );
                if idx.is_multiple_of(20) || idx == n_total {
                    println!(
                        "[{}/{}] α={:.2} δ={:.2} run={} icc_ψ={:.3} med={:.3}",
                        idx, n_total, alpha, delta, run_idx, rep.icc_psi, rep.mediation_ratio
                    );
                }
                reps.push(rep);
            }
            record::log_cell_summary(&mut child, &reps);
            child.finish().expect("runvault: sweep 子 run の完了に失敗");
        }
    }

    stage.close();

    let dir = parent
        .finish()
        .expect("runvault: sweep 親 run の完了に失敗");
    println!("------------------------------------------------------------");
    println!("sweep done.");
    println!("親 run     → {}", dir.display());
    println!("セル {n_cells} 個 → 子 run (subcommand=sweep-point)．試行 1 本が events.jsonl の terminal 行 1 本．");
}

// --------------------------------------------------------------------------- //
// reproduce
// --------------------------------------------------------------------------- //

/// 帯は原著の数ではなくこちらが決めた許容幅なので，`reference.csv` にも指標にも
/// 入れずコンソールに残す．
fn band(name: &str, value: f64, lo: f64, hi: f64) -> String {
    let ok = value >= lo && value <= hi;
    format!(
        "  {name:<22} = {value:>7.3}   ([{lo:.2}, {hi:.2}]: {})",
        if ok { "PASS" } else { "off-anchor" }
    )
}

fn cmd_reproduce(args: ReproduceArgs, scratch: bool) {
    let mode = parse_decision_mode(&args.decision_mode).unwrap_or_else(|e| panic!("{e}"));
    let runs = args.runs.max(1);

    let base_cfg = Config {
        n_teams: args.n_teams,
        team_size: args.team_size,
        decision_mode: mode,
        t_max: args.t_max,
        runs,
        seed: args.seed,
        output_dir: args.output_dir.clone(),
        ..Config::default()
    };

    // 試行は自分の時系列を残さない (各試行の断面統計しか見ない) ので，子 run には
    // 割らず run 1 本の terminal 行 1 本ずつにする．
    let mut run = Run::start(
        RunOptions::new(EXPERIMENT, "reproduce")
            .scratch(scratch)
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&args.output_dir)
            .parameters(&ReplicateGroupParameters {
                condition: ConditionParameters::from_config(&base_cfg),
                runs,
                base_seed: args.seed,
            })
            .expect("runvault: reproduce の parameters の組み立てに失敗")
            .hash_exclude(HASH_EXCLUDE)
            .seed_pointers(GROUP_SEED_POINTERS)
            .master_seed(args.seed)
            .replication(record::replication()),
    )
    .expect("runvault: reproduce の run の開始に失敗");

    println!("=== edmondson-reproduce ({} mode) ===", mode.label());
    println!("output: {}", run.dir().display());

    // Each independent trial yields one set of anchor statistics on its own
    // team cross-section (n_teams ≈ the paper's 51); we report the mean across
    // trials, matching the §6 "30 trials, average ± 95% CI" plan. Pooling all
    // teams into one giant regression would inflate the statistical power far
    // beyond the paper's design (and make every effect spuriously significant).
    let mut reps = Vec::with_capacity(runs);
    let mut stage = run.stage("steps", runs * args.t_max as usize);
    for run_idx in 0..runs {
        let seed = record::replicate_seed(args.seed, run_idx);
        let cfg = Config {
            seed,
            runs: 1,
            ..base_cfg.clone()
        };
        let result = run_trial(&cfg, &mut stage);
        let rep = anchor_report_from_result(&result);
        record::log_trial(
            &mut run,
            run_idx,
            seed,
            result.final_round,
            args.t_max,
            &rep,
        );
        reps.push(rep);
    }
    stage.close();
    record::log_reproduce_summary(&mut run, &reps);
    record::log_paper_reference(&mut run);
    let rep = record::mean_report(&reps);
    let dir = run
        .finish()
        .expect("runvault: reproduce の run の完了に失敗");

    println!(
        "per-trial team cross-section averaged over {} runs ({} teams/trial):",
        runs, args.n_teams,
    );
    println!("{}", band("ICC(ψ)", rep.icc_psi, 0.25, 0.55));
    println!("{}", band("ICC(L)", rep.icc_learning, 0.15, 0.40));
    println!("{}", band("ψ→L  B", rep.beta_psi_l, 0.50, 1.00));
    println!("{}", band("ψ→L  R²", rep.r2_psi_l, 0.48, 0.78));
    println!("{}", band("L→Π  R²", rep.r2_l_pi, 0.16, 0.36));
    println!("{}", band("support→ψ B", rep.beta_support_psi, 0.35, 0.77));
    println!(
        "  {:<22} = {:>7.3}   (mediation residual p>.10: {})",
        "ψ residual p",
        rep.beta_psi_residual_p,
        if rep.beta_psi_residual_p > 0.10 {
            "PASS"
        } else {
            "review"
        }
    );
    println!(
        "  {:<22} = {:>7.3}   (mediation ratio ≥ .5: {})",
        "mediation ratio",
        rep.mediation_ratio,
        if rep.mediation_ratio >= 0.5 {
            "PASS"
        } else {
            "review"
        }
    );
    println!(
        "  {:<22} = {:>7.3}   (efficacy |t|<2, H5/H8 unsupported: {})",
        "efficacy |t|",
        rep.efficacy_partial_t.abs(),
        if rep.efficacy_partial_t.abs() < 2.0 {
            "PASS"
        } else {
            "review"
        }
    );
    println!(
        "  {:<22} = {:>5}/8   (≥ 5/8: {})",
        "hypotheses supported",
        rep.hypotheses_supported,
        if rep.hypotheses_supported >= 5 {
            "PASS"
        } else {
            "review"
        }
    );
    println!();
    println!("run → {}", dir.display());
    println!("試行 {runs} 本が events.jsonl の terminal 行．原著の報告値は reference.csv．");
    println!();
    println!("For the full Table 4-8-style Baron & Kenny report + bootstrap mediation");
    println!("95% BC CI + the efficacy discriminant, run the Python tool:");
    println!("  uv run edmondson-tools reproduce");
}

// --------------------------------------------------------------------------- //
// main
// --------------------------------------------------------------------------- //

fn main() {
    let cli = Cli::parse();
    let scratch = cli.scratch;
    if let Some(host) = cli.ollama_host.as_deref() {
        std::env::set_var("OLLAMA_HOST", host);
    }
    match cli.command {
        Commands::Run(args) => cmd_run(args, scratch),
        Commands::Sweep(args) => cmd_sweep(args, scratch),
        Commands::Reproduce(args) => cmd_reproduce(args, scratch),
    }
}
