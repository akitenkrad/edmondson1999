//! Edmondson (1999) — Psychological Safety & Team Learning CLI.
//!
//! `run`       : single configuration; `--decision-mode {rule|llm}`.
//! `sweep`     : Cartesian product over ψ-update params × seeds; one row per cell.
//! `reproduce` : team-level cross-section against the §5 calibration anchors.

use std::fs;
use std::path::Path;

use clap::{Parser, Subcommand};

use edmondson_team::config::{
    parse_decision_mode, parse_network_kind, Config, LearningWeights, LlmSettings, NetworkKind,
    PsiParams, VoiceBeta,
};
use edmondson_team::simulation::{
    anchor_report_from_result, ensure_output_dir, run, save_individuals, save_llm_meta,
    save_metrics, save_teams, SimulationResult,
};

use socsim_core::derive_seed;
use socsim_results::{refresh_latest_symlink, timestamp, write_csv, write_json};

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
    /// Ollama 接続先 URL（指定時は環境変数 OLLAMA_HOST を上書きする）．
    #[arg(long, global = true)]
    ollama_host: Option<String>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run a single configuration.
    Run(RunArgs),
    /// Sweep ψ-update parameters across seeds; aggregate into `sweep_summary.csv`.
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
    /// Number of independent runs (the run outputs reflect a *pooled* cross-section).
    #[arg(long, default_value_t = 30)]
    runs: usize,
    /// Random seed (governs the socsim core layer).
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
    /// Output base directory.
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
    /// Output base directory.
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
    /// Runs (pooled into the cross-section).
    #[arg(long, default_value_t = 30)]
    runs: usize,
    /// Output base directory.
    #[arg(long, default_value = "results")]
    output_dir: String,
}

// --------------------------------------------------------------------------- //
// CSV rows
// --------------------------------------------------------------------------- //

#[derive(serde::Serialize)]
struct SweepRow {
    decision_mode: String,
    alpha: f64,
    delta: f64,
    lambda: f64,
    run: usize,
    seed: u64,
    icc_psi: f64,
    beta_psi_l: f64,
    r2_psi_l: f64,
    r2_l_pi: f64,
    mediation_ratio: f64,
    beta_support_psi: f64,
    efficacy_t: f64,
    hypotheses_supported: u8,
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

// --------------------------------------------------------------------------- //
// run
// --------------------------------------------------------------------------- //

fn cmd_run(args: RunArgs) {
    let timestamp = timestamp();
    let output_dir = format!("{}/{}", args.output_dir, timestamp);
    ensure_output_dir(&output_dir);

    let mut base_cfg = cfg_from_run_args(&args);
    base_cfg.output_dir = output_dir.clone();
    if base_cfg.decision_mode.is_llm() {
        if let Some(parent) = Path::new(&args.cache_path).parent() {
            let _ = fs::create_dir_all(parent);
        }
    }

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
        base_cfg.runs,
        base_cfg.seed,
    );
    println!("output: {output_dir}");
    println!("----------------------------------------------------------------------");

    {
        let path = format!("{output_dir}/config.json");
        write_json(&base_cfg.to_run_config_json(), &path).expect("failed to write config.json");
    }

    // Run all repeats; keep the last for the long-format CSVs, and pool the
    // final-tick team cross-sections across runs for the printed anchors.
    let mut last_result: Option<SimulationResult> = None;
    let runs = base_cfg.runs.max(1);
    let (mut pooled_icc, mut pooled_beta_psi_l, mut pooled_r2, mut pooled_med) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut cross_section: Vec<edmondson_team::simulation::CrossSectionRow> = Vec::new();
    for run_idx in 0..runs {
        let seed = derive_seed(base_cfg.seed, &[run_idx as u64]);
        let cfg = Config {
            seed,
            ..base_cfg.clone()
        };
        let result = run(&cfg).unwrap_or_else(|e| panic!("run failed: {e}"));
        let rep = anchor_report_from_result(&result);
        pooled_icc.push(rep.icc_psi);
        pooled_beta_psi_l.push(rep.beta_psi_l);
        pooled_r2.push(rep.r2_psi_l);
        pooled_med.push(rep.mediation_ratio);
        cross_section.extend(edmondson_team::simulation::cross_section_rows(
            &result, run_idx,
        ));
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
        last_result = Some(result);
    }

    let result = last_result.expect("at least one run");
    save_teams(&result, &output_dir);
    save_individuals(&result, &output_dir);
    save_metrics(&result, &output_dir);
    edmondson_team::simulation::save_cross_section(&cross_section, &output_dir);
    save_llm_meta(&result, &base_cfg, &output_dir);

    let _ = refresh_latest_symlink(&args.output_dir, &timestamp);

    println!("----------------------------------------------------------------------");
    println!(
        "pooled over {} runs: icc_ψ={:.3} ψ→L B={:.3} R²={:.3} med_ratio={:.3}",
        runs,
        mean(&pooled_icc),
        mean(&pooled_beta_psi_l),
        mean(&pooled_r2),
        mean(&pooled_med),
    );
    println!(
        "LLM calls: {} | cache-hit: {} ({:.1}%) | model: {}",
        result.metadata.total(),
        result.metadata.cache_hits(),
        result.metadata.cache_hit_rate() * 100.0,
        result.llm_model,
    );
    println!("teams       → {output_dir}/teams.csv");
    println!("individuals → {output_dir}/individuals.csv");
    println!("metrics     → {output_dir}/metrics.csv");
    println!("cross_section→ {output_dir}/team_cross_section.csv");
    println!("llm_meta    → {output_dir}/llm_meta.json");
    println!("config      → {output_dir}/config.json");
}

// --------------------------------------------------------------------------- //
// sweep
// --------------------------------------------------------------------------- //

fn cmd_sweep(args: SweepArgs) {
    let mode = parse_decision_mode(&args.decision_mode).unwrap_or_else(|e| panic!("{e}"));
    let timestamp = timestamp();
    let dir_name = format!("{timestamp}_sweep");
    let sweep_dir = format!("{}/{}", args.output_dir, dir_name);
    fs::create_dir_all(&sweep_dir).expect("failed to create sweep dir");

    let alphas = frange(args.alpha_min, args.alpha_max, args.alpha_step);
    let deltas = frange(args.delta_min, args.delta_max, args.delta_step);
    let n_cells = alphas.len() * deltas.len();
    let n_total = n_cells * args.runs;

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
    println!("output: {sweep_dir}");
    println!("------------------------------------------------------------");

    {
        let config_json = serde_json::json!({
            "command": "sweep",
            "decision_mode": mode.label(),
            "n_teams": args.n_teams,
            "team_size": args.team_size,
            "alpha_values": alphas,
            "delta_values": deltas,
            "lambda": args.lambda,
            "runs": args.runs,
            "t_max": args.t_max,
            "seed": args.seed,
        });
        let path = format!("{sweep_dir}/sweep_config.json");
        write_json(&config_json, &path).expect("failed to write sweep_config.json");
    }

    let mut rows: Vec<SweepRow> = Vec::with_capacity(n_total);
    let mut idx = 0usize;
    for &alpha in &alphas {
        for &delta in &deltas {
            for run_idx in 0..args.runs {
                idx += 1;
                let seed = derive_seed(
                    args.seed,
                    &[
                        (alpha * 1000.0) as u64,
                        (delta * 1000.0) as u64,
                        run_idx as u64,
                    ],
                );
                let cfg = Config {
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
                    runs: 1,
                    seed,
                    ..Config::default()
                };
                let result = run(&cfg).unwrap_or_else(|e| panic!("sweep run failed: {e}"));
                let rep = anchor_report_from_result(&result);
                rows.push(SweepRow {
                    decision_mode: mode.label().to_string(),
                    alpha,
                    delta,
                    lambda: args.lambda,
                    run: run_idx,
                    seed,
                    icc_psi: rep.icc_psi,
                    beta_psi_l: rep.beta_psi_l,
                    r2_psi_l: rep.r2_psi_l,
                    r2_l_pi: rep.r2_l_pi,
                    mediation_ratio: rep.mediation_ratio,
                    beta_support_psi: rep.beta_support_psi,
                    efficacy_t: rep.efficacy_partial_t,
                    hypotheses_supported: rep.hypotheses_supported,
                });
                if idx.is_multiple_of(20) || idx == n_total {
                    println!(
                        "[{}/{}] α={:.2} δ={:.2} run={} icc_ψ={:.3} med={:.3}",
                        idx, n_total, alpha, delta, run_idx, rep.icc_psi, rep.mediation_ratio
                    );
                }
            }
        }
    }

    let path = format!("{sweep_dir}/sweep_summary.csv");
    write_csv(&rows, &path).expect("failed to write sweep_summary.csv");

    let _ = refresh_latest_symlink(&args.output_dir, &dir_name);
    println!("------------------------------------------------------------");
    println!("sweep done.");
    println!("summary → {sweep_dir}/sweep_summary.csv");
    println!("config  → {sweep_dir}/sweep_config.json");
}

// --------------------------------------------------------------------------- //
// reproduce
// --------------------------------------------------------------------------- //

fn band(name: &str, value: f64, lo: f64, hi: f64) -> String {
    let ok = value >= lo && value <= hi;
    format!(
        "  {name:<22} = {value:>7.3}   ([{lo:.2}, {hi:.2}]: {})",
        if ok { "PASS" } else { "off-anchor" }
    )
}

fn cmd_reproduce(args: ReproduceArgs) {
    let mode = parse_decision_mode(&args.decision_mode).unwrap_or_else(|e| panic!("{e}"));
    println!("=== edmondson-reproduce ({} mode) ===", mode.label());

    // Each independent trial yields one set of anchor statistics on its own
    // team cross-section (n_teams ≈ the paper's 51); we report the mean across
    // trials, matching the §6 "30 trials, average ± 95% CI" plan. Pooling all
    // teams into one giant regression would inflate the statistical power far
    // beyond the paper's design (and make every effect spuriously significant).
    let runs = args.runs.max(1);
    let mut reps = Vec::with_capacity(runs);
    for run_idx in 0..runs {
        let seed = derive_seed(args.seed, &[run_idx as u64]);
        let cfg = Config {
            n_teams: args.n_teams,
            team_size: args.team_size,
            decision_mode: mode,
            t_max: args.t_max,
            runs: 1,
            seed,
            ..Config::default()
        };
        let result = run(&cfg).unwrap_or_else(|e| panic!("reproduce run failed: {e}"));
        reps.push(edmondson_team::simulation::anchor_report_from_result(
            &result,
        ));
    }
    let mean_f = |f: &dyn Fn(&edmondson_team::simulation::AnchorReport) -> f64| -> f64 {
        reps.iter().map(f).sum::<f64>() / reps.len() as f64
    };
    let rep = edmondson_team::simulation::AnchorReport {
        icc_psi: mean_f(&|r| r.icc_psi),
        icc_learning: mean_f(&|r| r.icc_learning),
        beta_psi_l: mean_f(&|r| r.beta_psi_l),
        r2_psi_l: mean_f(&|r| r.r2_psi_l),
        r2_l_pi: mean_f(&|r| r.r2_l_pi),
        beta_l_pi: mean_f(&|r| r.beta_l_pi),
        beta_psi_residual: mean_f(&|r| r.beta_psi_residual),
        beta_psi_residual_p: mean_f(&|r| r.beta_psi_residual_p),
        mediation_ratio: mean_f(&|r| r.mediation_ratio),
        beta_support_psi: mean_f(&|r| r.beta_support_psi),
        beta_support_psi_p: mean_f(&|r| r.beta_support_psi_p),
        efficacy_partial_t: mean_f(&|r| r.efficacy_partial_t),
        hypotheses_supported: (mean_f(&|r| r.hypotheses_supported as f64)).round() as u8,
    };

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
    println!("For the full Table 4-8-style Baron & Kenny report + bootstrap mediation");
    println!("95% BC CI + the efficacy discriminant, run the Python tool:");
    println!("  uv run edmondson-tools reproduce --results-dir results/latest");
}

// --------------------------------------------------------------------------- //
// main
// --------------------------------------------------------------------------- //

fn main() {
    let cli = Cli::parse();
    if let Some(host) = cli.ollama_host.as_deref() {
        std::env::set_var("OLLAMA_HOST", host);
    }
    match cli.command {
        Commands::Run(args) => cmd_run(args),
        Commands::Sweep(args) => cmd_sweep(args),
        Commands::Reproduce(args) => cmd_reproduce(args),
    }
}
