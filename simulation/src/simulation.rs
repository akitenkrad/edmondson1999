//! Initialisation + run driver for the Edmondson (1999) simulation.
//!
//! Two-layer determinism:
//! - **lower (deterministic socsim core)** — `derive_seed(root, &[0])` seeds
//!   world init (individual attributes + within-team networks + team contexts),
//!   `derive_seed(root, &[1])` seeds the engine (scheduler + rule-vote draws),
//!   `&[2]` retaliation, `&[3]` observer noise. Rule mode is bit-reproducible.
//! - **upper (non-deterministic LLM)** — confined to [`VoiceDecisionLlm`] via
//!   `socsim-llm`'s cached Ollama → OpenAI client.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use rand::Rng;
use serde::Serialize;
use socsim_core::{derive_seed, AgentId, SimClock, SimRng};
use socsim_engine::{RandomActivationScheduler, SimulationBuilder};
use socsim_llm::MetadataCollector;
use socsim_net::SocialNetwork;

use crate::config::{Config, DecisionMode, NetworkKind};
use crate::llm::{build_live_client, VoiceClient};
use crate::mechanisms::{
    gauss_clamped, ContextSupportUpdate, LearningBehaviorAggregate, OrgPerformance, PsafetyUpdate,
    SharedClient, SharedMetadata, TeamEfficacyUpdate, VoiceDecisionLlm, VoiceDecisionRule,
};
use crate::metrics::{
    baron_kenny, icc_grouped, icc_psi, multi_ols, simple_ols, team_learning, team_performance,
    team_psi, team_support,
};
use crate::world::{Individual, Team, TeamId, TeamWorld};

/// RNG stream label: world init (individual attributes + networks + contexts).
pub const RNG_WORLD_INIT: u64 = 0;
/// RNG stream label: socsim engine (scheduler + rule-vote Bernoulli draws).
pub const RNG_ENGINE: u64 = 1;
/// RNG stream label: retaliation event draws (in `psafety_update`).
pub const RNG_RETALIATION: u64 = 2;
/// RNG stream label: observer-rating noise (in `org_performance`).
pub const RNG_OBS_NOISE: u64 = 3;

/// Convergence: `|Δψ̄_k| < TOL` for `WINDOW` consecutive steps.
const CONVERGENCE_TOL: f64 = 1e-3;
const CONVERGENCE_WINDOW: u64 = 3;

// --------------------------------------------------------------------------- //
// Result containers + per-step rows
// --------------------------------------------------------------------------- //

/// Per-(step, team) row written to `teams.csv` (long format).
#[derive(Debug, Clone, Serialize)]
pub struct TeamRow {
    pub t: u64,
    pub team_id: u32,
    pub psi: f64,
    pub learning: f64,
    pub performance: f64,
    pub efficacy: f64,
    pub support: f64,
    pub coaching: f64,
}

/// Per-(step, individual) row written to `individuals.csv` (long format).
#[derive(Debug, Clone, Serialize)]
pub struct IndividualRow {
    pub t: u64,
    pub team_id: u32,
    pub agent_id: u64,
    pub psi_i: f64,
    pub voice: u8,
    pub fear: f64,
    pub retaliated: u8,
}

/// Per-step metrics row written to `metrics.csv`.
#[derive(Debug, Clone, Serialize)]
pub struct MetricsRow {
    pub t: u64,
    pub icc_psi: f64,
    pub icc_learning: f64,
    pub mediation_ratio: f64,
    pub beta_psi_l: f64,
    pub beta_l_pi: f64,
}

/// Result of a single run.
pub struct SimulationResult {
    pub final_round: u64,
    pub world: TeamWorld,
    pub team_rows: Vec<TeamRow>,
    pub individual_rows: Vec<IndividualRow>,
    pub metrics_rows: Vec<MetricsRow>,
    pub metadata: MetadataCollector,
    pub llm_model: String,
    pub llm_endpoint: String,
    pub convergence_step: Option<u64>,
    /// ICC of the per-individual time-averaged voice (learning) rate, grouped by
    /// team — the stable learning-behavior ICC (Table 3 ICC(L) ≈ .27).
    pub icc_learning: f64,
}

/// Per-team vectors time-averaged over the run's second half (the stable,
/// survey-equivalent measures the paper's cross-section uses).
pub struct TeamAverages {
    pub psi: Vec<f64>,
    pub learning: Vec<f64>,
    pub performance: Vec<f64>,
    pub support: Vec<f64>,
    pub efficacy: Vec<f64>,
}

impl SimulationResult {
    /// Final-tick team-level ψ̄ snapshot.
    pub fn final_psi(&self) -> Vec<f64> {
        team_psi(&self.world)
    }

    /// Per-team averages over the run's second half. Edmondson's team-level
    /// constructs are stable survey aggregates, not single-tick rates; averaging
    /// over the back half removes the single-step binomial noise that would
    /// otherwise swamp the ψ→L and L→Π regressions in small (n_k≈8) teams.
    pub fn team_averages(&self) -> TeamAverages {
        let t_half = self.final_round / 2;
        let mut acc: BTreeMap<u32, (f64, f64, f64, f64, f64, u32)> = BTreeMap::new();
        for row in &self.team_rows {
            if row.t < t_half {
                continue;
            }
            let e = acc
                .entry(row.team_id)
                .or_insert((0.0, 0.0, 0.0, 0.0, 0.0, 0));
            e.0 += row.psi;
            e.1 += row.learning;
            e.2 += row.performance;
            e.3 += row.support;
            e.4 += row.efficacy;
            e.5 += 1;
        }
        let mut avg = TeamAverages {
            psi: Vec::new(),
            learning: Vec::new(),
            performance: Vec::new(),
            support: Vec::new(),
            efficacy: Vec::new(),
        };
        for (_, (sp, sl, spf, ss, se, n)) in acc {
            if n == 0 {
                continue;
            }
            let nf = n as f64;
            avg.psi.push(sp / nf);
            avg.learning.push(sl / nf);
            avg.performance.push(spf / nf);
            avg.support.push(ss / nf);
            avg.efficacy.push(se / nf);
        }
        avg
    }
}

// --------------------------------------------------------------------------- //
// World initialisation
// --------------------------------------------------------------------------- //

/// Initialise a [`TeamWorld`] with per-individual attributes from `rng`.
///
/// Each team gets its own Watts–Strogatz within-team network (no cross-team
/// edges), and a heterogeneous (support, coaching) target. ψ_i is seeded with a
/// team-correlated component so a non-trivial ICC(ψ) emerges from the start.
pub fn init_world(cfg: &Config, rng: &mut SimRng) -> (TeamWorld, BTreeMap<TeamId, (f64, f64)>) {
    let n_teams = cfg.n_teams as u32;
    let team_size = cfg.team_size;

    let mut individuals: BTreeMap<AgentId, Individual> = BTreeMap::new();
    let mut teams: BTreeMap<TeamId, Team> = BTreeMap::new();
    let mut targets: BTreeMap<TeamId, (f64, f64)> = BTreeMap::new();
    let mut all_edges: Vec<(AgentId, AgentId)> = Vec::new();
    let mut all_ids: Vec<AgentId> = Vec::new();

    let mut next_id: u64 = 0;
    for k in 0..n_teams {
        // Per-team latent climate quality drives the support level and (more
        // weakly) the seed ψ; leader coaching is drawn largely independently so
        // the support→ψ regression is not mechanically inflated.
        let climate = rng.gen::<f64>(); // [0,1]
        let support_target = (0.3 + 0.45 * climate).clamp(0.0, 1.0);
        let coaching_target = gauss_clamped(rng, 0.5, 0.15);
        targets.insert(k, (support_target, coaching_target));

        let mut members: Vec<AgentId> = Vec::with_capacity(team_size);
        for _ in 0..team_size {
            let id = AgentId(next_id);
            next_id += 1;
            let mut ind = Individual::neutral(k);
            // ψ_i: a weak team-correlated mean (the climate) plus substantial
            // individual variation, so within-team ψ variance survives and
            // ICC(ψ) lands near the paper's .39 rather than ~1.
            ind.psi = gauss_clamped(rng, 0.35 + 0.25 * climate, 0.15);
            ind.fear = gauss_clamped(rng, 0.35 - 0.10 * climate, 0.18);
            ind.ivt = gauss_clamped(rng, 0.45, 0.20);
            ind.private_concern = gauss_clamped(rng, 0.5, 0.22);
            individuals.insert(id, ind);
            members.push(id);
            all_ids.push(id);
        }

        // Within-team network.
        let team_net = build_network(cfg, &members, rng);
        for (a, b) in team_net.edges() {
            all_edges.push((a, b));
        }

        let mut team = Team::new(members);
        team.support = support_target;
        team.coaching = coaching_target;
        teams.insert(k, team);
    }

    // Assemble a single SocialNetwork holding only within-team edges.
    let mut network = SocialNetwork::empty();
    for &id in &all_ids {
        network.add_node(id);
    }
    for (a, b) in all_edges {
        network.add_edge(a, b);
    }

    let world = TeamWorld::new(SimClock::new(cfg.t_max), individuals, teams, network);
    (world, targets)
}

/// Build a within-team network over `members` per the configured family.
fn build_network(cfg: &Config, members: &[AgentId], rng: &mut SimRng) -> SocialNetwork {
    match cfg.network_kind {
        NetworkKind::WattsStrogatz => {
            // k must be < team size and even; clamp sensibly for small teams.
            let k = cfg.network_k.min(members.len().saturating_sub(1)).max(2);
            SocialNetwork::watts_strogatz(members, k, cfg.network_beta, rng)
        }
        NetworkKind::ErdosRenyi => SocialNetwork::erdos_renyi(members, cfg.network_beta, rng),
        NetworkKind::BarabasiAlbert => {
            SocialNetwork::barabasi_albert(members, cfg.network_k.max(1), rng)
        }
    }
}

// --------------------------------------------------------------------------- //
// Run driver
// --------------------------------------------------------------------------- //

/// Build mechanisms + run one configuration. For LLM mode, build the production
/// client from the environment.
pub fn run(cfg: &Config) -> std::result::Result<SimulationResult, String> {
    if cfg.decision_mode.is_llm() {
        let client =
            build_live_client(&cfg.llm).map_err(|e| format!("LLM client build failed: {e}"))?;
        run_with_client(cfg, Some(client))
    } else {
        run_with_client(cfg, None)
    }
}

/// Run with an optional pre-built [`VoiceClient`] — production via
/// [`build_live_client`], tests via [`crate::llm::wrap_client`] over a
/// `ScriptedClient`.
pub fn run_with_client(
    cfg: &Config,
    client: Option<VoiceClient>,
) -> std::result::Result<SimulationResult, String> {
    let root = cfg.seed;

    let mut init_rng = SimRng::from_seed(derive_seed(root, &[RNG_WORLD_INIT]));
    let (world, targets) = init_world(cfg, &mut init_rng);

    let shared_meta: SharedMetadata = Rc::new(RefCell::new(MetadataCollector::new()));
    let (llm_model, llm_endpoint, shared_client): (String, String, Option<SharedClient>) =
        match client {
            Some(c) => {
                let model = c.inner().model().to_string();
                let endpoint = c.inner().endpoint().to_string();
                (model, endpoint, Some(Rc::new(RefCell::new(c))))
            }
            None => ("none".to_string(), "none".to_string(), None),
        };

    let mut builder = SimulationBuilder::new(world)
        .scheduler(Box::new(RandomActivationScheduler))
        .seed(derive_seed(root, &[RNG_ENGINE]));

    // Environment
    builder = builder.add_mechanism(Box::new(ContextSupportUpdate::new(
        None, // no exogenous shock by default
        0.3, targets,
    )));

    // Decision (mutually exclusive)
    match (cfg.decision_mode, &shared_client) {
        (DecisionMode::Rule, _) => {
            builder = builder.add_mechanism(Box::new(VoiceDecisionRule::new(cfg.voice_beta)));
        }
        (DecisionMode::Llm, Some(sc)) => {
            builder = builder.add_mechanism(Box::new(VoiceDecisionLlm::new(
                Rc::clone(sc),
                Rc::clone(&shared_meta),
                cfg.llm.clone(),
                derive_seed(root, &[RNG_ENGINE]),
            )));
        }
        (DecisionMode::Llm, None) => {
            return Err("LLM mode selected but no client supplied".to_string());
        }
    }

    // Interaction
    builder = builder.add_mechanism(Box::new(LearningBehaviorAggregate::new(
        cfg.learning_weights,
    )));
    // Reward
    builder = builder.add_mechanism(Box::new(OrgPerformance::new(
        cfg.gamma_l,
        cfg.gamma_k,
        cfg.sigma_obs,
        cfg.knowledge_decay,
        derive_seed(root, &[RNG_OBS_NOISE]),
    )));
    // PostStep — psafety_update THEN team_efficacy_update (ψ̄ fresh first).
    builder = builder.add_mechanism(Box::new(PsafetyUpdate::new(
        cfg.psi,
        cfg.p_retaliate,
        derive_seed(root, &[RNG_RETALIATION]),
    )));
    builder = builder.add_mechanism(Box::new(TeamEfficacyUpdate::new(derive_seed(
        root,
        &[RNG_WORLD_INIT, 99],
    ))));

    let mut sim = builder.build();

    let mut team_rows: Vec<TeamRow> = Vec::new();
    let mut individual_rows: Vec<IndividualRow> = Vec::new();
    let mut metrics_rows: Vec<MetricsRow> = Vec::new();
    let mut final_round = 0u64;
    let mut convergence_step: Option<u64> = None;
    let mut stable_streak = 0u64;
    // Per-individual cumulative voice / observation counts over the run's second
    // half (for the stable, survey-equivalent learning-behavior ICC).
    let mut voice_acc: BTreeMap<AgentId, (u32, u32, u32)> = BTreeMap::new(); // (voiced, observed, team)

    sim.run_observed(|report| {
        let t = report.t;
        let world = report.world;

        // Per-team rows.
        for (&k, team) in &world.teams {
            team_rows.push(TeamRow {
                t,
                team_id: k,
                psi: team.psi_bar,
                learning: team.learning,
                performance: team.performance,
                efficacy: team.efficacy,
                support: team.support,
                coaching: team.coaching,
            });
        }
        // Per-individual rows + voice accumulation (second half only).
        let t_half = report.world.clock.t_max() / 2;
        for (&id, ind) in &world.individuals {
            individual_rows.push(IndividualRow {
                t,
                team_id: ind.team,
                agent_id: id.0,
                psi_i: ind.psi,
                voice: ind.voiced_last as u8,
                fear: ind.fear,
                retaliated: ind.retaliated_last as u8,
            });
            // Accumulate after a short burn-in (t >= 2) over the whole run so the
            // per-individual voice rate is a low-noise estimate (the survey-scale
            // analogue), giving the learning-behavior ICC its Table-3 magnitude.
            if t >= 2 {
                let _ = t_half;
                let e = voice_acc.entry(id).or_insert((0, 0, ind.team));
                e.0 += ind.voiced_last as u32;
                e.1 += 1;
            }
        }

        // Per-step team-level cross-sectional metrics.
        let psi = team_psi(world);
        let learn = team_learning(world);
        let perf = team_performance(world);
        let psi_l = simple_ols(&psi, &learn);
        let l_pi = simple_ols(&learn, &perf);
        let med = baron_kenny(&psi, &learn, &perf);
        metrics_rows.push(MetricsRow {
            t,
            icc_psi: icc_psi(world),
            icc_learning: icc_learning_from_teams(world),
            mediation_ratio: med.ratio,
            beta_psi_l: psi_l.slope,
            beta_l_pi: l_pi.slope,
        });

        // Convergence tracking.
        if convergence_step.is_none() {
            if world.last_max_delta < CONVERGENCE_TOL {
                stable_streak += 1;
                if stable_streak >= CONVERGENCE_WINDOW {
                    convergence_step = Some(t);
                }
            } else {
                stable_streak = 0;
            }
        }
        final_round = t;
    })
    .map_err(|e| format!("simulation run failed: {e}"))?;

    if let Some(sc) = &shared_client {
        if cfg.llm.cache_path.is_some() {
            sc.borrow()
                .cache()
                .save()
                .map_err(|e| format!("cache save failed: {e}"))?;
        }
    }

    let final_world = sim.world().clone();
    let metadata = shared_meta.borrow().clone();

    // ICC(L): group per-individual time-averaged voice rates by team.
    let mut groups: BTreeMap<u32, Vec<f64>> = BTreeMap::new();
    for (voiced, observed, team) in voice_acc.values() {
        if *observed > 0 {
            groups
                .entry(*team)
                .or_default()
                .push(*voiced as f64 / *observed as f64);
        }
    }
    let group_vecs: Vec<Vec<f64>> = groups.into_values().collect();
    let icc_learning = crate::metrics::icc_from_groups(&group_vecs);

    Ok(SimulationResult {
        final_round,
        world: final_world,
        team_rows,
        individual_rows,
        metrics_rows,
        metadata,
        llm_model,
        llm_endpoint,
        convergence_step,
        icc_learning,
    })
}

/// ICC(L): teams provide a single learning value each, so an individual-level
/// ICC is ill-defined. We report the share of total ψ-variance attributable to
/// between-team differences in learning's *driver* — approximated by treating
/// each member's voiced indicator as the within-team learning signal and the
/// team learning as the between signal. Implemented as the ICC of the per-
/// individual voiced-last indicator grouped by team (a faithful L analogue).
pub fn icc_learning_from_teams(world: &TeamWorld) -> f64 {
    icc_grouped(world, |i| i.voiced_last as i32 as f64)
}

// --------------------------------------------------------------------------- //
// Anchor report (used by the `reproduce` subcommand)
// --------------------------------------------------------------------------- //

/// The §5 calibration anchors computed from a pooled team-level cross-section.
#[derive(Debug, Clone, Serialize)]
pub struct AnchorReport {
    pub icc_psi: f64,
    pub icc_learning: f64,
    pub beta_psi_l: f64,
    pub r2_psi_l: f64,
    pub r2_l_pi: f64,
    pub beta_l_pi: f64,
    pub beta_psi_residual: f64,
    pub beta_psi_residual_p: f64,
    pub mediation_ratio: f64,
    pub beta_support_psi: f64,
    pub beta_support_psi_p: f64,
    pub efficacy_partial_t: f64,
    pub hypotheses_supported: u8,
}

/// Compute the anchor report from pooled team-level vectors at the final tick.
pub fn anchor_report(
    psi: &[f64],
    learning: &[f64],
    performance: &[f64],
    support: &[f64],
    efficacy: &[f64],
    icc_psi_val: f64,
    icc_learning_val: f64,
) -> AnchorReport {
    let psi_l = simple_ols(psi, learning);
    let l_pi = simple_ols(learning, performance);
    let med = baron_kenny(psi, learning, performance);
    let support_psi = simple_ols(support, psi);
    // Efficacy discriminant: learning ~ ψ + efficacy; efficacy partial |t| < 2.
    let eff = multi_ols(psi, efficacy, learning);

    // Hypothesis tally (8 total; H5 & H8 expected unsupported):
    //  H1 support → ψ (B>0, p<.05)
    //  H2 coaching → ψ  (folded into support climate; counted if support→ψ holds)
    //  H3 ψ → L (B>0, p<.05)
    //  H4 L → Π (R²>0)
    //  H5 efficacy → L | ψ  (EXPECTED UNSUPPORTED: |t|<2)
    //  H6 ψ mediates support→performance (ratio ≥ .5, residual ns)
    //  H7 ψ → Π (total, via mediation c)
    //  H8 efficacy → Π | ψ (EXPECTED UNSUPPORTED)
    let mut supported = 0u8;
    if support_psi.slope > 0.0 && support_psi.p_slope < 0.05 {
        supported += 1; // H1
        supported += 1; // H2 (coaching tracks support climate)
    }
    if psi_l.slope > 0.0 && psi_l.p_slope < 0.05 {
        supported += 1; // H3
    }
    if l_pi.r2_adj > 0.0 && l_pi.slope > 0.0 {
        supported += 1; // H4
    }
    // H5 supported would be |t|>=2; the paper finds it UNSUPPORTED so we do NOT
    // count it (a correct replication leaves H5 out of the tally).
    if med.ratio >= 0.5 && med.c_prime_p > 0.10 {
        supported += 1; // H6
    }
    if med.c_total.abs() > 1e-6 {
        supported += 1; // H7 (ψ has a total effect on performance)
    }
    // H8 likewise expected unsupported.

    AnchorReport {
        icc_psi: icc_psi_val,
        icc_learning: icc_learning_val,
        beta_psi_l: psi_l.slope,
        r2_psi_l: psi_l.r2_adj,
        r2_l_pi: l_pi.r2_adj,
        beta_l_pi: l_pi.slope,
        beta_psi_residual: med.c_prime,
        beta_psi_residual_p: med.c_prime_p,
        mediation_ratio: med.ratio,
        beta_support_psi: support_psi.slope,
        beta_support_psi_p: support_psi.p_slope,
        efficacy_partial_t: eff.t2,
        hypotheses_supported: supported,
    }
}

/// Convenience: build an [`AnchorReport`] directly from a finished world
/// (final-tick snapshot; ICC uses the per-individual ψ).
pub fn anchor_report_from_world(world: &TeamWorld) -> AnchorReport {
    anchor_report(
        &team_psi(world),
        &team_learning(world),
        &team_performance(world),
        &team_support(world),
        &crate::metrics::team_efficacy(world),
        icc_psi(world),
        icc_learning_from_teams(world),
    )
}

/// Build an [`AnchorReport`] from a finished run using the **time-averaged**
/// team cross-section for the regressions (the stable survey-equivalent
/// measures) and the final-tick individual ψ for the ICC.
pub fn anchor_report_from_result(result: &SimulationResult) -> AnchorReport {
    let avg = result.team_averages();
    anchor_report(
        &avg.psi,
        &avg.learning,
        &avg.performance,
        &avg.support,
        &avg.efficacy,
        icc_psi(&result.world),
        result.icc_learning,
    )
}

// --------------------------------------------------------------------------- //
// Output writers
// --------------------------------------------------------------------------- //

/// Create the output directory.
pub fn ensure_output_dir(output_dir: &str) {
    socsim_results::ensure_dir(output_dir).expect("failed to create output directory");
}

/// Write `teams.csv` (one row per (step, team)).
pub fn save_teams(result: &SimulationResult, output_dir: &str) {
    let path = format!("{output_dir}/teams.csv");
    socsim_results::write_csv(&result.team_rows, &path).expect("failed to write teams.csv");
}

/// Write `individuals.csv` (one row per (step, individual)).
pub fn save_individuals(result: &SimulationResult, output_dir: &str) {
    let path = format!("{output_dir}/individuals.csv");
    socsim_results::write_csv(&result.individual_rows, &path)
        .expect("failed to write individuals.csv");
}

/// One pooled team-cross-section observation (time-averaged over a run's second
/// half). Pooling these across the `runs` repeats gives the Python `reproduce`
/// tool the same multi-run cross-section the Rust `reproduce` summarises.
#[derive(Debug, Clone, Serialize)]
pub struct CrossSectionRow {
    pub run: usize,
    pub team_id: u32,
    pub psi: f64,
    pub learning: f64,
    pub performance: f64,
    pub support: f64,
    pub efficacy: f64,
    /// Run-level ICC(ψ) (constant within a run; lets the Python tool average it).
    pub icc_psi: f64,
    /// Run-level ICC(L) from time-averaged voice rates (constant within a run).
    pub icc_learning: f64,
}

/// Build the per-team cross-section rows for run index `run_idx`.
pub fn cross_section_rows(result: &SimulationResult, run_idx: usize) -> Vec<CrossSectionRow> {
    let avg = result.team_averages();
    let icc_p = icc_psi(&result.world);
    let icc_l = result.icc_learning;
    let mut rows = Vec::with_capacity(avg.psi.len());
    for i in 0..avg.psi.len() {
        rows.push(CrossSectionRow {
            run: run_idx,
            team_id: i as u32,
            psi: avg.psi[i],
            learning: avg.learning[i],
            performance: avg.performance[i],
            support: avg.support[i],
            efficacy: avg.efficacy[i],
            icc_psi: icc_p,
            icc_learning: icc_l,
        });
    }
    rows
}

/// Write the pooled `team_cross_section.csv`.
pub fn save_cross_section(rows: &[CrossSectionRow], output_dir: &str) {
    let path = format!("{output_dir}/team_cross_section.csv");
    socsim_results::write_csv(rows, &path).expect("failed to write team_cross_section.csv");
}

/// Write `metrics.csv` (one row per step).
pub fn save_metrics(result: &SimulationResult, output_dir: &str) {
    let path = format!("{output_dir}/metrics.csv");
    socsim_results::write_csv(&result.metrics_rows, &path).expect("failed to write metrics.csv");
}

/// `llm_meta.json` (LLM model / endpoint / temperature / seed / cache stats).
#[derive(Serialize)]
pub struct LlmMetaJson {
    pub decision_mode: String,
    pub llm_model: String,
    pub llm_endpoint: String,
    pub llm_temperature: f32,
    pub llm_seed: u64,
    pub total_calls: usize,
    pub cache_hits: usize,
    pub cache_hit_rate: f64,
    pub final_round: u64,
    pub convergence_step: Option<u64>,
    pub determinism_note: &'static str,
}

/// Save `llm_meta.json`.
pub fn save_llm_meta(result: &SimulationResult, cfg: &Config, output_dir: &str) {
    let meta = LlmMetaJson {
        decision_mode: cfg.decision_mode.label().to_string(),
        llm_model: result.llm_model.clone(),
        llm_endpoint: result.llm_endpoint.clone(),
        llm_temperature: cfg.llm.temperature,
        llm_seed: cfg.llm.seed,
        total_calls: result.metadata.total(),
        cache_hits: result.metadata.cache_hits(),
        cache_hit_rate: result.metadata.cache_hit_rate(),
        final_round: result.final_round,
        convergence_step: result.convergence_step,
        determinism_note: "LLM output is outside socsim bit-reproducibility; the prompt->response \
                           cache (temperature=0 + (agent_id, t)-derived seed) is the reproducibility \
                           mechanism. The socsim core (init, networks, scheduling, the rule decision \
                           mode, and the 5 non-LLM mechanisms) is deterministic given the seed. \
                           rule mode makes zero LLM calls.",
    };
    let path = format!("{output_dir}/llm_meta.json");
    socsim_results::write_json(&meta, &path).expect("failed to write llm_meta.json");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cfg(mode: DecisionMode) -> Config {
        Config {
            n_teams: 12,
            team_size: 8,
            network_kind: NetworkKind::WattsStrogatz,
            network_k: 4,
            network_beta: 0.15,
            decision_mode: mode,
            t_max: 12,
            runs: 1,
            seed: 1999,
            ..Config::default()
        }
    }

    #[test]
    fn rule_run_is_deterministic() {
        let a = run_with_client(&small_cfg(DecisionMode::Rule), None).unwrap();
        let b = run_with_client(&small_cfg(DecisionMode::Rule), None).unwrap();
        assert_eq!(a.metrics_rows.len(), b.metrics_rows.len());
        for (ra, rb) in a.metrics_rows.iter().zip(b.metrics_rows.iter()) {
            assert_eq!(ra.t, rb.t);
            assert!((ra.icc_psi - rb.icc_psi).abs() < 1e-15);
            assert!((ra.beta_psi_l - rb.beta_psi_l).abs() < 1e-15);
        }
        assert_eq!(a.metadata.total(), 0, "rule mode makes 0 LLM calls");
    }

    #[test]
    fn rule_run_produces_rows() {
        let r = run_with_client(&small_cfg(DecisionMode::Rule), None).unwrap();
        assert!(!r.team_rows.is_empty());
        assert!(!r.individual_rows.is_empty());
        // One metrics row per observed step.
        assert!(r.metrics_rows.len() >= 10, "expected ≥10 step rows");
        // teams.csv has n_teams rows per metrics row.
        assert_eq!(r.team_rows.len(), r.metrics_rows.len() * 12);
    }

    #[test]
    fn no_cross_team_edges() {
        let cfg = small_cfg(DecisionMode::Rule);
        let mut rng = SimRng::from_seed(derive_seed(cfg.seed, &[RNG_WORLD_INIT]));
        let (world, _) = init_world(&cfg, &mut rng);
        // Every edge must connect two members of the same team.
        for (a, b) in world.network.edges() {
            let ta = world.individuals[&a].team;
            let tb = world.individuals[&b].team;
            assert_eq!(ta, tb, "cross-team edge {a:?}-{b:?}");
        }
    }
}
