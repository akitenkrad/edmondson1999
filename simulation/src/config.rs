//! Simulation configuration for the Edmondson (1999) psychological-safety model.
//!
//! Holds every knob surfaced by the `run` / `sweep` / `reproduce` CLI: org /
//! network shape, the ψ-update diff-eq weights `(λ, α, β, γ, δ)`, the voice
//! logit `β` group, the learning-aggregate `w` weights, the org-performance
//! `(γ_L, γ_K, σ_obs)` parameters, and the LLM settings used in `--decision-mode
//! llm`.

use serde::Serialize;

// --------------------------------------------------------------------------- //
// DecisionMode — rule-based logit vs LLM-driven voice_decision
// --------------------------------------------------------------------------- //

/// Decision-mechanism selector (mutually exclusive, like detert2011's switch).
///
/// The driver wires **exactly one** voice-decision mechanism:
/// - `Rule` → `voice_decision_rule` logit (default; zero LLM calls, deterministic)
/// - `Llm`  → `voice_decision` (LLM via the socsim-llm shared harness)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionMode {
    /// `voice_decision_rule` — logit; default, bit-deterministic, 0 LLM calls.
    Rule,
    /// `voice_decision` — LLM-driven (Phase-3 ablation).
    Llm,
}

impl DecisionMode {
    /// Stable snake_case label (CSV / JSON / directory friendly).
    pub fn label(&self) -> &'static str {
        match self {
            DecisionMode::Rule => "rule",
            DecisionMode::Llm => "llm",
        }
    }

    /// Whether this mode reaches the LLM layer.
    pub fn is_llm(&self) -> bool {
        matches!(self, DecisionMode::Llm)
    }
}

/// Parse a [`DecisionMode`] from a CLI string.
pub fn parse_decision_mode(s: &str) -> Result<DecisionMode, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "rule" | "rules" | "logit" => Ok(DecisionMode::Rule),
        "llm" | "ollama" | "openai" => Ok(DecisionMode::Llm),
        _ => Err(format!("invalid decision-mode: \"{s}\" (rule / llm)")),
    }
}

// --------------------------------------------------------------------------- //
// LLM settings (re-exported from socsim-llm)
// --------------------------------------------------------------------------- //

/// LLM-layer settings (`temperature`, `seed`, `cache_path`) — re-exported from
/// `socsim-llm::harness` so every replication shares one struct.
pub use socsim_llm::LlmSettings;

// --------------------------------------------------------------------------- //
// PsiParams — ψ-update diff-eq weights
// --------------------------------------------------------------------------- //

/// Weights of the ψ-update difference equation (§4.3):
///
/// `ψ_i(t+1) = (1−λ)·ψ_i + λ·[α·s_k + β·c_k − γ·1[retaliated] + δ·ψ̄_{−i,k}]`.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct PsiParams {
    /// Learning rate `λ ∈ (0, 1]`.
    pub lambda: f64,
    /// Context-support weight `α`.
    pub alpha: f64,
    /// Leader-coaching weight `β`.
    pub beta: f64,
    /// Retaliation-shock weight `γ`.
    pub gamma: f64,
    /// Shared-belief convergence weight `δ`.
    pub delta: f64,
}

impl Default for PsiParams {
    fn default() -> Self {
        // Calibrated toward the §5 anchors: ICC(ψ) ≈ .39 (δ in [.30,.45]),
        // support→ψ B ≈ .56, ψ→L B ≈ .76.
        PsiParams {
            lambda: 0.10,
            alpha: 0.30,
            beta: 0.25,
            gamma: 0.50,
            delta: 0.35,
        }
    }
}

// --------------------------------------------------------------------------- //
// VoiceBeta — voice-decision logit coefficients
// --------------------------------------------------------------------------- //

/// Coefficient group for the rule-mode VOICE logit (§4.3):
///
/// `P(VOICE) = σ(β0 + β_ψ·ψ_i − β_f·f_i − β_θ·θ_i − β_c·c_i)`.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct VoiceBeta {
    /// Intercept `β0`.
    pub intercept: f64,
    /// `β_ψ` — psychological safety (positive for VOICE; Table 5A B≈.76 → ≈1.2).
    pub beta_psafety: f64,
    /// `β_f` — fear (subtracted).
    pub beta_fear: f64,
    /// `β_θ` — implicit-voice-theory strength (subtracted).
    pub beta_ivt: f64,
    /// `β_c` — private concern (subtracted; higher concern → more self-censorship).
    pub beta_concern: f64,
}

impl Default for VoiceBeta {
    fn default() -> Self {
        // β_ψ calibrated so team-level ψ̄ drives a steep VOICE response across
        // the observed ψ̄ range, reproducing the strong ψ→L slope (Table 5A
        // B≈.76); the intercept centers the logit near the mean ψ̄ so the
        // response is maximally sensitive there.
        VoiceBeta {
            intercept: -2.05,
            beta_psafety: 4.1,
            beta_fear: 0.8,
            beta_ivt: 0.6,
            beta_concern: 0.3,
        }
    }
}

// --------------------------------------------------------------------------- //
// LearningWeights — learning-behavior aggregation weights
// --------------------------------------------------------------------------- //

/// Weights for `L_k = w_v·(V_k/n_k) + w_h·(H_k/n_k) + w_e·(E_k/n_k)` (§4.3).
#[derive(Debug, Clone, Copy, Serialize)]
pub struct LearningWeights {
    /// Voice weight `w_v`.
    pub w_voice: f64,
    /// Help-request weight `w_h`.
    pub w_help: f64,
    /// Error-talk weight `w_e`.
    pub w_error: f64,
}

impl Default for LearningWeights {
    fn default() -> Self {
        // Scaled so team learning L spans roughly [0, 0.8] across the ψ̄ range,
        // giving the ψ→L slope its Table-5A magnitude (B≈.76) while keeping L
        // bounded. Voice is the broad construct; help / error-talk are the
        // narrower, higher-risk behaviors.
        LearningWeights {
            w_voice: 0.85,
            w_help: 0.55,
            w_error: 0.50,
        }
    }
}

// --------------------------------------------------------------------------- //
// NetworkKind
// --------------------------------------------------------------------------- //

/// Within-team network family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NetworkKind {
    /// Watts–Strogatz small-world (default — design §4.3).
    WattsStrogatz,
    /// Erdős–Rényi G(n,p) — sensitivity.
    ErdosRenyi,
    /// Barabási–Albert preferential attachment — sensitivity.
    BarabasiAlbert,
}

/// Parse a [`NetworkKind`] from a CLI string.
pub fn parse_network_kind(s: &str) -> Result<NetworkKind, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "ws" | "watts-strogatz" | "watts_strogatz" | "small-world" => {
            Ok(NetworkKind::WattsStrogatz)
        }
        "er" | "erdos-renyi" | "erdos_renyi" => Ok(NetworkKind::ErdosRenyi),
        "ba" | "barabasi-albert" | "scale-free" => Ok(NetworkKind::BarabasiAlbert),
        _ => Err(format!(
            "invalid network kind: \"{s}\" (watts-strogatz / erdos-renyi / barabasi-albert)"
        )),
    }
}

// --------------------------------------------------------------------------- //
// Config
// --------------------------------------------------------------------------- //

/// Configuration for a single run.
#[derive(Debug, Clone)]
pub struct Config {
    // ── organisation shape ─────────────────────────────────────────────────
    pub n_teams: usize,
    pub team_size: usize,

    // ── within-team network ─────────────────────────────────────────────────
    pub network_kind: NetworkKind,
    pub network_k: usize,
    pub network_beta: f64,

    // ── decision-mode switch ───────────────────────────────────────────────
    pub decision_mode: DecisionMode,

    // ── ψ-update diff-eq ────────────────────────────────────────────────────
    pub psi: PsiParams,

    // ── voice logit ─────────────────────────────────────────────────────────
    pub voice_beta: VoiceBeta,

    // ── learning aggregation ─────────────────────────────────────────────────
    pub learning_weights: LearningWeights,

    // ── org performance ──────────────────────────────────────────────────────
    /// Learning → performance coefficient `γ_L` (≈ B=.60 mediated).
    pub gamma_l: f64,
    /// Knowledge-stock → performance coefficient `γ_K`.
    pub gamma_k: f64,
    /// Observer-rating noise sd `σ_obs`.
    pub sigma_obs: f64,
    /// Knowledge-stock decay (per step).
    pub knowledge_decay: f64,

    // ── retaliation ──────────────────────────────────────────────────────────
    /// Base per-individual per-step retaliation probability when voicing.
    pub p_retaliate: f64,

    // ── horizon / repeats ──────────────────────────────────────────────────
    pub t_max: u64,
    pub runs: usize,
    pub seed: u64,

    // ── LLM settings (used iff `decision_mode == Llm`) ─────────────────────
    pub llm: LlmSettings,

    // ── output ─────────────────────────────────────────────────────────────
    pub output_dir: String,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            n_teams: 90,
            team_size: 8,
            network_kind: NetworkKind::WattsStrogatz,
            network_k: 6,
            network_beta: 0.15,
            decision_mode: DecisionMode::Rule,
            psi: PsiParams::default(),
            voice_beta: VoiceBeta::default(),
            learning_weights: LearningWeights::default(),
            gamma_l: 0.60,
            gamma_k: 0.20,
            sigma_obs: 0.22,
            knowledge_decay: 0.10,
            p_retaliate: 0.06,
            t_max: 24,
            runs: 30,
            seed: 1999,
            llm: LlmSettings::default(),
            output_dir: "results".to_string(),
        }
    }
}

impl Config {
    /// Total number of individuals.
    pub fn n_individuals(&self) -> usize {
        self.n_teams.saturating_mul(self.team_size)
    }
}

/// JSON representation of a `run`'s `config.json`.
#[derive(Serialize)]
pub struct RunConfigJson {
    pub command: &'static str,
    pub n_teams: usize,
    pub team_size: usize,
    pub n_individuals: usize,
    pub network_kind: NetworkKind,
    pub network_k: usize,
    pub network_beta: f64,
    pub decision_mode: DecisionMode,
    pub psi: PsiParams,
    pub voice_beta: VoiceBeta,
    pub learning_weights: LearningWeights,
    pub gamma_l: f64,
    pub gamma_k: f64,
    pub sigma_obs: f64,
    pub knowledge_decay: f64,
    pub p_retaliate: f64,
    pub t_max: u64,
    pub runs: usize,
    pub seed: u64,
    pub llm_temperature: f32,
    pub llm_seed: u64,
    pub llm_cache_path: Option<String>,
    pub output_dir: String,
}

impl Config {
    /// Build the `config.json` representation.
    pub fn to_run_config_json(&self) -> RunConfigJson {
        RunConfigJson {
            command: "run",
            n_teams: self.n_teams,
            team_size: self.team_size,
            n_individuals: self.n_individuals(),
            network_kind: self.network_kind,
            network_k: self.network_k,
            network_beta: self.network_beta,
            decision_mode: self.decision_mode,
            psi: self.psi,
            voice_beta: self.voice_beta,
            learning_weights: self.learning_weights,
            gamma_l: self.gamma_l,
            gamma_k: self.gamma_k,
            sigma_obs: self.sigma_obs,
            knowledge_decay: self.knowledge_decay,
            p_retaliate: self.p_retaliate,
            t_max: self.t_max,
            runs: self.runs,
            seed: self.seed,
            llm_temperature: self.llm.temperature,
            llm_seed: self.llm.seed,
            llm_cache_path: self.llm.cache_path.clone(),
            output_dir: self.output_dir.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_decision_mode_variants() {
        assert_eq!(parse_decision_mode("rule").unwrap(), DecisionMode::Rule);
        assert_eq!(parse_decision_mode("LLM").unwrap(), DecisionMode::Llm);
        assert!(parse_decision_mode("bogus").is_err());
    }

    #[test]
    fn is_llm_flag() {
        assert!(!DecisionMode::Rule.is_llm());
        assert!(DecisionMode::Llm.is_llm());
    }

    #[test]
    fn parse_network_kind_variants() {
        assert_eq!(
            parse_network_kind("watts-strogatz").unwrap(),
            NetworkKind::WattsStrogatz
        );
        assert_eq!(parse_network_kind("ER").unwrap(), NetworkKind::ErdosRenyi);
        assert_eq!(
            parse_network_kind("ba").unwrap(),
            NetworkKind::BarabasiAlbert
        );
    }

    #[test]
    fn default_n_individuals() {
        let cfg = Config::default();
        assert_eq!(cfg.n_individuals(), 90 * 8);
    }
}
