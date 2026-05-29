//! Edmondson (1999) — Psychological Safety and Team Learning simulation.
//!
//! A socsim-based team-level ABM of the causal chain
//!
//! ```text
//! context support / coaching → ψ (psychological safety) → learning behavior L → team performance Π
//! ```
//!
//! Individuals belong to fixed teams and live on a **within-team** Watts–Strogatz
//! network (no cross-team edges). Each step runs six mechanisms across socsim's
//! 6-phase loop:
//!
//! 1. `context_support_update` (Environment) — evolve team support / coaching.
//! 2. `voice_decision_rule` / `voice_decision` (Decision) — **mutually exclusive**
//!    per-individual VOICE/SILENCE: a rule-based logit (default, deterministic,
//!    zero LLM calls) or an LLM-driven decision (`--decision-mode llm`).
//! 3. `learning_behavior_aggregate` (Interaction) — `L_k` from team voice/help/
//!    error-talk counts.
//! 4. `org_performance` (Reward) — `Π_k = γ_L·L_k + γ_K·K_k + N(0, σ_obs²)`.
//! 5. `psafety_update` (PostStep) — the **core** ψ difference equation.
//! 6. `team_efficacy_update` (PostStep) — the discriminant efficacy construct.
//!
//! The §5 calibration anchors (ICC(ψ)≈.39, ψ→L B≈.76 R²≈.63, L→Π R²≈.26,
//! support→ψ B≈.56, mediated ψ residual ns, efficacy |t|<2) are reconstructed
//! by `metrics.rs` (local OLS + ICC + Baron & Kenny three-step) and surfaced by
//! the `reproduce` subcommand and the Python `edmondson-tools reproduce`.
//!
//! See `simulation/src/main.rs` for the `run` / `sweep` / `reproduce` CLI.

pub mod config;
pub mod llm;
pub mod mechanisms;
pub mod metrics;
pub mod prompts;
pub mod simulation;
pub mod world;
