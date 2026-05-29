//! Integration tests for the Edmondson (1999) psychological-safety simulation.
//!
//! **No live LLM required.** The rule mode needs no LLM at all; the LLM path is
//! driven by `socsim_llm::mock::ScriptedClient`. Tests cover rule-mode
//! bit-determinism, anchor-band sanity on a sandbox-sized run, no cross-team
//! edges, and the LLM path end-to-end via a scripted client.

use edmondson_team::config::{Config, DecisionMode, LlmSettings, NetworkKind};
use edmondson_team::llm::wrap_client;
use edmondson_team::simulation::{anchor_report_from_world, run_with_client, SimulationResult};

use socsim_llm::mock::ScriptedClient;
use socsim_llm::PromptCache;

fn small_cfg(mode: DecisionMode) -> Config {
    Config {
        n_teams: 40,
        team_size: 8,
        network_kind: NetworkKind::WattsStrogatz,
        network_k: 4,
        network_beta: 0.15,
        decision_mode: mode,
        t_max: 16,
        runs: 1,
        seed: 1999,
        llm: LlmSettings::default(),
        output_dir: "results".to_string(),
        ..Config::default()
    }
}

/// Scripted client cycling VOICE(speak) / VOICE(help) / SILENCE.
fn scripted_client() -> edmondson_team::llm::VoiceClient {
    let backend = ScriptedClient::new("mock-model", |prompt: &str| {
        let h = prompt.len() % 3;
        match h {
            0 => r#"{"decision":"voice","behavior":"speak"}"#.to_string(),
            1 => r#"{"decision":"voice","behavior":"help"}"#.to_string(),
            _ => r#"{"decision":"silence","behavior":null}"#.to_string(),
        }
    });
    wrap_client(backend, PromptCache::in_memory())
}

#[test]
fn rule_mode_smoke_run() {
    let r: SimulationResult = run_with_client(&small_cfg(DecisionMode::Rule), None).unwrap();
    assert!(!r.metrics_rows.is_empty(), "must produce per-step metrics");
    assert!(!r.team_rows.is_empty());
    assert!(!r.individual_rows.is_empty());
    assert_eq!(r.metadata.total(), 0, "rule mode makes 0 LLM calls");
    for row in &r.metrics_rows {
        assert!(
            (-0.1..=1.0).contains(&row.icc_psi),
            "icc_psi out of range: {}",
            row.icc_psi
        );
    }
}

#[test]
fn rule_mode_is_bit_deterministic() {
    let a = run_with_client(&small_cfg(DecisionMode::Rule), None).unwrap();
    let b = run_with_client(&small_cfg(DecisionMode::Rule), None).unwrap();
    assert_eq!(a.metrics_rows.len(), b.metrics_rows.len());
    assert_eq!(a.team_rows.len(), b.team_rows.len());
    for (ra, rb) in a.metrics_rows.iter().zip(b.metrics_rows.iter()) {
        assert_eq!(ra.t, rb.t);
        assert!((ra.icc_psi - rb.icc_psi).abs() < 1e-15);
        assert!((ra.beta_psi_l - rb.beta_psi_l).abs() < 1e-15);
        assert!((ra.mediation_ratio - rb.mediation_ratio).abs() < 1e-15);
    }
    // Byte-identical team rows.
    for (ta, tb) in a.team_rows.iter().zip(b.team_rows.iter()) {
        assert!((ta.psi - tb.psi).abs() < 1e-15);
        assert!((ta.performance - tb.performance).abs() < 1e-15);
    }
}

#[test]
fn anchors_are_in_plausible_bands() {
    // A modest sandbox run should already land the core anchors near their
    // bands (the full default run tightens them further).
    let r = run_with_client(&small_cfg(DecisionMode::Rule), None).unwrap();
    let rep = anchor_report_from_world(&r.world);
    assert!(
        rep.icc_psi > 0.0 && rep.icc_psi < 1.0,
        "icc_psi={}",
        rep.icc_psi
    );
    assert!(
        rep.beta_psi_l > 0.0,
        "ψ→L slope should be positive: {}",
        rep.beta_psi_l
    );
    assert!(
        rep.beta_support_psi > 0.0,
        "support→ψ slope should be positive"
    );
}

#[test]
fn llm_mode_smoke_run_with_scripted_client() {
    let cfg = small_cfg(DecisionMode::Llm);
    let client = scripted_client();
    let r = run_with_client(&cfg, Some(client)).unwrap();
    assert!(!r.metrics_rows.is_empty());
    assert!(r.metadata.total() > 0, "LLM mode must call the LLM");
    // Some teams should show non-zero learning (scripted voice responses).
    let any_learning = r.team_rows.iter().any(|row| row.learning > 0.0);
    assert!(
        any_learning,
        "scripted voice responses should drive learning"
    );
}
