//! 6 mechanisms across the socsim 6-phase loop.
//!
//! | # | Mechanism                  | Phase       | Role |
//! |---|----------------------------|-------------|------|
//! | 1 | `context_support_update`   | Environment | Update each team's support `s_k` / coaching `c_k` (mild mean-reversion + optional shock); clear transient event counters |
//! | 2 | `voice_decision_rule` / `voice_decision` | Decision | **★ mutually exclusive**: per-individual VOICE/SILENCE — logit (rule) or LLM. Writes `voiced_last` + per-team voice/help/error-talk counters |
//! | 3 | `learning_behavior_aggregate` | Interaction | `L_k ← w_v·V_k/n + w_h·H_k/n + w_e·E_k/n` (event counts over the team) |
//! | 4 | `org_performance`          | Reward      | `Π_k ← γ_L·L_k + γ_K·K_k + N(0, σ_obs²)`; decay-update the knowledge stock K_k |
//! | 5 | `psafety_update`           | PostStep    | **★ core diff-eq** ψ_i ← (1−λ)ψ_i + λ[α·s + β·c − γ·1[retal] + δ·ψ̄_{−i}]; retaliation draw; recompute ψ̄_k. Synchronous (snapshot → batch write) |
//! | 6 | `team_efficacy_update`     | PostStep    | η_k evolves independently of ψ (the H5/H8 discriminant); runs after `psafety_update` so ψ̄ is fresh |
//!
//! The decision mechanisms **snapshot all individuals at step start** and write
//! the new `voiced_last` from the snapshot (synchronous update).

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use rand::Rng;
use socsim_core::{
    derive_seed, AgentId, Mechanism, Phase, Result, SocsimError, StepContext, WorldState,
};
use socsim_llm::MetadataCollector;

use crate::config::{LearningWeights, LlmSettings, PsiParams, VoiceBeta};
use crate::llm::{llm_config, VoiceClient};
use crate::prompts::{build_voice_prompt, parse_voice_decision, Behavior};
use crate::world::{TeamId, TeamWorld};

// --------------------------------------------------------------------------- //
// Shared LLM client / metadata wrappers (mirrors detert2011 / knoll2013)
// --------------------------------------------------------------------------- //

/// Shared LLM client between driver + mechanism (`Rc<RefCell>` pattern).
pub type SharedClient = Rc<RefCell<VoiceClient>>;
/// Shared metadata collector for cache-hit rate / call count.
pub type SharedMetadata = Rc<RefCell<MetadataCollector>>;

/// Sigmoid (logistic) function.
#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Approximately-Gaussian draw clamped to `[0, 1]` (sum-of-uniforms; cheap and
/// deterministic given the RNG).
pub fn gauss_clamped(rng: &mut socsim_core::SimRng, mean: f64, sd: f64) -> f64 {
    let u: f64 = (0..4).map(|_| rng.gen::<f64>()).sum::<f64>() / 4.0;
    let z = (u - 0.5) * (12.0_f64).sqrt();
    (mean + sd * z).clamp(0.0, 1.0)
}

// --------------------------------------------------------------------------- //
// 1. ContextSupportUpdate  (Environment)
// --------------------------------------------------------------------------- //

/// Updates each team's context support `s_k` / coaching `c_k` with mild
/// mean-reversion toward their own (heterogeneous) target, plus an optional
/// step-`shock_t` exogenous downsizing dip. Also clears the transient per-team
/// event counters at step start.
pub struct ContextSupportUpdate {
    decay: f64,
    shock_t: Option<u64>,
    shock_magnitude: f64,
    /// Per-team support / coaching targets (heterogeneous, frozen at init).
    targets: BTreeMap<TeamId, (f64, f64)>,
}

impl ContextSupportUpdate {
    pub fn new(
        shock_t: Option<u64>,
        shock_magnitude: f64,
        targets: BTreeMap<TeamId, (f64, f64)>,
    ) -> Self {
        ContextSupportUpdate {
            decay: 0.10,
            shock_t,
            shock_magnitude,
            targets,
        }
    }
}

impl Mechanism<TeamWorld> for ContextSupportUpdate {
    fn name(&self) -> &str {
        "context_support_update"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Environment]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, TeamWorld>) -> Result<()> {
        ctx.world.reset_counters();
        let shock = self.shock_t.map(|ts| ctx.clock.t() == ts).unwrap_or(false);
        let team_ids: Vec<TeamId> = ctx.world.teams.keys().copied().collect();
        for k in team_ids {
            let (ts, tc) = self.targets.get(&k).copied().unwrap_or((0.5, 0.5));
            if let Some(team) = ctx.world.teams.get_mut(&k) {
                let mut s = team.support + self.decay * (ts - team.support);
                let mut c = team.coaching + self.decay * (tc - team.coaching);
                if shock {
                    s = (s - self.shock_magnitude).clamp(0.0, 1.0);
                    c = (c - self.shock_magnitude).clamp(0.0, 1.0);
                }
                team.support = s.clamp(0.0, 1.0);
                team.coaching = c.clamp(0.0, 1.0);
            }
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 2a. VoiceDecisionRule  (Decision) — logit
// --------------------------------------------------------------------------- //

/// `voice_decision_rule` mechanism — the §4.3 VOICE logit. Each individual votes
/// VOICE/SILENCE by an independent Bernoulli draw; the learning behavior on
/// voice is chosen by a second draw weighted toward speak.
pub struct VoiceDecisionRule {
    beta: VoiceBeta,
}

impl VoiceDecisionRule {
    pub fn new(beta: VoiceBeta) -> Self {
        VoiceDecisionRule { beta }
    }
}

/// Snapshot of the features the VOICE logit consumes for one individual.
struct VoiceFeatures {
    id: AgentId,
    team: TeamId,
    psi: f64,
    fear: f64,
    ivt: f64,
    concern: f64,
}

impl Mechanism<TeamWorld> for VoiceDecisionRule {
    fn name(&self) -> &str {
        "voice_decision_rule"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Decision]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, TeamWorld>) -> Result<()> {
        // Synchronous: snapshot first, then write.
        let ids: Vec<AgentId> = ctx.world.agent_ids();
        let mut snapshot: Vec<VoiceFeatures> = Vec::with_capacity(ids.len());
        for id in &ids {
            let ind = &ctx.world.individuals[id];
            snapshot.push(VoiceFeatures {
                id: *id,
                team: ind.team,
                psi: ind.psi,
                fear: ind.fear,
                ivt: ind.ivt,
                concern: ind.private_concern,
            });
        }

        // (id, voiced, behavior)
        let mut updates: Vec<(AgentId, TeamId, bool, Option<Behavior>)> =
            Vec::with_capacity(ids.len());
        for f in snapshot {
            let logit = self.beta.intercept + self.beta.beta_psafety * f.psi
                - self.beta.beta_fear * f.fear
                - self.beta.beta_ivt * f.ivt
                - self.beta.beta_concern * f.concern;
            let p_voice = sigmoid(logit);
            let u_voice: f64 = ctx.rng.gen();
            // Draw the behavior selector up front so the RNG sequence is
            // independent of the VOICE/SILENCE branch taken.
            let u_beh: f64 = ctx.rng.gen();
            if u_voice < p_voice {
                // Behavior mix: speak 0.45 / help 0.30 / error_talk 0.25.
                let beh = if u_beh < 0.45 {
                    Behavior::Speak
                } else if u_beh < 0.75 {
                    Behavior::Help
                } else {
                    Behavior::ErrorTalk
                };
                updates.push((f.id, f.team, true, Some(beh)));
            } else {
                updates.push((f.id, f.team, false, None));
            }
        }
        write_decisions(ctx.world, updates);
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 2b. VoiceDecisionLlm  (Decision) — LLM-driven
// --------------------------------------------------------------------------- //

/// LLM-driven voice decision (Phase-3 ablation).
pub struct VoiceDecisionLlm {
    client: SharedClient,
    metadata: SharedMetadata,
    settings: LlmSettings,
    /// `derive_seed` root for the (agent_id, t) LLM seed stream.
    llm_seed_root: u64,
}

impl VoiceDecisionLlm {
    pub fn new(
        client: SharedClient,
        metadata: SharedMetadata,
        settings: LlmSettings,
        llm_seed_root: u64,
    ) -> Self {
        VoiceDecisionLlm {
            client,
            metadata,
            settings,
            llm_seed_root,
        }
    }
}

impl Mechanism<TeamWorld> for VoiceDecisionLlm {
    fn name(&self) -> &str {
        "voice_decision"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Decision]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, TeamWorld>) -> Result<()> {
        let ids: Vec<AgentId> = ctx.world.agent_ids();
        let t = ctx.clock.t();
        let mut prompts: Vec<(AgentId, TeamId, String, u64)> = Vec::with_capacity(ids.len());
        for id in ids {
            let team = ctx.world.individuals[&id].team;
            let prompt = build_voice_prompt(ctx.world, id);
            let llm_seed = derive_seed(self.llm_seed_root, &[3, id.0, t]);
            prompts.push((id, team, prompt, llm_seed));
        }

        let mut updates: Vec<(AgentId, TeamId, bool, Option<Behavior>)> =
            Vec::with_capacity(prompts.len());
        for (id, team, prompt, llm_seed) in prompts {
            let mut cfg = llm_config(&self.settings);
            cfg.seed = llm_seed;
            let text = {
                let mut client = self.client.borrow_mut();
                let resp = client.complete(&prompt, &cfg).map_err(|e| {
                    SocsimError::Mechanism(format!("voice_decision LLM call failed: {e}"))
                })?;
                self.metadata.borrow_mut().record(resp.metadata.clone());
                resp.text
            };
            let v = parse_voice_decision(&text);
            updates.push((id, team, v.voiced, v.behavior));
        }
        write_decisions(ctx.world, updates);
        Ok(())
    }
}

/// Apply a batch of decisions: write `voiced_last` and bump per-team counters.
fn write_decisions(world: &mut TeamWorld, updates: Vec<(AgentId, TeamId, bool, Option<Behavior>)>) {
    for (id, team, voiced, beh) in updates {
        if let Some(ind) = world.individuals.get_mut(&id) {
            ind.voiced_last = voiced;
        }
        if voiced {
            *world.voice_count_by_team.entry(team).or_insert(0) += 1;
            match beh {
                Some(Behavior::Help) => {
                    *world.help_count_by_team.entry(team).or_insert(0) += 1;
                }
                Some(Behavior::ErrorTalk) => {
                    *world.error_talk_by_team.entry(team).or_insert(0) += 1;
                }
                _ => {}
            }
        }
    }
}

// --------------------------------------------------------------------------- //
// 3. LearningBehaviorAggregate  (Interaction)
// --------------------------------------------------------------------------- //

/// `L_k ← w_v·(V_k/n_k) + w_h·(H_k/n_k) + w_e·(E_k/n_k)` from the transient
/// per-team event counters.
pub struct LearningBehaviorAggregate {
    weights: LearningWeights,
}

impl LearningBehaviorAggregate {
    pub fn new(weights: LearningWeights) -> Self {
        LearningBehaviorAggregate { weights }
    }
}

impl Mechanism<TeamWorld> for LearningBehaviorAggregate {
    fn name(&self) -> &str {
        "learning_behavior_aggregate"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Interaction]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, TeamWorld>) -> Result<()> {
        let team_ids: Vec<TeamId> = ctx.world.teams.keys().copied().collect();
        for k in team_ids {
            let n = ctx.world.teams[&k].members.len().max(1) as f64;
            let v = *ctx.world.voice_count_by_team.get(&k).unwrap_or(&0) as f64;
            let h = *ctx.world.help_count_by_team.get(&k).unwrap_or(&0) as f64;
            let e = *ctx.world.error_talk_by_team.get(&k).unwrap_or(&0) as f64;
            let l = self.weights.w_voice * (v / n)
                + self.weights.w_help * (h / n)
                + self.weights.w_error * (e / n);
            if let Some(team) = ctx.world.teams.get_mut(&k) {
                team.learning = l.clamp(0.0, 1.0);
            }
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 4. OrgPerformance  (Reward)
// --------------------------------------------------------------------------- //

/// `Π_k ← γ_L·L_k + γ_K·K_k + N(0, σ_obs²)`, decay-updating the knowledge stock
/// `K_k` (stored in `efficacy`-adjacent state via a private accumulator). Here
/// the knowledge stock is folded into the performance signal through a per-team
/// EMA carried in a side map.
pub struct OrgPerformance {
    gamma_l: f64,
    gamma_k: f64,
    sigma_obs: f64,
    decay: f64,
    obs_seed_root: u64,
    /// Per-team knowledge stock `K_k(t)` (EMA of learning).
    knowledge: BTreeMap<TeamId, f64>,
}

impl OrgPerformance {
    pub fn new(gamma_l: f64, gamma_k: f64, sigma_obs: f64, decay: f64, obs_seed_root: u64) -> Self {
        OrgPerformance {
            gamma_l,
            gamma_k,
            sigma_obs,
            decay,
            obs_seed_root,
            knowledge: BTreeMap::new(),
        }
    }
}

impl Mechanism<TeamWorld> for OrgPerformance {
    fn name(&self) -> &str {
        "org_performance"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Reward]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, TeamWorld>) -> Result<()> {
        let t = ctx.clock.t();
        let team_ids: Vec<TeamId> = ctx.world.teams.keys().copied().collect();
        for k in team_ids {
            let l = ctx.world.teams[&k].learning;
            // Knowledge stock EMA: K ← (1−decay)·K + L.
            let k_stock = {
                let entry = self.knowledge.entry(k).or_insert(0.0);
                *entry = (1.0 - self.decay) * *entry + l;
                *entry
            };
            // Observer-rating noise from a dedicated, reproducible stream. A
            // per-team *frozen* component (rater idiosyncrasy / halo) plus a
            // small per-tick component. The frozen part does not average out
            // over ticks, so it caps the L→Π R² near the paper's .26 (otherwise
            // time-averaging would wash the noise away and inflate R²).
            let noise = {
                let frozen_seed = derive_seed(self.obs_seed_root, &[k as u64]);
                let mut rng_f = socsim_core::SimRng::from_seed(frozen_seed);
                let uf: f64 = (0..12).map(|_| rng_f.gen::<f64>()).sum::<f64>() - 6.0;
                let tick_seed = derive_seed(self.obs_seed_root, &[k as u64, t]);
                let mut rng_t = socsim_core::SimRng::from_seed(tick_seed);
                let ut: f64 = (0..12).map(|_| rng_t.gen::<f64>()).sum::<f64>() - 6.0;
                (uf * 0.85 + ut * 0.35) * self.sigma_obs
            };
            let pi = self.gamma_l * l + self.gamma_k * k_stock + noise;
            if let Some(team) = ctx.world.teams.get_mut(&k) {
                team.performance = pi;
            }
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 5. PsafetyUpdate  (PostStep) — ★ core diff-eq
// --------------------------------------------------------------------------- //

/// Weight of the individual fear disposition in the ψ target (within-team).
const DISP_FEAR: f64 = 0.60;
/// Weight of the individual implicit-voice-norm disposition in the ψ target.
const DISP_IVT: f64 = 0.40;
/// Weight of the (centered) private-concern disposition in the ψ target.
const DISP_CONCERN: f64 = 0.60;

/// `ψ_i(t+1) = (1−λ)ψ_i + λ[α·s_k + β·c_k − γ·1[retaliated] + δ·ψ̄_{−i,k}]`.
///
/// Retaliation is drawn here: a voicing individual is retaliated against with
/// probability `p_retaliate · (1 − s_k)` (low support → higher retaliation risk),
/// from a dedicated reproducible stream. Synchronous: snapshot ψ → batch write →
/// recompute ψ̄_k → track convergence.
pub struct PsafetyUpdate {
    params: PsiParams,
    p_retaliate: f64,
    retal_seed_root: u64,
}

impl PsafetyUpdate {
    pub fn new(params: PsiParams, p_retaliate: f64, retal_seed_root: u64) -> Self {
        PsafetyUpdate {
            params,
            p_retaliate,
            retal_seed_root,
        }
    }
}

impl Mechanism<TeamWorld> for PsafetyUpdate {
    fn name(&self) -> &str {
        "psafety_update"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::PostStep]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, TeamWorld>) -> Result<()> {
        let t = ctx.clock.t();
        let ids: Vec<AgentId> = ctx.world.agent_ids();
        let p = self.params;

        // Snapshot ψ̄_{-i} (computed from the *old* ψ snapshot) + retaliation.
        let prev_psi_bar: BTreeMap<TeamId, f64> = ctx
            .world
            .teams
            .iter()
            .map(|(k, t)| (*k, t.psi_bar))
            .collect();

        let mut new_psi: Vec<(AgentId, f64, bool)> = Vec::with_capacity(ids.len());
        for id in &ids {
            let ind = &ctx.world.individuals[id];
            let team = ind.team;
            let (support, coaching) = {
                let t = &ctx.world.teams[&team];
                (t.support, t.coaching)
            };
            // Retaliation draw: only voicing individuals are exposed; risk rises
            // as support falls. Dedicated reproducible stream.
            let retaliated = if ind.voiced_last {
                let seed = derive_seed(self.retal_seed_root, &[id.0, t]);
                let mut rng = socsim_core::SimRng::from_seed(seed);
                let risk = self.p_retaliate * (1.0 - support);
                rng.gen::<f64>() < risk
            } else {
                false
            };
            let psi_bar_excl = ctx.world.psi_bar_excluding(team, *id);
            // Persistent individual disposition: fear and implicit-voice norms
            // lower an individual's perceived safety. These vary *within* a team,
            // so members do not collapse to one shared value — the residual
            // within-team variance is what keeps ICC(ψ) finite (Kenny & LaVoie
            // 1985: group vs individual effects), reproducing the paper's
            // ICC(ψ) ≈ .39 rather than a degenerate 1.0.
            let disposition = -DISP_FEAR * ind.fear - DISP_IVT * ind.ivt
                + DISP_CONCERN * (ind.private_concern - 0.5);
            let target = p.alpha * support + p.beta * coaching
                - p.gamma * (retaliated as i32 as f64)
                + p.delta * psi_bar_excl
                + disposition;
            let psi_new = ((1.0 - p.lambda) * ind.psi + p.lambda * target).clamp(0.0, 1.0);
            new_psi.push((*id, psi_new, retaliated));
        }

        // Batch write.
        for (id, psi, retaliated) in new_psi {
            if let Some(ind) = ctx.world.individuals.get_mut(&id) {
                ind.psi = psi;
                ind.retaliated_last = retaliated;
            }
        }

        // Recompute ψ̄_k and convergence delta.
        ctx.world.recompute_psi_bar();
        let mut max_delta = 0.0f64;
        for (k, prev) in &prev_psi_bar {
            if let Some(team) = ctx.world.teams.get(k) {
                max_delta = max_delta.max((team.psi_bar - prev).abs());
            }
        }
        ctx.world.last_max_delta = max_delta;
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 6. TeamEfficacyUpdate  (PostStep) — discriminant construct
// --------------------------------------------------------------------------- //

/// Evolves team efficacy `η_k` toward a per-team **independent** latent target
/// (drawn once per team from a dedicated stream), so that — controlling for ψ —
/// η carries little information about learning behavior (the paper's H5/H8
/// discriminant: |t| < 2). Efficacy is a *distinct* construct from ψ: a team can
/// feel collectively capable without feeling interpersonally safe. Runs after
/// `psafety_update` so ψ̄ is fresh.
pub struct TeamEfficacyUpdate {
    rate: f64,
    /// Per-team independent efficacy target (frozen at first apply).
    targets: BTreeMap<TeamId, f64>,
    target_seed_root: u64,
}

impl TeamEfficacyUpdate {
    pub fn new(target_seed_root: u64) -> Self {
        TeamEfficacyUpdate {
            rate: 0.15,
            targets: BTreeMap::new(),
            target_seed_root,
        }
    }
}

impl Mechanism<TeamWorld> for TeamEfficacyUpdate {
    fn name(&self) -> &str {
        "team_efficacy_update"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::PostStep]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, TeamWorld>) -> Result<()> {
        let team_ids: Vec<TeamId> = ctx.world.teams.keys().copied().collect();
        for k in team_ids {
            let target = *self.targets.entry(k).or_insert_with(|| {
                let seed = derive_seed(self.target_seed_root, &[k as u64]);
                let mut rng = socsim_core::SimRng::from_seed(seed);
                gauss_clamped(&mut rng, 0.5, 0.18)
            });
            if let Some(team) = ctx.world.teams.get_mut(&k) {
                team.efficacy =
                    ((1.0 - self.rate) * team.efficacy + self.rate * target).clamp(0.0, 1.0);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_at_zero_is_half() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn gauss_clamped_in_range() {
        let mut rng = socsim_core::SimRng::from_seed(3);
        for _ in 0..100 {
            let v = gauss_clamped(&mut rng, 0.5, 0.2);
            assert!((0.0..=1.0).contains(&v));
        }
    }
}
