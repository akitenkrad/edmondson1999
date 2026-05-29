//! World state for the Edmondson (1999) psychological-safety & team-learning model.
//!
//! Implements socsim's [`WorldState`] over `Individual`s who belong to fixed
//! `Team`s and live on a within-team [`SocialNetwork`] (Watts–Strogatz
//! small-world; **no cross-team edges**). Each individual carries a persistent
//! psychological-safety belief `ψ_i`, fear propensity `f_i`, implicit-voice-
//! theory strength `θ_i` (ivt), a private concern, and transient flags
//! (`retaliated_last`, `voiced_last`). Each team aggregates the shared belief
//! `ψ̄_k`, learning behavior `L_k`, performance `Π_k`, and efficacy `η_k`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use socsim_core::{AgentId, SimClock, WorldState};
use socsim_net::SocialNetwork;

/// Stable team identifier.
pub type TeamId = u32;

// --------------------------------------------------------------------------- //
// Individual
// --------------------------------------------------------------------------- //

/// Per-individual (employee) state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Individual {
    /// Team membership.
    pub team: TeamId,
    /// Individual-level psychological safety `ψ_i ∈ [0, 1]` (7-point scale → [0,1]).
    pub psi: f64,
    /// Fear propensity `f_i ∈ [0, 1]` (Kish-Gephart et al. 2009).
    pub fear: f64,
    /// Implicit voice theory strength `θ_i ∈ [0, 1]` (Detert & Edmondson 2011).
    pub ivt: f64,
    /// Private concern intensity `c_i ∈ [0, 1]`.
    pub private_concern: f64,
    /// Whether this individual was retaliated against in the current step
    /// (transient buffer, written by `psafety_update`'s retaliation draw).
    pub retaliated_last: bool,
    /// Whether this individual voiced in the current step (transient buffer).
    pub voiced_last: bool,
}

impl Individual {
    /// Initialise a "neutral" individual; per-attribute random draws happen at
    /// the call site (`simulation::init_world`).
    pub fn neutral(team: TeamId) -> Self {
        Individual {
            team,
            psi: 0.5,
            fear: 0.3,
            ivt: 0.45,
            private_concern: 0.5,
            retaliated_last: false,
            voiced_last: false,
        }
    }
}

// --------------------------------------------------------------------------- //
// Team
// --------------------------------------------------------------------------- //

/// Per-team aggregate state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Team {
    /// Sorted member [`AgentId`]s.
    pub members: Vec<AgentId>,
    /// Context support `s_k(t) ∈ [0, 1]` (Hackman 1987 work-design support).
    pub support: f64,
    /// Leader coaching `c_k(t) ∈ [0, 1]`.
    pub coaching: f64,
    /// Aggregated shared belief `ψ̄_k` (mean of members' `ψ_i`).
    pub psi_bar: f64,
    /// Learning behavior `L_k(t) ∈ [0, 1]` (7-item scale composite).
    pub learning: f64,
    /// Observer-rated team performance `Π_k(t)`.
    pub performance: f64,
    /// Team efficacy `η_k(t)` (discriminant construct; H5/H8 check).
    pub efficacy: f64,
}

impl Team {
    /// A team with the given members and neutral aggregates.
    pub fn new(members: Vec<AgentId>) -> Self {
        Team {
            members,
            support: 0.5,
            coaching: 0.5,
            psi_bar: 0.5,
            learning: 0.0,
            performance: 0.0,
            efficacy: 0.5,
        }
    }
}

// --------------------------------------------------------------------------- //
// TeamWorld
// --------------------------------------------------------------------------- //

/// World state for the psychological-safety & team-learning model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TeamWorld {
    pub clock: SimClock,
    /// Individuals keyed by sorted [`AgentId`] (sorted keys = determinism).
    pub individuals: BTreeMap<AgentId, Individual>,
    /// Teams keyed by sorted [`TeamId`].
    pub teams: BTreeMap<TeamId, Team>,
    /// Within-team inter-individual network (Watts–Strogatz; no cross-team edges).
    pub network: SocialNetwork,
    /// Transient per-team voice-event counters (cleared each step start).
    pub voice_count_by_team: BTreeMap<TeamId, u32>,
    /// Transient per-team help-request counters (cleared each step start).
    pub help_count_by_team: BTreeMap<TeamId, u32>,
    /// Transient per-team error-talk counters (cleared each step start).
    pub error_talk_by_team: BTreeMap<TeamId, u32>,
    /// Largest `|Δψ̄_k|` observed last step (convergence tracking).
    pub last_max_delta: f64,
}

impl TeamWorld {
    /// Build a world from individuals + teams + a within-team network.
    pub fn new(
        clock: SimClock,
        individuals: BTreeMap<AgentId, Individual>,
        teams: BTreeMap<TeamId, Team>,
        network: SocialNetwork,
    ) -> Self {
        let mut w = TeamWorld {
            clock,
            individuals,
            teams,
            network,
            voice_count_by_team: BTreeMap::new(),
            help_count_by_team: BTreeMap::new(),
            error_talk_by_team: BTreeMap::new(),
            last_max_delta: f64::INFINITY,
        };
        w.reset_counters();
        w.recompute_psi_bar();
        w
    }

    /// Number of individuals.
    pub fn n_individuals(&self) -> usize {
        self.individuals.len()
    }

    /// Number of teams.
    pub fn n_teams(&self) -> usize {
        self.teams.len()
    }

    /// Clear the transient per-team event counters (called at step start).
    pub fn reset_counters(&mut self) {
        self.voice_count_by_team.clear();
        self.help_count_by_team.clear();
        self.error_talk_by_team.clear();
        for &k in self.teams.keys() {
            self.voice_count_by_team.insert(k, 0);
            self.help_count_by_team.insert(k, 0);
            self.error_talk_by_team.insert(k, 0);
        }
    }

    /// Recompute every team's `ψ̄_k` from its members' current `ψ_i`.
    pub fn recompute_psi_bar(&mut self) {
        let ids: Vec<TeamId> = self.teams.keys().copied().collect();
        for k in ids {
            let members = self.teams[&k].members.clone();
            let psi_bar = if members.is_empty() {
                0.0
            } else {
                let s: f64 = members
                    .iter()
                    .filter_map(|id| self.individuals.get(id))
                    .map(|i| i.psi)
                    .sum();
                s / members.len() as f64
            };
            if let Some(team) = self.teams.get_mut(&k) {
                team.psi_bar = psi_bar;
            }
        }
    }

    /// Mean `ψ_{-i,k}` over team `k`'s members *excluding* `i` (shared-belief
    /// convergence term). Falls back to the team mean for a singleton team.
    pub fn psi_bar_excluding(&self, team: TeamId, exclude: AgentId) -> f64 {
        let Some(t) = self.teams.get(&team) else {
            return 0.0;
        };
        let mut sum = 0.0;
        let mut n = 0usize;
        for id in &t.members {
            if *id == exclude {
                continue;
            }
            if let Some(ind) = self.individuals.get(id) {
                sum += ind.psi;
                n += 1;
            }
        }
        if n == 0 {
            self.individuals.get(&exclude).map(|i| i.psi).unwrap_or(0.0)
        } else {
            sum / n as f64
        }
    }
}

impl WorldState for TeamWorld {
    fn agent_ids(&self) -> Vec<AgentId> {
        // BTreeMap keys are already sorted — canonical activation order.
        self.individuals.keys().copied().collect()
    }

    fn clock(&self) -> &SimClock {
        &self.clock
    }

    fn clock_mut(&mut self) -> &mut SimClock {
        &mut self.clock
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use socsim_core::SimRng;

    fn tiny_world() -> TeamWorld {
        let mut rng = SimRng::from_seed(7);
        let ids: Vec<AgentId> = (0..4).map(AgentId).collect();
        let net = SocialNetwork::erdos_renyi(&ids, 0.5, &mut rng);
        let mut inds: BTreeMap<AgentId, Individual> = BTreeMap::new();
        for (i, &id) in ids.iter().enumerate() {
            let mut ind = Individual::neutral(0);
            ind.psi = 0.2 * (i as f64 + 1.0); // 0.2, 0.4, 0.6, 0.8
            inds.insert(id, ind);
        }
        let mut teams: BTreeMap<TeamId, Team> = BTreeMap::new();
        teams.insert(0, Team::new(ids.clone()));
        TeamWorld::new(SimClock::new(1), inds, teams, net)
    }

    #[test]
    fn psi_bar_is_member_mean() {
        let w = tiny_world();
        assert!((w.teams[&0].psi_bar - 0.5).abs() < 1e-12);
    }

    #[test]
    fn psi_bar_excluding_drops_one_member() {
        let w = tiny_world();
        // Exclude agent 0 (ψ=0.2): mean of {0.4, 0.6, 0.8} = 0.6.
        assert!((w.psi_bar_excluding(0, AgentId(0)) - 0.6).abs() < 1e-12);
    }

    #[test]
    fn psi_bar_excluding_singleton_falls_back_to_self() {
        let mut rng = SimRng::from_seed(1);
        let ids = vec![AgentId(0)];
        let net = SocialNetwork::erdos_renyi(&ids, 0.0, &mut rng);
        let mut inds = BTreeMap::new();
        let mut ind = Individual::neutral(0);
        ind.psi = 0.7;
        inds.insert(AgentId(0), ind);
        let mut teams = BTreeMap::new();
        teams.insert(0, Team::new(ids));
        let w = TeamWorld::new(SimClock::new(1), inds, teams, net);
        assert!((w.psi_bar_excluding(0, AgentId(0)) - 0.7).abs() < 1e-12);
    }

    #[test]
    fn reset_counters_zeroes_all_teams() {
        let mut w = tiny_world();
        w.voice_count_by_team.insert(0, 5);
        w.reset_counters();
        assert_eq!(w.voice_count_by_team[&0], 0);
    }
}
