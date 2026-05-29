//! Paper-specific metrics for the Edmondson (1999) model.
//!
//! ICC, simple/multiple OLS regression, and the Baron & Kenny (1986) three-step
//! mediation are **paper-specific** and implemented locally (not pushed to
//! `socsim-metrics`). The canonical `mean` / `variance` from `socsim-metrics`
//! feed the ICC MS_between / MS_within computation.
//!
//! All cross-sectional regressions operate on **team-level** vectors taken from
//! the final-tick snapshot (one observation per team), matching the paper's
//! 51-team cross-section.

use socsim_metrics::stats;

use crate::world::TeamWorld;

// --------------------------------------------------------------------------- //
// Team-level extractors
// --------------------------------------------------------------------------- //

/// Team-level `ψ̄_k` over teams (sorted by `TeamId`).
pub fn team_psi(world: &TeamWorld) -> Vec<f64> {
    world.teams.values().map(|t| t.psi_bar).collect()
}

/// Team-level learning `L_k`.
pub fn team_learning(world: &TeamWorld) -> Vec<f64> {
    world.teams.values().map(|t| t.learning).collect()
}

/// Team-level performance `Π_k`.
pub fn team_performance(world: &TeamWorld) -> Vec<f64> {
    world.teams.values().map(|t| t.performance).collect()
}

/// Team-level context support `s_k`.
pub fn team_support(world: &TeamWorld) -> Vec<f64> {
    world.teams.values().map(|t| t.support).collect()
}

/// Team-level efficacy `η_k`.
pub fn team_efficacy(world: &TeamWorld) -> Vec<f64> {
    world.teams.values().map(|t| t.efficacy).collect()
}

// --------------------------------------------------------------------------- //
// ICC — intraclass correlation
// --------------------------------------------------------------------------- //

/// One-way-ANOVA ICC(1) for a per-individual value grouped by team.
///
/// `ICC = (MS_between − MS_within) / (MS_between + (k̄ − 1)·MS_within)` where the
/// MS terms come from the between/within sums of squares and `k̄` is the mean
/// group size. Returns 0 for degenerate input. `value` maps an `Individual` to
/// the quantity of interest (e.g. `|i| i.psi`).
pub fn icc_grouped<F>(world: &TeamWorld, value: F) -> f64
where
    F: Fn(&crate::world::Individual) -> f64,
{
    // Gather per-team value vectors.
    let mut groups: Vec<Vec<f64>> = Vec::new();
    for team in world.teams.values() {
        let mut g = Vec::with_capacity(team.members.len());
        for id in &team.members {
            if let Some(ind) = world.individuals.get(id) {
                g.push(value(ind));
            }
        }
        if !g.is_empty() {
            groups.push(g);
        }
    }
    icc_from_groups(&groups)
}

/// ICC(1) from already-grouped observations.
pub fn icc_from_groups(groups: &[Vec<f64>]) -> f64 {
    let k = groups.len();
    if k < 2 {
        return 0.0;
    }
    let n_total: usize = groups.iter().map(|g| g.len()).sum();
    if n_total == 0 {
        return 0.0;
    }
    let all: Vec<f64> = groups.iter().flatten().copied().collect();
    let grand = stats::mean(&all);

    // Between-group SS: Σ n_j (mean_j − grand)^2, df = k − 1.
    let mut ss_between = 0.0;
    let mut ss_within = 0.0;
    for g in groups {
        let m = stats::mean(g);
        ss_between += g.len() as f64 * (m - grand).powi(2);
        for &x in g {
            ss_within += (x - m).powi(2);
        }
    }
    let df_between = (k - 1) as f64;
    let df_within = (n_total - k) as f64;
    if df_within <= 0.0 {
        return 0.0;
    }
    let ms_between = ss_between / df_between;
    let ms_within = ss_within / df_within;

    // Mean group size correction (k̄, the "n0" of unbalanced ANOVA).
    let sum_n_sq: f64 = groups.iter().map(|g| (g.len() as f64).powi(2)).sum();
    let k_bar = (n_total as f64 - sum_n_sq / n_total as f64) / df_between;
    let k_bar = if k_bar <= 0.0 { 1.0 } else { k_bar };

    let denom = ms_between + (k_bar - 1.0) * ms_within;
    if denom.abs() < 1e-12 {
        0.0
    } else {
        (ms_between - ms_within) / denom
    }
}

/// ICC(ψ) at the current world state.
pub fn icc_psi(world: &TeamWorld) -> f64 {
    icc_grouped(world, |i| i.psi)
}

// --------------------------------------------------------------------------- //
// OLS regression
// --------------------------------------------------------------------------- //

/// Result of a simple `y = b0 + b1·x` OLS fit.
#[derive(Debug, Clone, Copy)]
pub struct SimpleOls {
    pub intercept: f64,
    pub slope: f64,
    /// Adjusted R².
    pub r2_adj: f64,
    /// t-statistic for the slope.
    pub t_slope: f64,
    /// Two-sided p-value for the slope (normal approximation).
    pub p_slope: f64,
    pub n: usize,
}

/// Fit a simple OLS `y ~ x`.
pub fn simple_ols(x: &[f64], y: &[f64]) -> SimpleOls {
    let n = x.len().min(y.len());
    if n < 3 {
        return SimpleOls {
            intercept: 0.0,
            slope: 0.0,
            r2_adj: 0.0,
            t_slope: 0.0,
            p_slope: 1.0,
            n,
        };
    }
    let nf = n as f64;
    let mx = stats::mean(&x[..n]);
    let my = stats::mean(&y[..n]);
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    let mut syy = 0.0;
    for i in 0..n {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    if sxx.abs() < 1e-12 {
        return SimpleOls {
            intercept: my,
            slope: 0.0,
            r2_adj: 0.0,
            t_slope: 0.0,
            p_slope: 1.0,
            n,
        };
    }
    let slope = sxy / sxx;
    let intercept = my - slope * mx;
    // Residual SS.
    let mut ss_res = 0.0;
    for i in 0..n {
        let yhat = intercept + slope * x[i];
        ss_res += (y[i] - yhat).powi(2);
    }
    let r2 = if syy.abs() < 1e-12 {
        0.0
    } else {
        1.0 - ss_res / syy
    };
    let r2_adj = 1.0 - (1.0 - r2) * (nf - 1.0) / (nf - 2.0);
    let se = (ss_res / (nf - 2.0)).sqrt() / sxx.sqrt();
    let t_slope = if se.abs() < 1e-12 { 0.0 } else { slope / se };
    let p_slope = two_sided_p(t_slope);
    SimpleOls {
        intercept,
        slope,
        r2_adj,
        t_slope,
        p_slope,
        n,
    }
}

/// Result of a two-predictor `y = b0 + b1·x1 + b2·x2` OLS fit.
#[derive(Debug, Clone, Copy)]
pub struct MultiOls {
    pub intercept: f64,
    pub b1: f64,
    pub b2: f64,
    pub r2_adj: f64,
    pub t1: f64,
    pub p1: f64,
    pub t2: f64,
    pub p2: f64,
    pub n: usize,
}

/// Fit a two-predictor OLS `y ~ x1 + x2` via the normal equations.
pub fn multi_ols(x1: &[f64], x2: &[f64], y: &[f64]) -> MultiOls {
    let n = x1.len().min(x2.len()).min(y.len());
    let zero = MultiOls {
        intercept: 0.0,
        b1: 0.0,
        b2: 0.0,
        r2_adj: 0.0,
        t1: 0.0,
        p1: 1.0,
        t2: 0.0,
        p2: 1.0,
        n,
    };
    if n < 4 {
        return zero;
    }
    let nf = n as f64;
    // Center the predictors and response (drops the intercept from the system).
    let m1 = stats::mean(&x1[..n]);
    let m2 = stats::mean(&x2[..n]);
    let my = stats::mean(&y[..n]);
    let (mut s11, mut s12, mut s22, mut s1y, mut s2y, mut syy) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 0..n {
        let a = x1[i] - m1;
        let b = x2[i] - m2;
        let c = y[i] - my;
        s11 += a * a;
        s12 += a * b;
        s22 += b * b;
        s1y += a * c;
        s2y += b * c;
        syy += c * c;
    }
    let det = s11 * s22 - s12 * s12;
    if det.abs() < 1e-12 {
        return zero;
    }
    let b1 = (s22 * s1y - s12 * s2y) / det;
    let b2 = (s11 * s2y - s12 * s1y) / det;
    let intercept = my - b1 * m1 - b2 * m2;
    // Residual SS + standard errors.
    let mut ss_res = 0.0;
    for i in 0..n {
        let yhat = intercept + b1 * x1[i] + b2 * x2[i];
        ss_res += (y[i] - yhat).powi(2);
    }
    let r2 = if syy.abs() < 1e-12 {
        0.0
    } else {
        1.0 - ss_res / syy
    };
    let p = 2.0; // number of predictors
    let df = nf - p - 1.0;
    let r2_adj = if df > 0.0 {
        1.0 - (1.0 - r2) * (nf - 1.0) / df
    } else {
        0.0
    };
    let sigma2 = if df > 0.0 { ss_res / df } else { 0.0 };
    // (X'X)^-1 for the centered predictors → cov(b1,b2) = sigma2 · inv.
    let inv11 = s22 / det;
    let inv22 = s11 / det;
    let se1 = (sigma2 * inv11).sqrt();
    let se2 = (sigma2 * inv22).sqrt();
    let t1 = if se1.abs() < 1e-12 { 0.0 } else { b1 / se1 };
    let t2 = if se2.abs() < 1e-12 { 0.0 } else { b2 / se2 };
    MultiOls {
        intercept,
        b1,
        b2,
        r2_adj,
        t1,
        p1: two_sided_p(t1),
        t2,
        p2: two_sided_p(t2),
        n,
    }
}

// --------------------------------------------------------------------------- //
// Baron & Kenny (1986) three-step mediation
// --------------------------------------------------------------------------- //

/// Baron & Kenny three-step mediation of `X → M → Y`.
///
/// Step 1: `Y ~ X`  (total effect `c`).
/// Step 2: `M ~ X`  (path `a`).
/// Step 3: `Y ~ X + M` (direct `c'` = residual X coefficient; `b` = M coefficient).
/// Mediation ratio (proportion mediated) = `(a·b) / c`.
#[derive(Debug, Clone, Copy)]
pub struct Mediation {
    /// Total effect of X on Y.
    pub c_total: f64,
    /// X → M path.
    pub a_path: f64,
    /// M → Y path (controlling X).
    pub b_path: f64,
    /// Direct X → Y path controlling M (the residual; near 0 under full mediation).
    pub c_prime: f64,
    /// p-value of the residual X coefficient (should be > .10 under mediation).
    pub c_prime_p: f64,
    /// Proportion mediated `(a·b) / c`.
    pub ratio: f64,
}

/// Run the Baron & Kenny three-step mediation.
pub fn baron_kenny(x: &[f64], m: &[f64], y: &[f64]) -> Mediation {
    let step1 = simple_ols(x, y);
    let step2 = simple_ols(x, m);
    let step3 = multi_ols(x, m, y);
    let c = step1.slope;
    let a = step2.slope;
    let c_prime = step3.b1; // residual X coefficient
    let b = step3.b2; // M coefficient
    let ratio = if c.abs() < 1e-9 { 0.0 } else { (a * b) / c };
    Mediation {
        c_total: c,
        a_path: a,
        b_path: b,
        c_prime,
        c_prime_p: step3.p1,
        ratio,
    }
}

// --------------------------------------------------------------------------- //
// p-value helper (normal approximation to the t distribution)
// --------------------------------------------------------------------------- //

/// Two-sided p-value for a t/z statistic via the standard-normal CDF
/// (adequate for the team-level n the design targets; the Python `reproduce`
/// path uses the exact t distribution via statsmodels).
pub fn two_sided_p(t: f64) -> f64 {
    let z = t.abs();
    2.0 * (1.0 - normal_cdf(z))
}

/// Standard-normal CDF via the erf approximation (Abramowitz & Stegun 7.1.26).
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_ols_recovers_slope() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|v| 2.0 + 3.0 * v).collect();
        let f = simple_ols(&x, &y);
        assert!((f.slope - 3.0).abs() < 1e-9);
        assert!((f.intercept - 2.0).abs() < 1e-9);
        assert!(f.r2_adj > 0.99);
    }

    #[test]
    fn multi_ols_recovers_coeffs() {
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x2 = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let y: Vec<f64> = (0..6).map(|i| 1.0 + 2.0 * x1[i] - 0.5 * x2[i]).collect();
        let f = multi_ols(&x1, &x2, &y);
        assert!((f.b1 - 2.0).abs() < 1e-6);
        assert!((f.b2 + 0.5).abs() < 1e-6);
    }

    #[test]
    fn icc_high_when_groups_separated() {
        // Two tight groups far apart → ICC near 1.
        let groups = vec![vec![0.10, 0.11, 0.09, 0.10], vec![0.90, 0.91, 0.89, 0.90]];
        let icc = icc_from_groups(&groups);
        assert!(icc > 0.9, "icc was {icc}");
    }

    #[test]
    fn icc_low_when_groups_overlap() {
        let groups = vec![vec![0.10, 0.90, 0.20, 0.80], vec![0.15, 0.85, 0.25, 0.75]];
        let icc = icc_from_groups(&groups);
        assert!(icc < 0.2, "icc was {icc}");
    }

    #[test]
    fn baron_kenny_full_mediation_zeroes_direct() {
        // M = 0.8X + own variation; Y = 1.5M only. Then the direct X effect c'
        // is ≈ 0 and the proportion mediated ≈ 1.
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let extra = [0.0, 0.3, -0.2, 0.1, -0.4, 0.2, -0.1, 0.3];
        let m: Vec<f64> = x.iter().zip(extra).map(|(v, e)| 0.8 * v + e).collect();
        let y: Vec<f64> = m.iter().map(|v| 1.5 * v).collect();
        let med = baron_kenny(&x, &m, &y);
        assert!(med.c_prime.abs() < 1e-3, "c' was {}", med.c_prime);
        assert!((med.ratio - 1.0).abs() < 1e-2, "ratio was {}", med.ratio);
    }

    #[test]
    fn two_sided_p_symmetric() {
        assert!((two_sided_p(0.0) - 1.0).abs() < 1e-6);
        assert!(two_sided_p(1.96) < 0.06 && two_sided_p(1.96) > 0.04);
    }
}
