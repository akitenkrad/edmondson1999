**English** | [日本語](cli.ja.md)

# CLI reference

The `edmondson` binary has three subcommands.

## `run` — single configuration

Runs `--runs` independent replicates, each a fresh seed derived from `--seed`.

A replicate keeps its own time series, so **one replicate is one child run**: the parent
declares the condition and the replicate list (and, being no single simulation, claims no
`master_seed` — the base seed lives in `/base_seed` and reaches `execution_hash` through
`seed_pointers`), and each child carries its own `master_seed` and `replicate_index`.
Replicates of the same condition share a `config_hash`, so "which of these are repeats of
the same condition" is answered by the machine rather than by the directory names.

Each child records:

| what | where |
|------|-------|
| per-step ICC / regression coefficients | `metrics.csv`, `scope=run` step metrics |
| the team panel (per step, per team) | `events.jsonl`, `observation` (`unit_id` = team) |
| the team cross-section (second-half averages) | `events.jsonl`, `terminal` (one row per team) |
| the individual panel | `artifacts/individuals.csv`, as a table |
| `icc_learning_stable` / `final_round` / `n_units` / `convergence_step` | `metrics.csv`, run-scope rows |

| Flag | Default | Meaning |
|------|---------|---------|
| `--decision-mode` | `rule` | `rule` (logit, no LLM) or `llm` (socsim-llm). |
| `--n-teams` | `90` | Number of teams. |
| `--team-size` | `8` | Individuals per team. |
| `--network-model` | `watts-strogatz` | `watts-strogatz` / `erdos-renyi` / `barabasi-albert` (within-team). |
| `--network-k` | `6` | Watts–Strogatz / Barabási–Albert degree parameter. |
| `--network-beta` | `0.15` | Watts–Strogatz rewiring β (or Erdős–Rényi p). |
| `--lambda` | `0.10` | ψ-update learning rate λ. |
| `--alpha` | `0.30` | ψ-update context-support weight α. |
| `--beta` | `0.25` | ψ-update coaching weight β. |
| `--gamma` | `0.50` | ψ-update retaliation-shock weight γ. |
| `--delta` | `0.35` | ψ-update shared-belief convergence weight δ. |
| `--sigma-obs` | `0.22` | Observer-rating noise sd σ_obs. |
| `--t-max` | `24` | Steps per trial (months). |
| `--runs` | `30` | Independent replicates (one child run each). |
| `--seed` | `1999` | Base seed; each replicate's seed is derived from it. |
| `--llm-temperature` | `0.0` | LLM temperature (llm mode). |
| `--llm-seed` | `0` | LLM seed offset (llm mode). |
| `--cache-path` | `.llm_cache/cache.json` | Prompt→response cache path (llm mode). |
| `--output-dir` | `results` | runvault results root. |

## `sweep` — α × δ sensitivity

Cartesian product over `α × δ`, `--runs` trials per cell.

A sweep trial keeps only its final cross-section statistics, so it has no time series of its
own and one `terminal` row says everything about it. **One cell is one child run**, and one
trial is one `terminal` row in that child's `events.jsonl`; the cell's own means are its
run-scope `mean_*` metrics. (The shape follows what was observed — it is not a blanket rule:
`run` splits its replicates into child runs because those *do* keep a time series.)

| Flag | Default | Meaning |
|------|---------|---------|
| `--alpha-min/max/step` | `0.10` / `0.50` / `0.10` | α sweep grid. |
| `--delta-min/max/step` | `0.10` / `0.60` / `0.10` | δ sweep grid. |
| `--lambda` | `0.10` | λ held fixed across the sweep. |
| `--n-teams` / `--team-size` | `90` / `8` | Org shape. |
| `--runs` | `30` | Trials per cell. |
| `--t-max` / `--seed` | `24` / `1999` | Horizon / root seed. |

## `reproduce` — anchor report

Runs `--runs` trials, computes the §5 anchors on each trial's team cross-section, and prints the **mean across trials** with PASS / off-anchor verdicts (averaging per-trial statistics keeps the n≈51-team statistical power, rather than inflating it by pooling).

Like `sweep`, the trials keep no time series, so this is a single run: one `terminal` row per
trial, and the mean as run-scope metrics named after the paper's quantities. The eight numbers
Edmondson (1999) printed go to `reference.csv` with their table citations, so the difference
between "what the paper reported" and "what this reproduction got" is computable afterwards.
The anchor **bands** (`[.25, .55]` and friends) are ours, not the paper's, so they stay on the
console and out of both files.

| Flag | Default | Meaning |
|------|---------|---------|
| `--decision-mode` | `rule` | Mode to report. |
| `--n-teams` / `--team-size` | `90` / `8` | Org shape. |
| `--t-max` / `--runs` / `--seed` | `24` / `30` / `1999` | Horizon / trials / root seed. |
