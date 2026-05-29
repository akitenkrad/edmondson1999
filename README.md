<p align="center"><img src="docs/assets/hero.svg" width="100%"></p>

**English** | [日本語](README.ja.md)

# Edmondson (1999) — Psychological Safety & Team Learning

A team-level agent-based replication of **Edmondson (1999), "Psychological Safety and Learning Behavior in Work Teams"** (*Administrative Science Quarterly*, 44(2), 350–383; DOI: 10.2307/2666999).

The paper's field study is reconstructed as a **dynamic agent-based model of its causal chain**:

```
context support / leader coaching  →  ψ (team psychological safety)  →  learning behavior L  →  team performance Π
```

Individuals belong to fixed teams and sit on a **within-team Watts–Strogatz** small-world network (no cross-team edges). Each step runs six mechanisms across socsim's six-phase loop; the **core** is the psychological-safety difference equation

```
ψ_i(t+1) = (1−λ)·ψ_i + λ·[ α·s_k + β·c_k − γ·1[retaliated] + δ·ψ̄_{−i,k} ]
```

Two **mutually exclusive** voice-decision mechanisms are selected with `--decision-mode`:

- `rule` (default) — a deterministic VOICE logit `σ(β0 + β_ψ·ψ − β_f·f − β_θ·θ − β_c·c)`. Zero LLM calls, bit-for-bit reproducible.
- `llm` — an LLM (`socsim-llm`, Ollama-first → OpenAI fallback) decides VOICE/SILENCE and the learning behavior given the individual's inner state and team context.

Team learning `L_k` aggregates the within-team voice / help-request / error-talk counts; team performance is `Π_k = γ_L·L_k + γ_K·K_k + N(0, σ_obs²)`. The model reconstructs the paper's Baron & Kenny (1986) mediation and ICC anchors on the simulated team cross-section.

## Two-layer determinism

- **Deterministic socsim core** — individual initialisation, within-team network generation, scheduling, and the five non-LLM mechanisms. Given a seed, the `rule` mode reproduces bit-for-bit.
- **Non-deterministic LLM layer** — the `voice_decision` mechanism only. Pseudo-determinised by `socsim-llm`'s `CachingClient` (`hash(prompt+model)` → response cache), `temperature=0`, and a fixed `(agent_id, t)`-derived seed. The cache — not the model — is the reproducibility mechanism.

Each run writes `llm_meta.json` recording decision-mode / model / endpoint / temperature / seed / cache-hit rate.

## Install & Quick start

```bash
# Build the Rust simulation (fetches socsim incl. socsim-llm with Ollama+OpenAI backends).
cargo build --release

# === Baseline reproduction (rule mode — no LLM) ===
cargo run --release -- run \
    --n-teams 90 --team-size 8 \
    --lambda 0.10 --alpha 0.30 --beta 0.25 --gamma 0.50 --delta 0.35 \
    --network-model watts-strogatz --network-k 6 --network-beta 0.15 \
    --t-max 24 --runs 30 --seed 1999

# === Per-trial anchor report against the §5 calibration targets ===
cargo run --release -- reproduce --decision-mode rule --runs 30 --seed 1999

# === Sensitivity sweep (α × δ) ===
cargo run --release -- sweep \
    --alpha-min 0.10 --alpha-max 0.50 --alpha-step 0.05 \
    --delta-min 0.10 --delta-max 0.60 --delta-step 0.10 \
    --n-teams 90 --runs 30 --seed 1999

# === LLM-driven ablation (Ollama first) ===
#   ollama pull llama3.1
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=llama3.1
cargo run --release -- run --decision-mode llm \
    --cache-path runs/edmondson_cache.json \
    --n-teams 90 --runs 10 --seed 1999

# Python visualization & analysis tools (workspace root)
uv sync
uv run edmondson-tools visualize                 # ψ/L/Π series + mediation scatter + ICC trace
uv run edmondson-tools visualize-sweep           # mediation / R² / ICC heatmaps over α × δ
uv run edmondson-tools show-experiment-settings  # config / sweep_config / llm_meta
uv run edmondson-tools reproduce                 # Table 4-8-style Baron & Kenny report + bootstrap CI
```

## Repository layout

```
edmondson1999/
├── simulation/                       # Rust socsim ABM
│   ├── Cargo.toml                    # socsim-{core,engine,net,metrics,llm,results} git deps
│   ├── src/
│   │   ├── lib.rs / main.rs          # CLI: run / sweep / reproduce
│   │   ├── config.rs                 # Config / DecisionMode / PsiParams / VoiceBeta / NetworkKind
│   │   ├── world.rs                  # TeamWorld + Individual + Team
│   │   ├── mechanisms.rs             # 6 mechanisms × 6 phases; rule vs LLM decision (exclusive)
│   │   ├── prompts.rs                # voice-decision prompt + decision JSON parser
│   │   ├── llm.rs                    # socsim-llm shared-harness re-export shim
│   │   ├── simulation.rs             # init_world + run_with_client + CSV/JSON writers + anchors
│   │   └── metrics.rs                # ICC + OLS + Baron & Kenny three-step (paper-specific)
│   └── tests/integration_test.rs     # rule bit-determinism + scripted-LLM smoke
├── tools/                            # Python edmondson-tools
│   └── src/edmondson_tools/{cli,visualize,visualize_sweep,show_experiment_settings,
│                            reproduce_paper}.py
├── docs/                             # bilingual: architecture, cli, usecases, visualization, reproduction
└── results/                          # runtime outputs (gitignored)
    ├── latest -> {YYYYMMDD_HHMMSS}/
    └── {YYYYMMDD_HHMMSS}/
        ├── config.json | sweep_config.json
        ├── teams.csv                # t, team_id, psi, learning, performance, efficacy, support, coaching
        ├── individuals.csv          # t, team_id, agent_id, psi_i, voice, fear, retaliated
        ├── metrics.csv              # t, icc_psi, icc_learning, mediation_ratio, beta_psi_l, beta_l_pi
        ├── team_cross_section.csv   # pooled per-team second-half averages (one row per team per run)
        ├── sweep_summary.csv        # sweep: one row per (α, δ, run)
        └── llm_meta.json            # LLM provenance + cache-hit + determinism note
```

## Documentation

- [Architecture](docs/architecture.md) — world state, six-mechanism × six-phase table, two-layer determinism
- [CLI reference](docs/cli.md) — `run` / `sweep` / `reproduce` flags
- [Usecases](docs/usecases.md) — baseline, sweep, and LLM-ablation workflows
- [Visualization](docs/visualization.md) — what the Python tools produce
- [Reproduction](docs/reproduction.md) — how the model maps to the Edmondson 1999 numbers

## References

- Edmondson, A. C. (1999). Psychological Safety and Learning Behavior in Work Teams. *Administrative Science Quarterly*, 44(2), 350–383.
- Baron, R. M., & Kenny, D. A. (1986). The moderator–mediator variable distinction in social psychological research. *Journal of Personality and Social Psychology*, 51(6), 1173–1182.
- Simulation engine: [socsim (rs-social-simulation-tools)](https://github.com/akitenkrad/rs-social-simulation-tools).

## License

MIT — see [LICENSE](LICENSE).

---
*This file was generated by Claude Code.*
