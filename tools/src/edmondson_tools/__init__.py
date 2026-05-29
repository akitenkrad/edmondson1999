"""edmondson-tools — visualization, sweep analysis, and reproduction utilities
for the Edmondson (1999) Psychological Safety & Team Learning replication.

Modules:
- `visualize`              — ψ / L / Π time series, mediation scatter, ICC trace.
- `visualize_sweep`        — mediation-ratio / R² heatmaps over α × δ.
- `show_experiment_settings` — pretty-print a results directory's config / meta.
- `reproduce_paper`        — Table 4-8-style Baron & Kenny three-step OLS report
                             + bootstrap mediation 95% BC CI + the efficacy
                             discriminant, compared against the §5 anchors.

All subcommands dispatch through `edmondson_tools.cli:main` — see
`edmondson-tools --help`.
"""

__version__ = "0.1.0"
