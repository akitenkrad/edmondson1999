"""edmondson-tools — unified CLI dispatcher.

    edmondson-tools visualize                 # ψ/L/Π time series + mediation scatter + ICC trace
    edmondson-tools visualize-sweep           # mediation-ratio / R² heatmaps over α×δ
    edmondson-tools show-experiment-settings  # print config / sweep_config / llm_meta
    edmondson-tools reproduce                 # Table 4-8-style Baron & Kenny report + bootstrap CI

Arguments after the subcommand are passed verbatim to that subcommand's argparse.
Add `--help` after a subcommand for its own help.

The dispatcher assembly is delegated to the shared helper
`socsim_tools.cli.build_dispatcher`.
"""

from __future__ import annotations

from socsim_tools.cli import build_dispatcher

main = build_dispatcher(
    prog="edmondson-tools",
    description="Edmondson (1999) Psychological Safety & Team Learning — visualization + reproduction",
    subcommands={
        "visualize": (
            "single-run visualization (ψ/L/Π time series + mediation scatter + ICC trace)",
            "edmondson_tools.visualize:main",
        ),
        "visualize-sweep": (
            "sweep visualization (mediation-ratio / R² heatmaps over α×δ)",
            "edmondson_tools.visualize_sweep:main",
        ),
        "show-experiment-settings": (
            "print a results directory's settings (config / sweep_config / llm_meta)",
            "edmondson_tools.show_experiment_settings:main",
        ),
        "reproduce": (
            "Table 4-8-style Baron & Kenny three-step OLS report + bootstrap mediation 95% BC CI",
            "edmondson_tools.reproduce_paper:main",
        ),
    },
)


if __name__ == "__main__":
    main()
