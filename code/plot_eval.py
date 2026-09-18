"""
Plot evaluation results: mean score vs. mean cost, colored by system.

Reads output/eval_scored.csv and produces output/eval_plot.png.

Usage:
    uv run python code/plot_eval.py
"""

import sys
from pathlib import Path

import pandas as pd
from plotnine import (
    ggplot,
    aes,
    geom_point,
    geom_text,
    labs,
    scale_color_manual,
    theme_minimal,
    theme,
    element_text,
    ggsave,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    """Read scored evaluation results and produce a quality-vs-cost dotplot.

    Reads ``output/eval_scored.csv``, aggregates scores and costs by
    system × query (averaging over replicates), and saves a
    ``plotnine``-based scatter plot to ``output/eval_plot.png``.  Also
    prints a summary table of mean score, mean cost, and total cost per
    system to stdout.
    """
    input_path = PROJECT_ROOT / "output" / "eval_scored.csv"
    output_path = PROJECT_ROOT / "output" / "eval_plot.png"

    if not input_path.exists():
        print(f"Error: {input_path} not found. Run run_eval.py first and score the results.")
        sys.exit(1)

    df = pd.read_csv(input_path)

    # Aggregate: one dot per system x query_id
    agg = (
        df.groupby(["system", "query_id", "category"])
        .agg(
            mean_score=("score", "mean"),
            mean_cost=("cost_usd", "mean"),
        )
        .reset_index()
    )

    # Capitalize and order system labels
    agg["system"] = pd.Categorical(
        agg["system"].str.capitalize(),
        categories=["Conversation", "Workflow", "Agent"],
        ordered=True,
    )

    # Convert cost to millicents for readability on axis
    agg["cost_millicents"] = agg["mean_cost"] * 100_000

    colors = {
        "Conversation": "#E63946",
        "Workflow": "#457B9D",
        "Agent": "#2A9D8F",
    }

    p = (
        ggplot(agg, aes(x="mean_cost", y="mean_score", color="system"))
        + geom_point(size=3.5, alpha=0.8)
        + geom_text(
            aes(label="query_id"),
            nudge_y=0.12,
            size=7,
            alpha=0.7,
        )
        + scale_color_manual(values=colors)
        + labs(
            x="Mean Cost per Query (USD)",
            y="Mean Quality Score (1-5)",
            color="System",
            title="Quality vs. Cost by System",
            subtitle="Each dot = one system x query (averaged over 3 replicates)",
        )
        + theme_minimal()
        + theme(
            figure_size=(10, 6),
            plot_title=element_text(size=14, weight="bold"),
            plot_subtitle=element_text(size=10),
        )
    )

    ggsave(p, output_path, dpi=300)
    print(f"✓ Saved plot to {output_path}")

    # Also print summary table
    summary = (
        df.groupby("system")
        .agg(
            mean_score=("score", "mean"),
            mean_cost=("cost_usd", "mean"),
            total_cost=("cost_usd", "sum"),
        )
        .round(4)
    )
    print(f"\n{summary}")


if __name__ == "__main__":
    main()
