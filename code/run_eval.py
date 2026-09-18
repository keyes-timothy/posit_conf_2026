"""
Run evaluation: 3 systems x 10 queries x N replicates.

Systems:
- conversation: Bare LLM call with no notes context
- workflow: Summarize each note then combine (imported from example_1)
- agent: LLM-driven tool loop that selectively reads notes (imported from example_2)

The workflow and agent systems are imported from the example scripts
to ensure a single source of truth for each system's logic.

Outputs: output/eval_responses.csv

Usage:
    uv run python code/run_eval.py
    uv run python code/run_eval.py --replicates 5
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _setup import get_llm_client, load_notes_df, UsageTracker, extract_text, MODEL
from example_0_conversation import run_conversation
from example_1_simple_workflow import run_workflow
from example_2_single_agent import run_agent

# ─── Pricing (per token) ─────────────────────────────────────────────────────

MINI_INPUT_COST = 0.75 / 1_000_000
MINI_OUTPUT_COST = 4.50 / 1_000_000


def compute_cost_from_tracker(tracker: UsageTracker) -> float:
    """Compute USD cost from a UsageTracker using module-level pricing.

    Args:
        tracker: A :class:`UsageTracker` with recorded token usage.

    Returns:
        float: The total cost in USD.
    """
    return (tracker.total_input_tokens * MINI_INPUT_COST
            + tracker.total_output_tokens * MINI_OUTPUT_COST)


# ─── Tracker Bridge ──────────────────────────────────────────────────────────


def _run_with_tracker(fn, client, notes_df, query):
    """Run a system function with a fresh UsageTracker and return metrics.

    Creates a temporary :class:`UsageTracker`, passes it to ``fn``, and
    extracts token counts and cost from the tracker after the run.

    Args:
        fn: A callable with signature
            ``(client, notes_df, query, tracker, verbose) -> str``.
        client: An OpenAI-compatible client instance.
        notes_df: The clinical-notes DataFrame (may be ``None`` for the
            conversation system).
        query: The user's clinical question.

    Returns:
        tuple[str, int, int, float]: A 4-tuple of
            ``(response_text, input_tokens, output_tokens, cost_usd)``.
    """
    tracker = UsageTracker()
    text = fn(client, notes_df, query, tracker, verbose=False)
    cost = compute_cost_from_tracker(tracker)
    return text, tracker.total_input_tokens, tracker.total_output_tokens, cost


# ─── Main ─────────────────────────────────────────────────────────────────────

SYSTEMS = {
    "conversation": run_conversation,
    "workflow": run_workflow,
    "agent": run_agent,
}


def main():
    """Run the full evaluation harness across all systems and queries.

    Parses ``--replicates`` from the command line, loads the 10 eval
    queries from ``code/eval_queries.json``, runs each of the 3 systems
    (conversation, workflow, agent) against every query for the
    specified number of replicates, and writes the results to
    ``output/eval_responses.csv``.  Prints a cost summary to stdout.
    """
    parser = argparse.ArgumentParser(description="Run evaluation across 3 systems")
    parser.add_argument("--replicates", type=int, default=3)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    queries_path = project_root / "code" / "eval_queries.json"
    output_path = project_root / "output" / "eval_responses.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(queries_path) as f:
        queries = json.load(f)

    client = get_llm_client()
    notes_df = load_notes_df()

    total_runs = len(queries) * len(SYSTEMS) * args.replicates
    print(f"Running {total_runs} evaluations "
          f"({len(queries)} queries x {len(SYSTEMS)} systems x {args.replicates} reps)\n")

    rows = []
    run_num = 0
    for q in queries:
        for system_name, system_fn in SYSTEMS.items():
            for rep in range(1, args.replicates + 1):
                run_num += 1
                print(f"  [{run_num}/{total_runs}] {q['query_id']} / "
                      f"{system_name} / rep {rep}...", end=" ", flush=True)

                text, inp, out, cost = _run_with_tracker(
                    system_fn, client, notes_df, q["query"]
                )

                rows.append({
                    "query_id": q["query_id"],
                    "category": q["category"],
                    "query": q["query"],
                    "system": system_name,
                    "replicate": rep,
                    "response": text,
                    "input_tokens": inp,
                    "output_tokens": out,
                    "cost_usd": round(cost, 6),
                })
                print(f"${cost:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved {len(df)} rows to {output_path}")

    summary = df.groupby("system").agg(
        total_cost=("cost_usd", "sum"),
        mean_cost=("cost_usd", "mean"),
    ).round(4)
    print(f"\n{summary}")


if __name__ == "__main__":
    main()

