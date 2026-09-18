"""
Example 0: LLM Conversation (no tools, no notes, no orchestration).

Strategy:
A single LLM call with only the user's query — no patient data is
provided.  The model can only rely on its training data, so it will
typically ask for more information or produce a generic response.

This is the baseline: it illustrates what happens when the LLM has
no access to the patient's chart.

Usage:
    uv run python code/example_0_conversation.py
    uv run python code/example_0_conversation.py --query "What medications is the patient on?"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _setup import get_llm_client, UsageTracker, timed_response, extract_text, MODEL

DEFAULT_QUERY = "Summarize the workup for this patient's lightheadedness"


def run_conversation(client, notes_df, query: str, tracker, verbose: bool = False) -> str:
    """Run a bare LLM conversation with no patient-data context.

    Sends only the user's query to the LLM.  The ``notes_df`` parameter
    is accepted for signature compatibility with the other example
    systems but is intentionally unused — the whole point of this
    baseline is that the model has **no access** to the patient's chart.

    Args:
        client: An OpenAI-compatible client instance.
        notes_df: Unused.  Accepted for a uniform interface with the
            workflow and agent systems.  Typically ``None``.
        query: The user's clinical question.
        tracker: A :class:`UsageTracker` for recording token usage.
        verbose: If ``True``, print progress to stdout.

    Returns:
        str: The LLM's response text (likely generic or a request for
            more information, since no chart data is provided).
    """
    if verbose:
        print("  Sending query to LLM (no patient context)...", end=" ", flush=True)

    response = timed_response(
        client,
        tracker,
        label="conversation",
        model=MODEL,
        instructions="You are a helpful clinical assistant.",
        input=query,
        temperature=0.3,
    )

    if verbose:
        print("Done.")

    return extract_text(response)


def main():
    """Run the conversation baseline demo end-to-end.

    Parses ``--query`` from the command line, sends the query to the LLM
    with no patient context, and prints the response along with
    performance metrics.
    """
    parser = argparse.ArgumentParser(description="Example 0: LLM Conversation (baseline)")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY, help="Clinical question to ask")
    args = parser.parse_args()

    print("=" * 60)
    print("Example 0: LLM Conversation (baseline)")
    print("=" * 60)
    print(f'Query: "{args.query}"\n')

    client = get_llm_client()
    tracker = UsageTracker()

    tracker.start()
    response_text = run_conversation(client, None, args.query, tracker, verbose=True)
    tracker.stop()

    print("\n" + "─" * 60)
    print("RESPONSE")
    print("─" * 60)
    print(response_text)

    tracker.print_summary()


if __name__ == "__main__":
    main()
