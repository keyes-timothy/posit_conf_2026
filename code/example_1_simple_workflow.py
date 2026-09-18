"""
Example 1: Simple LLM Workflow (no tools, no agent).

Strategy:
1. Load all clinical notes for the patient
2. Summarize each note individually (focused on the user's query)
3. Pass all individual summaries into a final "meta-summary" call
4. Return the comprehensive patient summary

This is the simplest approach — purely sequential, no decision-making.

Usage:
    uv run python code/example_1_simple_workflow.py
    uv run python code/example_1_simple_workflow.py --query "Summarize the workup for this patient's lightheadedness"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _setup import get_llm_client, load_notes_df, UsageTracker, timed_response, extract_text, MODEL

DEFAULT_QUERY = "Summarize the workup for this patient's lightheadedness"


def summarize_single_note(client, tracker, note_row: dict, query: str) -> str:
    """Summarize one clinical note, focused on the user's query.

    Sends the note text to the LLM with instructions to produce a 2–3
    sentence summary relevant to ``query``.  If the note contains nothing
    relevant, the LLM returns a sentinel string.

    Args:
        client: An OpenAI-compatible client instance.
        tracker: A :class:`UsageTracker` for recording token usage.
        note_row: A dict representing one row of the notes DataFrame
            (must contain ``note_id``, ``date``, ``specialty``,
            ``provider``, and ``text``).
        query: The user's clinical question to focus the summary on.

    Returns:
        str: A short summary of the note, or a "No relevant information"
            sentinel if the note is not pertinent to the query.
    """
    response = timed_response(
        client,
        tracker,
        label=f"summarize_{note_row['note_id']}",
        model=MODEL,
        instructions=(
            "You are a clinical summarization assistant. "
            "Summarize the following clinical note in 2-3 sentences. "
            "Focus specifically on information relevant to this request: "
            f'"{query}". '
            "If the note contains nothing relevant, respond with: "
            "'No relevant information in this note.'"
        ),
        input=(
            f"Note from {note_row['date']} — "
            f"{note_row['specialty']} ({note_row['provider']}):\n\n"
            f"{note_row['text']}"
        ),
        temperature=0.3,
    )
    return extract_text(response)


def generate_meta_summary(client, tracker, summaries: list[str], query: str) -> str:
    """Combine individual note summaries into a final patient summary.

    Filters out summaries that indicate no relevant information, then
    asks the LLM to synthesize the remaining summaries into a single
    comprehensive answer.

    Args:
        client: An OpenAI-compatible client instance.
        tracker: A :class:`UsageTracker` for recording token usage.
        summaries: List of per-note summary strings produced by
            :func:`summarize_single_note`.
        query: The user's original clinical question.

    Returns:
        str: A comprehensive answer to the user's query, or a message
            indicating that no relevant information was found.
    """
    # Filter out irrelevant notes
    relevant = [
        (i, s) for i, s in enumerate(summaries)
        if "no relevant information" not in s.lower()
    ]

    if not relevant:
        return "No relevant information found across the patient's notes."

    numbered = "\n\n".join(f"[Note {i+1}] {s}" for i, s in relevant)

    response = timed_response(
        client,
        tracker,
        label="meta_summary",
        model=MODEL,
        instructions=(
            "You are a clinical summarization assistant. "
            "Given the individual note summaries below, produce a "
            "comprehensive answer to the user's request. "
            "Be concise but thorough. Cite specific dates where relevant."
        ),
        input=f"User's request: {query}\n\nNote summaries:\n\n{numbered}",
        temperature=0.3,
    )
    return extract_text(response)


def run_workflow(client, notes_df, query: str, tracker, verbose: bool = False) -> str:
    """Run the full map-reduce workflow and return the final summary text.

    Summarizes each note individually (map step), then combines the
    relevant summaries into a final answer (reduce step).

    Args:
        client: An OpenAI-compatible client instance.
        notes_df: The clinical-notes DataFrame (one row per note).
        query: The user's clinical question.
        tracker: A :class:`UsageTracker` for recording token usage.
        verbose: If ``True``, print progress for each note to stdout.

    Returns:
        str: A comprehensive answer to the user's query synthesized
            from the individual note summaries.
    """
    summaries = []
    for _, row in notes_df.iterrows():
        row_dict = row.to_dict()
        if verbose:
            print(f"  {row_dict['note_id']} ({row_dict['date']}, {row_dict['specialty']})...", end=" ", flush=True)
        summary = summarize_single_note(client, tracker, row_dict, query)
        summaries.append(summary)
        if verbose:
            print("✓")

    return generate_meta_summary(client, tracker, summaries, query)


def main():
    """Run the simple workflow demo end-to-end.

    Parses ``--query`` from the command line, loads clinical notes,
    summarizes each note individually (map step), combines the relevant
    summaries into a final answer (reduce step), and prints the result
    along with performance metrics.
    """
    parser = argparse.ArgumentParser(description="Example 1: Simple LLM Workflow")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY, help="What to summarize")
    args = parser.parse_args()

    print("=" * 60)
    print("Example 1: Simple LLM Workflow")
    print("=" * 60)
    print(f'Query: "{args.query}"\n')

    # Setup
    client = get_llm_client()
    notes_df = load_notes_df()
    tracker = UsageTracker()
    print(f"Loaded {len(notes_df)} clinical notes.\n")

    tracker.start()

    # Run the workflow
    print("Running workflow (map-reduce over all notes)...")
    final_summary = run_workflow(client, notes_df, args.query, tracker, verbose=True)

    tracker.stop()

    # Output
    print("\n" + "─" * 60)
    print("FINAL SUMMARY")
    print("─" * 60)
    print(final_summary)

    tracker.print_summary()


if __name__ == "__main__":
    main()
