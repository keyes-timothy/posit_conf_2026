"""
Example 2: Single Agent with Tools.

The agent explores the patient's chart autonomously by calling tools:
- list_notes: See all available notes (date, specialty, type)
- preview_clinical_note: Read first 300 chars of a note
- read_clinical_note: Read the complete note text

The agent decides which notes to pull based on the user's query,
and produces a targeted summary when it has enough context.

Usage:
    uv run python code/example_2_single_agent.py
    uv run python code/example_2_single_agent.py --query "Summarize the workup for this patient's lightheadedness"
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _setup import get_llm_client, load_notes_df, UsageTracker, timed_response, extract_text, MODEL

DEFAULT_QUERY = "Summarize the workup for this patient's lightheadedness"


# ─── Tool Definitions ────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "name": "list_notes",
        "description": "List all available clinical notes for the patient. Returns note_id, date, specialty, provider, and note_type.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "preview_clinical_note",
        "description": "Preview the first 300 characters of a clinical note. Useful for deciding whether to read the full note.",
        "parameters": {
            "type": "object",
            "properties": {
                "note_id": {
                    "type": "string",
                    "description": "The note ID, e.g. 'NOTE-001'",
                },
            },
            "required": ["note_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "read_clinical_note",
        "description": "Read the complete text of a clinical note.",
        "parameters": {
            "type": "object",
            "properties": {
                "note_id": {
                    "type": "string",
                    "description": "The note ID, e.g. 'NOTE-001'",
                },
            },
            "required": ["note_id"],
            "additionalProperties": False,
        },
    },
]


# ─── Tool Implementations ─────────────────────────────────────────────────────


def execute_tool(tool_name: str, raw_arguments: str, notes_df: pd.DataFrame) -> str:
    """Execute a tool call and return the result as a string.

    Dispatches to the appropriate handler based on ``tool_name``:

    - ``list_notes``: returns a formatted table of all notes.
    - ``preview_clinical_note``: returns the first 300 characters of a
      note identified by ``note_id``.
    - ``read_clinical_note``: returns the full text of a note with its
      date and specialty header.

    Args:
        tool_name: The name of the tool to invoke.
        raw_arguments: A JSON string of tool arguments (parsed
            internally).
        notes_df: The clinical-notes DataFrame to query against.

    Returns:
        str: The tool's output, or ``"Unknown tool"`` if ``tool_name``
            is not recognized.
    """
    args = json.loads(raw_arguments)

    if tool_name == "list_notes":
        listing = notes_df[["note_id", "date", "specialty", "provider", "note_type"]].copy()
        listing["date"] = listing["date"].dt.strftime("%Y-%m-%d")
        return listing.to_string(index=False)

    elif tool_name == "preview_clinical_note":
        note_id = args["note_id"]
        match = notes_df[notes_df["note_id"] == note_id]
        if match.empty:
            return f"Error: Note '{note_id}' not found."
        text = match.iloc[0]["text"]
        return text[:300] + ("..." if len(text) > 300 else "")

    elif tool_name == "read_clinical_note":
        note_id = args["note_id"]
        match = notes_df[notes_df["note_id"] == note_id]
        if match.empty:
            return f"Error: Note '{note_id}' not found."
        row = match.iloc[0]
        return (
            f"Date: {row['date'].strftime('%Y-%m-%d')}\n"
            f"Specialty: {row['specialty']}\n"
            f"Provider: {row['provider']}\n"
            f"Type: {row['note_type']}\n\n"
            f"{row['text']}"
        )

    return f"Error: Unknown tool '{tool_name}'"


# ─── Agent Loop ───────────────────────────────────────────────────────────────

AGENT_INSTRUCTIONS = """\
You are a clinical summarization agent. Your job is to answer the user's \
question about a patient by exploring their medical chart.

You have tools to explore the chart:
1. Start by listing available notes to see what's in the chart.
2. Preview or read notes that seem relevant to the user's question.
3. You do NOT need to read every note — focus on the most relevant ones.
4. When you have enough information, produce your final answer.

Your final answer should be well-organized, cite specific dates, and \
directly address the user's question.
"""


def run_agent(client, notes_df: pd.DataFrame, query: str, tracker: UsageTracker, verbose: bool = True) -> str:
    """Run the agent loop until it produces a final text response.

    The agent iteratively calls tools (list, preview, read) to explore
    the patient's chart, building up conversational context, until it
    decides it has enough information and returns a text answer.  The
    loop is capped at 15 iterations as a safety limit.

    Args:
        client: An OpenAI-compatible client instance.
        notes_df: The clinical-notes DataFrame the tools operate on.
        query: The user's clinical question.
        tracker: A :class:`UsageTracker` for recording token usage.
        verbose: If ``True``, print iteration progress and tool-call
            details to stdout.

    Returns:
        str: The agent's final text answer, or an error message if the
            maximum iteration count is exceeded.
    """
    input_messages = [{"role": "user", "content": query}]

    max_iterations = 15
    for iteration in range(max_iterations):
        if verbose:
            print(f"\n  [Iteration {iteration + 1}] Calling LLM...", end=" ", flush=True)

        response = timed_response(
            client, tracker,
            label=f"agent_iter_{iteration + 1}",
            model=MODEL,
            instructions=AGENT_INSTRUCTIONS,
            input=input_messages,
            tools=TOOLS,
            temperature=0.2,
        )

        # Separate tool calls from text messages
        tool_calls = [item for item in response.output if item.type == "function_call"]

        # If no tool calls, the agent is done
        if not tool_calls:
            if verbose:
                print("Done (final response).")
            return extract_text(response)

        # Process tool calls
        if verbose:
            print(f"Tool calls: {[tc.name for tc in tool_calls]}")

        # Accumulate: add model output + tool results to the conversation
        input_messages += response.output
        for tc in tool_calls:
            result = execute_tool(tc.name, tc.arguments, notes_df)
            if verbose:
                preview = result[:80].replace("\n", " ")
                print(f"    → {tc.name}: {preview}...")

            input_messages.append({
                "type": "function_call_output",
                "call_id": tc.call_id,
                "output": result,
            })

    return "Error: Agent exceeded maximum iterations."


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    """Run the single-agent demo end-to-end.

    Parses ``--query`` from the command line, initializes the LLM client
    and notes DataFrame, runs the tool-using agent loop, and prints the
    final summary along with performance metrics.
    """
    parser = argparse.ArgumentParser(description="Example 2: Single Agent with Tools")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY)
    args = parser.parse_args()

    print("=" * 60)
    print("Example 2: Single Agent with Tools")
    print("=" * 60)
    print(f'Query: "{args.query}"\n')

    client = get_llm_client()
    notes_df = load_notes_df()
    tracker = UsageTracker()
    print(f"Loaded {len(notes_df)} clinical notes into DataFrame.")
    print("Starting agent...\n")

    tracker.start()
    summary = run_agent(client, notes_df, args.query, tracker)
    tracker.stop()

    print("\n" + "─" * 60)
    print("FINAL SUMMARY (Agent-Generated)")
    print("─" * 60)
    print(summary)

    tracker.print_summary()


if __name__ == "__main__":
    main()
