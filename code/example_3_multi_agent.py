"""
Example 3: Multi-Agent System (Orchestrator + Summarizer + Critic).

Architecture:
- Orchestrator: Delegates summarization and evaluates quality
- Summarizer Agent: Same as Example 2 — explores chart with tools
- Critic Agent: Evaluates if the summary answers the user's query well

The orchestrator loops: summarizer produces a draft, critic evaluates,
and if the critic says "refine", feedback goes back to the summarizer.

Usage:
    uv run python code/example_3_multi_agent.py
    uv run python code/example_3_multi_agent.py --query "Summarize the workup for this patient's lightheadedness"
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _setup import get_llm_client, load_notes_df, UsageTracker, timed_response, extract_text, MODEL

DEFAULT_QUERY = "Summarize the workup for this patient's lightheadedness"
DEFAULT_CRITERIA = (
    "The summary should include specific dates, current medications, "
    "and any outstanding follow-up items."
)
MAX_REFINEMENTS = 2


# ─── Tool Definitions (same as Example 2) ────────────────────────────────────

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
      date, specialty, and provider header.

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


# ─── Summarizer Agent ─────────────────────────────────────────────────────────

SUMMARIZER_INSTRUCTIONS = """\
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


def run_summarizer(
    client,
    notes_df: pd.DataFrame,
    query: str,
    tracker: UsageTracker,
    feedback: str | None = None,
    verbose: bool = True,
) -> str:
    """Run the summarizer agent. Optionally incorporate critic feedback.

    Works identically to the agent loop in Example 2: iteratively calls
    tools to explore the chart, then produces a text summary.  When
    ``feedback`` is provided (from a previous critic pass), the feedback
    is prepended to the user message so the agent can address it.

    Args:
        client: An OpenAI-compatible client instance.
        notes_df: The clinical-notes DataFrame the tools operate on.
        query: The user's clinical question.
        tracker: A :class:`UsageTracker` for recording token usage.
        feedback: Optional critic feedback from a prior refinement pass.
            When provided, the agent is instructed to address these
            issues in its revised summary.
        verbose: If ``True``, print iteration progress and tool-call
            details to stdout.

    Returns:
        str: The summarizer agent's final text answer, or an error
            message if the maximum iteration count is exceeded.
    """
    if feedback:
        user_content = (
            f"{query}\n\n"
            f"[CRITIC FEEDBACK — please address these issues in your revised summary]\n"
            f"{feedback}"
        )
    else:
        user_content = query

    input_messages = [{"role": "user", "content": user_content}]

    max_iterations = 15
    for iteration in range(max_iterations):
        if verbose:
            print(f"    [Summarizer iter {iteration + 1}]", end=" ", flush=True)

        response = timed_response(
            client, tracker,
            label=f"summarizer_iter_{iteration + 1}",
            model=MODEL,
            instructions=SUMMARIZER_INSTRUCTIONS,
            input=input_messages,
            tools=TOOLS,
            temperature=0.2,
        )

        tool_calls = [item for item in response.output if item.type == "function_call"]

        if not tool_calls:
            if verbose:
                print("→ Final response.")
            return extract_text(response)

        if verbose:
            print(f"→ Tools: {[tc.name for tc in tool_calls]}")

        # Accumulate: add model output + tool results to the conversation
        input_messages += response.output
        for tc in tool_calls:
            result = execute_tool(tc.name, tc.arguments, notes_df)
            input_messages.append({
                "type": "function_call_output",
                "call_id": tc.call_id,
                "output": result,
            })

    return "Error: Summarizer exceeded maximum iterations."


# ─── Critic Agent ─────────────────────────────────────────────────────────────

CRITIC_INSTRUCTIONS_TEMPLATE = """\
You are a clinical documentation critic. You evaluate whether a patient \
summary adequately answers the user's original question.

Evaluate the summary against these standard criteria:
1. RELEVANCE: Does it directly address the user's question?
2. COMPLETENESS: Are key dates, medications, and findings included?
3. ACCURACY: Is the information internally consistent?
4. ORGANIZATION: Is it well-structured and easy to follow?

In addition, the user has specified these priorities for the summary:
"{criteria}"

Weight the user's stated priorities heavily in your evaluation.

Return your evaluation as structured output.
"""


class CriticVerdict(BaseModel):
    """Structured evaluation result from the critic agent.

    Attributes:
        verdict: Either ``"PASS"`` (summary is acceptable) or
            ``"REFINE"`` (summary needs improvement).
        feedback: Explanation of what is missing or needs improvement
            (when ``verdict`` is ``"REFINE"``), or a brief confirmation
            that the summary is adequate.
    """
    verdict: str = Field(description="Either 'PASS' or 'REFINE'")
    feedback: str = Field(description="If REFINE, explain what's missing or needs improvement. If PASS, say 'Summary is adequate.'")


def run_critic(
    client,
    query: str,
    summary: str,
    criteria: str,
    tracker: UsageTracker,
    verbose: bool = True,
) -> CriticVerdict:
    """Evaluate a summary using the critic agent with structured output.

    Sends the summary and the user's original question to the LLM with
    instructions to judge quality against the provided ``criteria``.
    The response is constrained to the :class:`CriticVerdict` JSON
    schema (``verdict`` + ``feedback``).

    Args:
        client: An OpenAI-compatible client instance.
        query: The user's original clinical question.
        summary: The summarizer agent's draft answer to evaluate.
        criteria: Human-readable quality criteria the critic should
            prioritize (e.g. "include specific dates, medications…").
        tracker: A :class:`UsageTracker` for recording token usage.
        verbose: If ``True``, print evaluation progress to stdout.

    Returns:
        CriticVerdict: A structured verdict with ``verdict`` (PASS or
            REFINE) and ``feedback``.
    """
    if verbose:
        print("    [Critic] Evaluating summary...", end=" ", flush=True)

    instructions = CRITIC_INSTRUCTIONS_TEMPLATE.format(criteria=criteria)

    response = timed_response(
        client, tracker,
        label="critic",
        model=MODEL,
        instructions=instructions,
        input=(
            f"User's original question: {query}\n\n"
            f"Summary to evaluate:\n{summary}"
        ),
        temperature=0.2,
        text={"format": {"type": "json_schema", "name": "CriticVerdict", "schema": {**CriticVerdict.model_json_schema(), "additionalProperties": False}, "strict": True}},
    )

    raw_text = extract_text(response)
    evaluation = CriticVerdict.model_validate_json(raw_text)

    if verbose:
        print(f"→ {evaluation.verdict}")

    return evaluation


# ─── Orchestrator ─────────────────────────────────────────────────────────────


def orchestrator(
    client,
    notes_df: pd.DataFrame,
    query: str,
    criteria: str,
    tracker: UsageTracker,
    verbose: bool = True,
) -> str:
    """Orchestrate the summarizer + critic loop until the summary passes.

    On each iteration the summarizer agent drafts (or refines) a
    summary, then the critic evaluates it.  If the critic returns
    ``"PASS"``, the summary is returned immediately.  If the critic
    returns ``"REFINE"``, its feedback is forwarded to the summarizer
    for the next attempt, up to ``MAX_REFINEMENTS`` extra passes.

    Args:
        client: An OpenAI-compatible client instance.
        notes_df: The clinical-notes DataFrame the tools operate on.
        query: The user's clinical question.
        criteria: Quality criteria forwarded to the critic agent.
        tracker: A :class:`UsageTracker` for recording token usage.
        verbose: If ``True``, print progress to stdout.

    Returns:
        str: The final (critic-approved or best-effort) summary text.
    """
    feedback = None

    for attempt in range(1, MAX_REFINEMENTS + 2):  # +1 for initial + max refinements
        if verbose:
            label = "Initial" if attempt == 1 else f"Refinement {attempt - 1}"
            print(f"\n  ── {label} {'─' * 40}")

        # Run summarizer (with optional feedback from previous critic pass)
        summary = run_summarizer(client, notes_df, query, tracker, feedback=feedback, verbose=verbose)

        # Run critic
        evaluation = run_critic(client, query, summary, criteria, tracker, verbose=verbose)

        if evaluation.verdict == "PASS":
            if verbose:
                print(f"\n  ✓ Summary approved by critic on attempt {attempt}.")
            return summary

        # Critic wants refinement
        if attempt <= MAX_REFINEMENTS:
            feedback = evaluation.feedback
            if verbose:
                print(f"    Critic feedback: {feedback[:100]}...")
        else:
            if verbose:
                print(f"\n  ⚠ Max refinements reached. Returning latest summary.")
            return summary

    return summary


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    """Run the multi-agent demo end-to-end.

    Parses ``--query`` and ``--criteria`` from the command line,
    initializes the LLM client and notes DataFrame, runs the
    orchestrator (summarizer + critic loop), and prints the final
    summary along with performance metrics.
    """
    parser = argparse.ArgumentParser(description="Example 3: Multi-Agent System")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY,
                        help="What to summarize from the patient's chart")
    parser.add_argument("--criteria", type=str, default=DEFAULT_CRITERIA,
                        help="What the critic should prioritize when evaluating the summary")
    args = parser.parse_args()

    print("=" * 60)
    print("Example 3: Multi-Agent System")
    print("  (Orchestrator + Summarizer + Critic)")
    print("=" * 60)
    print(f'Query:    "{args.query}"')
    print(f'Criteria: "{args.criteria}"\n')

    client = get_llm_client()
    notes_df = load_notes_df()
    tracker = UsageTracker()
    print(f"Loaded {len(notes_df)} clinical notes into DataFrame.")
    print("Starting orchestrator...\n")

    tracker.start()
    summary = orchestrator(client, notes_df, args.query, args.criteria, tracker)
    tracker.stop()

    print("\n" + "─" * 60)
    print("FINAL SUMMARY (Multi-Agent)")
    print("─" * 60)
    print(summary)

    tracker.print_summary()


if __name__ == "__main__":
    main()

