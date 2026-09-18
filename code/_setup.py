"""
Shared setup for posit::conf 2026 demo scripts.

Provides:
- get_llm_client(): OpenAI client via Azure KeyVault -> LiteLLM proxy
- load_notes_df(): Load synthetic clinical notes into a pandas DataFrame
- UsageTracker: Accumulates token usage and latency across LLM calls

Usage:
    from _setup import get_llm_client, load_notes_df, UsageTracker, MODEL
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from dotenv import dotenv_values
from openai import OpenAI
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient


# ─── Azure KeyVault / LiteLLM Configuration ──────────────────────────────────
# Loaded from .env at the project root.  Copy .env.example to .env and fill in
# your values.  See README.md for details.

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_env = dotenv_values(_PROJECT_ROOT / ".env")

KEYVAULT_URL = _env["KEYVAULT_URL"]
LITELLM_KEY_SECRET_NAME = _env["LITELLM_KEY_SECRET_NAME"]
LITELLM_BASE_URL = _env["LITELLM_BASE_URL"]

MODEL = "gpt-5.4-mini"

def get_llm_client() -> OpenAI:
    """Create an OpenAI-compatible client via Azure KeyVault -> LiteLLM.

    Authenticates using ``DefaultAzureCredential``, retrieves the LiteLLM
    API key from Azure Key Vault, and returns an OpenAI client configured
    to use the LiteLLM proxy as its base URL.

    Returns:
        OpenAI: A configured client instance ready to make LLM API calls.
    """
    credential = DefaultAzureCredential()
    keyvault_client = SecretClient(credential=credential, vault_url=KEYVAULT_URL)
    litellm_key = keyvault_client.get_secret(LITELLM_KEY_SECRET_NAME).value
    return OpenAI(base_url=LITELLM_BASE_URL, api_key=litellm_key)


def load_notes_df() -> pd.DataFrame:
    """Load synthetic clinical notes JSON into a pandas DataFrame.

    Reads ``synthetic_data/clinical_notes.json`` relative to the project
    root and returns a DataFrame with one row per clinical note.  The
    ``date`` column is converted to ``datetime64``.

    Returns:
        pd.DataFrame: DataFrame with columns ``note_id``, ``date``,
            ``specialty``, ``provider``, ``note_type``, and ``text``.
    """
    project_root = Path(__file__).resolve().parent.parent
    notes_path = project_root / "synthetic_data" / "clinical_notes.json"

    with open(notes_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data["notes"])
    df["date"] = pd.to_datetime(df["date"])
    return df


# ─── Usage Tracker ────────────────────────────────────────────────────────────


@dataclass
class UsageTracker:
    """Track token usage and latency across multiple LLM calls.

    Usage:
        tracker = UsageTracker()
        tracker.start()
        # ... make LLM calls via timed_response() ...
        tracker.stop()
        tracker.print_summary()
    """

    calls: list[dict] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_calls: int = 0
    _start_time: float | None = field(default=None, repr=False)
    _total_latency: float | None = field(default=None, repr=False)

    def start(self) -> None:
        """Start the wall-clock timer for the entire run.

        Call this before the first LLM request.  Pair with :meth:`stop`
        after the last request to record total elapsed time.
        """
        self._start_time = time.perf_counter()

    def stop(self) -> None:
        """Stop the wall-clock timer and record total elapsed time.

        After calling this, :attr:`total_latency_seconds` returns the
        frozen duration between :meth:`start` and this call.
        """
        if self._start_time is not None:
            self._total_latency = time.perf_counter() - self._start_time

    @property
    def total_latency_seconds(self) -> float:
        """Total wall-clock time in seconds from :meth:`start` to :meth:`stop`.

        Returns:
            float: Elapsed seconds.  Returns the frozen value after
                :meth:`stop`, the live elapsed time if the timer is still
                running, or ``0.0`` if :meth:`start` was never called.
        """
        if self._total_latency is not None:
            return self._total_latency
        if self._start_time is not None:
            return time.perf_counter() - self._start_time
        return 0.0

    def record(self, response, elapsed: float, label: str = "") -> None:
        """Record token-usage metrics from a single API response.

        Args:
            response: An OpenAI Responses API response object whose
                ``usage`` attribute contains ``input_tokens`` and
                ``output_tokens``.
            elapsed: Wall-clock seconds the API call took.
            label: Optional human-readable label for the call (e.g.
                ``"summarize_NOTE-001"``).
        """
        input_tokens = getattr(response.usage, "input_tokens", 0) or 0
        output_tokens = getattr(response.usage, "output_tokens", 0) or 0

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_calls += 1

        self.calls.append({
            "label": label,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "elapsed_seconds": round(elapsed, 3),
        })

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output) consumed across all recorded calls.

        Returns:
            int: Sum of :attr:`total_input_tokens` and
                :attr:`total_output_tokens`.
        """
        return self.total_input_tokens + self.total_output_tokens

    def print_summary(self) -> None:
        """Print a formatted performance-metrics summary to stdout.

        Displays total LLM calls, input/output tokens, and total
        wall-clock latency in a bordered text block.
        """
        print("\n" + "─" * 50)
        print("PERFORMANCE METRICS")
        print("─" * 50)
        print(f"  Total LLM calls:    {self.total_calls:>8,}")
        print(f"  Input tokens:       {self.total_input_tokens:>8,}")
        print(f"  Output tokens:      {self.total_output_tokens:>8,}")
        print(f"  Total latency:      {self.total_latency_seconds:>8.1f}s")
        print("─" * 50)

    def to_dict(self, query: str = "", model: str = MODEL) -> dict:
        """Export accumulated metrics as a JSON-serializable dictionary.

        Args:
            query: The user query that was evaluated (stored as metadata).
            model: The LLM model name (stored as metadata).

        Returns:
            dict: Keys include ``query``, ``model``, ``total_calls``,
                ``input_tokens``, ``output_tokens``,
                ``total_latency_seconds``, and ``per_call`` (list of
                per-call dicts).
        """
        return {
            "query": query,
            "model": model,
            "total_calls": self.total_calls,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_latency_seconds": round(self.total_latency_seconds, 2),
            "per_call": self.calls,
        }

    def save(self, path: str | Path, query: str = "", model: str = MODEL) -> None:
        """Save accumulated metrics to a JSON file.

        Creates parent directories if they do not exist.

        Args:
            path: Destination file path (absolute or relative).
            query: The user query that was evaluated (stored as metadata).
            model: The LLM model name (stored as metadata).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(query=query, model=model), f, indent=2)
        print(f"  Metrics saved to: {path}")


def timed_response(client, tracker: UsageTracker, label: str = "", **kwargs):
    """Call ``client.responses.create()`` with automatic timing and tracking.

    Measures wall-clock time for the API call and records the result in
    the provided :class:`UsageTracker`.

    Args:
        client: An OpenAI-compatible client instance.
        tracker: The :class:`UsageTracker` to record metrics into.
        label: Optional human-readable label for this call.
        **kwargs: Keyword arguments forwarded directly to
            ``client.responses.create()`` (e.g. ``model``,
            ``instructions``, ``input``, ``temperature``).

    Returns:
        The raw response object from ``client.responses.create()``.
    """
    start = time.perf_counter()
    response = client.responses.create(**kwargs)
    elapsed = time.perf_counter() - start
    tracker.record(response, elapsed, label=label)
    return response


def extract_text(response) -> str:
    """Extract concatenated text content from a Responses API response.

    Iterates over the response's output items, collecting text from all
    ``output_text`` blocks within ``message``-type items.

    Args:
        response: An OpenAI Responses API response object.

    Returns:
        str: The concatenated text content.  Returns an empty string if
            no text blocks are found.
    """
    parts = []
    for item in response.output:
        if item.type == "message":
            for block in item.content:
                if block.type == "output_text":
                    parts.append(block.text)
    return "".join(parts)
