"""
Generate synthetic clinical notes for the posit::conf 2026 talk demo.

Creates 30 SOAP-format clinical notes for a fictional patient spanning
multiple specialties and 3 years of care. Notes are written to
synthetic_data/clinical_notes.json.

Usage:
    uv run python code/generate_notes.py
    uv run python code/generate_notes.py --num-notes 10
    uv run python code/generate_notes.py --output synthetic_data/clinical_notes.json

The patient profile and specialty distribution can be edited directly in
this file (see PATIENT_PROFILE and NOTE_SCHEDULE below).
"""

import argparse
import json
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _setup import get_llm_client


# ─── Patient Profile ─────────────────────────────────────────────────────────

PATIENT_PROFILE = """
Patient: John TestPatient
Age: 68 years old (DOB: 1957-11-03)
Sex: Male
MRN: 9283741

Medical History:
- Hypertension (diagnosed 2010, on lisinopril 40mg + amlodipine 5mg)
- Type 2 Diabetes Mellitus (diagnosed 2015, on metformin 1000mg BID + empagliflozin 10mg daily)
- Mild osteoarthritis (bilateral knees, managed conservatively)
- Transient ischemic attack (TIA, 2025-03, resolved — initially attributed to small-vessel disease, started on aspirin + atorvastatin)
- Mild non-proliferative diabetic retinopathy (noted 2025-08)
- GERD (intermittent, on omeprazole PRN)
- Obesity (BMI 32)
- Recent episodic "lightheadedness" (2026-03, workup in progress — new-onset atrial fibrillation discovered on ED EKG; orthostatic hypotension also documented, possibly medication-related)

Current Medications:
- Lisinopril 40mg daily
- Amlodipine 5mg daily
- Metformin 1000mg BID
- Empagliflozin 10mg daily
- Aspirin 81mg daily
- Atorvastatin 40mg daily
- Omeprazole 20mg PRN
- (After afib diagnosis: apixaban 5mg BID, metoprolol 25mg BID added)

Social History:
- Retired schoolteacher, lives with wife
- Never smoker, no alcohol, no drugs
- Active in community garden, walks 20 min/day

Family History:
- Father: MI at age 72, T2DM
- Mother: Breast cancer at age 65 (survived), HTN
- Sister: T2DM
"""

# ─── Note Schedule ────────────────────────────────────────────────────────────
# Each entry: (date, specialty, provider, note_type)
# Spanning 2023-06 to 2026-07 (3 years of care)

NOTE_SCHEDULE = [
    ("2023-06-15", "Primary Care", "Dr. Sarah Patel", "Progress Note"),
    ("2023-08-22", "Endocrinology", "Dr. Ravi Krishnamurthy", "Consult Note"),
    ("2023-09-10", "Ophthalmology", "Dr. Linda Zhao", "Progress Note"),
    ("2023-11-02", "Primary Care", "Dr. Sarah Patel", "Progress Note"),
    ("2024-01-30", "Primary Care", "Dr. Sarah Patel", "Progress Note"),
    ("2024-03-14", "Cardiology", "Dr. Michael Torres", "Consult Note"),
    ("2024-04-20", "Gastroenterology", "Dr. Priya Sharma", "Progress Note"),
    ("2024-05-15", "Orthopedics", "Dr. Kevin Park", "Consult Note"),
    ("2024-06-28", "Primary Care", "Dr. Sarah Patel", "Progress Note"),
    ("2024-08-05", "Endocrinology", "Dr. Ravi Krishnamurthy", "Progress Note"),
    ("2024-10-01", "Primary Care", "Dr. Sarah Patel", "Progress Note"),
    ("2024-11-19", "Cardiology", "Dr. Michael Torres", "Progress Note"),
    ("2024-12-03", "Rheumatology", "Dr. Angela Morrison", "Consult Note"),
    ("2025-01-14", "Primary Care", "Dr. Sarah Patel", "Progress Note"),
    ("2025-02-20", "Gastroenterology", "Dr. Priya Sharma", "Progress Note"),
    ("2025-03-08", "Emergency Medicine", "Dr. Carlos Rivera", "ED Note"),         # TIA presentation
    ("2025-03-09", "Neurology", "Dr. Helen Wu", "Consult Note"),
    ("2025-03-11", "Primary Care", "Dr. Sarah Patel", "Hospital Discharge Follow-up"),
    ("2025-04-15", "Neurology", "Dr. Helen Wu", "Progress Note"),
    ("2025-05-20", "Cardiology", "Dr. Michael Torres", "Progress Note"),
    ("2025-06-10", "Primary Care", "Dr. Sarah Patel", "Progress Note"),
    ("2025-08-01", "Ophthalmology", "Dr. Linda Zhao", "Progress Note"),
    ("2025-09-18", "Endocrinology", "Dr. Ravi Krishnamurthy", "Progress Note"),
    ("2025-10-22", "Orthopedics", "Dr. Kevin Park", "Progress Note"),
    ("2026-01-15", "Primary Care", "Dr. Sarah Patel", "Progress Note"),
    ("2026-03-18", "Emergency Medicine", "Dr. Carlos Rivera", "ED Note"),         # Lightheadedness — afib discovered
    ("2026-03-19", "Cardiology", "Dr. Michael Torres", "Consult Note"),           # Inpatient cardiology consult for new afib
    ("2026-03-22", "Primary Care", "Dr. Sarah Patel", "Hospital Discharge Follow-up"),
    ("2026-04-10", "Cardiology", "Dr. Michael Torres", "Progress Note"),          # Afib follow-up, Holter results
    ("2026-05-15", "Primary Care", "Dr. Sarah Patel", "Progress Note"),           # Post-afib follow-up, med reconciliation
]


# ─── Generation Logic ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a medical documentation specialist generating realistic synthetic \
clinical notes for educational purposes. All data is entirely fictional.

You will generate a single clinical note in SOAP format (Subjective, Objective, \
Assessment, Plan). The note should:
- Be realistic in style, length (200-400 words), and clinical detail
- Use standard medical abbreviations where appropriate
- Reference the patient's known history naturally
- Include vital signs and relevant exam findings in the Objective section
- Be internally consistent with the patient's overall clinical trajectory
- Reflect the specialty and note type appropriately

Format the note with clear SOAP headers:
S:
O:
A:
P:

Do NOT include any preamble or explanation — output only the note text.
"""


def generate_single_note(
    client: OpenAI,
    model: str,
    date: str,
    specialty: str,
    provider: str,
    note_type: str,
    previous_notes_summary: str,
) -> str:
    """Generate one synthetic SOAP clinical note using the Responses API.

    Builds a prompt that incorporates the patient profile, the target
    note's metadata (date, specialty, provider, type), and a brief
    summary of all previously generated notes to ensure narrative
    continuity across the chart.

    Args:
        client: An OpenAI-compatible client instance.
        model: The LLM model name to use for generation.
        date: The note date as a string (e.g. ``"2024-03-14"``).
        specialty: Medical specialty (e.g. ``"Cardiology"``).
        provider: Provider name (e.g. ``"Dr. Michael Torres"``).
        note_type: The type of note (e.g. ``"Consult Note"``).
        previous_notes_summary: A running summary of notes generated so
            far, used to maintain continuity.

    Returns:
        str: The generated SOAP-format clinical note text.
    """
    user_prompt = (
        f"Generate a clinical note with the following details:\n\n"
        f"Patient Profile:\n{PATIENT_PROFILE}\n\n"
        f"Note Details:\n"
        f"- Date: {date}\n"
        f"- Specialty: {specialty}\n"
        f"- Provider: {provider}\n"
        f"- Note Type: {note_type}\n\n"
        f"Context from previous notes (for continuity):\n"
        f"{previous_notes_summary}\n\n"
        f"Write the SOAP note now."
    )

    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=user_prompt,
        temperature=0.8,
    )

    # Extract text from response output
    text_parts = []
    for item in response.output:
        if item.type == "message":
            for content_block in item.content:
                if content_block.type == "output_text":
                    text_parts.append(content_block.text)
    return "".join(text_parts)


def build_running_summary(notes_so_far: list[dict]) -> str:
    """Create a brief summary of notes generated so far for continuity.

    Produces a bullet-list of existing notes showing the date,
    specialty, provider, and first 100 characters of text.  This is
    passed to the LLM when generating subsequent notes so that each new
    note can reference prior findings.

    Args:
        notes_so_far: List of note dicts already generated, each with
            keys ``date``, ``specialty``, ``provider``, and ``text``.

    Returns:
        str: A multi-line string summarising prior notes, or
            ``"No previous notes."`` if the list is empty.
    """
    if not notes_so_far:
        return "This is the first note in the patient's chart."

    recent = notes_so_far[-5:]
    lines = []
    for note in recent:
        snippet = note["text"][:200].replace("\n", " ")
        lines.append(
            f"- {note['date']} ({note['specialty']}, {note['provider']}): "
            f"{snippet}..."
        )
    return "\n".join(lines)


def generate_all_notes(
    client: OpenAI,
    model: str,
    schedule: list[tuple],
    verbose: bool = True,
) -> list[dict]:
    """Generate all notes sequentially, maintaining narrative continuity.

    Iterates through the note schedule, generating one note at a time.
    Each call receives a running summary of all prior notes so the LLM
    can maintain a coherent patient storyline.

    Args:
        client: An OpenAI-compatible client instance.
        model: The LLM model name to use for generation.
        schedule: A list of ``(date, specialty, provider, note_type)``
            tuples defining the notes to generate.
        verbose: If ``True``, print progress for each note to stdout.

    Returns:
        list[dict]: A list of note dicts, each containing ``note_id``,
            ``date``, ``specialty``, ``provider``, ``note_type``, and
            ``text``.
    """
    notes = []

    for i, (date, specialty, provider, note_type) in enumerate(schedule):
        if verbose:
            print(
                f"  [{i+1}/{len(schedule)}] {date} — {specialty} "
                f"({provider})...",
                end=" ",
                flush=True,
            )

        previous_summary = build_running_summary(notes)
        text = generate_single_note(
            client=client,
            model=model,
            date=date,
            specialty=specialty,
            provider=provider,
            note_type=note_type,
            previous_notes_summary=previous_summary,
        )

        note = {
            "note_id": f"NOTE-{i+1:03d}",
            "date": date,
            "specialty": specialty,
            "provider": provider,
            "note_type": note_type,
            "text": text.strip(),
        }
        notes.append(note)

        if verbose:
            print("✓")

    return notes


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    """Generate synthetic clinical notes and write them to JSON.

    Parses ``--output``, ``--model``, and ``--num-notes`` from the
    command line, connects to the LLM via Azure KeyVault, generates
    SOAP-format notes according to ``NOTE_SCHEDULE``, and writes the
    result to the specified JSON output path (default:
    ``synthetic_data/clinical_notes.json``).
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic SOAP clinical notes for demo."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="synthetic_data/clinical_notes.json",
        help="Output path for JSON file (default: synthetic_data/clinical_notes.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.4-mini",
        help="LLM model to use for generation (default: gpt-5.4-mini)",
    )
    parser.add_argument(
        "--num-notes",
        type=int,
        default=None,
        help="Generate only the first N notes (default: all 30)",
    )
    args = parser.parse_args()

    # Resolve output path relative to project root
    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine how many notes to generate
    schedule = NOTE_SCHEDULE[: args.num_notes] if args.num_notes else NOTE_SCHEDULE

    print(f"Generating {len(schedule)} synthetic clinical notes...")
    print(f"  Model: {args.model}")
    print(f"  Output: {output_path}\n")

    # Initialize client
    print("  Connecting to LiteLLM proxy via Azure KeyVault...")
    client = get_llm_client()
    print("  ✓ Client ready\n")

    # Generate notes
    notes = generate_all_notes(client=client, model=args.model, schedule=schedule)

    # Write output
    output_data = {
        "patient": {
            "name": "John TestPatient",
            "mrn": "9283741",
            "dob": "1957-11-03",
            "sex": "Male",
        },
        "notes": notes,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Done! Wrote {len(notes)} notes to {output_path}")


if __name__ == "__main__":
    main()

