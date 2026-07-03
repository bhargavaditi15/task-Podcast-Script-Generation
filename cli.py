#!/usr/bin/env python
"""Terminal fallback for the podcast script generator.

Runs the exact same backend as the Streamlit app end-to-end: parse documents,
extract topics, generate the script, and (interactively) apply unlimited
modification cycles -- all from the command line, no UI required.

Example:
    python cli.py \\
        --docs sample_docs/electric_vehicles.txt \\
        --host-name Asha --guest-name Rahul \\
        --host-gender female --guest-gender male \\
        --host-speed 100 --guest-speed 90 \\
        --duration 15 \\
        --provider "Mock (offline/dev)" \\
        --output sample_outputs/sample_script.txt
"""

import argparse
import pathlib
import sys

from src.config import DURATION_OPTIONS_MIN, LLM_PROVIDERS
from src.documents.parser import DocumentParseError, extract_text
from src.llm.base import LLMError
from src.llm.factory import get_llm_client
from src.script.generator import generate_script, modify_script
from src.script.models import Speaker
from src.topics.extractor import classify_manual_topics, extract_topics
from src.utils.validation import validate_setup_inputs


def parse_args():
    # Parse command-line arguments for the CLI fallback.
    p = argparse.ArgumentParser(description="Generate a two-speaker podcast script from documents (CLI fallback).")
    p.add_argument("--docs", nargs="+", required=True, help="Paths to one or more PDF/DOC/DOCX/TXT files.")
    p.add_argument("--host-name", required=True)
    p.add_argument("--guest-name", required=True)
    p.add_argument("--host-gender", required=True, choices=["male", "female"])
    p.add_argument("--guest-gender", required=True, choices=["male", "female"])
    p.add_argument("--host-speed", type=int, default=100, help="50-150 scale, slow to fast.")
    p.add_argument("--guest-speed", type=int, default=100, help="50-150 scale, slow to fast.")
    p.add_argument("--duration", type=int, required=True, choices=DURATION_OPTIONS_MIN, help="Target duration in minutes.")

    p.add_argument("--provider", required=True, choices=LLM_PROVIDERS)
    p.add_argument("--api-key", default="", help="API key for the chosen provider (or set the matching env var).")
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default="", help="Only for 'Custom (OpenAI-compatible)' provider.")

    p.add_argument("--topics", default="", help="Comma-separated topics to include (default: all extracted topics).")
    p.add_argument("--extra-topics", default="", help="Comma-separated manually-typed topics to check against the documents.")
    p.add_argument("--list-topics-only", action="store_true", help="Extract and print topics, then exit.")

    p.add_argument("--output", default="script_output.txt", help="Where to save the generated script.")
    p.add_argument("--non-interactive", action="store_true", help="Skip the interactive modification loop.")
    return p.parse_args()


def load_documents(paths: list[str]) -> dict[str, str]:
    # Load and parse each provided document into plain text.
    doc_texts = {}
    for path_str in paths:
        path = pathlib.Path(path_str)
        if not path.exists():
            print(f"[warn] '{path}' does not exist -- skipping.", file=sys.stderr)
            continue
        try:
            doc_texts[path.name] = extract_text(path.read_bytes(), path.name)
        except DocumentParseError as exc:
            print(f"[warn] {exc}", file=sys.stderr)
    return doc_texts


def print_script(result):
    # Print the generated script and summary metrics to the terminal.
    print("\n" + "=" * 70)
    print(result.full_text)
    print("=" * 70)
    print(f"Target words: ~{result.target_words} | Actual words: {result.actual_words}")
    for w in result.warnings:
        print(f"[warning] {w}")


def main():
    # Main CLI orchestration: validate inputs, initialize LLM, extract topics,
    # generate the script, and optionally enter a modify loop.
    args = parse_args()

    doc_texts = load_documents(args.docs)

    errors = validate_setup_inputs(
        host_name=args.host_name,
        guest_name=args.guest_name,
        host_gender=args.host_gender,
        guest_gender=args.guest_gender,
        host_speed=args.host_speed,
        guest_speed=args.guest_speed,
        doc_texts=doc_texts,
        duration_minutes=args.duration,
        llm_provider=args.provider,
        llm_model=args.model,
        llm_api_key=args.api_key,
    )
    if errors:
        print("Cannot continue -- please fix the following:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

    try:
        llm = get_llm_client(provider=args.provider, api_key=args.api_key, model=args.model, base_url=args.base_url)
        if args.provider != "Mock (offline/dev)":
            llm.test_connection()
    except LLMError as exc:
        print(f"LLM setup failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Extracting topics from documents...")
    try:
        extracted_topics = extract_topics(doc_texts, llm)
    except LLMError as exc:
        print(f"Topic extraction failed: {exc}", file=sys.stderr)
        sys.exit(1)
    if not extracted_topics:
        print(
            "No extractable topics were found in the uploaded document(s) -- they may be too thin. "
            "Please provide additional/better document(s) and restart.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.list_topics_only:
        print("Extracted topics:")
        for t in extracted_topics:
            print(f"  - {t}")
        return

    selected = [t.strip() for t in args.topics.split(",") if t.strip()] if args.topics else list(extracted_topics)

    if args.extra_topics:
        manual = [t.strip() for t in args.extra_topics.split(",") if t.strip()]
        try:
            included, ignored = classify_manual_topics(manual, extracted_topics, doc_texts, llm)
        except LLMError as exc:
            print(f"Topic classification failed: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"Topics included (found in documents): {included or '(none)'}")
        print(f"Topics ignored (not found in documents): {ignored or '(none)'}")
        selected.extend(t for t in included if t not in selected)

    if not selected:
        print("No topics selected -- nothing to generate.", file=sys.stderr)
        sys.exit(1)

    host = Speaker(name=args.host_name, gender=args.host_gender, speed=args.host_speed)
    guest = Speaker(name=args.guest_name, gender=args.guest_gender, speed=args.guest_speed)

    print(f"Generating script for {len(selected)} topic(s), target duration {args.duration} min...")
    result = generate_script(
        topics=selected,
        host=host,
        guest=guest,
        duration_minutes=args.duration,
        doc_texts=doc_texts,
        llm=llm,
        progress_callback=lambda i, n, s: print(f"  [{i}/{n}] generated section: {s.kind} {s.topic or ''}".strip()),
    )
    print_script(result)

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.full_text, encoding="utf-8")
    print(f"\nSaved to {out_path}")

    if args.non_interactive or not sys.stdin.isatty():
        return

    while True:
        instruction = input("\nEnter a modification instruction (or press Enter to finish): ").strip()
        if not instruction:
            break
        print("Regenerating full script with your modification...")
        result = modify_script(result, instruction, host, guest, doc_texts, llm)
        print_script(result)
        out_path.write_text(result.full_text, encoding="utf-8")
        print(f"Saved updated script to {out_path}")


if __name__ == "__main__":
    main()
