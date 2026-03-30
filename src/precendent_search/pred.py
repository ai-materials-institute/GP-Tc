#!/usr/bin/env python3
"""
Query materials from combined_predictions.csv using Google Gemini API.

This script processes the CSV file in batches, querying the Gemini API to identify
related superconductor materials mentioned in the formatted_answer column.

Based on phantom-wiki's __main__.py structure with async batch processing.

Authors
-------
Albert Gong and Anmol Kabra
"""

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path
import re
import pandas as pd

from gemini_utils import GeminiChat, InferenceGenerationConfig, LLMChatResponse


SCRIPT_DIR = Path(__file__).resolve().parent

def create_prompt(reduced_formula: str, formatted_answer: str, prompt_template: str) -> str:
    """Create prompt from the prompt template."""
    # assert that "{{reduced_formula}}" and "{{formatted_answer}}" are in the prompt_template
    assert "{{reduced_formula}}" in prompt_template, "Prompt template must contain '{{reduced_formula}}'"
    assert "{{formatted_answer}}" in prompt_template, "Prompt template must contain '{{formatted_answer}}'"
    # import pdb; pdb.set_trace()
    prompt = prompt_template.replace("{{reduced_formula}}", reduced_formula).replace("{{formatted_answer}}", formatted_answer)
    return prompt

def extract_boxed_answer(answer: str) -> str:
    """Extract text within \\boxed{...} from LaTeX answer string.
    
    Args:
        answer: LaTeX formatted answer string
    Returns:
        Extracted text
    
    """
    pattern = r"\\boxed\{(.*?)\}"
    match = re.search(pattern, answer)
    if match:
        return match.group(1)
    else:
        return ""

def save_preds(
    pred_path: Path,
    batch_df: pd.DataFrame,
    prompts: list[str],
    responses: list[LLMChatResponse],
    args: argparse.Namespace,
    batch_number: int,
) -> None:
    """
    Save predictions to JSON file as a list of dictionaries.

    Matches the format from query_materials_with_edison.py for compatibility
    with combine_predictions.py.
    """
    preds = []
    batch_size = len(batch_df)

    for i, (row_idx, row) in enumerate(batch_df.iterrows()):
        reduced_formula = str(row.get("reduced_formula", ""))
        formatted_answer = str(row.get("formatted_answer", ""))
        pred = responses[i].pred

        pred_dict = {
            "icsd_id": row.get("icsd_id", None),
            "reduced_formula": reduced_formula,
            "batch_number": row.get("batch_number", None),
            "query": row.get("query", None),
            "formatted_answer": formatted_answer,
            "prompt": prompts[i],
            "gemini_response": pred,
            "gemini_answer": extract_boxed_answer(pred),
            "error": responses[i].error,
            "status": "success" if responses[i].error is None else "error",
            "metadata": {
                "model": args.model_name,
                "batch_size": batch_size,
                "batch_number": batch_number,
                "row_index": str(row_idx),  # Store as string for JSON compatibility
            },
            "usage": responses[i].usage,
        }
        preds.append(pred_dict)

    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pred_path, "w") as f:
        json.dump(preds, f, indent=2, default=str)
        f.flush()

    print(f"✓ Saved predictions to {pred_path}")


async def process_batch(
    llm_chat: GeminiChat,
    batch_df: pd.DataFrame,
    inf_gen_config: InferenceGenerationConfig,
    args: argparse.Namespace,
    batch_number: int,
    batch_size: int,
    prompt_template: str,
    preds_dir: Path,
) -> None:
    """
    Process a single batch of rows.

    Uses asyncio.gather to process rows in parallel.
    """
    # Create output path matching the edison pattern: gemini_batch={N}__bs={size}.json
    run_name = f"gemini_batch={batch_number}__bs={batch_size}"
    pred_path = preds_dir / f"{run_name}.json"

    # Skip if output already exists and --force is not set
    if pred_path.exists() and not args.force:
        print(f"⚠ Skipping {pred_path} as it already exists. Use --force to overwrite.")
        return

    print(f"\n{'='*60}")
    print(f"Processing batch {batch_number}")
    print(f"Rows: {len(batch_df)}")
    print(f"{'='*60}")

    # Process all rows in batch using asyncio.gather
    print(f"Querying Gemini API for {len(batch_df)} rows...")
    prompts = [
        create_prompt(
            str(row.get("reduced_formula", "")),
            str(row.get("formatted_answer", "")),
            prompt_template,
        )
        for _, row in batch_df.iterrows()
    ]
    responses: list[LLMChatResponse] = await asyncio.gather(
        *[
            llm_chat.chat_async(prompt, inf_gen_config)
            for prompt in prompts
        ]
    )

    # Count successes and errors
    num_success = sum(1 for r in responses if r.error is None)
    num_errors = len(responses) - num_success
    print(f"✓ Completed: {num_success} successful, {num_errors} errors")

    # Save predictions
    save_preds(pred_path, batch_df, prompts, responses, args, batch_number)


async def main(args: argparse.Namespace) -> None:
    """Main function to run the script."""
    # Load data
    print(f"Loading data from {args.input}...")
    try:
        df = pd.read_csv(args.input)
        print(f"✓ Loaded {len(df)} rows")
        if True:
            # filter out rows where known_superconductor is TRUE
            df = df[df["known_superconductor"] != True]
            print(f"✓ Filtered down to {len(df)} rows where known_superconductor = FALSE")
    except FileNotFoundError:
        print(f"✗ Error: Input file not found: {args.input}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        sys.exit(1)

    # Initialize Gemini client
    print(f"Initializing Gemini client with model '{args.model_name}'...")
    try:
        llm_chat = GeminiChat(model_name=args.model_name)
        print("✓ Gemini client initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing Gemini client: {e}")
        sys.exit(1)

    # load prompt template
    prompt_path = SCRIPT_DIR / "prompts" / f"{args.prompt}.md"
    with open(prompt_path, "r") as f:
        prompt_template = f.read()
    print(f"✓ Loaded prompt template from {prompt_path}")

    # Setup inference configuration
    inf_gen_config = InferenceGenerationConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        max_retries=args.max_retries,
        wait_seconds=args.wait_seconds,
    )

    # Process batches
    num_rows = len(df)
    batch_size = args.batch_size
    num_batches = math.ceil(num_rows / batch_size)
    preds_dir = args.output_dir / args.prompt
    preds_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing Configuration")
    print(f"{'='*60}")
    print(f"Total rows: {num_rows}")
    print(f"Batch size: {batch_size}")
    print(f"Total batches: {num_batches}")
    print(f"{'='*60}")

    for batch_number in range(1, num_batches + 1):
        # Skip if specific batch number is requested and this isn't it
        if args.batch_number is not None and batch_number != args.batch_number:
            continue

        # Calculate batch indices
        batch_start_idx = (batch_number - 1) * batch_size
        batch_end_idx = min(batch_start_idx + batch_size, num_rows)

        print(
            f"\nBatch {batch_number}/{num_batches}: rows [{batch_start_idx}, {batch_end_idx})"
        )

        # Get batch data
        batch_df = df.iloc[batch_start_idx:batch_end_idx]

        # Process batch
        try:
            await process_batch(
                llm_chat, batch_df, inf_gen_config, args, batch_number, batch_size,
                prompt_template, preds_dir,
            )
        except Exception as e:
            print(f"✗ Error processing batch {batch_number}: {e}")
            import traceback

            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print("All batches complete!")
    print(f"{'='*60}")


def get_parser() -> argparse.ArgumentParser:
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        description="Query materials using Google Gemini API"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=SCRIPT_DIR / "combined_predictions.csv",
        help="Input CSV file with materials (default: combined_predictions.csv)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "preds",
        help="Output directory for results (default: preds)",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="floor",
        help="Prompt template to use (default: floor)",
    )

    parser.add_argument(
        "--batch-size",
        "-bs",
        type=int,
        default=100,
        help="Number of rows to process in each batch (default: 100)",
    )

    parser.add_argument(
        "--batch-number",
        "-bn",
        type=int,
        default=None,
        help="Specific batch number to process (1-indexed). If not set, processes all batches",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="gemini-3-flash-preview",
        help="Gemini model name (default: gemini-3-flash-preview)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing results",
    )

    # Inference configuration
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Maximum tokens in response (default: 1024)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for generation (default: 1.0)",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling (default: use model default)",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p (nucleus) sampling (default: use model default)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries on API errors (default: 3)",
    )

    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=1.0,
        help="Initial wait time between retries in seconds (default: 1.0)",
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    asyncio.run(main(args))
