#!/usr/bin/env python3
"""
Hypergraph Dataset Converter.

Converts tab-separated hyperedge lists (where each line is a space/tab separated 
list of vertex IDs representing a hyperedge) into a space-separated CSV format.
"""

import argparse
import logging
import sys
from pathlib import Path

# Configure clean, production-ready logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def convert_hypergraph(input_path: Path, output_path: Path) -> None:
    """
    Reads a tab-separated input file and standardizes it into a space-separated CSV.
    
    Handles empty lines, structural spaces, and trailing white space variations.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Source file not found at: {input_path}")

    logger.info(f"Processing input file: {input_path}")
    processed_count = 0

    # Read and write using streaming context managers for low memory overhead
    with open(input_path, mode='r', encoding='utf-8') as infile, \
         open(output_path, mode='w', encoding='utf-8', newline='') as outfile:
        
        for line_num, line in enumerate(infile, start=1):
            # Clean up leading/trailing whitespaces and line breaks
            cleaned_line = line.strip()
            
            # Defensive check: skip accidental empty lines elegantly
            if not cleaned_line:
                continue
                
            # Split elements by any whitespace (tabs, consecutive spaces, etc.)
            tokens = cleaned_line.split()
            
            # Join tokens uniformly with a single spaces to mirror toy_hypernetwork structure
            standardized_line = " ".join(tokens)
            
            # Write out to destination CSV path
            outfile.write(standardized_line + "\n")
            processed_count += 1

    logger.info(f"Successfully converted {processed_count} lines. Output saved to: {output_path}")


def main() -> None:
    """Configures argument passing interfaces and handles top-level execution."""
    parser = argparse.ArgumentParser(
        description="Convert tab-separated hypergraphs to space-separated network structures."
    )
    parser.add_argument(
        "input", 
        type=str, 
        help="Path to the input text file (e.g., senate-commitees.txt)"
    )
    parser.add_argument(
        "output", 
        type=str, 
        help="Destination path for the final output (e.g., toy_hypernetwork.csv)"
    )

    args = parser.parse_args()

    try:
        convert_hypergraph(Path(args.input), Path(args.output))
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()