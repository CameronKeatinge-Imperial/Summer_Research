#!/usr/bin/env python3
"""
Hyperedge Duplicate Checker for Tab-Separated Text Files.

Usage: python check_duplicates.py <dataset-name>
Example: python check_duplicates.py senate-commitees
"""

import sys
from collections import Counter
from pathlib import Path


def analyze_duplicate_hyperedges(target_name: str) -> None:
    # Build path targeting your raw input folder structure
    file_path = Path(f"hypergraph_datasets/hyperedges/{target_name}.txt")

    if not file_path.exists():
        print(f"[-] Error: Target hyperedge file does not exist at: {file_path}")
        print("    Please ensure your file is at that exact path.")
        sys.exit(1)

    edge_registry = Counter()
    total_lines = 0

    print(f"[>] Scanning raw text file: {file_path}")

    # Open using generic text mode; Python handles \r\n automatically
    with open(file_path, mode="r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            cleaned_line = line.strip()
            
            # Defensive check: safely skip empty rows or trailing lines
            if not cleaned_line:
                continue

            total_lines += 1

            try:
                # .split() with no arguments splits on ANY whitespace (tabs or spaces)
                # This guarantees smooth parsing of your tab-separated rows
                node_ids = list(map(int, cleaned_line.split()))
                
                # 1. Cast to a set to remove duplicates inside the single row
                unique_nodes = set(node_ids)

                # 2. Sort the nodes to make the signature order-agnostic
                canonical_shape = tuple(sorted(unique_nodes))

                # 3. Increment the occurrence count for this specific shape
                edge_registry[canonical_shape] += 1

            except ValueError:
                print(f"[!] Warning: Non-integer data encountered on line {line_num}. Skipping row.")
                continue

    # --- Metrics Calculations ---
    unique_shapes_count = len(edge_registry)
    total_duplicate_edges = sum(count - 1 for count in edge_registry.values() if count > 1)
    duplicate_groups = sum(1 for count in edge_registry.values() if count > 1)

    # --- Terminal UI Output ---
    print("\n" + "=" * 50)
    print(f" DATA INTEGRITY REPORT FOR TARGET: {target_name.upper()}")
    print("=" * 50)
    print(f" Total rows parsed from file:   {total_lines}")
    print(f" Unique topological structures: {unique_shapes_count}")
    print(f" Redundant/Duplicate edges:     {total_duplicate_edges}")
    print(f" Distinct overlapping groups:   {duplicate_groups}")
    print("=" * 50)

    # Show the top duplicated node shapes if any exist
    if total_duplicate_edges > 0:
        print("\n[!] Top 3 Most Frequently Repeated Hyperedge Shapes:")
        most_common = edge_registry.most_common(3)
        for shape, count in most_common:
            if count > 1:
                print(f"    - Nodes {list(shape)} appeared {count} times in the data.")
    print()


if __name__ == "__main__":
    # Handle CLI bounds checks
    if len(sys.argv) < 2:
        print("[-] Error: Missing dataset target string name.")
        print("Usage:   python check_duplicates.py <dataset-name>")
        print("Example: python check_duplicates.py senate-commitees")
        sys.exit(1)

    analyze_duplicate_hyperedges(sys.argv[1])