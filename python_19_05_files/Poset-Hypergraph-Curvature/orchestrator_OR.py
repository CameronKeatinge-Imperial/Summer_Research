import sys
import datetime
import subprocess
from pathlib import Path

def run_targeted_pipeline():
    # 1. Guard Clause: Ensure the user actually typed a name
    if len(sys.argv) < 2:
        print("[-] Error: Missing dataset name.")
        print("Usage:   python orchestrator.py <dataset-name>")
        print("Example: python orchestrator.py senate-committees")
        sys.exit(1)

    # Grab the target name directly from your command line entry
    token = sys.argv[1]

    # 2. Generate a Web-Safe Timestamp Folder Name
    # Format: YYYY-MM-DD_HH-MM (e.g., 2026-05-20_12-52)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_folder_name = f"{token}_{timestamp}"

    # 3. Define System Paths
    script_file1 = Path("bipartite_graph.py")
    script_file2 = Path("OR_calculate.py")

    # Build the exact input paths based on your rules
    node_path = Path(f"hypergraph_datasets/nodes/{token}.txt")
    hyperedge_path = Path(f"hypergraph_datasets/hyperedges/{token}.txt")

    # Route outputs into the dynamic, timestamped folder layout
    inter_dir = (
        Path("pipeline_outputs_OR") / run_folder_name / "intermediate"
    )
    final_dir = Path("pipeline_outputs_OR") / run_folder_name / "final"

    # 4. Safety Check: Verify files exist before doing any heavy lifting
    if not node_path.exists():
        print(f"[-] Error: Node file does not exist at: {node_path}")
        sys.exit(1)

    if not hyperedge_path.exists():
        print(f"[-] Error: Hyperedge file does not exist at: {hyperedge_path}")
        sys.exit(1)

    # Ensure target output directories exist
    # parents=True allows pathlib to recursively generate missing parent directories automatically
    inter_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    print(f"[>>>] Launching Isolated Pipeline for: {token}")
    print(f"[>>>] Target Output Sandbox: pipeline_outputs/{run_folder_name}/")

    # --- PHASE 1: FILE1.py ---
    # Generates 3 outputs based on the node and hyperedge files
    out1 = inter_dir / f"{token}_out1.txt"
    out2 = inter_dir / f"{token}_out2.txt"
    out3 = inter_dir / f"{token}_out3.txt"

    file1_command = [
        sys.executable,
        str(script_file1),
        str(node_path),
        str(hyperedge_path),
        str(out1),
        str(out2),
        str(out3),
    ]

    try:
        print("  └─ Running Phase 1: FILE1.py...")
        subprocess.run(file1_command, check=True, capture_output=True, text=True)
        print("  [✓] Phase 1 complete.")
    except subprocess.CalledProcessError as e:
        print(
            f"  [X] Phase 1 Failed. Pipeline aborted.\nDetails: {e.stderr.strip()}"
        )
        sys.exit(1)

    # --- PHASE 2: FILE2.py ---
    # Takes the 3 intermediate outputs and generates 1 final output
    final_out = final_dir / f"{token}_final_out.txt"

    file2_command = [
        sys.executable,
        str(script_file2),
        str(out1),
        str(out2),
        str(out3),
        str(final_out),
    ]

    try:
        print("  └─ Running Phase 2: FILE2.py...")
        subprocess.run(
            file2_command, check=True, capture_output=True, text=True
        )
        print(f"  [✓] Phase 2 complete. Output saved to {final_dir}")
        print(f"[✓✓✓] Execution successful for target: {token}")
    except subprocess.CalledProcessError as e:
        print(f"  [X] Phase 2 Failed.\nDetails: {e.stderr.strip()}")
        sys.exit(1)

if __name__ == "__main__":
    run_targeted_pipeline()