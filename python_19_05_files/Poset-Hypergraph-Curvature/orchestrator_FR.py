import datetime
import subprocess
import sys
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
    script_poset = Path("generate_poset.py")
    script_geometry = Path("poset_geometry.py")

    # Build the exact input paths based on your rules
    node_path = Path(f"hypergraph_datasets/nodes/{token}.txt")
    hyperedge_path = Path(f"hypergraph_datasets/hyperedges/{token}.txt")

    # Route outputs into the dynamic, timestamped folder layout
    inter_dir = (
        Path("pipeline_outputs_FR") / run_folder_name / "intermediate"
    )
    final_dir = Path("pipeline_outputs_FR") / run_folder_name / "final"

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

    # --- PHASE 1: GENERATE POSET ---
    out1 = inter_dir / f"{token}_out1.txt"
    out2 = inter_dir / f"{token}_out2.txt"
    out3 = inter_dir / f"{token}_out3.txt"
    out4 = inter_dir / f"{token}_out4.txt"
    out5 = inter_dir / f"{token}_out5.txt"

    poset_command = [
        sys.executable,
        str(script_poset),
        str(node_path),
        str(hyperedge_path),
        str(out1),
        str(out2),
        str(out3),
        str(out4),
        str(out5),
    ]

    try:
        print("  └─ Running Phase 1: generate_poset.py...")
        subprocess.run(poset_command, check=True, capture_output=True, text=True)
        print("  [✓] Phase 1 complete.")
    except subprocess.CalledProcessError as e:
        print(
            f"  [X] Phase 1 Failed. Pipeline aborted.\nDetails: {e.stderr.strip()}"
        )
        sys.exit(1)

    # --- PHASE 2: POSET GEOMETRY ---
    geom_out1 = final_dir / f"{token}_geom1.txt"
    geom_out2 = final_dir / f"{token}_geom2.txt"
    geom_out3 = final_dir / f"{token}_geom3.txt"
    geom_out4 = final_dir / f"{token}_geom4.txt"

    geometry_command = [
        sys.executable,
        str(script_geometry),
        str(out1),
        str(out3),
        str(out5),
        str(geom_out1),
        str(geom_out2),
        str(geom_out3),
        str(geom_out4),
    ]

    try:
        print("  └─ Running Phase 2: poset_geometry.py...")
        subprocess.run(
            geometry_command, check=True, capture_output=True, text=True
        )
        print(f"  [✓] Phase 2 complete. Outputs saved to {final_dir}")
        print(f"[✓✓✓] Execution successful for target: {token}")
    except subprocess.CalledProcessError as e:
        print(f"  [X] Phase 2 Failed.\nDetails: {e.stderr.strip()}")
        sys.exit(1)


if __name__ == "__main__":
    run_targeted_pipeline()