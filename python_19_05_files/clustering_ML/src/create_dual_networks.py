##dual for ORC
import os
import networkx as nx
from itertools import combinations
from datetime import datetime
from pathlib import Path

# add to a class

def process_and_save_dual_complexes(data_source,  dataset_name):
    """
    Scans the input file, loops through every discovered k value, 
    and saves a nodes file and an edges file for each.
    """
    data_source_path = Path(data_source)
    input_file = data_source_path / "hypernetwork_form" / "edges" / f"{dataset_name}.txt"
    key_file = data_source_path / "dual_networks" / "hyperedge_node_key" / f"{dataset_name}.txt"


    # 2. Parse and group input data by dimension
    groups = parse_mixed_input_file(input_file,key_file)
    
    if not groups:
        print("No valid data found.")
        return

    # 3. Iterate through each detected dimension k
    for k in sorted(groups.keys()):
        simplices = groups[k]
        vertex_count = k + 1
        print(f"\nProcessing dimension k = {k} ({len(simplices)} cells found)...")
        
        nodes_filename = data_source_path / "dual_networks" / "nodes" / dataset_name / f"nodes_k{k}.txt"
        edges_filename = data_source_path / "dual_networks" / "edges" / dataset_name / f"edges_k{k}.txt"
        
        #nodes_filename
        try:
            nodes_filename.parent.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            # This catches the scenario where a file shares the name of a planned folder
            print(f"Could not create directory '{nodes_filename.parent.mkdir}'.")
            print("A file already exists in this location.")
        #edge_filename
        try:
            edges_filename.parent.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            # This catches the scenario where a file shares the name of a planned folder
            print(f"Could not create directory '{edges_filename.parent.mkdir}'.")
            print("A file already exists in this location.")

        # --- STEP A: Build Node Map & Write Nodes File ---
        face_to_simplices = {}
        
        with open(nodes_filename, 'w') as node_file:
            for idx, vertices in enumerate(simplices):
                # Write the node file exactly as requested: [v1, v2, v3...]
                node_file.write(f"{vertices}\n")
                
                # Compute all (k-1) dimensional faces (subsets of size k)
                for face in combinations(vertices, k):
                    face_key = frozenset(face)
                    if face_key not in face_to_simplices:
                        face_to_simplices[face_key] = []
                    # Store the actual vertex representation of the simplex
                    face_to_simplices[face_key].append(vertices)

        # --- STEP B: Find Neighbors & Write Edges File ---
        edge_count = 0
        with open(edges_filename, 'w') as edge_file:
            for face, sharing_simplices in face_to_simplices.items():
                if len(sharing_simplices) > 1:
                    # Connect any pair of simplices sharing this face
                    for u, v in combinations(sharing_simplices, 2):
                        # Write the edge exactly as requested: [vertices_A], [vertices_B]
                        edge_file.write(f"{u}, {v}\n")
                        edge_count += 1
                        
        print(f" -> Saved nodes to: {nodes_filename}")
        print(f" -> Saved {edge_count} dual connections to: {edges_filename}")

    print("\nAll dimensions processed successfully!")


def parse_mixed_input_file(file_path, output_mapping_path):
    """
    Reads an input file, automatically groups simplices by their dimension k,
    assigns a unique Node ID to each unique hyperedge, and optionally writes 
    the bijective mapping to a file.
    """
    # Dictionary to hold lists of simplices: { k: [ [v1, v2..], [v1, v2..] ] }
    # Track unique hyperedges to maintain a strict bijection and avoid duplicates

    dimension_groups = {}
    seen_hyperedges = {}
    node_counter = 0

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file '{file_path}' not found.")

    with open(file_path, 'r') as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Handle commas or spaces
            delimiter = ',' if ',' in line else None
            try:
                vertices = [int(v) for v in line.split(delimiter) if v.strip()]
                if vertices:
                    # Sort vertices to ensure unique, unoriented face hashing downstream
                    vertices.sort() 
                    
                    # Convert to a tuple so it can be used as a dictionary key
                    hyperedge_tuple = tuple(vertices)
                    
                    # If we haven't seen this hyperedge yet, assign it a unique Node ID
                    if hyperedge_tuple not in seen_hyperedges:
                        seen_hyperedges[hyperedge_tuple] = node_counter
                        node_counter += 1
                    
                    # Number of vertices = k + 1 -> therefore k = len(vertices) - 1
                    k = len(vertices) - 1
                    
                    if k not in dimension_groups:
                        dimension_groups[k] = []
                    
                    # To prevent duplicate entries within the same dimension group
                    if vertices not in dimension_groups[k]:
                        dimension_groups[k].append(vertices)
                        
            except ValueError:
                print(f"Skipping line {line_idx}: Could not parse integers.")

    # Writing Component: Save the bijection if a path is provided
    if output_mapping_path and seen_hyperedges:
        # Sort by Node ID so the output file is cleanly organized (0, 1, 2...)
        sorted_mapping = sorted(seen_hyperedges.items(), key=lambda item: item[1])
        
        with open(output_mapping_path, 'w') as out_f:
            out_f.write("# node_id:hyperedge_definition\n") # Optional header
            for hyperedge, node_id in sorted_mapping:
                edge_str = ",".join(map(str, hyperedge))
                out_f.write(f"{node_id}:{edge_str}\n")
        print(f"Success: Bijective mapping saved to '{output_mapping_path}'")
                
    return dimension_groups