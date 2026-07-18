#dual for ORC
import os
import networkx as nx
from itertools import combinations
import itertools as it
from datetime import datetime
import re
import numpy as np
from pathlib import Path

from src.iterative_clustering import StatisticalModelTemplate
from src.create_dual_networks import process_and_save_dual_complexes

class OllivierRicciDualClustering(StatisticalModelTemplate):
    
    def __init__(self):
        super().__init__()
        self.network_decomposition = "dual_networks"

    def construct_network(self,data_source,dataset_name):
        #from the hypernetwork file, get all the subnetworks
        #can add if need the functionality?
        SCRIPT_DIR = Path(__file__).resolve().parent
        BASE_DATA_DIR = SCRIPT_DIR.parent / "data"
        data_source = os.path.join(BASE_DATA_DIR, data_source)
        process_and_save_dual_complexes(data_source,dataset_name)

    def files_for_network(self,source,name):
        needed_info = ["nodes","edges"]
        paths_search = []
        base_dir = Path("data")
        #but now need to get the networks of all the networks of different cardinality
        file_of_nodes = os.path.join(base_dir, source, self.network_decomposition, "nodes", name)
        self.hyperedge_cardinalities = self.extract_cardinalities_from_files(file_of_nodes)

        for c in self.hyperedge_cardinalities:
            #add these as a sublist/array, so now 2d
            cardinality_pairs = []
            for n in needed_info:
                n_path_string = os.path.join(base_dir, source, self.network_decomposition, n, name, f"{n}_k{c}.txt")
                cardinality_pairs.append(n_path_string)
            paths_search.append(cardinality_pairs)
        return paths_search
    

    def extract_cardinalities_from_files(self, folder_path):
        """
        read all the files from this file location, but from their names in the form nodes_k{number}.txt
        """
        cardinalities = []
        
        # Define a regex pattern: 'nodes_k' followed by one or more digits (\d+), ending in '.txt'
        # The parenthesis () create a capture group for just the digits
        pattern = re.compile(r'^nodes_k(\d+)\.txt$')
        
        try:
            # List all files in the given directory
            for filename in os.listdir(folder_path):
                match = pattern.match(filename)
                if match:
                    # Extract the captured number string and convert it to an int
                    number = int(match.group(1))
                    cardinalities.append(number)
        except FileNotFoundError:
            print(f"Error: The folder '{folder_path}' does not exist.")
            return []

        # Return the numbers sorted for easier processing later
        return sorted(cardinalities)

    def network_from_files(self,file_location_verified,paths_tuples):
        #initialNetwork will be a dictionary of initial networks
        self.initialNetwork = {c: None for c in self.hyperedge_cardinalities}
        print(range(len(self.hyperedge_cardinalities)))
        for i in range(len(self.hyperedge_cardinalities)):
            smallNetworkPaths = paths_tuples[i]
            #extract cardinality, so checking that paths_tuples[i] matches self.hyperedge_cardinalities
            pattern = re.compile(r'_k(\d+)\.txt$')  
            match = pattern.search(smallNetworkPaths[0])
            if match:
                # 3. Extract the actual string digits (e.g., "5") and convert to an integer
                network_card = int(match.group(1))

            print(f"Construct graph object for cardinality {network_card}")
            G = nx.Graph()
            # 1. Stream edges into the graph
            # (line.split() handles both spaces and tabs automatically)
            # 2. Stream nodes to ensure isolated nodes (nodes with no edges) are included
            with open(smallNetworkPaths[0], 'r') as f:
                node_generator = (line.strip() for line in f if line.strip())
                G.add_nodes_from(node_generator)

            with open(smallNetworkPaths[1], 'r') as f:
                edge_generator = (line.split() for line in f if line.strip())
                G.add_edges_from(e for e in edge_generator if len(e) >= 2)
                
            self.initialNetwork[network_card] = G
    
    def hyperedge_removal(self):
        pass
    def initialise_curvature(self):
        pass
    def recalculate_curvature(self):
        pass




def parse_mixed_input_file(file_path):
    """
    Reads an input file and automatically groups simplices by their dimension k
    based on the number of vertices found in each entry.
    """
    # Dictionary to hold lists of simplices: { k: [ [v1, v2..], [v1, v2..] ] }
    dimension_groups = {}
    
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
                    
                    # Number of vertices = k + 1 -> therefore k = len(vertices) - 1
                    k = len(vertices) - 1
                    
                    if k not in dimension_groups:
                        dimension_groups[k] = []
                    dimension_groups[k].append(vertices)
            except ValueError:
                print(f"Skipping line {line_idx}: Could not parse integers.")
                
    return dimension_groups


def process_and_save_dual_complexes(input_file, chosen_string):
    """
    Scans the input file, loops through every discovered k value, 
    and saves a nodes file and an edges file for each.
    """
    # 1. Setup timestamp and naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 2. Parse and group input data by dimension
    print("Scanning input file and sorting by cell dimensions...")
    groups = parse_mixed_input_file(input_file)
    
    if not groups:
        print("No valid data found.")
        return

    # 3. Iterate through each detected dimension k
    for k in sorted(groups.keys()):
        simplices = groups[k]
        vertex_count = k + 1
        print(f"\nProcessing dimension k = {k} ({len(simplices)} cells found)...")
        
        # Define output filenames exactly matching your specifications
        nodes_filename = f"pipeline_outputs_OR_dual/{chosen_string}/nodes_k{k}.txt"
        edges_filename = f"pipeline_outputs_OR_dual/{chosen_string}/edges_k{k}.txt"
        
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


# --- Example Execution Execution ---
if __name__ == "__main__":
    # Change these values to match your files
    #TARGET_DIMENSION = 3      # k=2 means we are analyzing triangles sharing an edge
    #OUTPUT_FILENAME = "dual_network.graphml"
    DATASET_NAME = "trial_mini2"
    INPUT_FILENAME = f"./hypergraph_datasets/hyperedges/{DATASET_NAME}.txt"

    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #OUTPUT_FILENAME = f"pipeline_outputs_OR_dual/dual_network_{DATASET_NAME}_{timestamp}.graphml"
    
    # Generate the graph
    dual_net = process_and_save_dual_complexes(
        input_file=INPUT_FILENAME, 
        chosen_string=DATASET_NAME
    )