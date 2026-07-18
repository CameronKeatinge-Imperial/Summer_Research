import networkx as nx
import hypernetx as hnx
import pandas as pd
import os
import pathlib as Path
import sys

from src2.posetNetworkClass import PosetNetworkObject
from src2.hypernetwork_class import HypernetworkObject
from src2.data_object import MappingOfHyperedges
from src2.algorithm_class import NetworkProcessor
from src.data_processing import load_config

# ---------------------------------------------------------
# 1. Mock Imports / Instantiations (Replace with your actual imports)
# ---------------------------------------------------------
# from your_module import Network1, MappingOfHyperedges, initialHypernetwork

# ---------------------------------------------------------
# 2. Test Script Definition
# ---------------------------------------------------------
def run_mapping_tests(network, hypernetwork, mapping):
    """
    Tests whether the nodes and hyperedges defined in the mapping 
    exist in the provided network and hypernetwork respectively.
    """
    print("Starting Mapping Validation Tests...\n")
    
    # Trackers for our test metrics
    node_matches = 0
    node_total = len(mapping._to_spec2)
    
    hyperedge_matches = 0
    hyperedge_total = len(mapping._to_spec1)
    
    # ---------------------------------------------------------
    # TEST 1: Check if nodes from mapping exist in Network1
    # ---------------------------------------------------------
    # Assuming `network` is a NetworkX graph or similar object with a `.nodes` collection
    network_nodes = set(network.nodes)
    
    for node_id in mapping._to_spec2.keys():
        if node_id in network_nodes:
            node_matches += 1
        else:
            print(f"[Warning] Node {node_id} from mapping not found in Network1.")
            
    # ---------------------------------------------------------
    # TEST 2: Check if hyperedges from mapping exist in Hypernetwork
    # ---------------------------------------------------------
    # HyperNetX edges are stored as sets of nodes. We extract all sets to check against.
    # Note: hnx.edges[e] behaves like an iterable of nodes in that specific hyperedge
    existing_hyperedges = []
    for e in hypernetwork.edges:
        # Extract the nodes for each hyperedge and store as a frozenset for easy comparison
        nodes_in_edge = frozenset(hypernetwork.edges[e])
        existing_hyperedges.append(nodes_in_edge)
        
    for hyperedge_str in mapping._to_spec1.keys():
        # The key file snippet notes this is a "comma string" (e.g., "1, 2, 3")
        try:
            # Parse the string into a frozenset of integers
            hyperedge_nodes = frozenset(int(n.strip()) for n in hyperedge_str.split(','))
            
            # Check if this exact set of nodes exists as an edge in the hypernetwork
            if hyperedge_nodes in existing_hyperedges:
                hyperedge_matches += 1
            else:
                print(f"[Warning] Hyperedge '{hyperedge_str}' not found in Hypernetwork.")
                
        except ValueError:
            print(f"[Error] Could not parse hyperedge string: '{hyperedge_str}'")

    # ---------------------------------------------------------
    # 3. Output Results
    # ---------------------------------------------------------
    print("\n" + "="*40)
    print("TEST RESULTS")
    print("="*40)
    
    print(f"Node Match: {node_matches} / {node_total} "
          f"({(node_matches/node_total)*100:.2f}%)")
    
    print(f"Hyperedge Match: {hyperedge_matches} / {hyperedge_total} "
          f"({(hyperedge_matches/hyperedge_total)*100:.2f}%)")
    
    if node_matches == node_total and hyperedge_matches == hyperedge_total:
        print("\nSUCCESS: All mapped entities are present in their respective networks!")
    else:
        print("\nFAILURE: Some mapped entities are missing. Check the warnings above.")

# ---------------------------------------------------------
# Example Execution Trigger
# ---------------------------------------------------------
if __name__ == "__main__":
    # Assuming mapping is instantiated like this:
    config_path = "../config/config.yaml"
    config = load_config(config_path)
    paths_obj = NetworkProcessor(config)

    paths_obj.files_for_network()
    paths_obj.files_for_hypernetwork()
    paths_obj.hyperedge_key_file()


    #base_dir = Path("data")
    #source = "real-world data"
    #dataset_name = "senate-commitees"

    my_mapping = MappingOfHyperedges(paths_obj.hyperedge_key_file())
    network_obj = PosetNetworkObject(paths_obj.files_for_network())
    hypernetwork_obj = HypernetworkObject(paths_obj.files_for_hypernetwork())
    network_in, a = network_obj.network_from_files(paths_obj.files_for_network())
    hypernetwork_in = hypernetwork_obj.hypernetwork_from_files(paths_obj.files_for_hypernetwork())
    print("files successfully read in")
    run_mapping_tests(network=network_in, hypernetwork=hypernetwork_in, mapping=my_mapping)
