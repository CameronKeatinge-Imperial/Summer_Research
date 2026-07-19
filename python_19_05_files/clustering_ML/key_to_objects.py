import networkx as nx
import hypernetx as hnx
import pandas as pd
import os
import pathlib as Path
import sys
import xgi 

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
            
    '''
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
    '''
    
    # ---------------------------------------------------------
    # TEST 2: Check if hyperedges from mapping exist in Hypernetwork (XGI UPDATED)
    # ---------------------------------------------------------
    existing_hyperedges = []
    
    # In XGI, hypernetwork.edges.members() returns an iterable of the node sets 
    # for all edges. We can convert them directly to frozensets.
    for members in hypernetwork.edges.members():
        existing_hyperedges.append(frozenset(members))
        
    for hyperedge_str in mapping._to_spec1.keys():
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
    # TEST 3: Check Bijection (Node -> Hyperedge -> Node)
    # ---------------------------------------------------------
    bijection_n2h_matches = 0
    
    for node_id in mapping._to_spec2.keys():
        try:
            # Map forward then backward
            mapped_hyperedge = mapping.node_to_hyperedge_map(node_id)
            reversed_node = mapping.hyperedge_to_node_map(mapped_hyperedge)
            
            if reversed_node == node_id:
                bijection_n2h_matches += 1
            else:
                print(f"[Warning] Bijection failed: Node {node_id} mapped back to {reversed_node}")
        except KeyError:
            print(f"[Error] Bijection broken: Missing reverse mapping for Node {node_id}")

    # ---------------------------------------------------------
    # TEST 4: Check Bijection (Hyperedge -> Node -> Hyperedge)
    # ---------------------------------------------------------
    bijection_h2n_matches = 0
    
    for hyperedge_str in mapping._to_spec1.keys():
        try:
            # Map forward then backward
            mapped_node = mapping.hyperedge_to_node_map(hyperedge_str)
            reversed_hyperedge = mapping.node_to_hyperedge_map(mapped_node)
            
            if reversed_hyperedge == hyperedge_str:
                bijection_h2n_matches += 1
            else:
                print(f"[Warning] Bijection failed: Hyperedge '{hyperedge_str}' mapped back to '{reversed_hyperedge}'")
        except KeyError:
            print(f"[Error] Bijection broken: Missing reverse mapping for Hyperedge '{hyperedge_str}'")
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

    print(f"Bijection (Node -> Hyperedge -> Node): {bijection_n2h_matches} / {node_total}")
    print(f"Bijection (Hyperedge -> Node -> Hyperedge): {bijection_h2n_matches} / {hyperedge_total}")

def run_hyperedge_to_edge_id_tests(hypernetwork_obj):

    print("Starting hyperedge_to_edge_id() Tests...\n")
    passed = 0
    total = 0

    existing_edge_ids = list(hypernetwork_obj.itertative_H.edges)
    sample_nodes, expected_edge_id = next(iter(hypernetwork_obj.node_to_edge_id_map.items()))

    # 1. Passthrough
    total += 1
    test_edge_id = existing_edge_ids[0]
    result = hypernetwork_obj.hyperedge_to_edge_id(test_edge_id)
    if result == test_edge_id:
        print(f"[Pass] Edge id passthrough works for {test_edge_id}")
        passed += 1
    else:
        print(f"[Fail] Passthrough expected {test_edge_id}, got {result}")

    # 2. Node list -> edge id
    total += 1
    result = hypernetwork_obj.hyperedge_to_edge_id(list(sample_nodes))
    if result == expected_edge_id:
        print(f"[Pass] Nodes {sample_nodes} -> edge id {expected_edge_id}")
        passed += 1
    else:
        print(f"[Fail] Nodes {sample_nodes} -> {result}, expected {expected_edge_id}")
    # Diagnostic: confirm 'in' works against itertative_H.edges directly
        real_edge_id = existing_edge_ids[0]
        direct_check = real_edge_id in hypernetwork_obj.itertative_H.edges
        print(f"[Diag] Direct 'in' check for {real_edge_id!r}: {direct_check}")
    # 3. Order independence
    total += 1
    shuffled = list(reversed(sample_nodes))
    result = hypernetwork_obj.hyperedge_to_edge_id(shuffled)
    if result == expected_edge_id:
        print(f"[Pass] Unordered nodes {shuffled} -> edge id {expected_edge_id}")
        passed += 1
    else:
        print(f"[Fail] Unordered nodes {shuffled} -> {result}")

    # 4. String coercion
    total += 1
    str_nodes = [str(n) for n in sample_nodes]
    result = hypernetwork_obj.hyperedge_to_edge_id(str_nodes)
    if result == expected_edge_id:
        print(f"[Pass] String nodes {str_nodes} coerced correctly")
        passed += 1
    else:
        print(f"[Fail] String nodes {str_nodes} -> {result}")

    # 5. Nonexistent combo
    total += 1
    fake_nodes = [999999, 888888]
    result = hypernetwork_obj.hyperedge_to_edge_id(fake_nodes)
    if result is None:
        print("[Pass] Nonexistent nodes correctly returned None")
        passed += 1
    else:
        print(f"[Fail] Nonexistent nodes -> {result}, expected None")

    # 6. Non-numeric input
    total += 1
    result = hypernetwork_obj.hyperedge_to_edge_id(["a", "b"])
    if result is None:
        print("[Pass] Non-numeric input correctly returned None")
        passed += 1
    else:
        print(f"[Fail] Non-numeric input -> {result}, expected None")

    # 7. Known ambiguity: a node list equal to a valid edge id
    total += 1
    if isinstance(test_edge_id, (list, tuple)):
        result = hypernetwork_obj.hyperedge_to_edge_id(test_edge_id)
        print(f"[Info] Collision case: passing edge-id-shaped nodes {test_edge_id} -> {result}")
    else:
        print(f"[Info] Edge ids are scalar ({type(test_edge_id).__name__}); "
              f"collision with node-list input not directly testable here, "
              f"but note edge ids ({existing_edge_ids}) and node ids overlap in range — "
              f"flag as a design risk.")

    # 8. Scalar edge id that fails membership check should not silently
    #    fall into the char-iteration bug
    total += 1
    fake_edge_id = "not_a_real_edge"
    result = hypernetwork_obj.hyperedge_to_edge_id(fake_edge_id)
    if result is None:
        print("[Pass] Unknown scalar edge id correctly returned None (not via crash)")
        passed += 1
    else:
        print(f"[Fail] Unknown scalar edge id -> {result}")
    # 9. List input must not crash on the edge-id membership check
    total += 1
    try:
        result = hypernetwork_obj.hyperedge_to_edge_id(list(sample_nodes))
        if result == expected_edge_id:
            print(f"[Pass] List input handled without crashing -> {expected_edge_id}")
            passed += 1
        else:
            print(f"[Fail] List input returned {result}, expected {expected_edge_id}")
    except TypeError as e:
        print(f"[Fail] List input crashed the function: {e}")

    # 10. Set input (also unhashable-adjacent — set of ints is hashable
    #     as an object only if frozenset; a plain set itself is unhashable
    #     too, so this exercises the same guard)
    total += 1
    try:
        result = hypernetwork_obj.hyperedge_to_edge_id(set(sample_nodes))
        if result == expected_edge_id:
            print(f"[Pass] Set input handled without crashing -> {expected_edge_id}")
            passed += 1
        else:
            print(f"[Fail] Set input returned {result}, expected {expected_edge_id}")
    except TypeError as e:
        print(f"[Fail] Set input crashed the function: {e}")

    print(f"\n{passed}/{total} deterministic tests passed.")

    return passed, total
     
def build_test_mapping():
    """Constructs an xgi Hypergraph + a minimal mapping object exposing
    itertative_H, node_to_edge_id_map, hyperedge_to_edge_id(), and remove_hyperedge()."""

    H = xgi.Hypergraph()
    H.add_edge([1, 2, 3])   # edge id 0
    H.add_edge([2, 4])      # edge id 1
    H.add_edge([1, 5, 6])   # edge id 2

    node_to_edge_id_map = {
        tuple(sorted(members)): edge_id
        for edge_id, members in H.edges.members(dtype=dict).items()
    }

    class Mapping:
        def __init__(self, H, node_to_edge_id_map):
            self.itertative_H = H
            self.node_to_edge_id_map = node_to_edge_id_map

        def hyperedge_to_edge_id(self, nodes):
            # Case 1: already a valid edge id
            try:
                if nodes in self.itertative_H.edges:
                    return nodes
            except TypeError:
                pass

            # Case 2: comma-separated string, e.g. "7,12,57,76" or "1, 2, 3"
            if isinstance(nodes, str):
                try:
                    int_nodes = [int(n.strip()) for n in nodes.split(',')]
                except ValueError:
                    return None
                sorted_nodes = tuple(sorted(int_nodes))
                return self.node_to_edge_id_map.get(sorted_nodes)

            # Case 3: collection of node ids
            if not isinstance(nodes, (list, tuple, set, frozenset)):
                return None

            try:
                int_nodes = [int(n) for n in nodes]
            except (ValueError, TypeError):
                return None

            sorted_nodes = tuple(sorted(int_nodes))
            return self.node_to_edge_id_map.get(sorted_nodes)

        def remove_hyperedge(self, hyperedge_nodes):
            hyperedge = self.hyperedge_to_edge_id(hyperedge_nodes)
            if hyperedge is not None:
                print(f"Removing hyperedge {hyperedge} containing nodes {hyperedge_nodes}")
                self.itertative_H.remove_edge(hyperedge)
            else:
                print(f"[Warning] Could not find a hyperedge with exactly these nodes: {hyperedge_nodes}")

    return Mapping(H, node_to_edge_id_map)


def run_remove_hyperedge_tests():
    """
    Tests remove_hyperedge() end-to-end, including the comma-string
    input format that caused the original production failure.

    Builds and rebuilds its own fixture internally, since removals
    mutate state and each destructive test needs a clean graph.
    """
    print("Starting remove_hyperedge() Tests...\n")
    passed = 0
    total = 0

    # --- Test 1: comma string, no spaces ---
    total += 1
    hypernetwork_obj = build_test_mapping()
    sample_nodes, expected_edge_id = next(
        (nodes, eid) for nodes, eid in hypernetwork_obj.node_to_edge_id_map.items()
        if len(nodes) >= 3
    )
    edges_before = set(hypernetwork_obj.itertative_H.edges)
    comma_str = ",".join(str(n) for n in sample_nodes)
    hypernetwork_obj.remove_hyperedge(comma_str)
    edges_after = set(hypernetwork_obj.itertative_H.edges)
    removed = edges_before - edges_after
    if removed == {expected_edge_id}:
        print(f"[Pass] Comma string '{comma_str}' removed edge {expected_edge_id}")
        passed += 1
    else:
        print(f"[Fail] Comma string '{comma_str}' removed {removed}, expected {{{expected_edge_id}}}")

    # --- Test 2: comma string with spaces ---
    total += 1
    hypernetwork_obj = build_test_mapping()
    sample_nodes, expected_edge_id = next(
        (nodes, eid) for nodes, eid in hypernetwork_obj.node_to_edge_id_map.items()
        if len(nodes) >= 3
    )
    edges_before = set(hypernetwork_obj.itertative_H.edges)
    spaced_str = ", ".join(str(n) for n in sample_nodes)
    hypernetwork_obj.remove_hyperedge(spaced_str)
    edges_after = set(hypernetwork_obj.itertative_H.edges)
    removed = edges_before - edges_after
    if removed == {expected_edge_id}:
        print(f"[Pass] Spaced string '{spaced_str}' removed edge {expected_edge_id}")
        passed += 1
    else:
        print(f"[Fail] Spaced string '{spaced_str}' removed {removed}, expected {{{expected_edge_id}}}")

    # --- Test 3: direct edge id (no regression on Case 1) ---
    total += 1
    hypernetwork_obj = build_test_mapping()
    real_edge_id = next(iter(hypernetwork_obj.itertative_H.edges))
    edges_before = set(hypernetwork_obj.itertative_H.edges)
    hypernetwork_obj.remove_hyperedge(real_edge_id)
    edges_after = set(hypernetwork_obj.itertative_H.edges)
    removed = edges_before - edges_after
    if removed == {real_edge_id}:
        print(f"[Pass] Direct edge id {real_edge_id} removed correctly")
        passed += 1
    else:
        print(f"[Fail] Direct edge id {real_edge_id} removed {removed}")

    # --- Test 4: list input (no regression on Case 3) ---
    total += 1
    hypernetwork_obj = build_test_mapping()
    sample_nodes, expected_edge_id = next(iter(hypernetwork_obj.node_to_edge_id_map.items()))
    edges_before = set(hypernetwork_obj.itertative_H.edges)
    hypernetwork_obj.remove_hyperedge(list(sample_nodes))
    edges_after = set(hypernetwork_obj.itertative_H.edges)
    removed = edges_before - edges_after
    if removed == {expected_edge_id}:
        print(f"[Pass] List input removed edge {expected_edge_id}")
        passed += 1
    else:
        print(f"[Fail] List input removed {removed}, expected {{{expected_edge_id}}}")

    # --- Test 5: nonexistent comma string -> no removal ---
    total += 1
    hypernetwork_obj = build_test_mapping()
    edges_before = set(hypernetwork_obj.itertative_H.edges)
    fake_str = "9001,9002,9003"
    hypernetwork_obj.remove_hyperedge(fake_str)
    edges_after = set(hypernetwork_obj.itertative_H.edges)
    if edges_before == edges_after:
        print(f"[Pass] Nonexistent nodes '{fake_str}' left graph unchanged")
        passed += 1
    else:
        print(f"[Fail] Nonexistent nodes '{fake_str}' unexpectedly removed {edges_before - edges_after}")

    # --- Test 6: malformed comma string -> no crash, no removal ---
    total += 1
    hypernetwork_obj = build_test_mapping()
    try:
        edges_before = set(hypernetwork_obj.itertative_H.edges)
        hypernetwork_obj.remove_hyperedge("7,abc,12")
        edges_after = set(hypernetwork_obj.itertative_H.edges)
        if edges_before == edges_after:
            print("[Pass] Malformed string handled gracefully, no removal")
            passed += 1
        else:
            print(f"[Fail] Malformed string caused unexpected removal: {edges_before - edges_after}")
    except Exception as e:
        print(f"[Fail] Malformed string crashed remove_hyperedge: {e}")

    print(f"\n{passed}/{total} remove_hyperedge tests passed.")
    return passed, total

# ---------------------------------------------------------
# Example Execution Trigger
# ---------------------------------------------------------
if __name__ == "__main__":
    # Assuming mapping is instantiated like this:
    config_path = "config/config.yaml"
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
    #run_mapping_tests(network=network_in, hypernetwork=hypernetwork_in, mapping=my_mapping)
    #run_hyperedge_to_edge_id_tests(hypernetwork_obj)
    run_remove_hyperedge_tests()
