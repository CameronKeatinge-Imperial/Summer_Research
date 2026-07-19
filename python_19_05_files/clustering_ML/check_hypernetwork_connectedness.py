import networkx as nx
from src2.posetNetworkClass import PosetNetworkObject
from src2.hypernetwork_class import HypernetworkObject
from src2.data_object import MappingOfHyperedges
from src2.algorithm_class import NetworkProcessor
from src.data_processing import load_config
import hypernetx as hnx
import xgi
# HyperNetX allows you to project to a standard bipartite or line graph
def test_edge_removal(H):
    """
    Removes one hyperedge from the hypernetwork and verifies removal.

    Args:
        H: hypernetwork object

    Returns:
        Modified hypernetwork
    """

    # Get current hyperedges
    edges_before = list(H.edges())

    if len(edges_before) == 0:
        print("No edges available to remove.")
        return H

    # Select first hyperedge
    edge_to_remove = edges_before[0]

    print(f"Removing edge: {edge_to_remove}")

    # Remove edge
    H = H.remove_edges(edge_to_remove)

    # Check remaining edges
    edges_after = list(H.edges())

    if edge_to_remove in edges_after:
        print("ERROR: Edge was not removed!")
    else:
        print("SUCCESS: Edge removed correctly.")

    print(f"Edges before: {len(edges_before)}")
    print(f"Edges after: {len(edges_after)}")

    return H


def test_sequential_component_tracking(H):
    """
    Sequentially removes edges one by one from the hypergraph and tracks 
    how the number of connected components changes.

    Args:
        H: hypergraph object (xgi.Hypergraph)

    Returns:
        H_test: The modified (empty) hypergraph.
        component_history: A list of integers representing the number of 
                           components after each edge removal.
    """
    print("\n========== SEQUENTIAL REMOVAL TEST ==========")

    # Create a copy so we don't destroy the original hypergraph
    H_test = H.copy()
    
    edges_to_remove = list(H_test.edges)
    total_edges = len(edges_to_remove)
    
    print(f"Starting nodes: {H_test.num_nodes}")
    print(f"Starting edges: {total_edges}")

    if total_edges == 0:
        print("No edges to remove.")
        return H_test, []

    # Get baseline components before any removal
    try:
        baseline_components = len(list(xgi.connected_components(H_test)))
        print(f"Initial connected components: {baseline_components}\n")
    except Exception as e:
        print("Initial connected_components failed:", e)
        return H_test, []

    component_history = []

    # --------------------------------------------------
    # SEQUENTIAL REMOVAL LOOP
    # --------------------------------------------------
    print("Beginning sequential removal...")
    
    for i, edge in enumerate(edges_to_remove, start=1):
        # 1. Remove the edge
        H_test.remove_edge(edge)
        
        # 2. Calculate the new number of components
        try:
            current_components = len(list(xgi.connected_components(H_test)))
            component_history.append(current_components)
            
            # Print update (you might want to comment this out for very large networks)
            print(f"Removed edge '{edge}' ({i}/{total_edges}) | Components: {current_components}")
            
        except Exception as e:
            print(f"Failed to calculate components after removing '{edge}': {e}")
            break

    # --------------------------------------------------
    # FINAL VERIFICATION
    # --------------------------------------------------
    print("\n========== TEST COMPLETE ==========")
    print(f"Final edges remaining: {H_test.num_edges}")
    print(f"Final isolated nodes remaining: {H_test.num_nodes}")
    
    if component_history:
        print(f"Components grew from {baseline_components} -> {component_history[-1]}")
        # Note: In a fully disconnected XGI graph, the number of components 
        # should exactly equal the number of nodes!
        if component_history[-1] == H_test.num_nodes:
            print("SUCCESS: Final component count perfectly matches total isolated nodes.")

    return H_test, component_history

def test_hyperedge_removal(H):
    """
    Full diagnostic test for hyperedge removal and connected components in XGI.

    Args:
        H: hypergraph object (xgi.Hypergraph)

    Returns:
        Modified hypergraph after removing all edges.
    """

    print("\n========== INITIAL STATE ==========")
    
    # XGI uses properties (.edges, .nodes) rather than methods
    edges_before = list(H.edges)
    nodes_before = list(H.nodes)

    print(f"Number of edges: {len(edges_before)}")
    print(f"Number of nodes: {len(nodes_before)}")

    if len(edges_before) == 0:
        print("No edges to test.")
        return H


    # --------------------------------------------------
    # TEST 1: Remove a single hyperedge
    # --------------------------------------------------

    print("\n========== SINGLE EDGE REMOVAL ==========")

    edge_to_remove = edges_before[0]
    print("Removing:", edge_to_remove)

    # Create a copy so we don't destroy the original hypergraph passed into the test
    H_test = H.copy()
    
    # XGI modifies in-place using remove_edge() for a single ID
    H_test.remove_edge(edge_to_remove)

    edges_after = list(H_test.edges)

    print(f"Edges before: {len(edges_before)}")
    print(f"Edges after: {len(edges_after)}")

    if edge_to_remove not in edges_after:
        print("PASS: Edge removed correctly")
    else:
        print("FAIL: Edge still exists")

    # Check components after single removal
    print("\nConnected components after removing one edge:")

    try:
        # In XGI, algorithms are called from the xgi namespace
        components = list(xgi.connected_components(H_test))
        print(f"Number of components: {len(components)}")
        for c in components:
            print(c)
    except Exception as e:
        print("connected_components failed:")
        print(e)


    # --------------------------------------------------
    # TEST 2: Remove all hyperedges
    # --------------------------------------------------

    print("\n========== REMOVE ALL EDGES ==========")

    H_empty = H_test
    remaining_edges = list(H_empty.edges)

    print(f"Removing {len(remaining_edges)} remaining edges")

    # XGI provides remove_edges_from() for batch removal in-place
    H_empty.remove_edges_from(remaining_edges)

    final_edges = list(H_empty.edges)
    final_nodes = list(H_empty.nodes)

    print("\nAfter removing all edges:")
    print("Edges:", final_edges)
    # Unlike HNX, XGI keeps isolated nodes. This will match the initial node count.
    print(f"Nodes: {len(final_nodes)} isolated nodes remain.")


    # --------------------------------------------------
    # TEST 3: Empty graph connectivity
    # --------------------------------------------------

    print("\n========== FINAL CONNECTIVITY TEST ==========")

    if len(final_nodes) == 0:
        print("Hypergraph is completely empty (no nodes).")
    else:
        try:
            # This will now successfully return a component for every isolated node
            components = list(xgi.connected_components(H_empty))
            print(f"Connected components: {len(components)}")
            # Commenting out the print loop here so it doesn't spam the console 
            # if you have thousands of isolated nodes.
            # for c in components:
            #     print(c)

        except Exception as e:
            print("connected_components failed:")
            print(e)

    print("\n========== TEST COMPLETE ==========")

    return H_empty

if __name__ == "__main__":
    config_path = "config/config.yaml"
    config = load_config(config_path)
    paths_obj = NetworkProcessor(config)

    paths_obj.files_for_hypernetwork()

    hypernetwork_obj = HypernetworkObject(paths_obj.files_for_hypernetwork())
    #bipartite_graph = hypernetwork_obj.initialHypernetwork.bipartite()
    #is_connected = nx.is_connected(bipartite_graph)
    #print(f"Is the hypernetwork fully connected? {is_connected}")

    test_hyperedge_removal(hypernetwork_obj.initialHypernetwork)
    test_sequential_component_tracking(hypernetwork_obj.initialHypernetwork)
