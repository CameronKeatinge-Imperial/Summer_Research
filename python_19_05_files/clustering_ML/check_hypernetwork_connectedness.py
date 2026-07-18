import networkx as nx
from src2.posetNetworkClass import PosetNetworkObject
from src2.hypernetwork_class import HypernetworkObject
from src2.data_object import MappingOfHyperedges
from src2.algorithm_class import NetworkProcessor
from src.data_processing import load_config
import hypernetx as hnx
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

def test_remove_and_partition(H):
    """
    Diagnostic test for hyperedge removal and connected components.

    Args:
        H: hypergraph object

    Returns:
        Modified hypergraph after removing one edge
    """

    print("\n===== INITIAL STATE =====")

    edges_before = list(H.edges())
    nodes_before = list(H.nodes())

    print(f"Number of edges: {len(edges_before)}")
    print(f"Edges: {edges_before}")
    print(f"Number of nodes: {len(nodes_before)}")

    components_before = list(H.connected_components())
    print(f"Connected components before removal ({len(components_before)}):")
    for c in components_before:
        print(c)


    if len(edges_before) == 0:
        print("No edges found. Cannot test removal.")
        return H


    # Pick an edge to remove
    edge_to_remove = edges_before[0]

    print("\n===== REMOVING EDGE =====")
    print(f"Removing: {edge_to_remove}")


    # Perform removal
    H_after = H.remove_edges(edge_to_remove)


    print("\n===== AFTER REMOVAL =====")

    edges_after = list(H_after.edges())

    print(f"Number of edges: {len(edges_after)}")
    print(f"Edges: {edges_after}")


    # Check if removal worked
    if edge_to_remove in edges_after:
        print("❌ FAIL: Edge still exists after removal")
    else:
        print("✅ PASS: Edge removed successfully")


    # Compare edge counts
    if len(edges_after) == len(edges_before) - 1:
        print("✅ PASS: Edge count decreased by exactly one")
    else:
        print(
            f"⚠️ WARNING: Expected {len(edges_before)-1} edges, "
            f"got {len(edges_after)}"
        )


    # Check connectivity
    print("\n===== CONNECTED COMPONENTS AFTER REMOVAL =====")

    components_after = list(H_after.connected_components())

    print(
        f"Connected components after removal ({len(components_after)}):"
    )

    for c in components_after:
        print(c)


    # Check if components changed
    if components_before == components_after:
        print("⚠️ WARNING: Components did not change")
    else:
        print("✅ Components changed after removal")


    return H_after

def test_remove_all_edges(H):
    """
    Removes every hyperedge and checks behaviour.
    """

    print("Initial number of edges:", len(list(H.edges())))
    print("Initial number of nodes:", len(list(H.nodes())))

    # Remove all edges
    for e in list(H.edges()):
        H = H.remove_edges([e])

    print("\nAfter removal:")
    print("Edges:", list(H.edges()))
    print("Nodes:", list(H.nodes()))

    # Try connected components
    try:
        components = list(H.connected_components())

        print("Number of components:", len(components))
        for c in components:
            print(c)

    except Exception as error:
        print("connected_components failed:")
        print(error)

        print("\nLikely cause:")
        print("Hypergraph has nodes but no edges.")

    return H

def test_hyperedge_removal(H):
    """
    Full diagnostic test for hyperedge removal and connected components.

    Args:
        H: hypergraph object (e.g. xgi.Hypergraph)

    Returns:
        Modified hypergraph after removing all edges.
    """

    print("\n========== INITIAL STATE ==========")

    edges_before = list(H.edges())
    nodes_before = list(H.nodes())

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

    H_test = H.remove_edges([edge_to_remove])

    edges_after = list(H_test.edges())

    print(f"Edges before: {len(edges_before)}")
    print(f"Edges after: {len(edges_after)}")


    if edge_to_remove not in edges_after:
        print("PASS: Edge removed correctly")
    else:
        print("FAIL: Edge still exists")


    # Check components after single removal
    print("\nConnected components after removing one edge:")

    try:
        components = list(H_test.connected_components())

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

    remaining_edges = list(H_empty.edges())

    print(f"Removing {len(remaining_edges)} remaining edges")

    for e in remaining_edges:
        H_empty = H_empty.remove_edges([e])


    final_edges = list(H_empty.edges())
    final_nodes = list(H_empty.nodes())

    print("\nAfter removing all edges:")
    print("Edges:", final_edges)
    print("Nodes:", final_nodes)


    # --------------------------------------------------
    # TEST 3: Empty graph connectivity
    # --------------------------------------------------

    print("\n========== FINAL CONNECTIVITY TEST ==========")

    if len(final_nodes) == 0:
        print("Hypergraph is completely empty.")
        print("connected_components cannot run on empty graph.")

    else:
        try:
            components = list(H_empty.connected_components())

            print(
                f"Connected components: {len(components)}"
            )

            for c in components:
                print(c)

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
    bipartite_graph = hypernetwork_obj.initialHypernetwork.bipartite()
    is_connected = nx.is_connected(bipartite_graph)
    print(f"Is the hypernetwork fully connected? {is_connected}")

    test_hyperedge_removal(hypernetwork_obj.initialHypernetwork)
    #print(hnx.__version__)