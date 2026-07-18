#data processing
import networkx as nx
import yaml
import itertools as it
import os
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
#def read_data(nodes,edges,input_structure):
#    if input_structure == "polyhedral_complex" or input_structure == "dual":
#        graph_output = read_standard_network(nodes,edges)
#        return graph_output

def number_of_hypernetwork_nodes(hypernetwork_node_file):
    with open(hypernetwork_node_file, "r") as f:
        return sum(1 for _ in f)

def read_hypernetwork():
    pass

def read_ph_network(path):
    # Handle NaNs, scale features, etc.
    hypernetwork_node_file = path["nodes"]
    hypernetwork_edge_file = path["edges"]
    hypernetwork_triangle_file = path["triangles"]
    hypernetwork_cardinality_file = path["cardinality"]

    ###########################
    # PART 1, HYPEREDGE NODES #
    ###########################
    # matching cardinality

    cardinality_1_nodes = set()

    with open(hypernetwork_node_file, 'r') as f_node, open(hypernetwork_cardinality_file, 'r') as f_card:
        # 1. Generators ignore completely blank lines
        nodes = (line.strip() for line in f_node if line.strip())
        cardinalities = (line.strip() for line in f_card if line.strip())
        
        # 2. Pair them up line-by-line
        # Pro-Tip: If you use Python 3.10+, change this to zip(nodes, cardinalities, strict=True)
        for node_name, card_value in zip(nodes, cardinalities):
            try:
                if int(card_value) == 1:
                    cardinality_1_nodes.add(node_name)
            except ValueError:
                # 3. Safeguard: Prevents crashing if there's a header row like "Degree" or a corrupted string
                print(f"Warning: Skipping invalid cardinality value '{card_value}' for node '{node_name}'")

    #now construct the graph
    G = nx.Graph()
    # 1. Stream edges into the graph
    # (line.split() handles both spaces and tabs automatically)
    with open(hypernetwork_edge_file, 'r') as f:
        edge_generator = (line.split() for line in f if line.strip())
        G.add_edges_from(e for e in edge_generator if len(e) >= 2)
        
    # 2. Stream nodes to ensure isolated nodes (nodes with no edges) are included
    with open(hypernetwork_node_file, 'r') as f:
        node_generator = (line.strip() for line in f if line.strip())
        G.add_nodes_from(node_generator)
    
    with open(hypernetwork_triangle_file, 'r') as f:
        for line in f:
            # 1. Parse the line safely
            temp = list(map(int, line.strip().split()))
            
            # 2. Update Node Triangle Counts safely
            for i in temp:
                # Ensure the node exists in the graph first
                if i not in G:
                    G.add_node(i)
                # Use .get('triangles', 0) to handle missing attributes safely
                G.nodes[i]['triangles'] = G.nodes[i].get('triangles', 0) + 1
                
            # 3. Update Edge Triangle Counts safely
            for i, j in it.combinations(temp, 2):
                # Ensure the edge exists, regardless of node order
                if not G.has_edge(i, j):
                    G.add_edge(i, j)
                    
                # Safely get the existing edge dictionary object
                edge_data = G.edges[i, j]
                edge_data['triangles'] = edge_data.get('triangles', 0) + 1
    return G, cardinality_1_nodes

def read_dual_network(path):
    # Handle NaNs, scale features, etc.
    hypernetwork_node_file = path["nodes"]
    hypernetwork_edge_file = path["edges"]

    G = nx.Graph()
    
    # 1. Stream edges into the graph
    # (line.split() handles both spaces and tabs automatically)
    with open(hypernetwork_edge_file, 'r') as f:
        edge_generator = (line.split() for line in f if line.strip())
        G.add_edges_from(e for e in edge_generator if len(e) >= 2)
        
    # 2. Stream nodes to ensure isolated nodes (nodes with no edges) are included
    with open(hypernetwork_node_file, 'r') as f:
        node_generator = (line.strip() for line in f if line.strip())
        G.add_nodes_from(node_generator)
    return G

def hyp_to_polyh_complex():
    #leave this to the existing files.
    #by doing this, I can do this once for each data set.
    pass

def hyp_to_bipartite_graph():
    pass

def read_true_labels(data_source,dataset_name):
    """
    Reads a file containing line-separated community/cluster labels 
    and returns them as a list of integers.
    """
    #SCRIPT_DIR = Path(__file__).resolve().parent
    #not needed as imported into a file
    #BASE_DATA_DIR = SCRIPT_DIR.parent / "data"
    #data_source = os.path.join(BASE_DATA_DIR, data_source)
    file_path = os.path.join("data", data_source, "true_clusters", f"{dataset_name}.txt")
    with open(file_path, 'r') as file:
        labels = [int(line.strip()) for line in file if line.strip()]
    return labels