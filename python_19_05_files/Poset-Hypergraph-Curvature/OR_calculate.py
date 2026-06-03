#construct 5-step neighbourhood subgraphs
#first get the nodes 

import numpy as np
from scipy.sparse.csgraph import shortest_path
import sys
import ot
import pandas as pd


def subgraph_curvature_cpu(SG,central_node,bar_reg,w_reg):
    def construct_adj_matrix():
        sparse_matrix = nx.to_scipy_sparse_array(SG)
        return sparse_matrix

    def construct_cost_matrix(adjacency_matrix):
        cm = distances = shortest_path(csgraph=adjacency_matrix, directed=False, unweighted=True)
        return cm

    def node_probability_measure(adj, node):
        N = adj.shape[0]
        indptr = adj.indptr
        indices = adj.indices

        probs = np.zeros(N)

        hyperedges = indices[indptr[node]:indptr[node + 1]]

        deg_i = len(hyperedges)
        if deg_i == 0:
            return probs

        for e in hyperedges:

            neighbors = indices[indptr[e]:indptr[e + 1]]

            valid = [v for v in neighbors if v != node]

            if len(valid) == 0:
                continue

            w = 1.0 / deg_i / len(valid)

            for v in valid:
                probs[v] += w

        return probs

    local_to_global = np.array(SG.nodes())
    local_center_node = np.where(local_to_global == central_node)[0][0]
    adj_matrix = construct_adj_matrix()
    #ALL REINDEXED NOW

    #for all nodes which are one step from the node of interest
    #step 2
    indptr = adj_matrix.indptr
    indices = adj_matrix.indices
    total_local_nodes = len(indptr) - 1  # This equals 583
    num_neighbours = indptr[local_center_node + 1] - indptr[local_center_node]
    prob_measures = np.zeros((total_local_nodes,num_neighbours))
    
    #for all nodes adjacent to the hyperedge:
    counter = 0
    for n in indices[indptr[local_center_node]:indptr[local_center_node + 1]]:
        h = node_probability_measure(adj_matrix,n)
        #add to A
        prob_measures[:,counter] = h
        counter += 1

    #step 3
    cost_matrix = construct_cost_matrix(adj_matrix)    

    #OK this may be problem on graphs that massively sprawl, but should be fine for now
    #step 4
    barycenter_distribution = ot.bregman.barycenter(prob_measures, cost_matrix, bar_reg)

    #step 5
    #where A is all the probability measures of nodes surrounding the hyperedge.
    individual_distances = ot.sinkhorn2(barycenter_distribution, prob_measures, cost_matrix, w_reg)

    #step 6
    total_w_distance = np.sum(individual_distances)
    magnitude_of_hyperedge = len(indices[indptr[local_center_node]:indptr[local_center_node + 1]])
    curvature = 1 - total_w_distance / ( 2 * (magnitude_of_hyperedge-1))

    return prob_measures,total_w_distance, curvature


def subgraph_curvature_gpu(edges, target_nodes, k_hops=5, reg=0.1, n_iter=50):
    # STEP 1: Multi-source BFS
    frontier = cudf.DataFrame({
        "node": target_nodes, "source": target_nodes, 
        "depth": cp.zeros(len(target_nodes), dtype=cp.int32)
    })
    
    current_frontier = frontier
    all_frontiers = [frontier]
    for d in range(1, k_hops + 1):
        next_frontier = current_frontier.merge(edges, left_on="node", right_on="src", how="inner")
        next_frontier = next_frontier[["dst", "source"]]
        next_frontier.columns = ["node", "source"]
        next_frontier["depth"] = cp.int32(d)
        current_frontier = next_frontier.drop_duplicates(subset=["source", "node"])
        all_frontiers.append(current_frontier)

    # Combine all hops
    frontier = cudf.concat(all_frontiers).groupby(["source", "node"], as_index=False).depth.min()

    # STEP 2: Induced subgraph construction
    induced = frontier.merge(edges, left_on="node", right_on="src", how="inner")
    induced = induced.merge(frontier, left_on=["source", "dst"], right_on=["source", "node"], 
                           how="inner", suffixes=("_src", "_dst"))
    induced = induced[induced["depth_src"] <= k_hops]

    # STEP 3: Probability Measures
    deg = induced.groupby(["source", "src"]).size().reset_index(name="deg")
    mu_edges = induced.merge(deg, on=["source", "src"])
    mu_edges["mass"] = 1.0 / mu_edges["deg"]
    mu = mu_edges.groupby(["source", "dst"]).mass.sum().reset_index()

    # STEP 4: Per-source Sinkhorn calculation
    results = []
    sources = mu["source"].unique().to_cupy()

    for s in sources:
        # Isolate local subgraph data
        edges_s = induced[induced["source"] == s]
        mu_s = mu[mu["source"] == s]
        
        # Local Mapping (0 to N_local-1)
        unique_nodes = cudf.concat([edges_s["src"], edges_s["dst"]]).unique()
        local_map = cudf.DataFrame({"node": unique_nodes}).reset_index().rename(columns={"index": "local_id"})
        
        # Map edges and mu to local IDs
        edges_s = edges_s.merge(local_map, left_on="src", right_on="node").merge(local_map, left_on="dst", right_on="node", suffixes=("_src", "_dst"))
        mu_s = mu_s.merge(local_map, left_on="dst", right_on="node")
        
        # Build Tensors
        N_l = len(unique_nodes)
        a = cp.zeros(N_l); a[mu_s["local_id"].to_cupy()] = mu_s["mass"].to_cupy()
        a /= cp.sum(a)
        b = cp.ones(N_l) / N_l
        
        C = cp.zeros((N_l, N_l))
        C[edges_s["local_id_src"].to_cupy(), edges_s["local_id_dst"].to_cupy()] = edges_s["depth_src"].to_cupy()
        C = cp.minimum(C, C.T) # Symmetrize
        
        # Sinkhorn (with normalization for stability)
        C_norm = C / (k_hops + 1e-9)
        K = cp.exp(-C_norm / reg)
        u, v = cp.ones(N_l), cp.ones(N_l)
        for _ in range(n_iter):
            u = a / (K @ v + 1e-12)
            v = b / (K.T @ u + 1e-12)
        
        dist = cp.sum((cp.diag(u) @ K @ cp.diag(v)) * C)
        curvature = 1.0 - (dist / (2.0 * (N_l - 1)))
        results.append((int(s), float(curvature)))
        
    return results

#to get the 5-step neighbourhoods, the decision is whether to:
# 1. do it from each hyperedge node from the edge pairings
# 2. construct the adjacency matrix from the edges, then find the neighbourhoods

barycenter_reg = 0.3
wasserstein_reg = 0.3


#read in the data
#the list of nodes
nodes = np.loadtxt(sys.argv[1], dtype=int)
#cardinalities
cardinalities = np.loadtxt(sys.argv[3], dtype=int)

# 2. Load the edges .txt file into a Pandas DataFrame
edges_io = pd.read_csv(
    sys.argv[2], 
    sep=r'\s+', 
    names=['src', 'dst'], 
    header=None
)

def check_gpu_availability():
    """
    Dynamically checks if the RAPIDS environment and a functional 
    NVIDIA GPU/CUDA driver are available on this machine.
    """
    try:
        import cudf
        import cugraph
        # Force a tiny GPU allocation to ensure the CUDA driver is actually responsive
        _ = cudf.Series([1])
        return True
    except Exception:
        # Catches ModuleNotFoundError, Driver Missing, or CUDA Initialization errors
        print("--- GPU/CUDA environment not detected. Falling back to CPU path. ---")
        return False
    
use_gpu = check_gpu_availability()
#use_gpu = False

output_data = []
if use_gpu:
    pass
else:
    import networkx as nx
    G_cpu = nx.from_pandas_edgelist(
        edges_io,
        source='src',
        target='dst',
        create_using=nx.Graph()
    )

    target_nodes = nodes[cardinalities > 1]    

    for node in target_nodes:
        if G_cpu.has_node(node):
            local_subgraph = nx.ego_graph(G_cpu, n=node, radius=5)
            _,_, result = subgraph_curvature_cpu(local_subgraph, node, barycenter_reg, wasserstein_reg)
            print(f"Successfully extracted CPU neighborhood for node {node}.")
            output_data.append((node, result))
        else:
            print(f"Warning: Node {node} was not found in the CPU graph. Skipping.")

    with open(sys.argv[4], 'w') as out_file:
        for node, result in output_data:
            out_file.write(f"{node}\t{result}\n")