#comments:
#needs to handle infinity in the cost matrix for disconnected parts of the graphs

import numpy as np
import ot
from scipy.sparse.csgraph import shortest_path

def find_continuous_barycenter(adj_matrix, target_nodes, reg=0.1):
    V = len(adj_matrix)
    adj_array = np.array(adj_matrix)
    
    cost_matrix = shortest_path(
        csgraph=adj_array,
        directed=False,
        unweighted=True
    )

    if np.isinf(cost_matrix).any():
        raise ValueError("The graph must be connected.")
    
    A = np.zeros((V, len(target_nodes)))
    for i, target in enumerate(target_nodes):
        A[target, i] = 1.0
        
    barycenter_distribution = ot.bregman.barycenter(A, cost_matrix, reg)
    

    individual_distances = ot.sinkhorn2(barycenter_distribution, A, cost_matrix, reg)
    
    total_w_distance = np.sum(individual_distances)

    return barycenter_distribution, individual_distances, total_w_distance

if __name__ == "__main__":
    # Example unweighted adjacency matrix
    adjacency_matrix = [
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
    ]

    targets = [0, 1]

    reg_val = 0.1
    barycenter, distances, total_sum, direct = find_continuous_barycenter(
            adjacency_matrix, targets, reg=reg_val
        )
    
    print(f"Target Nodes: {targets}\n")
    print("Continuous Barycenter Distribution:")
    print("-" * 35)
    for node_id, mass in enumerate(barycenter):
        print(f"Node {node_id}: {mass:.4f}")
        
    for target_node, distance in zip(targets, distances):
        print(f"Distance from Barycenter to Target Node {target_node}: {distance:.4f}")

    print(f"Total sum of W_1 distance {total_sum:.4f}")
