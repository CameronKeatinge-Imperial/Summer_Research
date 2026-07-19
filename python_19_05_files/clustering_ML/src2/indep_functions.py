import numpy as np
import networkx as nx
import hypernetx.algorithms.hypergraph_modularity as hmod

def initialise_curvatures_forman_ricci(graph,mapping_edges,mapping_nodes):
    '''
    Graph will already contain information about nodes, edges and triangles
    mapping = self.edge_to_index(())
    '''
    # returning edge and node curvatures, from current_scores_e, current_scores_n =
    node_curvature = np.zeros(graph.number_of_nodes(), dtype=int)
    edge_curvature = np.zeros(graph.number_of_edges(), dtype=int)
    
    #graph.edges[u,v]['RicE'] = temp
    #this is super useful, allows you to assign additional information to edges and nodes.
    #this allows you to aggegrate
    for u, v in graph.edges():
        # Safely get triangles, defaulting to 0 if the key doesn't exist
        triangles = graph.edges[u, v].get('triangles', 0)
        
        # Run your curvature calculation using the safe variable
        temp = 4 - graph.degree(u) - graph.degree(v) + 3 * triangles
        edge_curvature[mapping_edges[(u, v)]] = temp
    for n in graph.nodes():
        temp = 0
        for u,v in graph.edges(n):
            temp = temp + edge_curvature[mapping_edges[(u, v)]]
        node_curvature[mapping_nodes[n]] = temp
    return node_curvature, edge_curvature

def edge_forman_ricci(graph,edge_point_1,edge_point_2):
    #edge is in form (u,v)
    temp = 4 - graph.degree(edge_point_1) - graph.degree(edge_point_2) + 3*graph.edges[edge_point_1,edge_point_2]['triangles']
    return temp
   
def node_forman_ricci(graph,node,edge_curvature,mapping_edges):
    #temp = 0
    #for u,v in graph.edges(node):
    #    temp = temp + edge_curvature[mapping_edges((u, v))]
    #return(temp)    
    return sum(
    edge_curvature[mapping_edges[(u, v)]]
    for u, v in graph.edges(node)
    )

def n_step_neighbourhood_nodes(graph,source_node,k):
    lengths = nx.single_source_shortest_path_length(graph, source=source_node, cutoff=k)
    neigbbourhood_set = set(lengths.keys())
    return neigbbourhood_set

def n_step_neighbourhood_nodes_from_edge(graph,source_node1,source_node2,k):
    lengths = nx.multi_source_shortest_path_length(graph, source=[source_node1,source_node2], cutoff=k)
    neigbbourhood_set = set(lengths.keys())
    return neigbbourhood_set

def n_step_greater_than_k_neighbourhood_nodes(graph, a, b, inner_set=None, central_node=None):
    """
    Blazing fast retrieval of the outer node layer (distance a to b).
    Generates the inner_set automatically if it isn't provided.
    """
    if a >= b:
        raise ValueError("Parameter 'a' must be strictly less than 'b' (a < b)")
        
    # 1. Fallback: If no inner_set is provided, compute it using the central_node
    if inner_set is None:
        if central_node is None:
            raise ValueError("Must provide either 'inner_set' or 'central_node'.")
            
        # Wrap central_node in a list, and search out to distance 'a'
        inner_lengths = nx.multi_source_shortest_path_length(graph, sources=[central_node], cutoff=a)
        inner_set = set(inner_lengths.keys())
        
        # Optional: If you want to make absolutely sure the central node itself 
        # isn't accidentally caught in the final outer layer calculation:
        inner_set.add(central_node) 
    else:
        if not isinstance(inner_set, set):
            inner_set = set(inner_set)
    remaining_steps = b - a
    # 3. Multi-source BFS expansion from the inner core outward
    lengths = nx.multi_source_shortest_path_length(graph, sources=inner_set, cutoff=remaining_steps)
    all_expanded_nodes = set(lengths.keys())
    # 4. The outer set is the newly discovered nodes minus the inner core
    outer_set = all_expanded_nodes - inner_set
    
    return outer_set



def calculate_modularity_ext(hypergraph_object, current_partitions, modularity_param):
    '''
    IF THIS FUNCTION USES A hnx OBJECT THAT COULD BE FINE, AS USES THE SAME ONE THROUGHOUT
    hypernetwork is HyperNetX object
    options of wdc=strict,majority,linear - make this a config val
    '''
    # just checking for any nodes that aren't part of the hypernetwork, as they are isolated nodes

    param_mapping = {
        'linear': hmod.linear,
        'majority': hmod.majority,
        'strict': hmod.strict
    }
    
    # NOTE: You are overwriting the function argument `modularity_param` here.
    # You might want to remove the next line if you want to pass 'linear' or 'strict' dynamically!
    
    chosen_wdc_function = param_mapping[modularity_param]
    
    #print(f"partition", current_partitions)
    modularity_score = hmod.modularity(hypergraph_object, current_partitions, wdc=chosen_wdc_function)
    #print(f"modularity_score in external function",modularity_score)
    return modularity_score