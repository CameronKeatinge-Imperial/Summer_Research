#need to figure out how to do modularity from different forms of networks
#note it needs to be done at each stage of the algorithm

#could do with hyperedge node key + partitions knowledge
#can tranform each hyperedge given the classification of the nodes within
import hypernetx.algorithms.hypergraph_modularity as hmod

#USE SMALLER DATASET
def calculate_modularity_ext(hypergraph_object, current_partitions, modularity_param):
    '''
    hypernetwork is HyperNetX object
    options of wdc=strict,majority,linear - make this a config val
    '''
    partition_for_modularity = network_clusters_to_hypernetwork(hypergraph_object, current_partitions)
    # just checking for any nodes that aren't part of the hypernetwork, as they are isolated nodes

    param_mapping = {
        'linear': hmod.linear,
        'majority': hmod.majority,
        'strict': hmod.strict
    }
    
    # NOTE: You are overwriting the function argument `modularity_param` here.
    # You might want to remove the next line if you want to pass 'linear' or 'strict' dynamically!
    
    chosen_wdc_function = param_mapping[modularity_param]
    
    modularity_score = hmod.modularity(hypergraph_object, partition_for_modularity, wdc=chosen_wdc_function)
    return modularity_score


def network_clusters_to_hypernetwork(hypergraph_object1, network_partitions):
    # use in clusters -> modularity

    # 2. Convert List of Sets to a Dictionary {node: community_id}
    # This gives you the format needed for dictionary-based filtering
    partition_dict = {}
    for cid, community in enumerate(network_partitions):
        for node in community:
            partition_dict[node] = cid

    # 3. Apply your safe-filtering code on the dictionary
    # (Ensures only nodes actually present in your hypergraph object are included)
    filtered_dict = {
        node: cid for node, cid in partition_dict.items() 
        if node in hypergraph_object1
    }

    # 4. Convert the filtered dictionary BACK to HyperNetX's required list of sets
    # We gather the unique community IDs found in our filtered dictionary
    unique_cids = set(filtered_dict.values())

    filtered_partition_list = []
    for cid in unique_cids:
        # Gather all nodes that belong to this specific community ID
        community_set = {node for node, c_id in filtered_dict.items() if c_id == cid}
        filtered_partition_list.append(community_set)
        
    # FIX: Return the reconstructed list of sets
    return filtered_partition_list


from sklearn.metrics import normalized_mutual_info_score

def normalised_mutual_information_calc(algo_assignments,true_assignments):
    '''
    expected form
    labels_true = [0, 0, 1, 1, 2, 2]
    labels_pred = [0, 0, 1, 2, 2, 2]
    '''
    print("Calculating NMI")
    print("algo_assignments")
    print(algo_assignments)
    print("true_assignments")
    print(true_assignments)
    #algo_assignments = partition_to_labels_temp(algo_assignments)
    nmi_score = normalized_mutual_info_score(algo_assignments, true_assignments)
    print(f"NMI Score: {nmi_score}") 
    # 1.0 is a perfect match, 0.0 means no mutual information

def partition_to_labels(nodes, partition_list_of_sets):
    """Converts a list of sets into a 1D array of community labels based on a fixed node order."""
    #######
    # FIX #
    #######
    
    label_dict = {}
    for cid, community in enumerate(partition_list_of_sets):
        for node in community:
            label_dict[node] = cid
            
    # Map back to the exact node list order to ensure alignment
    return [label_dict[node] for node in nodes]


def adj_rand_calc():
    pass

    def n_step_neighbourhood_nodes(self,graph,source_node,k):
        lengths = nx.single_source_shortest_path_length(graph, source=source_node, cutoff=k)
        neigbbourhood_set = set(lengths.keys())
        return neigbbourhood_set
    
    def n_step_neighbourhood_nodes_from_edge(self,graph,source_node1,source_node2,k):
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