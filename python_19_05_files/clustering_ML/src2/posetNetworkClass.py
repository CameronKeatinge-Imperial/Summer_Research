import numpy as np
import itertools as it
import networkx as nx
from src2.indep_functions import initialise_curvatures_forman_ricci, edge_forman_ricci, node_forman_ricci, n_step_greater_than_k_neighbourhood_nodes, n_step_neighbourhood_nodes, n_step_neighbourhood_nodes_from_edge

class PosetNetworkObject():
    '''
    Need to change is_removed from considering every node to know only some are relevant
    
    #for edges, need mapping from edges to integers
    #for nodes the same, means nodes can be NON-INTEGER values
    WHEN NOT affecting/manipulating an array, use node, otherwise use node_mapping
    '''
    def __init__(self,paths):
        '''
        Reason for having network_is_node_removed is for faster lookup
        Need to check whether network_object nodes are all integers
        '''
        self.initialPoset, self.hyperedge_nodes = self.network_from_files(paths)
        self.last_node_removed = None
        self.network_is_node_removed = np.zeros(self.initialPoset.number_of_nodes(), dtype=bool)
        #all nodes not in hyperedge_nodes should NEVER be removed
        self.init_edge_hashmap(self.initialPoset)
        self.init_node_hashmap(self.initialPoset)
        #so to check, use self.network_is_node_removed[node_hashmap(node)]

        self.iterative_G = self.initialPoset
        self.extra_model_parameters()

    def extra_model_parameters(self):
        '''
        Find justifications in FR_subclass file
        '''
        self.last_node_removed = None
        #defined by number of connections from the chosen node that the topology is changed
        #CHANGE TOPOLOGY
        self.distance_topology_change = 1
        self.radius_for_edge_curv_calc_from_edge = 1 #so neighbourhood of incident nodes
        #RECALCULATE CURVATURES
        self.distance_edge_curv_change = self.distance_topology_change + self.radius_for_edge_curv_calc_from_edge
        #USE TO RECALCULATE CURVATURES
        self.edge_curv_change_radius = self.distance_edge_curv_change + self.radius_for_edge_curv_calc_from_edge
        self.subgraph_update_radius = self.edge_curv_change_radius
        self.radius_for_node_summation = 1
        self.radius_for_node_update = self.distance_edge_curv_change + self.radius_for_node_summation

    def initialise_curvature(self):
        '''
        Initialses for 
        '''
        self.node_curvature, self.edge_curvature = initialise_curvatures_forman_ricci(self.initialPoset, self.edge_hashmap, self.node_hashmap)
        
        return self.nodes_curv_to_queue_pairs(self.initialPoset.nodes)

    def remove_node_and_adj_edges(self,node):
        '''
        Need to think about index matching and similar
        '''
        self.last_node_removed = node
        print(f"self.network_obj.last_node_removed", node)

        topology_change_neighbourhood = n_step_neighbourhood_nodes(self.iterative_G,node,self.distance_topology_change)
        #in this case radius of size 1
        self.network_is_node_removed[self.node_hashmap[node]] = True
        '''
        for node_p in topology_change_neighbourhood:
            if self.network_is_node_removed[self.node_hashmap[node_p]] == False:
                #remove node -> node_p edge
                self.iterative_G.remove_edge(node, node_p)
        '''
        edges_to_remove = [
            (node, node_p) for node_p in topology_change_neighbourhood
            if not self.network_is_node_removed[self.node_hashmap[node_p]]
        ]
        self.iterative_G.remove_edges_from(edges_to_remove)

    def update_neighbourhood_scores(self, node):
        '''
        Calculates updated scores for the neighbourhood and returns them
        as a list of (score, node) tuples to be queued by the caller.
        '''
        # all the nearby nodes
        # can make this more efficient in one function and reducing computation
        '''
        CLAUDE CHANGES
        MAYBE I ALSO NEED TO CHANGE SUCH FOR EDGES, DOESN'T WORK BY SUBNETWORK BUT WORKS OVER THE WHOLE NETOWKR
        AS I THINK IT MIGHT BE QUITE OPTIMISED ANYWAY
        subgraph_update_full = n_step_neighbourhood_nodes(self.iterative_G, node, self.subgraph_update_radius)
        subgraph_update_edges = n_step_neighbourhood_nodes(self.iterative_G, node, self.distance_edge_curv_change)
        subgraph_update_nodes = n_step_neighbourhood_nodes(self.iterative_G, node, self.radius_for_node_update)

        local_subgraph = self.iterative_G.subgraph(subgraph_update_full)
        
        for u, v in local_subgraph.edges:
            if u in subgraph_update_edges and v in subgraph_update_edges:
                edge_q = (u, v)
                # self.current_curvature_edge[self.edge_to_index[edge_q]] = edge_forman_ricci(local_subgraph,u,v)

                smaller_subgraph = self.iterative_G.subgraph(self.radius_for_edge_curv_calc_from_edge)
                self.current_curvature_edge[self.edge_hashmap[edge_q]] = edge_forman_ricci(smaller_subgraph, u, v)
        '''
        # subgraph_update_nodes uses the same radius as subgraph_update_full — no need to recompute
        subgraph_update_full = n_step_neighbourhood_nodes(self.iterative_G, node, self.subgraph_update_radius)
        subgraph_update_edges = n_step_neighbourhood_nodes(self.iterative_G, node, self.distance_edge_curv_change)
        subgraph_update_nodes = subgraph_update_full

        local_subgraph = self.iterative_G.subgraph(subgraph_update_full)

        for u, v in local_subgraph.edges:
            if u in subgraph_update_edges and v in subgraph_update_edges:
                edge_q = (u, v)
                edge_neighbourhood = (
                    n_step_neighbourhood_nodes(self.iterative_G, u, self.radius_for_edge_curv_calc_from_edge)
                    | n_step_neighbourhood_nodes(self.iterative_G, v, self.radius_for_edge_curv_calc_from_edge)
                )
                smaller_subgraph = self.iterative_G.subgraph(edge_neighbourhood)
                self.current_curvature_edge[self.edge_hashmap[edge_q]] = edge_forman_ricci(smaller_subgraph, u, v)
        # Initialize the list that will hold the new scores for the orchestrator
        nodes_to_queue = []

        for node_q in subgraph_update_nodes:
            if self.network_is_node_removed[self.node_hashmap[node_q]]:
                continue

            self.current_curvature_node[self.node_hashmap[node_q]] = node_forman_ricci(
                self.iterative_G, node_q, self.current_curvature_edge, self.edge_to_index
            )

            # smaller_subgraph = self.iterative_G.subgraph(self.radius_for_node_summation)
            # self.current_curvature_node[self.node_to_index[node_q]] = node_forman_ricci(smaller_subgraph,node_q,self.current_curvature_edge,self.edge_to_index)

            # The is_removed == False check is no longer needed here since it was caught by the continue above
            if node_q in self.hyperedge_nodes:
                new_score = self.current_curvature_node[self.node_hashmap[node_q]]
                # Store the tuple instead of pushing directly to the queue
                nodes_to_queue.append((new_score, node_q))

        # Pass the values back up to the caller
        return nodes_to_queue

    def nodes_curv_to_queue_pairs(self,set_of_nodes):
        nodes_to_queue = []
        for node_q in set_of_nodes:
            if node_q in self.hyperedge_nodes:
                new_score = self.node_curvature[self.node_hashmap[node_q]]
                # Store the tuple instead of pushing directly to the queue
                nodes_to_queue.append((new_score, node_q))
        return nodes_to_queue
        
    def init_edge_hashmap(self,network):
        '''
        Should probably fix for storage reasons
        '''
        self.edge_hashmap = {}
        for idx, (u, v) in enumerate(network.edges()):
            self.edge_hashmap[(u, v)] = idx
            #tnis does double the size of the dictionary; edges can be forced to be ordered correctly using tuple(sorted((u, v)))
            self.edge_hashmap[(v, u)] = idx

    def init_node_hashmap(self,network):
        self.node_hashmap = {}         # Maps node -> index
        self.reverse_node_hashmap = {} # Maps index -> node
        
        for idx, n in enumerate(network.nodes()):
            self.node_hashmap[n] = idx
            self.reverse_node_hashmap[idx] = n

    def network_from_files(self, paths_to_read, p = None):
            # Unpack the paths in the exact order they were appended in files_for_network
            nodes_p, edges_p, triangles_p, cardinality_p = paths_to_read
                
            # Your specific file reading logic under the hood:
            # network_nodes = open(nodes_p).read()...
            cardinality_greater_1_nodes = set()

            with open(nodes_p, 'r') as f_node, open(cardinality_p, 'r') as f_card:
                # 1. Generators ignore completely blank lines
                nodes = (int(line.strip()) for line in f_node if line.strip())
                cardinalities = (int(line.strip()) for line in f_card if line.strip())
                
                # 2. Pair them up line-by-line
                # Pro-Tip: If you use Python 3.10+, change this to zip(nodes, cardinalities, strict=True)
                for node_name, card_value in zip(nodes, cardinalities):
                    try:
                        if int(card_value) > 1:
                            cardinality_greater_1_nodes.add(node_name)
                    except ValueError:
                        # 3. Safeguard: Prevents crashing if there's a header row like "Degree" or a corrupted string
                        print(f"Warning: Skipping invalid cardinality value '{card_value}' for node '{node_name}'")

            #now construct the graph
            G = nx.Graph()
            # 1. Stream edges into the graph
            # (line.split() handles both spaces and tabs automatically)
            with open(edges_p, 'r') as f:
                edge_generator = ([int(x) for x in line.split()] for line in f if line.strip())
                G.add_edges_from(e for e in edge_generator if len(e) >= 2)
                
            # 2. Stream nodes to ensure isolated nodes (nodes with no edges) are included
            with open(nodes_p, 'r') as f:
                node_generator = (int(line.strip()) for line in f if line.strip())
                G.add_nodes_from(node_generator)
            
            with open(triangles_p, 'r') as f:
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
            return G, cardinality_greater_1_nodes

    def get_network_curvature(self):
        self.node_curvature, self.edge_curvature = initialise_curvatures_forman_ricci(self.initialPoset, self.edge_hashmap, self.node_hashmap)
        #so return a dictionary of the node number and curvature, then use the key file to match them.
        node_list = list(self.hyperedge_nodes)
        node_indices = [self.node_hashmap[n] for n in node_list]

        hyperedge_curvature_dict = dict(zip(node_list, self.node_curvature[node_indices]))
        return hyperedge_curvature_dict

    def hypernetwork_nodes_curv(self):
        '''
        Use the aggregation of the hyperedge's curvature
        Just sum the curvature of any node that it is incident on
        '''
        agg_curv_dict = {}
        
        # Iterate through all nodes in the NetworkX graph
        for n in self.initialPoset.nodes():
            
            if n not in self.hyperedge_nodes:
                agg_node_curv = 0
                
                # Get the neighborhood of node n using NetworkX
                neighbors = self.initialPoset.neighbors(n)
                
                # Sum the curvature of each node in the neighborhood
                for neighbour in neighbors:
                    # Retrieve the curvature attribute (defaults to 0 if missing)
                    agg_node_curv += self.node_curvature[self.node_hashmap[neighbour]]
                
                # Assign the aggregated sum to the dictionary
                agg_curv_dict[n] = agg_node_curv
                
        return agg_curv_dict