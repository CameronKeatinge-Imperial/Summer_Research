## forman ricci file
import numpy as np
import os
import networkx as nx
import itertools as it
from pathlib import Path
from src.iterative_clustering import StatisticalModelTemplate
from src.create_poset_network import process_and_save_poset

class FormanRicciClustering(StatisticalModelTemplate):
    
    def __init__(self):
        super().__init__()
        self.network_decomposition = "poset_complex"
    
    def construct_network(self,data_source,dataset_name):
        '''
        need to draw from config
        '''
        #from the hypernetwork file
        #can add if need the functionality?
        #get file from outside
        #THIS IS CREATING DUAL!
        SCRIPT_DIR = Path(__file__).resolve().parent
        BASE_DATA_DIR = SCRIPT_DIR.parent / "data"
        data_source = os.path.join(BASE_DATA_DIR, data_source)
        process_and_save_poset(data_source,dataset_name)

    def files_for_network(self,source,name):
        needed_info = ["nodes","edges","triangles","cardinality"]
        paths_search = []
        base_dir = Path("data")
        for n in needed_info:
            n_path_string = os.path.join(base_dir, source, self.network_decomposition, n, f"{name}.txt")
            paths_search.append(n_path_string)
        return paths_search
    
    def network_from_files(self, paths_to_read, p = None):
        # Unpack the paths in the exact order they were appended in files_for_network
        nodes_p, edges_p, triangles_p, cardinality_p = paths_to_read
            
        # Your specific file reading logic under the hood:
        # network_nodes = open(nodes_p).read()...
        cardinality_1_nodes = set()

        with open(nodes_p, 'r') as f_node, open(cardinality_p, 'r') as f_card:
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
        with open(edges_p, 'r') as f:
            edge_generator = (line.split() for line in f if line.strip())
            G.add_edges_from(e for e in edge_generator if len(e) >= 2)
            
        # 2. Stream nodes to ensure isolated nodes (nodes with no edges) are included
        with open(nodes_p, 'r') as f:
            node_generator = (line.strip() for line in f if line.strip())
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
        self.initialNetwork = G
        self.hyperedge_nodes = cardinality_1_nodes
            #return G, cardinality_1_nodes
        
    def extra_model_parameters(self):
        self.last_node_removed = None
        #defined by number of connections from the chosen node that the topology is changed
        #CHANGE TOPOLOGY
        self.distance_topology_change = 1
        #number of nodes, if the incident nodes are zero steps
        self.radius_for_edge_curv_calc_from_edge = 1 #so neighbourhood of incident nodes

        #as the change affects all nodes within 2 steps
        #RECALCULATE CURVATURES
        self.distance_edge_curv_change = self.distance_topology_change + self.radius_for_edge_curv_calc_from_edge
        #from changed node, all the nodes that need included for recalcing all these edge's curvature
        #USE TO RECALCULATE CURVATURES
        self.edge_curv_change_radius = self.distance_edge_curv_change + self.radius_for_edge_curv_calc_from_edge
        self.subgraph_update_radius = self.edge_curv_change_radius
        
        '''
        #this is unnecessary extra variable
        self.radius_for_node_summation = 1 #weighted sum of edges.
        self.radius_for_node_update = self.distance_edge_curv_change + self.radius_for_node_summation
        '''
        self.radius_for_node_summation = 1 #weighted sum of edges.
        self.radius_for_node_update = self.distance_edge_curv_change + self.radius_for_node_summation

        #now can do:
        #get radius_for_update subgraph around node
        #for each edge within edge_curv_change_radius:
        #   create subgraph of size radius_for_edge_curv_calc_from_edge
        #   each subgraph corresponds to an edge, so calculate curv and assign new curvature to variable
        #for all nodes, within subgraph_update_radius -1
        #   recalculate node curvature summation

    def initialise_curvature(self):
        self.current_curvature_node, self.current_curvature_edge = initialise_curvatures_forman_ricci(self.initialNetwork,self.edge_to_index,self.node_to_index)
        self.is_removed = np.zeros(self.initialNetwork.number_of_nodes(), dtype=bool)
        self.initialise_queue(self.hyperedge_nodes)
        #should I use factory method here?
        
    def recalculate_curvature(self):
        #last_node_removal
        # FUNCTION 3: recalculate the curvatures of each edge and node; again mediated by a subnetwork change
        self.update_neighbourhood_scores(self.last_node_removed)
        

    def prune_neighbourhoods(self,node):
        '''
        notes
        '''
        topology_change_neighbourhood = self.n_step_neighbourhood_nodes(self.iterative_G,node,self.distance_topology_change)
        #in this case radius of size 1
        self.is_removed[self.node_to_index[node]] = True
        for node_p in topology_change_neighbourhood:
            if self.is_removed[self.node_to_index[node_p]] == False:
                #remove node -> node_p edge
                self.iterative_G.remove_edge(node, node_p)

    
    def update_neighbourhood_scores(self,node):
        '''
        can this be generalised, maybe wait until other methods written
        '''
        #all the nearby nodes
        
        #can make this more efficient in one function and reducing computation
        subgraph_update_full = self.n_step_neighbourhood_nodes(self.iterative_G,node,self.subgraph_update_radius)
        subgraph_update_edges = self.n_step_neighbourhood_nodes(self.iterative_G,node,self.distance_edge_curv_change)
        subgraph_update_nodes = self.n_step_neighbourhood_nodes(self.iterative_G,node,self.radius_for_node_update)

        local_subgraph = self.iterative_G.subgraph(subgraph_update_full)
        for u, v in local_subgraph.edges:
            if u in subgraph_update_edges and v in subgraph_update_edges:
                edge_q = (u, v)
                #self.current_curvature_edge[self.edge_to_index[edge_q]] = edge_forman_ricci(local_subgraph,u,v)

                smaller_subgraph = self.iterative_G.subgraph(self.radius_for_edge_curv_calc_from_edge)
                self.current_curvature_edge[self.edge_to_index[edge_q]] = edge_forman_ricci(smaller_subgraph,u,v)

        for node_q in subgraph_update_nodes:
            if self.is_removed[self.node_to_index[node_q]] == True:
                continue

            self.current_curvature_node[self.node_to_index[node_q]] = node_forman_ricci(self.iterative_G,node_q,self.current_curvature_edge,self.edge_to_index)

            #smaller_subgraph = self.iterative_G.subgraph(self.radius_for_node_summation)
            #self.current_curvature_node[self.node_to_index[node_q]] = node_forman_ricci(smaller_subgraph,node_q,self.current_curvature_edge,self.edge_to_index)


            if node_q in self.hyperedge_nodes and self.is_removed[self.node_to_index[node_q]] == False:
                new_score = self.current_curvature_node[self.node_to_index[node_q]]
                #push (new_score, node_q) onto self.node_queue
                self.node_queue.push(new_score,node_q)

    def hyperedge_removal(self):
        #print("Forman Ricci: remove hyperedge")
        self.last_node_removed = self.next_node_removal()  #maybe check the output form of node_removed
        if self.last_node_removed is None:
            return False
        self.prune_neighbourhoods(self.last_node_removed)
        
    def next_node_removal(self):
        '''
        Purpose: find hyperedge with the lowest curvature and remove
        
        Method: uses Queue()
        #note: this node removal is with respect to the poset complex, so actually represents a hyperedge
        '''
        node = None  # Safeguard if queue empties without finding a valid node
        while True:
            if self.node_queue.is_empty():
                break 
            score, node = self.node_queue.extract_lowest_score()  
            if self.is_removed[self.node_to_index[node]] == False and score == self.current_curvature_node[self.node_to_index[node]]:
                break 
        return node
    
    def map_hyperedge_for_removal(self):
        pass
        

        













########################
# FROM data_processing #
########################

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
