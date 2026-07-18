#model superclass

from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import heapq
import networkx as nx
from pathlib import Path
from networkx.algorithms import community
#from itertools import chain
from collections.abc import Iterable
import hypernetx as hnx
import os
import random
import hypernetx.algorithms.hypergraph_modularity as hmod

from src.modularity_funcs import calculate_modularity_ext

# This acts as both your High-Level Method and your "Interface"

class StatisticalModelTemplate(ABC):
    
    def __init__(self):
        self.existsNetworkFile = True

    def read_data(self,data_source,dataset_name):
        self.try_file_location_network(data_source,dataset_name)
        if (self.existsNetworkFile == False):
            self.construct_network(data_source,dataset_name)
            self.try_file_location_network(data_source,dataset_name)
        self.open_init_hypernetwork(data_source,dataset_name)
        #now self.initialNetwork and self.hyperedge_nodes are defined

    # 1. THE TEMPLATE METHOD
    # This dictates the overarching logic and loop structure.
    # We do not use @abstractmethod here, because we DON'T want subclasses changing this flow.
        #TO INITIALISE SAY FR, JUST CALL FormanRicciClustering()
        #model = FormanRicciClustering()
        #model.perform()
    def perform(self, modularity_equ,  max_iterations_multiple: int = 5,target_num_clusters: int = None) -> Any:
        print("Starting training process...")
        
        self.model_parameters(modularity_equ, max_iterations_multiple, target_num_clusters)

        for i in range(self.max_iter):
            if (i == 0):
                self.initialise_curvature()  # initialise curvature of network
            else:
                self.recalculate_curvature()
            self.hyperedge_removal()
            if self.hyperedge_removal() == False:
                break

            self.assess_clustering(modularity_equ)
            
        print(f"Best modularity:", self.best_modularity)
        print(f"Best partition:", self.best_partition)
        return self.best_partition

    # 2. THE REQUIRED INTERFACE STEPS
    # The @abstractmethod decorator forces subclasses to implement these.
    # The leading underscore denotes these are "protected" internal methods.
    
    def model_parameters(self,modularity_form, max_iter_multiple, target_no_clusters):
        #the network decomposition is already defined as self.initial_network
        self.iterative_G = self.initialNetwork
        self.interative_H = self.initial_hypernetwork
        self.modularity_equation = modularity_form
        self.target_num_clusters = target_no_clusters
        self.max_iter = target_no_clusters * max_iter_multiple
        self.optimal_partition = None  # Stores the final cluster assignments
        self.modularity_iters = []   # Tracks modularity over iterations
        self.node_queue = PriorityQueue()
        self.terminate_algorithm = False
        self.node_dictionary(self.initialNetwork)
        self.edge_dictionary(self.initialNetwork)

        #self.hyperedge_nodes = hyperedge_node_set -- already initialised
        self.best_partition = []
        self.best_modularity = -0.5 #lowest theoretical val
        self.extra_model_parameters()
        
    @abstractmethod
    def initialise_curvature(self):
        pass

    @abstractmethod
    def recalculate_curvature(self):
        pass

    def hyperedge_removal(self):
        hyperedge_for_removal = self.map_hyperedge_for_removal()
        self.interative_H.remove_edges(hyperedge_for_removal)
        
    @abstractmethod
    def map_hyperedge_for_removal(self):
        pass

    ####################
    # Clustering based #
    # General methods  #
    ####################
    def assess_clustering(self,modularity_equation):
        temp_partitions = self.new_partition(self.iterative_G)
        temp_modularity = self.calculate_modularity(temp_partitions)
        self.review_new_modularity(temp_partitions, temp_modularity)    

    def new_partition(self,graph):
        '''
        Partition should be configured through config
        May need to send to the network type subclass to calculate
        '''
        #for now, this definitely needs updated/changed
        return list(self.iterative_hypernetwork.connected_components())
        return list(nx.connected_components(graph))
        pass

    def calculate_modularity(self, network_partitions):
        """
        Needs to be config dependent
        """
        #return community.modularity(hypergraph, current_partitions)
        return calculate_modularity_ext(self.initial_hypernetwork, network_partitions,self.modularity_equation)
    
    def review_new_modularity(self,current_partition,current_modularity):
        if current_modularity > self.best_modularity:
                self.best_modularity = current_modularity
                self.best_partition = current_partition
                print(f"number of partitions", len(current_partition))
                print(current_modularity)
    
    def try_file_location_network(self,org_data_source,dataset_name1):
        #read the nodes and edges
        #specific_file_within
        paths = self.files_for_network(org_data_source,dataset_name1)
        verified_paths = []
        for path_str in self.flatten_paths(paths):
        #for path_str in chain.from_iterable(paths):
            path = Path(path_str)
            print(f"Checking file: {path.absolute()}")
            
            if not path.is_file():
                print(f"  [Error] Missing required file: {path_str}")
                self.existsNetworkFile = False
                return None # Stop processing immediately since the batch is incomplete
                
            verified_paths.append(str(path))
            
        # 2. All files exist, proceed to execution
        print("All required files verified. Processing...")
        return self.network_from_files(verified_paths,paths)

    def flatten_paths(self,paths_data):
        flat_list = []
        # Force single items into a loopable list if the top level isn't iterable
        if not isinstance(paths_data, Iterable) or isinstance(paths_data, (str, bytes)):
            paths_data = [paths_data]
            
        for item in paths_data:
            if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
                flat_list.extend(item) # Unpacks lists, sets, or tuples
            else:
                flat_list.append(item)
        return flat_list
    
    def open_init_hypernetwork(self, source , name):
        '''
        Commented out code would keep isolated nodes.
        Labels are preserved, so if 3 is an isolated nodes,
        the remaining nodes in hypernetwork would  be 1,2,4,5
        '''
        # 1. Read the file into a list of lists
        base_dir = Path("data")
        hyperedge_path = os.path.join(base_dir, source, self.network_decomposition, "edges", f"{name}.txt")
        hypernet_nodes_path = os.path.join(base_dir, source, self.network_decomposition, "nodes", f"{name}.txt")

        # 1. Read your reference node file
        #with open(hypernet_nodes_path, "r") as f:
        #    all_nodes = [int(line.strip()) for line in f if line.strip()]

        # 2. Your actual hyperedges data (for example purposes)
        with open(hyperedge_path, 'r') as f:
            # Split by spaces (or change to .split(',') for CSV lines)
            hyperedges = [line.strip().split() for line in f if line.strip()]

        # Convert edge data into a dictionary for full control over edge IDs
        #incidence_dict = {f"e{i}": edge for i, edge in enumerate(hyperedges)}
        #existing_nodes = set(node for edge in hyperedges for node in edge)
        #isolated_nodes = set(all_nodes) - existing_nodes
        # Add them as singleton edges
        #for node in isolated_nodes:
        #    incidence_dict[f"isolated_{node}"] = [node]
        #self.initial_hypernetwork = hnx.Hypergraph(incidence_dict)

        self.initial_hypernetwork = hnx.Hypergraph(hyperedges)
        
    @abstractmethod
    def construct_network(self,data_source,dataset_name):
        pass

    #############################
    # NOT ABSTRACT METHODS      # 
    # appilcable to all methods #
    #############################

    def init_edge_hashmap(self,network):
        self.edge_hashmap = {}
        for idx, (u, v) in enumerate(network.edges()):
            self.edge_to_index[(u, v)] = idx
            #tnis does double the size of the dictionary; edges can be forced to be ordered correctly using tuple(sorted((u, v)))
            self.edge_to_index[(v, u)] = idx
    '''
    def init_node_hashmap(self,network):
        self.node_hashmap = {}
        for idx, n in enumerate(network.nodes()):
            self.node_to_index[n] = idx
    '''
    
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
    
    def initialise_queue(self, all_nodes):
        """Populates the priority queue with all initial nodes and their scores."""
        # Ensure the queue is starting fresh
        self.node_queue = PriorityQueue() 
        
        for node in all_nodes:
            node_idx = self.node_to_index[node]            
            # 1. Ensure the node isn't marked as removed yet
            self.is_removed[node_idx] = False 
            # 2. Grab its initial score from your vector
            initial_score = self.current_curvature_node[node_idx]
            # 3. Push onto the queue
            self.node_queue.push(initial_score, node)

    def edge_dictionary(self,network):
        self.edge_to_index = {}
        for idx, (u, v) in enumerate(network.edges()):
            self.edge_to_index[(u, v)] = idx
            #tnis does double the size of the dictionary; edges can be forced to be ordered correctly using tuple(sorted((u, v)))
            self.edge_to_index[(v, u)] = idx

    def node_dictionary(self,network):
        self.node_to_index = {}
        for idx, n in enumerate(network.nodes()):
            self.node_to_index[n] = idx

class PriorityQueue:
    def __init__(self):
        # Initialize an empty list to serve as the heap
        self._queue = []

    def push(self, score, node):
        """Pushes a new (score, node) tuple onto the heap."""
        heapq.heappush(self._queue, (score, node))

    def extract_lowest_score(self):
        """Pops and returns the (score, node) tuple with the lowest score."""
        # Note: Your algorithm checks if it's empty before calling this, 
        # so we don't need to handle IndexError here unless you want to be extra safe.
        return heapq.heappop(self._queue)

    def is_empty(self):
        """Returns True if the queue is empty, False otherwise."""
        return len(self._queue) == 0
    