##algorithm
from pathlib import Path
import re

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
#from src.registry_local import get_file_reading_type
from src2.data_object import DataObject
from src2.frDataObject import FormanRicciDataObject
# This acts as both your High-Level Method and your "Interface"
class algorithmClass():
    
    def __init__(self,target_num_clusters=5,max_iterations=500,modularity_equ='linear'):
         #all the configurations here
         self.modularity_equ = modularity_equ
         self.target_num_clusters = target_num_clusters
         self.max_iter = max_iterations
         self.maximum_clusters = self.target_num_clusters * 4 # can change the 4 to a paramet

    def perform_algorithm(self,config_obj) -> Any:
            print("Starting training process...")
            
            #this object can now be referenced
            configNavigator = NetworkProcessor(config_obj)
            #create data object

            ##########################################
            # THIS NEEDS TO BE GENERALISED TO CONFIG #
            ##########################################
            my_data = FormanRicciDataObject(configNavigator.files_for_hypernetwork(),configNavigator.files_for_network(),configNavigator.hyperedge_key_file())
            #my_data.model_parameters
            terminate_condition = False
            for i in range(self.max_iter):
                if my_data.number_of_clusters < self.maximum_clusters:
                    print(f"iteration ", i)
                    print(f"number_of_clusters ", my_data.number_of_clusters)
                    if (i == 0):
                        my_data.initialise_curvature()  # initialise curvature of network
                    else:
                        my_data.recalculate_curvature()
                    terminate_condition = my_data.hyperedge_removal()
                    if terminate_condition == True:
                        break
                    my_data.assess_clustering(self.target_num_clusters)
                
            print(f"Best modularity:", my_data.best_modularity)
            print(f"Best partition:", my_data.best_partition)
            return my_data.best_partition
    

class NetworkProcessor:
    def __init__(self, config):
        # Load the configuration file once during initialization
        self.config = config
        self.curvature_discretisation = self.config["model"]["curvature_form"]
        self.source = self.config["data"]["data_source_type"]
        self.dataset_name = self.config["data"]["hypernetwork_name"]
        self.base_dir = Path("data")
        #####################
        # NEEDS GENERALISED #
        #####################
        self.network_decomposition = "poset_complex"

    def files_for_network(self) -> list[str]:
        '''
        NEED TO GENERALISE
        '''
        # 1. Safely extract settings from config with fallbacks/defaults
        #return get_file_reading_type()
        return self.forman_ricci_files()
        
    
    def files_for_hypernetwork(self) -> list[str]:            
        hyperedge_path = os.path.join(self.base_dir, self.source, "hypernetwork_form", "edges", f"{self.dataset_name}.txt")
        hypernet_nodes_path = os.path.join(self.base_dir, self.source, "hypernetwork_form", "nodes", f"{self.dataset_name}.txt")
        paths_search_h= []
        paths_search_h.append(hyperedge_path)
        paths_search_h.append(hypernet_nodes_path)
        return paths_search_h
    
    def forman_ricci_files(self):
        needed_info=["nodes", "edges", "triangles", "cardinality"]
        paths_search= []
        for n in needed_info:
            n_path_string = os.path.join(self.base_dir, self.source, self.network_decomposition, n, f"{self.dataset_name}.txt")
            paths_search.append(n_path_string)
        return paths_search
    
    def orc_dual_files(self):
        needed_info = ["nodes","edges"]
        paths_search = []
        #but now need to get the networks of all the networks of different cardinality
        file_of_nodes = os.path.join(self.base_dir, self.source, self.network_decomposition, "nodes", self.dataset_name)
        self.hyperedge_cardinalities = self.extract_cardinalities_from_files(file_of_nodes)

        for c in self.hyperedge_cardinalities:
            #add these as a sublist/array, so now 2d
            cardinality_pairs = []
            for n in needed_info:
                n_path_string = os.path.join(self.base_dir, self.source, self.network_decomposition, n, self.dataset_name, f"{n}_k{c}.txt")
                cardinality_pairs.append(n_path_string)
            paths_search.append(cardinality_pairs)
        return paths_search
    
        
    def extract_cardinalities_from_files(self, folder_path):
        """
        read all the files from this file location, but from their names in the form nodes_k{number}.txt
        """
        cardinalities = []
        
        # Define a regex pattern: 'nodes_k' followed by one or more digits (\d+), ending in '.txt'
        # The parenthesis () create a capture group for just the digits
        pattern = re.compile(r'^nodes_k(\d+)\.txt$')
        
        try:
            # List all files in the given directory
            for filename in os.listdir(folder_path):
                match = pattern.match(filename)
                if match:
                    # Extract the captured number string and convert it to an int
                    number = int(match.group(1))
                    cardinalities.append(number)
        except FileNotFoundError:
            print(f"Error: The folder '{folder_path}' does not exist.")
            return []

        # Return the numbers sorted for easier processing later
        return sorted(cardinalities)
    
    def hyperedge_key_file(self):
        '''
        needed_info = ["hyperedge_node_key"]
        paths_search= []
        for n in needed_info:
            n_path_string = os.path.join(self.base_dir, self.source, n, f"{self.dataset_name}.txt")
            paths_search.append(n_path_string)
        return paths_search
        '''
        return os.path.join(self.base_dir, self.source, self.network_decomposition, "hyperedge_node_key", f"{self.dataset_name}.txt")
