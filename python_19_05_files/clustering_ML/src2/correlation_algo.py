# algorithm
# essentially parallel version of clustering, but returning just output file
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
import os

from src2.data_object import DataObject
from src2.frDataObject import FormanRicciDataObject
from src2.registry_measures import get_model_type
from src.create_dual_networks import process_and_save_dual_complexes
from src.create_poset_network import process_and_save_poset

class measuresClass():
    def __init__(self):
         #all the configurations here
        self.name = "A"

    def performing_analysis(self,config_obj) -> Any:
            print("Analysing data")
            
            #this object can now be referenced
            self.configNavigator = NetworkProcessor(config_obj)
            data_obj = get_model_type(config_obj['model']['curvature_form'])
            my_data = data_obj(self.configNavigator.files_for_hypernetwork(),self.configNavigator.files_for_network(),self.configNavigator.hyperedge_key_file())
            #once the data object is built, just go straight to some calculation
            self.my_data = my_data
            return my_data.return_init_curvature()

    def change_hyperedge_keys(self,input_dict):
        new_dict = {
            self.my_data.labelMapping.node_to_hyperedge_map(old_key): value 
            for old_key, value in input_dict.items()
        }
        return new_dict


    def save_dataframes(self,measure_name,edge_dict,node_dict=None,):
        #for edges
        self.configNavigator.save_dict_to_file(edge_dict,'hyperedge',measure_name)
        #for nodes
        if not node_dict == None:
            self.configNavigator.save_dict_to_file(node_dict,'node',measure_name)



class NetworkProcessor:
    '''
    1. Add validation that paths exist
    2. If paths do not exist, trigger create files
    '''
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
        self.initial_check_hypernetwork_files()

    def initial_check_hypernetwork_files(self):
        hyperedge_path = os.path.join(self.base_dir, self.source, "hypernetwork_form", "edges", f"{self.dataset_name}.txt")
        hypernet_nodes_path = os.path.join(self.base_dir, self.source, "hypernetwork_form", "nodes", f"{self.dataset_name}.txt")
        paths_search_h = []
        paths_search_h.append(hyperedge_path)
        paths_search_h.append(hypernet_nodes_path)
        if (self.validate_files_populated(paths_search_h)==True):
            print("Hypernetwork paths already exist")
        else:
            if (self.validate_files_populated([hyperedge_path])==False):
                print("No hypernetwork hyperedge file")
            else:
                self.make_nodes_file_from_hyperedges(hyperedge_path)

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
        if (self.validate_files_populated(paths_search_h)==True):
            print("Hypernetwork paths exist (for reading)")
        return paths_search_h
    
    def forman_ricci_files(self):
        needed_info=["nodes", "edges", "triangles", "cardinality"]
        paths_search= []
        for n in needed_info:
            n_path_string = os.path.join(self.base_dir, self.source, self.network_decomposition, n, f"{self.dataset_name}.txt")
            paths_search.append(n_path_string)
        if (self.validate_files_populated(paths_search)==True):
            print("Forman Ricci file paths exist")
        else:
            self.base_dir = Path("data")
            source_path = os.path.join(self.base_dir, self.source)
            process_and_save_poset(source_path,self.dataset_name)
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

        if (self.validate_files_populated(paths_search)==True):
            print("Ollivier Ricci Dual file paths exist")
        else:
            self.base_dir = Path("data")
            source_path = os.path.join(self.base_dir, self.source)
            process_and_save_dual_complexes(source_path,self.dataset_name)
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

    def validate_files_populated(self, testing_paths):
        '''
        Binary output of whether all the files exist and are non-empty
        '''
        #read the nodes and edges
        #specific_file_within
        verified_paths = []
        for path_str in self.flatten_paths(testing_paths):
        #for path_str in chain.from_iterable(paths):
            path = Path(path_str)
            print(f"Checking file: {path.absolute()}")

            if not path.is_file():
                print(f"  [Error] Missing required file: {path_str}")
                self.existsNetworkFile = False
                print("A required files NOT verified.")

                return False # Stop processing immediately since the batch is incomplete

            if path.stat().st_size == 0:
                print(f"  [Error] File is empty: {path_str}")
                self.existsNetworkFile = False
                print("A required files NOT verified.")

                return False # Stop processing immediately since the batch has an empty file

            verified_paths.append(str(path))

        # 2. All files exist and are non-empty, proceed to execution
        print("All required files verified. Processing...")
        return True
    
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
    
    def make_nodes_file_from_hyperedges(self,file_loc):
        '''
        Read a file of comma-separated numbers (one group per line),
        find the largest number in it, then write a new file containing
        1 through that max number, one per line.
        '''
        print("Trying to write into hyperedge node file")
        max_num = 0
        with open(file_loc) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                nums = [int(x) for x in line.split(',')]
                max_num = max(max_num, max(nums))
        hypernet_nodes_path = os.path.join(self.base_dir, self.source, "hypernetwork_form", "nodes", f"{self.dataset_name}.txt")
        with open(hypernet_nodes_path, 'w') as f:
            for i in range(1, max_num + 1):
                f.write(f'{i}\n')

    def save_dict_to_file(self,my_dictionary,node_or_edge: str, measure: str):
        results_dir = Path("results")

        path = os.path.join(results_dir, self.source, self.dataset_name, node_or_edge, f"{measure}.txt")

        # 1. Create all necessary parent directories if they don't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 2. Open the file and write the dictionary contents
        with open(path, 'w', encoding='utf-8') as f:
            for key, value in my_dictionary.items():
                f.write(f"{key} : {value}\n")