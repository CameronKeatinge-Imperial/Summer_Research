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

from src2.registry_measures import get_model_type
from src2.network_processing_obj import NetworkProcessor
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
        return self.my_data.change_hyperedge_keys(input_dict)
    
    def save_dataframes(self,curv_type,measure_name,edge_dict,node_dict=None):
        #for edges
        print(edge_dict)
        file_name = curv_type + measure_name
        self.configNavigator.save_dict_to_file(edge_dict,'hyperedge',file_name)
        #for nodes
        print(node_dict)
        if not node_dict == None:
            self.configNavigator.save_dict_to_file(node_dict,'node',file_name)