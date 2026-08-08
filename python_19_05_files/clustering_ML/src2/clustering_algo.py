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

from src2.registry_measures import get_model_type
from src2.network_processing_obj import NetworkProcessor

class algorithmClass():
    def __init__(self,target_num_clusters=2,max_iterations=500,modularity_equ='linear'):
         #all the configurations here
         self.modularity_equ = modularity_equ
         self.target_num_clusters = target_num_clusters
         self.max_iter = max_iterations
         self.maximum_clusters = self.target_num_clusters * 4 # can change the 4 to a paramet

    def perform_algorithm(self,config_obj) -> Any:
            print("Starting training process...")
            
            #this object can now be referenced
            self.configNavigator = NetworkProcessor(config_obj)
            #create data object
            data_obj = get_model_type(config_obj['model']['curvature_form'])
            my_data = data_obj(self.configNavigator.files_for_hypernetwork(),self.configNavigator.files_for_network(),self.configNavigator.hyperedge_key_file())
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