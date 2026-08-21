##algorithm
from typing import Any
from src2.registry_measures import get_model_type
from src2.network_processing_obj import NetworkProcessor

class algorithmClass():
    def __init__(self,modularity_equ,target_dist,max_iterations=100,target_num_clusters=None):
         #all the configurations here
         self.modularity_equ = modularity_equ
         self.target_num_clusters = target_num_clusters
         self.return_num_clusters = target_num_clusters * 2
         self.maximum_clusters = self.target_num_clusters * 20

         self.max_iter = max_iterations
         self.hyperedges_removal_ordering = []

         if target_dist == "None":
            self.target_dist = target_dist
         else:
            self.target_dist = [0, 0] + target_dist # either None or a distribution   
         

    def perform_algorithm(self,config_obj) -> Any:
        print("Starting training process...")
        
        #this object can now be referenced
        self.configNavigator = NetworkProcessor(config_obj)
        #create data object
        data_obj = get_model_type(config_obj['model']['curvature_form'])
        my_data = data_obj(self.configNavigator.files_for_hypernetwork(),self.configNavigator.files_for_network(),self.configNavigator.hyperedge_key_file(),self.target_dist)
        #my_data.model_parameters
        terminate_condition = False
        for i in range(self.max_iter):
            # if my_data.number_of_clusters < self.maximum_clusters:
            print(f"iteration ", i)
            print(f"number_of_clusters ", my_data.number_of_clusters)
            if (i == 0):
                my_data.initialise_curvature()  # initialise curvature of network
            else:
                my_data.recalculate_curvature()
            terminate_condition, removed_hyperedge = my_data.hyperedge_removal(self.target_dist)
            self.hyperedges_removal_ordering.append(removed_hyperedge)
            if terminate_condition == True or my_data.cluster_size_terminate(self.target_num_clusters):
                break
            #can drop this one line back if I know modularity is increasing function as algo works its magic
        my_data.assess_clustering(self.target_num_clusters,self.return_num_clusters)
            
        print(f"Best modularity:", my_data.best_modularity)
        print(f"Best partition:", my_data.best_partition)
        # print(f"Order of removal: ", self.hyperedges_removal_ordering)
        return my_data.best_partition, self.hyperedges_removal_ordering