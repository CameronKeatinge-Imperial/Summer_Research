#inside the data object both the network and the hypernetwork are processed
from src2.hypernetwork_class import HypernetworkObject
from abc import ABC, abstractmethod
import heapq
import numpy as np

class DataObject(ABC):
    def __init__(self,hypernetwork_location,network_location,hyp_key_file):
        self.construct_network_and_hypernetwork(hypernetwork_location,network_location)
        self.modularity_iters = []   # Tracks modularity over iterations
        self.best_partition = []
        self.best_modularity = -0.5 #lowest theoretical val
        self.number_of_clusters = 1
        '''
        NEED CONFIGS
        self.modularity_equation = modularity_form
        self.target_num_clusters = target_no_clusters
        self.max_iter = target_no_clusters * max_iter_multiple
        '''

    @abstractmethod
    def initialise_curvature(self):
        pass
        
    @abstractmethod
    def recalculate_curvature(self):
        pass

    @abstractmethod
    def hyperedge_removal(self):
        pass

    def assess_clustering(self, optimal_cluster_number, greedy_clusters_number):
        #self.network_obj.hyperedge_removal()
        #temp_partitions = self.hypernetwork_obj.get_partitions()
        
        temp_partitions = self.hypernetwork_obj.run_iteration(greedy_clusters_number, size_by="nodes")
        temp_partitions = self.hypernetwork_obj.optimal_attach_clusters(temp_partitions,optimal_cluster_number)
        self.number_of_clusters = len(temp_partitions)
        temp_modularity = self.hypernetwork_obj.calculate_modularity(temp_partitions)
        self.review_new_modularity(temp_partitions, temp_modularity)   
        
   
    def review_new_modularity(self,current_partition,current_modularity):
        self.modularity_iters.append(current_modularity)
        if current_modularity > self.best_modularity:
            self.best_modularity = current_modularity
            self.best_partition = current_partition
        
    @abstractmethod
    def construct_network_and_hypernetwork():
        '''
        Need to ensure the bijection of hyperedges to network_object is understood
        '''
        pass

    @abstractmethod
    def return_init_curvature():
        pass
    #need to operate self.posetNetworkClass in here .. maybe in frDataObject

    def cluster_size_terminate(self, target_cluster_no):
        '''
        Conditions for termination
        '''
        if target_cluster_no < 2:
            raise ValueError("target_cluster_no must be at least 2")

        clusters = self.hypernetwork_obj.get_partitions()
        partition = [set(c) for c in clusters]

        # Descending, so "largest" is [0] and "n largest" is [:n]
        sizes = sorted(
            (self.hypernetwork_obj._cluster_size(c, "nodes") for c in clusters),
            reverse=True,
        )

        total = sum(sizes)
        if total == 0:
            raise ValueError("total cluster size is zero")
        frac = [s / total for s in sizes]

        if frac[0] < 1 / (target_cluster_no - 1):
            print(f"Terminating due to largest cluster less than size 1 / ", (target_cluster_no - 1))
            return True
        if sum(frac[:(target_cluster_no+1)]) < (target_cluster_no - 1) / (target_cluster_no):
            print(f"Terminating due to ",(target_cluster_no+1), "largest clusters less than size ", (target_cluster_no - 1), " / ", target_cluster_no)
            return True
        return False

#translating between hyperedges and objects in the network decomposition?
class MappingOfHyperedges:
    '''
    Of the mapping file, the first column will be the node number.
    The second will be the full hyperedge.
    ''' 
    def __init__(self, mapping_file_path):
        # Two dictionaries for O(1) lookups in both directions
        self._to_spec1 = {}
        self._to_spec2 = {}
        self._load_bijection(mapping_file_path)

    def _load_bijection(self, filepath):
        # Assuming a simple CSV or text file where each line is "label1,label2"
        with open(filepath, 'r') as f:
            for line in f:
                l1, l2 = line.strip().split(':')
                # Store the bijection both ways
                l1 = int(l1.strip())
                l2 = l2.strip() # This removes the leading space from the comma string
                self._to_spec1[l2] = l1
                self._to_spec2[l1] = l2

    def hyperedge_to_node_map(self, label2):
        return self._to_spec1[label2]

    def node_to_hyperedge_map(self, label1):
        return self._to_spec2[label1]

    def get_node_to_hyp_map(self):
        return self._to_spec2