#inside the data object both the network and the hypernetwork are processed
from src2.hypernetwork_class import HypernetworkObject
from abc import ABC, abstractmethod
import heapq
import numpy as np

class DataObject(ABC):
    def __init__(self,hypernetwork_location,network_location,hyp_key_file):
        self.construct_network_and_hypernetwork(hypernetwork_location,network_location)
        self.hyperedge_queue = PriorityQueue()
        self.labelMapping = MappingOfHyperedges(hyp_key_file)
        
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


    def initialise_curvature(self):
        '''
        Need to add that initialise curvature adds gets back all the curvatures of hyperedges
        Then adds them to queue object
        '''
        #just creates curvature objects in network
        initial_curv_vals = self.network_obj.initialise_curvature()
        #the key is the nodes within the network
        #print(self.labelMapping._to_spec2.keys())
        self.is_hyp_node_removed = dict.fromkeys(self.labelMapping._to_spec2.keys(), False)
        #NEED TO POPULATE
        self.hyperedge_queue.push_mult(initial_curv_vals)
        
    @abstractmethod
    def recalculate_curvature(self):
        pass

    @abstractmethod
    def hyperedge_removal(self):
        pass

    def assess_clustering(self, target_clusters_number):
        #self.network_obj.hyperedge_removal()
        #temp_partitions = self.hypernetwork_obj.get_partitions()
        
        temp_partitions = self.hypernetwork_obj.run_iteration(target_clusters_number, size_by="nodes")
        self.number_of_clusters = len(temp_partitions)
        temp_modularity = self.hypernetwork_obj.calculate_modularity(temp_partitions)
        self.review_new_modularity(temp_partitions, temp_modularity)   
        
   
    def review_new_modularity(self,current_partition,current_modularity):
        self.modularity_iters.append(current_modularity)
        if current_modularity > self.best_modularity:
                self.best_modularity = current_modularity
                self.best_partition = current_partition
                #print(f"number of partitions", len(current_partition))
                #print(current_modularity)
        
    @abstractmethod
    def construct_network_and_hypernetwork():
        '''
        Need to ensure the bijection of hyperedges to network_object is understood
        '''
        pass
    
    #need to operate self.posetNetworkClass in here .. maybe in frDataObject


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


class PriorityQueue:
    def __init__(self):
        # Initialize an empty list to serve as the heap
        self._queue = []

    def push(self, score, node):
        """Pushes a new (score, node) tuple onto the heap."""
        heapq.heappush(self._queue, (score, node))

    def push_mult(self, pairs):
        """Pushes an array of (score, node) pairs onto the heap."""
        for score, node in pairs:
            heapq.heappush(self._queue, (score, node))

    def extract_lowest_score(self):
        """Pops and returns the (score, node) tuple with the lowest score."""
        # Note: Your algorithm checks if it's empty before calling this, 
        # so we don't need to handle IndexError here unless you want to be extra safe.
        return heapq.heappop(self._queue)

    def is_empty(self):
        """Returns True if the queue is empty, False otherwise."""
        return len(self._queue) == 0
    