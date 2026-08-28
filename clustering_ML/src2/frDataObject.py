from src2.data_object import DataObject
from src2.hypernetwork_class import HypernetworkObject
from src2.posetNetworkClass import PosetNetworkObject
from src2.data_object import MappingOfHyperedges
from src2.indep_functions import cardinality_distribution
from src2.multi_queue_object import MultiPriorityQueue         
import numpy as np

class FormanRicciDataObject(DataObject):
    '''
    In this object, generally use network form as shorter
    The exception is for the Queue, use network nodes (+ full hyperedge is quite verbose)
    '''
    def __init__(self,hypernetwork_location,network_location,hyp_key_file,target_dist):
        super().__init__(hypernetwork_location,network_location,hyp_key_file)
        self.labelMapping = MappingOfHyperedges(hyp_key_file)
        #so Queue, labelMapping object, self.is_hyp_node_removed exists
        #this is obviously from the posetObject
        if target_dist == "None":
            self.hyperedge_queue = MultiPriorityQueue() 
            self.maximum_cardinality = 1
        else:
            self.hyperedge_queue = MultiPriorityQueue(target_dist) 
            self.maximum_cardinality = len(target_dist)
            #so there exists a PriorityQueue for every cardinality that is non-zero in the target distribution
        self.new_queue_entries = None

    def construct_network_and_hypernetwork(self,hypernetwork_files,network_files):
        '''
        Need to ensure the bijection of hyperedges to network_object is understood.
        So this is specific for FRC and posets
        GOAL OF: define self.network_obj and self.hypernetwork_obj

        With both, just read in the files and allow nodes to have non-integer names
        '''
        self.hypernetwork_obj = HypernetworkObject(hypernetwork_files)
        self.number_of_nodes = self.hypernetwork_obj.number_of_nodes()
        print(f"Number of nodes", self.number_of_nodes)
        #this defines self.hypernetwork_obj
        self.network_obj = PosetNetworkObject(network_files)

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
        self.hyperedge_queue.push_mult(initial_curv_vals,self.labelMapping.get_node_to_hyp_map())
        
    def recalculate_curvature(self):
        '''
        maybe this should be self.network_obj.values_to_add
        '''
        if (self.network_obj.last_node_removed != None):
            self.new_queue_entries = self.network_obj.update_neighbourhood_scores(self.network_obj.last_node_removed)
            self.hyperedge_queue.push_mult(self.new_queue_entries,self.labelMapping.get_node_to_hyp_map())
        else:
            print("error")
            
        #now map the nodes to hyperedges

    def hyperedge_removal(self,target_dist1):
        #get the lowest value
        current_distribution = cardinality_distribution(self.hypernetwork_obj.itertative_H,self.maximum_cardinality)
        hyperedge_node_for_removal = self.next_hyperedge_removal(target_dist1,current_distribution)
        if hyperedge_node_for_removal != None :
            #remove for network and hypernetwork
            self.network_obj.remove_node_and_adj_edges(hyperedge_node_for_removal)
            self.hypernetwork_obj.remove_hyperedge(self.labelMapping.node_to_hyperedge_map(hyperedge_node_for_removal))
            return False, self.labelMapping.node_to_hyperedge_map(hyperedge_node_for_removal)
        else:
            return True, None

    def next_hyperedge_removal(self,target_distribution,hypernetwork_distribution):
        '''
        old logic
        #working from the highest cardinality
        for i in range(len(target_distribution) - 1, -1, -1):
            #print(i)
            if target_distribution[i] < hypernetwork_distribution[i]:
                #print("length ",self.hyperedge_queue.hyperedge_queues[i].is_empty())
                #if self.hyperedge_queue.hyperedge_queues[i].is_empty() == False
                if self.hyperedge_queue.is_empty_cardinality(i) == False:
                    score, node = self.hyperedge_queue.extract_lowest_score(cardinality=i)
                    break
        else:
            return None
        '''
        print(f"Target distribution", target_distribution)
        print(f"Current hypernetwork distribution", hypernetwork_distribution)
        node = None  # Safeguard if queue empties without finding a valid node
        while True:
            if self.hyperedge_queue.is_empty():
                break 
            if target_distribution == "None":
                score, node = self.hyperedge_queue.extract_lowest_score()
            else:
                candidates = []  # list A: (score, node, cardinality)
                for i in range(len(target_distribution)):
                    if target_distribution[i]==0:
                        continue
                    if self.hyperedge_queue.is_empty_cardinality(i):
                        continue
                    score, node = self.hyperedge_queue.peek_lowest_score(cardinality=i)
                    candidates.append((score, node, i))

                candidates.sort(key=lambda entry: entry[0], reverse=True)

                for score, node, i in candidates:
                    # 2.1 only remove if this cardinality is over-represented
                    if target_distribution[i] < hypernetwork_distribution[i]:
                        # 2.2 valid -> actually pop it
                        score, node = self.hyperedge_queue.extract_lowest_score(cardinality=i)
                        break
                    # 2.3 otherwise fall through to the next entry in the list
                else:
                    return None

            #print(f"Curvature of hyperedge", score)
            #second condition is just a check
            if self.is_hyp_node_removed[node] == False and score == self.network_obj.node_curvature[self.network_obj.node_hashmap[node]]:
                break 
        self.is_hyp_node_removed[node] = True
        return node

    def return_init_curvature(self):
        hyperedge_curv_dict = self.network_obj.get_network_curvature()
        nodes_curv_dict = self.network_obj.hypernetwork_nodes_curv()
        return hyperedge_curv_dict, nodes_curv_dict

    def change_hyperedge_keys(self,input_dict):
        new_dict = {
            self.labelMapping.node_to_hyperedge_map(old_key): value 
            for old_key, value in input_dict.items()
        }
        return new_dict