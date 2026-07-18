from src2.data_object import DataObject
from src2.hypernetwork_class import HypernetworkObject
from src2.posetNetworkClass import PosetNetworkObject

import numpy as np

class FormanRicciDataObject(DataObject):
    '''
    In this object, generally use network form as shorter
    The exception is for the Queue, use network nodes (+ full hyperedge is quite verbose)
    '''
    def __init__(self,hypernetwork_location,network_location,hyp_key_file):
        super().__init__(hypernetwork_location,network_location,hyp_key_file)
        #so Queue, labelMapping object, self.is_hyp_node_removed exists
        #this is obviously from the posetObject
        self.new_queue_entries = None     

    def construct_network_and_hypernetwork(self,hypernetwork_files,network_files):
        '''
        Need to ensure the bijection of hyperedges to network_object is understood.
        So this is specific for FRC and posets
        GOAL OF: define self.network_obj and self.hypernetwork_obj

        With both, just read in the files and allow nodes to have non-integer names
        '''
        self.hypernetwork_obj = HypernetworkObject(hypernetwork_files)
        #this defines self.hypernetwork_obj
        self.network_obj = PosetNetworkObject(network_files)

    def recalculate_curvature(self):
        '''
        maybe this should be self.network_obj.values_to_add
        '''
        if (self.network_obj.last_node_removed != None):
            self.new_queue_entries = self.network_obj.update_neighbourhood_scores(self.network_obj.last_node_removed)
            self.hyperedge_queue.push_mult(self.new_queue_entries)
        else:
            print("error")
            
        #now map the nodes to hyperedges

    def hyperedge_removal(self):
        #get the lowest value
        hyperedge_node_for_removal = self.next_hyperedge_removal()
        if hyperedge_node_for_removal != None :
            #remove for network and hypernetwork
            self.network_obj.remove_node_and_adj_edges(hyperedge_node_for_removal)
            self.hypernetwork_obj.remove_hyperedge(self.labelMapping.node_to_hyperedge_map(hyperedge_node_for_removal))
            return False
        else:
            return True

    def next_hyperedge_removal(self):
        '''
        Purpose: find hyperedge with the lowest curvature and remove
        
        Method: uses Queue()
        #note: this node removal is with respect to the poset complex, so actually represents a hyperedge

        Returns: the network_node to be removed
        '''
        node = None  # Safeguard if queue empties without finding a valid node
        while True:
            if self.hyperedge_queue.is_empty():
                break 
            score, node = self.hyperedge_queue.extract_lowest_score()  
            #print(f"Curvature of hyperedge", score)
            #second condition is just a check
            if self.is_hyp_node_removed[node] == False and score == self.network_obj.node_curvature[self.network_obj.node_hashmap[node]]:
                break 
        self.is_hyp_node_removed[node] = True
        return node