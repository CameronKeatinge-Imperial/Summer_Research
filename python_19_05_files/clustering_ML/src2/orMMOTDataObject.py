from src2.data_object import DataObject
from src2.hypernetwork_MMOT import MMOTHypernetworkObject
from src2.posetNetworkClass import PosetNetworkObject

class OllivierRicciMMOTDataObject(DataObject):
    def __init__(self,hypernetwork_location,network_location):
        self.hypernetwork_obj = MMOTHypernetworkObject(hypernetwork_location)
        #this I just need the hypernetwork object?
        # THIS OBJECT SHOULD ONLY EVER HAVE SET OF NODES VISIBILITY (NOT NODE NAMES)
        self.queue_id_to_hyperedge_bijection()

    def queue_id_to_hyperedge_bijection(self):
        self.edge_label_to_nodes = self.hypernetwork_obj.return_edge_dict()

    def construct_network_and_hypernetwork(self,hypernetwork_files):
        pass

    def initialise_curvature(self):
        #just creates curvature objects in network
        initial_curv_vals = self.hypernetwork_obj.initialise_curvature()
        #should return list of edge_id & curvature
        self.hyperedge_queue.push_mult(initial_curv_vals)
        
    def recalculate_curvature(self):
        '''
        maybe this should be self.network_obj.values_to_add
        '''
        if (self.network_obj.last_node_removed != None):
            self.new_queue_entries = self.network_obj.update_neighbourhood_scores(self.network_obj.last_node_removed)
            self.hyperedge_queue.push_mult(self.new_queue_entries)
        else:
            print("error")
            

    def next_hyperedge_removal(self):
        '''
        Purpose: find hyperedge with the lowest curvature and remove
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


    def hyperedge_removal(self):
        #get the lowest value
        hyperedge_to_remove = self.next_hyperedge_removal()
        if hyperedge_to_remove != None :
            #passing in edge_id
            nodes_of_hyperedge = list[self.edge_label_to_nodes[hyperedge_to_remove]]
            self.hypernetwork_obj.remove_hyperedge(nodes_of_hyperedge)
            return False
        else:
            return True

    def return_init_curvature(self):
        # hypernetwork_obj -- these functions need written
        hyperedge_curv_dict = self.hypernetwork_obj.get_network_curvature()
        nodes_curv_dict = self.hypernetwork_obj.hypernetwork_nodes_curv()
        return hyperedge_curv_dict, nodes_curv_dict