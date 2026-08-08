from src2.data_object import DataObject
from src2.hypernetwork_MMOT import MMOTHypernetworkObject
#
class OllivierRicciMMOTDataObject(DataObject):
    def __init__(self,hypernetwork_location,network_location,extra_file):
        #network_location , extra_file are obselete
        super().__init__(hypernetwork_location,network_location,extra_file)
        #this I just need the hypernetwork object?
        # THIS OBJECT SHOULD ONLY EVER HAVE SET OF NODES VISIBILITY (NOT NODE NAMES)
        self.queue_id_to_hyperedge_bijection()

    def queue_id_to_hyperedge_bijection(self):
        self.edge_label_to_nodes = self.hypernetwork_obj.return_edge_dict()

    def construct_network_and_hypernetwork(self,hypernetwork_files,netw_file):
        self.hypernetwork_obj = MMOTHypernetworkObject(hypernetwork_files)

    def initialise_curvature(self):
        #just creates curvature objects in network
        curvature_dict = self.hypernetwork_obj.get_network_curvature()
        initial_curv_vals = [(score, edge_id) for edge_id, score in curvature_dict.items()]
        self.is_hyp_node_removed = dict.fromkeys(self.edge_label_to_nodes.keys(), False)
        self.hyperedge_queue.push_mult(initial_curv_vals)
        
    def recalculate_curvature(self):
        '''
        maybe this should be self.network_obj.values_to_add
        
        if (self.network_obj.last_node_removed != None):
            self.new_queue_entries = self.network_obj.update_neighbourhood_scores(self.network_obj.last_node_removed)
            self.hyperedge_queue.push_mult(self.new_queue_entries)
        else:
            print("error")
        '''
        if getattr(self, "last_removed_hyp_members", None):
            # node distributions + clique expansion are stale after a removal
            self.hypernetwork_obj._build_state()

            updates = self.hypernetwork_obj.update_neighbourhood_scores(self.last_removed_hyp_members)
            # update_neighbourhood_scores yields [eid, curvature]; the queue needs (score, id)
            self.new_queue_entries = [(curvature, eid) for eid, curvature in updates]

            # keep the live store in sync, or next_hyperedge_removal's score check never matches
            for score, eid in self.new_queue_entries:
                self.hypernetwork_obj.edge_curvatures[eid] = score

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
            if self.is_hyp_node_removed[node] == False and score == self.hypernetwork_obj.edge_curvatures[node]:
                break 
        self.is_hyp_node_removed[node] = True
        return node


    def hyperedge_removal(self):
        #get the lowest value
        hyperedge_to_remove = self.next_hyperedge_removal()
        if hyperedge_to_remove != None :
            #passing in edge_id
            nodes_of_hyperedge = list(self.edge_label_to_nodes[hyperedge_to_remove])
            self.hypernetwork_obj.remove_hyperedge(nodes_of_hyperedge)
            self.last_removed_hyp_members = self.hypernetwork_obj.last_removed_hyp_members
            return False
        else:
            return True

    def return_init_curvature(self):
        # hypernetwork_obj -- these functions need written
        hyperedge_curv_dict = self.hypernetwork_obj.get_network_curvature()
        nodes_curv_dict = self.hypernetwork_obj.hypernetwork_nodes_curv()
        return hyperedge_curv_dict, nodes_curv_dict

    def change_hyperedge_keys(self,input_dict):
        #new_dict = {
        #    self.edge_label_to_nodes[eid]: curvature
        #    for eid, curvature in input_dict.items()
        #}
        new_dict = {
            ",".join(map(str, sorted(self.edge_label_to_nodes[eid]))): curvature
            for eid, curvature in input_dict.items()
        }
        return new_dict