from src2.data_object import DataObject
from src2.hypernetwork_random import randomHypernetworkObject
from src2.indep_functions import cardinality_distribution
import numpy as np

class randomDataObject(DataObject):
    def __init__(self,hypernetwork_location,network_location,extra_file,target_dist):
        super().__init__(hypernetwork_location,network_location,extra_file)
        if target_dist == "None":
            self.maximum_cardinality = 1
        else:
            self.maximum_cardinality = len(target_dist)

    def construct_network_and_hypernetwork(self,hypernetwork_files,netw_file):
        self.hypernetwork_obj = randomHypernetworkObject(hypernetwork_files)
        self.number_of_nodes = self.hypernetwork_obj.number_of_nodes()
        print(f"Number of nodes", self.number_of_nodes)
        self.edge_label_to_nodes = self.hypernetwork_obj.return_edge_dict()
        self.is_hyp_node_removed = dict.fromkeys(self.edge_label_to_nodes.keys(), False)

    def initialise_curvature(self):
        #just creates curvature objects in network
        pass
        
    def recalculate_curvature(self):
        pass

    def next_hyperedge_removal(self,target_distribution,hypernetwork_distribution):
        '''
        Node variable is actually a hyperedge here
        '''
        print(f"Target distribution", target_distribution)
        print(f"Current hypernetwork distribution", hypernetwork_distribution)
        node = None  # Safeguard if queue empties without finding a valid node
        while True:
            if self.hypernetwork_obj.no_hyperedges_left():
                print("no hyperedges left")
                return None
            if target_distribution == "None":
                node = self.hypernetwork_obj.get_random_hyperedge()
                print(node)
            else:
                complete = False
                while complete == False:
                    node = self.hypernetwork_obj.get_random_hyperedge()
                    cardinality = self.hypernetwork_obj.get_hyperedge_cardinality(node)
                    #might be slow if there are few of the correct cardinality available
                    if target_distribution[cardinality] < hypernetwork_distribution[cardinality]:
                        #print("Random removal")
                        break

            if not self.is_hyp_node_removed[node]:
                break
        self.is_hyp_node_removed[node] = True
        return node


    def hyperedge_removal(self,target_dist1):
        #get the lowest value
        current_distribution = cardinality_distribution(self.hypernetwork_obj.itertative_H,self.maximum_cardinality)
        hyperedge_to_remove = self.next_hyperedge_removal(target_dist1,current_distribution)
        if hyperedge_to_remove != None :
            #passing in edge_id
            nodes_of_hyperedge = list(self.edge_label_to_nodes[hyperedge_to_remove])
            self.hypernetwork_obj.remove_hyperedge(nodes_of_hyperedge)
            self.last_removed_hyp_members = self.hypernetwork_obj.last_removed_hyp_members
            return False, self.edge_label_to_nodes[hyperedge_to_remove]
        else:
            return True, None

    def return_init_curvature(self):
        pass

    def change_hyperedge_keys(self,input_dict):
        new_dict = {
            ",".join(map(str, sorted(self.edge_label_to_nodes[eid]))): curvature
            for eid, curvature in input_dict.items()
        }
        return new_dict