import os
import networkx as nx
import re
from pathlib import Path
from src2.data_object import DataObject
from src2.hypernetwork_class import HypernetworkObject
from src2.dualNetworkClass import DualNetworkObject
from src.create_dual_networks import process_and_save_dual_complexes
from src2.queue_object import PriorityQueue

class OllivierRicciDualDataObject(DataObject):
    def __init__(self,hypernetwork_location,network_location,extra1=0,extra2=0):
        super().__init__(hypernetwork_location,network_location,extra1)
        self.network_decomposition = "dual_networks"
        self.hypernetwork_location = hypernetwork_location
        self.hypernetwork_obj = HypernetworkObject(hypernetwork_location)
        self.network_obj = DualNetworkObject(network_location)
        self.hyperedge_queue = PriorityQueue()        


    def construct_network_and_hypernetwork(self,hyp_loc,net_loc):
        '''
        Need to ensure the bijection of hyperedges to network_object is understood.
        So this is specific for FRC and creating the poset
        '''
        self.hypernetwork_obj = HypernetworkObject(hyp_loc)
        self.network_obj = DualNetworkObject(net_loc)

    def construct_network(self,data_source,dataset_name):
        #from the hypernetwork file, get all the subnetworks
        #can add if need the functionality?
        SCRIPT_DIR = Path(__file__).resolve().parent
        BASE_DATA_DIR = SCRIPT_DIR.parent / "data"
        data_source = os.path.join(BASE_DATA_DIR, data_source)
        process_and_save_dual_complexes(data_source,dataset_name)

    def files_for_network(self,source,name):
        needed_info = ["nodes","edges"]
        paths_search = []
        base_dir = Path("data")
        #but now need to get the networks of all the networks of different cardinality
        file_of_nodes = os.path.join(base_dir, source, self.network_decomposition, "nodes", name)
        self.hyperedge_cardinalities = self.extract_cardinalities_from_files(file_of_nodes)

        for c in self.hyperedge_cardinalities:
            #add these as a sublist/array, so now 2d
            cardinality_pairs = []
            for n in needed_info:
                n_path_string = os.path.join(base_dir, source, self.network_decomposition, n, name, f"{n}_k{c}.txt")
                cardinality_pairs.append(n_path_string)
            paths_search.append(cardinality_pairs)
        return paths_search
    

    def extract_cardinalities_from_files(self, folder_path):
        """
        read all the files from this file location, but from their names in the form nodes_k{number}.txt
        """
        cardinalities = []
        
        # Define a regex pattern: 'nodes_k' followed by one or more digits (\d+), ending in '.txt'
        # The parenthesis () create a capture group for just the digits
        pattern = re.compile(r'^nodes_k(\d+)\.txt$')
        
        try:
            # List all files in the given directory
            for filename in os.listdir(folder_path):
                match = pattern.match(filename)
                if match:
                    # Extract the captured number string and convert it to an int
                    number = int(match.group(1))
                    cardinalities.append(number)
        except FileNotFoundError:
            print(f"Error: The folder '{folder_path}' does not exist.")
            return []

        # Return the numbers sorted for easier processing later
        return sorted(cardinalities)

    def network_from_files(self,file_location_verified,paths_tuples):
        #initialNetwork will be a dictionary of initial networks
        self.initialNetwork = {c: None for c in self.hyperedge_cardinalities}
        print(range(len(self.hyperedge_cardinalities)))
        for i in range(len(self.hyperedge_cardinalities)):
            smallNetworkPaths = paths_tuples[i]
            #extract cardinality, so checking that paths_tuples[i] matches self.hyperedge_cardinalities
            pattern = re.compile(r'_k(\d+)\.txt$')  
            match = pattern.search(smallNetworkPaths[0])
            if match:
                # 3. Extract the actual string digits (e.g., "5") and convert to an integer
                network_card = int(match.group(1))

            print(f"Construct graph object for cardinality {network_card}")
            G = nx.Graph()
            # 1. Stream edges into the graph
            # (line.split() handles both spaces and tabs automatically)
            # 2. Stream nodes to ensure isolated nodes (nodes with no edges) are included
            with open(smallNetworkPaths[0], 'r') as f:
                node_generator = (line.strip() for line in f if line.strip())
                G.add_nodes_from(node_generator)

            with open(smallNetworkPaths[1], 'r') as f:
                edge_generator = (line.split() for line in f if line.strip())
                G.add_edges_from(e for e in edge_generator if len(e) >= 2)
                
            self.initialNetwork[network_card] = G
    
    def hyperedge_removal(self):
        pass
    def initialise_curvature(self):
        pass
    def recalculate_curvature(self):
        pass

    def return_init_curvature(self):
        hyperedge_curv_dict = self.network_obj.get_network_curvature()
        nodes_curv_dict = self.hypernetwork_nodes_curv(hyperedge_curv_dict)
        return hyperedge_curv_dict, nodes_curv_dict

    def hypernetwork_nodes_curv(self,hyperedge):

        #all_keys_union = set().union(*hyperedge.keys())

        node_curvature = {}
        n_defined = {}
        for members, v in hyperedge.items():
            for k in members:
                node_curvature.setdefault(k, 0.0)
                n_defined.setdefault(k, 0)
                if v == v:                    # False only for NaN
                    node_curvature[k] += v
                    n_defined[k] += 1

        for k, count in n_defined.items():
            if count == 0:
                #maybe change this to zero, as for the nodes themselves
                node_curvature[k] = float("nan")

        return node_curvature

    def change_hyperedge_keys(self,input_dict):
        new_dict = {
            ",".join(map(str, sorted(eid))): curvature
            for eid, curvature in input_dict.items()
        }
        return new_dict