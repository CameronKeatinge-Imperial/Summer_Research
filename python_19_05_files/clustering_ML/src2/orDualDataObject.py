from src2.data_object import DataObject
from src2.hypernetwork_class import HypernetworkObject
from src2.posetNetworkClass import PosetNetworkObject

class OllivierRicciDualDataObject(DataObject):
    def __init__(self,hypernetwork_location,network_location):
        self.hypernetwork_obj = HypernetworkObject(hypernetwork_location)
        self.network_obj = PosetNetworkObject(network_location)

    def construct_network_and_hypernetwork():
        '''
        Need to ensure the bijection of hyperedges to network_object is understood.
        So this is specific for FRC and creating the poset
        '''
        self.network_from_files()


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
    
