import hypernetx as hnx
import pandas as pd
from src2.indep_functions import calculate_modularity_ext
class HypernetworkObject():
    def __init__(self,file_in):
        self.initialHypernetwork = self.hypernetwork_from_files(file_in)
        self.itertative_H = self.initialHypernetwork
        self.previous_partition = None
        self.previous_modularity = None

    def hypernetwork_from_files(self, file):
        '''
        Commented out code would keep isolated nodes.
        Labels are preserved, so if 3 is an isolated nodes,
        the remaining nodes in hypernetwork would  be 1,2,4,5
        '''

        #paths_search_h= []
        #paths_search_h.append(hyperedge_path)
        #paths_search_h.append(hypernet_nodes_path)
        # We will build a list of rows for a DataFrame directly
        edge_node_rows = []
        edge_dict = {}
        file = file[0]
        
        with open(file, 'r') as f:
            for edge_id, line in enumerate(f):  
                stripped = line.strip()
                
                if not stripped:
                    continue
                    
                try:
                    # 1. Split and convert to integers, using set() to drop duplicates
                    nodes_set = set(int(n) for n in stripped.split())
                except ValueError:
                    # Skip header rows or lines that can't be converted to ints
                    continue 
                
                # 2. Skip if empty
                if not nodes_set:
                    continue
                
                # 3. Create the edge name and assign the node set to the dictionary
                edge_name = f"e{edge_id}" 
                edge_dict[edge_name] = nodes_set
                
        # 4. Initialize Hypergraph directly from the dictionary (No Pandas needed!)
        self.initialHypernetwork = hnx.Hypergraph(edge_dict)
        
        # Quick check to ensure the topology parsed correctly
        print(f"Loaded: {len(self.initialHypernetwork.nodes)} nodes and {len(self.initialHypernetwork.edges)} edges.")
        
        return self.initialHypernetwork

    def remove_hyperedge(self,hyperedge):
        print(f"removing ", hyperedge)
        self.itertative_H = self.itertative_H.remove_edges([hyperedge])

    def get_partitions(self):
        '''
        RETURN TO THIS LATER AS A REPLACEMENT FOR PREFERENTIAL ATTACHMENT
        H.s_connected_components(s=2)
        '''
        print(f"Number of hyperedges: {self.itertative_H.edges()}")

        num_components = len(list(self.itertative_H.connected_components()))
        print(f"Number of connected components: {num_components}")
        return list(self.itertative_H.connected_components())
        #return list(self.itertative_H.s_components(s=1, edges=False))
        #false just measn that the nodes are returned
    
    def calculate_modularity(self,partitions):
        if partitions == self.previous_partition:
            return self.previous_modularity
        else:
            new_modularity = calculate_modularity_ext(self.itertative_H,partitions,'linear')
            print(f"new_modularity",new_modularity)
            self.previous_modularity = new_modularity
            return new_modularity