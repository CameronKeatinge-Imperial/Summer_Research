import xgi
import random
from src2.hypernetwork_base import BaseHypernetworkObject


class randomHypernetworkObject(BaseHypernetworkObject):
    def __init__(self, file_in):
        super().__init__(file_in)
        self.mod_key = {}  # {frozenset(nodes): per-cluster modularity contribution}

    def get_random_hyperedge(self, rng=random):
        '''
        rng means can set seed
        '''
        remaining = list(self._edge_to_nodes)
        if not remaining:
            return None
        return rng.choice(remaining)

    def return_edge_dict(self):
        return dict(self._edge_to_nodes)
    
    def no_hyperedges_left(self):
        return not self._edge_to_nodes

    def get_hyperedge_cardinality(self,node):
        return len(self._edge_to_nodes[node])