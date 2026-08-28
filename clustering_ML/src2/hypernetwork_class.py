import xgi
from src2.hypernetwork_base import BaseHypernetworkObject


class HypernetworkObject(BaseHypernetworkObject):
    def __init__(self, file_in):
        super().__init__(file_in)
        self.mod_key = {}  # {frozenset(nodes): per-cluster modularity contribution}