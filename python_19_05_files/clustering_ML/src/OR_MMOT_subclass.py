## forman ricci file
import numpy as np
import os
import networkx as nx
import itertools as it
from pathlib import Path
from src.iterative_clustering import StatisticalModelTemplate

class OllivierRicciMMOTClustering(StatisticalModelTemplate):
    
    def __init__(self):
        super().__init__()
        self.network_decomposition = "bipartite_graph"