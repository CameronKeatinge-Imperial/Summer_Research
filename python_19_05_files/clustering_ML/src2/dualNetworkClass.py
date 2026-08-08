from typing import Any
import numpy as np
import networkx as nx
from pathlib import Path
from collections.abc import Iterable
import os
import random
import numpy as np
import ot
# from src.data_processing import load_config
# from src.create_dual_networks import process_and_save_dual_complexes

class DualNetworkObject():

    def __init__(self,network_files):
        self.int_to_hyperedge = {}
        self.hyperedge_to_int = {}
        self.edge_dependent_node = {}
        self.card_networks = {}
        self.networks_from_files(network_files)

    def networks_from_files(self,paths_to_read, p = None):
        '''
        Paths to read is an N x 2 list
        '''
        #define dictionary

        for i in range(len(paths_to_read)):
            print(paths_to_read[i])
            self.card_networks[self.get_cardinality(paths_to_read[i])] = self.construct_indiv_dual_network(paths_to_read[i])
            print(f"constructing network number", i)

    @staticmethod
    def _parse_hyperedge_str(s):
        """'[23, 25, 49, 66]' -> [23, 25, 49, 66]"""
        s = s.strip().strip('[]')
        return frozenset(int(x) for x in s.strip('[]').split(',')) if s else frozenset()
        #return frozenset(int(x) for x in s.split(',')) if s else {}

    def get_cardinality(self,files):
        nodes_file = files[0]
        with open(nodes_file, 'r') as f:
            first_line = f.readline()
        cardinality = len(self._parse_hyperedge_str(first_line))
        return cardinality

    def construct_indiv_dual_network(self, files):
        '''
        Should be length 2, start here tomorrow.
        '''
        nodes_p = files[0]
        edges_p = files[1]
        G = nx.Graph()

        # 1. Register every hyperedge listed in nodes_p as a dual-graph node.
        with open(nodes_p, 'r') as f:
            hyperedge_member_sets = (
                self._parse_hyperedge_str(line) for line in f if line.strip()
            )
            nodes = self.map_hyperedge_nodes_to_int(hyperedge_member_sets)
            G.add_nodes_from(nodes)

        # 2. Stream edges into the graph. Each line: "[..]:[..]"
        with open(edges_p, 'r') as f:
            hyperedge_edges = [
                tuple(self._parse_hyperedge_str(part) for part in line.split(':'))
                for line in f if line.strip()
            ]

            edges_in_dual = self.hyperedge_to_node_dual_edge_transform(hyperedge_edges)
            edges_in_dual = self.map_edge_to_dependent_node(hyperedge_edges)
            if edges_in_dual != None:
                G.add_edges_from(e for e in edges_in_dual if len(e) >= 2)

        return G

    def map_hyperedge_nodes_to_int(self, hyperedges_list):
        '''
        Init should initiate this hashmap
        Should have key as the node number, then value as the set of nodes (in hypernetwork)
        Each time new hyperedge is passed in (hyperedges will be a list), add to the key sequentially as integers
        '''
        nodes_to_return = []
        for hyperedge in hyperedges_list:
            if hyperedge not in self.hyperedge_to_int:
                new_id = len(self.int_to_hyperedge)
                self.int_to_hyperedge[new_id] = hyperedge
                self.hyperedge_to_int[hyperedge] = new_id
                nodes_to_return.append(new_id)
        return nodes_to_return

    def hyperedge_to_node_dual_edge_transform(self, hyperedges_list):
        '''
        Get all the edges in the form of being between two hyperedges
        Transform that to an edge between two nodes, using mapping defined in map_hyperedge_nodes_to_int()
        '''
        edges = []
        for h1, h2 in hyperedges_list:
            shared = frozenset(h1) & frozenset(h2)
            id1 = self.hyperedge_to_int[frozenset(h1)]
            id2 = self.hyperedge_to_int[frozenset(h2)]
            edges.append((id1, id2))
        return edges


    def get_dependent_node(self, h1, h2):
        shared = frozenset(h1) & frozenset(h2)
        try:
            return self.hyperedge_to_int[shared]
        except KeyError:
            #so if there are no hyperedges that can be removed below, then same none
            #particularly if a standard edge, with nodes that cannot be removed by hyperedge removal
            return None


    def map_edge_to_dependent_node(self, hyperedges_list):
        '''
        For each edge, map to a node of cardinality (of one less)
        Of the two hyperedges that form an edge/pair, they will share all but one node within their hyperedge
        This intersection itself will be a node, which can be found with the bijection defined in map_hyperedge_int
        '''
        for h1, h2 in hyperedges_list:
            id1 = self.hyperedge_to_int[frozenset(h1)]
            id2 = self.hyperedge_to_int[frozenset(h2)]
            dep_id = self.get_dependent_node(h1, h2)
            self.edge_dependent_node[tuple(sorted((id1,id2)))] = dep_id

    def get_network_curvature(self):
        node_curvature_by_network = {}

        for network in self.card_networks.values():
            network_curv = self.get_indiv_dual_network_curvature(network)
            node_curvature_by_network.update(network_curv)
        return node_curvature_by_network
    
    def get_indiv_dual_network_curvature(self,network):
        dist_matrix = dict(nx.all_pairs_shortest_path_length(network, cutoff=4))

        edge_curvature = {}

        for edge in network.edges():
            u, v = edge

            if u == v:
                continue  # skip self-loops

            u_neighbors = list(network.neighbors(u))
            v_neighbors = list(network.neighbors(v))

            u_mass = self.neighbour_distribution(u, u_neighbors, 0.05)
            v_mass = self.neighbour_distribution(v, v_neighbors, 0.05)

            u_support = [u] + u_neighbors
            v_support = [v] + v_neighbors

            cost_matrix = np.array([
                [dist_matrix[a][b] for b in v_support]
                for a in u_support
            ], dtype=float)

            transport_plan = ot.sinkhorn(
                np.array(u_mass), np.array(v_mass),
                cost_matrix, reg=0.01
            )
            w1_distance = np.sum(transport_plan * cost_matrix)

            edge_curvature[edge] = 1 - w1_distance


        # now sum curvature of every edge incident on each node
        node_curvature = {}
        for (u, v), curvature in edge_curvature.items():
            u_key = self.int_to_hyperedge[u]
            v_key = self.int_to_hyperedge[v]
            node_curvature[u_key] = node_curvature.get(u_key, 0) + curvature
            node_curvature[v_key] = node_curvature.get(v_key, 0) + curvature

        for node in network.nodes():
            key = self.int_to_hyperedge[node]
            if key not in node_curvature:
                node_curvature[key] = 0

        return node_curvature

# DO I NEED THIS
def neighbour_distribution(node, neighbors, alpha):
    '''
    Standard ORC mass distribution: alpha on self, (1-alpha) split
    uniformly among neighbors.
    '''
    if not neighbors:
        return [1.0]  # isolated node, all mass on itself
    mass = [alpha] + [(1 - alpha) / len(neighbors)] * len(neighbors)
    return mass

