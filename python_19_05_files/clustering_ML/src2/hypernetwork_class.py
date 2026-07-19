import hypernetx as hnx
import xgi
import pandas as pd
from collections import Counter
from math import comb
from src2.indep_functions import calculate_modularity_ext
class HypernetworkObject():
    def __init__(self,file_in):
        self.initialHypernetwork = self.hypernetwork_from_files(file_in)
        self.itertative_H = self.initialHypernetwork
        self.previous_partition = None
        self.previous_modularity = None
        #for modularity
        self.hnx_initialHypernetwork = self.xgi_to_hnx(self.initialHypernetwork)
        self.wdc = "linear"
        self.mod_key = {}  # {frozenset(nodes): per-cluster modularity contribution}

    def hypernetwork_from_files(self, file):
        '''
        Commented out code would keep isolated nodes.
        Labels are preserved, so if 3 is an isolated nodes,
        the remaining nodes in hypernetwork would  be 1,2,4,5
        '''
        # We will build a list of rows for a DataFrame directly
        edge_node_rows = []
        edge_dict = {}
        self.node_to_edge_id_map = {}
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

                sorted_nodes = tuple(sorted(nodes_set))
                self.node_to_edge_id_map[sorted_nodes] = edge_name
                
        # 4. Initialize Hypergraph directly from the dictionary (No Pandas needed!)
        self.initialHypernetwork = xgi.Hypergraph(edge_dict)
        
        # Quick check to ensure the topology parsed correctly
        #print(f"Loaded: {len(self.initialHypernetwork.nodes)} nodes and {len(self.initialHypernetwork.edges)} edges.")
        
        return self.initialHypernetwork

    def xgi_to_hnx(self,H_xgi):
        '''
        Get hnx initial_network for modularity calculations
        Note that isolated nodes will be dropped
        '''
        edge_dict = H_xgi.edges.members(dtype=dict)
        H_hnx = hnx.Hypergraph(edge_dict)
        return H_hnx

    def remove_hyperedge(self, hyperedge_nodes):
        hyperedge = self.hyperedge_to_edge_id(hyperedge_nodes)
        if hyperedge is not None:
            print(f"Removing hyperedge {hyperedge} containing nodes {hyperedge_nodes}")
            self.itertative_H.remove_edge(hyperedge)
        else:
            print(f"[Warning] Could not find a hyperedge with exactly these nodes: {hyperedge_nodes}")
            
    def hyperedge_to_edge_id(self, nodes):
        '''
        Just handle the edge id case in here.
        Check if nodes are alreadya actually passing in a valid edge
        '''
        # Case 1: already a valid edge id
        try:
            if nodes in self.itertative_H.edges:
                return nodes
        except TypeError:
            pass

        # Case 2: comma-separated string, e.g. "7,12,57,76" or "1, 2, 3"
        if isinstance(nodes, str):
            try:
                int_nodes = [int(n.strip()) for n in nodes.split(',')]
            except ValueError:
                print("ValError fail")
                return None
            sorted_nodes = tuple(sorted(int_nodes))
            return self.node_to_edge_id_map.get(sorted_nodes)

        # Case 3: collection of node ids
        if not isinstance(nodes, (list, tuple, set, frozenset)):
            print("is_instance failure")
            return None

        try:
            int_nodes = [int(n) for n in nodes]
        except (ValueError, TypeError):
            print("ValError fail")
            return None

        sorted_nodes = tuple(sorted(int_nodes))
        return self.node_to_edge_id_map.get(sorted_nodes)

    def get_partitions(self):
        '''
        RETURN TO THIS LATER AS A REPLACEMENT FOR PREFERENTIAL ATTACHMENT
        H.s_connected_components(s=2)
        '''
        print(f"Number of hyperedges: {self.itertative_H.num_edges}")
        components = list(xgi.connected_components(self.itertative_H))
        print(f"Number of connected components: {len(components)}")
        return components

    def attach_partitions(self):
        pass

    def calculate_modularity(self,partitions):
        if partitions == self.previous_partition:
            return self.previous_modularity
        else:
            new_modularity = calculate_modularity_ext(self.hnx_initialHypernetwork,partitions,'linear')
            print(f"new_modularity",new_modularity)
            self.previous_modularity = new_modularity
            return new_modularity
 
    # ------------------------------------------------------------------
    # step 1: disconnected partitions
    # ------------------------------------------------------------------
    def get_disconnected_partitions(self):
        components = [set(cc) for cc in xgi.connected_components(self.itertative_H)]
        print(f"Number of disconnected components",len(components))
        return components
 
    # ------------------------------------------------------------------
    # steps 2-4: memoized per-cluster modularity contribution
    # ------------------------------------------------------------------
    def cluster_contribution(self, cluster):
        """
        Return the modularity contribution of `cluster` as a standalone
        community: e_H(cluster) - E[e_H(cluster)].
        Looks up self.mod_key first; computes + caches on a miss.
        """
        key = frozenset(cluster)
        if key not in self.mod_key:
            self.mod_key[key] = self._compute_contribution(key)
        return self.mod_key[key]
 
    def score_partitions(self, partitions):
        """
        Run steps 2-4 of the iteration over a list of candidate clusters:
        for each, look up or compute+cache its modularity contribution.
        Returns {frozenset(cluster): contribution}.
        """
        return {frozenset(p): self.cluster_contribution(p) for p in partitions}
 
    def _compute_contribution(self, cluster):
        cluster = set(cluster)
        edges = self.itertative_H.edges
 
        # observed term: sum of edge weights whose dominant-community
        # weight counts toward `cluster`
        observed = 0.0
        for eid in edges:
            e = set(edges.members(eid))
            d = len(e)
            c = len(e & cluster)
            observed += self._chi(d, c)
 
        # expected term (Chung-Lu-style degree tax)
        degrees = self.itertative_H.nodes.degree
        vol_cluster = sum(degrees[n] for n in cluster)
        vol_total = sum(degrees[n] for n in self.itertative_H.nodes)
        p = vol_cluster / vol_total if vol_total else 0.0
 
        sizes = Counter(len(edges.members(eid)) for eid in edges)
        total_edges = sum(sizes.values())
        expected = 0.0
        for d, count in sizes.items():
            Pd = count / total_edges
            expected += Pd * self._expected_chi(d, p)
 
        return observed - expected
 
    def _chi(self, d, c):
        """Edge-contribution function, per strict/majority/linear form."""
        if self.wdc == "strict":
            return 1.0 if c == d else 0.0
        if self.wdc == "majority":
            return 1.0 if c > d / 2 else 0.0
        if self.wdc == "linear":
            return (2 * c - d) / d if c > d / 2 else 0.0
        raise ValueError(f"Unknown wdc form: {self.wdc}")
 
    def _expected_chi(self, d, p):
        """Expectation of chi(d, C) where C ~ Binomial(d, p)."""
        if self.wdc == "strict":
            return p ** d
        total = 0.0
        for c in range(d + 1):
            prob = comb(d, c) * (p ** c) * ((1 - p) ** (d - c))
            total += prob * self._chi(d, c)
        return total
 
    # ------------------------------------------------------------------
    # cluster attachment
    # ------------------------------------------------------------------
    def _cluster_size(self, cluster, size_by="nodes"):
        if size_by == "nodes":
            return len(cluster)
        if size_by == "volume":
            degrees = self.itertative_H.nodes.degree
            return sum(degrees[n] for n in cluster)
        raise ValueError(f"Unknown size_by: {size_by}")
 
    def attach_clusters(self, partition, target_number, size_by="nodes"):
        """
        Repeatedly take the smallest cluster and merge it into whichever of
        the `target_number` largest OTHER clusters most increases modularity,
        until only `target_number` clusters remain.
        """
        partition = [set(c) for c in partition]
 
        while len(partition) > target_number:
            s_idx = min(
                range(len(partition)),
                key=lambda i: self._cluster_size(partition[i], size_by),
            )
            s = partition[s_idx]
 
            host_idxs = sorted(
                (i for i in range(len(partition)) if i != s_idx),
                key=lambda i: self._cluster_size(partition[i], size_by),
                reverse=True,
            )[:target_number]
 
            best_idx, best_delta = None, float("-inf")
            s_contrib = self.cluster_contribution(s)
            for h in host_idxs:
                host = partition[h]
                merged = host | s
                delta = (
                    self.cluster_contribution(merged)
                    - self.cluster_contribution(host)
                    - s_contrib
                )
                if delta > best_delta:
                    best_delta, best_idx = delta, h
 
            partition[best_idx] = partition[best_idx] | s
            del partition[s_idx]
 
        return partition
 
    # ------------------------------------------------------------------
    # everything that belongs inside one iteration of the (uncoded) outer loop
    # ------------------------------------------------------------------
    def run_iteration(self, target_number, size_by="nodes"):
        partitions = self.get_disconnected_partitions()
        self.score_partitions(partitions)          # populates/uses mod_key
        if len(partitions) > target_number:
            partitions = self.attach_clusters(partitions, target_number, size_by)
        return partitions
 