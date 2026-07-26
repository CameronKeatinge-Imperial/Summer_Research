from collections import Counter
from math import comb
from dataclasses import dataclass

import networkx as nx
import numpy as np
import scipy.sparse as sp
import ot
import xgi
from scipy.sparse.csgraph import dijkstra
from src2.indep_functions import calculate_modularity_ext

'''
For better efficiency, manage cache
'''

@dataclass
class EdgeBarycenterResult:
    edge_id: object
    nodes: list
    support: list
    costs: np.ndarray
    mean_cost: float
    curvature: float
    method: str

class HypergraphDistance:
    def __init__(self, A: "sp.csr_matrix", node_to_idx: dict, idx_to_node: np.ndarray,
                 radius: float = 5.0):
        #A is clique
        self.A = A
        self.node_to_idx = node_to_idx
        self.idx_to_node = idx_to_node
        self.radius = radius

    def __call__(self, u, v) -> float:
        #gets the distance between u and v.
        if u == v:
            return 0.0
        u_idx = self.node_to_idx[u]
        v_idx = self.node_to_idx[v]
        row = dijkstra(csgraph=self.A, directed=False, indices=u_idx,
                        unweighted=True, limit=self.radius)
        d = row[v_idx]
        return np.inf if np.isinf(d) else float(d)

    def submatrix(self, U: list) -> np.ndarray:
        #|U|x|U| matrix
        idx_U = np.array([self.node_to_idx[u] for u in U])
        D = dijkstra(csgraph=self.A, directed=False, indices=idx_U,
                      unweighted=True, limit=self.radius)
        return D[:, idx_U]


class MMOTHypernetworkObject():
    def __init__(self, file_in):
        self.initialHypernetwork = self.hypernetwork_from_files(file_in)
        self.itertative_H = self.initialHypernetwork
        self.previous_partition = None
        self.previous_modularity = None

    def additional_parameters(self, reg: float = 0.05, maxiter: int = 500,
                               tol: float = 1e-6, lp_threshold: int = 150,
                               lazy_support: bool = True):
        self.wdc = "linear"

        self.reg = reg
        self.maxiter = maxiter
        self.tol = tol
        self.lp_threshold = lp_threshold
        self.lazy_support = lazy_support

        # state is rebuilt fresh each time it's needed -- no staleness
        # flag, no persisted "is this still valid" tracking
        self.measures = None
        self.A = None
        self.dist = None
        self.node_to_idx = None
        self.idx_to_node = None

    def hypernetwork_from_files(self, file):
        edge_dict = {}
        self.node_to_edge_id_map = {}
        file = file[0]

        with open(file, 'r') as f:
            for edge_id, line in enumerate(f):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    nodes_set = set(int(n) for n in stripped.replace(",", " ").split())
                except ValueError:
                    continue
                if not nodes_set:
                    continue

                edge_name = f"e{edge_id}"
                edge_dict[edge_name] = nodes_set
                sorted_nodes = tuple(sorted(nodes_set))
                self.node_to_edge_id_map[sorted_nodes] = edge_name

        self.initialHypernetwork = xgi.Hypergraph(edge_dict)
        return self.initialHypernetwork

    def return_edge_dict(self):
        edge_dict = {eid: set(self.initialHypernetwork.edges.members(eid))
             for eid in self.initialHypernetwork.edges}
        return edge_dict
    
    def remove_hyperedge(self, hyperedge_nodes):
        hyperedge = self.hyperedge_to_edge_id(hyperedge_nodes)
        if hyperedge is not None:
            print(f"Removing hyperedge {hyperedge} containing nodes {hyperedge_nodes}")
            removed_members = set(self.itertative_H.edges.members(hyperedge))

            self.itertative_H.remove_edge(hyperedge)
            self.last_edge_removed = hyperedge
        else:
            print(f"[Warning] Could not find a hyperedge with exactly these nodes: {hyperedge_nodes}")

    def update_neighbourhood_scores(self, removed_nodes=None) -> list:
        """
        Recomputes curvature for every hyperedge whose current membership
        intersects the removed hyperedge's nodes. Since nothing is cached,
        this filter only limits *which* edges get recomputed -- each one
        that qualifies is still computed fully from scratch.
        """
        if removed_nodes is None:
            print("Done")
        if not removed_nodes:
            return []
        removed_nodes = set(removed_nodes)

        #THIS MAY NEED CHANGED
        affected_edges = [
            eid for eid in self.itertative_H.edges
            if removed_nodes & set(self.itertative_H.edges.members(eid))
        ]

        updates = []
        for eid in affected_edges:
            result = self.compute_edge(eid)
            updates.append([eid, result.curvature])
        #THIS GIVES THE NEW HYPEREDGE CURVATURES
        return updates

    def hyperedge_to_edge_id(self, nodes):
        try:
            if nodes in self.itertative_H.edges:
                return nodes
        except TypeError:
            pass

        if isinstance(nodes, str):
            try:
                int_nodes = [int(n.strip()) for n in nodes.split(',')]
            except ValueError:
                print("ValError fail")
                return None
            sorted_nodes = tuple(sorted(int_nodes))
            return self.node_to_edge_id_map.get(sorted_nodes)

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

    # ------------------------------------------------------------------
    # curvature / MMOT barycenter -- fully recomputed, no cache
    # ------------------------------------------------------------------
    def _build_state(self):
        """
        Rebuilds node distributions, adjacency, and the distance helper
        from the current itertative_H. Called unconditionally -- there is
        no flag deciding whether this is "necessary"; it always runs.
        """
        self.measures = node_distributions(self.itertative_H)
        self.A, self.node_to_idx, self.idx_to_node = build_sparse_adjacency(self.itertative_H)
        self.dist = HypergraphDistance(self.A, self.node_to_idx, self.idx_to_node)

    def compute_edge(self, e) -> EdgeBarycenterResult:
        """Compute barycenter result for a single hyperedge e, from scratch."""
        self._build_state()

        nodes_e = list(self.itertative_H.edges.members(e))
        N = len(nodes_e)

        if N <= 1:
            return EdgeBarycenterResult(
                edge_id=e, nodes=nodes_e, support=nodes_e,
                costs=np.array([]), mean_cost=0.0, curvature=0.0, method="trivial",
            )

        U = get_expanded_support_matrix(
            nodes_e=nodes_e, mus=self.measures, A=self.A,
            node_to_idx=self.node_to_idx, idx_to_node=self.idx_to_node,
            lazy=self.lazy_support,
        )
        idx = {u: k for k, u in enumerate(U)}
        n = len(U)

        A_marginals = np.zeros((n, N))
        for col, i in enumerate(nodes_e):
            for target, w in self.measures[i].items():
                A_marginals[idx[target], col] = w

        C = self.dist.submatrix(U)
        weights = np.full(N, 1.0 / N)

        if n <= self.lp_threshold:
            beta, costs = lp_barycenter_with_costs(A_marginals, C, weights)
            method = "lp"
        else:
            beta, costs = ibp_barycenter_with_costs(
                A_marginals, C, self.reg, weights,
                maxiter=self.maxiter, tol=self.tol,
            )
            method = "ibp"

        return EdgeBarycenterResult(
            edge_id=e, nodes=nodes_e, support=U,
            costs=costs, mean_cost=float(costs.mean()),
            curvature=float(1 - costs.sum() / (len(nodes_e) - 1)),
            method=method,
        )

    def compute_all(self) -> dict:
        self._build_state()
        return {e: self.compute_edge(e) for e in self.itertative_H.edges}

    
    ##############
    # CLUSTERING #
    ##############
    def get_partitions(self):
        print(f"Number of hyperedges: {self.itertative_H.num_edges}")
        components = list(xgi.connected_components(self.itertative_H))
        print(f"Number of connected components: {len(components)}")
        return components

    def attach_partitions(self):
        pass

    def calculate_modularity(self, partitions):
        # no memoization: recomputed every call regardless of whether
        # partitions matches the previous call
        new_modularity = calculate_modularity_ext(self.hnx_initialHypernetwork, partitions, 'linear')
        print(f"new_modularity", new_modularity)
        self.previous_modularity = new_modularity
        return new_modularity

    def get_disconnected_partitions(self):
        components = [set(cc) for cc in xgi.connected_components(self.itertative_H)]
        print(f"Number of disconnected components", len(components))
        return components

    def cluster_contribution(self, cluster):
        # previously memoized in self.mod_key; now always recomputed
        return self._compute_contribution(frozenset(cluster))

    def score_partitions(self, partitions):
        return {frozenset(p): self.cluster_contribution(p) for p in partitions}

    def _compute_contribution(self, cluster):
        cluster = set(cluster)
        edges = self.itertative_H.edges

        observed = 0.0
        for eid in edges:
            e = set(edges.members(eid))
            d = len(e)
            c = len(e & cluster)
            observed += self._chi(d, c)

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
        if self.wdc == "strict":
            return 1.0 if c == d else 0.0
        if self.wdc == "majority":
            return 1.0 if c > d / 2 else 0.0
        if self.wdc == "linear":
            return (2 * c - d) / d if c > d / 2 else 0.0
        raise ValueError(f"Unknown wdc form: {self.wdc}")

    def _expected_chi(self, d, p):
        if self.wdc == "strict":
            return p ** d
        total = 0.0
        for c in range(d + 1):
            prob = comb(d, c) * (p ** c) * ((1 - p) ** (d - c))
            total += prob * self._chi(d, c)
        return total

    def _cluster_size(self, cluster, size_by="nodes"):
        if size_by == "nodes":
            return len(cluster)
        if size_by == "volume":
            degrees = self.itertative_H.nodes.degree
            return sum(degrees[n] for n in cluster)
        raise ValueError(f"Unknown size_by: {size_by}")

    def attach_clusters(self, partition, target_number, size_by="nodes"):
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

    def run_iteration(self, target_number, size_by="nodes"):
        partitions = self.get_disconnected_partitions()
        self.score_partitions(partitions)
        if len(partitions) > target_number:
            partitions = self.attach_clusters(partitions, target_number, size_by)
        return partitions
    
###################################
# EXTERNAL CALCULATION ALGORITHMS #
###################################

def ibp_barycenter_with_costs(A: np.ndarray, C: np.ndarray, reg: float,
                               weights: np.ndarray, maxiter: int = 500,
                               tol: float = 1e-6):
    n, N = A.shape
    K = np.exp(-C / reg)
    eps = 1e-300

    u = np.ones((n, N))
    beta = np.full(n, 1.0 / n)

    for _ in range(maxiter):
        v = A / np.maximum(K.T @ u, eps)
        Kv = K @ v
        log_beta = (np.log(np.maximum(Kv, eps)) * weights).sum(axis=1)
        beta_new = np.exp(log_beta)
        beta_new /= beta_new.sum()

        converged = np.max(np.abs(beta_new - beta)) < tol
        beta = beta_new
        u = beta[:, None] / np.maximum(Kv, eps)
        if converged:
            break

    CV = C[:, :, None] * v[None, :, :]
    KCV = np.einsum('jl,jlk->jk', K, CV)
    costs = (u * KCV).sum(axis=0)

    return beta, costs

def lp_barycenter_with_costs(A: np.ndarray, C: np.ndarray, weights: np.ndarray):
    beta = ot.lp.barycenter(A, C, weights)
    N = A.shape[1]
    costs = np.array([ot.emd2(A[:, k], beta, C) for k in range(N)])
    return beta, costs


def _single_node_distribution(H: "xgi.Hypergraph", i) -> dict:
    """
    mu_i for one node, factored out of node_distributions() so the
    incremental path can recompute a single node's distribution without
    re-walking the whole hypergraph.
    """
    incident_edges = list(H.nodes.memberships(i))
    deg = len(incident_edges)

    if deg == 0:
        return {i: 1.0}

    mu_i: dict = {}
    share_per_edge = 1.0 / deg
    for e in incident_edges:
        members = H.edges.members(e)
        others = [m for m in members if m != i]

        if len(others) == 0:
            raise ValueError(
                f"Node {i} has a singleton incident hyperedge {e} "
                f"(only member). Distribution is undefined for this "
                f"case per current assumptions."
            )

        w = share_per_edge / len(others)
        for o in others:
            mu_i[o] = mu_i.get(o, 0.0) + w

    return mu_i


def node_distributions(H: "xgi.Hypergraph") -> dict:
    """Build mu_i for every node i. Thin wrapper over the per-node helper."""
    return {i: _single_node_distribution(H, i) for i in H.nodes}

def build_sparse_adjacency(H: "xgi.Hypergraph"):
    """Sparse clique-expansion adjacency matrix + node<->index mappings."""
    nodes_list = list(H.nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes_list)}
    idx_to_node = np.array(nodes_list)

    I = xgi.incidence_matrix(H, sparse=True)
    A = I @ I.T
    A.data = np.ones_like(A.data)
    A.setdiag(0)
    A.eliminate_zeros()

    return A, node_to_idx, idx_to_node


def get_expanded_support_matrix(nodes_e: list, mus: dict, A: sp.csr_matrix,
                                 node_to_idx: dict, idx_to_node: np.ndarray,
                                 lazy: bool = True) -> list:
    """Matrix-based (SpMV) support expansion around a hyperedge."""
    N = A.shape[0]

    base_support_nodes = set(nodes_e)
    for i in nodes_e:
        base_support_nodes.update(mus[i].keys())

    if lazy:
        x = np.zeros(N)
        for u in nodes_e:
            x[node_to_idx[u]] = 1.0

        h1 = A @ x
        h2 = A @ h1
        reachable_mask = (x + h1 + h2) > 0
        reachable_nodes = set(idx_to_node[np.where(reachable_mask)[0]])

    else:
        k = len(nodes_e)
        X = np.zeros((N, k))
        for col_idx, u in enumerate(nodes_e):
            X[node_to_idx[u], col_idx] = 1.0

        H1 = A @ X
        H2 = A @ H1
        R = (X + H1 + H2) > 0
        reach_counts = R.sum(axis=1)
        reachable_nodes = set(idx_to_node[np.where(reach_counts >= 2)[0]])

    return sorted(base_support_nodes.union(reachable_nodes))

