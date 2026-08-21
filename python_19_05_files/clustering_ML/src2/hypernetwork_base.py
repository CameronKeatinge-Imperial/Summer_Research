from abc import ABC, abstractmethod
from collections import Counter
from math import comb, floor

import xgi


class BaseHypernetworkObject(ABC):

    def __init__(self, file_in):
        self.initialHypernetwork = self.hypernetwork_from_files(file_in)
        self._install_working_state(self._initial_edge_to_nodes, self._nodes)
        # independent copy: removals must not mutate the baseline
        self.itertative_H = xgi.Hypergraph(
            {eid: set(m) for eid, m in self._initial_edge_to_nodes.items()}
        )

        self.previous_partition = None
        self.previous_modularity = None
        self.last_removed_hyp_members = None
        self.wdc = "linear"

        self._build_modularity_invariants()
        self._state_dirty = True
        

    # ------------------------------------------------------------------
    # loading / state
    # ------------------------------------------------------------------
    def hypernetwork_from_files(self, file):
        print("reading hypergraph")
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

        self._initial_edge_to_nodes = {
            eid: frozenset(members) for eid, members in edge_dict.items()
        }
        self._nodes = sorted({n for m in edge_dict.values() for n in m})

        self.initialHypernetwork = xgi.Hypergraph(edge_dict)
        return self.initialHypernetwork

    def _install_working_state(self, edge_to_nodes, nodes):
        """
        Dict-backed mirror of the hypergraph that every shared method below
        reads from. Kept here because `remove_hyperedge`,
        `_compute_contribution` and `_build_modularity_invariants` all depend
        on it.
        """
        self._edge_to_nodes = {
            eid: frozenset(members) for eid, members in edge_to_nodes.items()
        }
        self._nodes = sorted(nodes)
        self._node_to_edges = {n: set() for n in self._nodes}
        for eid, members in self._edge_to_nodes.items():
            for n in members:
                self._node_to_edges[n].add(eid)

    def hyperedge_to_edge_id(self, nodes):
        '''
        Just handle the edge id case in here.
        Check if nodes are alreadya actually passing in a valid edge
        '''
        # Case 1: already a valid edge id
        try:
            if nodes in self.itertative_H.edges:
                return nodes
        except (TypeError, AttributeError):
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

    def remove_hyperedge(self, hyperedge_nodes):
        hyperedge = self.hyperedge_to_edge_id(hyperedge_nodes)
        if hyperedge is None or hyperedge not in self._edge_to_nodes:
            print(f"[Warning] Could not find a hyperedge with exactly these nodes: {hyperedge_nodes}")
            return

        print(f"Removing hyperedge {hyperedge} containing nodes {hyperedge_nodes}")
        members = self._edge_to_nodes.pop(hyperedge)
        self.last_removed_hyp_members = set(members)

        for n in members:
            self._node_to_edges[n].discard(hyperedge)

        self.itertative_H.remove_edge(hyperedge)
        self._state_dirty = True

    def number_of_nodes(self):
        '''
        return number of nodes
        '''
        return self.initialHypernetwork.num_nodes

    # ------------------------------------------------------------------
    # partitions
    # ------------------------------------------------------------------
    def get_partitions(self):
        components = [set(cc) for cc in xgi.connected_components(self.itertative_H)]
        print(f"Number of disconnected components", len(components))
        return components

    def score_partitions(self, partitions):
        return {frozenset(p): self.cluster_contribution(p) for p in partitions}

    def _cluster_size(self, cluster, size_by="nodes"):
        '''
        Size by defines what the smallest hyperedges are, either nodes or volume
        '''
        if size_by == "nodes":
            return len(cluster)
        raise ValueError(f"Unknown size_by: {size_by}")

    # ------------------------------------------------------------------
    # modularity
    # ------------------------------------------------------------------
    def _eta_product(self, d, c):
        """Edge-contribution function, per strict/majority/linear form."""
        if self.wdc == "strict":
            return 1.0 if c == d else 0.0
        if self.wdc == "majority":
            return 1.0 if  c >= floor(d / 2) + 1 else 0.0
        if self.wdc == "linear":
            return c / d if c >= floor(d / 2) + 1 else 0.0
        if self.wdc == "quadratic":
            return (c / d)**2 if c >= floor(d / 2) + 1 else 0.0
        raise ValueError(f"Unknown wdc form: {self.wdc}")

    def _expected_chi(self, d, p):
        if self.wdc == "strict":
            return p ** d
        total = 0.0
        for c in range(floor(d / 2) + 1, d + 1):
            prob = comb(d, c) * (p ** c) * ((1 - p) ** (d - c))
            total += prob * self._eta_product(d, c)
        return total
    
    def _modularity_edges(self, edge_to_nodes=None):
        src = self._initial_edge_to_nodes if edge_to_nodes is None else edge_to_nodes
        return [frozenset(m) for m in src.values() if len(m) >= 2]

    def _build_modularity_invariants(self):
        members_list = self._modularity_edges()

        self._degrees = Counter()
        for members in members_list:
            for n in members:
                self._degrees[n] += 1
        self._vol_total = sum(self._degrees.values())

        self._edge_size_hist = Counter(len(m) for m in members_list)
        self._total_edges = sum(self._edge_size_hist.values())
        self._modularity_members = members_list  # cached, avoids re-filtering per cluster

    def _compute_contribution(self, cluster):
        if getattr(self, "_degrees", None) is None:
            self._build_modularity_invariants()

        cluster = set(cluster)

        observed = 0.0
        for members in self._modularity_members:
            observed += self._eta_product(len(members), len(members & cluster))

        vol_cluster = sum(self._degrees.get(n, 0) for n in cluster)
        p = vol_cluster / self._vol_total if self._vol_total else 0.0

        expected = 0.0
        if self._total_edges:
            for d, count in self._edge_size_hist.items():
                expected += (count / self._total_edges) * self._expected_chi(d, p)

        return (observed / self._total_edges if self._total_edges else 0.0) - expected
        #return observed - expected

    def _modularity_from_edges(self, edge_to_nodes, partitions):
        """
        Whole-partition modularity over an arbitrary edge set -- the drop-in
        replacement for hnx `hmod.modularity`.

        Same quantity: (1/|E|) * sum_e chi(d, c_A(e)) summed over the parts,
        minus the Chung-Lu degree tax sum_d P(d) * E[chi(d, Binom(d, p_A))].
        Written against an explicit `edge_to_nodes` so the caller chooses the
        baseline (initial hypergraph) rather than the working one.
        """
        all_members = self._modularity_edges(edge_to_nodes)

        degrees = Counter()
        for members in all_members:
            for n in members:
                degrees[n] += 1
        vol_total = sum(degrees.values())

        size_hist = Counter(len(m) for m in all_members)
        total_edges = sum(size_hist.values())
        if not total_edges:
            return 0.0

        modularity = 0.0
        for part in partitions:
            part = set(part)

            observed = 0.0
            for members in all_members:
                observed += self._eta_product(len(members), len(members & part))
                
            vol_part = sum(degrees.get(n, 0) for n in part)
            p = vol_part / vol_total if vol_total else 0.0

            expected = 0.0
            for d, count in size_hist.items():
                expected += (count / total_edges) * self._expected_chi(d, p)

            modularity += observed / total_edges - expected

        return modularity

    def run_iteration(self, target_number, size_by="nodes"):
        partitions = self.get_partitions()
        self.score_partitions(partitions)
        if len(partitions) > target_number:
            partitions = self.attach_clusters(partitions, target_number, size_by)
        return partitions

    def attach_clusters(self, partition, target_number, size_by="nodes"):
            """
            Same as attach clusters in the other code.
            """
            partition = [set(c) for c in partition]
            print(sorted(self._cluster_size(c, size_by) for c in partition))

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

    def calculate_modularity(self, partitions):
        if partitions == self.previous_partition:
            return self.previous_modularity
        else:
            new_modularity = self._modularity_from_edges(
                self._initial_edge_to_nodes, partitions
            )
            print(f"new_modularity", new_modularity)
            self.previous_partition = partitions
            self.previous_modularity = new_modularity
            return new_modularity
        
    def cluster_contribution(self, cluster):
        return self._compute_contribution(frozenset(cluster))

    def optimal_attach_clusters(self, clusters, target_number=None):
        """
        Exhaustive counterpart to `attach_clusters`: given 2n clusters, score
        every way of merging them into exactly n and return the best.

        Scoring sums per-block `cluster_contribution`, which is exact because
        sum_i cluster_contribution(A_i) == calculate_modularity(A) on the
        baseline hypergraph. Contributions are cached per block: there are at
        most 2^len(clusters) - 1 distinct blocks but far more partitions
        built from them, so each merged set is evaluated once.
        """
        clusters = [set(c) for c in clusters]
        m = len(clusters)
        if target_number is None:
            target_number = m // 2
        if not 1 <= target_number <= m:
            raise ValueError(f"target_number must be in [1, {m}], got {target_number}")

        # S(m, target_number) -- how many candidates are about to be scored
        stirling = [1] + [0] * target_number
        for _ in range(m):
            for j in range(target_number, 0, -1):
                stirling[j] = j * stirling[j] + stirling[j - 1]
            stirling[0] = 0
        print(f"scoring {stirling[target_number]} partitions of {m} clusters into {target_number}")

        cache = {}

        def block_score(block):
            """block: tuple of indices into `clusters`, ascending"""
            score = cache.get(block)
            if score is None:
                merged = set()
                for i in block:
                    merged |= clusters[i]
                score = self.cluster_contribution(merged)
                cache[block] = score
            return score

        blocks = []
        best_score, best_blocks = float("-inf"), None

        def recurse(i):
            nonlocal best_score, best_blocks
            if i == m:
                if len(blocks) == target_number:
                    score = sum(block_score(tuple(b)) for b in blocks)
                    if score > best_score:
                        best_score, best_blocks = score, [tuple(b) for b in blocks]
                return
            if len(blocks) + (m - i) < target_number:
                return  # too few items left to open the remaining blocks

            # cluster i joins an existing block ...
            for b in blocks:
                b.append(i)
                recurse(i + 1)
                b.pop()
            # ... or opens the next one. Blocks stay ordered by their smallest
            # member, so each set partition is generated exactly once.
            if len(blocks) < target_number:
                blocks.append([i])
                recurse(i + 1)
                blocks.pop()

        recurse(0)

        best = [set().union(*(clusters[i] for i in b)) for b in best_blocks]
        print(f"best modularity {best_score}")
        return best