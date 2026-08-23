"""
MMOT hypergraph curvature.

Used Generative Tools to help with optimisation

Shared topology/modularity code lives in
`src2.hypernetwork_base.BaseHypernetworkObject`.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import os

import warnings
import numpy as np
import scipy.sparse as sp
import ot
import xgi
from scipy.sparse.csgraph import connected_components, dijkstra
from src2.hypernetwork_base import BaseHypernetworkObject

# distance code used for "further apart than `radius`" in the uint8 blocks
UNREACHABLE = 255 #replacement for infinity
SAFE_RADIUS = 5

@dataclass
class EdgeBarycenterResult:
    edge_id: object
    nodes: list
    support: list
    costs: np.ndarray
    mean_cost: float
    curvature: float
    method: str
    degenerate: bool = False
    n_support: int = 0


# ----------------------------------------------------------------------
# graph primitives
# ----------------------------------------------------------------------

def _bfs_ball(indptr, indices, source_idx, depth):
    """
    Global indices within `depth` hops of any source, as a sorted int array.

    Frontier expansion over the CSR arrays (A4). Touches only the edges inside
    the ball -- nothing proportional to |V| is allocated.
    """
    frontier = np.unique(np.asarray(source_idx, dtype=np.int64))
    visited = frontier
    for _ in range(int(depth)):
        if frontier.size == 0:
            break
        starts = indptr[frontier]
        ends = indptr[frontier + 1]
        if int((ends - starts).sum()) == 0:
            break
        nbrs = np.unique(np.concatenate(
            [indices[s:e] for s, e in zip(starts, ends)]
        )).astype(np.int64)
        frontier = nbrs[np.isin(nbrs, visited, assume_unique=True, invert=True)]
        if frontier.size:
            visited = np.union1d(visited, frontier)
    return visited


class HypergraphDistance:
    """
    BFS distances on the clique expansion, capped at `radius`.
    """

    def __init__(self, A, node_to_idx, idx_to_node, radius=5):
        self.A = A.tocsr()
        self.node_to_idx = node_to_idx
        self.idx_to_node = idx_to_node
        self.radius = int(radius) # radius is the cap on distance measures
        #by having it equal 5, it means on the subnetworks everything is calculated
        self.indptr = self.A.indptr
        self.indices = self.A.indices

    def _block(self, idx_U):
        """uint8 |U|x|U| distance block over the given global indices."""
        idx_U = np.asarray(idx_U, dtype=np.int64)
        ball = _bfs_ball(self.indptr, self.indices, idx_U, self.radius // 2)
        sub = self.A[ball][:, ball]
        local_U = np.searchsorted(ball, idx_U)

        D = dijkstra(csgraph=sub, directed=False, indices=local_U,
                     unweighted=True, limit=self.radius)
        D = D[:, local_U]

        out = np.full(D.shape, UNREACHABLE, dtype=np.uint8)
        finite = np.isfinite(D)
        out[finite] = D[finite].astype(np.uint8)
        return out

    def __call__(self, u, v):
        if u == v:
            return 0.0
        block = self._block([self.node_to_idx[u], self.node_to_idx[v]])
        d = block[0, 1]
        return np.inf if d == UNREACHABLE else float(d)

    def submatrix(self, U):
        return self._block([self.node_to_idx[u] for u in U])


def build_sparse_adjacency(edge_to_nodes, nodes=None):
    """
    Clique-expansion adjacency plus the co-occurrence count matrix.

    Returns (A, W, node_to_idx, idx_to_node). W holds integer co-occurrence
    counts (how many hyperedges contain both endpoints); A is its boolean
    pattern. W is not used by the current path -- it is what an incremental
    removal update would decrement instead of rebuilding.
    """
    if nodes is None:
        nodes = {n for members in edge_to_nodes.values() for n in members}
    nodes_list = sorted(nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes_list)}
    idx_to_node = np.array(nodes_list)

    rows, cols = [], []
    for col, members in enumerate(edge_to_nodes.values()):
        for n in members:
            rows.append(node_to_idx[n])
            cols.append(col)

    incidence = sp.coo_matrix(
        (np.ones(len(rows), dtype=np.int32), (rows, cols)),
        shape=(len(nodes_list), max(len(edge_to_nodes), 1)),
    ).tocsr()

    W = (incidence @ incidence.T).tocsr()
    W.setdiag(0)
    W.eliminate_zeros()

    A = W.copy()
    A.data = np.ones_like(A.data)

    return A, W, node_to_idx, idx_to_node


# ----------------------------------------------------------------------
# node distributions
# ----------------------------------------------------------------------

def _single_node_distribution(edge_to_nodes, node_to_edges, i, alpha=0.01):
    """mu_i for one node, from the dict-backed state (A7)."""
    incident_edges = node_to_edges.get(i, ())
    deg = len(incident_edges)

    if deg == 0:
        return {i: 1.0}

    mu_i = {i: alpha} if alpha > 0.0 else {}
    share_per_edge = (1.0 - alpha) / deg

    for e in incident_edges:
        members = edge_to_nodes[e]
        n_others = len(members) - 1
        if n_others <= 0:
            raise ValueError(
                f"Node {i} has a singleton incident hyperedge {e} "
                f"(only member). Distribution is undefined for this "
                f"case per current assumptions."
            )
        w = share_per_edge / n_others
        for o in members:
            if o != i:
                mu_i[o] = mu_i.get(o, 0.0) + w

    return mu_i


def node_distributions(edge_to_nodes, node_to_edges, alpha=0.01):
    """Build mu_i for every node."""
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    return {
        i: _single_node_distribution(edge_to_nodes, node_to_edges, i, alpha)
        for i in node_to_edges
    }


def get_local_matrix_for_calc(nodes_e, measures, A, node_to_idx, idx_to_node,
                              lazy=True):
    """
    Support for a hyperedge: its members, the supports of their measures, and
    the 2-hop reachable set. Frontier expansion (A4) -- no |V|-sized arrays.
    """
    base_support_nodes = set(nodes_e)
    for i in nodes_e:
        base_support_nodes.update(measures[i].keys())

    indptr, indices = A.indptr, A.indices
    src = np.fromiter((node_to_idx[u] for u in nodes_e),
                      dtype=np.int64, count=len(nodes_e))

    if lazy:
        # any node within two steps of the hyperedge
        reach = _bfs_ball(indptr, indices, src, 2)
    else:
        # only nodes reached from at least two distinct members
        per_source = [_bfs_ball(indptr, indices, np.array([s]), 2) for s in src]
        if per_source:
            vals, counts = np.unique(np.concatenate(per_source),
                                     return_counts=True)
            reach = vals[counts >= 2]
        else:
            reach = np.empty(0, dtype=np.int64)

    reachable_nodes = set(idx_to_node[reach].tolist()) if reach.size else set()
    return sorted(base_support_nodes | reachable_nodes)


# ----------------------------------------------------------------------
# barycentre solvers
# ----------------------------------------------------------------------


def ibp_barycenter_with_costs(A, C, reg, weights, maxiter=500, tol=1e-6,
                              K=None, M=None):
    """
    Iterative Bregman projection barycentre.

    K and M = K o C may be supplied by the caller (built by lookup table, A5).
    The final cost block is (u * (M @ v)).sum(axis=0) -- algebraically the same
    as the old einsum over an n x n x N tensor, but O(n^2) memory and a single
    BLAS gemm (A1).
    """
    n, N = A.shape

    if K is None or M is None:
        Cf = np.asarray(C, dtype=float)
        finite = np.isfinite(Cf)
        Csafe = np.where(finite, Cf, 0.0)
        K = np.where(finite, np.exp(-Csafe / reg), 0.0)
        M = np.where(finite, K * Csafe, 0.0)

    eps = 1e-300
    u = np.ones((n, N))
    v = np.array(A, dtype=float, copy=True)
    beta = np.full(n, 1.0 / n)

    for _ in range(maxiter):
        # K is symmetric (C is), so K.T @ u is just K @ u (A5)
        v = A / np.maximum(K @ u, eps)
        Kv = K @ v
        log_beta = (np.log(np.maximum(Kv, eps)) * weights).sum(axis=1)
        beta_new = np.exp(log_beta)

        total = beta_new.sum()
        if not np.isfinite(total) or total <= 0.0:
            break
        beta_new /= total

        converged = np.max(np.abs(beta_new - beta)) < tol
        beta = beta_new
        u = beta[:, None] / np.maximum(Kv, eps)
        if converged:
            break

    costs = (u * (M @ v)).sum(axis=0)
    return beta, costs

def lp_barycenter_with_costs(A, C, weights=None, solver="highs-ds"):
    """
    Exact LP barycentre. Same contract as `ibp_barycenter_with_costs`:
    returns (beta, costs) with costs[k] = <pi_k, C> for source k, unweighted.

    Solves one joint LP over all N plans plus the free barycentre, so there is
    no `reg` and no entropic blur. Cost is N*n^2 + n variables -- gate on n.
    `highs-ds` (dual simplex) returns a vertex solution; `highs-ipm` returns the
    analytic centre of the optimal face, which smears beta over ties.
    """
    n, N = A.shape
    C = np.ascontiguousarray(C, dtype=float)
    if not np.isfinite(C).all():
        raise ValueError("LP barycentre requires a finite cost matrix")

    beta, sol = ot.lp.barycenter(A, C, weights=weights, log=True, solver=solver)

    if getattr(sol, "status", 0) != 0:
        raise RuntimeError(f"LP barycentre did not solve: {getattr(sol, 'message', '')}")

    # sol.x lays out the N plans row-major, then beta in the trailing n slots.
    # sol.fun is the *weighted* objective; recover per-source costs to match IBP.
    n2 = n * n
    x = np.asarray(sol.x)
    costs = np.array([
        float(np.sum(x[k * n2:(k + 1) * n2].reshape(n, n) * C))
        for k in range(N)
    ])
    return np.ascontiguousarray(beta), costs
# ----------------------------------------------------------------------
# process pool worker (A8)
# ----------------------------------------------------------------------

_WORKER = None
_LIMITS = None          # keep the limiter alive for the process lifetime

def _worker_init(edge_to_nodes, nodes, params, blas_threads=1):
    global _WORKER, _LIMITS
    try:
        from threadpoolctl import threadpool_limits
        _LIMITS = threadpool_limits(blas_threads)
    except ImportError:
        pass
    _WORKER = MMOTHypernetworkObject.__new__(MMOTHypernetworkObject)
    _WORKER._install_working_state(edge_to_nodes, nodes)
    _WORKER.itertative_H = None
    _WORKER.last_removed_hyp_members = None
    _WORKER.additional_parameters(**params)
    _WORKER._build_state()


def _worker_edge(eid):
    return eid, _WORKER.compute_edge(eid)


class MMOTHypernetworkObject(BaseHypernetworkObject):
    def __init__(self, file_in):
        super().__init__(file_in)
        self.additional_parameters()

    def additional_parameters(self, reg: float = 0.1, maxiter: int = 500,
                              tol: float = 1e-6,
                              lazy_support: bool = False, radius: int = 5,
                              alpha: float = 0.01, n_jobs: int = 1,
                              lp_solver: str = "highs-ds",
                              lp_strict: bool = False):
        self.wdc = "linear"
        self.reg = reg
        self.maxiter = maxiter
        self.tol = tol
        self.lazy_support = lazy_support
        self.radius = int(radius)
        self.alpha = alpha
        self.n_jobs = n_jobs

        self.measures = None
        self.clique_expansion = None
        self.cooccurrence = None
        self.dist = None
        self.node_to_idx = None
        self.idx_to_node = None
        self._c_lut = None # the cost itself - 0.0, 1.0, 2.0, 3.0
        self._k_lut = None # the Gibbs kernel - 1.0, 4.5e-05, 2.1e-09, 9.4e-14
        self._m_lut = None # product kernel - 0.0, 4.5e-05, 4.1e-09, 2.8e-13
        self._degrees = None
        self._state_dirty = True
        self._sentinel_possible = self.radius < SAFE_RADIUS
        self.lp_solver = lp_solver
        self.lp_strict = lp_strict
        self.use_lp_param = False

    def _tuning_params(self):
        return dict(reg=self.reg, maxiter=self.maxiter, tol=self.tol,
                    lp_solver=self.lp_solver, lp_strict=self.lp_strict,
                    lazy_support=self.lazy_support, radius=self.radius,
                    alpha=self.alpha, n_jobs=1)

    def return_edge_dict(self):
        return dict(self._edge_to_nodes)

    def update_neighbourhood_scores(self, removed_nodes=None) -> list:
        """
        Recompute curvature for every hyperedge sharing a node with the removed
        one, and return [[edge_id, curvature], ...].

        This is the mu-dirty set only. Removing a hyperedge also changes
        distances, which can alter the curvature of hyperedges sharing no node
        with it; those are not captured here.
        """
        if not removed_nodes:
            if removed_nodes is None:
                print("Done")
            return []

        affected = set()
        for n in removed_nodes:
            affected |= self._node_to_edges.get(n, set())

        if self.dist is None or self._state_dirty:
            self._build_state()

        return [[eid, self.compute_edge(eid).curvature] for eid in sorted(affected)]

    # ---------------- state ----------------

    def _build_state(self):
        """Rebuild distributions, adjacency, distance helper and lookup tables."""
        print("executing _build_state")
        self.measures = node_distributions(
            self._edge_to_nodes, self._node_to_edges, self.alpha
        )
        (self.clique_expansion, self.cooccurrence,
         self.node_to_idx, self.idx_to_node) = build_sparse_adjacency(
            self._edge_to_nodes, self._nodes
        )
        self.dist = HypergraphDistance(
            self.clique_expansion, self.node_to_idx, self.idx_to_node, self.radius
        )
        self._build_luts()
        self._build_modularity_invariants()
        self._state_dirty = False

    def update_state(self, last_removed_nodes):
        """
        Placeholder for the incremental update (co-occurrence decrement,
        decremental distances, dirty-set propagation). Not implemented --
        `remove_hyperedge` marks the state dirty and it is rebuilt in full.
        """
        self._state_dirty = True

    def _build_luts(self):
        """
        A5: C takes values in {0..R} plus a sentinel, so K = exp(-C/reg) and
        M = K o C have at most R+2 distinct entries each. Index a table rather
        than calling exp on n^2 floats.
        """
        R = self.radius
        d = np.arange(R + 1, dtype=float)
        k = np.exp(-d / self.reg)
        self._k_lut = np.append(k, 0.0)
        self._m_lut = np.append(k * d, 0.0)
        # the LP path cannot take inf, so cap the sentinel one step past R
        self._c_lut = np.append(d, float(R + 1))

    def _kernels(self, C_u8):
        if self._sentinel_possible:
            codes = np.where(C_u8 == UNREACHABLE, self.radius + 1,
                            C_u8).astype(np.int64)
        else:
            codes = C_u8.astype(np.int64)
        return self._k_lut[codes], self._m_lut[codes], self._c_lut[codes]

    # ---------------- curvature ----------------

    def _restrict_to_member_component(self, support, C_u8, nodes_e):
        """
        A6: the solver cannot handle a support that splits under the radius
        cap. Drop support nodes outside the members' component; if the members
        themselves span more than one component, report it rather than letting
        zeros and NaNs propagate through IBP.
        """
        finite = (C_u8 != UNREACHABLE)
        n_comp, labels = connected_components(
            sp.csr_matrix(finite), directed=False
        )
        if n_comp <= 1:
            return support, C_u8, True

        pos = {u: k for k, u in enumerate(support)}
        member_labels = {labels[pos[i]] for i in nodes_e}
        if len(member_labels) > 1:
            return support, C_u8, False

        keep = np.where(labels == member_labels.pop())[0]
        return ([support[k] for k in keep],
                C_u8[np.ix_(keep, keep)], True)

    def compute_edge(self, e) -> EdgeBarycenterResult:
        """Barycentre result for a single hyperedge."""
        if self.dist is None or self._state_dirty:
            self._build_state()

        nodes_e = sorted(self._edge_to_nodes[e])
        N = len(nodes_e)

        if N <= 1:
            return EdgeBarycenterResult(
                edge_id=e, nodes=nodes_e, support=nodes_e,
                costs=np.array([]), mean_cost=0.0, curvature=0.0,
                method="trivial", degenerate=False, n_support=len(nodes_e),
            )

        support = get_local_matrix_for_calc(
            nodes_e=nodes_e, measures=self.measures, A=self.clique_expansion,
            node_to_idx=self.node_to_idx, idx_to_node=self.idx_to_node,
            lazy=self.lazy_support,
        )

        C_u8 = self.dist.submatrix(support)

        if self._sentinel_possible:
            support, C_u8, connected = self._restrict_to_member_component(
                support, C_u8, nodes_e
            )
            if not connected:
                return EdgeBarycenterResult(
                    edge_id=e, nodes=nodes_e, support=support,
                    costs=np.array([]), mean_cost=float("nan"),
                    curvature=float("nan"), method="disconnected",
                    degenerate=True, n_support=len(support),
                )
        else:
            assert not (C_u8 == UNREACHABLE).any(), (
                f"edge {e}: sentinel at radius={self.radius} >= {SAFE_RADIUS}; "
                f"the support-diameter bound has been broken"
            )

        idx = {u: k for k, u in enumerate(support)}
        n = len(support)

        A_marginals = np.zeros((n, N))
        for col, i in enumerate(nodes_e):
            for target, w in self.measures[i].items():
                A_marginals[idx[target], col] = w

        col_mass = A_marginals.sum(axis=0)
        if np.any(np.abs(col_mass - 1.0) > 1e-9):
            A_marginals /= np.maximum(col_mass, 1e-300)

        K, M, C = self._kernels(C_u8)
        weights = np.full(N, 1.0 / N)

        # A sentinel means "unreachable". IBP encodes that as K=0 (forbidden);
        # _c_lut caps it at R+1, which the LP would happily route through at
        # finite cost. Different problems -- so refuse the LP if any survive.
        has_sentinel = self._sentinel_possible and bool((C_u8 == UNREACHABLE).any())
        use_lp = self.use_lp_param and not has_sentinel

        method = "ibp"
        if use_lp:
            try:
                beta, costs = lp_barycenter_with_costs(
                    A_marginals, C, weights, solver=self.lp_solver,
                )
                method = "lp"
            except Exception as exc:
                if self.lp_strict:
                    raise
                warnings.warn(f"edge {e}: LP barycentre failed ({exc}); using IBP")
                use_lp = False

        if not use_lp:
            beta, costs = ibp_barycenter_with_costs(
                A_marginals, C, self.reg, weights,
                maxiter=self.maxiter, tol=self.tol, K=K, M=M,
            )

        '''
        before adding the snippet above
        beta, costs = ibp_barycenter_with_costs(
            A_marginals, C, self.reg, weights,
            maxiter=self.maxiter, tol=self.tol, K=K, M=M,
        )
        method = "ibp"
        '''

        return EdgeBarycenterResult(
            edge_id=e, nodes=nodes_e, support=support,
            costs=costs, mean_cost=float(costs.mean()),
            curvature=float(1 - (costs.sum() / (N - 1))),
            method=method, degenerate=False, n_support=n,
        )

    def compute_all(self) -> dict:
        print(f"Using lp:", self.use_lp_param)
        self._build_state()
        edges = list(self._edge_to_nodes)

        n_jobs = self.n_jobs
        if n_jobs == -1:
            try:
                n_jobs = len(os.sched_getaffinity(0))
            except AttributeError:          # not available on macOS/Windows
                n_jobs = os.cpu_count() or 1
        if n_jobs > 1 and len(edges) > 1:
            return self._compute_all_parallel(edges, n_jobs)

        return {e: self.compute_edge(e) for e in edges}

    def _compute_all_parallel(self, edges, n_jobs, blas_threads=1):
        """A8: hyperedges are independent given fixed state."""
        # cost is O(maxiter * n^2 * N); dispatch the expensive edges first
        # so the stragglers overlap with the rest of the queue
        edges = sorted(edges, key=lambda e: -len(self._edge_to_nodes[e]))

        results = {}
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_worker_init,
            initargs=(dict(self._edge_to_nodes), list(self._nodes),
                      self._tuning_params(), blas_threads),
        ) as pool:
            for eid, result in pool.map(_worker_edge, edges, chunksize=1):
                results[eid] = result
        return results

    def get_network_curvature(self) -> dict:
        """Curvature for every hyperedge; also stored on self."""
        results = self.compute_all()
        self.edge_curvatures = {eid: r.curvature for eid, r in results.items()}
        self.edge_results = results
        return self.edge_curvatures

    def hypernetwork_nodes_curv(self) -> dict:
        """Sum of incident hyperedge curvatures, per node."""
        if not hasattr(self, "edge_curvatures") or self.edge_curvatures is None:
            raise RuntimeError(
                "self.edge_curvatures not found. Call get_network_curvature() "
                "before hypernetwork_nodes_curv()."
            )

        node_curvatures = {
            node: sum(self.edge_curvatures[eid]
                      for eid in self._node_to_edges.get(node, ()))
            for node in self._nodes
        }
        self.node_curvatures = node_curvatures
        return node_curvatures