"""
Verification for the A1-A8 rewrite of src2/hypernetwork_MMOT.py.

Run from anywhere:
    python -B check_mmot.py
or point it at a different checkout:
    python -B check_mmot.py <path-to-clustering_ML>

Writes nothing except a temporary toy hypergraph, which it deletes on exit.
`-B` stops Python writing .pyc files into your repo (optional).

Sections marked [EQUIV] are the ones that matter: they assert the rewritten
code returns the same answers as the original. Sections marked [INFO] just
exercise a code path and print numbers for you to eyeball.
"""

import os
import random
import sys
import tempfile
import traceback

DEFAULT_ROOT = r"C:\Users\ckeat\github-projects\Summer_Research\python_19_05_files\clustering_ML"
ROOT = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT
sys.path.insert(0, ROOT)

import numpy as np
from scipy.sparse.csgraph import dijkstra

from src2.hypernetwork_MMOT import (
    MMOTHypernetworkObject, get_local_matrix_for_calc, UNREACHABLE,
)

failures = []


def check(name, condition, detail=""):
    if condition:
        print(f"  [PASS] {name}")
    else:
        failures.append(f"{name} {detail}")
        print(f"  [FAIL] {name} {detail}")


# ----------------------------------------------------------------------
print("\n[EQUIV] A1 -- barycentre cost block")
# The rewrite replaces an n x n x N tensor with (K o C) @ v. Assert the two
# are numerically identical on random inputs.
rng = np.random.default_rng(0)
worst = 0.0
for _ in range(5):
    n, N = 40, 6
    C = rng.integers(0, 5, size=(n, n)).astype(float)
    C = np.minimum(C, C.T)
    np.fill_diagonal(C, 0.0)
    K = np.exp(-C / 0.5)
    u = rng.random((n, N))
    v = rng.random((n, N))

    old = (u * np.einsum('jl,jlk->jk', K, C[:, :, None] * v[None, :, :])).sum(axis=0)
    new = (u * ((K * C) @ v)).sum(axis=0)
    worst = max(worst, float(np.max(np.abs(old - new) / np.maximum(np.abs(old), 1e-30))))
check("old einsum == new gemm", worst < 1e-12, f"(max rel err {worst:.2e})")


# ----------------------------------------------------------------------
# toy hypergraph: 120 hyperedges of size 2-5 over 60 nodes
random.seed(1)
lines = []
for _ in range(120):
    k = random.choice([2, 2, 3, 3, 4, 5])
    lines.append(",".join(str(x) for x in random.sample(range(1, 61), k)))

fd, toy = tempfile.mkstemp(suffix="_mmot_toy_edges.txt", text=True)
with os.fdopen(fd, "w") as f:
    f.write("\n".join(lines) + "\n")

try:
    obj = MMOTHypernetworkObject([toy, None])
    obj._build_state()

    # ------------------------------------------------------------------
    print("\n[EQUIV] A4 -- support construction")
    # Original: dense |V|-vectors and |V| x k matrices via A @ x.
    # Rewrite:  CSR frontier expansion. Must return identical node sets.
    def old_support(nodes_e, measures, A, node_to_idx, idx_to_node, lazy):
        Nn = A.shape[0]
        base = set(nodes_e)
        for i in nodes_e:
            base.update(measures[i].keys())
        if lazy:
            x = np.zeros(Nn)
            for u_ in nodes_e:
                x[node_to_idx[u_]] = 1.0
            h1 = A @ x
            h2 = A @ h1
            reach = set(idx_to_node[np.where((x + h1 + h2) > 0)[0]].tolist())
        else:
            k = len(nodes_e)
            X = np.zeros((Nn, k))
            for c, u_ in enumerate(nodes_e):
                X[node_to_idx[u_], c] = 1.0
            H1 = A @ X
            H2 = A @ H1
            R = (X + H1 + H2) > 0
            reach = set(idx_to_node[np.where(R.sum(axis=1) >= 2)[0]].tolist())
        return sorted(base | reach)

    for lazy in (True, False):
        bad = []
        for eid in list(obj._edge_to_nodes)[:40]:
            ne = sorted(obj._edge_to_nodes[eid])
            a = get_local_matrix_for_calc(ne, obj.measures, obj.clique_expansion,
                                          obj.node_to_idx, obj.idx_to_node, lazy=lazy)
            b = old_support(ne, obj.measures, obj.clique_expansion,
                            obj.node_to_idx, obj.idx_to_node, lazy=lazy)
            if a != b:
                bad.append((eid, set(a) ^ set(b)))
        check(f"frontier support == dense support (lazy={lazy})",
              not bad, f"{len(bad)} mismatched edges, first: {bad[:1]}")

    # ------------------------------------------------------------------
    print("\n[EQUIV] A3 -- distance block")
    # Original: |U|-source Dijkstra over the whole graph.
    # Rewrite:  same Dijkstra restricted to the ball of radius floor(R/2)
    #           around U. The claim is that this is exact, not approximate.
    A = obj.clique_expansion
    bad = []
    for eid in list(obj._edge_to_nodes)[:30]:
        ne = sorted(obj._edge_to_nodes[eid])
        U = get_local_matrix_for_calc(ne, obj.measures, A, obj.node_to_idx,
                                      obj.idx_to_node, lazy=True)
        new = obj.dist.submatrix(U)

        iu = np.array([obj.node_to_idx[u_] for u_ in U])
        D = dijkstra(csgraph=A, directed=False, indices=iu,
                     unweighted=True, limit=obj.radius)[:, iu]
        ref = np.full(D.shape, UNREACHABLE, dtype=np.uint8)
        fin = np.isfinite(D)
        ref[fin] = D[fin].astype(np.uint8)

        if not np.array_equal(new, ref):
            bad.append((eid, int((new != ref).sum())))
    check("local-ball block == global Dijkstra block",
          not bad, f"{len(bad)} mismatched edges, first: {bad[:1]}")

    # ------------------------------------------------------------------
    print("\n[INFO] end-to-end curvature")
    curv = obj.get_network_curvature()
    nodec = obj.hypernetwork_nodes_curv()
    finite = np.array([v for v in curv.values() if np.isfinite(v)])
    nonfinite = sum(1 for v in curv.values() if not np.isfinite(v))
    if finite.size:
        print(f"  {len(curv)} hyperedges, {len(nodec)} nodes")
        print(f"  curvature range [{finite.min():.4f}, {finite.max():.4f}], "
              f"mean {finite.mean():.4f}")
        print(f"  non-finite (disconnected support, A6): {nonfinite}")
    else:
        check("some finite curvature produced", False, "(all values non-finite)")

    # ------------------------------------------------------------------
    print("\n[INFO] removal path")
    edict = obj.return_edge_dict()
    n_before = len(edict)
    target = list(edict)[0]
    obj.remove_hyperedge(list(edict[target]))
    upd = obj.update_neighbourhood_scores(obj.last_removed_hyp_members)
    print(f"  removed {target}; {len(upd)} neighbourhood curvatures recomputed")
    print(f"  working edges: {n_before} -> {len(obj._edge_to_nodes)}")
    check("baseline hypergraph not mutated by removal",
          len(obj._initial_edge_to_nodes) == n_before,
          f"({len(obj._initial_edge_to_nodes)} vs {n_before})")

    # ------------------------------------------------------------------
    print("\n[INFO] clustering / modularity")
    try:
        parts = obj.run_iteration(4, size_by="nodes")
        mod = obj.calculate_modularity(parts)
        print(f"  {len(parts)} partitions, modularity = {mod:.6f}")
    except Exception:
        print("  modularity path raised:")
        traceback.print_exc(limit=3)
        failures.append("calculate_modularity raised")

    # ------------------------------------------------------------------
    print("\n[INFO] LP vs IBP at reg=0.5 (A2 calibration)")
    # Entropic regularisation biases costs upward, so IBP curvature sits below
    # the exact LP answer. This prints the size of that bias so you can decide
    # whether reg=0.5 is acceptable, or sweep it.
    try:
        obj2 = MMOTHypernetworkObject([toy, None])
        obj2.additional_parameters(reg=0.5, lp_threshold=10**9)  # force LP
        obj2._build_state()
        sample = list(obj2._edge_to_nodes)[:15]
        lp = np.array([obj2.compute_edge(e).curvature for e in sample])
        ibp = np.array([curv[e] for e in sample])
        d = np.abs(lp - ibp)
        print(f"  {len(sample)} edges: max |LP - IBP| = {np.nanmax(d):.4f}, "
              f"mean = {np.nanmean(d):.4f}")
        print(f"  signed mean (LP - IBP) = {np.nanmean(lp - ibp):+.4f} "
              f"(positive => IBP under-reports curvature, as expected)")
    except Exception:
        print("  LP comparison raised (POT version may lack ot.lp.barycenter):")
        traceback.print_exc(limit=3)

finally:
    try:
        os.unlink(toy)
    except OSError:
        pass

# ----------------------------------------------------------------------
print("\n" + "=" * 60)
if failures:
    print(f"{len(failures)} CHECK(S) FAILED:")
    for f_ in failures:
        print(f"  - {f_}")
    sys.exit(1)
print("ALL EQUIVALENCE CHECKS PASSED")