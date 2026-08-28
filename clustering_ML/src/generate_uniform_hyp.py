import numpy as np
import xgi
from math import comb


def generate_hsbm(
    n=300,                      # number of nodes
    m=3,                        # hyperedge size (uniform)
    q=3,                        # number of communities
    n_hyperedges=1000,          # target TOTAL number of hyperedges
    frac_between=0.10,          # proportion of hyperedges spanning >1 community
    sizes=None,                 # community sizes; default = as equal as possible
    seed=1,
    calibrate=True,             # rescale p_in/p_out to actually hit the targets
    calibration_rounds=8,
    attach_within=True,
    edge_file="hyperedges.txt",
    cluster_file="clusters.txt",
    node_offset=1,              # 1 => write nodes as 1..n
    label_offset=1,             # 1 => write communities as 1..q
    verbose=True,
):
    # ---- community sizes and node->community map -------------------------
    if sizes is None:
        base, rem = divmod(n, q)
        sizes = [base + (1 if c < rem else 0) for c in range(q)]
    sizes = list(sizes)
    assert len(sizes) == q and sum(sizes) == n, "sizes must have q entries summing to n"

    labels = np.repeat(np.arange(q), sizes)   # node i (0-indexed) -> community

    # ---- how many candidate m-subsets are within vs between --------------
    n_within  = sum(comb(s, m) for s in sizes)
    n_between = comb(n, m) - n_within
    if n_between == 0:
        raise ValueError("No cross-community m-subsets exist for these sizes.")

    target_within  = n_hyperedges * (1.0 - frac_between)
    target_between = n_hyperedges * frac_between

    # p_in / p_out implied by the two targets
    p_in  = target_within / n_within
    p_out = target_between / n_between
    if p_in > 1 or p_out > 1:
        raise ValueError(
            f"Requested density impossible: p_in={p_in:.3g}, p_out={p_out:.3g}. "
            "Lower n_hyperedges or raise n."
        )

    def build_p(p_in, p_out):
        p = np.full((q,) * m, p_out, dtype=float)
        for c in range(q):
            p[(c,) * m] = p_in
        return p

    def counts(H):
        w = b = 0
        for e in H.edges.members():
            e = list(e)
            if len(e) != m:
                continue
            if len(set(labels[i] for i in e)) == 1:
                w += 1
            else:
                b += 1
        return w, b

    # ---- generate (optionally correcting for XGI's ordered-tuple sampling
    #      and its discarding of degenerate edges) --------------------------
    H = xgi.uniform_HSBM(n, m, build_p(p_in, p_out), sizes, seed=seed)

    if calibrate:
        for _ in range(calibration_rounds):
            w, b = counts(H)
            if w == 0 or b == 0:
                break
            r_in  = target_within / w
            r_out = target_between / b
            if abs(r_in - 1) < 0.02 and abs(r_out - 1) < 0.02:
                break
            p_in  = min(1.0, p_in * r_in)
            p_out = min(1.0, p_out * r_out)
            H = xgi.uniform_HSBM(n, m, build_p(p_in, p_out), sizes, seed=seed)

    # ---- write the two files ---------------------------------------------
    # ---- attach isolated nodes so every node appears in >= 1 hyperedge ----
    edges = [sorted(e) for e in H.edges.members() if len(e) == m]

    rng = np.random.default_rng(seed)
    covered = set().union(*edges) if edges else set()
    isolates = [i for i in range(n) if i not in covered]

    existing = {tuple(e) for e in edges}
    for v in isolates:
        if attach_within:
            pool = [u for u in np.flatnonzero(labels == labels[v]) if u != v]
        else:
            pool = list(np.flatnonzero(labels != labels[v]))
        for _ in range(100):                       # retry on collision
            partners = rng.choice(pool, size=m - 1, replace=False)
            e = tuple(sorted([v, *partners.tolist()]))
            if e not in existing:
                existing.add(e)
                edges.append(list(e))
                break
        else:
            raise RuntimeError(f"could not place isolate {v}")

    if verbose and isolates:
        kind = "within" if attach_within else "between"
        print(f"attached {len(isolates)} isolate(s) via {kind}-community edges: "
              f"{[i + node_offset for i in isolates]}")

    with open(edge_file, "w") as f:
        for e in edges:
            f.write(",".join(str(i + node_offset) for i in e) + "\n")

    with open(cluster_file, "w") as f:
        for i in range(n):                      # every node gets a line, incl. isolates
            f.write(f"{labels[i] + label_offset}\n")

    if verbose:
        w, b = counts(H)
        print(f"p_in = {p_in:.6g}   p_out = {p_out:.6g}")
        print(f"hyperedges: {len(edges)} (target {n_hyperedges})")
        print(f"  within : {w}  ({w / max(len(edges),1):.3f})")
        print(f"  between: {b}  ({b / max(len(edges),1):.3f}, target {frac_between})")
        print(f"wrote {edge_file} and {cluster_file}")

    return H, labels


if __name__ == "__main__":
    edge =r"MY_PATH\clustering_ML\data\synthetic_data\hypernetwork_form\edges\uniform_cluster_id3.txt"
    cluster =r"MY_PATH\clustering_ML\data\synthetic_data\true_clusters\uniform_cluster_id3.txt"

    H, labels = generate_hsbm(
        n=1000, m=3, q=4,
        n_hyperedges=2000,
        frac_between=0.20,
        seed=1,
        edge_file=edge,
        cluster_file=cluster,
    )
