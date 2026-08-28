from pathlib import Path
from collections import defaultdict
from itertools import combinations
from math import comb
import xgi

# Output folder, relative to wherever you run this from
OUT_DIR = Path("data/synthetic_data/hypernetwork_form")
FILE_NAME = "ID_H.txt"
EDGE_FILE = OUT_DIR / "edges" / FILE_NAME
NODE_FILE = OUT_DIR / "nodes" / FILE_NAME

OUT_DIR.mkdir(parents=True, exist_ok=True)

def hypergraph_from_counts(n, counts):
    """Wrapper for fast random hypergraph"""
    ps = []
    for size, m in enumerate(counts, start=2):
        total = comb(n, size)
        if m > total:
            raise ValueError(f"can't fit {m} edges of size {size}; only {total} exist")
        ps.append(m / total)
    return xgi.fast_random_hypergraph(n, ps, seed=1)    

def get_adjacency(hypergraph):
    #additional information
    pair_to_edges = defaultdict(list)
    for e, nodes in hypergraph.edges.members(dtype=dict).items():
        for pair in combinations(sorted(nodes), 2):
            pair_to_edges[pair].append(e)

    shared = {e for edges in pair_to_edges.values() if len(edges) > 1 for e in edges}

    print(f"{len(shared)} of {hypergraph.num_edges} edges share 2+ nodes with another edge")

###############
# TWO CHOICES #
###############

H = hypergraph_from_counts(400, [200,200,200,200,200])
# H = xgi.watts_strogatz_hypergraph(n=200, d=3, k=4, l=1, p=0.5, seed=1)

# One hyperedge per line, nodes space-separated
with open(EDGE_FILE, "w") as f:
    for edge in H.edges.members():
        f.write(" ".join(str(node) for node in sorted(edge)) + "\n")

# One node per line
with open(NODE_FILE, "w") as f:
    for node in sorted(H.nodes):
        f.write(f"{node}\n")

print(f"Wrote {H.num_edges} hyperedges to {EDGE_FILE}")
print(f"Wrote {H.num_nodes} nodes to {NODE_FILE}")

#additional information
get_adjacency(H)