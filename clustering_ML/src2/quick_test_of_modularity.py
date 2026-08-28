import pandas as pd
import hypernetx as hnx
import hypernetx.algorithms.hypergraph_modularity as hmod

print(hnx.__version__)
print(pd.__version__)

edges = {
    "e1": {"A", "B", "C"},
    "e2": {"B", "C"},
    "e3": {"D", "E"},
    "e4": {"C", "D"}
}

H = hnx.Hypergraph(edges)

partition = [
    {"A", "B", "C"},
    {"D", "E"}
]

Q_linear = hmod.modularity(H, partition, wdc=hmod.linear)
Q_strict = hmod.modularity(H, partition, wdc=hmod.strict)
Q_majority = hmod.modularity(H, partition, wdc=hmod.majority)

print(Q_linear)
print(Q_strict)
print(Q_majority)