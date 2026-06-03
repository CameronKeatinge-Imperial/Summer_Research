'''
simpler version

Input files:-
arg1 -> nodelist of hypergraph
arg2 -> hyperedges of the hypergraph

Output files:-
arg3 -> each line contains the ID of a node in the poset complex
arg4 -> REMOVED
arg5 -> 2-dimensional poset complex (in edgelist format)
arg6 -> cardinalities of nodes
arg7 -> REMOVED
'''
#when tested, returns the correct number of nodes corresponding to edges.
import time
import sys

start = time.time()

output1 = open(sys.argv[3], 'w')
output3 = open(sys.argv[4], 'w')
output4 = open(sys.argv[5], 'w')

print(sys.argv[2])
print('%fs\tReading files and generating bipartite graph' % (0))

raw_to_internal = {}
count = 0

with open(sys.argv[1], 'r') as f:
    for line in f:
        if not line.strip():
            continue
        raw_node = int(line.strip())
        
        raw_to_internal[raw_node] = count
        
        output1.write(f"{count}\n")
        output4.write("1\n")
        count += 1

with open(sys.argv[2], 'r') as f:
    for line in f:
        if not line.strip():
            continue
        raw_nodes = list(map(int, line.split()))
        
        output1.write(f"{count}\n")
        output4.write(f"{len(raw_nodes)}\n")
        
        for raw_node in raw_nodes:
            base_id = raw_to_internal.get(raw_node)
            if base_id is not None:
                output3.write(f"{base_id}\t{count}\n")          
        count += 1

output1.close()
output3.close()
output4.close()

end = time.time()
print('%fs\tDone' % (end - start))
print('=======================================================')