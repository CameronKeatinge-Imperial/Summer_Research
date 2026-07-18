import os
import networkx as nx
from itertools import combinations
from datetime import datetime
from pathlib import Path
import time
import sys
start = time.time()

def process_and_save_poset(data_source,  dataset_name):
    data_source_path = Path(data_source)

    input_1 = os.path.join(data_source_path, "hypernetwork_form", "nodes", f"{dataset_name}.txt")
    input_2 = os.path.join(data_source_path, "hypernetwork_form", "edges", f"{dataset_name}.txt")
    output1_file = os.path.join(data_source_path, "poset_complex", "nodes", f"{dataset_name}.txt")
    output3_file = os.path.join(data_source_path, "poset_complex", "edges", f"{dataset_name}.txt")
    output4_file = os.path.join(data_source_path, "poset_complex", "cardinality", f"{dataset_name}.txt")
    output5_file = os.path.join(data_source_path, "poset_complex", "triangles", f"{dataset_name}.txt")
    output6_file = os.path.join(data_source_path, "poset_complex", "hyperedge_node_key", f"{dataset_name}.txt")

    output1 = open(output1_file, 'w')
    output3 = open(output3_file, 'w')
    output4 = open(output4_file, 'w')
    output5 = open(output5_file, 'w')
    output6 = open(output6_file, 'w')
    # Now you can access them via list indexing:
    # outputs[0] is your old output1, outputs[1] is output2, etc.

    #create dictionaries of new nodes (old hyperedges) and their cardinalities, each node indexed by whole numbers
    print(input_2)
    print('%fs\tReading hyperedge files'%(0))
    nodes = {}
    k_values = {}
    with open(input_1,'r') as f:
        count = 0
        for line in f:
            nodes[count] = {int(line.strip())}
            k_values[count] = 1
            output1.write(str(count)+'\n')
            output4.write(str(k_values[count])+'\n')
            count+=1
    with open(input_2,'r') as f:
        for line in f:
            nodes[count] = set(map(int,line.split()))
            k_values[count] = len(nodes[count])
            output1.write(str(count)+'\n')
            output4.write(str(k_values[count])+'\n')

            #added by me
            output6.write(f"{count} : {','.join(map(str, sorted(nodes[count])))}\n")
            count+=1

    print('%fs\tGenerating poset complex'%(time.time()-start))

    k_max = max(k_values.values())
    hyperedges = {k:[i for i,j in k_values.items() if j == k] for k in range(1,k_max+1)}

    parent = [set()  for i in nodes]
    print('%fs\tSize of parent list: %dbytes'%(time.time()-start,sys.getsizeof(parent)))

    for k in range(k_max,1,-1):
        for i in hyperedges[k]:
            for d in range(1,k):
                temp = [j for j in hyperedges[k-d] if nodes[j].issubset(nodes[i])]
                for j in temp:
                    parent[j].add(i)
                    parent[j]=parent[j].difference(parent[i])

    print('%fs\tPoset generated. Writing to output files'%(time.time()-start))
    #print to output2, output3 and output5
    for i,val in enumerate(parent):
        new_edges = set()
        faces = set()
        for j in val:
            output3.write(str(i)+'\t'+str(j)+'\n')
            for k in parent[j]:
                new_edges.add((i,k))
                faces.add((i,j,k))
        for item in new_edges: output3.write(str(item[0])+'\t'+str(item[1])+'\n')
        for item in faces: output5.write(str(item[0])+'\t'+str(item[1])+ '\t' + str(item[2])+'\n')

    output1.close()
    output3.close()
    output4.close()
    output5.close()
    output6.close()

    end = time.time()
    print('%fs\tDone'%(end-start))
    print('=======================================================')
