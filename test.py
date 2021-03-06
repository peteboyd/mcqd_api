#!/usr/bin/env python
from copy import deepcopy
import numpy as np
from scipy.spatial import distance
import sys
import itertools
sys.path.append("/home/pboyd/codes_in_development/mcqd_api/build/lib.linux-x86_64-2.7")
import _mcqd as mcqd

def random(size=50):
    #WARNING the dtype MUST be np.int32 for the array to be passed correctly
    # to the _mcqd api... not sure why this is.
    test_graph = np.random.randint(0,2,(size,size))
    return mcqd.maxclique(np.array(test_graph, dtype=np.int32), size)

def ones(size=50):
    return mcqd.maxclique(np.ones((size, size), dtype=np.int32), size)

def test_clq():
    with open('test.clq', 'r') as f:
        first_line = f.readline()
        graph_size = int(first_line.split()[2])
        #WARNING the dtype MUST be np.int32 for the array to be passed correctly
        # to the _mcqd api... not sure why this is.
        edge_matrix = np.zeros((graph_size, graph_size), dtype=np.int32)
        for line in f:
            ind1 = int(line.split()[1])-1
            ind2 = int(line.split()[2])-1
            edge_matrix[ind1][ind2] = 1
            edge_matrix[ind2][ind1] = 1

        print edge_matrix.size
    return mcqd.maxclique(edge_matrix, graph_size)
#check = ones(99)
#check = random(11644)
#check = test_clq() 
#print check
def corr():
    atoms1 = ["C", "O", "C", "O"]
    a1 = [1, 2, 3, 4]
    #coord1 = np.random.random((len(atoms1),3))*np.random.randint(3)
    #dist1 = distance.cdist(coord1,coord1)
    atoms2 = ["C", "O", "C", "C", "O", "O", "H", "C", "C"]
    a2 = [1, 2, 0, 0, 3, 0, 0, 4, 0]

    adj = [1 if i==j else 0 for (i,j) in itertools.product(a1, a2)]
    size = len(list(itertools.product(atoms1, atoms2)))
    q = np.zeros((size, size))

    for ind, i in enumerate(q):
        for indj, j in enumerate(i):
            q[ind][indj] = adj[ind]*adj[indj]
            
    q = q - np.identity(len(adj))
    #coord2 = np.random.random((len(atoms2),3))*np.random.randint(3)
    #dist2 = distance.cdist(coord2,coord2)
    #p = mcqd.correspondence(atoms1, atoms2)
    #q = mcqd.correspondence_edges(p,dist1, dist2, 0.1)
    #print mcqd.maxclique(q, q.size)
    print mcqd.maxclique(q, len(adj))
corr()

