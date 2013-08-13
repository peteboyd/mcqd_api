#!/usr/bin/env python

import numpy as np
import sys
sys.path.append("/home/pboyd/codes_in_development/mcqd_api/build/lib.linux-x86_64-2.7")
import _mcqd as mcqd

def random(size=50):
    #WARNING the dtype MUST be np.int32 for the array to be passed correctly
    # to the _mcqd api... not sure why this is.
    test_graph = np.random.randint(0,2,(size,size))
    mcqd.maxclique(np.array(test_graph, dtype=np.int32), size)

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

    mcqd.maxclique(edge_matrix, graph_size)

random(56)
