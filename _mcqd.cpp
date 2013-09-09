#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <stdio.h>
#include <set>
#include <string>
#include <map>
#include <cmath>
#include <numpy/arrayobject.h>
#include <mcqd.h>

static PyObject * mxclique(PyObject *self, PyObject *args);
static PyObject * correspondence(PyObject *self, PyObject *args);
static PyObject * correspondence_edges(PyObject *self, PyObject *args);

static PyMethodDef functions[] = {
    {"maxclique", mxclique, METH_VARARGS},
    {"correspondence", correspondence, METH_VARARGS},
    {"correspondence_edges", correspondence_edges, METH_VARARGS},
    {NULL, NULL, 0, NULL} //Not sure why I need this but will not import properly otherwise
};

PyMODINIT_FUNC init_mcqd(void)
{
    Py_InitModule("_mcqd", functions);
    import_array();
    return;
}

static PyObject * mxclique(PyObject *self, PyObject *args)
{
    PyArrayObject* conn;
    PyObject *ret_array;
    int size;
    int *qmax;
    int qsize;
    if (!PyArg_ParseTuple(args, "Oi",
                          &conn,
                          &size)){
        return NULL;
    };
    /* Convert numpy array of edge matrix to boolean array 
     * WARNING: the depreciated numpy api allows access to 
     * ndarray internals so including the definition
     * #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION 
     * breaks this program as it has no access to ->data.
     */
    //const double (*pp)[size] = (double(*)[size])conn->data;
    //const int (*pp)[size] = (int(*)[size])conn->data; #include <numpy/arrayobject.h>
    const int (*pp)[size] = (int(*)[size])PyArray_DATA(conn);
    int i, j;
    //declare a list of pointers to pointers (dynamic 2x2 array)
    bool** e = new bool*[size];
    for (i=0; i<size; i++){
        e[i] = new bool[size];
        memset(e[i], 0, size *sizeof(bool));
    }
    for (i=0; i < size; i++){
       for (j=0; j<size; j++){
           if (pp[i][j] == 1.){
               e[i][j] = true;
               e[j][i] = true;
           }
           //std::cout<<pp[i][j];
           //std::cout<<e[i][j];
       }
       //std::cout<<std::endl;
    }
    
    Maxclique m(e, size);
    m.mcq(qmax, qsize);  // run max clique with improved coloring
    //delete [] qmax;
    //Maxclique md(e, size, 0.025);  //(3rd parameter is optional - default is 0.025 - this heuristics parameter enables you to use dynamic resorting of vertices (time expensive)
    // on the part of the search tree that is close to the root - in this case, approximately 2.5% of the search tree -
    // you can probably find a more optimal value for your graphs
    //md.mcqdyn(qmax, qsize);  // run max clique with improved coloring and dynamic sorting of vertices
    ret_array = PyList_New(qsize);
     
    for (i=0; i<qsize; i++){
        PyList_SetItem(ret_array, i, PyInt_FromLong((long) qmax[i]));
    }

    return ret_array; 
};

//Computes the correspondence graphs for a pair of arrays.
static PyObject * correspondence(PyObject * self, PyObject *args)
{
    PyObject* elem1;
    PyObject* elem2;
    PyObject* attr;
    PyObject* pair = PyTuple_New(2); 
    PyObject *nodes = PyList_New(0);
    if (!PyArg_ParseTuple(args, "OO",
                          &elem1,
                          &elem2
                          )){
        return NULL;
    };
    int size1, size2, i, j, inc;
    size1 = PySequence_Length(elem1);
    size2 = PySequence_Length(elem2);
    const char* atoms1[size1];
    const char* atoms2[size2];
    
    for (i=0; i < size1; i++){
        attr = PyList_GetItem(elem1, i); // now convert to c++ string
        atoms1[i] = PyString_AsString(attr);
        //std::cout<<atoms1[i]<<std::endl;
    }
    for (i=0; i < size2; i++){
        attr = PyList_GetItem(elem2, i); // now convert to c++ string
        atoms2[i] = PyString_AsString(attr);
        //std::cout<<atoms2[i]<<std::endl;
    }
    //Converted all the data, now create correspondence and adjacency matrix.
    //declare a list of pointers to pointers (dynamic 2x2 array)
    int max = size1*size2;
    inc = 0;
    for (i=0; i < size1; i++){
        for (j=0; j< size2; j++){
            if ((*atoms1[i]) == (*atoms2[j])){
                //add the pair to the correspondence graph
                //ipo = PyInt_FromLong(i);
                //jpo = PyInt_FromLong(j);
                pair = Py_BuildValue("(ii)", (i), (j));
                //PyTuple_SetItem(pair, 0, ipo);
                //PyTuple_SetItem(pair, 1, jpo);
                PyList_Append(nodes, pair);
                Py_DECREF(pair);
                inc++;
                //std::cout<<i<<" "<<j<<std::endl;
            }
        }
    }

    return nodes;
}

static PyObject * correspondence_edges(PyObject * self, PyObject *args){
    //Adjacency matrix stuff
    PyArrayObject* dist1;
    PyArrayObject* dist2;
    PyObject* value;
    PyObject* node1;
    PyObject* node2;
    PyObject* nodes;
    double tol;
    if (!PyArg_ParseTuple(args, "OOOd",
                          &nodes,
                          &dist1,
                          &dist2,
                          &tol
                          )){
        return NULL;
    };
    int inc = PySequence_Length(nodes);
    int i, j;
    const int size1 = PyArray_DIM(dist1, 0);
    const int size2 = PyArray_DIM(dist2,0); 
    double (*d1)[size1] = (double(*)[size1])PyArray_DATA(dist1);
    double (*d2)[size2] = (double(*)[size2])PyArray_DATA(dist2);

    int** adj = new int*[inc];
    for (i=0; i<inc; i++){
        adj[i] = new int[inc];
        memset(adj[i], 0, inc *sizeof(int));
    }
    npy_intp dims[2];
    PyArrayObject * adj_array;
    //PyObject * value;
    dims[0] = (npy_intp) inc;
    dims[1] = (npy_intp) inc;
    double di, dj;
    int *i_1, *i_2, *j_1, *j_2;
    void *arrptr;
    char *charptr;
    adj_array = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_INT);
    //void **vp = (void**)adj_array->data;
    for (i=0; i<inc; i++){
        for (j=0; j<inc; j++){
            if (i != j){
                node1 = PyList_GET_ITEM(nodes, i);
                node2 = PyList_GET_ITEM(nodes, j);
                i_1 = (int(*))PyTuple_GetItem(node1, 0);
                i_2 = (int(*))PyTuple_GetItem(node2, 0);
                j_1 = (int(*))PyTuple_GetItem(node1, 1);
                j_2 = (int(*))PyTuple_GetItem(node2, 1);
                //Py_DECREF(node1);
                //Py_DECREF(node2);
                di = d1[(*i_1)][(*i_2)];
                dj = d2[(*j_1)][(*j_2)];
                if (std::abs(di-dj) < tol){
                    adj[i][j] = 1;
                    adj[j][i] = 1;
                    value = PyInt_FromLong(1);
                    arrptr = PyArray_GETPTR2(adj_array, i, j);
                    charptr = (char*) arrptr;
                    PyArray_SETITEM(adj_array, charptr, value);
                }
                else{
                    value = PyInt_FromLong(0);
                    arrptr = PyArray_GETPTR2(adj_array, i, j);
                    charptr = (char*) arrptr;
                    PyArray_SETITEM(adj_array, charptr, value);
                    adj[i][j] = 0;
                    adj[j][i] = 0;
                }
            }
            else{
                value = PyInt_FromLong(0);
                arrptr = PyArray_GETPTR2(adj_array, i, j);
                charptr = (char*) arrptr;
                PyArray_SETITEM(adj_array, charptr, value);
                adj[i][j] = 0;
                adj[j][i] = 0;
            }
        }
    }
    //adj_array = PyArray_FromDimsAndData(2, dims, NPY_INT32, (char*) adj);
    return PyArray_Return(adj_array); 
}
