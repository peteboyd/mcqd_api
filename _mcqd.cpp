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
    PyObject *x;
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
    int i;
    //declare a list of pointers to pointers (dynamic 2x2 array)
    //Maxclique m(conn, size);
    //m.mcq(qmax, qsize);  // run max clique with improved coloring
    Maxclique md(conn, size, 0.025);  //(3rd parameter is optional - default is 0.025 - this heuristics parameter enables you to use dynamic resorting of vertices (time expensive)
    // on the part of the search tree that is close to the root - in this case, approximately 2.5% of the search tree -
    // you can probably find a more optimal value for your graphs
    md.mcqdyn(qmax, qsize);  // run max clique with improved coloring and dynamic sorting of vertices
    ret_array = PyList_New(qsize);
     
    for (i=0; i<qsize; i++){
        x = PyInt_FromLong((long) qmax[i]);
        PyList_SetItem(ret_array, i, x);
    }
    return ret_array;
};

//Computes the correspondence graphs for a pair of arrays.
static PyObject * correspondence(PyObject * self, PyObject *args)
{
    PyObject* elem1;
    PyObject* elem2;
    PyObject* pair = PyTuple_New(2); 
    PyObject *nodes = PyList_New(0);
    if (!PyArg_ParseTuple(args, "OO",
                          &elem1,
                          &elem2
                          )){
        return NULL;
    };
    int size1, size2, i, j;
    size1 = PySequence_Length(elem1);
    size2 = PySequence_Length(elem2);
    
    //Converted all the data, now create correspondence and adjacency matrix.
    //declare a list of pointers to pointers (dynamic 2x2 array)
    for (i=0; i < size1; i++){
        for (j=0; j< size2; j++){
            //Here have to compare atoms with strings greater than 1
            //Try some PyString comparison methods?
            PyObject * py_atom1 = PyList_GetItem(elem1, i);
            PyObject * py_atom2 = PyList_GetItem(elem2, j);
            if (PyObject_RichCompareBool(py_atom1, py_atom2, Py_EQ) == 1){
                //add the pair to the correspondence graph
                pair = Py_BuildValue("(ii)", (i), (j));
                PyList_Append(nodes, pair);
                Py_DECREF(pair);
            }
        }
    }
    return nodes;
    //return Py_BuildValue("O", nodes); 
}

static PyObject * correspondence_edges(PyObject * self, PyObject *args){
    //Adjacency matrix stuff
    PyArrayObject* dist1;
    PyArrayObject* dist2;
    PyArrayObject* adj_array = NULL;
    PyObject* node1;
    PyObject* node2;
    PyObject* nodes;
    PyObject* tol;
    PyObject* val;
    PyObject *py_di, *py_dj;
    npy_intp dims[2];
    int i, j;
    double di, dj;
    Py_ssize_t zero = 0;
    Py_ssize_t one = 1;
    long i_1, i_2, j_1, j_2;
    void *adj_ptr, *dist1_ptr, *dist2_ptr;
    char *adj_charptr, *dist1_charptr, *dist2_charptr;
    if (!PyArg_ParseTuple(args, "OOOO",
                          &nodes,
                          &dist1,
                          &dist2,
                          &tol
                          )){
        return NULL;
    };
    int inc = PySequence_Length(nodes);
    //ensure that inc is > 0
    if (inc == 0){
        Py_INCREF(Py_None);
        return Py_None;
    }
    dims[0] = (npy_intp) inc;
    dims[1] = (npy_intp) inc;
    adj_array = (PyArrayObject*) PyArray_ZEROS(2, dims, NPY_INT, 0);
    //Mem check return a 'python None' value if the allocation failed.
    if (adj_array == NULL){
        Py_XDECREF(adj_array);
        Py_INCREF(Py_None);
        return Py_None;
    }

    for (i=0; i<inc; i++){
        for (j=i+1; j<inc; j++){
            //if (i != j){
            //get node indices as pyintegers,
            //get distances from the py objects dist1 and dist2 
            //instead of dist
            node1 = PyList_GET_ITEM(nodes, i);
            node2 = PyList_GET_ITEM(nodes, j);
            i_1 = PyInt_AS_LONG(PyTuple_GET_ITEM(node1, zero));
            i_2 = PyInt_AS_LONG(PyTuple_GET_ITEM(node2, zero));
            j_1 = PyInt_AS_LONG(PyTuple_GET_ITEM(node1, one));
            j_2 = PyInt_AS_LONG(PyTuple_GET_ITEM(node2, one));
            
            if ((i_1 != i_2) && (j_1 != j_2)){
                dist1_ptr = PyArray_GETPTR2(dist1, i_1, i_2);
                //dist1_charptr = (char*) dist1_ptr;
                py_di = PyArray_GETITEM(dist1, (char*) dist1_ptr);
                //di = PyFloat_AS_DOUBLE(py_di);
                dist2_ptr = PyArray_GETPTR2(dist2, j_1, j_2);
                //dist2_charptr = (char*) dist2_ptr;
                py_dj = PyArray_GETITEM(dist2, (char*) dist2_ptr);
                //dj = PyFloat_AS_DOUBLE(py_dj);
                //if (std::abs(di-dj) < tol){
                PyObject* diff = PyNumber_Subtract(py_di, py_dj);
                PyObject* abs_diff = PyNumber_Absolute(diff);
                if (PyObject_RichCompareBool(abs_diff, tol, Py_LT) == 1){
                    adj_ptr = PyArray_GETPTR2(adj_array, i, j);
                    //adj_charptr = (char*) adj_ptr;
                    val = PyInt_FromLong(1);
                    PyArray_SETITEM(adj_array, (char*) adj_ptr, val);
                    Py_DECREF(val);
                    adj_ptr = PyArray_GETPTR2(adj_array, j, i);
                    //adj_charptr = (char*) adj_ptr;
                    val = PyInt_FromLong(1);
                    PyArray_SETITEM(adj_array, (char*) adj_ptr, val);
                    Py_DECREF(val);
                }
                Py_DECREF(py_di);
                Py_DECREF(py_dj);
                Py_DECREF(diff);
                Py_DECREF(abs_diff);
            }
        }
    }
    
    return PyArray_Return(adj_array);
}
