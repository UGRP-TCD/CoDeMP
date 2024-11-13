#include <Python.h>
#include <numpy/arrayobject.h>

// Declare the fslic function from the original code
double* fslic(PyObject* arg, int w, int h, int depth, int m, double compactness, int max_iterations, double p, double q);

// Wrapper function to be called from Python
static PyObject* py_fslic(PyObject* self, PyObject* args) {
    PyObject* input_array;
    int w, h, depth, m, max_iterations;
    double compactness, p, q;

    if (!PyArg_ParseTuple(args, "Oiiiiiddd", &input_array, &w, &h, &depth, &m, &max_iterations, &compactness, &p, &q)) {
        return NULL;
    }

    double* result = fslic(input_array, w, h, depth, m, compactness, max_iterations, p, q);

    npy_intp dims[3] = {h, w, depth};
    PyObject* output_array = PyArray_SimpleNewFromData(3, dims, NPY_DOUBLE, result);
    PyArray_ENABLEFLAGS((PyArrayObject*)output_array, NPY_ARRAY_OWNDATA);

    return output_array;
}

static PyMethodDef FslicMethods[] = {
    {"fslic", py_fslic, METH_VARARGS, "Execute FSLIC algorithm"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fslicmodule = {
    PyModuleDef_HEAD_INIT,
    "fslic",
    NULL,
    -1,
    FslicMethods
};

PyMODINIT_FUNC PyInit_fslic(void) {
    import_array();
    return PyModule_Create(&fslicmodule);
}