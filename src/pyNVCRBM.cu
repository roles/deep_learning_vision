extern "C"{
#include <Python.h>
}
#include <iostream>
#include "matrix.h"
#include "nvmatrix.cuh"
#include "crbm.cuh"
#include <cuda.h>
#include <ctime>
#include <cstdlib>
#include "utils.h"

using namespace std;

static CRBM* crbm = NULL;

static PyArrayObject* copy_host_matrix(Matrix& src){
    PyArrayObject* ret;
    int dims[2];

    dims[0] = src.get_row_num();
    dims[1] = src.get_col_num();
    ret = (PyArrayObject*)PyArray_FromDims(2, dims, NPY_FLOAT);

    memcpy(ret->data, src.get_data(), sizeof(float) * src.get_ele_num());
    
    return ret;
}

static PyObject*
init(PyObject *self, PyObject *args){
    PyArrayObject *pyfilter;
    PyArrayObject *pyinit_filter, *pyinit_hbias, *pyinit_vbias;
    int filter_num;
    int filter_size;
    int input_num;
    int input_size;
    int input_group_num;
    int pooling_rate;
    int left_upper_padding, right_low_padding;

    if(!PyArg_ParseTuple(args, "iiiiiiiiO!O!O!", 
                &filter_num, &filter_size,
                &input_num, &input_size,
                &input_group_num, &pooling_rate,
                &left_upper_padding, &right_low_padding,
                &PyArray_Type, &pyinit_filter,
                &PyArray_Type, &pyinit_hbias,
                &PyArray_Type, &pyinit_vbias))
        return NULL;

    Matrix init_filter(pyinit_filter);
    Matrix init_hbias(pyinit_hbias);
    Matrix init_vbias(pyinit_vbias);

    crbm = new CRBM(filter_num, filter_size,
              input_num, input_size, input_group_num,
              left_upper_padding, right_low_padding,
              pooling_rate, &init_filter, //&filter,
              &init_hbias, &init_vbias);

    pyfilter = copy_host_matrix(*crbm->CPU_filters);

    return PyArray_Return(pyfilter);
    //return Py_BuildValue("i", 0);
}

static PyObject*
run_batch(PyObject *self, PyObject *args){
    PyArrayObject *pybatch_data;
    int cur_trail, cur_image, cur_batch;

    if(!PyArg_ParseTuple(args, "iiiO!", 
        &cur_trail, &cur_image, &cur_batch,
        &PyArray_Type, &pybatch_data)){
        return NULL;
    }
    Matrix batch_data(pybatch_data);

    crbm->run_batch(cur_trail, cur_image, cur_batch, batch_data);
    
    return Py_BuildValue("i", 0);
}

static PyObject*
get_gpu_filters(PyObject *self, PyObject *args){
    PyArrayObject *pyfilter;

    Matrix *tmp_filter = new Matrix(*crbm->CPU_filters);
    crbm->GPU_filters->assign(*tmp_filter);
    pyfilter = copy_host_matrix(*tmp_filter);
    delete tmp_filter;

    return PyArray_Return(pyfilter);
}

static PyObject*
get_cpu_filters(PyObject *self, PyObject *args){
    PyArrayObject *pyfilter;

    pyfilter = copy_host_matrix(*crbm->CPU_filters);

    return PyArray_Return(pyfilter);
}

static PyObject*
get_gpu_hbias(PyObject *self, PyObject *args){
    PyArrayObject *pyhbias;

    Matrix *tmp_hbias = new Matrix(*crbm->CPU_hbias);
    crbm->GPU_hbias->assign(*tmp_hbias);
    pyhbias = copy_host_matrix(*tmp_hbias);
    delete tmp_hbias;

    return PyArray_Return(pyhbias);
}

static PyObject*
print_result(PyObject *self, PyObject *args){
    cout << "ferr : " << crbm->ferr / 20.0 << endl;
    cout << "sparsity : " << crbm->sparsity / 20.0 << endl;
    crbm->ferr = 0.0;
    crbm->sparsity = 0.0;
    return Py_BuildValue("i", 0);
}

static PyMethodDef PyNVcrbmMethods[] = {
    {"get_gpu_filters", get_gpu_filters, METH_VARARGS, "Get the filter weight matrix"},
    {"get_cpu_filters", get_cpu_filters, METH_VARARGS, "Get the filter weight matrix"},
    {"get_gpu_hbias", get_gpu_hbias, METH_VARARGS, "Get gpu hidden layer bias"},
    {"print_result", print_result, METH_VARARGS, "print result"},
    {"run_batch", run_batch, METH_VARARGS, "Run a batch"},
    {"init", init, METH_VARARGS, "Initialize the convolutional RBM"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initnvcrbm(void){
    (void)Py_InitModule("nvcrbm", PyNVcrbmMethods);
    _import_array();
    srand(1234);
    //srand(time(NULL));
}
