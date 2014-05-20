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

static PyObject*
run(PyObject *self, PyObject *args){
    PyArrayObject *data, *pyfilter;
    int filter_num;
    int filter_size;
    int input_num;
    int input_size;
    int input_group_num;
    int pooling_rate;
    int left_upper_padding, right_low_padding;

    if(!PyArg_ParseTuple(args, "iiiiiiiiO!O!", 
                &filter_num, &filter_size,
                &input_num, &input_size,
                &input_group_num, &pooling_rate,
                &left_upper_padding, &right_low_padding,
                &PyArray_Type, &data,
                &PyArray_Type, &pyfilter))
        return NULL;

    Matrix input_data(data);
    Matrix filter(pyfilter);
    crbm = new CRBM(filter_num, filter_size,
              input_num, input_size, input_group_num,
              left_upper_padding, right_low_padding,
              pooling_rate, NULL, //&filter,
              NULL, NULL, &input_data);

    struct timeval _start_time, _end_time;
    //crbm->CPU_convolution_forward();
    timeFunc(crbm->CPU_convolution_forward(crbm->CPU_input->get_data(),
            crbm->CPU_filters->get_data(), crbm->CPU_y_h->get_data(),
            crbm->CPU_hbias->get_data()), "CPU convolutional forward");
    
    timeFunc(crbm->CPU_max_pooling(crbm->CPU_y_h->get_data(),
            crbm->CPU_y_h_probs->get_data(), crbm->CPU_y_p->get_data()), 
            "CPU max pooling");

    timeFunc(crbm->CPU_convolution_backward(crbm->CPU_y_h_probs->get_data(),
            crbm->CPU_filters->get_data(), crbm->CPU_vbias->get_data(),
            crbm->CPU_y_v_probs->get_data(), crbm->CPU_y_v->get_data()), 
            "CPU convolutional backward");

    timeFunc(crbm->GPU_convolution_forward(crbm->GPU_input->get_data(),
            crbm->GPU_filters->get_data(), crbm->GPU_y_h->get_data(),
            crbm->GPU_hbias->get_data()), "GPU convolutional forward");
    
    timeFunc(crbm->GPU_max_pooling(crbm->GPU_y_h->get_data(),
            crbm->GPU_y_h_probs->get_data(), crbm->GPU_y_p->get_data()), 
            "GPU max pooling");
    
    timeFunc(crbm->GPU_convolution_backward(crbm->GPU_y_h_probs->get_data(),
            crbm->GPU_filters->get_data(), crbm->GPU_vbias->get_data(),
            crbm->GPU_y_v_probs->get_data(), crbm->GPU_y_v->get_data()), 
            "GPU convolutional backward");

    Matrix* tmp_y_h_probs = new Matrix(crbm->CPU_y_h_probs->get_row_num(),
                                 crbm->CPU_y_h_probs->get_col_num());
    crbm->GPU_y_h_probs->assign(*tmp_y_h_probs);
    crbm->CPU_y_h_probs->equal_value(*tmp_y_h_probs);
    delete tmp_y_h_probs;

    Matrix* tmp_y_v_probs = new Matrix(crbm->CPU_y_v_probs->get_row_num(),
                                 crbm->CPU_y_v_probs->get_col_num());
    crbm->GPU_y_v_probs->assign(*tmp_y_v_probs);
    crbm->CPU_y_v_probs->equal_value(*tmp_y_v_probs);
    delete tmp_y_v_probs;

    return Py_BuildValue("i", 0);
}

static PyObject*
test(PyObject *self, PyObject *args){
    return Py_BuildValue("i", 0);
}

static PyMethodDef PyNVcrbmMethods[] = {
    {"test", test, METH_VARARGS, "Test"},
    {"run", run, METH_VARARGS, "Initialize the convolutional RBM"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initnvcrbm(void){
    (void)Py_InitModule("nvcrbm", PyNVcrbmMethods);
    _import_array();
    srand(1234);
    //srand(time(NULL));
}
