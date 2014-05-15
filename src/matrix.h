#ifndef _MATRIX_H
#define _MATRIX_H

extern "C"{
#include "Python.h"
#include "arrayobject.h"
}

class Matrix {
    private:
        int nrow, ncol;
        bool trans;
        bool own;
        float *data;
        void _init(int, int, float*);
    public:
        Matrix(PyArrayObject *);
        Matrix(int nrow, int ncol, float low, float upper);
        Matrix(int nrow, int ncol);
        Matrix(Matrix&);
        ~Matrix();
        inline float& operator()(int i, int j){
            return this->data[i * this->ncol + j];
        }
        int get_row_num();
        int get_col_num();
        bool get_trans();
        float* get_data();
        bool equal_value(Matrix&);
};

#endif
