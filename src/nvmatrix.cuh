#ifndef _NVMATRIX_H
#define _NVMATRIX_H

#include "matrix.h"

class NVMatrix {
    private:
        int nrow, ncol;
        bool trans;
        bool own;
        float *data;
    public:
        NVMatrix(Matrix&);
        NVMatrix(int nrow, int ncol);
        ~NVMatrix();
        inline float& operator()(int i, int j){
            return this->data[i * this->ncol + j];
        }
        int get_row_num();
        int get_col_num();
        bool get_trans();
        float* get_data();

        void assign(Matrix&);
};

#endif
