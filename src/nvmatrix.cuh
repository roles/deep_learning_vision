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
        int get_ele_num();
        bool get_trans();
        float* get_data();

        void assign(Matrix&);
        void ele_scale(float scaler, NVMatrix& target);
        void ele_scale(float scaler);
        void mat_sum(int axis, NVMatrix &target);
};

#endif
