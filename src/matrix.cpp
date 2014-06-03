#include "matrix.h"
#include "utils.h"
#include <iostream>
#include <cstring>

using namespace std;

Matrix::Matrix(PyArrayObject *pyarr){
    this->nrow = PyArray_DIM(pyarr, 0);
    this->ncol = PyArray_DIM(pyarr, 1);
    /*
    if(pyarr->flags & NPY_ARRAY_C_CONTIGUOUS){
        this->data = reinterpret_cast<float*>(pyarr->data);
        this->own = false;
    }else{*/
  
        this->data = new float[this->nrow * this->ncol];
        for(int i = 0; i < this->nrow; i++){
            for(int j = 0; j < this->ncol; j++){
                (*this)(i, j) = *reinterpret_cast<float*>(PyArray_GETPTR2(pyarr, i, j));
            }
        }
        this->own = true;
    //}
    this->trans = false;
}

Matrix::Matrix(int nrow, int ncol, float low, float upper){
    this->nrow  = nrow;
    this->ncol  = ncol;
    this->own   = true;
    this->trans = false;
    this->data  = new float[nrow * ncol];
    for(int i = 0; i < nrow; i++){
        for(int j = 0; j < ncol; j++){
            (*this)(i, j) = random_float(low, upper);
        }
    }
}

Matrix::Matrix(int nrow, int ncol){
    this->nrow  = nrow;
    this->ncol  = ncol;
    this->own   = true;
    this->trans = false;
    this->data  = new float[nrow * ncol];
    for(int i = 0; i < nrow; i++){
        for(int j = 0; j < ncol; j++){
            (*this)(i, j) = 0;
        }
    }
}

Matrix::Matrix(Matrix& target){
    this->nrow = target.nrow;
    this->ncol = target.ncol;
    this->own   = true;
    this->trans = target.trans; 
    this->data  = new float[nrow * ncol];
    memcpy(this->data, target.get_data(), nrow * ncol * sizeof(float)); 
}

Matrix::~Matrix(){
    if(this->own)
        delete this->data;
}

float* Matrix::get_data(){
    return this->data;
}

int Matrix::get_row_num(){
    return this->nrow;
}

int Matrix::get_col_num(){
    return this->ncol;
}

int Matrix::get_ele_num(){
    return nrow * ncol;
}

bool Matrix::get_trans(){
    return this->trans;
}

void Matrix::mat_init(float val){
    for(int i = 0; i < nrow; i++){
        for(int j = 0; j < ncol; j++){
            (*this)(i, j) = val;
        }
    }
}

bool Matrix::equal_value(Matrix &target){
    this->equal_value(target, 1e-3);
}

bool Matrix::equal_value(Matrix &target, float e){
    assert(this->nrow == target.get_row_num() &&
           this->ncol == target.get_col_num());
    for(int i = 0; i < this->nrow; i++)
        for(int j = 0; j < this->ncol; j++){
            if(!float_equal((*this)(i, j), target(i,j), e)){
                cout << "this(" << i << j << "):" << (*this)(i, j) << endl;
                cout << "target(" << i << j << "):" << target(i, j) << endl;
                return false;
            }
        }

    cout << "same" << endl;
    return true;
}

void Matrix::ele_add(float val){
    ele_add(val, *this);
}

void Matrix::ele_add(float val, Matrix& target){
    for(int i = 0; i < this->nrow; i++)
        for(int j = 0; j < this->ncol; j++){
            target(i, j) = (*this)(i, j) + val;
        }
}

void Matrix::ele_scale(float scaler){
    ele_scale(scaler, *this);
}

void Matrix::ele_scale(float scaler, Matrix& target){
    for(int i = 0; i < this->nrow; i++)
        for(int j = 0; j < this->ncol; j++){
            target(i, j) = (*this)(i, j) * scaler;
        }
}

void Matrix::mat_sum(int axis, Matrix& target){
    if(axis == 0){
        for(int i = 0; i < this->nrow; i++){
            float sum = 0.0;
            for(int j = 0; j < this->ncol; j++){
                sum += (*this)(i, j);
            }
            target(i, 0) = sum;
        }
    }else{
        for(int i = 0; i < this->ncol; i++){
            float sum = 0.0;
            for(int j = 0; j < this->nrow; j++){
                sum += (*this)(j, i);
            }
            target(0, i) = sum;
        }
    }
}

void Matrix::mat_add(Matrix& m, float sb){
    mat_add(m, *this, 1.0, sb);
}

void Matrix::mat_add(Matrix& m, Matrix& target, float sa, float sb){
    for(int i = 0; i < this->nrow; i++)
        for(int j = 0; j < this->ncol; j++){
            target(i, j) = sa * (*this)(i, j) + sb * m(i, j);
        }
}

void Matrix::assign(Matrix& target){
    for(int i = 0; i < this->nrow; i++)
        for(int j = 0; j < this->ncol; j++){
            target(i, j) = (*this)(i, j);
        }
}

float Matrix::ele_mean(){
    float mean = 0.0f;
    for(int i = 0; i < this->nrow; i++)
        for(int j = 0; j < this->ncol; j++){
            mean += (*this)(i, j);
        }
    mean /= get_ele_num();
    return mean;
}

void Matrix::mat_mul(Matrix& m, Matrix& target){
    for(int i = 0; i < this->nrow; i++)
        for(int j = 0; j < this->ncol; j++){
            target(i, j) = (*this)(i, j) * m(i, j);
        }
}

void Matrix::mat_mul(Matrix& m){
    mat_mul(m, *this);
}

void Matrix::reshape(int nrow, int ncol){
    this->nrow = nrow;
    this->ncol = ncol;
}
