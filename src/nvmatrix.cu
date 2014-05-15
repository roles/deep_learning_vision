#include "nvmatrix.cuh"
#include <cuda.h>

using namespace std;

NVMatrix::NVMatrix(Matrix &m){
    this->nrow  = m.get_row_num();
    this->ncol  = m.get_col_num();
    this->own   = true; 
    this->trans = m.get_trans();
    cudaMalloc((void**)&this->data, nrow * ncol * sizeof(float));
    cudaMemcpy(this->data, m.get_data(), nrow * ncol * sizeof(float),
            cudaMemcpyHostToDevice);
}

NVMatrix::~NVMatrix(){
    cudaFree(this->data);
}

float* NVMatrix::get_data(){
    return this->data;
}

int NVMatrix::get_row_num(){
    return this->nrow;
}

int NVMatrix::get_col_num(){
    return this->ncol;
}

bool NVMatrix::get_trans(){
    return this->trans;
}

void NVMatrix::assign(Matrix &target){
    assert(this->nrow == target.get_row_num() 
        && this->ncol == target.get_col_num());
    cudaMemcpy(target.get_data(), this->data, nrow * ncol * sizeof(float),
            cudaMemcpyDeviceToHost);
}
