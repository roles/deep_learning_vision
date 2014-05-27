#include <cuda.h>
#include <cublas.h>
#include <cmath>
#include "nvmatrix.cuh"
#include "nvmatrix_kernel.cuh"

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

NVMatrix::NVMatrix(int nrow, int ncol){
    this->nrow  = nrow;
    this->ncol  = ncol;
    this->own   = true; 
    this->trans = false;
    cudaMalloc((void**)&this->data, nrow * ncol * sizeof(float));
    _init_mat<<<ceil(nrow*ncol/64.0), 64>>>(this->get_data(), 0.0f, nrow*ncol);
    cudaDeviceSynchronize();
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

int NVMatrix::get_ele_num(){
    return this->ncol * this->nrow;
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

void NVMatrix::ele_scale(float scaler, NVMatrix& target){
    int len = nrow * ncol;
    _ele_scale<<<ceil(len / 64.0), 64>>>(this->get_data(), target.get_data(),
            scaler, len);
    cudaDeviceSynchronize();
}

void NVMatrix::ele_scale(float scaler){
    ele_scale(scaler, *this);
}

void NVMatrix::mat_sum(int axis, NVMatrix& target){
    if(axis == 1){      //column sum
        dim3 blocks = dim3(ceil(nrow / 64.0), 1);
        dim3 threads = dim3(64, 1);
        _mat_sum_col<<<blocks, threads>>>(get_data(), target.get_data(), nrow, ncol);
        cudaDeviceSynchronize();
    }else{              //row sum
        /*
        dim3 blocks = dim3(ceil(ncol / 64.0), 1);
        dim3 threads = dim3(64, 1);
        _mat_sum_row<<<blocks, threads>>>(get_data(), target.get_data(), nrow, ncol);
        */
        
        int cur_col = ncol;
        NVMatrix* cur_sum_mat = this;
        while(cur_col > 1){
            int agg_col = ceil(cur_col * 1.0 / NUM_THREAD_PER_ROW);
            NVMatrix* agg_sum_mat = new NVMatrix(nrow,  agg_col);

            dim3 blocks = dim3(agg_col, nrow);
            dim3 threads = dim3(NUM_THREAD_PER_ROW, 1);

            _mat_sum_row_fast<<<blocks, threads>>>(cur_sum_mat->get_data(), agg_sum_mat->get_data(),
                nrow, cur_col, agg_col);

            if(cur_sum_mat != this)
                delete cur_sum_mat;
            delete agg_sum_mat;

            cur_sum_mat = agg_sum_mat;
            cur_col     = agg_col;
        }
        _copy_mat<<<ceil(nrow/64.0), 64>>>(cur_sum_mat->get_data(), target.get_data(), 
                cur_sum_mat->get_ele_num());

        if(cur_sum_mat != this)
            delete cur_sum_mat;
        cudaDeviceSynchronize();
    }
}
