#include "nvmatrix_kernel.cuh"

__global__ void _init_mat(float *m, float val, int len){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < len){
        m[tid] = val;
    }
}

__global__ void _copy_mat(float *m, float* target, int len){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < len){
        target[tid] = m[tid];
    }
}

__global__ void _ele_scale(float *m, float *target, float scaler, int len){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < len){
        target[tid] = scaler * m[tid]; 
    }
}
__global__ void _mat_add(float *ma, float *mb, float *target, float sa, float sb, int len){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < len){
        target[tid] = sa * ma[tid] + sb * mb[tid]; 
    }
}

__global__ void _mat_sum_row_fast(float *m, float *target,int nrow, int ncol, int agg_col){
    int tx = blockIdx.x * blockDim.x + threadIdx.x; 

    __shared__ float accum[NUM_THREAD_PER_ROW];

    if(tx < ncol){
        accum[threadIdx.x] = m[blockIdx.y*ncol+tx];
    }else{
        accum[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    if(NUM_THREAD_PER_ROW >= 512){
        if(threadIdx.x < 256)
            accum[threadIdx.x] += accum[threadIdx.x+256];
        __syncthreads();
    }

    if(NUM_THREAD_PER_ROW >= 256){
        if(threadIdx.x < 128)
            accum[threadIdx.x] += accum[threadIdx.x+128];
        __syncthreads();
    }

    //NUM_THREAD_PER_ROW at least 128
    if(threadIdx.x < 64)
        accum[threadIdx.x] += accum[threadIdx.x+64];
    __syncthreads();

    if(threadIdx.x < 32){
        accum[threadIdx.x] += accum[threadIdx.x+32];
        accum[threadIdx.x] += accum[threadIdx.x+16];
        accum[threadIdx.x] += accum[threadIdx.x+8];
        accum[threadIdx.x] += accum[threadIdx.x+4];
        accum[threadIdx.x] += accum[threadIdx.x+2];
        accum[threadIdx.x] += accum[threadIdx.x+1];
    }
    target[blockIdx.y*agg_col+blockIdx.x] = accum[0];
}

__global__ void _mat_sum_row(float *m, float *target,int nrow, int ncol){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < nrow){
        float sum = 0;
        for(int i = 0; i < ncol; i++){
            sum += m[tid*ncol+i];
        }
        target[tid] = sum;
    }
}

__global__ void _mat_sum_col(float *m, float *target,int nrow, int ncol){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < ncol){
        float sum = 0;
        for(int i = 0; i < nrow; i++){
            sum += m[i*ncol+tid];
        }
        target[tid] = sum;
    }
}
