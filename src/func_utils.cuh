#ifndef _FUNC_UITLS_H
#define _FUNC_UITLS_H

#include <cuda.h>
#include <curand_kernel.h>
#include <cmath>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
                         printf("Error at %s:%d\n", __FILE__, __LINE__); \
                         exit(1); }} while(0)

__global__ void setup_curand_kernel(curandState *state, int count){
    int id = threadIdx.x + blockIdx.x * 64;
    if(id < count){
        curand_init(1234, id, 0, &state[id]);
    }
}

void setup_curand(curandState **state, int count){
    CUDA_CALL(cudaMalloc((void**)state, count * sizeof(curandState)));
    setup_curand_kernel<<< ceil(count/64.0), 64>>>(*state, count);
}

#endif
