#ifndef _FUNC_UITLS_H
#define _FUNC_UITLS_H

#include <cuda.h>
#include <curand_kernel.h>

__global__ void setup_curand_kernel(curandState *state, int count){
    int id = threadIdx.x + blockIdx.x * 64;
    if(id < count)
        curand_init(1234, id, 0, &state[id]);
}

void setup_curand(curandState *state, int count){
    cudaMalloc((void**)&state, count * sizeof(curandState));
    setup_curand_kernel<<<(count / 64), 64>>>(state, count);
}

#endif
