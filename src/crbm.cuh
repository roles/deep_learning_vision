#ifndef _CRBM_H
#define _CRBM_H

#include "matrix.h"
#include "nvmatrix.cuh"
#include <curand_kernel.h>
#include <cuda.h>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
                         printf("Error at %s:%d\n", __FILE__, __LINE__); \
                         exit(1); }} while(0)

class CRBM {
    public:
        int filter_num;
        int filter_size;
        int input_num;
        int input_size;
        int channel_num;
        int feature_map_size;
        int pooling_rate;
        int subsample_size;
        int left_upper_padding, right_low_padding;

        Matrix *CPU_input;
        Matrix *CPU_filters;
        Matrix *CPU_vbias, *CPU_hbias;
        Matrix *CPU_y_h, *CPU_y_h_probs;
        Matrix *CPU_y_p;
        Matrix *CPU_y_v, *CPU_y_v_probs;
        Matrix *CPU_d_w;

        void CPU_convolution_forward(float*, float*, float*, float*);
        void CPU_max_pooling(float*, float*, float*);
        void CPU_convolution_backward(float*, float*, float*, float*, float*);
        void CPU_compute_d_w(float*, float*, float*, bool);

        NVMatrix *GPU_input;
        NVMatrix *GPU_filters;
        NVMatrix *GPU_vbias, *GPU_hbias;
        NVMatrix *GPU_y_h, *GPU_y_h_probs;
        NVMatrix *GPU_y_p;
        NVMatrix *GPU_y_v, *GPU_y_v_probs;
        NVMatrix *GPU_d_w;
        curandState *rnd_state;
        int rnd_state_num;

        void GPU_convolution_forward(float*, float*, float*, float*);
        void GPU_max_pooling(float*, float*, float*);
        void GPU_convolution_backward(float*, float*, float*, float*, float*);
        void GPU_compute_d_w(float*, float*, float*, bool);

        CRBM(int, int, int, int, int, int, int, int,
             Matrix*, Matrix*, Matrix*, Matrix*);
        void start();
        ~CRBM();


    private:
        Matrix* filter_init(int, int, int);
};

#endif
