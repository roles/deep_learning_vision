#ifndef _CRBM_H
#define _CRBM_H

#include "matrix.h"
#include "nvmatrix.cuh"
#include <curand_kernel.h>
#include <cuda.h>

#define MAX_FILETER_SIZE 8
#define MAX_POOLING_RATE 3
#define RAND_SIZE 10000

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
        Matrix* CPU_y_h, *CPU_y_h_probs;
        Matrix* CPU_y_p;

        void CPU_convolution_forward();
        void CPU_max_pooling();

        NVMatrix *GPU_input;
        NVMatrix *GPU_filters;
        NVMatrix *GPU_vbias, *GPU_hbias;
        NVMatrix *GPU_y_h, *GPU_y_h_probs;
        NVMatrix *GPU_y_p;
        curandState *rnd_state;
        int rnd_state_num;

        void GPU_convolution_forward();
        void GPU_max_pooling();

        CRBM(int, int, int, int, int, int, int, int,
             Matrix*, Matrix*, Matrix*, Matrix*);
        ~CRBM();


    private:
        Matrix* filter_init(int, int, int);
};

#endif
