#ifndef _CRBM_H
#define _CRBM_H

#include "matrix.h"
#include "nvmatrix.cuh"

#define MAX_FILETER_SIZE 8

class CRBM {
    public:
        int filter_num;
        int filter_size;
        int input_num;
        int input_size;
        int channel_num;
        int feature_map_size;
        int pooling_rate;
        int left_upper_padding, right_low_padding;

        Matrix *CPU_input;
        Matrix *CPU_filters;
        Matrix *CPU_vbias, *CPU_hbias;
        Matrix* CPU_y_h;


        void CPU_convolution_forward();

        NVMatrix *GPU_input;
        NVMatrix *GPU_filters;
        NVMatrix *GPU_vbias, *GPU_hbias;
        NVMatrix *GPU_y_h;

        void GPU_convolution_forward();

        CRBM(int, int, int, int, int, int, int, int,
             Matrix*, Matrix*, Matrix*, Matrix*);
        ~CRBM();


    private:
        Matrix* filter_init(int, int, int);
};

#endif
