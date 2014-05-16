#include "matrix.h"
#include "crbm.cuh"
#include <iostream>
#include "utils.h"

using namespace std;

CRBM::CRBM(int filter_num, int filter_size,
        int input_num, int input_size, int channel_num,
        int left_upper_padding, int right_low_padding,
        int pooling_rate, 
        Matrix *filters, Matrix *vbias,
        Matrix *hbias, Matrix *input){
    this->filter_num    = filter_num;  
    this->filter_size   = filter_size;
    this->input_num     = input_num;
    this->input_size    = input_size; 
    this->pooling_rate  = pooling_rate;
    this->channel_num = channel_num;
    this->left_upper_padding = left_upper_padding;
    this->right_low_padding = right_low_padding;
    this->feature_map_size = input_size + left_upper_padding +
        right_low_padding - filter_size + 1;
    this->subsample_size = feature_map_size / pooling_rate;

    if(filters == NULL){
        this->CPU_filters = filter_init(filter_size, filter_num, channel_num); 
    }else{
        this->CPU_filters = filters;
    }

    if(hbias == NULL){
        this->CPU_hbias = new Matrix(filter_num, 1);
    }else{
        this->CPU_hbias = hbias;
    }

    if(vbias == NULL){
        this->CPU_vbias = new Matrix(channel_num, 1);
    }else{
        this->CPU_vbias = vbias;
    }
    this->CPU_input = input;
    this->CPU_y_h = new Matrix(input_num ,
            filter_num * feature_map_size * feature_map_size);
    this->CPU_y_h_probs = new Matrix(input_num ,
            filter_num * feature_map_size * feature_map_size);
    this->CPU_y_p = new Matrix(input_num, 
            filter_num * subsample_size * subsample_size);

    this->GPU_filters = new NVMatrix(*this->CPU_filters);
    this->GPU_hbias = new NVMatrix(*this->CPU_hbias);
    this->GPU_vbias = new NVMatrix(*this->CPU_vbias);
    this->GPU_input = new NVMatrix(*this->CPU_input);
    this->GPU_y_h = new NVMatrix(*this->CPU_y_h);
    this->GPU_y_h_probs = new NVMatrix(*this->CPU_y_h);
    this->GPU_y_p = new NVMatrix(*this->CPU_y_p);
}

CRBM::~CRBM(){
    delete this->CPU_filters;
    delete this->CPU_hbias;
    delete this->CPU_vbias;
    delete this->CPU_y_h;
    delete this->CPU_y_h_probs;
    delete this->CPU_y_p;

    delete this->GPU_filters;
    delete this->GPU_hbias;
    delete this->GPU_vbias;
    delete this->GPU_y_h;
    delete this->GPU_y_h_probs;
    delete this->GPU_y_p;
}

Matrix* CRBM::filter_init(int filter_size, int filter_num, int channel_num){
    float low   = - 4 * sqrt(6.0 / (2 * filter_size * filter_size * channel_num)); 
    float upper = -low;
    return new Matrix(filter_num, channel_num*filter_size*filter_size, low, upper);
    //return new Matrix(filter_num, channel_num*filter_size*filter_size, 1.5, 1.5);
}

void CRBM::CPU_convolution_forward(){
    float* input = this->CPU_input->get_data();
    float* target = this->CPU_y_h->get_data();
    float* filter = this->CPU_filters->get_data();

    for(int img = 0; img < input_num; img++){
        for(int fil = 0; fil < filter_num; fil++){

            float *curBias = this->CPU_hbias->get_data() + fil;

            for(int r = 0; r < feature_map_size; r++){
                for(int c = 0; c < feature_map_size; c++){

                    float *curFilter = filter + fil * channel_num * filter_size * filter_size;

                    float* curTarget = target + img * filter_num * feature_map_size * feature_map_size +
                        fil * feature_map_size * feature_map_size +
                        r * feature_map_size + c;

                    for(int k = 0; k < channel_num; k++){

                        float* curInput = input + img * channel_num * input_size * input_size +
                            k * input_size * input_size + 
                            (r < left_upper_padding ? 0 : r - left_upper_padding) * input_size +
                            (c < left_upper_padding ? 0 : c - left_upper_padding);

                        for(int i = 0; i < filter_size; i++){

                            if(!((r+i) < left_upper_padding || 
                                 (r+i) >= (left_upper_padding + input_size))){

                                int step = 0;

                                for(int j = 0; j < filter_size; j++){ 
                                    if(!((c+j) < left_upper_padding ||
                                         (c+j) >= (left_upper_padding + input_size))){
                                        *curTarget += curFilter[i*filter_size+j] * (*curInput);
                                        curInput++;
                                        step++;
                                    }
                                }
                                curInput += input_size - step;

                            }
                        }
                        curFilter += filter_size * filter_size;
                    }
                    *curTarget += *curBias;
                }
            }
        }
    }
}

static int max_pooling_multinomial(float *probs, int len){
    float rnd = random_float(0, 1);
    int i;

    for(i = 0; rnd > probs[i]; i++, probs[i] += probs[i-1]);

    return i;
}

void CRBM::CPU_max_pooling(){
    float pooling_area[MAX_POOLING_RATE*MAX_FILETER_SIZE+1];

    for(int img = 0; img < input_num; img++){
        for(int fil = 0; fil < filter_num; fil++){
            float *fm = CPU_y_h->get_data() + 
                img * filter_num * feature_map_size * feature_map_size +
                fil * feature_map_size * feature_map_size;
            float *probs = CPU_y_h_probs->get_data() + 
                img * filter_num * feature_map_size * feature_map_size +
                fil * feature_map_size * feature_map_size;
            float *target = CPU_y_p->get_data() +
                img * filter_num * subsample_size * subsample_size +
                fil * subsample_size * subsample_size;

            for(int i = 0; i < feature_map_size; i += pooling_rate){
                for(int j = 0; j < feature_map_size; j += pooling_rate){
                    
                    float sum = 0;
                    for(int pi = 0; pi < pooling_rate; pi++){
                        for(int pj = 0; pj < pooling_rate; pj++){
                            float *cur_fm = fm + (i+pi) * feature_map_size + (j+pj);
                            *cur_fm = expf(*cur_fm);
                            sum += *cur_fm;
                        }
                    }
                    for(int pi = 0; pi < pooling_rate; pi++){
                        for(int pj = 0; pj < pooling_rate; pj++){
                            float *cur_fm = fm + (i+pi) * feature_map_size + (j+pj);
                            float *cur_probs = probs + (i+pi) * feature_map_size + (j+pj);
                            *cur_probs = *cur_fm / (1 +sum);
                            pooling_area[pi*pooling_rate+pj] = *cur_probs;
                            *cur_fm = 0;
                        }
                    }
                    pooling_area[pooling_rate*pooling_rate] = 1.0/(1+sum);
                    int pooling_idx = max_pooling_multinomial(pooling_area, 
                            pooling_rate*pooling_rate+1);
                    if(pooling_idx == pooling_rate*pooling_rate){
                        target[(i/pooling_rate)*subsample_size+(j/pooling_rate)] = 1;
                    }else{
                        target[(i/pooling_rate)*subsample_size+(j/pooling_rate)] = 0;
                        int pi = pooling_idx / pooling_rate;
                        int pj = pooling_idx % pooling_rate;
                        fm[(i+pi) * feature_map_size + (j+pj)] = 1;
                    }
                }
            }
        }
    }
}

__global__ void convolution_forward_kernel(float *input, 
        float *filters, float *feature_map, float *hbias, int input_size, 
        int channel_num, int feature_map_size, int filter_size,
        int filter_num, int lu_padding){
    __shared__ float shImg[32+MAX_FILETER_SIZE-1][32+MAX_FILETER_SIZE-1];
    __shared__ float shFilter[MAX_FILETER_SIZE][MAX_FILETER_SIZE];

    int imgIdx = blockIdx.y / (input_size / 32);
    int filterIdx = blockIdx.x / (input_size / 32);
    int tx = blockIdx.x % (input_size / 32) * 32 + threadIdx.x;
    int ty = blockIdx.y % (input_size / 32) * 32 + threadIdx.y;

    float *target = feature_map + 
        imgIdx * feature_map_size * feature_map_size * filter_num + 
        feature_map_size * feature_map_size * filterIdx +
        ty * feature_map_size + tx;


    for(int g = 0; g < channel_num; g++){

        if(threadIdx.x < filter_size && threadIdx.y < filter_size){
            shFilter[threadIdx.y][threadIdx.x] = 
                filters[filterIdx * channel_num * filter_size * filter_size +
                + g * filter_size * filter_size +
                threadIdx.y * filter_size + threadIdx.x];
        }
        __syncthreads();

        float *img = input + imgIdx * input_size * input_size * channel_num
            + g * input_size * input_size;

        float *shImgLoad = &shImg[threadIdx.y][threadIdx.x];
        if(tx < lu_padding || ty < lu_padding){
            *shImgLoad = 0;
        }else{
            *shImgLoad = img[(ty-lu_padding) * input_size + (tx-lu_padding)];
        }

        if(threadIdx.x < MAX_FILETER_SIZE-1){
            shImgLoad = &shImg[threadIdx.y][threadIdx.x+32];
            if(ty < lu_padding || (tx+32) >= (input_size+lu_padding)){
                *shImgLoad = 0; 
            }else{
                *shImgLoad = img[(ty-lu_padding) * input_size + 
                    (tx+32-lu_padding)];
            }
        }

        if(threadIdx.y < MAX_FILETER_SIZE-1){
            shImgLoad = &shImg[threadIdx.y+32][threadIdx.x];
            if(tx < lu_padding || (ty+32) >= (input_size+lu_padding)){
                *shImgLoad = 0; 
            }else{
                *shImgLoad = img[(ty+32-lu_padding) * input_size + 
                    (tx-lu_padding)];
            }

            if(threadIdx.x < MAX_FILETER_SIZE-1){
                shImgLoad = &shImg[threadIdx.y+32][threadIdx.x+32];
                if((ty+32) >= (input_size+lu_padding) || 
                   (tx+32) >= (input_size+lu_padding)){
                    *shImgLoad = 0; 
                }else{
                    *shImgLoad = img[(ty+32-lu_padding) * input_size + 
                        (tx+32-lu_padding)];
                }
            }
        }
        __syncthreads();

        float *imgPtr = &shImg[threadIdx.y][threadIdx.x];

        for(int i = 0; i < filter_size; i++){
            for(int j = 0; j < filter_size; j++){
                *target += imgPtr[j] * shFilter[i][j];
            }
            imgPtr += 32 + MAX_FILETER_SIZE - 1;
        }

        __syncthreads();

    }

    *target += hbias[filterIdx];

}

__global__ void max_pooling_kernel(float *feature_map, float *probs, float *target,
        int feature_map_size, int feature_map_num, int pooling_rate){
    __shared__ float shFm[16*MAX_POOLING_RATE][16*MAX_POOLING_RATE];

    int imgIdx = blockIdx.y / (feature_map_size / 16 / pooling_rate);
    int fmIdx = blockIdx.x / (feature_map_size / 16 / pooling_rate);
    int tx = (blockIdx.x % (feature_map_size / pooling_rate / 16)) * 16 + threadIdx.x;
    int ty = (blockIdx.y % (feature_map_size / pooling_rate / 16)) * 16 + threadIdx.y;

    float *fm = feature_map + imgIdx * feature_map_num * feature_map_size * feature_map_size +
        fmIdx * feature_map_size * feature_map_size;

    probs = probs + imgIdx * feature_map_num * feature_map_size * feature_map_size +
        fmIdx * feature_map_size * feature_map_size; 

    for(int i = 0; i < pooling_rate; i++){
        for(int j = 0; j < pooling_rate; j++){
            shFm[threadIdx.y*pooling_rate+i][threadIdx.x*pooling_rate+j] = 
                fm[(ty*pooling_rate+i) * feature_map_size + (tx*pooling_rate+j)];
        }
    }

    __syncthreads();

    float sum = 0;
    for(int i = 0; i < pooling_rate; i++){
        for(int j = 0; j < pooling_rate; j++){
            shFm[threadIdx.y*pooling_rate+i][threadIdx.x*pooling_rate+j] =
                __expf(shFm[threadIdx.y*pooling_rate+i][threadIdx.x*pooling_rate+j]);
            sum += shFm[threadIdx.y*pooling_rate+i][threadIdx.x*pooling_rate+j];
        }
    }
    for(int i = 0; i < pooling_rate; i++){
        for(int j = 0; j < pooling_rate; j++){
            shFm[threadIdx.y*pooling_rate+i][threadIdx.x*pooling_rate+j] /= (1 + sum);
            probs[(ty*pooling_rate+i) * feature_map_size + (tx*pooling_rate+j)] = 
                shFm[threadIdx.y*pooling_rate+i][threadIdx.x*pooling_rate+j];
        }
    }
}

void CRBM::GPU_convolution_forward(){
    dim3 blocks = dim3(input_size / 32 * filter_num, input_size / 32 * input_num);
    dim3 threads = dim3(32, 32);
    convolution_forward_kernel<<<blocks, threads>>>(GPU_input->get_data(),
        GPU_filters->get_data(), GPU_y_h->get_data(), GPU_hbias->get_data(), input_size,
        channel_num, feature_map_size, filter_size, filter_num, left_upper_padding);
}

void CRBM::GPU_max_pooling(){
    dim3 blocks = dim3(feature_map_size / pooling_rate / 16 * filter_num, 
            feature_map_size / pooling_rate / 16 * input_num);
    dim3 threads = dim3(16, 16);
    max_pooling_kernel<<<blocks, threads>>>(GPU_y_h->get_data(), GPU_y_h_probs->get_data(),
            GPU_y_p->get_data(), feature_map_size, filter_num, pooling_rate);
}
