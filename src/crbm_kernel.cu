#include "crbm_kernel.cuh"

using namespace std;

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
        int feature_map_size, int feature_map_num, int pooling_rate,
        curandState *rnd_state, int rnd_state_num){
    __shared__ float shFm[16*MAX_POOLING_RATE][16*MAX_POOLING_RATE];

    int imgIdx = blockIdx.y / (feature_map_size / 16 / pooling_rate);
    int fmIdx = blockIdx.x / (feature_map_size / 16 / pooling_rate);
    int tx = (blockIdx.x % (feature_map_size / pooling_rate / 16)) * 16 + threadIdx.x;
    int ty = (blockIdx.y % (feature_map_size / pooling_rate / 16)) * 16 + threadIdx.y;
    int subsample_size = feature_map_size / pooling_rate;

    float rnd = curand_uniform(rnd_state);

    float *fm = feature_map + imgIdx * feature_map_num * feature_map_size * feature_map_size +
        fmIdx * feature_map_size * feature_map_size;

    probs = probs + imgIdx * feature_map_num * feature_map_size * feature_map_size +
        fmIdx * feature_map_size * feature_map_size; 

    target = target + imgIdx * feature_map_num * subsample_size * subsample_size +
        fmIdx * subsample_size * subsample_size; 

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
            fm[(ty*pooling_rate+i) * feature_map_size + (tx*pooling_rate+j)] = 0;
        }
    }

    sum = 0;
    bool isStop = false;
    for(int i = 0; i < pooling_rate && !isStop; i++){
        for(int j = 0; j < pooling_rate && !isStop; j++){
            sum += shFm[threadIdx.y*pooling_rate+i][threadIdx.x*pooling_rate+j];
            if(rnd < sum){
                fm[(ty*pooling_rate+i) * feature_map_size + (tx*pooling_rate+j)] = 1;
                isStop = true;
            }
        }
    }
    if(!isStop){
        target[threadIdx.y*subsample_size+threadIdx.x] = 1;
    }else{
        target[threadIdx.y*subsample_size+threadIdx.x] = 0;
    }
}

__global__ void convolution_backward_kernel(float *y_h, float *filters, float *vibas,
        float *target, float *y_v,
        int input_size, int lu_padding, int channel_num, int feature_map_size, 
        int filter_num, int filter_size){
    int imgIdx = blockIdx.y / (input_size / 32); 
    int channelIdx = blockIdx.x / (input_size / 32);
    int tx = (blockIdx.x % (input_size / 32)) * 32 + threadIdx.x;
    int ty = (blockIdx.y % (input_size / 32)) * 32 + threadIdx.y;
    int padding = (filter_size - 1);

    __shared__ float shHidden[32+2*(MAX_FILETER_SIZE-1)][32+2*(MAX_FILETER_SIZE-1)];
    __shared__ float shFlipFilter[MAX_FILETER_SIZE][MAX_FILETER_SIZE];
    float local_target = 0.0f;

    target = target + imgIdx * channel_num * input_size * input_size +
        channelIdx * input_size * input_size;

    __syncthreads();


    for(int f = 0; f < filter_num; f++){
        float *cur_y_h = y_h + imgIdx * filter_num * feature_map_size * feature_map_size +
            f * feature_map_size * feature_map_size;

        float *cur_filter = filters + f * channel_num * filter_size * filter_size +
            channelIdx * filter_size * filter_size;
        
        if(threadIdx.x < filter_size && threadIdx.y < filter_size){
            shFlipFilter[threadIdx.y][threadIdx.x] = 
                cur_filter[(filter_size-1-threadIdx.y)*filter_size + filter_size-1-threadIdx.x];
        }

        float *shHiddenLoad = &shHidden[threadIdx.y][threadIdx.x];
        if(tx < padding || ty < padding){
            *shHiddenLoad = 0;
        }else{
            *shHiddenLoad = cur_y_h[(ty-padding) * input_size + 
                (tx-padding)];
        }

        if(threadIdx.x < 2 * padding){
            shHiddenLoad = &shHidden[threadIdx.y][threadIdx.x+32];
            if(ty < padding || (tx+32) >= (feature_map_size+padding)){
                *shHiddenLoad = 0; 
            }else{
                *shHiddenLoad = cur_y_h[(ty-padding) * feature_map_size + 
                    (tx+32-padding)];
            }
        }

        if(threadIdx.y < 2 * padding){
            shHiddenLoad = &shHidden[threadIdx.y+32][threadIdx.x];
            if(tx < padding || (ty+32) >= (feature_map_size+padding)){
                *shHiddenLoad = 0; 
            }else{
                *shHiddenLoad = cur_y_h[(ty+32-padding) * feature_map_size + 
                    (tx-padding)];
            }

            if(threadIdx.x < 2 * padding){
                shHiddenLoad = &shHidden[threadIdx.y+32][threadIdx.x+32];
                if((ty+32) >= (feature_map_size+padding) || 
                   (tx+32) >= (feature_map_size+padding)){
                    *shHiddenLoad = 0; 
                }else{
                    *shHiddenLoad = cur_y_h[(ty+32-padding) * feature_map_size + 
                        (tx+32-padding)];
                }
            }
        }

        __syncthreads();

        for(int i = 0; i < filter_size; i++){
            for(int j = 0; j < filter_size; j++){
                target[ty*input_size+tx] += 
                local_target +=
                    shHidden[threadIdx.y+i+lu_padding][threadIdx.x+j+lu_padding] *
                    shFlipFilter[i][j];
            }
        }

        __syncthreads();
    }
    local_target = expf(-local_target);
    local_target = 1.0f / (1.0f + local_target);
    target[ty*input_size+tx] = local_target;
}
