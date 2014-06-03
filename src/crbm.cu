#include "matrix.h"
#include "crbm.cuh"
#include <iostream>
#include "utils.h"
#include "crbm_kernel.cuh"

using namespace std;

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

CRBM::CRBM(int filter_num, int filter_size,
        int input_num, int input_size, int channel_num,
        int left_upper_padding, int right_low_padding,
        int pooling_rate, 
        Matrix *filters, Matrix *hbias,
        Matrix *vbias){
    
    this->epsilon       = 1;
    this->momentum      = 0.5;
    this->l2reg         = 0.01;
    this->ph_lambda     = 5;
    this->ph            = 0.002;

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
        this->CPU_filters = new Matrix(*filters);
    }

    if(hbias == NULL){
        this->CPU_hbias = new Matrix(filter_num, 1, -0.1f, -0.1f);
    }else{
        this->CPU_hbias = new Matrix(*hbias);
    }

    if(vbias == NULL){
        this->CPU_vbias = new Matrix(channel_num, 1);
    }else{
        this->CPU_vbias = new Matrix(*vbias);
    }
    this->CPU_input = new Matrix(input_num, channel_num * input_size * input_size);

    this->CPU_y_h = new Matrix(input_num ,
            filter_num * feature_map_size * feature_map_size);
    this->CPU_y_h_probs = new Matrix(input_num ,
            filter_num * feature_map_size * feature_map_size);
    this->CPU_y_h2 = new Matrix(input_num ,
            filter_num * feature_map_size * feature_map_size);
    this->CPU_y_h2_probs = new Matrix(input_num ,
            filter_num * feature_map_size * feature_map_size);
            //filter_num * feature_map_size * feature_map_size, 1, 1);
    this->CPU_y_p = new Matrix(input_num, 
            filter_num * subsample_size * subsample_size);
    this->CPU_y_v = new Matrix(this->CPU_input->get_row_num(),
            this->CPU_input->get_col_num());
    this->CPU_y_v_probs = new Matrix(this->CPU_input->get_row_num(),
            this->CPU_input->get_col_num());
    this->CPU_d_w = new Matrix(this->CPU_filters->get_row_num(),
            this->CPU_filters->get_col_num());
    this->CPU_d_w_pre = new Matrix(this->CPU_filters->get_row_num(),
            this->CPU_filters->get_col_num());
    this->CPU_d_hbias = new Matrix(this->CPU_hbias->get_row_num(),
            this->CPU_hbias->get_col_num());
    this->CPU_d_hbias_pre = new Matrix(this->CPU_hbias->get_row_num(),
            this->CPU_hbias->get_col_num());
    this->CPU_d_hbias_tmp = new Matrix(this->CPU_hbias->get_row_num(),
            this->CPU_hbias->get_col_num());
    this->CPU_d_h_sum_tmp = new Matrix(1, this->CPU_y_h->get_col_num());

    this->GPU_filters = new NVMatrix(*this->CPU_filters);
    this->GPU_hbias = new NVMatrix(*this->CPU_hbias);
    this->GPU_vbias = new NVMatrix(*this->CPU_vbias);
    this->GPU_input = new NVMatrix(*this->CPU_input);
    this->GPU_y_h = new NVMatrix(*this->CPU_y_h);
    this->GPU_y_h_probs = new NVMatrix(*this->CPU_y_h_probs);
    this->GPU_y_h2 = new NVMatrix(*this->CPU_y_h2);
    this->GPU_y_h2_probs = new NVMatrix(*this->CPU_y_h2_probs);
    this->GPU_y_p = new NVMatrix(*this->CPU_y_p);
    this->GPU_y_v = new NVMatrix(*this->CPU_y_v);
    this->GPU_y_v_probs = new NVMatrix(*this->CPU_y_v_probs);
    this->GPU_d_w = new NVMatrix(*this->CPU_d_w);
    this->GPU_d_w_pre = new NVMatrix(*this->CPU_d_w);
    this->GPU_d_hbias = new NVMatrix(*this->CPU_d_hbias);
    this->GPU_d_hbias_pre = new NVMatrix(*this->CPU_d_hbias);
    this->GPU_d_hbias_tmp = new NVMatrix(*this->CPU_d_hbias_tmp);
    this->GPU_d_h_sum_tmp= new NVMatrix(*this->CPU_d_h_sum_tmp);

    this->rnd_state_num = 1000;
    setup_curand(&this->rnd_state, this->rnd_state_num);
}

CRBM::~CRBM(){
    delete this->CPU_filters;
    delete this->CPU_hbias;
    delete this->CPU_vbias;
    delete this->CPU_y_h;
    delete this->CPU_y_h_probs;
    delete this->CPU_y_p;
    delete this->CPU_y_v;
    delete this->CPU_y_v_probs;
    delete this->CPU_d_w;
    delete this->CPU_d_w_pre;
    delete this->CPU_d_hbias;
    delete this->CPU_d_hbias_pre;
    delete this->CPU_d_hbias_tmp;
    delete this->CPU_d_h_sum_tmp;

    delete this->GPU_filters;
    delete this->GPU_hbias;
    delete this->GPU_vbias;
    delete this->GPU_y_h;
    delete this->GPU_y_h_probs;
    delete this->GPU_y_p;
    delete this->GPU_y_v;
    delete this->GPU_y_v_probs;
    delete this->GPU_d_w;
    delete this->GPU_d_w_pre;
    delete this->GPU_d_hbias;
    delete this->GPU_d_hbias_pre;
    delete this->GPU_d_hbias_tmp;
    delete this->GPU_d_h_sum_tmp;

    cudaFree(this->rnd_state);
}

Matrix* CRBM::filter_init(int filter_size, int filter_num, int channel_num){
    float low   = - 4 * sqrt(6.0 / (2 * filter_size * filter_size * channel_num)); 
    float upper = -low;
    return new Matrix(filter_num, channel_num*filter_size*filter_size, low, upper);
}

void CRBM::CPU_convolution_forward(float *input, float *filter, float *target, float *hbias){

    bzero(target, input_num * filter_num * feature_map_size * feature_map_size * sizeof(float));

    for(int img = 0; img < input_num; img++){
        for(int fil = 0; fil < filter_num; fil++){

            float *curBias = hbias + fil;

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

void CRBM::CPU_max_pooling(float *y_h, float *y_h_probs, float *y_p){
    float pooling_area[MAX_POOLING_RATE*MAX_FILETER_SIZE+1];

    for(int img = 0; img < input_num; img++){
        for(int fil = 0; fil < filter_num; fil++){
            float *fm = y_h + 
                img * filter_num * feature_map_size * feature_map_size +
                fil * feature_map_size * feature_map_size;
            float *probs = y_h_probs + 
                img * filter_num * feature_map_size * feature_map_size +
                fil * feature_map_size * feature_map_size;
            float *target = y_p +
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
                        target[(i/pooling_rate)*subsample_size+(j/pooling_rate)] = 0;
                    }else{
                        target[(i/pooling_rate)*subsample_size+(j/pooling_rate)] = 1;
                        int pi = pooling_idx / pooling_rate;
                        int pj = pooling_idx % pooling_rate;
                        fm[(i+pi) * feature_map_size + (j+pj)] = 1;
                    }
                }
            }
        }
    }
}

void CRBM::CPU_convolution_backward(float *y_h, float *filters, float *vbias,
        float *y_v_probs, float *y_v){
    float tmp_recon[MAX_IMGAG_SIZE][MAX_IMGAG_SIZE];
    int padding = filter_size-1;
    int input_padding_size = feature_map_size + filter_size - 1;
    int lu_padding = left_upper_padding;

    bzero(tmp_recon, sizeof(tmp_recon));

    for(int img = 0; img < input_num; img++){
        for(int cha = 0; cha < channel_num; cha++){
            float *target = y_v_probs +
                img * channel_num * input_size * input_size +
                cha * input_size * input_size;

            float *target_y_v = y_v +
                img * channel_num * input_size * input_size +
                cha * input_size * input_size;

            for(int fil = 0; fil < filter_num; fil++){
                float *filter = filters +
                    fil * filter_size * filter_size * channel_num +
                    cha * filter_size * filter_size;

                float *fm = y_h +
                    img * filter_num * feature_map_size * feature_map_size +
                    fil * feature_map_size * feature_map_size;

                for(int r = 0; r < feature_map_size + filter_size - 1; r++){
                    for(int c = 0; c < feature_map_size + filter_size - 1; c++){

                        for(int i = r; i < r+filter_size; i++){
                            for(int j = c; j < c+filter_size; j++){
                                if(!(i < padding || j < padding ||
                                            i >= (padding + feature_map_size) ||
                                            j >= (padding + feature_map_size))){
                                    tmp_recon[r][c] += 
                                        fm[(i-padding)*feature_map_size + (j-padding)] *
                                        filter[(filter_size-1-(i-r))*filter_size + (filter_size-1-(j-c))];
                                }
                            }
                        }
                    }
                }
            }

            for(int i = 0; i < input_size; i++){
                for(int j = 0; j < input_size; j++){
                    target[i*input_size+j] = logisitc(tmp_recon[i+lu_padding][j+lu_padding]);
                    target_y_v[i*input_size+j] =
                        (random_float(0,1) < target[i*input_size+j]) ? 1 : 0;
                }
            }
            bzero(tmp_recon, sizeof(tmp_recon));
        }
    }
}

/*
 * 分为positive phase和negative phase
 * is_init为true则计算positive phase, dw初始化为0
 * is_init为false则计算negative phase, dw -= new_dw
 */
void CRBM::CPU_compute_d_w(float *v, float *h, float *dw, bool is_init){

    float sign;
    int lu_padding = left_upper_padding;
    if(is_init){
        bzero(dw, filter_num * channel_num * filter_size * filter_size * sizeof(float));
        sign = 1.0f;
    }else{
        sign = -1.0f;
    }

    for(int img = 0; img < input_num; img++){
        for(int fil = 0; fil < filter_num; fil++){

            float *this_h = h + img * filter_num * feature_map_size * feature_map_size +
                fil * feature_map_size * feature_map_size;

            for(int cha = 0; cha < channel_num; cha++){

                float *this_v = v + img * channel_num * input_size * input_size +
                    cha * input_size * input_size;

                float *this_dw = dw + fil * channel_num * filter_size * filter_size +
                    cha * filter_size * filter_size;

                for(int r = 0; r < filter_size; r++){
                    for(int c = 0; c < filter_size; c++){

                        float *cur_v = this_v + (r-lu_padding) * input_size +
                            (c-lu_padding);

                        for(int i = 0; i < feature_map_size; i++){
                            for(int j = 0; j < feature_map_size; j++){
                                if(!((r+i) < lu_padding ||
                                            (c+j) < lu_padding ||
                                            (r+i) >= (lu_padding+input_size) ||
                                            (c+j) >= (lu_padding+input_size))){

                                    this_dw[r*filter_size+c] += 
                                        sign * cur_v[j] * this_h[i*feature_map_size+j];
                                }
                            }
                            cur_v += input_size;
                        }
                    }
                }
            }
        }
    }
}

void CRBM::GPU_convolution_forward(float *input, float *filters, float *y_h, float *hbias){
    dim3 blocks = dim3(input_size / 32 * filter_num, input_size / 32 * input_num);
    dim3 threads = dim3(32, 32);
    convolution_forward_kernel<<<blocks, threads>>>(input, filters, y_h, 
            hbias, input_size, channel_num, feature_map_size, filter_size, 
            filter_num, left_upper_padding);
    cudaDeviceSynchronize();
}

void CRBM::GPU_max_pooling(float *y_h, float *y_h_probs, float *y_p){
    dim3 blocks = dim3(feature_map_size / pooling_rate / 16 * filter_num, 
            feature_map_size / pooling_rate / 16 * input_num);
    dim3 threads = dim3(16, 16);
    max_pooling_kernel<<<blocks, threads>>>(y_h, y_h_probs, y_p,
            feature_map_size, filter_num, pooling_rate, rnd_state, rnd_state_num);
    cudaDeviceSynchronize();
}

void CRBM::GPU_convolution_backward(float *y_h, float *filters, float *vbias, 
        float *y_v_probs, float *y_v){
    dim3 blocks = dim3(input_size / 32 * channel_num, input_size / 32 * input_num);
    dim3 threads = dim3(32, 32);

    convolution_backward_kernel<<<blocks, threads>>>(y_h,
            filters, vbias, y_v_probs, y_v, input_size, left_upper_padding,
            channel_num, feature_map_size, filter_num, filter_size, rnd_state, rnd_state_num);
    cudaDeviceSynchronize();
}

void CRBM::GPU_compute_d_w(float *v, float *h, float *dw, bool is_init){
    if(is_init){
        cudaMemset(dw, 0, filter_num * channel_num * filter_size * filter_size * 
                sizeof(float));
    }
    dim3 blocks = dim3(channel_num * filter_num * feature_map_size / 32, 
            input_num * feature_map_size / 32);
    dim3 threads = dim3(filter_size, filter_size);

    compute_d_w_kernel<<<blocks, threads>>>(v, h, dw, is_init, input_size, left_upper_padding,
            channel_num, filter_num, filter_size, feature_map_size);
    cudaDeviceSynchronize();
}

void CRBM::run_batch(Matrix& batch_data){
    batch_data.assign(*this->CPU_input);
    this->GPU_input->copyFromHost(*this->CPU_input);

    start();
}

void CRBM::start(){
    bool cheak_euqality = false;
    bool run_CPU = true;
    bool run_GPU = false;

    struct timeval _start_time, _end_time;

    if(run_CPU){
    /* CPU computation */
    /**********************************/
        timeFunc(this->CPU_convolution_forward(this->CPU_input->get_data(),
                    this->CPU_filters->get_data(), this->CPU_y_h->get_data(),
                    this->CPU_hbias->get_data()), "CPU convolutional forward");

        timeFunc(this->CPU_max_pooling(this->CPU_y_h->get_data(),
                    this->CPU_y_h_probs->get_data(), this->CPU_y_p->get_data()), 
                "CPU max pooling");

        timeFunc(this->CPU_convolution_backward(this->CPU_y_h->get_data(),
                    this->CPU_filters->get_data(), this->CPU_vbias->get_data(),
                    this->CPU_y_v_probs->get_data(), this->CPU_y_v->get_data()), 
                "CPU convolutional backward");

        timeFunc(this->CPU_convolution_forward(this->CPU_y_v->get_data(),
                    this->CPU_filters->get_data(), this->CPU_y_h2->get_data(),
                    this->CPU_hbias->get_data()), "CPU convolutional forward");

        timeFunc(this->CPU_max_pooling(this->CPU_y_h2->get_data(),
                    this->CPU_y_h2_probs->get_data(), this->CPU_y_p->get_data()), 
                "CPU max pooling");

        timeFunc(this->CPU_compute_d_w(this->CPU_input->get_data(),
                    this->CPU_y_h_probs->get_data(), this->CPU_d_w->get_data(),
                    true), "CPU compute dw positive phase");

        timeFunc(this->CPU_compute_d_w(this->CPU_y_v->get_data(),
                    this->CPU_y_h2_probs->get_data(), this->CPU_d_w->get_data(),
                    false), "CPU compute dw negative phase");

        this->CPU_d_w->ele_scale(1.0 / (input_num * feature_map_size * feature_map_size));

        this->CPU_y_h_probs->mat_sum(1, *this->CPU_d_h_sum_tmp);
        this->CPU_d_h_sum_tmp->reshape(filter_num, feature_map_size * feature_map_size);
        this->CPU_d_h_sum_tmp->mat_sum(0, *this->CPU_d_hbias);
        this->CPU_d_h_sum_tmp->reshape(1, filter_num * feature_map_size * feature_map_size);

        this->CPU_y_h2_probs->mat_sum(1, *this->CPU_d_h_sum_tmp);
        this->CPU_d_h_sum_tmp->reshape(filter_num, feature_map_size * feature_map_size);
        this->CPU_d_h_sum_tmp->mat_sum(0, *this->CPU_d_hbias_tmp);
        this->CPU_d_h_sum_tmp->reshape(1, filter_num * feature_map_size * feature_map_size);

        this->CPU_d_hbias->mat_add(*this->CPU_d_hbias_tmp, -1.0f);
        this->CPU_d_hbias->ele_scale(1.0 / (input_num * feature_map_size * feature_map_size));

        this->CPU_y_h->mat_sum(1, *this->CPU_d_h_sum_tmp);
        this->CPU_d_h_sum_tmp->reshape(filter_num, feature_map_size * feature_map_size);
        this->CPU_d_h_sum_tmp->mat_sum(0, *this->CPU_d_hbias_tmp);
        this->CPU_d_h_sum_tmp->reshape(1, filter_num * feature_map_size * feature_map_size);

        this->CPU_d_hbias_tmp->ele_scale(1.0 / (input_num * feature_map_size * feature_map_size));
        this->CPU_d_hbias_tmp->ele_add(-this->ph);
        this->CPU_d_hbias_tmp->ele_scale(this->ph_lambda);
        this->CPU_d_hbias->mat_add(*this->CPU_d_hbias_tmp, -1.0f);

        this->CPU_d_w->mat_add(*this->CPU_d_w_pre, *this->CPU_d_w, epsilon, momentum);
        this->CPU_d_w->assign(*this->CPU_d_w_pre);
        this->CPU_filters->mat_add(*this->CPU_d_w, 1.0f);

        this->CPU_d_hbias->mat_add(*this->CPU_d_hbias_pre, *this->CPU_d_hbias, epsilon, momentum);
        this->CPU_d_hbias->assign(*this->CPU_d_hbias_pre);
        this->CPU_hbias->mat_add(*this->CPU_d_hbias, 1.0f);

        this->CPU_y_v->mat_add(*this->CPU_input, -1.0f);
        this->CPU_y_v->mat_mul(*this->CPU_y_v);
        float ferr = this->CPU_y_v->ele_mean();
        float sparsity = this->CPU_y_h->ele_mean();

        cout << "ferr : " << ferr << endl;
        cout << "sparsity : " << sparsity << endl;
    }
    /**********************************/

    if(run_GPU){
    /* GPU computation */
    /**********************************/

        timeFunc(this->GPU_convolution_forward(this->GPU_input->get_data(),
                    this->GPU_filters->get_data(), this->GPU_y_h->get_data(),
                    this->GPU_hbias->get_data()), "GPU convolutional forward");

        timeFunc(this->GPU_max_pooling(this->GPU_y_h->get_data(),
                    this->GPU_y_h_probs->get_data(), this->GPU_y_p->get_data()), 
                "GPU max pooling");

        timeFunc(this->GPU_convolution_backward(this->GPU_y_h->get_data(),
                    this->GPU_filters->get_data(), this->GPU_vbias->get_data(),
                    this->GPU_y_v_probs->get_data(), this->GPU_y_v->get_data()), 
                "GPU convolutional backward");

        timeFunc(this->GPU_convolution_forward(this->GPU_y_v->get_data(),
                    this->GPU_filters->get_data(), this->GPU_y_h2->get_data(),
                    this->GPU_hbias->get_data()), "GPU convolutional forward");

        timeFunc(this->GPU_max_pooling(this->GPU_y_h2->get_data(),
                    this->GPU_y_h2_probs->get_data(), this->GPU_y_p->get_data()), 
                "GPU max pooling");

        timeFunc(this->GPU_compute_d_w(this->GPU_input->get_data(),
                    this->GPU_y_h_probs->get_data(), this->GPU_d_w->get_data(),
                    true), "GPU compute dw positive phase");

        timeFunc(this->GPU_compute_d_w(this->GPU_y_v->get_data(),
                    this->GPU_y_h2_probs->get_data(), this->GPU_d_w->get_data(),
                    false), "GPU compute dw negative phase");
        
        this->GPU_d_w->ele_scale(1.0 / (input_num * feature_map_size * feature_map_size));

        this->GPU_y_h_probs->mat_sum(1, *this->GPU_d_h_sum_tmp);
        this->GPU_d_h_sum_tmp->reshape(filter_num, feature_map_size * feature_map_size);
        this->GPU_d_h_sum_tmp->mat_sum(0, *this->GPU_d_hbias);
        this->GPU_d_h_sum_tmp->reshape(1, filter_num * feature_map_size * feature_map_size);

        this->GPU_y_h2_probs->mat_sum(1, *this->GPU_d_h_sum_tmp);
        this->GPU_d_h_sum_tmp->reshape(filter_num, feature_map_size * feature_map_size);
        this->GPU_d_h_sum_tmp->mat_sum(0, *this->GPU_d_hbias_tmp);
        this->GPU_d_h_sum_tmp->reshape(1, filter_num * feature_map_size * feature_map_size);

        this->GPU_d_hbias->mat_add(*this->GPU_d_hbias_tmp, -1.0f);
        this->GPU_d_hbias->ele_scale(1.0 / (input_num * feature_map_size * feature_map_size));

        this->GPU_y_h->mat_sum(1, *this->GPU_d_h_sum_tmp);
        this->GPU_d_h_sum_tmp->reshape(filter_num, feature_map_size * feature_map_size);
        this->GPU_d_h_sum_tmp->mat_sum(0, *this->GPU_d_hbias_tmp);
        this->GPU_d_h_sum_tmp->reshape(1, filter_num * feature_map_size * feature_map_size);

        this->GPU_d_hbias_tmp->ele_scale(1.0 / (input_num * feature_map_size * feature_map_size));
        this->GPU_d_hbias_tmp->ele_add(-this->ph);
        this->GPU_d_hbias_tmp->ele_scale(this->ph_lambda);
        this->GPU_d_hbias->mat_add(*this->GPU_d_hbias_tmp, -1.0f);

        this->GPU_d_w->mat_add(*this->GPU_d_w_pre, *this->GPU_d_w, epsilon, momentum);
        this->GPU_d_w->assign(*this->GPU_d_w_pre);
        this->GPU_filters->mat_add(*this->GPU_d_w, 1.0f);

        this->GPU_d_hbias->mat_add(*this->GPU_d_hbias_pre, *this->GPU_d_hbias, epsilon, momentum);
        this->GPU_d_hbias->assign(*this->GPU_d_hbias_pre);
        this->GPU_hbias->mat_add(*this->GPU_d_hbias, 1.0f);

        this->GPU_y_v->mat_add(*this->GPU_input, -1.0f);
        this->GPU_y_v->mat_mul(*this->GPU_y_v);
        float ferr = this->GPU_y_v->ele_mean();
        float sparsity = this->GPU_y_h->ele_mean();

        cout << "ferr : " << ferr << endl;
        cout << "sparsity : " << sparsity << endl;
    }

    if(cheak_euqality){
        /*
         * CPU and GPU equality test
         */
        cout << "y_h_probs : ";
        Matrix* tmp_y_h_probs = new Matrix(this->CPU_y_h_probs->get_row_num(),
                                     this->CPU_y_h_probs->get_col_num());
        this->GPU_y_h_probs->assign(*tmp_y_h_probs);
        this->CPU_y_h_probs->equal_value(*tmp_y_h_probs);
        delete tmp_y_h_probs;

        cout << "y_v_probs : ";
        Matrix* tmp_y_v_probs = new Matrix(this->CPU_y_v_probs->get_row_num(),
                                     this->CPU_y_v_probs->get_col_num());
        this->GPU_y_v_probs->assign(*tmp_y_v_probs);
        this->CPU_y_v_probs->equal_value(*tmp_y_v_probs);
        delete tmp_y_v_probs;

        cout << "y_h2_probs : ";
        Matrix* tmp_y_h2_probs = new Matrix(this->CPU_y_h2_probs->get_row_num(),
                                     this->CPU_y_h2_probs->get_col_num());
        this->GPU_y_h2_probs->assign(*tmp_y_h2_probs);
        this->CPU_y_h2_probs->equal_value(*tmp_y_h2_probs);
        delete tmp_y_h2_probs;

        cout << "d_w : ";
        Matrix* tmp_d_w = new Matrix(this->CPU_d_w->get_row_num(),
                                     this->CPU_d_w->get_col_num());
        this->GPU_d_w->assign(*tmp_d_w);
        this->CPU_d_w->equal_value(*tmp_d_w, 1);
        delete tmp_d_w;

        cout << "d_hbias : ";
        Matrix* tmp_d_hbias = new Matrix(this->CPU_d_hbias->get_row_num(),
                                     this->CPU_d_hbias->get_col_num());
        this->GPU_d_hbias->assign(*tmp_d_hbias);
        this->CPU_d_hbias->equal_value(*tmp_d_hbias, 1);
        delete tmp_d_hbias;

        cout << "filter : ";
        Matrix* tmp_filters = new Matrix(this->CPU_filters->get_row_num(),
                                     this->CPU_filters->get_col_num());
        this->GPU_filters->assign(*tmp_filters);
        this->CPU_filters->equal_value(*tmp_filters);
        delete tmp_filters;

        cout << "hbias : ";
        Matrix* tmp_hbias = new Matrix(this->CPU_hbias->get_row_num(),
                                     this->CPU_hbias->get_col_num());
        this->GPU_hbias->assign(*tmp_hbias);
        this->CPU_hbias->equal_value(*tmp_hbias);
        delete tmp_hbias;
    }
}
