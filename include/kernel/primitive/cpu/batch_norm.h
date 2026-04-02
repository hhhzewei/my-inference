//
// Created by hzw on 2026/4/1.
//

#ifndef MY_INFERENCE_BATCH_NORM_H
#define MY_INFERENCE_BATCH_NORM_H

namespace my_inference::cpu::primitive {
    template<typename T>
    void batchNorm(T *x, const float *scale, const float *B,const T *mean, T *var,
                   T *y,
                   float eps,
                   const int64_t N, const int64_t C, const int64_t stride) {
        const int64_t N_stride = C * stride;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < C; ++j) {
                for (int k = 0; k < stride; ++k) {
                    int64_t offset = i * N_stride + j * stride + k;
                    y[offset] = (x[offset] - mean[j]) * scale[j] / std::sqrt(var[j] + eps) + B[j];
                }
            }
        }
    }
}
#endif //MY_INFERENCE_BATCH_NORM_H
